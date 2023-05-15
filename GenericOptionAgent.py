from typing import Optional, Callable
import numpy as np
import numpy.random
import tensorflow as tf
from keras.optimizers.legacy.adam import Adam
from tensorflow.python.ops.gen_dataset_ops import BatchDataset
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments import tf_py_environment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent, DdqnAgent
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.replay_buffers import TFUniformReplayBuffer
from tf_agents.replay_buffers.tf_uniform_replay_buffer import BufferInfo
from os import environ
from tf_agents.specs import array_spec, ArraySpec
from tf_agents.trajectories import time_step as ts, policy_step, trajectory
from tf_agents.typing import types
from tf_agents.utils import nest_utils
from tf_agents.networks import network
from tf_agents.utils.common import element_wise_squared_loss

import FourRooms
from DirectionalPseudoRewards import pseudo_up, pseudo_down, pseudo_left, pseudo_right

num_episodes = 10
num_steps = 10000
exam_steps = 10000
add_and_train_policies_interval = 1000
train_steps = 5
train_batch_size = 64
print_step_interval = 500
verbose = False
verbose_agent = False


class PolicyOverOptions(PyPolicy):
    def __init__(self, time_step_spec: ts.TimeStep, action_spec: types.NestedArraySpec,
                 policy_state_spec=ArraySpec((2, ), np.int32), rng_seed=150):
        super(PolicyOverOptions, self).__init__(time_step_spec, action_spec, policy_state_spec)
        self._outer_dims = None
        numpy.random.seed(rng_seed)
        self.rng_seed = rng_seed
        self._rng = numpy.random
        self.options: [PyPolicy] = []
        self.option_prob = 0.3
        self.option_termination_prob = 0.1

    def _action(self, time_step: ts.TimeStep, policy_state: types.NestedArray,
                seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:

        outer_dims = self._outer_dims
        if outer_dims is None:
            outer_dims = nest_utils.get_outer_array_shape(
                time_step.observation, self._time_step_spec.observation
            )

        if policy_state[1] == 1:    # if we are in a policy, don't leave the policy
            if self._rng.random() < self.option_termination_prob:
                print("\tExiting option.")
            else:
                print(f"\tContinuing with option {policy_state[0]}...")
                return self.options[policy_state[0]].action(time_step, ())

        if len(self.options) == 0 or self._rng.random() > self.option_prob:
            action_ = array_spec.sample_spec_nest(self._action_spec, self._rng, outer_dims=outer_dims)
            state_ = (0, 0)     # zero indicates nop
            info_ = array_spec.sample_spec_nest(self._info_spec, self._rng, outer_dims=outer_dims)
            return policy_step.PolicyStep(action_, state_, info_)

        else:
            # sample a policy over options (assume globally-available options, and uniform distribution over options)
            option = self._rng.randint(len(self.options))
            print(f"Executing option {option}")
            p_step = self.options[option].action(time_step, ())
            return p_step

    def add_option(self, opt):
        self.options.append(opt)


class RepeatActionPolicy(PyPolicy):
    def __init__(self, time_step_spec: ts.TimeStep, action_spec: types.NestedArraySpec, policy_goal='u'):
        super().__init__(time_step_spec, action_spec)
        self.policy_goal = policy_goal

    def _action(self, time_step: ts.TimeStep, policy_state: types.NestedArray, seed: Optional[types.Seed] = None) \
            -> policy_step.PolicyStep:

        if self.policy_goal == 'u':
            action_ = 0
        elif self.policy_goal == 'r':
            action_ = 1
        elif self.policy_goal == 'd':
            action_ = 2
        elif self.policy_goal == 'l':
            action_ = 3
        else:
            raise ValueError("Invalid policy goal...")

        state_ = (action_, 1)  # zero indicates nop
        action_ = np.array(action_)
        info_ = ()

        return policy_step.PolicyStep(action_, state_, info_)


# policy with distribution over available options.
# options are added using policy.add_option([init, policy]) where init: obs -> p(option)
class PolicyOverNonGlobalOptions(PyPolicy):
    def __init__(self,
                 time_step_spec: ts.TimeStep,
                 action_spec: types.NestedArraySpec,
                 policy_state_spec=(ArraySpec((), np.int32), ArraySpec((), np.int32)),
                 rng_seed=150,
                 option_train_steps=5,
                 option_prob=0.3,
                 option_termination_prob=0.1,
                 adjust_sample_weights=True,
                 verbose=False,
                 show_actions=False):
        super(PolicyOverNonGlobalOptions, self).__init__(time_step_spec, action_spec, policy_state_spec)
        self._sa = show_actions
        self._outer_dims = None
        numpy.random.seed(rng_seed)
        self.rng_seed = rng_seed
        self._rng = numpy.random.RandomState(rng_seed)
        self.adjust_sample_weights = adjust_sample_weights

        self.options: [(Callable, PyPolicy)] = []           # array of [initiation_func, option]
        self.agents: [DqnAgentPseudo] = []      # array of [agents] allowing this policy access to option.train()
        self.option_prob = option_prob          # probability of executing an option at a time-step (if not already)
        self.option_termination_prob = option_termination_prob
        self.option_train_steps = option_train_steps
        self.V = verbose

    def _action(self, time_step: ts.TimeStep,
                policy_state: types.NestedArray,
                seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:

        outer_dims = self._outer_dims
        if outer_dims is None:
            outer_dims = nest_utils.get_outer_array_shape(
                time_step.observation, self._time_step_spec.observation
            )

        if policy_state[1] == 1:    # if we are in a policy, don't leave the policy unless we roll B_o()
            if self._rng.random() < self.option_termination_prob:
                if self.V:
                    tf.print("\tExiting option.")
                if self._sa:
                    tf.print("option terminate.")
            else:
                if self.V:
                    tf.print(f"\tContinuing with option {policy_state[0]}...")
                if self._sa:
                    tf.print(f"option {policy_state[0]} continue.")

                option_policy = self.options[policy_state[0]][1]
                action_, q_state_, info_ = option_policy.action(time_step, ())
                p_step = policy_step.PolicyStep(action_, (policy_state[0], np.int32(1)), info_)
                return p_step

        option_probabilities = np.zeros(len(self.options))
        for idx, opt in enumerate(self.options):
            # tf.where(~) [0][1] because we get [[n, idx]] from tf.where
            avail = opt[0](int(tf.squeeze(tf.where(time_step.observation != 0)[0][1])))
            option_probabilities[idx] = avail

        sum_probs = np.sum(option_probabilities)
        if sum_probs != 0:
            # we need to normalize option probabilities, np.random.choice won't do this for us
            option_probabilities /= sum_probs

        if self._rng.random() > self.option_prob or sum_probs == 0:
            if self.V or self._sa:
                tf.print("random action.")
            action_ = array_spec.sample_spec_nest(self._action_spec, self._rng, outer_dims=outer_dims)
            state_ = np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32)    # zero indicates nop
            info_ = array_spec.sample_spec_nest(self._info_spec, self._rng, outer_dims=outer_dims)
            return policy_step.PolicyStep(action_, state_, info_)

        else:
            # sample an option policy
            if self._sa:
                tf.print("option start.")
            option = self._rng.choice(a=np.arange(option_probabilities.shape[0]), size=1, p=option_probabilities)[0]
            if self.V:
                tf.print("option_probs: ", option_probabilities)
                tf.print(f"Executing option {option} on state {time_step.observation}")
            option_policy = self.options[option][1]
            action_, q_state_, info_ = option_policy.action(time_step, ())
            if self.V:
                tf.print("option policy: ", action_, q_state_, info_)
            p_step = policy_step.PolicyStep(action_, (option, np.int32(1)), info_)
            return p_step

    def add_option(self, initiation_function, dqn_agent):
        self.options.append((initiation_function, dqn_agent.policy))
        self.agents.append(dqn_agent)

    def train_with_batched_ds(self, experience_batched_dataset: BatchDataset):
        train_max_loss = 0

        # create an iterator over the batched dataset we have been passed
        train_data_iter = iter(experience_batched_dataset)
        for train_step in range(self.option_train_steps):
            tf.print(f"\tTrain step: {train_step}/{self.option_train_steps - 1}")

            # get the next trajectory from our batched dataset (train_traj shape: Trajectory(batch_size, inner_dims))
            try:
                train_traj = next(train_data_iter)
            except StopIteration:
                if self.V:
                    tf.print("\tAborting train (no more samples).")
                return train_max_loss
            if len(train_traj) == 2:        # if we have received a tuple of (Traj, BufferInfo)...
                train_traj = train_traj[0]  # discard the BufferInfo

            # for each agent, call pseudo-reward function, create new trajectory with pseudo-rewards, and train on it
            for ag_idx, a in enumerate(self.agents):
                # call the agent's pseudo-reward function on this trajectory
                psr_func = a.pr
                psr = psr_func(train_traj)
                new_traj = trajectory.Trajectory(train_traj.step_type, train_traj.observation, train_traj.action,
                                                 train_traj.policy_info, train_traj.next_step_type, psr,
                                                 train_traj.discount)
                this_loss = a.train_on_batch(new_traj, adjust_sample_weights=self.adjust_sample_weights)
                if train_max_loss < this_loss:
                    train_max_loss = this_loss

        return train_max_loss

    def train_on_experience_batch(self, eb: trajectory.Trajectory, train_all_policies=True):
        max_loss = 0
        for ag_idx, a in enumerate(self.agents if train_all_policies else [self.agents[0]]):
            if self.V:
                tf.print("Calling pseudo-reward function.")
            # pseudo_reward = a.pr(eb)
            if self.V:
                tf.print("Creating new trajectory.")
            new_trajectory = trajectory.Trajectory(eb.step_type, eb.observation, eb.action, eb.policy_info,
                                                   eb.next_step_type, eb.reward[:, :, ag_idx], eb.discount)
            if self.V:
                tf.print("Training agent on loss.")
            loss = a.train_on_batch(new_trajectory, adjust_sample_weights=self.adjust_sample_weights)
            max_loss = loss if loss > max_loss else max_loss
        return max_loss

    def set_mu(self, new_value):
        self.option_prob = new_value

    def _get_initial_state(self, batch_size: Optional[int] = None) -> types.NestedArray:
        if not tf.executing_eagerly():
            return tf.zeros((1,), dtype=tf.int32), tf.zeros((1,), dtype=tf.int32)
        else:
            return np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32)


class DqnAgentPseudo(DqnAgent):
    def _serialize_to_tensors(self):
        return super()._serialize_to_tensors()

    def _restore_from_tensors(self, restored_tensors):
        return super()._restore_from_tensors(restored_tensors)

    def __init__(self, time_step_spec: ts.TimeStep, action_spec: types.NestedTensorSpec,
                 q_network: network.Network, optimizer: types.Optimizer, 
                 pseudo_reward: Callable, training_batch_size, target_update_tau=1.0, target_update_period=1):
        super().__init__(time_step_spec, action_spec, q_network,
                         optimizer, epsilon_greedy=0.0,
                         td_errors_loss_fn=element_wise_squared_loss,
                         target_update_tau=target_update_tau, target_update_period=target_update_period)
        self.pr = pseudo_reward
        self.training_batch_size = training_batch_size

    def train_on_batch(self, experience_batch, adjust_sample_weights=True):
        self.train_step_counter.assign_add(1)
        if experience_batch.observation.shape[0] != self.training_batch_size:
            raise ValueError("agent.train() called on an improperly batched experience_batch...")

        if adjust_sample_weights:
            action_distribution = tf.nn.softmax(self._q_network.call(experience_batch.observation[:, 0])[0])
            taken_action = experience_batch.action[:, 0]
            indices = tf.stack((tf.range(self.training_batch_size), taken_action), axis=1)
            weights = tf.gather_nd(action_distribution, indices)
            return self.train(experience_batch, weights).loss
        else:
            return self.train(experience_batch).loss
        # below lies legacy code for training on a single element (unbatching training stuff, bad idea...)
        # for unbatched_elem in range(train_batch_size):
        #
        #     # if we have received an incomplete batch due to end of buffer
        #     if experience_batch.observation.shape[0] <= unbatched_elem:
        #         break
        #
        #     replayed_ts = ts.TimeStep(experience_batch.step_type[:][unbatched_elem],
        #                               experience_batch.reward[:][unbatched_elem],
        #                               experience_batch.discount[:][unbatched_elem],
        #                               experience_batch.observation[:][unbatched_elem])
        #
        #     # call our q-net on the observation to get the suggested q-action
        #     # index 0, 0 because 1, state_0
        #     if adjust_sample_weights:
        #         action_distribution = tf.nn.softmax(self._q_network.call(replayed_ts.observation)[0][0])
        #
        #         # index train_step, 0 because experience_batch.action is a pair of actions...
        #         actual_action = experience_batch.action[unbatched_elem, 0]
        #
        #         # weights = tf.gather(action_distribution, actual_action)
        #         weights = action_distribution[actual_action]
        #
        #         train_out = super().train(experience_batch, weights)
        #     else:
        #         train_out = super().train(experience_batch)
        #     if train_step_max_loss < train_out.loss:
        #         train_step_max_loss = train_out.loss
        #
        # return train_step_max_loss



class DdqnAgentPseudo(DdqnAgent):
    def _serialize_to_tensors(self):
        return super()._serialize_to_tensors()

    def _restore_from_tensors(self, restored_tensors):
        return super()._restore_from_tensors(restored_tensors)

    def __init__(self, time_step_spec: ts.TimeStep, action_spec: types.NestedTensorSpec,
                 q_network: network.Network, optimizer: types.Optimizer,
                 pseudo_reward: Callable, training_batch_size, target_update_tau=1.0, target_update_period=1):
        super().__init__(time_step_spec, action_spec, q_network,
                         optimizer, epsilon_greedy=0.005,
                         td_errors_loss_fn=element_wise_squared_loss,
                         target_update_tau=target_update_tau, target_update_period=target_update_period)
        self.pr = pseudo_reward
        self.training_batch_size = training_batch_size

    def train_on_batch(self, experience_batch, adjust_sample_weights=True):
        self.train_step_counter.assign_add(1)
        if experience_batch.observation.shape[0] != self.training_batch_size:
            raise ValueError("agent.train() called on an improperly batched experience_batch...")

        if adjust_sample_weights:
            action_distribution = tf.nn.softmax(self._q_network.call(experience_batch.observation[:, 0])[0])
            taken_action = experience_batch.action[:, 0]
            indices = tf.stack((tf.range(self.training_batch_size), taken_action), axis=1)
            weights = tf.gather_nd(action_distribution, indices)
            return self.train(experience_batch, weights).loss
        else:
            return self.train(experience_batch).loss
        # below lies legacy code for training on a single element (unbatching training stuff, bad idea...)
        # for unbatched_elem in range(train_batch_size):
        #
        #     # if we have received an incomplete batch due to end of buffer
        #     if experience_batch.observation.shape[0] <= unbatched_elem:
        #         break
        #
        #     replayed_ts = ts.TimeStep(experience_batch.step_type[:][unbatched_elem],
        #                               experience_batch.reward[:][unbatched_elem],
        #                               experience_batch.discount[:][unbatched_elem],
        #                               experience_batch.observation[:][unbatched_elem])
        #
        #     # call our q-net on the observation to get the suggested q-action
        #     # index 0, 0 because 1, state_0
        #     if adjust_sample_weights:
        #         action_distribution = tf.nn.softmax(self._q_network.call(replayed_ts.observation)[0][0])
        #
        #         # index train_step, 0 because experience_batch.action is a pair of actions...
        #         actual_action = experience_batch.action[unbatched_elem, 0]
        #
        #         # weights = tf.gather(action_distribution, actual_action)
        #         weights = action_distribution[actual_action]
        #
        #         train_out = super().train(experience_batch, weights)
        #     else:
        #         train_out = super().train(experience_batch)
        #     if train_step_max_loss < train_out.loss:
        #         train_step_max_loss = train_out.loss
        #
        # return train_step_max_loss


def reshape_reward(traj: trajectory.Trajectory, info: BufferInfo):
    pr = pseudo_up(traj)
    # pr = pseudo_top(traj)
    new_traj = trajectory.Trajectory(traj.step_type, traj.observation,
                                     traj.action, traj.policy_info,
                                     traj.next_step_type, pr, traj.discount)
    return new_traj, info


def initialize_directional_agents():
    up_net = QNetwork(env.observation_spec(),
                      env.action_spec(),
                      fc_layer_params=(32, 32),
                      q_layer_activation_fn=tf.nn.softmax)
    up_agent = DqnAgentPseudo(env.time_step_spec(), env.action_spec(), up_net,
                              optimizer=Adam(),
                              pseudo_reward=pseudo_up,
                              training_batch_size=train_batch_size)

    down_net = QNetwork(env.observation_spec(),
                        env.action_spec(),
                        fc_layer_params=(32, 32),
                        q_layer_activation_fn=tf.nn.softmax)
    down_agent = DqnAgentPseudo(env.time_step_spec(), env.action_spec(), down_net,
                                optimizer=Adam(),
                                pseudo_reward=pseudo_down,
                                training_batch_size=train_batch_size)
    left_net = QNetwork(env.observation_spec(),
                        env.action_spec(),
                        fc_layer_params=(32, 32),
                        q_layer_activation_fn=tf.nn.softmax)
    left_agent = DqnAgentPseudo(env.time_step_spec(), env.action_spec(), left_net,
                                optimizer=Adam(),
                                pseudo_reward=pseudo_left,
                                training_batch_size=train_batch_size)
    
    right_net = QNetwork(env.observation_spec(),
                         env.action_spec(),
                         fc_layer_params=(32, 32),
                         q_layer_activation_fn=tf.nn.softmax)
    right_agent = DqnAgentPseudo(env.time_step_spec(), env.action_spec(),
                                 right_net,
                                 optimizer=Adam(),
                                 pseudo_reward=pseudo_right,
                                 training_batch_size=train_batch_size)

    return up_agent, down_agent, left_agent, right_agent


if __name__ == '__main__':
    environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TF-Metal throws loads of useless info to stderr. Get rid of this.
    tf.print("Creating environment and agents.")
    env = FourRooms.FourRooms()
    env = tf_py_environment.TFPyEnvironment(env)

    ngo_policy = PolicyOverNonGlobalOptions(env.time_step_spec(), env.action_spec(), verbose=verbose_agent)
    # go_pol = TFPyPolicy(go_pol)
    test_agents = []

    directional_agents = initialize_directional_agents()
    
    buf2 = TFUniformReplayBuffer(directional_agents[0].collect_policy.collect_data_spec, 1, max_length=10000)
    driver = DynamicStepDriver(env, ngo_policy, observers=[buf2.add_batch])

    initialization_functions = (lambda x: int(x < 39), lambda x: int(x > 13*10),
                                lambda x: int(x % 13 >= 10), lambda x: int(x % 13 <= 2))

    terms = [lambda: np.random.random() < 1/10] * 4

    for i in range(4):
        test_agents.append((initialization_functions[i], directional_agents[i]))

    step = env.reset()
    p_state = np.zeros(2)
    tf.print("Beginning trajectory generation.")

    for i in range(num_steps):
        if verbose:
            tf.print(f"Explore step {i}")
        step, p_state = driver.run(step, p_state)
        if i % print_step_interval == 0 and i != 0:
            tf.print(f"\tgen step: {i}/{num_steps}")

        if i % add_and_train_policies_interval == 0 and i != 0:
            if len(test_agents) > 0:
                tf.print("Augmenting NGO Meta-Policy with new policy.")
                init_fn, agent_to_add = test_agents.pop(0)
                ngo_policy.add_option(init_fn, agent_to_add)

            tf.print("Training Q-Policy off-policy.")

            # convert experience replay buffer to dataset, batch it into training_batches, and reshape the reward
            buf_ds = buf2.as_dataset(None, 2, single_deterministic_pass=True)
            batched_dataset = buf_ds.batch(train_batch_size, drop_remainder=True)

            ngo_policy.train_with_batched_ds(batched_dataset)
            tf.print("DQN Option Training Done. Regenerating trajectory.")

        # if i > 1000:
            # env.render('human')

        i += 1

    step = env.reset()
    p_state = np.zeros(2)
    tf.print("Beginning exam step.")
    for ex in range(exam_steps):
        step, p_state = driver.run(step, p_state)
        env.render('human')
