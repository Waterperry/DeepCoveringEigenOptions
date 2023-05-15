from typing import Optional
import numpy as np
import tensorflow as tf
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.trajectories import time_step as ts, policy_step
from tf_agents.typing import types
from tf_agents.utils import nest_utils


class EGreedyPolicyWrapper(PyPolicy):
    def __init__(self, time_step_spec: ts.TimeStep, action_spec: types.NestedArraySpec,
                 random_policy, greedy_policy, epsilon=0.1, rng_seed=None, verbose=False, policy_state_spec=(),
                 expand_greedy_dims=False, show_actions=False):
        super().__init__(time_step_spec, action_spec, policy_state_spec=policy_state_spec)
        self.epsilon = epsilon
        self.random_policy = random_policy
        self.greedy_policy = greedy_policy
        self._rng = np.random.RandomState(seed=rng_seed)
        self._V = verbose
        self._egd = expand_greedy_dims
        self._sa = show_actions

    def _action(self, time_step: ts.TimeStep, policy_state: types.NestedArray,
                seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:
        if self._V:
            tf.print("Sampling action from E-Greedy Policy Wrapper.")

        if policy_state[1] >= 1:       # if we are in an option, stay exploring
            if self._V:
                tf.print(f"\t({policy_state[1]} left) Dropping through to option continue.")
            if self._sa:
                tf.print("option cont.")
            return self.random_policy.action(time_step, policy_state, seed)
        if self._rng.random() < self.epsilon:
            if self._V:
                tf.print("\tRolled less than epsilon, sampling random exploratory action/option.")
            if self._sa:
                tf.print("explore.")
            return self.random_policy.action(time_step, (np.int32(0), np.int32(0)), seed)
        else:
            # self.greedy_policy is (D)DQN, so we need to transform policy_state to reflect not being in an option
            if self._V:
                tf.print("\tSample greedy action.")
            if self._sa:
                tf.print("greedy.")
            if self._egd:
                action_, _, info_ = nest_utils.unbatch_nested_array(
                    self.greedy_policy.action(nest_utils.batch_nested_array(time_step), policy_state, seed))
            else:
                action_, _, info_ = self.greedy_policy.action(time_step, policy_state, seed)

            state_ = (np.int32(0), np.int32(0))
            return policy_step.PolicyStep(action_, state_, info_)

    def get_wrapped_policy(self):
        return self.greedy_policy
