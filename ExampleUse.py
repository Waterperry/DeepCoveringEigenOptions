from pygame.event import get as pygame_event_get
from tf_agents.environments import tf_py_environment
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
import DceoPolicy
import FourRooms

DceoPolicy.dceo_batch_size = dceo_batch_size = 128


def run_dceo_on_four_rooms(num_explore_episodes=100, num_explore_steps=1000, num_eval_episodes=100):
    # create our Python and TF Environments
    py_env = FourRooms.FourRooms()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)

    # create our DceoPolicy using information from the TENSORFLOW environment. Also, some custom parameters.
    agent = DceoPolicy.DceoPolicy(tf_env.time_step_spec(), tf_env.action_spec(), tf_env.observation_spec(),
                                  num_options=4, main_policy_train_steps=1,
                                  option_train_steps=1, mu=0.1, D=5.0, dceo_batch_size=dceo_batch_size,
                                  dceo_neuron_count=16, replay_buffer_max_size=num_explore_steps*num_explore_episodes,
                                  show_nn_train_output=True)

    # create a driver for our evaluation phase. Note that we use agent.add_batch - this method adds to the state
    #   visitation hash-table. This is necessary for defining state visitation. This can be transformed by
    #   passing a lambda hash function to our DCEO Policy.
    driver = DynamicStepDriver(tf_env, agent.get_exploit_policy(), observers=[agent.add_batch])

    # create our driver for the exploration phase
    explore_driver = DynamicStepDriver(tf_env, agent.get_explore_policy(), observers=[agent.add_batch])

    print("Beginning initial trajectory generation. This may take some time...")
    for explore_ep in range(num_explore_episodes):
        step_number = 0
        step = tf_env.reset()

        # set the initial state, which is usually just (0, 0)
        policy_state = agent.get_initial_state()
        while step_number <= num_explore_steps:
            step, policy_state = explore_driver.run(step, policy_state)
            step_number += 1

        print(f"END explore episode {explore_ep}.")
        # now that we have filled the agent's replay buffer with some sample state transitions, allow it to train
        #   its internal representation.
        agent.train_ef_representation(explore_ep, visualize_ef=True, env='fourrooms')

    for ep_number in range(num_eval_episodes):
        step = tf_env.reset()
        policy_state = agent.get_initial_state()

        reward = 0.
        while not step.is_last():
            step, policy_state = driver.run(step, policy_state)
            reward += step.reward

            tf_env.render(mode='human')
            pygame_event_get()  # workaround to prevent window freezing

            if agent.get_rb().num_frames() >= agent.get_batch_size():
                agent.eagerly_train_policies()

        print(f"{ep_number}: {reward}")


if __name__ == '__main__':
    run_dceo_on_four_rooms(num_explore_episodes=10, num_explore_steps=1000)
