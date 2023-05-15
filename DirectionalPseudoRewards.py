from tf_agents.trajectories import trajectory
import tensorflow as tf


def pseudo_up(traj: trajectory.Trajectory):
    tf.print("\tCalculating pseudo-reward for new batch.")
    # trajectory should be of shape (train_batch_size, 2, [inner_dims])
    observations = traj.observation
    # tf.print("obs shape: ", observations.shape)

    obs0 = observations[:, 0]
    obs1 = observations[:, 1]

    s0 = tf.where(obs0 == 1)[:, 1]     # (batch_size, )
    s1 = tf.where(obs1 == 1)[:, 1]     # (batch_size, )
    # tf.print("s0 shape: ", s0.shape)

    x0 = tf.math.mod(s0, 13)
    x1 = tf.math.mod(s1, 13)
    # tf.print("x0 shape: ", x0.shape)

    y0 = tf.divide(s0, 13)
    y1 = tf.divide(s1, 13)
    # tf.print("y0 shape: ", y0.shape)

    y0 = tf.floor(y0)
    y1 = tf.floor(y1)

    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)
    delta_y = y1 - y0
    rwd_del_y = tf.map_fn(lambda x: x if x > 0 else -10., delta_y)

    # append a zero to the last experience of the batch (b.c. we don't know what to reward it)
    rwd_del_y2 = tf.concat((rwd_del_y[1:], tf.zeros(1)), 0)

    rewards = tf.stack((rwd_del_y, rwd_del_y2), axis=1)
    return rewards


def pseudo_down(traj: trajectory.Trajectory):
    tf.print("\tCalculating pseudo-reward for new batch.")
    # trajectory should be of shape (train_batch_size, 2, [inner_dims])
    observations = traj.observation
    # tf.print("obs shape: ", observations.shape)

    obs0 = observations[:, 0]
    obs1 = observations[:, 1]

    s0 = tf.where(obs0 == 1)[:, 1]  # (batch_size, )
    s1 = tf.where(obs1 == 1)[:, 1]  # (batch_size, )
    # tf.print("s0 shape: ", s0.shape)

    x0 = tf.math.mod(s0, 13)
    x1 = tf.math.mod(s1, 13)
    # tf.print("x0 shape: ", x0.shape)

    y0 = tf.divide(s0, 13)
    y1 = tf.divide(s1, 13)
    # tf.print("y0 shape: ", y0.shape)

    y0 = tf.floor(y0)
    y1 = tf.floor(y1)

    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)
    delta_y = y0 - y1
    rwd_del_y = tf.map_fn(lambda x: x if x > 0 else -10., delta_y)

    # append a zero to the last experience of the batch (b.c. we don't know what to reward it)
    rwd_del_y2 = tf.concat((rwd_del_y[1:], tf.zeros(1)), 0)

    rewards = tf.stack((rwd_del_y, rwd_del_y2), axis=1)
    return rewards


def pseudo_left(traj: trajectory.Trajectory):
    tf.print("\tCalculating pseudo-reward for new batch.")
    # trajectory should be of shape (train_batch_size, 2, [inner_dims])
    observations = traj.observation
    # tf.print("obs shape: ", observations.shape)

    obs0 = observations[:, 0]
    obs1 = observations[:, 1]

    s0 = tf.where(obs0 == 1)[:, 1]  # (batch_size, )
    s1 = tf.where(obs1 == 1)[:, 1]  # (batch_size, )
    # tf.print("s0 shape: ", s0.shape)

    x0 = tf.math.mod(s0, 13)
    x1 = tf.math.mod(s1, 13)
    # tf.print("x0 shape: ", x0.shape)

    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    delta_x = x0 - x1
    rwd_del_x = tf.map_fn(lambda x: x if x > 0 else -10., delta_x)

    # append a zero to the last experience of the batch (b.c. we don't know what to reward it)
    rwd_del_x2 = tf.concat((rwd_del_x[1:], tf.zeros(1)), 0)

    rewards = tf.stack((rwd_del_x, rwd_del_x2), axis=1)
    return rewards


def pseudo_right(traj: trajectory.Trajectory):
    tf.print("\tCalculating pseudo-reward for new batch.")
    # trajectory should be of shape (train_batch_size, 2, [inner_dims])
    observations = traj.observation
    # tf.print("obs shape: ", observations.shape)

    obs0 = observations[:, 0]
    obs1 = observations[:, 1]

    s0 = tf.where(obs0 == 1)[:, 1]  # (batch_size, )
    s1 = tf.where(obs1 == 1)[:, 1]  # (batch_size, )
    # tf.print("s0 shape: ", s0.shape)

    x0 = tf.math.mod(s0, 13)
    x1 = tf.math.mod(s1, 13)
    # tf.print("x0 shape: ", x0.shape)

    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)

    delta_x = x1 - x0
    rwd_del_x = tf.map_fn(lambda x: x if x > 0 else -10., delta_x)

    # append a zero to the last experience of the batch (b.c. we don't know what to reward it)
    rwd_del_x2 = tf.concat((rwd_del_x[1:], tf.zeros(1)), 0)

    rewards = tf.stack((rwd_del_x, rwd_del_x2), axis=1)
    return rewards
