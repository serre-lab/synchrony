import tensorflow as tf
import numpy as np


def matt_loss(phase, target):
    # Synchrony loss based on absolute value of angles
    diffs = tf.expand_dims(phase, axis=1) - tf.expand_dims(phase, axis=2)
    diffs = tf.mod(tf.abs(diffs), tf.constant(np.pi, dtype=tf.float32))
    # diffs.shape=(batch, N, N)
    # do not count diags
    N = phase.get_shape().as_list()[1]
    valid_num = N ** 2 - N
    sync_loss = tf.reduce_mean(tf.reduce_sum(.5 * diffs, axis=[1, 2]) / valid_num)

    mask = tf.cast(target, tf.float32)
    masked_phase = tf.multiply(tf.expand_dims(phase, axis=1), mask)

    # self.masked_phase.shape=(batch, groups, N)
    sin_vec = tf.sin(masked_phase)
    cos_vec = tf.where(tf.equal(mask, tf.constant(0.)), masked_phase, tf.cos(masked_phase))

    sin_mean = tf.div_no_nan(tf.reduce_sum(sin_vec, axis=2), tf.reduce_sum(mask, axis=2))
    cos_mean = tf.div_no_nan(tf.reduce_sum(cos_vec, axis=2), tf.reduce_sum(mask, axis=2))
    # mean.shape=(batch, groups)

    sin_mat = tf.matmul(tf.expand_dims(sin_mean, axis=2), tf.expand_dims(sin_mean, axis=1))
    cos_mat = tf.matmul(tf.expand_dims(cos_mean, axis=2), tf.expand_dims(cos_mean, axis=1))

    # diags should be zeros
    diag = tf.zeros(shape=sin_mean.shape)
    sin_mat = tf.matrix_set_diag(sin_mat, diag)
    cos_mat = tf.matrix_set_diag(cos_mat, diag)

    desync_loss = 150 * tf.reduce_mean(tf.reduce_mean(tf.square(tf.add(sin_mat, cos_mat)), axis=[1, 2]) / valid_num)

    tot_loss = 0.2 * sync_loss + desync_loss
    return sync_loss, desync_loss, tot_loss


def coherence_loss(phase, target):
    # This loss function refers to wiki page of Directional Statistics
    # So first you have to find Re^(i*theta) within the group
    # not tough to implement
    mask = tf.cast(target, tf.float32)
    masked_phase = tf.multiply(tf.expand_dims(phase, axis=1), mask)

    # self.masked_phase.shape=(batch, groups, N)
    sin_vec = tf.sin(masked_phase)
    cos_vec = tf.where(tf.equal(mask, tf.constant(0.)), masked_phase, tf.cos(masked_phase))

    sin_mean = tf.div_no_nan(tf.reduce_sum(sin_vec, axis=2), tf.reduce_sum(mask, axis=2))
    cos_mean = tf.div_no_nan(tf.reduce_sum(cos_vec, axis=2), tf.reduce_sum(mask, axis=2))

    std = tf.sqrt(- 2 * log_no_nan(tf.sqrt(tf.square(sin_mean) + tf.square(cos_mean))))
    sync_loss = mean_no_zero(std, axis=1)
    # sync_loss.shape=(batch,)

    desync_loss = tf.tan(tf.sqrt(tf.square(mean_no_zero(sin_mean, axis=1)) +
                         tf.square(mean_no_zero(cos_mean, axis=1))))

    # desync_loss.shape=(batch,)
    sync_loss_mean = tf.reduce_mean(sync_loss)
    desync_loss_mean = tf.reduce_mean(desync_loss)
    tot_loss_mean = 0.2 * sync_loss_mean + desync_loss_mean
    return sync_loss_mean, desync_loss_mean, tot_loss_mean


def log_no_nan(tensor):
    return tf.where(tf.equal(tensor, 0.), tensor, tf.log(tensor))


def mean_no_zero(tensor, axis=None):
    # default all axis
    if axis is None:
        return tf.div(tf.reduce_sum(tensor), tf.reduce_sum(tf.abs(tf.sign(tensor))))
    else:
        return tf.div(tf.reduce_sum(tensor, axis=axis), tf.reduce_sum(tf.abs(tf.sign(tensor)), axis=axis))


def phase_up(phase, coupling_mat):
    """
    :return: update self.phase based on Kuramoto
    """
    diffs = tf.expand_dims(phase, axis=1) - tf.expand_dims(phase, axis=2)

    # diffs.shape=(batch, N, N)
    delta = tf.reduce_mean(tf.multiply(coupling_mat, tf.sin(diffs)), axis=2)

    # delta.shape=(batch, N)
    delta = .1 * delta

    # should not change the shape of phase and freq
    new_phase = tf.mod(tf.add(phase, delta), tf.constant(2*np.pi, dtype=tf.float32))

    return new_phase


@tf.function
def evolution(timestep, batch, N, coupling):
    # try to use zero intrinsic freq, good to start with
    phase = tf.random.uniform(maxval=2 * np.pi, shape=[batch, N])
    i = tf.constant(0)
    while i < timestep:
        phase = phase_up(phase, coupling)
        i = tf.add(i, tf.constant(1))
    return phase
