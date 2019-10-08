import numpy as np
import torch
import torch.distributions.uniform as uniform
import torch.distributions.cauchy as cauchy


def phase_up(phase, coupling_mat):
    diffs = phase.unsqueeze(1) - phase.unsqueeze(2)
    # For now intrinsic frequency just zeros
    delta = (coupling_mat * torch.sin(diffs)).sum(2)
    # print('phase', phase)
    # print('diffs', diffs)
    # print('delta', delta)
    return (phase + .1 * delta) % (2 * np.pi)


def evolution(timestep, coupling_mat, batch, N):
    m1 = uniform.Uniform(torch.tensor([0.0]), torch.tensor([2 * np.pi]))
    phase = m1.sample(sample_shape=torch.tensor([batch, N]))
    phase = phase.reshape([batch, N])

    #m2 = uniform.Uniform(torch.tensor([-0.1]), torch.tensor([0.1]))
    #pulse = m2.sample(sample_shape=torch.tensor([batch, N]))
    i = torch.tensor([0])
    while i < timestep:
        phase = phase_up(phase, coupling_mat)
        i = i + 1
    # return final state
    #pulse = pulse.reshape([batch, N])
    return phase


def matt_loss(phase, mask):
    diffs = phase.unsqueeze(1) - phase.unsqueeze(2)
    # return torch.abs(torch.sin(.5*diffs)).mean()
    synch_loss = .5 * torch.where(diffs < np.pi, torch.abs(diffs), 2 * np.pi - torch.abs(diffs)).mean()

    masked_phase = torch.mul(phase.unsqueeze(1), mask)
    sin_vec = torch.sin(masked_phase)
    cos_vec = torch.where(masked_phase == 0., masked_phase, torch.cos(masked_phase))

    # avg over groups
    sin_mean = div_no_nan(torch.sum(sin_vec, dim=2), torch.sum(mask, dim=2))
    cos_mean = div_no_nan(torch.sum(cos_vec, dim=2), torch.sum(mask, dim=2))

    sin_mat = torch.matmul(sin_mean.unsqueeze(2), sin_mean.unsqueeze(1))
    cos_mat = torch.matmul(cos_mean.unsqueeze(2), cos_mean.unsqueeze(1))

    diag_mat = (1 - torch.eye(sin_mat.shape[1], sin_mat.shape[1])).unsqueeze(0)
    sin_mat = sin_mat * diag_mat
    cos_mat = cos_mat * diag_mat

    desync_loss = ((sin_mat + cos_mat) ** 2).mean()

    tot_loss = 0.2 * synch_loss + desync_loss
    return tot_loss, synch_loss, desync_loss


def coherence_loss(phase, mask):
    masked_phase = torch.mul(phase.unsqueeze(1), mask)

    sin_vec = torch.sin(masked_phase)
    cos_vec = torch.where(masked_phase == torch.tensor(0.), masked_phase, torch.cos(masked_phase))

    # avg over groups
    sin_mean = torch.div(torch.sum(sin_vec, dim=2), torch.sum(mask, dim=2))
    cos_mean = torch.div(torch.sum(cos_vec, dim=2), torch.sum(mask, dim=2))

    std = - torch.log((sin_mean ** 2 + cos_mean ** 2) * 0.99 + 0.001)
    synch_loss = torch.mean(std)

    desynch_loss = torch.tan(torch.mean(sin_mean, dim=1) ** 2 +
                             torch.mean(cos_mean, dim=1) ** 2)
    """
    sin_mat = torch.matmul(sin_mean.unsqueeze(2), sin_mean.unsqueeze(1))
    cos_mat = torch.matmul(cos_mean.unsqueeze(2), cos_mean.unsqueeze(1))

    diag_mat = (1 - torch.eye(sin_mat.shape[1], sin_mat.shape[1])).unsqueeze(0)
    sin_mat = sin_mat * diag_mat
    cos_mat = cos_mat * diag_mat

    valid_num = sin_mat.shape[1] ** 2 - sin_mat.shape[1]

    desynch_loss = ((sin_mat + cos_mat) ** 2).sum(2).sum(1) / valid_num
    """
    # avg over batch
    sync_loss_mean = synch_loss.mean(0)
    desync_loss_mean = desynch_loss.mean(0)
    tot_loss_mean = 0.01 * sync_loss_mean + desync_loss_mean
    return tot_loss_mean, sync_loss_mean, desync_loss_mean


def div_no_nan(tensor1, tensor2):
    # works the same as tf.div_no_nan
    return torch.where(tensor2 != 0., tensor1 / tensor2, torch.zeros_like(tensor1))


def log_no_nan(tensor):
    return torch.where(tensor == 0., tensor, torch.log(tensor))


def mean_no_zero(tensor, axis=None):
    # default all axis
    if axis is None:
        return torch.sum(tensor) / torch.sum(torch.abs(torch.sign(tensor)))
    else:
        return torch.sum(tensor, dim=axis) / torch.sum(torch.abs(torch.sign(tensor)), dim=axis)


def IP_loss(phase, mask):
    osci_num = phase.shape[1]

    phase_sin = torch.sin(phase)
    phase_cos = torch.cos(phase)

    masked_sin = phase_sin.unsqueeze(1) * mask
    masked_cos = phase_cos.unsqueeze(1) * mask
    # mask.shape=(batch, groups, N)

    product1 = torch.matmul(masked_sin.unsqueeze(3), masked_sin.unsqueeze(2)) +\
               torch.matmul(masked_cos.unsqueeze(3), masked_cos.unsqueeze(2))

    sync_loss = (product1.sum(3).sum(2) - osci_num) / (osci_num ** 2 - osci_num)

    product2 = torch.matmul(masked_sin.unsqueeze(2).unsqueeze(4), masked_sin.unsqueeze(1).unsqueeze(3)) +\
               torch.matmul(masked_cos.unsqueeze(2).unsqueeze(4), masked_cos.unsqueeze(1).unsqueeze(3))

    product2 = product2.sum(2, keepdim=True) - product1.unsqueeze(2)
    desync_loss = product2.squeeze().sum(3).sum(2).sum(1) / 2 / torch.combinations(torch.arange(osci_num)).shape[0]

    sync_loss_mean = torch.exp(- sync_loss.mean())
    desync_loss_mean = torch.exp(desync_loss.mean())

    tot_loss_mean = 0.1 * sync_loss_mean + desync_loss_mean
    return tot_loss_mean, sync_loss_mean, desync_loss_mean
