import numpy as np
import torch
import ipdb


def matt_loss_torch(phase, mask, eta=.5):
    # Get image parameters

    # Pixels per group
    num_group_el = mask.sum(-1)
    # Number of groups
    num_groups = (num_group_el > 0).sum(1).unsqueeze(1).float()
    # Normalization factor for each group, even if empty
    per_group_norm = torch.where(num_group_el == 0, torch.ones_like(num_group_el), num_group_el)
    # Image size
    img_size = phase.shape[1]
    # Mask phase: dim 1 is group index
    masked_phase = phase.unsqueeze(1) * mask
    # Group real part
    group_real = (torch.cos(masked_phase).sum(-1) - (img_size - num_group_el)) / per_group_norm
    # Group imag part
    group_imag = torch.sin(masked_phase).sum(-1) / per_group_norm


    # Mean field unnormalized per group
    mean_field = (group_real**2 + group_imag**2)

    # Synch loss is 1 - mean_field averaged across batch
    synch_loss = 1 - (mean_field.sum(1) / num_groups).mean()

    # Mask for where group angle is undefined
    und_mask = (group_real==0)*(group_imag==0) 
    # Temporarily replace imag part at masked indices
    tmp_group_imag = torch.where(und_mask, torch.ones_like(group_imag), group_imag)

    # Calculate phase of group 
    group_phase = torch.atan2(tmp_group_imag,group_real)

    # Mask all phase pairs where phase is defined
    desynch_mask = ((1 - und_mask).unsqueeze(1) * (1 - und_mask).unsqueeze(2)).float()

    # Phase diffs
    group_phase_diffs = desynch_mask * torch.abs(torch.cos(group_phase.unsqueeze(1) - group_phase.unsqueeze(2)))**2

    # Desynch loss is frame potential between groups normalized to be between 0 and 1
    desynch_loss = (group_phase_diffs.sum((1,2)) / num_groups**2 - (1/ num_groups)).mean()

    # Total loss is convex combination of the two losses
    tot_loss = eta*synch_loss + (1-eta)*desynch_loss
    return tot_loss, synch_loss, desynch_loss

def matt_loss_np(phase,mask,eta=.5):
    t = []
    s = []
    d = []

    ma = torch.tensor(mask)
    for i in range(phase.shape[0]):
        ph = torch.tensor(phase[i,:]).unsqueeze(0)
        total, synch, desynch = matt_loss_torch(ph, ma, eta=eta)
        t.append(total.data.numpy())
        s.append(synch.data.numpy())
        d.append(desynch.data.numpy())
    return np.array(t), np.array(s), np.array(d)

def frame_pt_np(phase):
    diffs = np.expand_dims(phase, 1) - np.expand_dims(phase, 2)
    return np.round(np.mean(np.cos(diffs) ** 2, axis=(1, 2)), 5)


def fpt_btw_groups_np(phase, mask):
    masked_phase = np.expand_dims(phase, axis=1) * mask

    sin = np.sin(masked_phase)
    cos = np.where(masked_phase == 0., masked_phase, np.cos(masked_phase))

    sin_mean = div_no_nan_np(np.sum(sin, axis=2), np.sum(mask, axis=2))
    cos_mean = div_no_nan_np(np.sum(cos, axis=2), np.sum(mask, axis=2))

    sin_mat = np.matmul(np.expand_dims(sin_mean, axis=2),
                        np.expand_dims(sin_mean, axis=1))
    cos_mat = np.matmul(np.expand_dims(cos_mean, axis=2),
                        np.expand_dims(cos_mean, axis=1))

    diag_mat = np.expand_dims(1 - np.eye(sin_mat.shape[1], sin_mat.shape[1]),
                              axis=0)
    sin_mat = sin_mat * diag_mat
    cos_mat = cos_mat * diag_mat

    return np.round(((sin_mat + cos_mat) ** 2).mean(2).mean(1), 5)


def fpt_btw_groups_torch(phase, mask):
    masked_phase = phase.unsqueeze(1) * mask

    sin = torch.sin(masked_phase)
    cos = torch.where(masked_phase == 0., masked_phase, torch.cos(masked_phase))

    sin_mean = div_no_nan_torch(sin.sum(2), mask.sum(2))
    cos_mean = div_no_nan_torch(cos.sum(2), mask.sum(2))

    sin_mat = torch.matmul(sin_mean.unsqueeze(2), sin_mean.unsqueeze(1))
    cos_mat = torch.matmul(cos_mean.unsqueeze(2), cos_mean.unsqueeze(1))

    diag_mat = (1 - torch.eye(sin_mat.shape[1], sin_mat.shape[1])).unsqueeze(0)

    sin_mat = sin_mat * diag_mat
    cos_mat = cos_mat * diag_mat

    return ((sin_mat + cos_mat) ** 2).mean(2).mean(1)


def frame_pt_torch(phase):
    valid_num = phase.shape[1] ** 2 - phase.shape[1]
    diffs = phase.unsqueeze(1) - phase.unsqueeze(2)
    return (torch.cos(diffs) ** 2).sum(2).sum(1) / valid_num


def inp_np(phase):
    diffs = np.expand_dims(phase, 1) - np.expand_dims(phase, 2)
    return np.round(np.mean(np.cos(diffs), axis=(1, 2)), 5)


def inp_btw_groups_np(phase, mask):
    groups_size = np.sum(mask, axis=2)
    groups_size_mat = np.matmul(np.expand_dims(groups_size, 2),
                                np.expand_dims(groups_size, 1))
    groups_size_sum = groups_size_mat.sum(2).sum(1) - np.sum(groups_size ** 2, axis=1)

    masked_sin = np.sin(np.expand_dims(phase, axis=1)) * mask
    masked_cos = np.cos(np.expand_dims(phase, axis=1)) * mask

    in_groups_product = np.matmul(np.expand_dims(masked_sin, axis=3),
                                  np.expand_dims(masked_sin, axis=2)) +\
        np.matmul(np.expand_dims(masked_cos, axis=3),
                  np.expand_dims(masked_cos, axis=2))

    product = np.matmul(np.expand_dims(np.expand_dims(masked_sin, axis=2), axis=4),
                        np.expand_dims(np.expand_dims(masked_sin, axis=1), axis=3)) +\
        np.matmul(np.expand_dims(np.expand_dims(masked_cos, axis=2), axis=4),
                  np.expand_dims(np.expand_dims(masked_cos, axis=1), axis=3))

    product = np.sum(product, axis=2, keepdims=True) -\
              np.expand_dims(in_groups_product, axis=2)
    #return np.round(np.exp(np.squeeze(product, axis=2).sum(3).sum(2).sum(1)\
           #/ 2 / [*torch.combinations(torch.arange(osci_num)).shape][0]), 5)
    return np.round(np.squeeze(product, axis=2).sum(3).sum(2).sum(1) / groups_size_sum, 5)


def exinp_btw_groups_torch(phase, mask):
    groups_size = torch.sum(mask, dim=2)
    groups_size_mat = torch.matmul(groups_size.unsqueeze(2),
                                   groups_size.unsqueeze(1))
    groups_size_sum = groups_size_mat.sum(2).sum(1) - torch.sum(groups_size ** 2, dim=1)

    masked_sin = torch.sin(phase.unsqueeze(1)) * mask
    masked_cos = torch.cos(phase.unsqueeze(1)) * mask

    in_groups_product = torch.matmul(masked_sin.unsqueeze(3),
                                     masked_sin.unsqueeze(2)) + \
                        torch.matmul(masked_cos.unsqueeze(3),
                                     masked_cos.unsqueeze(2))

    product = torch.matmul(masked_sin.unsqueeze(2).unsqueeze(4),
                           masked_sin.unsqueeze(1).unsqueeze(3)) + \
              torch.matmul(masked_cos.unsqueeze(2).unsqueeze(4),
                           masked_cos.unsqueeze(1).unsqueeze(3))

    product = torch.sum(product, dim=2, keepdim=True) - \
              in_groups_product.unsqueeze(2)
    # return np.round(np.exp(np.squeeze(product, axis=2).sum(3).sum(2).sum(1)\
    # / 2 / [*torch.combinations(torch.arange(osci_num)).shape][0]), 5)
    return torch.exp(torch.squeeze(product, dim=2).sum(3).sum(2).sum(1) / groups_size_sum)


def exp_inp_torch(phase):
    valid_num = phase.shape[1] ** 2 - phase.shape[1]
    diffs = phase.unsqueeze(1) - phase.unsqueeze(2)
    return torch.exp(torch.cos(diffs)).sum(2).sum(1) / valid_num


def coherence_np(phase):
    sin_mean = np.mean(np.sin(phase), axis=1)
    cos_mean = np.mean(np.cos(phase), axis=1)
    return np.round(np.sqrt(sin_mean ** 2 + cos_mean ** 2), 5)


def coh_btw_groups_np(phase, mask):
    masked_sin = np.sin(np.expand_dims(phase, axis=1)) * mask
    masked_cos = np.cos(np.expand_dims(phase, axis=1)) * mask

    sin_mean = div_no_nan_np(masked_sin.sum(2), mask.sum(2))
    cos_mean = div_no_nan_np(masked_cos.sum(2), mask.sum(2))

    return np.round(sin_mean.mean(1) ** 2 + cos_mean.mean(1) ** 2, 5)


def coh_in_groups_np(phase, mask):
    masked_sin = np.sin(np.expand_dims(phase, axis=1)) * mask
    masked_cos = np.cos(np.expand_dims(phase, axis=1)) * mask

    sin_mean = div_no_nan_np(masked_sin.sum(2), mask.sum(2))
    cos_mean = div_no_nan_np(masked_cos.sum(2), mask.sum(2))
    return np.round((sin_mean ** 2 + cos_mean ** 2).mean(1), 5)


def coh_btw_groups_torch(phase, mask):
    masked_sin = torch.sin(phase.unsqueeze(1)) * mask
    masked_cos = torch.cos(phase.unsqueeze(1)) * mask

    sin_mean = div_no_nan_torch(masked_sin.sum(2), mask.sum(2))
    cos_mean = div_no_nan_torch(masked_cos.sum(2), mask.sum(2))

    return sin_mean.mean(1) ** 2 + cos_mean.mean(1) ** 2


def coh_in_groups_torch(phase, mask):
    masked_sin = torch.sin(phase.unsqueeze(1)) * mask
    masked_cos = torch.cos(phase.unsqueeze(1)) * mask

    sin_mean = div_no_nan_torch(masked_sin.sum(2), mask.sum(2))
    cos_mean = div_no_nan_torch(masked_cos.sum(2), mask.sum(2))

    return (sin_mean ** 2 + cos_mean ** 2).mean(1)


def coherence_torch(phase):
    sin_mean = torch.mean(torch.sin(phase), dim=1)
    cos_mean = torch.mean(torch.cos(phase), dim=1)
    return sin_mean ** 2 + cos_mean ** 2


def abs_angle_diffs_np(phase):
    phase = np.squeeze(phase)
    diffs = np.expand_dims(phase, 1) - np.expand_dims(phase, 2)
    return np.round(.5 * np.where(diffs < np.pi, np.abs(diffs),
                    2 * np.pi - np.abs(diffs)).mean(2).mean(1), 5)


def abs_angle_diffs_torch(phase):
    diffs = phase.unsqueeze(1) - phase.unsqueeze(2)
    # return torch.abs(torch.sin(.5*diffs)).mean()
    return .5 * torch.where(diffs < np.pi, torch.abs(diffs),
                            2 * np.pi - torch.abs(diffs)).mean(2).mean(1)


def div_no_nan_np(tensor1, tensor2):
    return np.where(tensor2 == 0., tensor2, tensor1 / tensor2)


def div_no_nan_torch(tensor1, tensor2):
    return torch.where(tensor2 == 0., tensor2, tensor1 / tensor2)
