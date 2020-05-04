import numpy as np
import torch
import ipdb

def just_kuramoto(phase, device):
    import pdb; pdb; set_trace()
    phase = phase.to(device)
    phase_num = len(phase)
    comb = torch.cos(phase).sum()**2+torch.sin(phase).sum()
    R = torch.sqrt(comb)
    Rbar = R/phase_num
    return Rbar
    

def exinp_integrate_torch(phase, mask, transform, device):
    phase = phase.to(device)
    mask = mask.to(device)
    groups_size = torch.sum(mask, dim=2)
    groups_size_mat = torch.matmul(groups_size.unsqueeze(2),
                                  groups_size.unsqueeze(1))

    masked_sin = (torch.sin(phase.unsqueeze(1)) * mask)
    masked_cos = (torch.cos(phase.unsqueeze(1)) * mask)

    product = (torch.matmul(masked_sin.unsqueeze(2).unsqueeze(4),
               masked_sin.unsqueeze(1).unsqueeze(3)) +
               torch.matmul(masked_cos.unsqueeze(2).unsqueeze(4),
               masked_cos.unsqueeze(1).unsqueeze(3)))

    if transform == 'exp':
        diag_mat = (1 - torch.eye(groups_size_mat.shape[1], groups_size_mat.shape[1])).unsqueeze(0).to(device)
        product_ = product.sum(4).sum(3) / (groups_size_mat + 1e-8)
        product_1 = torch.exp(product_) * \
                    torch.where(groups_size_mat == 0,
                                torch.zeros_like(groups_size_mat),
                                torch.ones_like(groups_size_mat)) * diag_mat
        dl = (product_1.sum(2).sum(1) / torch.abs(torch.sign(product_1)).sum(2).sum(1))
        product_2 = torch.exp(1-product_) * \
                    torch.where(groups_size_mat == 0,
                                torch.zeros_like(groups_size_mat),
                                torch.ones_like(groups_size_mat)) * (1 - diag_mat)
        sl = (product_2.sum(2).sum(1) / torch.abs(torch.sign(product_2)).sum(2).sum(1))
    elif transform == 'linear':
        diag_mat = (1 - torch.eye(groups_size_mat.shape[1], groups_size_mat.shape[1])).unsqueeze(0).to(device)
        product_ = product.sum(4).sum(3) / (groups_size_mat + 1e-8)
        product_1 = (product_ + 1) * \
                    torch.where(groups_size_mat == 0,
                                torch.zeros_like(groups_size_mat),
                                torch.ones_like(groups_size_mat)) * diag_mat
        dl = (product_1.sum(2).sum(1) / torch.abs(torch.sign(product_1)).sum(2).sum(1))
        product_2 = (1 - product_ + 1) * \
                    torch.where(groups_size_mat == 0,
                                torch.zeros_like(groups_size_mat),
                                torch.ones_like(groups_size_mat)) * (1 - diag_mat)
        sl = (product_2.sum(2).sum(1) / torch.abs(torch.sign(product_2)).sum(2).sum(1))
    else:
        raise ValueError('transformation not included')
    return dl + sl


def exinp_integrate_torch2(phase, mask, transform, device):
    # With efficient coding
    # integrate the calculation of both synchrony and desynchrony losses that I am currently using
    # This will directly give you the summation of two losses
    # This does not avg over batch
    # During training, the minimum usually lies between 0.6-0.7
    phase = phase.to(device)
    groups_size = torch.sum(mask, dim=2)
    groups_size_mat = torch.matmul(groups_size.unsqueeze(2),
                                   groups_size.unsqueeze(1))
    mask = torch.transpose(mask.to(device), 1, 2)

    diag = (torch.eye(groups_size_mat.shape[1]) + 1 - 3 * torch.eye(groups_size_mat.shape[1])).to(device)
    product_s = torch.bmm(torch.sin(phase).unsqueeze(1).float(), mask)
    product_c = torch.bmm(torch.cos(phase).unsqueeze(1).float(), mask)
    prod = (torch.bmm(torch.transpose(product_s, 1, 2), product_s) +
            torch.bmm(torch.transpose(product_c, 1, 2), product_c)) / (groups_size_mat + 1e-8)
    if transform == 'exp':
        prod = torch.exp(prod * diag + torch.eye(groups_size_mat.shape[1]).to(device)) * \
               torch.where(groups_size_mat == 0, torch.zeros_like(groups_size_mat), torch.ones_like(groups_size_mat))
    elif transform == 'linear':
        prod = (prod * diag + torch.eye(groups_size_mat.shape[1]).to(device) + 1) * \
               torch.where(groups_size_mat == 0, torch.zeros_like(groups_size_mat), torch.ones_like(groups_size_mat))
    else:
        raise ValueError('transformation not included')
    return prod.sum(2).sum(1) / torch.abs(torch.sign(prod)).sum(2).sum(1)


def coupling_regularizer(coupling, mask, device):
    coupling = coupling.to(device)
    mask = torch.transpose(mask.to(device), 1, 2)
    prod = (torch.bmm(coupling, mask) * (1 - mask)).sum(2, keepdim=True)

    return \
        (mask.sum(1, keepdim=True) * torch.bmm(torch.transpose(prod ** 2, 1, 2), mask) -
         torch.bmm(torch.transpose(prod, 1, 2), mask) ** 2).squeeze().sum(1) * 0.5 / ((mask.sum(1) - 1) * mask.sum(1)).sum(1)


def matt_loss1(phase, mask,transform, device='cpu'):
    num_group_el = mask.sum(-1)
    # Number of groups
    num_groups = (num_group_el > 0).sum(1).float()
    # Normalization factor for each group, even if empty
    per_group_norm = torch.where(num_group_el == 0, torch.ones_like(num_group_el), num_group_el)
    # Image size
    img_size = phase.shape[1]
    # Mask phase: dim 1 is group index
    masked_phase = phase.unsqueeze(1) * mask
    # Group real part. Note that the cosine sum is adjusted to account for erroneous cosine(0) from the mask
    group_real = (torch.cos(masked_phase).sum(-1) - (img_size - num_group_el))
    # Group imag part
    group_imag = torch.sin(masked_phase).sum(-1)

    # Group order. Modulus gradient is undefined at 0 so real gradient is calculated on squared modulus, but printed loss is true modulus. 
    group_order = (group_real**2 + group_imag**2)
    cosmetic_scalar = (torch.sqrt(group_order.detach()) - group_order).detach()
    group_order = (group_order + cosmetic_scalar) / per_group_norm

    # Synch loss is 1 - mean_field averaged across batch
    synch_loss = 1 - (group_order.sum(1) / num_groups)

    # Mask for where group angle is undefined
    und_mask = (group_real==0)*(group_imag==0)
    # Temporarily replace imag part at masked indices
    tmp_group_imag = torch.where(und_mask, torch.ones_like(group_imag), group_imag)

    # Calculate phase of group 
    group_phase = torch.atan2(tmp_group_imag,group_real)

    # Mask all phase pairs where phase is defined
    desynch_mask = ((1 - und_mask.float()).unsqueeze(1) * (1 - und_mask.float()).unsqueeze(2))

    # Phase diffs
    group_phase_diffs = desynch_mask * torch.abs(torch.cos(group_phase.unsqueeze(1) - group_phase.unsqueeze(2)))**2

    # Desynch loss is frame potential between groups normalized to be between 0 and 1
    desynch_loss = (group_phase_diffs.sum((1,2)) - .5*num_groups**2) * (2./num_groups**2)

    # Total loss is convex combination of the two losses
    tot_loss = .5*synch_loss + .5*desynch_loss
    if tot_loss.mean() != tot_loss.mean():
        ipdb.set_trace()

    return tot_loss

def matt_loss2(phase, mask,transform, device='cpu'):
    num_group_el = mask.sum(-1)
    # Number of groups
    num_groups = (num_group_el > 0).sum(1).float()
    # Normalization factor for each group, even if empty
    per_group_norm = torch.where(num_group_el == 0, torch.ones_like(num_group_el), num_group_el)
    # Image size
    img_size = phase.shape[1]
    # Mask phase: dim 1 is group index
    masked_phase = phase.unsqueeze(1) * mask
    # Group real part. Note that the cosine sum is adjusted to account for erroneous cosine(0) from the mask
    group_real = (torch.cos(masked_phase).sum(-1) - (img_size - num_group_el))
    # Group imag part
    group_imag = torch.sin(masked_phase).sum(-1)

    # Group order. Modulus gradient is undefined at 0 so real gradient is calculated on squared modulus, but printed loss is true modulus. 
    group_order = (group_real**2 + group_imag**2)
    cosmetic_scalar = (torch.sqrt(group_order.detach()) - group_order).detach()
    group_order = (group_order + cosmetic_scalar) / per_group_norm

    # Synch loss is 1 - mean_field averaged across batch
    synch_loss = 1 - (group_order.sum(1) / num_groups)

    phase_diffs = (torch.abs(torch.cos(phase.unsqueeze(2) - phase.unsqueeze(1))))**2

    mask_unions = 1*(mask.unsqueeze(2) + mask.unsqueeze(1) > 0)
    mask_unions_intersections = torch.einsum('bijk,bijl->bijkl',mask_unions,mask_unions)
    masked_phase_diffs = phase_diffs.unsqueeze(1).unsqueeze(1) * mask_unions_intersections
    Z = mask_unions_intersections.sum((3,4))
    Z = torch.where(Z>0,Z,torch.ones_like(Z))

    group_disorder = masked_phase_diffs.sum((3,4)) / Z
    desynch_loss = (group_disorder.sum((1,2)) - .5*num_groups**2) * (2./num_groups**2)
    
    # Total loss is convex combination of the two losses
    tot_loss = .5*synch_loss + .5*desynch_loss
    if tot_loss.mean() != tot_loss.mean():
        ipdb.set_trace()

    return tot_loss

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


def inp_btw_groups_torch(phase, mask, transformation='linear', device='cpu'):
    # Just without exponential
    phase = phase.to(device)
    mask = mask.to(device)
    groups_size = torch.sum(mask, dim=2)
    groups_size_mat = torch.matmul(groups_size.unsqueeze(2),
                                   groups_size.unsqueeze(1))
    # punish = torch.ones_like(groups_size)
    # punish[:, 1, :] = 0.1
    # groups_size_sum = groups_size_mat.sum(2).sum(1) - torch.sum(groups_size ** 2, dim=1)

    masked_sin = (torch.sin(phase.unsqueeze(1)) * mask)
    masked_cos = (torch.cos(phase.unsqueeze(1)) * mask)

    product = (torch.matmul(masked_sin.unsqueeze(2).unsqueeze(4),
                            masked_sin.unsqueeze(1).unsqueeze(3)) +
               torch.matmul(masked_cos.unsqueeze(2).unsqueeze(4),
                            masked_cos.unsqueeze(1).unsqueeze(3)))
    diag_mat = (1 - torch.eye(groups_size_mat.shape[1], groups_size_mat.shape[1])).unsqueeze(0)
    product = product / (groups_size_mat.unsqueeze(3).unsqueeze(4) + 1e-6)
    if transformation == 'linear':
        product = product.sum(4).sum(3) * diag_mat.to(device)
    elif transformation == 'exponential':
        product = torch.exp(product.sum(4).sum(3)) * diag_mat.to(device)
    else:
        raise ValueError('transformation not included')
    product = product.sum(2).sum(1) / torch.abs(torch.sign(product)).sum(2).sum(1)

    return product.mean(), product.mean(), product.mean()


def exinp_btw_groups_torch(phase, mask, device):
    phase = phase.to(device)
    mask = mask.to(device)
    groups_size = torch.sum(mask, dim=2)
    groups_size_mat = torch.matmul(groups_size.unsqueeze(2),
                                   groups_size.unsqueeze(1))
    # groups_size_sum = groups_size_mat.sum(2).sum(1) - torch.sum(groups_size ** 2, dim=1)

    masked_sin = (torch.sin(phase.unsqueeze(1)) * mask)
    masked_cos = (torch.cos(phase.unsqueeze(1)) * mask)

    product = (torch.matmul(masked_sin.unsqueeze(2).unsqueeze(4),
                           masked_sin.unsqueeze(1).unsqueeze(3)) +
              torch.matmul(masked_cos.unsqueeze(2).unsqueeze(4),
                           masked_cos.unsqueeze(1).unsqueeze(3)))
    diag_mat = (1 - torch.eye(groups_size_mat.shape[1], groups_size_mat.shape[1])).unsqueeze(0).to(device)
    product = product / (groups_size_mat.unsqueeze(3).unsqueeze(4) + 1e-6)
    product = torch.exp(product.sum(4).sum(3)) * diag_mat
    product = product.sum(2).sum(1) / torch.abs(torch.sign(product)).sum(2).sum(1)
    return product


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


# SBD code coming from: 
# https://github.com/Wizaron/instance-segmentation-pytorch/blob/master/code/evaluate.py
def calc_dic(n_objects_gt, n_objects_pred):
    return np.abs(n_objects_gt - n_objects_pred)


def calc_dice(gt_seg, pred_seg):

    nom = 2 * np.sum(gt_seg * pred_seg)
    denom = np.sum(gt_seg) + np.sum(pred_seg)

    dice = float(nom) / float(denom)
    return dice


def calc_bd(ins_seg_gt, ins_seg_pred):

    gt_object_idxes = list(set(np.unique(ins_seg_gt)).difference([0]))
    pred_object_idxes = list(set(np.unique(ins_seg_pred)).difference([0]))

    best_dices = []
    for gt_idx in gt_object_idxes:
        _gt_seg = (ins_seg_gt == gt_idx).astype('bool')
        dices = []
        for pred_idx in pred_object_idxes:
            _pred_seg = (ins_seg_pred == pred_idx).astype('bool')

            dice = calc_dice(_gt_seg, _pred_seg)
            dices.append(dice)
        best_dice = np.max(dices)
        best_dices.append(best_dice)

    best_dice = np.mean(best_dices)

    return best_dice


def calc_sbd(ins_seg_gt, ins_seg_pred):

    _dice1 = calc_bd(ins_seg_gt, ins_seg_pred)
    _dice2 = calc_bd(ins_seg_pred, ins_seg_gt)
    return min(_dice1, _dice2)

def calc_pq(seg_gt, seg_pred, threshold=.5):

    seg_gt = np.array([1*(seg_gt == i) for i in np.unique(seg_gt)])
    seg_pred = np.array([1*(seg_pred == i) for i in np.unique(seg_pred)])

    all_intersections = np.einsum('gi,pi->gpi',seg_gt,seg_pred)
    all_unions = np.clip(seg_gt[:,np.newaxis,...] + seg_pred[np.newaxis,...],0,1)

    all_IoU = all_intersections.sum(2) / all_unions.sum((2))
    argmax_IoU= np.argmax(all_IoU,1)
    max_IoU = np.take_along_axis(all_IoU, argmax_IoU[:,np.newaxis],axis=1)
    
    TP = ((max_IoU > threshold)*1)[:,np.newaxis].sum()
    FN = seg_gt.shape[0] - TP
    FP = seg_pred.shape[0] - TP

    PQ = max_IoU.sum() / (TP + .5*FP + .5*FN)
    return PQ


