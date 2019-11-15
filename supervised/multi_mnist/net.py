import torch
import torch.nn as nn
import numpy as np
import kuramoto as km
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from scipy.linalg import toeplitz


def exinp_integrate_torch(phase, mask, device):
    # integrate the calculation of both losses
    # do not avg over batch
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
    return dl + sl


class simple_conv(nn.Module):
    def __init__(self):
        """
        For image of 36*36
        """
        super(simple_conv16, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 108, 108 * 8)

    def forward(self, input, device,
                kura_update_rate,
                anneal,
                episodes,
                initial_phase,
                connectivity):
        osci = km.kura_torch2(input.shape[1] * input.shape[2], device=device)
        osci.set_ep(kura_update_rate)
        osci.phase_init(initial_phase)

        conv1 = torch.relu(self.conv1(input.reshape(-1, 1, 36, 36)))
        conv2 = torch.relu(self.conv2(conv1))
        conv3 = torch.sigmoid(self.conv3(conv2))
        fc = self.fc(conv3.reshape(-1, 108 * 32)).reshape(-1, 1296, 8)

        phase_list = osci.evolution2(fc, connectivity, anneal=anneal, steps=episodes, record=True)
        return phase_list, fc


class criterion(nn.Module):
    def __init__(self):
        super(criterion, self).__init__()

    def forward(self, phase_list, mask, device, valid=False):
        if valid:
            loss = torch.tensor(0.).detach().to(device)
        else:
            loss = torch.tensor(0.).to(device)
        for t in range(len(phase_list)):
            loss += \
                exinp_integrate_torch(phase_list[t], mask, device).mean() * (t ** 2)
        return loss
