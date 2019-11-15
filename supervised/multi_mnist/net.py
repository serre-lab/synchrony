import torch
import torch.nn as nn
import numpy as np
import kuramoto as km
import loss_func_ex as lx
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from scipy.linalg import toeplitz


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
                lx.exinp_integrate_torch(phase_list[t], mask, device).mean() * (t ** 2)
        return loss
