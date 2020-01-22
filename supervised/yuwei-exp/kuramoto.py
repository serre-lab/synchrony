import numpy as np
import copy
from tqdm import tqdm
import torch
import nets
import ipdb

"""
DEFAULTS
updating rate: 0.1
K(mean field strength): 1 
batch_size: 1
phases: random 0~2*pi
intrinsic frequencies: 0s
coupling: 0s
no printing & recording intermediate phases
"""


class Kuramoto(object):
    """
    Add device choice,
    default batch size if 1
    """

    def __init__(self,
                 oscillator_number,
                 connectivity0,
                 batch_size=32,
                 time_steps=10,
                 record_steps=10,
                 anneal=0.0,
                 phase_initialization='random',
                 walk_step=.1,
                 intrinsic_frequencies='zero',
                 update_rate=0.1,
                 update_fn_number=3,
                 device='cpu'):

        self.N = oscillator_number
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.eps = update_rate
        self.anneal = anneal
        self.time_steps = time_steps
        self.record_steps = record_steps

        if update_fn_number == 1:
            self.update = self._update1
        elif update_fn_number == 2:
            self.update = self._update2
        elif update_fn_number == 3:
            self.update = self._update3
        self.device = device

        self.connectivity0 = connectivity0
        self.phase_init(initialization=phase_initialization, walk_step=walk_step)
        self.frequency_init(initialization=intrinsic_frequencies)

    def phase_init(self, initialization='random', **kwargs):
        if initialization == 'random':
            self.phase_0 = lambda: 2 * np.pi * torch.rand((self.batch_size, self.N)).to(self.device).float()
        elif initialization == 'fixed':
            self.current_phase = 2 * np.pi * torch.rand((1, self.N)).repeat(self.batch_size, 1).to(self.device).float()
            self.phase_0 = lambda: self.current_phase
        elif initialization == 'gaussian':
            self.phase_0 = lambda: torch.normal(0., .1, (self.batch_size, self.N)).to(self.device).float()
        elif initialization == 'categorical':
            self.phase_0 = lambda: torch.randint(0, 4, (self.batch_size, self.N)).to(
                self.device).float() * 2 * np.pi / 4.
        elif initialization == 'random_walk':
            self.current_phase = 2 * np.pi * torch.rand((self.batch_size, self.N)).to(self.device).float()
            self.gamma = kwargs['walk_step']

            def walk():
                self.current_phase += self.gamma * 2 * np.pi * torch.rand((self.batch_size, self.N)).to(
                    self.device).float()
                return self.current_phase

            self.phase_0 = walk
        return True

    def frequency_init(self, initialization='zero'):
        if initialization == 'gaussian':
            self.omega = lambda x: 2 * np.pi * torch.normal((self.batch_size, self.N)).to(self.device)
        elif initialization == 'zero':
            self.omega = lambda x: torch.zeros((self.batch_size, self.N)).to(self.device)
        elif initialization == 'learned':
            self.freq_net = nets.autoencoder().to(self.device)
            self.omega = lambda x: self.freq_net.forward(x).reshape(self.batch_size, -1)
        return True

    def _update1(self, coupling, omega):
        diffs = self.phase.unsqueeze(1) - self.phase.unsqueeze(2)
        # diffs.shape=(batch, osci_num, osci_num)
        # coupling.shape=(batch, osci_num, osci_num)
        # 0, B-A, C-A
        # A-B, 0, C-B
        # A-C, B-C, 0
        self.delta = self.eps * (self.in_frq + torch.sum(coupling * torch.sin(diffs), dim=2) / (self.N - 1))
        self.phase = self.phase + self.delta + omega
        return self.phase

    def _update2(self, coupling, omega):
        diffs = self.phase.unsqueeze(1) - self.phase.unsqueeze(2)
        self.delta = self.eps * (
                    torch.sum(coupling * torch.sin(diffs).gather(2, self.connectivity0), dim=2) / coupling.shape[2])
        self.phase = self.phase + self.delta + omega
        return self.phase

    def _update3(self, coupling, omega):
        # efficient, much less memory usage
        self.delta = self.eps * \
                     (torch.bmm(coupling, torch.sin(self.phase).unsqueeze(2).float()).squeeze(2) * torch.cos(
                         self.phase) -
                      torch.bmm(coupling, torch.cos(self.phase).unsqueeze(2).float()).squeeze(2) * torch.sin(
                                 self.phase)) / self.connectivity0.shape[2]
        self.phase = self.phase + self.delta + omega
        return self.phase

    def evolution(self, coupling, batch=None):
        # integrate the update3 function and offer a new variable
        # record_step(default:1) which will set the number of how many steps to record from the last step
        connectivity = self.connectivity0.repeat(coupling.shape[0], 1, 1).to(self.device)
        coupling = torch.zeros(coupling.shape[0],
                               coupling.shape[1],
                               coupling.shape[1]).to(self.device).scatter_(dim=2, index=connectivity, src=coupling)
        self.phase = self.phase_0()
        self.eps = self.update_rate
        omega = self.omega(batch)
        phase_list = [self.phase]
        for i in range(self.time_steps):
            new = self.update(coupling, omega)
            self.eps_anneal(i)
            if i > (self.time_steps - 1 - self.record_steps):
                phase_list.append(new)
        try:
            return phase_list, omega
        except RuntimeError:
            print('No updating')

    def eps_anneal(self, i):
        self.eps = self.eps - self.anneal * float(i) * self.eps / self.time_steps
        return True
