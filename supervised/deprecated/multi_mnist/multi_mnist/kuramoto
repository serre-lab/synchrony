import numpy as np
import copy
from tqdm import tqdm
import torch

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
device = 'cpu'


class kura_np(object):
    """
    numpy
    """
    def __init__(self,
                 oscillator_num,
                 updating_rate=0.1,
                 mean_field=1,
                 batch_size=1):

        self.ep = updating_rate
        self.K = mean_field
        self.batch = batch_size
        self.N = oscillator_num
        self.phase = np.random.rand(batch_size, oscillator_num) * 2 * np.pi
        self.in_frq = np.zeros_like(self.phase)
        self.delta = np.zeros_like(self.phase)
        self.coupling = np.zeros((oscillator_num, oscillator_num))

    def phase_init(self, initial_phase=None):
        if initial_phase is not None:
            self.phase = initial_phase
        else:
            self.phase = np.random.rand(self.batch, self.N) * 2 * np.pi
        initial_phase = copy.deepcopy(self.phase)
        return initial_phase

    def frequency_init(self, initial_frequency=None):
        if initial_frequency is not None:
            self.in_frq = initial_frequency
        else:
            self.in_frq = np.zeros((self.batch, self.N))

    def set_coupling(self, coupling=None):
        if coupling is not None:
            self.coupling = coupling
        else:
            self.coupling = np.zeros((self.N, self.N))

    def set_ep(self, updating_rate=None):
        if updating_rate is not None:
            self.ep = updating_rate
        else:
            self.ep = 0.1

    def set_mean_field(self, mean_field=None):
        if mean_field is not None:
            self.K = mean_field
        else:
            self.K = 1

    def _update(self):
        diffs = np.expand_dims(self.phase, axis=1) - np.expand_dims(self.phase, axis=2)
        # diffs
        # 0, B-A, C-A
        # A-B, 0, C-B
        # A-C, B-C, 0
        self.delta = self.ep * (self.in_frq + np.sum(self.coupling * np.sin(diffs), axis=2) / (self.N - 1))
        self.phase = (self.phase + self.delta) % (2 * np.pi)
        new_phase = self.phase
        freq = self.delta
        return new_phase, freq

    def evolution(self, steps=1, record=False, show=False, anneal=0):
        phases_list = [self.phase]
        freqs_list = [self.delta]
        if show:
            for i in tqdm(range(steps)):
                new, freq = self._update()
                self.eps_anneal(i, steps, anneal)
                phases_list.append(new)
                freqs_list.append(freq)
        else:
            for i in range(steps):
                new, freq = self._update()
                self.eps_anneal(i, steps, anneal)
                phases_list.append(new)
                freqs_list.append(freq)
        try:
            if record:
                return phases_list, freqs_list
            else:
                # only return final phase
                return new
        except RuntimeError:
            print('No updating')

    def set_params(self,
                   updating_rate=0.1,
                   mean_field=1,
                   coupling=None,
                   initial_phase=None,
                   initial_freq=None):
        self.set_ep(updating_rate)
        self.set_mean_field(mean_field)
        self.phase_init(initial_phase)
        self.frequency_init(initial_freq)
        self.set_coupling(coupling)

    def eps_anneal(self,
                   i,
                   steps,
                   rate=0):
        self.ep = self.ep - rate * float(i) * self.ep / steps


class kura_torch(object):
    """
    torch
    """
    def __init__(self,
                 oscillator_num,
                 update_rate=0.1,
                 mean_field=1,
                 batch_size=1):
        self.ep = update_rate
        self.K = mean_field
        self.batch = batch_size
        self.N = oscillator_num
        self.phase = (torch.rand(batch_size, oscillator_num) * 2).to(device)
        self.in_frq = torch.zeros_like(self.phase).to(device)
        self.delta = torch.zeros_like(self.phase).to(device)

    def phase_init(self, initial_phase=None):
        if initial_phase is not None:
            self.phase = initial_phase
        else:
            self.phase = torch.rand(self.batch, self.N) * 2 * np.pi
        return True

    def frequency_init(self, initial_frequency=None):
        if initial_frequency is not None:
            self.in_frq = initial_frequency
        else:
            self.in_frq = torch.zeros((self.batch, self.N))
        return True

    def set_ep(self, updating_rate=None):
        if updating_rate is not None:
            self.ep = updating_rate
        else:
            self.ep = 0.1
        return True

    def set_mean_field(self, mean_field=None):
        if mean_field is not None:
            self.K = mean_field
        else:
            self.K = 1

    def _update(self, coupling, test):
        diffs = self.phase.unsqueeze(1) - self.phase.unsqueeze(2)
        # diffs.shape=(batch, osci_num, osci_num)
        # coupling.shape=(batch, osci_num, osci_num)
        # 0, B-A, C-A
        # A-B, 0, C-B
        # A-C, B-C, 0
        self.delta = self.ep * (self.in_frq + torch.sum(coupling * torch.sin(diffs), dim=2) / (self.N - 1)).to(device)
        self.phase = self.phase + self.delta
        if test:
            self.phase = self.phase % (2 * np.pi)
        return self.phase, self.delta

    def evolution(self, coupling, steps=1, record=False, show=False, test=False, anneal=0):
        phases_list = [self.phase]
        freqs_list = [self.delta]
        if show:
            for i in tqdm(range(steps)):
                new, freq = self._update(coupling, test)
                self.eps_anneal(i, steps, anneal)
                phases_list.append(new)
                freqs_list.append(freq)
        else:
            for i in range(steps):
                new, freq = self._update(coupling, test)
                self.eps_anneal(i, steps, anneal)
                phases_list.append(new)
                freqs_list.append(freq)
        try:
            if record:
                return phases_list, freqs_list
            else:
                # only return final phase
                return self.phase
        except RuntimeError:
            print('No updating')

    def set_params(self,
                   updating_rate=0.1,
                   mean_field=1,
                   initial_phase=None,
                   initial_freq=None):
        self.set_ep(updating_rate)
        self.set_mean_field(mean_field)
        self.phase_init(initial_phase)
        self.frequency_init(initial_freq)
        return True

    def set_batch_size(self, batch_size):
        self.batch = batch_size
        return True

    def eps_anneal(self,
                   i,
                   steps,
                   rate=0):
        self.ep = self.ep - rate * float(i) * self.ep / steps
        return True


class kura_torch2(object):
    """
    Add device choice,
    default batch size if 1
    """
    def __init__(self,
                 oscillator_number,
                 update_rate=0.1,
                 device='cpu'):
        self.N = oscillator_number
        self.ep = update_rate
        self.device = device

        self.phase = (torch.rand(1, oscillator_number) * 2 * np.pi).to(self.device)
        self.in_frq = torch.zeros_like(self.phase).to(self.device)
        self.delta = torch.zeros_like(self.phase).to(self.device)

    def phase_init(self, initial_phase):
        self.phase = initial_phase.to(self.device)
        return True

    def frequency_init(self, batch=1, intrinsic_frequency=None):
        if intrinsic_frequency is not None:
            self.in_frq = intrinsic_frequency.to(self.device)
        else:
            self.in_frq = torch.zeros(batch, self.N).to(self.device)
        return True

    def _update(self, coupling):
        diffs = self.phase.unsqueeze(1) - self.phase.unsqueeze(2)
        # diffs.shape=(batch, osci_num, osci_num)
        # coupling.shape=(batch, osci_num, osci_num)
        # 0, B-A, C-A
        # A-B, 0, C-B
        # A-C, B-C, 0
        self.delta = self.ep * (self.in_frq + torch.sum(coupling * torch.sin(diffs), dim=2) / (self.N - 1))
        self.phase = self.phase + self.delta
        return self.phase

    def _update2(self, coupling, inds):
        diffs = self.phase.unsqueeze(1) - self.phase.unsqueeze(2)
        self.delta = self.ep * (torch.sum(coupling * torch.sin(diffs).gather(2, inds), dim=2) / coupling.shape[2])
        self.phase = self.phase + self.delta
        return self.phase

    def evolution(self, coupling, steps=1, record=False, show=False, anneal=0):
        phases_list = [self.phase]
        if show:
            for i in tqdm(range(steps)):
                new = self._update(coupling)
                self.eps_anneal(i, steps, anneal)
                phases_list.append(new)
        else:
            for i in range(steps):
                new = self._update(coupling)
                self.eps_anneal(i, steps, anneal)
                phases_list.append(new)
        try:
            if record:
                return phases_list
            else:
                # only return final phase
                return self.phase
        except RuntimeError:
            print('No updating')

    def evolution2(self, coupling, inds, steps=1, record=False, show=False, anneal=0):
        phases_list = [self.phase]
        if show:
            for i in tqdm(range(steps)):
                new = self._update2(coupling, inds)
                self.eps_anneal(i, steps, anneal)
                phases_list.append(new)
        else:
            for i in range(steps):
                new = self._update2(coupling, inds)
                self.eps_anneal(i, steps, anneal)
                phases_list.append(new)
        try:
            if record:
                return phases_list
            else:
                # only return final phase
                return self.phase
        except RuntimeError:
            print('No updating')

    def eps_anneal(self, i, steps, rate=0):
        self.ep = self.ep - rate * float(i) * self.ep / steps
        return True

    def set_ep(self, updating_rate=None):
        if updating_rate is not None:
            self.ep = updating_rate
        else:
            self.ep = 0.1
        return True


if __name__ == '__main__':
    print('numpy version kuramoto dynamics simulation tool\n')
    kura_tool = kura_np(5)
    initial_phase = copy.deepcopy(kura_tool.phase)
    kura_tool.set_coupling(np.ones((5, 5)))
    final_phase = kura_tool.evolution(steps=50)
    print('initial phase\n', initial_phase, '\n')
    print('coupling\n', kura_tool.coupling, '\n')
    print('final phase\n', final_phase, '\n')
