import numpy as np
import copy
from tqdm import tqdm
import torch
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

    def _update2(self, coupling, inds):
        # Faster but take a little more memory than _update3
        diffs = self.phase.unsqueeze(1) - self.phase.unsqueeze(2)
        self.delta = self.ep * (torch.sum(coupling * torch.sin(diffs).gather(2, inds), dim=2) / coupling.shape[2])
        self.phase = self.phase + self.delta
        return self.phase

    def _update3(self, coupling):
        # Amazingly shrinking memory usage but slower than _update2
        n = torch.abs(torch.sign(coupling)).sum(2)
        self.delta = self.ep * \
                     (torch.bmm(coupling, torch.sin(self.phase).unsqueeze(2).float()).squeeze(2) * torch.cos(self.phase) -
                      torch.bmm(coupling, torch.cos(self.phase).unsqueeze(2).float()).squeeze(2) * torch.sin(self.phase)) / n
        self.phase = self.phase + self.delta
        return self.phase

    def evolution2(self, coupling, inds, steps=1, record=False, show=False, anneal=0, initial_state=False):
        if initial_state:
            phases_list = [self.phase]
        else:
            phases_list = []
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

    def evolution3(self, coupling, inds, steps=1, show=False, anneal=0, initial_state=True, record_step=1):
        coupling = torch.zeros(coupling.shape[0],
                               coupling.shape[1],
                               coupling.shape[1]).to(self.device).scatter_(dim=2, index=inds, src=coupling)
        if initial_state:
            phases_list = [self.phase]
        else:
            phases_list = []

        if show:
            for i in tqdm(range(steps)):
                new = self._update3(coupling)
                self.eps_anneal(i, steps, anneal)
                if i > (steps - 1 - record_step):
                    phases_list.append(new)
        else:
            for i in range(steps):
                new = self._update3(coupling)
                self.eps_anneal(i, steps, anneal)
                if i > (steps - 1 - record_step):
                    phases_list.append(new)
        try:
            return phases_list, coupling
        except RuntimeError:
            print('No updating')

    def evolution4(self, local_coupling, global_coupling, inds, global_inds, steps=1,
                   show=False, anneal=0, initial_state=True, record_step=1):
        # make sure every operation is in_place
        ipdb.set_trace()
        local_coupling = torch.zeros(local_coupling.shape[0],
                                     local_coupling.shape[1],
                                     self.phase.shape[1]).to(self.device).scatter_(dim=2, index=inds, src=local_coupling)
        global_coupling = torch.zeros(global_coupling.shape[0],
                                      global_coupling.shape[1],
                                      self.phase.shape[1]).to(self.device).scatter_(dim=2, index=global_inds,
                                                                                    src=global_coupling)

        coupling = torch.zeros(self.phase.shape[0], self.phase.shape[1], self.phase.shape[1]).to(self.device)
        coupling[:, :local_coupling.shape[1], :] += local_coupling
        coupling[:, local_coupling.shape[1]:, :] += global_coupling

        if initial_state:
            phases_list = [self.phase]
        else:
            phases_list = []
        if show:
            for i in tqdm(range(steps)):
                new = self._update3(coupling)
                self.eps_anneal(i, steps, anneal)
                if i > (steps - 1 - record_step):
                    phases_list.append(new)
        else:
            for i in range(steps):
                new = self._update3(coupling)
                self.eps_anneal(i, steps, anneal)
                if i > (steps - 1 - record_step):
                    phases_list.append(new)
        try:
            return phases_list, coupling
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