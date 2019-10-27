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
phases initialization: random 0~2*pi
intrinsic frequencies: 0s
coupling: 0s
no printing & recording intermediate phases
"""


class kura_np(object):
    """
    numpy
    """
    def __init__(self,
                 oscillator_num,
                 updating_rate=0.1,
                 mean_field=1,
                 batch_size=1):

        self.eps_init = updating_rate
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

    def frequency_init(self, intrinsic_frequency=None):
        if intrinsic_frequency is not None:
            self.in_frq = intrinsic_frequency
        else:
            self.in_frq = np.zeros((self.batch, self.N))

    def set_coupling(self, coupling=None):
        if coupling is not None:
            self.coupling = coupling
        else:
            self.coupling = np.zeros((self.N, self.N))

    #def set_eps(self, updating_rate=None):
    #    if updating_rate is not None:
    #        self.ep = updating_rate
    #    else:
    #        self.ep = 0.1

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
        self.delta = self.eps * (self.in_frq + np.sum(self.coupling * np.sin(diffs), axis=2) / (self.N - 1))
        self.phase = (self.phase + self.delta) % (2 * np.pi)
        new_phase = copy.deepcopy(self.phase)
        freq = copy.deepcopy(self.delta)
        return new_phase, freq

    def evolution(self, steps=1, record=False, show=False, anneal=False, int_freq=None):
        phases_list = [copy.deepcopy(self.phase)]
        freqs_list = [copy.deepcopy(self.delta)]
        self.eps = self.eps_init
        self.frequency_init(int_freq)
        if show:
            for i in tqdm(range(steps)):
                new, freq = self._update()
                self.eps_anneal(i, steps, rate=float(anneal))
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
        #self.set_ep(updating_rate)
        self.set_mean_field(mean_field)
        self.phase_init(initial_phase)
        self.frequency_init(initial_freq)
        self.set_coupling(coupling)


    def eps_anneal(self,
                   i,
                   steps,
                   rate=0):
        self.eps = self.eps - rate * float(i) * self.eps / steps


class kura_torch(object):
    """
    torch
    """
    def __init__(self,
                 oscillator_num,
                 update_rate=0.1,
                 mean_field=1,
                 batch_size=1):
        self.eps_init = update_rate
        self.K = mean_field
        self.batch = batch_size
        self.N = oscillator_num
        self.phase = torch.rand(batch_size, oscillator_num) * 2
        self.in_frq = torch.zeros_like(self.phase)
        self.delta = torch.zeros_like(self.phase)

    def phase_init(self, initial_phase=None):
        if initial_phase is not None:
            self.phase = initial_phase
        else:
            self.phase = torch.rand(self.batch, self.N) * 2 * np.pi
        initial_phase = copy.deepcopy(self.phase)
        return initial_phase

    def frequency_init(self, initial_frequency=None):
        if initial_frequency is not None:
            self.in_frq = initial_frequency
        else:
            self.in_frq = torch.zeros((self.batch, self.N))

    #def set_eps(self, updating_rate=None):
    #    if updating_rate is not None:
    #        self.ep = updating_rate
    #    else:
    #        self.ep = 0.1

    def set_mean_field(self, mean_field=None):
        if mean_field is not None:
            self.K = mean_field
        else:
            self.K = 1

    def _update(self, coupling, test):
        diffs = self.phase.unsqueeze(1) - self.phase.unsqueeze(2)
        # diffs
        # 0, B-A, C-A
        # A-B, 0, C-B
        # A-C, B-C, 0
        self.delta = self.eps * (self.in_frq + torch.sum(coupling * torch.sin(diffs), dim=2) / (self.N - 1.) )
        self.phase = self.phase + self.delta
        if test:
            self.phase = self.phase % (2 * np.pi)
        return self.phase, self.delta

    def evolution(self, coupling, steps=1, record=False, show=False, test=False, anneal=False, in_freq=None, record_torch=False):
        phases_list = [self.phase] if record_torch else [self.phase.data.numpy()]
        freqs_list = [self.delta] if record_torch else [self.delta.data.numpy()]
        self.eps = self.eps_init
        self.frequency_init(in_freq)
        if show:
            for i in tqdm(range(steps)):
                new, freq = self._update(coupling, test)
                self.eps_anneal(i, steps, rate=float(anneal))
                if record_torch:
                    phases_list.append(new)
                    freqs_list.append(freq)
                else:
                    phases_list.append(new.data.numpy())
                    freqs_list.append(freq.data.numpy())
        else:
            for i in range(steps):
                new, freq = self._update(coupling, test)
                self.eps_anneal(i, steps, anneal)
                if record_torch:
                    phases_list.append(new)
                    freqs_list.append(freq)
                else:
                    phases_list.append(new.data.numpy())
                    freqs_list.append(freq.data.numpy())
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
        #self.set_ep(updating_rate)
        self.set_mean_field(mean_field)
        self.phase_init(initial_phase)
        self.frequency_init(initial_freq)

    def eps_anneal(self,
                   i,
                   steps,
                   rate=0):
        self.eps = self.eps - rate * float(i) * self.eps / steps
