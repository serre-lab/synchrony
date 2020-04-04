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
                 batch_size = 32,
                 num_global = 0,
                 time_steps = 10,
                 anneal=0.0,
                 phase_initialization = 'random',
                 walk_step=.1,
                 intrinsic_frequencies = 'zero',
                 connectivity0=8,
                 update_rate=0.1,
                 update_fn_number = 3,
                 device='cpu'):
  
        self.N = oscillator_number
        self.gN = num_global
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.eps = update_rate
        self.anneal = anneal
        self.time_steps = time_steps

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
            self.phase_0 = lambda b : 2*np.pi * torch.rand((self.batch_size, self.N + self.gN)).float()
        elif initialization == 'fixed':
            self.current_phase = 2*np.pi * torch.rand((1,self.N + self.gN)).float()
            self.phase_0 = lambda b : self.current_phase.repeat(b, 1)
        elif initialization == 'gaussian':
            self.phase_0 = lambda b : torch.normal(0., .1, (b, self.N + self.gN)).float()
        elif initialization == 'categorical':
            self.phase_0 = lambda b : torch.randint(0, 4, (b, self.N + self.gN)).float() * 2*np.pi / 4.
        elif initialization == 'random_walk':
            self.current_phase = 2*np.pi * torch.rand((1,self.N + self.gN)).float()
            self.gamma = kwargs['walk_step']
            def walk(b): 
                if self.current_phase.shape[0] != b: self.current_phase = self.current_phase.repeat(b,1)
                self.current_phase+= self.gamma*2*np.pi * torch.rand((b,self.N + self.gN)).float()
                return self.current_phase
            self.phase_0 = walk
        return True

    def frequency_init(self, initialization='zero'):
        if initialization == 'gaussian':
            self.omega = lambda x,b : 2*np.pi * torch.normal((b, self.N + self.gN))
        elif initialization == 'zero':
            self.omega = lambda x,b : torch.zeros((b,self.N + self.gN))
        elif initialization == 'learned':
            self.freq_net = nets.autoencoder(int(np.sqrt(self.N)), num_global_control=self.gN).to(self.device)
            self.omega = lambda x,b : self.freq_net.forward(x.to(self.freq_net.device())).reshape(b, -1)
            #self.omega = lambda x,b : self.freq_net.forward(x).reshape(b, -1)
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
        self.delta = self.eps * (torch.sum(coupling * torch.sin(diffs).gather(2, self.connectivity), dim=2) / coupling.shape[2])
        self.phase = self.phase + self.delta + omega
        return self.phase
    
    def _update3(self, phase, coupling, omega):
        # efficient, much less memory usage
        n = torch.abs(torch.sign(coupling)).sum(2)
        self.delta = self.eps * ( omega + \
                     (torch.bmm(coupling, torch.sin(phase).unsqueeze(2).float()).squeeze(2) * torch.cos(phase) -
                     torch.bmm(coupling, torch.cos(phase).unsqueeze(2).float()).squeeze(2) * torch.sin(phase)) / n)
        phase = phase + self.delta
        return phase

    def evolution(self, coupling, omega=None, batch=None, hierarchical=False):
        ipdb.set_trace()
        b = coupling[0].shape[0] if self.gN > 0 else coupling.shape[0] 
        dv = coupling[0].device if self.gN > 0 else coupling.device
        phase = self.phase_0(b)
        self.eps = self.update_rate
        if omega is None:
            omega = self.omega(batch,b)
        phase_list = [phase.to(dv)]
        if hierarchical:
            batch_lconnectivity = self.connectivity0[0].unsqueeze(0).repeat(coupling[0].shape[0],1,1).to(dv)
            batch_gconnectivity = self.connectivity0[1].unsqueeze(0).repeat(coupling[1].shape[0],1,1).to(dv)
            local_coupling=torch.zeros(coupling[0].shape[0],
                                       coupling[0].shape[1],
                                       phase.shape[1]).to(dv).scatter_(dim=2,
                                       index=batch_lconnectivity, src=coupling[0])

            global_coupling=torch.zeros(coupling[1].shape[0],
                                       coupling[1].shape[1],
                                       phase.shape[1]).to(dv).scatter_(dim=2,
                                       index=batch_gconnectivity,src=coupling[1])

            coupling = torch.zeros(phase.shape[0], phase.shape[1], phase.shape[1]).to(dv)
            coupling[:, :local_coupling.shape[1], :] += local_coupling
            coupling[:, local_coupling.shape[1]:, :] += global_coupling

        else:
            batch_lconnectivity = self.connectivity0.unsqueeze(0).repeat(coupling.shape[0],1,1).to(dv)
            coupling = torch.zeros(phase.shape[0], phase.shape[1], phase.shape[1]).to(dv).scatter_(dim=2,index=batch_lconnectivity, src=coupling)

        for i in range(self.time_steps):
            new = self.update(phase_list[-1].to(dv), coupling, omega.to(dv))
            self.eps_anneal(i)
            phase_list.append(new)
        try:
            return phase_list, coupling, omega.to(dv)
        except RuntimeError:
            print('No updating')
    
    def eps_anneal(self, i):
        self.eps = self.eps - self.anneal * float(i) * self.eps / self.time_steps
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
