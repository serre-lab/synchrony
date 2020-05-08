import numpy as np
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

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
                 batch_size=32,
                 num_global=0,
                 time_steps=10,
                 anneal=0.0,
                 phase_initialization='random',
                 walk_step=.1,
                 intrinsic_frequencies='zero',
                 connectivity0=8,
                 update_rate=0.1,
                 update_fn_number=3,
                 max_time=100,
                 device='cpu', method='odeint'):

        self.N = oscillator_number
        self.gN = num_global
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.eps = update_rate
        self.anneal = anneal
        self.time_steps = time_steps
        self.integration_time = torch.linspace(0., max_time, 20).float()

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
            self.phase_0 = lambda b: 2 * np.pi * torch.rand((self.batch_size, self.N + self.gN)).float()
        elif initialization == 'fixed':
            self.current_phase = 2 * np.pi * torch.rand((1, self.N + self.gN)).float()
            self.phase_0 = lambda b: self.current_phase.repeat(b, 1)
        elif initialization == 'gaussian':
            self.phase_0 = lambda b: torch.normal(0., .1, (b, self.N + self.gN)).float()
        elif initialization == 'categorical':
            self.phase_0 = lambda b: torch.randint(0, 4, (b, self.N + self.gN)).float() * 2 * np.pi / 4.
        elif initialization == 'random_walk':
            self.current_phase = 2 * np.pi * torch.rand((1, self.N + self.gN)).float()
            self.gamma = kwargs['walk_step']

            def walk(b):
                if self.current_phase.shape[0] != b: self.current_phase = self.current_phase.repeat(b, 1)
                self.current_phase += self.gamma * 2 * np.pi * torch.rand((b, self.N + self.gN)).float()
                return self.current_phase

            self.phase_0 = walk
        return True

    def frequency_init(self, initialization='zero'):
        if initialization == 'gaussian':
            self.omega = lambda x, b: 2 * np.pi * torch.normal((b, self.N + self.gN))
        elif initialization == 'zero':
            self.omega = lambda x, b: torch.zeros((b, self.N + self.gN))
        elif initialization == 'learned':
            self.freq_net = nets.autoencoder(int(np.sqrt(self.N)), num_global_control=self.gN).to(self.device)
            self.omega = lambda x, b: self.freq_net.forward(x.to(self.freq_net.device())).reshape(b, -1)
            # self.omega = lambda x,b : self.freq_net.forward(x).reshape(b, -1)
            return True

    def _update3(self, phase, coupling, omega):
        # efficient, much less memory usage
        n = torch.abs(torch.sign(coupling)).sum(2)

        self.delta = self.eps * \
                     (torch.bmm(coupling, torch.sin(phase).unsqueeze(2).float()).squeeze(2) * torch.cos(phase) -
                      torch.bmm(coupling, torch.cos(phase).unsqueeze(2).float()).squeeze(2) * torch.sin(phase)) / n
        phase = phase + self.delta + omega
        return phase

    def evolution(self, coupling, omega=None, batch=None, hierarchical=False):
        b = coupling[0].shape[0] if self.gN > 0 else coupling.shape[0]
        dv = coupling[0].device if self.gN > 0 else coupling.device
        phase = self.phase_0(b)
        self.eps = self.update_rate
        if omega is None:
            omega = self.omega(batch, b)
        phase_list = [phase.to(dv)]
        if hierarchical:
            batch_lconnectivity = self.connectivity0[0].unsqueeze(0).repeat(coupling[0].shape[0], 1, 1).to(dv)
            batch_gconnectivity = self.connectivity0[1].unsqueeze(0).repeat(coupling[1].shape[0], 1, 1).to(dv)
            local_coupling = torch.zeros(coupling[0].shape[0],
                                         coupling[0].shape[1],
                                         phase.shape[1]).to(dv).scatter_(dim=2,
                                                                         index=batch_lconnectivity, src=coupling[0])

            global_coupling = torch.zeros(coupling[1].shape[0],
                                          coupling[1].shape[1],
                                          phase.shape[1]).to(dv).scatter_(dim=2,
                                                                          index=batch_gconnectivity, src=coupling[1])

            coupling = torch.zeros(phase.shape[0], phase.shape[1], phase.shape[1]).to(dv)
            coupling[:, :local_coupling.shape[1], :] += local_coupling
            coupling[:, local_coupling.shape[1]:, :] += global_coupling

        else:
            batch_lconnectivity = self.connectivity0.unsqueeze(0).repeat(coupling.shape[0], 1, 1).to(dv)
            coupling = torch.zeros(phase.shape[0], phase.shape[1], phase.shape[1]).to(dv).scatter_(dim=2,
                                                                                                   index=batch_lconnectivity,
                                                                                                   src=coupling)

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

    # Defines the dynamic and solve the sytem for the specified self.integration_time.
    def ODE_evolution(self, coupling, omega=None, rtol=1e-4, atol=1e-4, method='euler', options=None):
        b = coupling[0].shape[0] if self.gN > 0 else coupling.shape[0]
        dv = coupling[0].device if self.gN > 0 else coupling.device
        phase_init = self.phase_0(b).to(dv)
        batch_lconnectivity = self.connectivity0.unsqueeze(0).repeat(coupling.shape[0], 1, 1).to(dv)
        couplings = torch.zeros(phase_init.shape[0], phase_init.shape[1], phase_init.shape[1]).to(dv).scatter_(dim=2,
                                                                                                               index=batch_lconnectivity,
                                                                                                               src=coupling)
        # ODE dynamic module is updated with coupling parameters and integrated through solver
        self.ODEDynamic.update(couplings, omega)
        phase_list = odeint(self.ODEDynamic, phase_init.flatten(), self.integration_time.type_as(phase_init), rtol,
                            atol, method, options)

        return list(phase_list.reshape(self.integration_time.shape[0], b, -1)), couplings

    # methods to track number of calls to the solver
    @property
    def nfe(self):
        return self.ODEDynamic.nfe

    @nfe.setter
    def nfe(self, value):
        self.ODEDynamic.nfe = value


class ODEDynamic(nn.Module):
    """
    torch.nn.Module that defines the infinitesimal evolution of the ODE : df/dt = module(t,\theta)
    - Handles batchs of images by flattening the bacth dim and treat everything as a single ODE
    - Requires the update of the couplings parameters at every call to get the gradient d(couplings)/dL
    """

    def __init__(self, args):
        super(ODEDynamic, self).__init__()
        # self.couplings = torch.nn.Parameter(torch.Tensor([args.batch_size,args.img_side,args.img_side]),requires_grad=True)
        self.nfe = 0

    def update(self, couplings, omega):
        self.couplings = torch.nn.Parameter(couplings, requires_grad=True)
        if omega is not None:
            self.omega = torch.nn.Parameter(omega, requires_grad=True)
        else:
            self.omega = None

    def forward(self, t, phase):
        phase = phase.reshape(self.couplings.shape[0], -1).float()
        n = torch.abs(torch.sign(self.couplings)).sum(2)
        delta_phase = (torch.bmm(self.couplings, torch.sin(phase).unsqueeze(2).float()).squeeze(2) * torch.cos(phase) -
                       torch.bmm(self.couplings, torch.cos(phase).unsqueeze(2).float()).squeeze(2) * torch.sin(
                    phase)) / n

        if self.omega is not None:
            delta_phase = delta_phase.flatten() + self.omega.flatten()
        else:
            delta_phase = delta_phase.flatten()
        self.nfe += 1
        return delta_phase

class ODEDynamic_general(nn.Module):
    """
    torch.nn.Module that defines the infinitesimal evolution of the ODE : df/dt = module(t,\theta)
    - Handles batchs of images by flattening the bacth dim and treat everything as a single ODE
    - Requires the update of the couplings parameters at every call to get the gradient d(couplings)/dL
    """

    def __init__(self, args):
        super(ODEDynamic_general, self).__init__()
        # self.couplings = torch.nn.Parameter(torch.Tensor([args.batch_size,args.img_side,args.img_side]),requires_grad=True)
        self.nfe = 0

    def update(self, vector_field, omega):
        self.couplings = torch.nn.Parameter(vector_field, requires_grad=True)
        if omega is not None:
            self.omega = torch.nn.Parameter(omega, requires_grad=True)
        else:
            self.omega = None

    def forward(self, t, phase):
        phase = phase.reshape(self.couplings.shape[0], -1).float()
        n = torch.abs(torch.sign(self.couplings)).sum(2)
        delta_phase = torch.bmm(self.couplings,phase.unsqueeze(2)).squeeze(2) / n
        if self.omega is not None:
            delta_phase = delta_phase.flatten() + self.omega.flatten()
        else:
            delta_phase = delta_phase.flatten()
        self.nfe += 1
        return delta_phase

#couplings are now defining a vector field


class Osci_AE(nn.Module):
    def __init__(self, args, connectivity,
                 update_rate=1, anneal=0, time_steps=20,
                 phase_initialization='random', walk_step=20,
                 intrinsic_frequencies='zero', device='cpu'):
        super(Osci_AE, self).__init__()
        """
        nn.module object for passing to odeint module for various image size, feature maps are all in the same shape as input
        """
        self.args = args
        self.n_osci = args.n_osci
        self.num_global = args.num_global_control
        self.connectivity = connectivity
        self.rank = 0
        self.osci = Kuramoto(self.n_osci, update_rate=update_rate, batch_size=args.batch_size,
                       anneal=anneal, time_steps=time_steps,
                       connectivity0=connectivity, num_global=self.num_global,
                       phase_initialization=phase_initialization,
                       walk_step=walk_step, device=device, max_time=args.max_time,
                       intrinsic_frequencies=intrinsic_frequencies)
        self.evolution = self.osci.evolution
        if args.dynamic_type == 'linear':
            self.osci.ODEDynamic = ODEDynamic_linear(args)
        elif args.dynamic_type == 'kura':
            self.osci.ODEDynamic = ODEDynamic(args)
        elif args.dynamic_type == 'general':
            self.osci.ODEDynamic = ODEDynamic_general(args)
        else:
            raise ValueError("Dynamic not implemented")

        self.num_cn = args.num_cn
        self.img_side = args.img_side
        if self.num_global > 0:
            self.out_channels += 1
        self.split = args.split
        self.depth = args.depth
        self.args = args
        self.sigma = nn.Sigmoid()
        self.dropout = nn.Dropout(args.dropout_p)
        self.channel_num = 3
        self.kernel_num = 1024
        self.feature_size = self.img_side // 8
        self.feature_volume = self.kernel_num * (self.feature_size ** 2)

        self.encoder = nn.Sequential(
            self._conv(self.channel_num, self.kernel_num // 4),
            self._conv(self.kernel_num // 4, self.kernel_num // 2),
            self._conv(self.kernel_num // 2, self.kernel_num))

        self.fc1 = nn.Linear(self.feature_volume, args.n_osci * args.num_cn)
        self.fc2 = nn.Linear(args.n_osci, self.feature_volume)
        self.batch_norm = nn.BatchNorm2d(self.kernel_num)

        self.decoder = nn.Sequential(
            self._deconv(self.kernel_num, self.kernel_num // 2),
            self._deconv(self.kernel_num // 2, self.kernel_num // 4),
            self._deconv(self.kernel_num // 4, self.channel_num))
        # ======
        # Layers
        # ======

    def _conv(self, channel_size, kernel_num):
        return nn.Sequential(
            nn.Conv2d(channel_size, kernel_num,kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU())

    def _deconv(self, channel_num, kernel_num):
        return nn.Sequential(
            nn.ConvTranspose2d(channel_num, kernel_num,kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU())

    def encode(self, x):
        h = self.encoder(x).view(-1, self.feature_volume)
        couplings = self.fc1(h).reshape(-1, self.n_osci, self.num_cn)
        couplings = couplings / couplings.norm(p=2, dim=2).unsqueeze(2)
        return couplings

    def decode(self, phi):
        h = self.fc2(phi).view(-1,self.kernel_num,self.feature_size,self.feature_size)
        h = self.batch_norm(h)
        x_rec = self.decoder(F.relu(h))
        return slef.sigma(x_rec)

    def forward(self, x):
        couplings = self.encode(x)
        phase_list , couplings = self.ODE_evolution(couplings, omega=None, method=self.args.solver)
        reco=[]
        for phase in phase_list[-1*args.record_steps:]:
            reco.append(self.decode(phase))
        return reco, phase_list, couplings

if __name__ == '__main__':
    print('numpy version kuramoto dynamics simulation tool\n')
    kura_tool = kura_np(5)
    initial_phase = copy.deepcopy(kura_tool.phase)
    kura_tool.set_coupling(np.ones((5, 5)))
    final_phase = kura_tool.evolution(steps=50)
    print('initial phase\n', initial_phase, '\n')
    print('coupling\n', kura_tool.coupling, '\n')
    print('final phase\n', final_phase, '\n')
