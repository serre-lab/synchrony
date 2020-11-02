import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torchdiffeq import odeint, odeint_adjoint
from data_generators import GMM, Moons, Spirals, Circles
from configparser import ConfigParser
import argparse
import os
import time
import ipdb
import csv

def load_data(**kwargs):
    data_name = kwargs['data_name']
    if data_name == 'GMM':
        num_components = int(kwargs['num_components'])
        centroids = [10*np.array([np.cos(2*np.pi * i / num_components), np.sin(2*np.pi * i / num_components)]) for i in range(num_components)]
        cov_matrices = .2*np.array([np.eye(2) for _ in range(num_components)])
        data_generator = GMM(n_components=num_components,
                             centroids=centroids,
                             cov_matrices=cov_matrices)
    elif data_name == 'Moons':
        data_generator = Moons(noise=float(kwargs['noise']), random_state=None)
    elif data_name == 'Spirals':
        data_generator = Spirals(noise = float(kwargs['noise']))
    elif data_name == 'Circles':
        data_generator = Circles(noise=float(kwargs['noise']),random_state=None)

    return data_generator

def laplacian(connectivity,sym_norm=True, batch=True):
    A = connectivity
    D = torch.diag_embed(torch.abs(A).sum(1))
    D_mask = torch.diag_embed(torch.ones_like(A.sum(1)))
    L = D-A
    Phi = torch.where(D_mask.bool(), D**(-.5),torch.zeros_like(D))
    #D2 = torch.where(D_mask.bool(), D**(.5),torch.zeros_like(D))
    Phi = torch.where(torch.isinf(Phi), torch.zeros_like(Phi),Phi)
    if sym_norm:
        if batch:
            return torch.bmm(Phi,torch.bmm(L,Phi))
        else:
            return torch.matmul(Phi, torch.matmul(L,Phi))   
    else:
        return L

def circular_moments_batch(phases, masks, lasts=50, split = False, burn_in_prop=.5):
    num_groups = masks.shape[0]
    group_size = masks.sum(1)
    group_size = torch.where(group_size == 0, torch.ones_like(group_size), group_size)
    T = phases.shape[0]
    
    burn_in_steps = int(burn_in_prop * T)
    masked_phases = phases.unsqueeze(1) * masks
    xx = torch.where(masks.bool(), torch.cos(masked_phases), torch.zeros_like(masked_phases))
    yy = torch.where(masks.bool(), torch.sin(masked_phases), torch.zeros_like(masked_phases))
    go = torch.sqrt((xx.sum(-1))**2 + (yy.sum(-1))**2) / group_size
    synch = 1 - go.sum(-1)/num_groups

    mean_angles = torch.atan2(yy.sum(-1),xx.sum(-1))
    desynch = 0
    for m in np.arange(1, int(np.floor(num_groups/2.))+1):
#         K_m = 1 if m < int(np.floor(num_groups/2.)) + 1 else -1
        desynch += (1.0 / (2* num_groups * m**2)) * (torch.cos(m*mean_angles).sum(-1)**2 + torch.sin(m*mean_angles).sum(-1)**2)
        
    loss = (synch + desynch)
    
    if split == True:
        return loss[burn_in_steps:,...].mean(), synch[burn_in_steps:,...].mean(), desynch[burn_in_steps:,...].mean()
    else:
        return loss[burn_in_steps:,...].mean()

class KuraODE(nn.Module):
        
    """
    torch.nn.Module that defines the infinitesimal evolution of the ODE : df/dt = module(t,\\theta),
    - Handles batchs of images by flattening the batch dim and treat everything as a single ODE,
    - Requires the update of the couplings parameters at every call to get the gradient d(couplings)/dL,
    """

    def __init__(self):
        super(KuraODE, self).__init__()
        #self.couplings = torch.nn.Parameter(torch.eye(1),requires_grad=True) #placeholder,

    def update(self, couplings):
        self.couplings = torch.nn.Parameter(couplings,requires_grad=True)
        self.nfe = 0

    def forward(self, t, phase):
        self.nfe+=1
        phase = phase.reshape(self.couplings.shape[0],-1).float()
        n = self.couplings.shape[0]
        delta_phase = (self.couplings.unsqueeze(0)@torch.sin(phase).float() * torch.cos(phase) - 
        self.couplings.unsqueeze(0)@torch.cos(phase).float() * torch.sin(phase)) / n 

        return delta_phase.squeeze(-1)

class KuraTrainer(object):
        def __init__(self, model, data_generator, batch_size,
                      LossFunc, num_features=32, optimizer='Adam'):
            super(KuraTrainer, self).__init__()
            self.data_generator = data_generator
            self.batch_size = batch_size
            self.LossFunc = LossFunc
            self.num_features=num_features
            self.model = model
            
        def init_phase(self,batch_size):
            #self.init = torch.zeros((batch_size,)).float().unsqueeze(0)
            self.init = torch.normal(np.pi,1,(batch_size,)).float().unsqueeze(0).to(self.model.device)
            #self.init = torch.FloatTensor(batch_size,).uniform_(0, 2*np.pi).unsqueeze(0)
            return self.init
            #return torch.normal(np.pi,0.05,(batch_size,)).float().unsqueeze(0)
        
        def plot_data(self,num_points):
            a, idx = self.data_generator.sample(num_points)
            plt.scatter(a[:,0],a[:,1],c=idx)
            plt.show()
            
        def train_same_batch(self,gradient_steps=50,lr=0.1, alpha=.1, normalized=False, eps=1e-12,num_loss=0,burn_in_prop=.5, verbose=0):
            self.lh = []
            opt = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.init = self.init_phase(self.batch_size)
            batch, idx = self.data_generator.sample(self.batch_size)
            self.batch = batch.float()
            self.idx = idx.float()
            masks = torch.FloatTensor(self.batch_size,self.data_generator.n_components).zero_()
            masks.scatter_(1,idx.unsqueeze(1),1).transpose_(1,0)
            
            for step in range(gradient_steps):
                
                init_phase = self.init #self.init_phase(self.batch_size)
                
                #Training by gradient descent via adjoint sensitivity method
                opt.zero_grad()
                phase, couplings = self.model(self.init, self.batch )
                loss = LossFunc(phase, masks,split=True, burn_in_prop=burn_in_prop)
                self.lh.append(loss[num_loss].detach().numpy())
                loss[num_loss].backward()
                if verbose > 0: 
                    if step % 10 == 0:
                        print('Step: {}. Loss: {}'.format(step, self.lh[-1]))
                        print('Gradient norm: {}'.format(torch.pow(self.model.ODEDynamic.couplings.grad,2).sum().detach().cpu().numpy()))
                self.model.couplings.backward(gradient=self.model.ODEDynamic.couplings.grad)
                opt.step()
            return
        
        def train(self,gradient_steps=50,optimizer='Adam', lr=0.1, momentum=0.0,
                  alpha=.1, normalized=False, eps=1e-12,num_loss=0,
                  burn_in_prop=.5, verbose=0):
            self.lh = []
            self.sh = []
            self.eh = []

            if optimizer == 'Adam':
                opt = torch.optim.Adam(self.model.parameters(), lr=lr)
            elif optimizer=='SGD':
                opt = torch.optim.SGD(self.model.parameters(),lr=lr,momentum=momentum)

            for _ in range(gradient_steps):

                with torch.no_grad():
                    batch, idx = self.data_generator.sample(self.batch_size)
                    self.batch = batch.float().to(self.model.device)
                    self.idx = idx.float()
                    masks = torch.FloatTensor(self.batch_size,self.data_generator.n_components).zero_()
                    masks.scatter_(1,idx.unsqueeze(1),1).transpose_(1,0)
                    masks = masks.to(self.model.device)
                    init_phase = self.init_phase(self.batch_size).to(self.model.device) #self.init_phase(self.batch_size)

                #Training by gradient descent via adjoint sensitivity method
                opt.zero_grad()
                phase, couplings = self.model(self.init, self.batch)
                degree = torch.abs(couplings).sum(1)
                d_max = ((degree.unsqueeze(1)+ degree.unsqueeze(0)) / 2.).reshape(-1).max().detach().cpu().numpy()
                L = laplacian(couplings, sym_norm=True, batch=False).detach().cpu().numpy()
                self.eh.append(np.linalg.eigvals(L).min() / (d_max - 1))
                self.sh.append((1./(2*self.batch_size**2)) * torch.sqrt(((couplings - couplings.transpose(1,0))**2).sum((0,1))).mean().detach().cpu().numpy())
                loss = LossFunc(phase, masks,split=True, burn_in_prop=burn_in_prop)
                self.lh.append(loss[num_loss].detach().cpu().numpy())
                loss[num_loss].backward()
                if verbose > 0: 
                    if _ % 100 == 0: 
                        print('Step: {}. Loss: {}'.format(_, self.lh[-1]))
                        print('Gradient norm: {} \n'.format(torch.pow(self.model.ODEDynamic.couplings.grad,2).sum().detach().cpu().numpy()))
                self.model.couplings.backward(gradient=self.model.ODEDynamic.couplings.grad)
                norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
#                 print('norm',norm)
                opt.step()
            return

class Kuramoto_features(nn.Module):

    def __init__(self,
                 batch_size = 32,
                 num_hid = 32,
                 input_size = 2,
                 feature_size = 10,
                 steps=100,
                 time=1500,
                 alpha=1e-1,
                 device='cpu',
                 adjoint=True,
                 symmetric=False,
                 gain=1.):

        super(Kuramoto_features, self).__init__()

        #Integration parameters
        self.adjoint=adjoint
        self.steps = steps
        self.time = time
        self.alpha=alpha
        self.integration_time =  torch.linspace(0., self.time, self.steps).float()
        self.ODEDynamic = KuraODE()
        self.tol = 1e-5
        self.atol = 1e-5
        self.max_steps = 1000

        #Problem dimensions
        self.batch_size = batch_size
        self.indim = input_size
        self.outdim = feature_size
        self.symmetric = symmetric
        self.device = device
        self.gain = gain

        # Definition of couplings
        #self.couplings = torch.nn.Parameter(torch.rand(batch_size,batch_size) - .5,requires_grad=True)

        # Definition of coupling inference model
#         self.layers = torch.nn.Sequential(torch.nn.Linear(self.indim, num_hid),
#                           torch.nn.ReLU(),
#                           torch.nn.Linear(num_hid,num_hid),
#                           torch.nn.ReLU(),             
#                           torch.nn.Linear(num_hid,self.outdim))
        self.layers = torch.nn.Sequential(
                torch.nn.Linear(2*self.indim, num_hid),
                torch.nn.BatchNorm1d(num_hid),
                torch.nn.ReLU(),
                torch.nn.Linear(num_hid, num_hid),
                torch.nn.BatchNorm1d(num_hid),
                torch.nn.ReLU(),
                torch.nn.Linear(num_hid,1,bias=False)
        )
    
        for layer in self.layers:
            try :
                #torch.nn.init.orthogonal_(layer.weight, gain=1)
                #torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='tanh')
                torch.nn.init.xavier_normal_(layer.weight, gain=self.gain)
                #torch.nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='tanh')
            except:
                pass

    def sample_couplings(self, batches, num_samples=64):
        all_couplings = []
        for batch in batches:
            all_couplings.append(self.layers(batch.reshape(1,-1)).reshape(self.batch_size,self.batch_size).detach())
        all_couplings = torch.stack(all_couplings).reshape(-1).numpy()
        return all_couplings

    def forward(self,init_phase, x):
        batch_size = x.shape[0]
        # Bx2 --> (bXb)x4 --> (bxb)x1 --> bxb
        x = torch.cat([x[:,None].repeat(1,batch_size,1),x[None,:].repeat(batch_size,1,1)],dim=2)
        
        #Infer couplings 
        couplings = self.layers(x.view(-1, x.shape[-1]))
        couplings = couplings.view(batch_size, batch_size)
#         couplings = couplings / torch.pow(couplings,2).sum(-1).unsqueeze(1) - 0.1
        couplings = torch.nn.functional.normalize(couplings, p=2, dim=1)
        #couplings = couplings - torch.mean(couplings)
        if self.symmetric:
            couplings = .5*(couplings + couplings.transpose(1,0))
        self.couplings = couplings
        
        self.ODEDynamic.update(self.couplings)

        #Run Dynamic with adaptative solver
        if self.adjoint:
            trajectory = odeint_adjoint(self.ODEDynamic, init_phase, self.integration_time,
                                   rtol=self.tol, atol=self.atol, method='euler')
        else:
            trajectory = []
            phase = init_phase
            for t in range(self.T):
                phase = phase + self.alpha*self.ODEDynamic.forward(t, phase)
                trajectory.append(phase)
            trajectory = torch.stack(trajectory)

        return trajectory, couplings


if __name__=='__main__':
    meta_parser = argparse.ArgumentParser()
    meta_parser.add_argument('--name', type=str, default='DEFAULT')
    meta_parser.add_argument('--num_seeds', type=int, default=10)
    meta_parser.add_argument('--best_seed', type=int, default=-1)
    meta_args = meta_parser.parse_args()

    if meta_args.best_seed >= 0 and meta_args.num_seeds > 0:
        raise ValueError('If you want to search over multiple random seeds, set argument best_seed to -1!')

    config = ConfigParser()
    config.read('experiments.cfg')
    config_dict = {}
    for (key, val) in config.items(meta_args.name):
        config_dict[key] = val

    exp_name   = config_dict['exp_name']
    data_name  = config_dict['data_name']
    save_dir   = config_dict['save_dir']
    batch_size = int(config_dict['batch_size'])
    num_hid    = int(config_dict['num_hid'])
    symmetric  = config.getboolean(meta_args.name, 'symmetric')
    device     = config_dict['device']
    adjoint    = config.getboolean(meta_args.name, 'adjoint')
    optimizer  = config_dict['optimizer']
    lr         = float(config_dict['lr'])
    momentum   = float(config_dict['momentum'])
    iters      = int(config_dict['iters'])
    burn_in    = float(config_dict['burn_in'])

    save_path = os.path.join(save_dir, exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    all_losses = []
    for seed in range(meta_args.num_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        Model = Kuramoto_features(batch_size=batch_size, num_hid=num_hid,
                                  input_size=2, device=device, symmetric=symmetric,
                                  adjoint=adjoint).to(device)
        generator = load_data(**config_dict)
        LossFunc = circular_moments_batch
        Trainer = KuraTrainer(Model, generator, LossFunc=LossFunc,
                              batch_size=batch_size)
        Trainer.train(gradient_steps=iters,optimizer=optimizer, lr=lr,
                      momentum=momentum,num_loss=0, burn_in_prop=burn_in,
                      verbose=1)
        all_losses.append(Trainer.lh[-1])

    # Retrain with best seed and save
    best_seed = np.argmin(all_losses) if meta_args.num_seeds > 0 else meta_args.best_seed
    np.random.seed(best_seed)
    torch.manual_seed(best_seed)
    Model = Kuramoto_features(batch_size=batch_size, num_hid=num_hid,
                                  input_size=2, device=device, symmetric=symmetric,
                                  adjoint=adjoint).to(device)
    generator = load_data(**config_dict)
    LossFunc = circular_moments_batch
    Trainer = KuraTrainer(Model, generator, LossFunc=LossFunc, batch_size=batch_size)
    Trainer.train(gradient_steps=iters,optimizer=optimizer, lr=lr,
                      momentum=momentum, num_loss=0, burn_in_prop=burn_in,
                      verbose=1)

    torch.save(Model.state_dict(), os.path.join(save_path, 'model.pt')) 
   
    dict = config_dict 
    dict['loss'] = np.array(Trainer.lh)
    dict['seed'] = best_seed
    dict['conflict'] = Trainer.eh

    np.set_printoptions(threshold=np.inf)
    fn = os.path.join(save_path, 'result.csv')
    w = csv.writer(open(fn, "w"))
    for key, val in dict.items():
        w.writerow([key, val])
