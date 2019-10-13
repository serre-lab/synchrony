import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')
import torch
from torch.autograd import Variable
import numpy as np
import os
from tqdm import tqdm
import ipdb

def potential(k,phase):
    '''Potential energy which is minimized during dynamics.'''
    return -1* k * torch.cos(phase[0] - phase[1])

def coherence(phase):
    '''Loss function optimized to learn coupling.'''
    return .5*(torch.sqrt(torch.cos(phase).sum()**2 + torch.sin(phase).sum()**2))

def inner_opt(k, num_steps=20, lr=1e-1):
   ''' Optimization loop for dynamics. '''
   phase = Variable(2*np.pi*torch.rand(2,), requires_grad=True)
   inner_optim = torch.optim.SGD((phase,), lr=lr)  

   energy_history = []
   for n in range(num_steps):
       inner_optim.zero_grad()
       energy = potential(k, phase)
       energy_history.append(energy.data.numpy())
       energy.backward()
       inner_optim.step()
   return phase, energy_history

def outer_opt(maximize=False, k_mag=1.0, num_steps=1000,lr=1e-4, inner_show=25, outer_show=25):
    ''' Optimization loop for coupling'''
    sign = -1.0 if maximize else 1.0
    k = Variable(torch.tensor(sign*k_mag), requires_grad=True)
    outer_optim = torch.optim.SGD((k,),lr=lr)
    k_history    = []
    loss_history = []
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for m in tqdm(range(num_steps)):
        outer_optim.zero_grad()
        min_phase, energy_history = inner_opt(k)
        loss      = sign*coherence(min_phase)
        loss_history.append(loss.data.numpy())
        k_history.append(k.clone().data.numpy())
        loss.backward()
        outer_optim.step()
        if m % outer_show ==0:
            for array, name in zip([energy_history, loss_history, k_history], ['potential', 'coherence', 'coupling']):
                if name == 'coherence' and maximize==True:
                    array = -1*np.array(array) 
                plt.plot(array) 
                plt.title(name)
                plt.savefig(os.path.join(save_dir, name + '.png'))
                plt.close()

if __name__=='__main__':
    '''This experiment tries to switch between coherence and incoherence of two oscillators using gradient descent. 
        * To switch from coherence to incoherence: maximize=True
        * To switch from incorherence to coherence: maximize=False
        k_mag controls the magnitude of the initial coupling. 
        When maximize is true, the sign of the initial coupling is negative. '''

    save_dir = os.path.join(os.path.expanduser('~'), 'oscillators')
    display_true = True
    if display_true:
        coh = []
        ks = np.linspace(-2,2,1000)
        for k in ks:
            phase, _ = inner_opt(k, num_steps=100)
            coh.append(coherence(phase).data.numpy())  
        plt.plot(ks,coh, 'b.')    
        plt.xlabel('K')
        plt.ylabel('Coherence')
        plt.savefig(os.path.join(save_dir, 'coupling_vs_coh.png'))
        plt.close()
    #outer_opt(maximize=True, k_mag = 1.0, lr=1e-4)
