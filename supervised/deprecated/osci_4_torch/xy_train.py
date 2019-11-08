import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')
import torch
import numpy as np
from nets import small_net, big_net
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
import os
from tqdm import tqdm 
import ipdb
from test_utils import cplx_imshow

def metropolis(phase, p, prop_phase, prop_p, coupling, p_sigma):
   '''Accept or reject samples according to the metropolis-hastings criterion (thresholded Botlzmann factor): min(1, exp(-H(proposal) + H(current)))'''

   # Calculate proposed and current energy
   prop_energy = potential(coupling, prop_phase) + p_sigma * (prop_p**2).sum((1,2)) 
   cur_energy = potential(coupling, phase) + p_sigma * (p**2).sum((1,2)) 

   # Acceptance criterion
   acceptance_prob = torch.clamp(torch.exp(-1*prop_energy + cur_energy), 0.0, 1.0)
   # Determine acceptance
   accept = (torch.rand(1,).double() < acceptance_prob).unsqueeze(1).unsqueeze(2)
   return torch.where(accept, prop_phase, phase)

def potential(coupling, phase, grad=False):
    '''Calculate potential energy (U), or possibly its gradient.'''
    batch_size  = phase.shape[0]
    phase_diffs = phase.view(batch_size,-1).unsqueeze(2) - phase.view(batch_size,-1).unsqueeze(1)

    # Eliminate self-couplings
    diag_mask   = 1 - torch.eye(phase_diffs.shape[1]).double()

    if not grad:
        # Energy contributed by one phase pair is k_{ij}cos(theta_i - theta_j)
        pairwise_energies = coupling * diag_mask * torch.cos(phase_diffs)
        # Potential is negative, multiply by .5 to get rid of double counting. Mean makes potential an intensive property (size invariant).
        return -.5 * pairwise_energies.mean((1,2))
    else:
        # If need grad, then look at sine differences and negate. Mean is taken across only one dimension, since we want the gradient.
        return .5*(coupling * torch.sin(phase_diffs)).mean(2)

def hmc_update(phase, p, coupling, eps=1e-2, leapfrog=True):
    '''One step of Hamiltonian update, possibly using the leapfrog method.
        We assume Energy is U + K and that K = p^TMp so that p is Gaussian.'''
    phase_diffs = phase.unsqueeze(1) - phase.unsqueeze(0)
    
    current_eps = eps /2.0 if leapfrog else eps

    # p_dot = -dU/dphase
    p = p - (current_eps) * potential(coupling, phase, grad=True).reshape(p.shape[0],p.shape[1],p.shape[2]) 
    # phase_dot = dK/dp 
    phase = phase + eps*p
   
    # If leapfrog, complete the p update 
    if leapfrog:
        p = p - (eps / 2.0) * potential(coupling, phase, grad=True).reshape(p.shape[0], p.shape[1], p.shape[2])
    return phase, p

def hmc(phase, coupling, p_sigma=1.0, num_steps=1):
    '''Run HMC for num_steps steps. 
       1. Sample from momentum (p) marginal, which we assume to be Gaussian with std p_sigma.
       2. Run num_steps-long Hamiltonian dynamics, negating final p. 
       3. Accept or reject new phase proposed by the dynamics in step 2.'''

    # Sample p
    p = torch.normal(torch.zeros_like(phase), p_sigma * torch.ones_like(phase))
    prop_phase = phase.clone()
    prop_p     = p.clone()

    # Run Hamilton's equations
    for n in range(num_steps):
        prop_phase, prop_p = hmc_update(prop_phase, prop_p, coupling, leapfrog=True)

    # Accept using metropollis-hastings criterion
    # TODO: YOU MIGHT NOT ACCEPT ANYTHING. CRITERION?
    phase = metropolis(phase, p, prop_phase, -1*prop_p, coupling, p_sigma)
    return phase

def evaluate(phase_batch, coupling, p_sigma=1.0, num_samples=16, num_steps=256, save_dir=None):
    '''Generate num_samples batches of successive samples using HMC and save them.'''
    phase = 2*np.pi*torch.rand_like(phase_batch).double()
    phase_history = []
    for n in tqdm(range(num_samples)):
        phase = hmc(phase, coupling, p_sigma=p_sigma, num_steps=num_steps)
        phase_history.append(phase.data.numpy())
    fig, axes = plt.subplots(4,4) 

    for a, ax in enumerate(axes.reshape(-1)):
        cplx_img = np.cos(phase_history[a][0,...]) + 1j*np.sin(phase_history[a][0,...])
        cplx_imshow(ax, cplx_img)
    plt.savefig(os.path.join(save_dir, 'samples.png'))
    plt.close()

def train(my_net, dl, num_epochs=100, lr=1e-3, hmc_steps=10):
    '''Loops over the data set, calculates learning statistics (including by Hamiltonian MCMC), and applies gradients
        Gradient of log-likelihood is E_(data)[d potential / d background net] - E_(model)[d potential / d background net]. 
        The first expectation is calculated directly, and the second is estimated from HMC samples. We call these
        the positive and negative phases, respectively.'''

    optimizer = torch.optim.Adam(my_net.parameters(), lr=lr)
    save_dir  = os.path.join(os.path.expanduser('~'), 'oscillators')
    for e in range(num_epochs):
        # Evaluate
        test_batch = next(iter(dl))[0]
        phase_batch = torch.atan2(test_batch[:,1,...], test_batch[:,0,...])
        img_batch = torch.sqrt(test_batch[:,0,...]**2 + test_batch[:,1,...]**2).unsqueeze(1)
        with torch.no_grad():
            print('Evaluating')
            coupling = my_net.forward(img_batch)
            evaluate(phase_batch, coupling, p_sigma=1.0, num_steps=32, save_dir=save_dir) 
        print('Running epoch {}'.format(e))
        for phasor, _ in tqdm(dl):
            # Phasor is 2-channel complex data. We will split into phase and modulus
            phase_batch     = torch.atan2(phasor[:,1,...], phasor[:,0,...])
            img_batch = torch.sqrt(phasor[:,0,...]**2 + phasor[:,1,...]**2).unsqueeze(1)
            # Acquire coupling matrix from background net
            coupling = my_net.forward(img_batch)
 
            # Positive phase
            pos_potential = potential(coupling, phase_batch)
            # Calculate gradients of potential wrt net, keeping the graph for subsequent gradient computation.
            (-1*pos_potential.mean()).backward(retain_graph=True)
            pos_grads = [p.grad.clone().data for p in my_net.parameters()]
            optimizer.zero_grad()
    
            # Negative phase
            # Sample from the model using HMC, starting from the data.
            sample  = hmc(phase_batch, coupling, num_steps=hmc_steps)
            neg_potential = potential(coupling, sample)
            # Now gradient with respect to the model distribution
            (-1*neg_potential.mean()).backward()
            neg_grads = [p.grad.clone().data for p in my_net.parameters()]
            optimizer.zero_grad() 

            # Update
            # Apply gradient.
            for p, param in enumerate(my_net.parameters()):
                param.data.add_(lr*(pos_grads[p] - neg_grads[p]))

if __name__=='__main__':
    '''Training a fully-connected XY model with weights parameterized by a deep net.'''

    # Set up data
    #device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device    = 'cpu'
    batch_size=32
    data_type   = 'PHASE_SHAPES'
    num_objects = 3
    data_dir  = os.path.join(os.path.expanduser('~'), 'data', 'synch_data', data_type, str(num_objects))
    dataset   = DatasetFolder(data_dir, np.load, ('npy',))
    dl        = DataLoader(dataset, batch_size=batch_size,shuffle=True)

    # Load net
    my_net    = big_net(20,1,5).double().to(device)

    # Train net 
    train(my_net, dl, lr=1e-1, hmc_steps=10)
