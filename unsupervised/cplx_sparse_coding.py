import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader
from torch import optim
import ipdb
from tqdm import tqdm
import subprocess
from test_utils import cplx_imshow
import sys, os

def lp(x,p=1,batch_reduce=None):
    if batch_reduce==None:
        return (torch.abs(x)**p).sum(1)
    elif batch_reduce=='mean':
        return (torch.abs(x)**p).sum(1).mean()
    elif batch_reduce=='sum':
        return (torch.abs(x)**p).sum()

def rho(z):
    z_real = z[:,0,...]
    z_imag = z[:,1,...]
    return torch.sqrt(z_real**2 + z_imag**2)

def arg(z):
    z_real = z[:,0,...]
    z_imag = z[:,1,...]
    return torch.atan2(z_imag,z_real)

def weighted_frame_potential(z, batch_reduce=None):
    pairwise_inp = torch.einsum('bci,bcj->bij',z,z)
    if batch_reduce==None:
        return (torch.abs(pairwise_inp)**2).sum((1,2)) 
    elif batch_reduce=='mean':
        return (torch.abs(pairwise_inp)**2).sum((1,2)).mean()
    elif batch_reduce=='sum':
        return (torch.abs(pairwise_inp)**2).sum()

def energy(x, Phi, w):
    # Reconstruction
    Phi = torch.cat((Phi.unsqueeze(0), torch.zeros_like(Phi.unsqueeze(0))), dim=0)
    reconstruction = torch.einsum('bcm,cmn->bcn',w,Phi)
    gauss_energy = lp(rho(reconstruction) - x,  p=2, batch_reduce=None)
    # Sparsity
    sparsity = lp(rho(w),p=1,batch_reduce=None)
    
    # Desynchrony
    fp = weighted_frame_potential(w)
    
    return gauss_energy, sparsity, fp 

def optimize_codes(batch, Phi, code_size, lr, gamma1=1e-3, gamma2=1e-3, num_steps=50):
    batch_size = batch.shape[0]
    init_w = torch.normal(0,1,size=(batch_size, 2, code_size))
    w = Variable(init_w, requires_grad=True)
    optim_in = optim.Adam((w,), lr=lr)
    inner_loss = []
    inner_ge = []
    inner_sp = []
    inner_fp = []
    for n in range(num_steps):
        optim_in.zero_grad()
        ge, sp, fp = energy(batch.reshape(batch_size,-1), Phi, w)
        loss = (ge + gamma1*sp + gamma2*fp).mean()
        loss.backward()
        inner_loss.append(loss.data.cpu().numpy())
        inner_ge.append(ge.mean().data.cpu().numpy())
        inner_sp.append(sp.mean().data.cpu().numpy())
        inner_fp.append(fp.mean().data.cpu().numpy())
        optim_in.step()

    return w.data, [inner_loss, inner_ge, inner_sp, inner_fp]

def optimize_dict(dl, save_dir, data_size, code_size=64, lr=[1e-1,1e-3], gamma1=1e-3, gamma2=1e-3, num_epochs=10, num_steps=50, show_every=25, device='cpu'):
    lr1, lr2 = lr
    init_Phi = torch.normal(0,1e-1, size=(code_size, data_size))
    Phi = Variable(init_Phi, requires_grad=True)
    optim_out = optim.Adam((Phi,), lr=lr1)
    counter = 0
    outer_loss = []
    outer_ge = []
    outer_sp = []
    outer_fp = []
    for e in range(num_epochs):
        for b, (batch, target) in tqdm(enumerate(dl)):
            # Dataset is complex, so kill the phase
            batch = batch.to(device).float()
            #if len(batch.shape) > 3:
            optim_out.zero_grad()
            w, inner_energies = optimize_codes(batch, Phi.data, code_size, lr2, gamma1=gamma1,gamma2=gamma2,num_steps=num_steps)
            ge, sp, fp = energy(batch.reshape(batch.shape[0],-1), Phi, w)
            loss = (ge + gamma1*sp + gamma2*fp).mean()
            loss.backward()
            outer_loss.append(loss.data.cpu().numpy())
            outer_ge.append(ge.mean().data.cpu().numpy())
            outer_sp.append(sp.mean().data.cpu().numpy())
            outer_fp.append(fp.mean().data.cpu().numpy())
            optim_out.step()

            if counter%show_every == 0:
               print('Total loss at interation {} is {}'.format(counter,loss.data.cpu().numpy()))
               disp(Phi, save_dir, batch, w, inner_energies, [outer_loss, outer_ge, outer_sp, outer_fp])
            counter+=1
    return outer_loss

def disp(Phi, save_dir, batch, w, inner_energies, outer_energies):
    for e, energies in enumerate([inner_energies, outer_energies]):
        name = 'inner_energies.png' if e == 0 else 'outer_energies.png'
        plt.plot(np.array(energies).T)
        plt.legend(('Total', 'Reconstruction', 'Sparsity', 'Desynchrony'))
        plt.savefig(os.path.join(save_dir, name))
        plt.close()
    
    fig, axes = plt.subplots(4,4)
    for a, ax in enumerate(axes.reshape(-1)):
        filt = Phi.data[a,...].reshape(batch.shape[2], batch.shape[3]).cpu().numpy()
        ax.imshow(filt) 
    plt.savefig(os.path.join(save_dir, 'weights.png'))
    plt.close()

    fig, axes = plt.subplots(1,2)
    Phi = torch.cat((Phi.unsqueeze(0), torch.zeros_like(Phi.unsqueeze(0))), dim=0)
    img = torch.cat((batch[0,0,...].unsqueeze(0), torch.zeros_like(batch[0,0,...].unsqueeze(0))), dim=0).data.cpu().numpy()
    reconstruction = torch.einsum('bcm,cmn->bcn',w,Phi)[0,...].reshape(2,batch.shape[2], batch.shape[3]).data.cpu().numpy()
    arrays = [img, reconstruction]
    for (ax,array) in zip(axes, arrays):
        cplx_array = array[0,...] + 1j*array[1,...]
        cplx_imshow(ax, cplx_array) 
    plt.savefig(os.path.join(save_dir, 'cplx_reconstruction.png'))
    plt.close()
   
if __name__=='__main__':
    save_dir = '/home/matt/figs/synch_learn/sparse'
    ipdb.set_trace()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    subprocess.call('rm {}/*.png'.format(save_dir), shell=True)    
    kernel_size=12
    num_features=64 # Number of image patches = 2*sum_i=1^nb (kernel_size choose i)
    #TODO Make real dataset analgoues
    #TODO: Make modulus biases
    #ds = datasets.DatasetFolder('/home/matt/data/synch_data/PHASE_BARS/{}'.format(nb), np.load, extensions=('npy')) 
    ds = datasets.MNIST('/home/matt/data/synch_data/', train=True, download=True,
                                    transform=transforms.ToTensor())
    dl = DataLoader(ds, batch_size=100, shuffle=True)
    outer_energy = optimize_dict(dl, save_dir, 784)
    np.save('/home/matt/figs/synch_learn/energy_{}.npy'.format(run),outer_energy)
