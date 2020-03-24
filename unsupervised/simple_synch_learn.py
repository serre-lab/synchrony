import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import subprocess
from test_utils import cplx_imshow, save_cplx_anim
import sys, os
import ipdb

class net(torch.nn.Module):
    def __init__(self, kernel_side, num_features, sigma=1.0):
        super(net, self).__init__()
        self.conv = torch.nn.Conv2d(num_features, num_features, kernel_side, bias=False,padding=kernel_side)
        self.conv.weight.data.fill_(0)
        for n in range(num_features):
            self.conv.weight.data[n,n,...].fill_(1.0 / kernel_size**2)
    def forward(self, x):
        x_real = x[:,0,...]
        x_imag = x[:,1,...]
        x_real = self.conv(x_real)
        x_imag = self.conv(x_imag)
        return torch.cat([x_real.unsqueeze(1), x_imag.unsqueeze(1)], dim=1)

def energy(x, eta=1.0, normalize=False):
    x_mod = x[:,0,...]**2 + x[:,1,...]**2
    local_coherence  = x_mod.mean()
    global_coherence = x.mean((3,4))
    if normalize:
        norm = torch.sqrt(global_coherence[:,0,:]**2 + global_coherence[:,1,:]**2).unsqueeze(1)
        global_coherence = (global_coherence / norm)
    #TODO: CHECK THIS
    frame_potential = (torch.abs(torch.einsum('abc,abc->ab',global_coherence, global_coherence))**2).mean()
     
    #global_coherence = (global_coherence[:,0]**2 + global_coherence[:,1]**2).mean()
     
    return eta*frame_potential - local_coherence 

def opt(image, net, num_steps=25, lr=.1, eta=1.0, normalize=False):
    init_phase = 2*np.pi*torch.rand(image.shape).cuda()
    phase = Variable(init_phase, requires_grad=True)
    optimizer = optim.Adam((phase,), lr=lr)
    energy_history = []
    out_history     = []
    image_history  = []
    cplx_image = torch.cat([(image*torch.cos(phase.data)).unsqueeze(1), (image*torch.sin(phase.data)).unsqueeze(1)], dim=1)
    image_history.append(cplx_image.data.cpu().numpy())
    for n in tqdm(range(num_steps)):
        optimizer.zero_grad()
        out = net.forward(cplx_image)
        obj = energy(out, eta=eta, normalize=normalize)
        energy_history.append(obj.data.cpu().numpy())
        out_history.append(out.data.cpu().numpy())
        obj.backward()
        optimizer.step()
        cplx_image = torch.cat([(image*torch.cos(phase)).unsqueeze(1), (image*torch.sin(phase)).unsqueeze(1)], dim=1)
        image_history.append(cplx_image.data.cpu().numpy())
    return energy_history, image_history, out_history

def disp(net, save_dir, image_history, out_history, energy_history):
    image_history = np.array(image_history).transpose(1,0,2,3,4,5).sum(3)
    cplx_image_history = image_history[:,:,0,...] + 1j*image_history[:,:,1,...]
    energy_history = np.array(energy_history)
    out_history = np.array(out_history)

    plt.plot(energy_history)
    plt.savefig(os.path.join(save_dir, 'energy.png'))
    plt.close()

    #mean_region_angles = []
    #for r in range(num_regions):
    #    mean_region_angles.append(np.angle(cplx_image_history[:,:,:,r*5:(r+1)*5]).mean((2,3)))
    #mean_region_angles = np.array(mean_region_angles)
    #pairwise_diffs = np.sin(.5*(np.expand_dims(mean_region_angles,0) - np.expand_dims(mean_region_angles,1))).mean(2)
    #pd_means = pairwise_diffs.mean((0,1))
    #pd_std = pairwise_diffs.std((0,1))
    #time = np.arange(len(pd_means))
    #plt.plot(time,pd_means, 'k', color='#3F7F4C')
    #plt.fill_between(time, pd_means - pd_std, pd_means + pd_std, edgecolor='#3F7F4C',facecolor='#7EFF99')
    #plt.savefig(os.path.join(save_dir, 'bar_coherence.png'))
    #plt.close()

    cplx_image_history = image_history[:,:,0,...] + 1j*image_history[:,:,1,...]
    save_cplx_anim(os.path.join(save_dir, 'visible'), cplx_image_history, type='gif', number=image_history.shape[1])

if __name__=='__main__':
    num_regions = 4
    batch_size  = 10
    #alphas      = np.linspace(1,2,num_regions)
    alphas      = np.ones(num_regions)
    sigma       = 1.0
    lr          = 1e-1
    eta         = 1e0
    normalize   = False
    num_steps   = 1024
    save_dir    = '/home/matt/figs/synch_learn'
    kernel_size = 5
    image_side  = 5*num_regions
    image = torch.zeros(batch_size,num_regions,image_side, image_side)
    for a, alpha in enumerate(alphas):
        image[:,a,:,a*int(image_side / 4.0):(a+1)*int(image_side / 4.0)] = alpha
    my_net = net(kernel_size, num_features=num_regions).cuda()
    energy, image_history, out_history = opt(image.cuda(), my_net, num_steps=num_steps, lr=lr, eta=eta, normalize=normalize)
    disp(my_net, save_dir, image_history, out_history, energy)

