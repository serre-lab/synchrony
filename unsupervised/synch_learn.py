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
from tqdm import tqdm
import subprocess
from test_utils import cplx_imshow
import sys, os
import ipdb

class net(torch.nn.Module):
    def __init__(self, num_features, kernel_side, bias=False, positive=False):
        super(net, self).__init__()
        self.conv = torch.nn.Conv2d(1,num_features,kernel_side, bias=bias)
        if positive:
            #self.conv.weight.data = torch.where(self.conv.weight.data >=0, self.conv.weight.data, torch.zeros_like(self.conv.weight.data))
            self.conv.weight.data.sub_(self.conv.weight.data.min())
        self.conv.weight.data.div_(torch.sqrt((self.conv.weight**2).sum()))
        self.bias=bias
        self.positive=positive
        if bias:
            self.conv.bias.data = torch.where(self.conv.bias.data < -1, -1*torch.ones_like(self.conv.bias.data), self.conv.bias)
            self.conv.bias.data = torch.where(self.conv.bias.data > 1,   1*torch.ones_like(self.conv.bias.data), self.conv.bias)
    def forward(self, x):
        x_real = x[:,0,...].unsqueeze(1)
        x_imag = x[:,1,...].unsqueeze(1)
        return torch.cat([self.conv(x_real).unsqueeze(1), self.conv(x_imag).unsqueeze(1)], dim=1)

def energy(y, normalize=False):
    lo = (y[:,0,...]**2 + y[:,1,...]**2).mean()
    feature_coherence = y.mean(dim=(3,4))
    if normalize==True:
        norm = torch.sqrt(feature_coherence[:,0,...]**2 + feature_coherence[:,1,...]**2).unsqueeze(1)
        feature_coherence = feature_coherence / norm
    fc_real = feature_coherence[:,0,...]
    fc_imag = feature_coherence[:,1,...]
    go = .5*(torch.einsum('bi,bj->bij', fc_real, fc_real) + torch.einsum('bi,bj->bij',fc_imag,fc_imag)).mean(0)
    go[range(len(go)), range(len(go))] = 0.0
    go = (torch.abs(go)**2).mean()
    return lo, go

def inner_opt(batch, net, lr, eta=1.0, num_steps=50, normalize=False):
    init_phase = 2*np.pi*torch.rand(batch.shape).cuda()
    phase = Variable(init_phase, requires_grad=True)
    optim_in = optim.Adam((phase,), lr=lr)
    inner_energy = []
    inner_lo = []
    inner_go = []
    init_batch = torch.cat([batch*torch.cos(phase.data), batch*torch.sin(phase.data)], dim=1)
    for n in range(num_steps):
        cplx_batch = torch.cat([batch*torch.cos(phase), batch*torch.sin(phase)], dim=1)
        optim_in.zero_grad()
        out = net.forward(cplx_batch)
        lo, go = energy(out, normalize=normalize)
        obj = eta*go-lo
        inner_energy.append(obj.data.cpu().numpy())
        inner_lo.append(lo.data.cpu().numpy())
        inner_go.append(go.data.cpu().numpy())
        obj.backward()
        optim_in.step()
    #TODO: Check if this has been updated

    final_batch = torch.cat([batch*torch.cos(phase.data), batch*torch.sin(phase.data)], dim=1)
    return final_batch,init_batch, [inner_energy, inner_lo, inner_go]

def outer_opt(net, dl, save_dir, lr=[1e-1,1e-3], eta=1.0, num_epochs=10,show_every=25, normalize=False):
    lr1, lr2 = lr
    optim_out = optim.Adam(net.parameters(), lr=lr1)
    counter = 0
    outer_energy = []
    lo_energy = []
    go_energy = []
    for e in range(num_epochs):
        for b, (batch, target) in tqdm(enumerate(dl)):
            # Dataset is complex, so kill the phase
            batch = batch.cuda().float()
            #if len(batch.shape) > 3:
            #    batch = (batch[:,0,...]**2 + batch[:,1,...]**2).unsqueeze(1).cuda().float()
            optim_out.zero_grad()
            argmax_batch, init_batch, inner_energies = inner_opt(batch, net, lr2, eta=eta,normalize=normalize)
            out = net.forward(argmax_batch) 
            lo, go = energy(out, normalize=normalize)
            obj = eta*go - lo
            outer_energy.append(obj.data.cpu().numpy())
            lo_energy.append(lo.data.cpu().numpy())
            go_energy.append(go.data.cpu().numpy())
            obj.backward()
            optim_out.step()
            if net.positive:
                #net.conv.weight.data = torch.where(net.conv.weight.data >=0, net.conv.weight.data, torch.zeros_like(net.conv.weight.data))
                net.conv.weight.data.sub_(net.conv.weight.data.min())
            net.conv.weight.data.div_(torch.sqrt((net.conv.weight**2).sum()))
            if net.bias:        
                net.conv.bias.data = torch.where(net.conv.bias.data < -1, -1*torch.ones_like(net.conv.bias.data), net.conv.bias)
                net.conv.bias.data = torch.where(net.conv.bias.data > 1,   1*torch.ones_like(net.conv.bias.data), net.conv.bias)

            if counter%show_every == 0:
               print('Energy at interation {} is {}'.format(counter,obj.data.cpu().numpy()))
               disp(net, save_dir, [init_batch, argmax_batch], inner_energies, [outer_energy, lo_energy, go_energy], out.data)
            counter+=1
    return outer_energy
def disp(net, save_dir, phase_arrays, inner_energies, outer_energies, out):
    for e, energies in enumerate([inner_energies, outer_energies]):
        name = 'inner_energies.png' if e == 0 else 'outer_energies.png'
        plt.plot(np.array(energies).T)
        plt.legend(('Total', 'Local Coherence', 'Global Coherence'))
        plt.savefig(os.path.join(save_dir, name))
        plt.close()
    
    fig, axes = plt.subplots(4,4)
    for a, ax in enumerate(axes.reshape(-1)):
        filt = net.conv.weight.data[a,...].reshape(kernel_size, kernel_size).cpu().numpy()
        ax.imshow(filt) 
    plt.savefig(os.path.join(save_dir, 'weights.png'))
    plt.close()

    fig, axes = plt.subplots(1,2)
    for (ax,array) in zip(axes, phase_arrays):
        array = array.data.cpu().numpy()[0,...]
        cplx_array = array[0,...] + 1j*array[1,...]
        cplx_imshow(ax, cplx_array) 
    plt.savefig(os.path.join(save_dir, 'inner_opt.png'))
    plt.close()

    fig, axes = plt.subplots(4,4)
    for a, ax in enumerate(axes.reshape(-1)):
        feature_map = out[0,:,a,...].cpu().numpy()
        cplx_feature_map = feature_map[0,...] + 1j*feature_map[1,...] 
        cplx_imshow(ax, cplx_feature_map)
    plt.savefig(os.path.join(save_dir, 'feature_maps.png'))
    plt.close()
    
    if net.bias:
        biases = net.conv.bias.data.cpu().numpy()
        plt.hist(biases)
        plt.savefig(os.path.join(save_dir, 'biases.png'))
        plt.close()
   
if __name__=='__main__':
    eta = float(sys.argv[1])
    print('Running experiment with eta={}'.format(eta))
    run = sys.argv[2] 
    save_dir = '/home/matt/figs/synch_learn/run_{}'.format(run)
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
    my_net = net(num_features, kernel_size, bias=False, positive=True).cuda()
    outer_energy = outer_opt(my_net, dl, save_dir, lr=[5e-4, 1e-1], eta=eta, show_every=25, num_epochs=3, normalize=True)
    np.save('/home/matt/figs/synch_learn/energy_{}.npy'.format(run),outer_energy)
