import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
from tqdm import tqdm
from test_utils import cplx_imshow, save_cplx_anim
import ipdb

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            #nn.Sigmoid(),#nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16,16,2,stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            #nn.Sigmoid(),#nn.ReLU(True),
            #nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
            nn.Conv2d(8,8,2,2,1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            #nn.Sigmoid(),#nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            #nn.Sigmoid(),#nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            #nn.Sigmoid()
        )

    def forward(self, d):
        res = []
        for c in range(d.shape[1]):
            x = d[:,c,...].unsqueeze(1)
            x = self.encoder(x)
            x = self.decoder(x)
            res.append(x)
        res = torch.stack(res).transpose(1,0)
        return res

def rho(z):
    return z[:,0,...]**2 + z[:,1,...]**2

def inner_opt(net, batch, inner_steps=50, inner_lr=1e-1, device='cpu'):
    phase_init = 2*np.pi*torch.rand_like(batch)
    phase = Variable(phase_init, requires_grad=True).to(device)
    phase_optim = torch.optim.Adam((phase,),lr=inner_lr)

    inner_loss = []
    complex_batch = torch.cat((batch*torch.cos(phase), batch*torch.sin(phase)), dim=1)
    in_phase_history = [complex_batch.data]
    out_phase_history = []

    for step in range(inner_steps):
        phase_optim.zero_grad()
        out = net.forward(complex_batch)
        out_phase_history.append(out.data)
        reconstruction = rho(out)
        loss = ((reconstruction - batch)**2).mean()
        inner_loss.append(loss.data)
        loss.backward()
        phase_optim.step()
        complex_batch = torch.cat((batch*torch.cos(phase), batch*torch.sin(phase)), dim=1)
        in_phase_history.append(complex_batch.data)
    return phase.data, inner_loss, in_phase_history, out_phase_history


def outer_opt(dg, lrs, inner_steps=25, show_every=25, device='cpu', save_dir='/media/data_cifs/cplx_auto/', make_gifs=False):
    net = autoencoder().to(device)
    outer_lr, inner_lr = lrs
    
    net_optim = torch.optim.Adam(net.parameters(), lr=outer_lr)
    
    outer_loss = []
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    l=0    
    for i, (batch, target) in tqdm(enumerate(dg)):
        net_optim.zero_grad()
        phase, inner_loss, in_phase_history, out_phase_history = inner_opt(net, batch.to(device), inner_steps=inner_steps, inner_lr=inner_lr) 
        cplx_batch = torch.cat((batch*torch.cos(phase), batch*torch.sin(phase)), dim=1)
        out = net.forward(cplx_batch)

        reconstruction = rho(out)
        loss = ((reconstruction - batch)**2).mean()
        l+=loss.data.cpu().numpy()
        loss.backward()
        net_optim.step()
        if i % show_every == 0 and i > 0:
            outer_loss.append(l / show_every)
            l=0
            plt.plot(outer_loss)
            plt.savefig(os.path.join(save_dir, 'outer_loss.png'.format(i))) 
            plt.close()

            plt.plot(inner_loss)
            plt.savefig(os.path.join(save_dir, 'inner_loss_{}.png'.format(i))) 
            plt.close()

            batch_ind = np.random.randint(batch.shape[0])
            # Reconstruction

            cplx_in = cplx_batch[batch_ind, 0, ...].data.cpu().numpy() + cplx_batch[batch_ind, 1, ...].data.cpu().numpy() * 1j
            cplx_out = out[batch_ind, 0, ...].data.cpu().numpy() + out[batch_ind, 1, ...].data.cpu().numpy() * 1j
            fig, axes = plt.subplots(1,4)
            axes[0].imshow(batch[batch_ind,...].squeeze().cpu().numpy())
            axes[1].imshow(reconstruction[batch_ind,...].data.squeeze().cpu().numpy())
            cplx_imshow(axes[2], cplx_in) 
            cplx_imshow(axes[3], cplx_out) 

            plt.savefig(os.path.join(save_dir, 'reconstruction_{}.png'.format(i)))
            plt.close() 
    
            if make_gifs:
                for name, phase_history in zip(['in', 'out'],[in_phase_history, out_phase_history]):
                    phase_history = torch.stack(phase_history).transpose(1,0).squeeze().numpy()
                    phase_history = phase_history[:,:,0,...] + phase_history[:,:,1,...]*1j
                    save_cplx_anim(os.path.join(save_dir, name + '_' + str(i) + '_'), phase_history,type='gif')
           
if __name__=='__main__':
    lrs = [1e-3, 1e-1]
    inner_steps = 10
    device='cuda'
    batch_size=32
    data_generator = torch.utils.data.DataLoader(
        datasets.MNIST('/media/data_cifs/yuwei/osci_save/data', train=True, download=True,
        transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)

    outer_opt(data_generator, lrs, inner_steps=inner_steps, show_every=25, make_gifs=True) 
