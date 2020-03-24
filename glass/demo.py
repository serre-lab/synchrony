import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')
import torch
import numpy as np
import os
from tqdm import tqdm
import ipdb

class net(torch.nn.Module):
    def __init__(self, n, num_layers=3, activation=None, batch_size=64):
        super(net, self).__init__()
        self.in_size = int(n / 2**num_layers)        
        self.batch_size=64
        self.test_layer = torch.nn.Linear(10,10)

        if activation is not None:
            raise NotImplemented

        self.layers = []
        current_size = self.in_size
        for i in range(num_layers):
            self.layers.append(torch.nn.Linear(current_size, current_size * 2))
            current_size *= 2

    def forward(self):
        x = torch.normal(0,1,(self.batch_size, self.in_size)) 
        for layer in self.layers:
            x = layer(x)
        return x

def loss(theta):
   G = (torch.cos(theta).sum(1)**2).mean() + (torch.sin(theta).sum(1)**2).mean()
   theta_diffs = torch.abs(theta.unsqueeze(2) - theta.unsqueeze(1))
   Q = (torch.cos(theta_diffs).mean(1)**2).mean() + (torch.sin(theta_diffs).mean(1)**2).mean()
   return G, Q

save_dir = '/media/data_cifs/glass'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
n = 128
batch_size=64
lr=1e-3
step_size = .01
training_steps = 100
kuramoto_steps = 100
K_net     = net(n**2) 
omega_net = net(n)
parameters = tuple([param for param in K_net.parameters()] + [param for param in omega_net.parameters()])
optimizer = torch.optim.Adam(parameters, lr=lr)
show_every = 10

fl = 0
fl_history = []
for s in tqdm(range(training_steps)):

    K     = K_net.forward().reshape(batch_size,n,n)
    omega = omega_net.forward()

    theta = 2*np.pi*torch.rand((batch_size, n))
    G, Q = loss(theta)
    G_history = [G.data.numpy()]
    Q_history = [Q.data.numpy()]
    for t in range(kuramoto_steps):
        theta_diffs = torch.sin(theta.unsqueeze(2) - theta.unsqueeze(1))
        delta = omega + (K * theta_diffs).mean(2)
        theta += step_size * delta
        G, Q  = loss(theta)
        G_history.append(G.data.numpy())
        Q_history.append(Q.data.numpy())
    full_loss = G - Q
    fl += full_loss.data.numpy()
    full_loss.backward()
    optimizer.step()
    if s % show_every == 0 and s > 0:
        fl_history.append(fl / show_every)
        fl = 0
        for name, history in zip(['full', 'G', 'Q'], [fl_history, G_history, Q_history]):
            plt.plot(history)
            plt.savefig(os.path.join(save_dir, name + '.png'))
            plt.close()
