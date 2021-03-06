import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import torch
import numpy as np
from make_data import polyomino_scenes
from oscillator import oscillator, run_osci
from tqdm import tqdm
import os
import ipdb

class coupling_net(torch.nn.Module):
    def __init__(self,img_side, num_layers, kernel_size, conv=False):
        super(coupling_net,self).__init__()
        self.img_side = img_side
        self.conv = conv
        if not conv:
            self.layers = torch.nn.ModuleList([torch.nn.Linear(img_side**2, img_side**2) for i in range(num_layers)]) 
        else:
            num_features = [1] + (num_layers + 1)*[4]
            minus_one      = 0 if kernel_size % 0 is True else 1
            pad_size     = int(np.floor((kernel_size - minus_one)/ 2.0))
            self.layers = torch.nn.ModuleList([torch.nn.Conv2d(num_features[i], num_features[i+1], kernel_size, padding=pad_size) for i in range(num_layers)])
        
    def forward(self,x):
        batch_size = x.shape[0]
        if not self.conv: x.view(batch_size, -1)
        for layer in self.layers:
            x = layer(x)
            x = torch.sigmoid(x)
        if self.conv:
            means = x.mean(1)
            stds  = x.std(1)
            x    = x - means.unsqueeze(1) 
            x    = x.view(batch_size, x.shape[1], -1)
            stds = stds.view(batch_size, -1)
            return torch.einsum('bci,bcj->bcij', x, x).mean(1) / torch.einsum('bi,bj->bij',stds,stds)
        else:
            return torch.einsum('bi,bj->bij',x,x)

if __name__=='__main__':
    loss_history = []
    polyomino_size = 1
    num_objects    = 2
    batch_size = 16
    img_side = 4
    num_layers  = 1
    rotations   = False
    training_steps = 5000
    lr             = 5e-3
    osc_steps      = 200
    show_every     = 32
    conv           = True
    kernel_size    = 1
    net = coupling_net(img_side, num_layers, kernel_size, conv=conv)
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)    
    generator = polyomino_scenes(polyomino_size, img_side, num_objects, batch_size=batch_size,rotations=rotations)
    for t in tqdm(range(training_steps)):
        batch, sd = generator.generate_batch()
        batch = torch.tensor(batch).float().unsqueeze(1)
        optimizer.zero_grad() 
        coupling_mats = net.forward(batch)
        osci = oscillator(batch, sd, epsilon=1e-1, K=1, img_side=img_side,coupling_mats=coupling_mats, classes=False, initial_frq=torch.zeros(batch_size, img_side**2))
        for i in range(osc_steps):
            osci.update()
        total_loss, synch_loss, desynch_loss = osci.seg_loss()
        total_loss.backward()
        if torch.any(net.layers[0].weight.grad != net.layers[0].weight.grad):
            ipdb.set_trace()
        optimizer.step()
        if t % show_every == 0:
            loss_history.append(np.array([total_loss.data.numpy(), synch_loss.data.numpy(), desynch_loss.data.numpy()]))
            plt.plot(np.array(loss_history))
            plt.legend(('Total', 'Synch', 'Desynch'))
            plt.savefig(os.path.join(osci.save_dir, 'coupling_net_loss.png'))
            plt.close()
