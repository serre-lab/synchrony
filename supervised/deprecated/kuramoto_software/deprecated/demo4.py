DISPLAY=False
if DISPLAY:
    import matplotlib as plt
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
from nets import big_net, deep_net, weights_init
import kuramoto as km
import kura_visual as kv
import numpy as np
import torch
import loss_func_ex
from make_data_new import polyomino_scenes, mask_pro
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import os, sys
import argparse
import ipdb
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu')
parser.add_argument('--eta', type=float, default=.5)
parser.add_argument('--sparsity', type=float, default=1e-3)
args = parser.parse_args()

#plt.style.use('seaborn-darkgrid')


# Experiment parameters

# Saving directories
home_dir = os.path.expanduser('~/')
save_dir = os.path.join(home_dir, 'oscillators', 'eta{}_sparsity{}'.format(args.eta, args.sparsity))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Model name
model_name = '64_oscis_optim'
save_name =  os.path.join(save_dir, model_name)

# Side of image
img_side = 16
# Polyomino type
n = 4
# Number of polyominoes
num_polys = 2
# Batch Size
batch_size=64

# Data generator
generator = polyomino_scenes(n, img_side, num_polys, batch_size, rotation=False, noisy=False)

# Model parameters
learning_rate = 1e-3
training_steps = 5000
init_steps     = 1600
kernel_size=5
num_conv_features = 32
out_kernel_side=None

# Dynamics duration
episodes = 100
update_rate = 1.0
anneal   = False

# Loss is sum of frame potential and within-group synchrony
#loss_func   = loss_func_ex.matt_loss_torch
loss_func    = loss_func_ex.inp_btw_groups_torch

# How frequently to report results
plot_when = 5
make_gif = True
gif_when  = 1

#Initialize network
if args.device is not 'cpu':
    device = 'cuda:{}'.format(args.device)
else:
    device = args.device
coupling_network = big_net(img_side, 1, kernel_size=kernel_size, return_coupling=True, normalize_output=True, out_kernel_side=out_kernel_side, pretrained=True).to(device)
#coupling_network = deep_net(img_side, 1,1, kernel_size=kernel_size, pretrained=False).to(device)
in_freq_network = big_net(img_side, 1, kernel_size=kernel_size,return_coupling=False, out_kernel_side=out_kernel_side).to(device)

# Initialize Coupling Net
#target_mean = 0.0
#target_std  = 1e-2
#init_batch_size = 128
#init_optim = torch.optim.Adam(coupling_network.parameters(), lr=1e-4)
#init_loss_history = []
#init_show_every = 25
#print('RUNNING INITIALIZATION')
#for steps in tqdm(range(init_steps)):
#   batch, _ = generator.generate_batch() 
#   batch = torch.tensor(batch).to(device).unsqueeze(1).float()
#   coupling = coupling_network.forward(batch)
#   empirical_mean = coupling.mean()
#   empirical_std  = coupling.std()
#
#   loss = .5*(empirical_mean - target_mean)**2 + .5*(empirical_std - target_std)**2
#   init_loss_history.append(loss.data.cpu().numpy())
#   if steps % init_show_every == 0: 
#       print('Initialization loss: {}. Empirical Mean: {} Empirical std: {}'.format(init_loss_history[-1], empirical_mean.data.cpu().numpy(), empirical_std.data.cpu().numpy()))
#       plt.plot(init_loss_history)
#       plt.title('Initialization Loss')
#       plt.savefig(os.path.join(save_dir, 'init_loss.png'))
#   loss.backward()
#   init_optim.step()

display = kv.displayer()

# Kuramoto object
osci = km.kura_torch(img_side**2, batch_size=batch_size, update_rate=update_rate)
loss_history = []
time_loss_history = []
# Optimizer
parameters = set()
for net in [coupling_network, in_freq_network]: parameters |= set(net.parameters())
train_op = torch.optim.Adam(coupling_network.parameters(), lr=learning_rate)

init_phase = 2*np.pi*torch.rand((batch_size, img_side**2))
# Run traning
for step in range(training_steps):
    ipdb.set_trace()
    train_op.zero_grad()

    # Get batch
    batch, sd = generator.generate_batch()
    mask  = torch.tensor(list(map(generator.mask_pro, sd))).float().to(device)
    batch -= .5

    # Produce coupling
    coupling = coupling_network(torch.tensor(batch).to(device).unsqueeze(1).float())
    coupling -= coupling.mean((1,2)).unsqueeze(1).unsqueeze(2)
    ipdb.set_trace()
    coupling /= coupling.max((1,2)).unsqueeze(1).unsqueeze(2)
    #in_freq = in_freq_network(torch.tensor(batch).to(device).unsqueeze(1).float()) 
    #in_freq *= .1
    in_freq = None

    # Run dynamics
    osci.phase_init(init_phase, device=device)
    phase_list, _ = osci.evolution(coupling, steps=episodes, anneal=anneal,in_freq=in_freq, record=True, record_torch=True)

    # Calculate loss
    time_loss = 0
    for t in range(episodes):
        #loss, synch, desynch = loss_func(phase_list[t], mask, eta=args.eta, desynch_type='COS')
        loss, synch, desynch = loss_func(phase_list[t], mask, device=device)
        time_loss += (t+1)**2 * loss
    time_loss /= float(episodes)
    time_loss_history.append(time_loss.data.cpu().numpy())

    # Sparsity
    sparsity_penalty = args.sparsity * torch.abs(coupling).mean()

    # Calculate gradients and backpropagate
    full_loss = time_loss + sparsity_penalty
    full_loss.backward()
    train_op.step()

    if step % plot_when == 0:
        loss_history.append([loss.data.cpu().numpy(), synch.data.cpu().numpy(), desynch.data.cpu().numpy()])
        plt.plot(np.array(loss_history))
        plt.legend(['Total', 'Synchrony Loss', 'Desynchrony Loss'],
               fontsize=8)
        plt.savefig(save_name + '.png')
        plt.close()
  
        plt.plot(np.array(time_loss_history))
        plt.title('Time-averaged loss')
        plt.savefig(os.path.join(save_dir, 'time_averaged_loss.png'))
        plt.close()

        fig, axes = plt.subplots(img_side,img_side, figsize=(10., 10.)) 
        min_cpl = coupling[0].min().data.cpu().numpy()
        max_cpl = coupling[0].max().data.cpu().numpy()
        for a, ax in enumerate(axes.reshape(-1)):
            cpl = coupling[0,a,...].data.cpu().numpy().reshape(img_side, img_side)
            ax.imshow(cpl, cmap='gray', vmin=min_cpl, vmax=max_cpl)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        #cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
        #fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=min_cpl, vmax=max_cpl)), cmap='gray', cax=cbar_ax) 
        plt.savefig(os.path.join(save_dir, 'example_coupling.png'))
        plt.close()
            

        #plt.imshow(coupling[0,...].data.cpu().numpy())
        #plt.title('Example coupling')
        #plt.savefig(os.path.join(save_dir, 'example_coupling.png'))
        #plt.close()

        plt.imshow(batch[0,...], cmap='gray')
        plt.title('Example image')
        plt.savefig(os.path.join(save_dir, 'example_image.png'))
        plt.close()

        for n, net in enumerate([coupling_network, in_freq_network]):
            fig, axes = plt.subplots(4,4)
            name = 'coupling' if n == 0 else 'int_freq'
            for a, ax in enumerate(axes.reshape(-1)):
                kernel = net.conv_layers[0].weight[a,...].view((kernel_size, kernel_size)).data.cpu().numpy()
                ax.imshow(kernel, cmap='gray')
                #if a == 15:
                #    ax.colorbar(kernel)
            plt.savefig(os.path.join(save_dir, name + '_kernels.png'))
            plt.close()

    print('STEPS: ' + str(step).ljust(4) + '--------------' + str(loss.data.cpu().numpy()))

    # Test and display
    if step % gif_when == 0 and make_gif:
        osci.phase_init(device=device)
        phase_list, freqs_list = osci.evolution(coupling, steps=episodes, record=True, show=True, test=True, anneal=anneal, in_freq=in_freq)
        display.set_phases(np.array(phase_list)[:,0,:])
        display.set_masks(np.expand_dims(mask.data.cpu().numpy()[0,:], 0))
        #display.compute_properties(eta=args.eta)
        display.compute_properties()
        display.animate_evol_compare(img_side, img_side, compare=batch[0,...], save_name=save_name + '_' + str(step))
        display.phase_evol(save_name=save_name + '_' + str(step))
