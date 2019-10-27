DISPLAY=False
if DISPLAY:
    import matplotlib as plt
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
from nets import big_net, weights_init
import kuramoto as km
import kura_visual as kv
import numpy as np
import torch
import loss_func_ex
from make_data_new import polyomino_scenes, mask_pro
import matplotlib.pyplot as plt
import os, sys
import ipdb

if len(sys.argv) > 1:
    device_num = sys.argv[1]
    device = 'cuda:{}'.format(device_num)
    eta = float(sys.argv[2])
else:
    device = 'cpu'

plt.style.use('seaborn-darkgrid')


# Experiment parameters

# Saving directories
home_dir = os.path.expanduser('~/')
save_dir = os.path.join(home_dir, 'oscillators', 'eta{}'.format(eta))
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
num_polys = 3
# Batch Size
batch_size=64

# Data generator
generator = polyomino_scenes(n, img_side, num_polys, batch_size, rotation=False, noisy=True)

# Model parameters
learning_rate = 1e-2
training_steps = 5000
kernel_size=5
out_kernel_side=None

# Dynamics duration
episodes = 100
update_rate = .1
anneal   = False

# Loss is sum of frame potential and within-group synchrony
loss_func   = loss_func_ex.matt_loss_torch

# How frequently to report results
plot_when = 5
make_gif = False
gif_when  = 1

#Initialize network
coupling_network = big_net(img_side, 1, kernel_size=kernel_size, return_coupling=True, normalize_output=False, out_kernel_side=out_kernel_side).to(device)
in_freq_network = big_net(img_side, 1, kernel_size=kernel_size,return_coupling=False, out_kernel_side=out_kernel_side).to(device)
for layer in coupling_network.layers:
    weights_init(layer,w_std=0.1, b_std=0.1, w_mean=.5, b_mean=.5)
for layer in in_freq_network.layers:
    weights_init(layer,w_std=.1, b_std=.1, w_mean=2.0, b_mean=2.0)
# Display object
display = kv.displayer()

# Kuramoto object
osci = km.kura_torch(img_side**2, batch_size=batch_size, update_rate=update_rate)
loss_history = []
full_loss_history = []
# Optimizer
parameters = set()
for net in [coupling_network, in_freq_network]: parameters |= set(net.parameters())
train_op = torch.optim.Adam(parameters, lr=learning_rate)

# Run traning
for step in range(training_steps):
    train_op.zero_grad()

    # Get batch
    batch, sd = generator.generate_batch()
    mask  = torch.tensor(list(map(generator.mask_pro, sd))).float().to(device)

    # Produce coupling
    coupling = coupling_network(torch.tensor(batch).to(device).unsqueeze(1).float())
    in_freq = in_freq_network(torch.tensor(batch).to(device).unsqueeze(1).float()) 
    in_freq *= .1

    # Run dynamics
    osci.phase_init(device=device)
    phase_list, _ = osci.evolution(coupling, steps=episodes, anneal=anneal,in_freq=in_freq, record=True, record_torch=True)

    # Calculate loss
    full_loss = 0
    for t in range(episodes):
        loss, synch, desynch = loss_func(phase_list[t], mask, eta=eta)
        full_loss += (t+1)**2 * loss
    full_loss /= float(episodes)
    full_loss_history.append(full_loss.data.cpu().numpy())

    # Calculate gradients and backpropagate
    full_loss.backward()
    train_op.step()

    if step % plot_when == 0:
        loss_history.append([loss.data.cpu().numpy(), synch.data.cpu().numpy(), desynch.data.cpu().numpy()])
        plt.plot(np.array(loss_history))
        plt.legend(['Total', 'Synchrony Loss', 'Desynchrony Loss'],
               fontsize=8)
        plt.savefig(save_name + '.png')
        plt.close()
  
        plt.plot(np.array(full_loss_history))
        plt.title('Time-averaged loss')
        plt.savefig(os.path.join(save_dir, 'time_averaged_loss.png'))
        plt.close()

        for n, net in enumerate([coupling_network, in_freq_network]):
            fig, axes = plt.subplots(4,4)
            name = 'coupling' if n == 0 else 'int_freq'
            for a, ax in enumerate(axes.reshape(-1)):
                kernel = net.layers[0].weight[a,...].view((kernel_size, kernel_size)).data.cpu().numpy()
                ax.imshow(kernel)
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
        display.compute_properties(eta=eta)
        display.animate_evol_compare(img_side, img_side, compare=batch[0,...], save_name=save_name + '_' + str(step))
        display.phase_evol(save_name=save_name + '_' + str(step))
