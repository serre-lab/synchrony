DISPLAY=False
if DISPLAY:
    import matplotlib as plt
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
import nets
import kuramoto as km
import kura_visual as kv
import numpy as np
import torch
import loss_func_ex
from make_data_new import polyomino_scenes, mask_pro
import matplotlib.pyplot as plt
import os
import ipdb

plt.style.use('seaborn-darkgrid')


# Experiment parameters

# Saving directories
home_dir = os.path.expanduser('~/')
save_dir = os.path.join(home_dir, 'oscillators')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Model name
model_name = '64_oscis_optim'
save_name =  os.path.join(save_dir, model_name)

# Side of image
img_side = 10
# Polyomino type
n = 4
# Number of polyominoes
num_polys = 3
# Batch Size
batch_size=64

# Data generator
generator = polyomino_scenes(n, img_side, num_polys, batch_size)

# Model parameters
learning_rate = 1e-4
training_steps = 5000

# Dynamics duration
episodes = 50

# Loss is sum of frame potential and within-group synchrony
loss_func   = loss_func_ex.matt_loss_torch

# How frequently to report results
report = 25
effect = 2

#Initialize network
network = nets.big_net(img_side, 3)

# Display object
display = kv.displayer()

# Kuramoto object
osci = km.kura_torch(img_side**2, batch_size=batch_size, update_rate=.25)
loss_history = []

# Optimizer
train_op = torch.optim.Adam(network.parameters(), lr=learning_rate)

# Run traning
for step in range(training_steps):
    train_op.zero_grad()

    # Get batch
    batch, sd = generator.generate_batch()
    mask  = torch.tensor(list(map(generator.mask_pro, sd))).float()

    # Produce coupling
    coupling = network(torch.tensor(batch).unsqueeze(1).float())

    # Run dynamics
    osci.phase_init()
    final_phase = osci.evolution(coupling, steps=episodes, anneal=True)

    # Calculate loss
    loss, synch, desynch = loss_func(final_phase, mask)

    # Calculate gradients and backpropagate
    loss.backward()
    train_op.step()
    print('STEPS: ' + str(step).ljust(4) + '--------------' + str(loss.data.numpy()))

    # Test and display
    if step % report == 0:
        osci.phase_init()
        phase_list, freqs_list = osci.evolution(coupling, steps=episodes, record=True, show=True, test=True, anneal=True)
        display.set_phases(np.array(phase_list)[:,0,:])
        display.set_masks(np.expand_dims(mask.data.numpy()[0,:], 0))
        display.compute_properties()
        display.animate_evol_compare(img_side, img_side, compare=batch[0,...], save_name=save_name + '_' + str(step))
        display.phase_evol(save_name=save_name + '_' + str(step))
        loss_history.append([loss.data.numpy(), synch.data.numpy(), desynch.data.numpy()])
        plt.plot(np.array(loss_history))
        plt.legend(['Total', 'Synchrony Loss', 'Desynchrony Loss'],
                   fontsize=8)
        plt.savefig(save_name + '.png')
        plt.close()
