import torch as tc
import kuramoto as km
import kura_visual as kv
import loss_func_ex as lx
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import os

data = np.load('./train-data/train_data1.npz')

save_path = './three/optim_64_k2'
if not os.path.exists(save_path):
    os.mkdir(save_path)
learning_rate = 1e-2
episodes = 50
training_step = 2000
show_every = 500

coupling = tc.ones(1, 64, 64, requires_grad=True)
osci = km.kura_torch(64)
osci.set_ep(0.1)
training_mask = tc.tensor(data['mask']).float()
# Start from a fixed initialization, just for simplicity
initial_phase = tc.rand(1, 64) * 2 * np.pi
loss_history = []
coupling_history = [coupling.data.numpy()[0]]

displayer = kv.displayer()

op = tc.optim.Adam((coupling,), lr=learning_rate)

for step in tqdm.tqdm(range(training_step)):
    op.zero_grad()

    osci.phase_init(initial_phase)
    phase_list, _ = osci.evolution(coupling, steps=episodes, record=True)

    full_loss = 0

    norm = np.sum(np.arange(1, episodes + 1) ** 2)

    for t in range(episodes):
        loss = lx.exinp_btw_groups_torch(phase_list[t], training_mask)
        full_loss += loss * (t**2)
    full_loss = full_loss / norm
    loss_history.append(full_loss.data.numpy())

    if (step == 0) | ((step + 1) % show_every == 0):
        coupling_history.append(coupling.data.numpy()[0])
        phase_list = [phase.data.numpy() for phase in phase_list]

        displayer.set_phases(phase_list)
        displayer.set_masks(training_mask.data.numpy())
        displayer.compute_properties()
        displayer.animate_evol_compare(8, 8, compare=data['image'],
                                       save_name=save_path + '/ani' + str(step))
        displayer.phase_evol(save_path + '/phase_' + str(step) + '.png')

        im = plt.imshow(coupling.data.numpy()[0])
        plt.title('coupling')
        plt.gca().grid('off')
        plt.colorbar(im)
        plt.savefig(save_path + '/coupling' + str(step) + '.png')
        plt.close()

    full_loss.backward()

    op.step()

plt.plot(np.array(loss_history))
plt.title('full loss')
plt.savefig(save_path + '/loss' + '.png')
plt.close()

np.save('coupling', np.array(coupling_history))
