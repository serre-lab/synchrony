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

home_dir = os.path.expanduser('~/')
save_dir = os.path.join(home_dir, 'oscillators')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model_name = '64_oscis_optim'
save_name =  os.path.join(save_dir, model_name)
img_side = 16
learning_rate = 1e-4
training_steps = 5000
episodes = 100
loss_func   = loss_func_ex.matt_loss_torch
report = 100
effect = 2
batch_size=32
generator = polyomino_scenes(5, img_side, 4, batch_size)


network = nets.big_net(img_side, 3)
display = kv.displayer() # visualization
osci = km.kura_torch(img_side**2, batch_size=batch_size, update_rate=.25)
loss_history = []

train_op = torch.optim.Adam(network.parameters(), lr=learning_rate)

for step in range(training_steps):
    train_op.zero_grad()

    batch, sd = generator.generate_batch()
    mask  = torch.tensor(list(map(generator.mask_pro, sd))).float()

    coupling = network(torch.tensor(batch).unsqueeze(1).float())
    osci.phase_init()
    final_phase = osci.evolution(coupling, steps=50, anneal=True)
    loss, synch, desynch = loss_func(final_phase, mask)


    loss.backward()
    train_op.step()
    print('STEPS: ' + str(step).ljust(4) + '--------------' + str(loss.data.numpy()))
    if step % report == 0:
        osci.phase_init()
        phase_list, freqs_list = osci.evolution(coupling, steps=episodes, record=True, show=True, test=True, anneal=True)
        #display.set_phases(phase_list[-1])
        display.set_phases(np.array(phase_list)[:,0,:])
        #display.set_masks(train_data['mask'])
        display.set_masks(np.expand_dims(mask.data.numpy()[0,:], 0))
        display.compute_properties()
        display.animate_evol_compare(img_side, img_side, compare=batch[0,...], save_name=save_name + '_' + str(step))
        display.phase_evol(save_name=save_name + '_' + str(step))
        loss_history.append([loss.data.numpy(), synch.data.numpy(), desynch.data.numpy()])
        plt.plot(np.squeeze(np.array(loss_history)))
        plt.legend(['Total', 'Coherence btw groups', 'Coherence in groups', 'Exp inner product', 'frame potential'],
                   fontsize=8)
        plt.savefig(save_name + '.png')
        plt.close()
