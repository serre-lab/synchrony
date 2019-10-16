import net
import kuramoto as km
import kura_visual as kv
import numpy as np
import torch
import loss_func_ex
from make_data_new import polyomino_scenes, mask_pro
import matplotlib.pyplot as plt
import os

plt.style.use('seaborn-darkgrid')

save_dir = './eight1 '
model_name = '64_oscis_optim'
save_name = './eight/' + model_name
learning_rate = 1e-4
training_steps = 5000
episodes = 100
loss_func1 = loss_func_ex.coh_btw_groups_torch
loss_func2 = loss_func_ex.coh_in_groups_torch
loss_func3 = loss_func_ex.exinp_btw_groups_torch
loss_func4 = loss_func_ex.fpt_btw_groups_torch
report = 100
effect = 2

train_data = np.load('train_data.npz')
train_img = train_data['image']
train_mask = torch.tensor(train_data['mask']).float()

network = net.net2() # architecture
network.apply(net.weights_init) # network initialization
display = kv.displayer() # visualization
osci = km.kura_torch(64)
loss_history = []

train_op = torch.optim.Adam(network.parameters(), lr=learning_rate)

for step in range(training_steps):
    train_op.zero_grad()

    coupling = network(train_img, effect=effect)
    osci.phase_init()
    final_phase = osci.evolution(coupling, steps=50)
    loss1 = loss_func1(final_phase, train_mask)
    loss2 = 0.2 * loss_func2(final_phase, train_mask)
    loss3 = loss_func3(final_phase, train_mask)
    loss4 = loss_func4(final_phase, train_mask)

    loss = loss1 + loss2 + loss3 + loss4

    loss.backward()
    train_op.step()
    print('STEPS: ' + str(step).ljust(4) + '--------------' + str(loss.data.numpy()))
    if step % report == 0:
        osci.phase_init()
        phase_list, freqs_list = osci.evolution(coupling, steps=episodes, record=True, show=True, test=True)
        display.set_phases(phase_list)
        display.set_masks(train_data['mask'])
        display.compute_properties()
        display.animate_evol_compare(8, 8, compare=train_data['image'], save_name=save_name + '_' + str(step))
        display.phase_evol(save_name=save_name + '_' + str(step))
        loss_history.append([loss.data.numpy(), loss1.data.numpy(), loss2.data.numpy(),
                             loss3.data.numpy(), loss4.data.numpy()])
        plt.plot(np.squeeze(np.array(loss_history)))
        plt.legend(['Total', 'Coherence btw groups', 'Coherence in groups', 'Exp inner product', 'frame potential'],
                   fontsize=8)
        plt.savefig(save_name + '.png')
        plt.close()
