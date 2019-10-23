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
loss_func1 = loss_func_ex.coh_btw_groups_torch
loss_func2 = loss_func_ex.coh_in_groups_torch
loss_func3 = loss_func_ex.exinp_btw_groups_torch
loss_func4 = loss_func_ex.fpt_btw_groups_torch
report = 100
effect = 2
batch_size=32
#train_data = np.load('train_data.npz')
#train_img = train_data['image']
#train_mask = torch.tensor(train_data['mask']).float()
generator = polyomino_scenes(5, img_side, 4, batch_size)


#network = net.net2() # architecture
network = nets.big_net(img_side, 3)
#network.apply(net.weights_init) # network initialization
display = kv.displayer() # visualization
osci = km.kura_torch(img_side**2)
loss_history = []

train_op = torch.optim.Adam(network.parameters(), lr=learning_rate)

for step in range(training_steps):
    train_op.zero_grad()

    batch, sd = generator.generate_batch()
    batch = torch.tensor(batch).unsqueeze(1).float()
    ipdb.set_trace()
    mask  = torch.tensor(list(map(generator.mask_pro, sd))).float()

    coupling = network(batch)
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
