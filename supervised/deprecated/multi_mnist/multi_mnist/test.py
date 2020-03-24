import numpy as np
import torch as tc
import net
import kura_visual as kv
import tqdm
import os

epoch = 19

load_path = '/media/data_cifs/yuwei/osci_save/data/multi-mnist3-new/3/test/0'
model_path = '/media/data_cifs/yuwei/osci_save/results/multi_mnist_v9'
save_path = '/media/data_cifs/yuwei/osci_save/results/multi_mnist_v9/epoch{}_test'.format(epoch)

if not os.path.exists(save_path):
    os.mkdir(save_path)

img_side = 36
group_size = 3
data_num = 16

checkpoint = tc.load(model_path + '/model{}.pt'.format(epoch), map_location=tc.device('cpu'))
model = tc.nn.DataParallel(net.simple_conv())

model.load_state_dict(checkpoint['model_state_dict'])
initial_phase = checkpoint['initial_phase']
connectivity = checkpoint['connectivity']
model.eval()

for i in tqdm.tqdm(range(data_num)):
    test_data = np.load(load_path + '/img_{}.npy'.format(str(i).rjust(4, '0')))
    test_image = tc.tensor(test_data[0, ...]).float().reshape(1, img_side, img_side).unsqueeze(1)
    test_mask = tc.tensor(test_data[1:, ...]).float().reshape(1, group_size, img_side ** 2)

    episodes = 8

    displayer = kv.displayer()

    phase_list, coupling = model(test_image, 'cpu', 1.8, 0.5, 8, initial_phase, connectivity)
    phase_list = [phase.data.numpy()[0] for phase in phase_list]

    test_mask_colored = (tc.arange(group_size).unsqueeze(0).unsqueeze(-1).to(tc.device('cpu'))
                              * test_mask / group_size).sum(1)

    displayer.set_phases(phase_list)
    displayer.set_masks(test_mask.data.numpy())
    displayer.compute_properties()
    displayer.animate_evol_compare(img_side, img_side, compare=test_image[0][0].data.numpy(),
                                   save_name=save_path + '/ani_test' + str(i))
    displayer.static_evol(img_side, img_side, test_image[0][0].data.numpy(), save_path + '/static_test' + str(i),
                          test_mask_colored)
    displayer.phase_evol2(save_path + '/phase_test' + str(i) + '.png')
