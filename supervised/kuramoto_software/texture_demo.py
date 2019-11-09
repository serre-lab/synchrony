DISPLAY=False
if DISPLAY:
    import matplotlib.pyplot as plt
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
import os
import time
import argparse
import torch as tc
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
import ipdb

# from kuramoto_software
import nets
import kuramoto as km
import kura_visual as kv
import loss_func_ex as lx


######################
# argument parser: if_cuda, data path, save path
parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-t', '--texture-types', type=int, default=2)
parser.add_argument('-sp', '--save-path', type=str, help='where to save data',default='new_experiment')
args = parser.parse_args()

######################
# device
args.device = None
if not args.disable_cuda and tc.cuda.is_available():
    args.device = tc.device('cuda')
else:
    args.device = tc.device('cpu')

######################
# parameters
img_side = 16
group_size = args.texture_types
train_data_num = 2048
train_batch_size = 32
train_epochs = 2000
show_every = 100
# one plausible way is to check coupling matrix, it should be pretty smooth not very structured
# If it is highly structured, need to change either network or learning rate
kura_update_rate = 0.5 # waiting to explore

episodes = 50 # If too small, strength needs to be pretty big (harm training); if too big, training is slow
learning_rate = 1e-4 # waiting to explore
sparsity_weight = 1e-4 # usually introduce large sparsity due to exploding gradients


######################
# check valid file function
def valid_file(file_path):
    if (int(file_path[-8:-4]) < train_data_num) and file_path.lower().endswith('npy'):
        return True
    else:
        return False


######################
# transform images into the right size
def transform(sample):
    return tc.tensor(sample).float().to(args.device)


######################
# path
load_dir = '/media/data_cifs/composite_textures_mini/'
save_dir = '/media/data_cifs/yuwei/osci_save/results/'

load_path = load_dir + '{}/train'.format(group_size)
save_path = save_dir + args.save_path

if not os.path.exists(save_path):
    os.mkdir(save_path)
######################
# training set
training_set = datasets.DatasetFolder(load_path, np.load, extensions=('.npy',), transform=transform)
#                                      is_valid_file=valid_file)
training_loader = DataLoader(training_set, batch_size=train_batch_size, shuffle=True)

######################
# initialization
model = nets.simple_conv().to(device=args.device)
#print('network contains {} parameters'.format(model.count_parameters(model))) # parameter number
#time.sleep(2)

osci = km.kura_torch2(img_side ** 2, device=args.device)
osci.set_ep(kura_update_rate)

rand_phase = (tc.rand(1, img_side ** 2) * 2 * np.pi).to(args.device)
batch_initial_phase = rand_phase.repeat(train_batch_size, 1).to(args.device)

loss_history = []
coupling_history = []

displayer = kv.displayer()
op = tc.optim.Adam(model.parameters(), lr=learning_rate)

######################
# test image
test_data = np.load(load_dir + '/{}/test/0/img_'.format(group_size) +
                    str(np.random.randint(0, 10000)).ljust(4, '0') + '.npy')
test_image = tc.tensor(test_data[0, ...].reshape(1, img_side, img_side)).float().to(args.device)
test_mask = tc.tensor(test_data[1:, ...].reshape(1, group_size, img_side ** 2)).float().to(args.device)
test_mask_colored = (tc.arange(group_size).unsqueeze(0).unsqueeze(-1).to(args.device).float()
                     * test_mask / group_size).sum(1)

######################
# training pipeline
for epoch in range(train_epochs):
    print('Training epoch {}'.format(epoch))
    # every epoch randomly select one from training set actually
    # it is not real validation data!

    valid_data = np.load(load_dir + '/{}/train/0/img_'.format(group_size) +
                         str(np.random.randint(0, train_data_num)).ljust(4, '0') + '.npy')
    valid_image = tc.tensor(valid_data[0, ...].reshape(1, img_side, img_side)).float().to(args.device)
    valid_mask = tc.tensor(valid_data[1:, ...].reshape(1, group_size, img_side ** 2)).float().to(args.device)

    valid_mask_colored = (tc.arange(group_size).unsqueeze(0).unsqueeze(-1).to(args.device).float()
                          * valid_mask / group_size).sum(1)
    if (epoch == 0) | ((epoch + 1) % show_every == 0):
        # validation example, save its coupling matrix
        osci.phase_init(initial_phase=rand_phase)
        coupling_valid = model(valid_image.unsqueeze(1))
        coupling_history.append(coupling_valid.cpu().data.numpy())
        phase_list_valid = osci.evolution(coupling_valid, steps=episodes, record=True)
        valid_phase_list = np.array([phase.cpu().data.numpy()[0] for phase in phase_list_valid])
        displayer.set_phases(valid_phase_list)
        displayer.set_masks(valid_mask.cpu().data.numpy())
        displayer.compute_properties()
        displayer.animate_evol_compare(img_side, img_side, compare=valid_image.cpu().data.numpy(),
                                       save_name=save_path + '/ani_valid_' + str(epoch))
        displayer.static_evol(img_side, img_side, valid_image.cpu().data.numpy(),
                              save_path + '/static_valid_' + str(epoch), valid_mask_colored.cpu().data.numpy())
        displayer.phase_evol(save_path + '/phase_valid_' + str(epoch) + '.png')

        im = plt.imshow(coupling_valid[0].cpu().data.numpy())
        plt.title('Coupling Matrix on Training Example (Epoch:{})'.format(epoch))
        plt.gca().grid(False)
        plt.axis('off')
        plt.colorbar(im)
        plt.savefig(save_path + '/coupling_valid_' + str(epoch) + '.png')
        plt.close()

        # save files
        tc.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': op.state_dict(),
                 'initial_phase': rand_phase}, save_path + '/model.pt')

        plt.plot(np.array(loss_history))
        plt.title('Time Averaged Loss')
        plt.savefig(save_path + '/loss' + '.png')
        plt.close()
        np.save('coupling.npy', np.array(coupling_history))

        # test example
        osci.phase_init(initial_phase=rand_phase)
        coupling_test = model(test_image.unsqueeze(1))
        phase_list_test = osci.evolution(coupling_test, steps=episodes, record=True)
        test_phase_list = np.array([phase.cpu().data.numpy()[0] for phase in phase_list_test])
        displayer.set_phases(test_phase_list)
        displayer.set_masks(test_mask.cpu().data.numpy())
        displayer.compute_properties()
        displayer.animate_evol_compare(img_side, img_side, compare=test_image.cpu().data.numpy(),
                                       save_name=save_path + '/ani_test_' + str(epoch))
        displayer.static_evol(img_side, img_side, test_image.cpu().data.numpy(),
                              save_path + '/static_test_' + str(epoch), test_mask_colored.cpu().data.numpy())
        displayer.phase_evol(save_path + '/phase_test_' + str(epoch) + '.png')

        im = plt.imshow(coupling_test[0].cpu().data.numpy())
        plt.title('Coupling Matrix on Test Example (Epoch:{})'.format(epoch))
        plt.gca().grid(False)
        plt.axis('off')
        plt.colorbar(im)
        plt.savefig(save_path + '/coupling_test_' + str(epoch) + '.png')
        plt.close()

    tavg_loss_batches = []

    for i, train_data in tqdm(enumerate(training_loader)):
        batch = train_data[0][:, 0, ...]
        mask = train_data[0][:, 1:, ...].reshape(-1, group_size, img_side * img_side)

        op.zero_grad()

        tavg_loss = tc.tensor(0.).to(args.device)

        coupling = model(batch.unsqueeze(1))

        osci.frequency_init(intrinsic_frequency=tc.zeros(batch.shape[0], batch.shape[1]*batch.shape[2]))
        osci.phase_init(initial_phase=batch_initial_phase[:batch.shape[0],...])
        phase_list = osci.evolution(coupling, steps=episodes, record=True)

        norm = np.sum(np.arange(1, episodes + 1) ** 2)
        for t in range(episodes):
            loss = lx.exinp_btw_groups_torch(phase_list[t], mask, args.device)
            # Originally this said '_torch2'
            tavg_loss += loss.mean() * (t ** 2)
        tavg_loss = tavg_loss / norm

        sparsity_penalty = sparsity_weight * tc.abs(coupling).mean()

        tavg_loss += sparsity_penalty

        tavg_loss_batches.append(tavg_loss.cpu().item())

        tavg_loss.backward()

        op.step()

    # avg loss for every epoch
    loss_history.append(np.mean(tavg_loss_batches))
