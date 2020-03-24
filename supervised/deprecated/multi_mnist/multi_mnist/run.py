import os
import net
import time
import argparse
import torch as tc
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import kura_visual as kv


"""
DataParallel
kuramoto and loss_func_ex are integrated into an nn.Module to operate parallel calculation
"""


def read_data(data_inds, path, valid=False):
    images = np.zeros((len(data_inds), img_side, img_side))
    masks = np.zeros((len(data_inds), group_size, img_side ** 2))
    b = 0
    for data_ind in data_inds:
        data = np.load(path + '/img_' + str(data_ind).rjust(4, '0') + '.npy')
        image = data[0, ...].reshape(1, img_side, img_side)
        mask = data[1:, ...].reshape(1, group_size, img_side ** 2)

        images[b, ...] = image
        masks[b, ...] = mask
        b += 1
    if not valid:
        return tc.tensor(images).float().to(args.device), tc.tensor(masks).float().to(args.device)
    else:
        return tc.tensor(images).float().detach().to(args.device), tc.tensor(masks).float().detach().to(args.device)


def display(displayer, phase_list, image, mask, coupling, path, name):
    colored_mask = (np.expand_dims(np.expand_dims(np.arange(group_size), axis=0), axis=-1) * mask / group_size).sum(1)
    displayer.set_phases(phase_list)
    displayer.set_masks(mask)
    displayer.compute_properties()

    displayer.animate_evol_compare(img_side, img_side, compare=image, save_name=path + '/ani_' + name)
    displayer.static_evol(img_side, img_side, image, path + '/static_' + name, colored_mask)
    displayer.phase_evol2(path + '/phase_' + name)

    im = plt.imshow(coupling)
    plt.title('Coupling Matrix')
    plt.gca().grid(False)
    plt.axis('off')
    plt.colorbar(im)
    plt.savefig(path + '/coupling_' + name)
    plt.close()


######################
# argument parser: if_cuda, data path, save path
parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-t', '--types', type=int)
parser.add_argument('-sp', '--save-path', type=str, help='where to save data')
args = parser.parse_args()

######################
# device
args.device = tc.device('cuda')
gpu_num = tc.cuda.device_count()
print("Assigned {} GPUs".format(gpu_num))

######################
# parameters
img_side = 36 # 36 is what Hinton's experiment used, with bounding box(20 *20, keep ratio) preserving <80% overlapping
group_size = args.types
num_cn = 8 # connected neighbours
critic_dist = 2 # distance from the center pixel
train_data_num = 10000 # to be changed
train_batch_size = 40 # to be changed
cv_batch_size = int(train_batch_size / 5)
real_train_data_num = train_data_num - int(train_data_num % train_batch_size)
train_epochs = 200
show_every = 20
# One plausible way is to check coupling matrix, it should be pretty smooth not very structured
# If it is highly structured, need to change either network or learning rate
kura_update_rate = 1.6 # waiting to explore

episodes = 8 # If too small, not stable; if too big, taking resources
anneal = 0.5 # waiting to explore
learning_rate = 1e-4 # This is probably the best
sparsity_weight = 1e-4 # usually introduce large sparsity due to exploding gradients
shuffle = True # shuffle reading order after every epoch

######################
# path
load_dir = '/media/data_cifs/yuwei/osci_save/data/multi-mnist3-new/'
save_dir = '/media/data_cifs/yuwei/osci_save/results/'

load_path = load_dir + '{}/train/0'.format(group_size)
save_path = save_dir + args.save_path

if not os.path.exists(save_path):
    os.mkdir(save_path)

######################
# save parameters
file = open(save_path + "/params.txt", "w")
L = ["img_side = {} \n".format(img_side),
     "group_size = {}\n".format(args.texture_types),
     "train_data_num = {}\n".format(train_data_num),
     "train_batch_size = {}\n".format(train_batch_size),
     "train_epochs = {}\n".format(train_epochs),
     "show_every = {}\n".format(show_every),
     "kura_update_rate = {}\n".format(kura_update_rate),
     "episodes = {}\n".format(episodes),
     "learning_rate = {}\n".format(learning_rate),
     "sparsity_weight = {}\n".format(sparsity_weight),
     "shuffle = {}\n".format(shuffle),
     "network=net.simple_conv\n",
     "num_cn={}\n".format(num_cn),
     "critic_dist={}\n".format(critic_dist),
     "anneal={}\n".format(anneal),
     "loss:exinp_integrate_torch"]
file.writelines(L)
file.close()

######################
# initialization
if tc.cuda.device_count() > 1:
    model = nn.DataParallel(net.simple_conv()).to(args.device)
    criterion = nn.DataParallel(net.criterion()).to(args.device)
else:
    model = net.simple_conv().to(args.device)
    criterion = net.criterion().to(args.device)
print('network contains {} parameters'.format(net.count_parameters(model))) # parameter number
time.sleep(2)

initial_phase = np.random.rand(1, img_side ** 2) * 2 * np.pi
rand_phase = tc.tensor(initial_phase).to('cpu')
batch_initial_phase = rand_phase.repeat(train_batch_size, 1).to(args.device)
cv_initial_phase = rand_phase.repeat(cv_batch_size, 1).detach().to(args.device)

loss_history = []
loss_cv_history = []
coupling_history = []

displayer = kv.displayer()
op = tc.optim.Adam(model.parameters(), lr=learning_rate)

######################
# connectivity
connectivity = np.zeros((img_side ** 2, num_cn))
for i in tqdm(range(img_side ** 2)):
    count = 0
    for j in range(img_side ** 2):
        x_1 = int(i % img_side)
        y_1 = int(i // img_side)
        x_2 = int(j % img_side)
        y_2 = int(j // img_side)
        dist = np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)
        if (dist < critic_dist) and (dist > 0):
            connectivity[i, count] = j
            count += 1
    while count < num_cn:
        connectivity[i, count:] = \
            sorted(np.delete(range(img_side ** 2), connectivity[i, ...]),
                   key=lambda k: np.random.random())[:int(num_cn - count)]
        count += 1
    change = sorted(range(num_cn), key=lambda k: np.random.random())[:int(num_cn / 2 + 1)]
    connectivity[i, ...][change] = \
        sorted(np.delete(range(img_side ** 2), connectivity[i, ...]),
               key=lambda k: np.random.random())[:int(num_cn / 2 + 1)]
connectivity = tc.tensor(connectivity).long().unsqueeze(0).to('cpu')
batch_connectivity = connectivity.repeat(train_batch_size, 1, 1).to(args.device)
cv_connectivity = connectivity.repeat(cv_batch_size, 1, 1).detach().to(args.device)

######################
# training pipeline
if not shuffle:
    training_order = np.arange(train_data_num)
    np.random.shuffle(training_order)
    training_order = training_order[:real_train_data_num].reshape(-1, train_batch_size)
norm = np.sum(np.arange(1, episodes + 1) ** 2)
for epoch in tqdm(range(train_epochs)):
    if shuffle:
        training_order = np.arange(train_data_num)
        np.random.shuffle(training_order)
        training_order = training_order[:real_train_data_num].reshape(-1, train_batch_size)

    for step in range(training_order.shape[0]):
        batch, mask_train = read_data(training_order[step], load_path)
        # print(batch.shape)
        # print(batch[0])

        op.zero_grad()

        phase_list_train, coupling_train = model(batch.unsqueeze(1), args.device,
                                                 kura_update_rate, anneal, episodes,
                                                 batch_initial_phase, batch_connectivity)

        tavg_loss = criterion(phase_list_train, mask_train, args.device)
        tavg_loss = tavg_loss.mean() / norm
        tavg_loss += sparsity_weight * tc.abs(coupling_train).mean()

        tavg_loss.backward()
        op.step()

        if step == training_order.shape[0] - 1:
            # cross-validation
            cv_ordering = np.arange(128)
            np.random.shuffle(cv_ordering)
            cv_image, cv_mask = read_data(cv_ordering[:cv_batch_size], load_dir + '{}/valid/0'.format(group_size),
                                          valid=True)

            phase_list_cv, coupling_cv = model(cv_image.unsqueeze(1), args.device,
                                               kura_update_rate, anneal, episodes,
                                               cv_initial_phase, cv_connectivity)

            # phase.shape=(time, batch, N)
            # mask.shape=(time, batch, group, N)
            # return loss.shape=(time * batch,)
            tavg_loss_cv = criterion(phase_list_cv, cv_mask, args.device, True)
            tavg_loss_cv = tavg_loss_cv.mean() / norm
            tavg_loss_cv += sparsity_weight * tc.abs(coupling_cv).mean()
            loss_cv_history.append(tavg_loss_cv.cpu().data.numpy())
            loss_history.append(tavg_loss.cpu().data.numpy())

        # visualize training
        if (step == 0) & ((epoch == 0) | ((epoch + 1) % show_every == 0)):
            train_ind = np.random.randint(train_batch_size)
            train_image = batch[train_ind].cpu().data.numpy()
            train_mask = mask_train[train_ind].cpu().unsqueeze(0).data.numpy()
            coupling_train_show = \
                tc.zeros(1, img_side ** 2, img_side ** 2).to('cpu').scatter_(dim=2, index=connectivity,
                                                        src=coupling_train[train_ind].unsqueeze(0).cpu()).data.numpy()[0]
            coupling_history.append(coupling_train_show)
            train_phase_list = np.array([phase.cpu().data.numpy()[train_ind, :] for phase in phase_list_train])
            display(displayer, train_phase_list, train_image, train_mask, coupling_train_show, save_path,
                    'train{}'.format(epoch))

    # visualize validation and save
    if (epoch == 0) | ((epoch + 1) % show_every == 0):
        # validation example, save its coupling matrix
        valid_ind = np.random.randint(cv_batch_size)
        valid_image = cv_image[valid_ind].cpu().data.numpy()
        valid_mask = cv_mask[valid_ind].cpu().unsqueeze(0).data.numpy()
        coupling_valid_show = \
            tc.zeros(1, img_side ** 2, img_side ** 2).to('cpu').scatter_(dim=2, index=connectivity,
                                                                         src=coupling_cv[valid_ind].cpu().unsqueeze(
                                                                             0)).data.numpy()[0]
        valid_phase_list = np.array([phase.cpu().data.numpy()[valid_ind, :] for phase in phase_list_cv])
        display(displayer, valid_phase_list, valid_image, valid_mask, coupling_valid_show, save_path,
                'valid{}'.format(epoch))

        # save files
        tc.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': op.state_dict(),
                 'initial_phase': rand_phase,
                 'connectivity': connectivity}, save_path + '/model{}.pt'.format(epoch))
        np.save('coupling.npy', np.array(coupling_history))

    plt.plot(np.array(loss_history))
    plt.plot(np.array(loss_cv_history))
    plt.title('Time Averaged Loss')
    plt.legend(['train', 'valid'])
    plt.savefig(save_path + '/loss' + '.png')
    plt.close()
