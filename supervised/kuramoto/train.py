import os
import nets
import time
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import display as disp
import sys
from utils import *
import ipdb
#import warnings
#warnings.filterwarnings('ignore')
"""
DataParallel
kuramoto and loss_func_ex are integrated into an nn.Module to operate parallel calculation
"""
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='composite_textures_mini')
parser.add_argument('--exp_name', type=str, default='four_texture_segment')
parser.add_argument('--model_name', type=str, default='simple_conv')
parser.add_argument('--disable_cuda', type=bool, default=False)
parser.add_argument('--interactive', type=str, default=False)
parser.add_argument('--img_side', type=int, default=32)
parser.add_argument('--segments', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--train_epochs', type=int, default=200)
parser.add_argument('--time_steps', type=int, default=8)
parser.add_argument('--time_weight', type=int, default=2)
parser.add_argument('--rand_init_phase', type=bool, default=False)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--update_rate', type=float, default=1.6)
parser.add_argument('--sparsity_weight', type=float, default=1e-5)
parser.add_argument('--small_world', type=bool, default=False)
args = parser.parse_args()

if args.interactive:
    import matplotlib.pyplot as plt
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
    
######################
# device
if not args.disable_cuda:
    gpu_num = torch.cuda.device_count()
    print("Assigned {} GPUs".format(gpu_num))
    device = 'cuda'
else:
    device = 'cpu'

 ######################
# parameters
num_cn = 8 # connected neighbours
critic_dist = 2 # distance from the center pixel
num_test = 1000
show_every = 20

anneal = 0.5 # waiting to explore
######################
# path
load_dir = os.path.join('/media/data_cifs/yuwei/osci_save/data/', args.data_name, str(args.segments))
save_dir = os.path.join('/media/data_cifs/yuwei/osci_save/results/', args.exp_name)
train_path = load_dir + '/train'
test_path = load_dir + '/test'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
		
######################
# Load data
#training set
training_set = datasets.DatasetFolder(train_path, np.load, extensions=('npy',), transform=transforms.ToTensor())
training_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
# testing set
testing_set = datasets.DatasetFolder(test_path, np.load, extensions=('npy',), transform=transforms.ToTensor())
testing_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True,
	drop_last=True)
    
######################
# save parameters
#file = open(save_path + "/params.txt", "w")
#L = [key + '= {}'.format(value) for (key, value) in kwargs.items]
#file.writelines(L)
#file.close()

######################
# initialization
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(nets.load_net(args.model_name, img_side=args.img_side)).to(device)
    criterion = nn.DataParallel(nets.criterion()).to(device)
else:
    model = nets.load_net(args.model_name, img_side=args.img_side).to(device)
    criterion = nets.criterion().to(device)
    print('network contains {} parameters'.format(nets.count_parameters(model))) # parameter number
time.sleep(2)

if not args.rand_init_phase:
    initial_phase = np.random.rand(1, args.img_side ** 2) * 2 * np.pi
    rand_phase = torch.tensor(initial_phase).to('cpu')
    batch_initial_phase = rand_phase.repeat(args.batch_size, 1).to(device)
    test_initial_phase = rand_phase.repeat(args.batch_size, 1).detach().to(device)

loss_history = []
loss_test_history = []
coupling_history = []

displayer = disp.displayer(interactive=args.interactive)
op = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

######################
# connectivity
print('Generating fixed adjancency matrix.')
if args.small_world:
    connectivity = sw_connectivity(args.img_side, num_cn, critic_dist)
else:
    connectivity = np.ones((args.img_side ** 2, num_cn))
connectivity = torch.tensor(connectivity).long().unsqueeze(0).to('cpu')
batch_connectivity = connectivity.repeat(args.batch_size, 1, 1).to(device)
test_connectivity = connectivity.repeat(args.batch_size, 1, 1).detach().to(device)

######################
# training pipeline
norm = np.sum(np.arange(1, args.time_steps + 1) ** 2)
counter = 0
for epoch in range(args.train_epochs):
    print('Epoch: {}'.format(epoch))

    for train_data, _ in tqdm(training_loader):
        batch = train_data[:, :,0, ...].to(device).float()
        #mask = train_data[:, :, 1:, ...].reshape(-1, args.segments, args.img_side * args.img_side)
        mask = train_data[:,:,1:,...].transpose(2,1).to(device).float()
        if args.rand_init_phase:
            batch_initial_phase = 2*np.pi*torch.rand((args.batch_size, args.img_side**2))

        op.zero_grad()
        phase_list_train, coupling_train = model(batch.unsqueeze(1), device,
                                             args.update_rate, anneal, args.time_steps,
                                             batch_initial_phase, batch_connectivity)

        ipdb.set_trace()
        tavg_loss = criterion(phase_list_train, mask, device)
        tavg_loss = tavg_loss.mean() / norm
        tavg_loss += args.sparsity_weight * torch.abs(coupling_train).mean()

        tavg_loss.backward()
        op.step()
		
        # visualize training
        if counter % show_every == 0:
            train_ind = np.random.randint(args.batch_size)
            train_image = batch[train_ind].cpu().data.numpy()
            train_mask = mask[train_ind].cpu().unsqueeze(0).data.numpy()
            coupling_train_show = \
                torch.zeros(1, args.img_side ** 2, args.img_side ** 2).to('cpu').scatter_(dim=2, index=connectivity,
                                                    src=coupling_train[train_ind].unsqueeze(0).cpu()).data.numpy()[0]
            coupling_history.append(coupling_train_show)
            train_phase_list = np.array([phase.cpu().data.numpy()[train_ind, :] for phase in phase_list_train])
            show(displayer, train_phase_list, train_image, train_mask, coupling_train_show, save_dir,
                'train{}'.format(epoch))

    for step, test_data in tqdm(enumerate(testing_loader)):
        # cross-validation
        batch = test_data[0][:, 0, ...]
        mask = test_data[0][:, 1:, ...].reshape(-1, args.segments, args.img_side * args.img_side)
     
        phase_list_test, coupling_test = model(test_image.unsqueeze(1), device,
                                           args.update_rate, anneal, args.time_steps,
                                           test_initial_phase, test_connectivity)
        # phase.shape=(time, batch, N)
        # mask.shape=(time, batch, group, N)
        # return loss.shape=(time * batch
        tavg_loss_test = criterion(phase_list_test, test_mask, device, True)
        tavg_loss_test = tavg_loss_test.mean() / norm
        tavg_loss_test += args.sparsity_weight * torch.abs(coupling_test).mean()
        loss_test_history.append(tavg_loss_test.cpu().data.numpy())
	loss_history.append(tavg_loss.cpu().data.numpy())
		
	if step*args.batch_size > num_test:
	    break

    # # visualize validation and save
    # if (epoch == 0) | ((epoch + 1) % show_every == 0):
        # # validation example, save its coupling matrix
        # valid_ind = np.random.randint(args.batch_size)
        # valid_image = cv_image[valid_ind].cpu().data.numpy()
        # valid_mask = cv_mask[valid_ind].cpu().unsqueeze(0).data.numpy()
        # coupling_valid_show = \
            # torch.zeros(1, args.img_side ** 2, args.img_side ** 2).to('cpu').scatter_(dim=2, index=connectivity,
                                                                     # src=coupling_cv[valid_ind].cpu().unsqueeze(
                                                                         # 0)).data.numpy()[0]
        # valid_phase_list = np.array([phase.cpu().data.numpy()[valid_ind, :] for phase in phase_list_cv])
        # show(displayer, valid_phase_list, valid_image, valid_mask, coupling_valid_show, save_path,
            # 'valid{}'.format(epoch))
    # save file s
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': op.state_dict(),
        'initial_phase': rand_phase,
        'connectivity': connectivity}, save_dir + '/model{}.pt'.format(epoch))
    np.save('coupling.npy', np.array(coupling_history))

    plt.plot(np.array(loss_history))
    plt.plot(np.array(loss_test_history))
    plt.title('Time Averaged Loss')
    plt.legend(['train', 'valid'])
    plt.savefig(save_dir + '/loss' + '.png')
    plt.close()
