import os, subprocess
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
import warnings
warnings.filterwarnings('ignore')
"""
DataParallel
kuramoto and loss_func_ex are integrated into an nn.Module to operate parallel calculation
"""
parser = argparse.ArgumentParser()

# Cosmetic parameters
parser.add_argument('--data_name', type=str, default='composite_textures_mini')
parser.add_argument('--exp_name', type=str, default='four_texture_segment')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--interactive', type=str, default=False)
parser.add_argument('--show_every', type=int,default=50)
# Model parameters
parser.add_argument('--model_name', type=str, default='simple_conv')
parser.add_argument('--in_channels', type=int, default=1)
parser.add_argument('--start_filts', type=int, default=32)
parser.add_argument('--depth', type=int, default=2)
parser.add_argument('--out_channels', type=int, default=32)
parser.add_argument('--split', type=int, default=4)
parser.add_argument('--kernel_size', type=str, default=5)
parser.add_argument('--num_cn', type=int, default=8)
parser.add_argument('--num_global_control', type=int, default=0)
parser.add_argument('--p_rewire', type=float, default=0.0)
parser.add_argument('--rp_field', type=str, default='arange')
parser.add_argument('--phase_initialization', type=str, default='random')
parser.add_argument('--intrinsic_frequencies', type=str, default='zero')
parser.add_argument('--update_rate', type=float, default=1.6)
parser.add_argument('--sw', type=bool, default=False)
parser.add_argument('--time_steps', type=int, default=8)
parser.add_argument('--record_steps', type=int, default=8)
parser.add_argument('--walk_step', type=float, default=.1)

# Data parameters
parser.add_argument('--img_side', type=int, default=32)
parser.add_argument('--segments', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=32)

# Learning parameters
parser.add_argument('--train_epochs', type=int, default=200)
parser.add_argument('--time_weight', type=int, default=2)
parser.add_argument('--anneal', type=float, default=0)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--sparsity_weight', type=float, default=1e-5)

# loss parameters
parser.add_argument('--transform', type=str, default='linear')

args = parser.parse_args()
args.kernel_size = [int(k) for k in args.kernel_size.split(',')]

if args.interactive:
    import matplotlib.pyplot as plt
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
    
######################
# device
if not args.device is not 'cpu':
    gpu_num = torch.cuda.device_count()
    print("Assigned {} GPUs".format(gpu_num))
 ######################
# parameters
num_test = 1000

######################
# path
load_dir = os.path.join('/media/data_cifs/yuwei/osci_save/data/', args.data_name, str(args.segments))
save_dir = os.path.join('/media/data_cifs/yuwei/osci_save/results/', args.exp_name)
model_dir = os.path.join('/media/data_cifs/yuwei/osci_save/models/', args.exp_name)
train_path = load_dir + '/train'
test_path = load_dir + '/test'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
subprocess.call('rm -rf {}'.format(os.path.join(save_dir, '*')), shell=True)
		
######################
# Load data
#training set
training_set = datasets.DatasetFolder(train_path, np.load, extensions=('npy',))
training_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
# testing set
testing_set = datasets.DatasetFolder(test_path, np.load, extensions=('npy',))
testing_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True,
	drop_last=True)

######################
# connectivity
#print('Generating fixed adjancency matrix.')
#if args.small_world:
#    connectivity = generate_connectivity(args.img_side, args.num_cn, args.critical_dist, 'small_world')
#else:
#    connectivity = generate_connectivity(args.img_side, args.num_cn, args.critical_dist, 'local')
#connectivity = torch.tensor(connectivity).long().unsqueeze(0).to('cpu')
#batch_connectivity = connectivity.repeat(args.batch_size, 1, 1).to(args.device)

connectivity, global_connectivity = \
    generate_connectivity(args.num_cn, args.img_side, sw=args.sw, 
                          num_global_control=args.num_global_control,
                          p_rewire=args.p_rewire, rp_field=args.rp_field)

connectivity = torch.tensor(connectivity).long().unsqueeze(0).to('cpu')
batch_connectivity = connectivity.repeat(args.batch_size, 1, 1).to(args.device)

if global_connectivity is not None:
    global_connectivity = torch.tensor(global_connectivity).long().unsqueeze(0).to('cpu')
    batch_gconnectivity = global_connectivity.repeat(args.batch_size, 1, 1).to(args.device)
    batch_connectivity = [batch_connectivity] + [batch_gconnectivity]
######################
# initialization
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(nets.load_net(args, batch_connectivity, args.num_global_control)).to(args.device)
    criterion = nn.DataParallel(nets.criterion(args.time_weight)).to(args.device)
else:
    model = nets.load_net(args, batch_connectivity, args.num_global_control).to(args.device)
    criterion = nets.criterion(args.time_weight).to(args.device)
    print('network contains {} parameters'.format(nets.count_parameters(model))) # parameter number
time.sleep(2)

loss_history = []
loss_history_test = []
coupling_history = []

displayer = disp.displayer(interactive=args.interactive)
if args.intrinsic_frequencies == 'learned':
    params = tuple([q1 for q1 in model.parameters()] + [q2 for q2 in model.osci.freq_net.parameters()])
else:
    params = model.parameters()
op = torch.optim.Adam(params, lr=args.learning_rate)

######################
# training pipeline
norm = np.sum(np.arange(1, args.time_steps + 1) ** 2)
counter = 0
for epoch in range(args.train_epochs):
    print('Epoch: {}'.format(epoch))

    l=0
    for step, (train_data, _) in tqdm(enumerate(training_loader)):
        batch = torch.tensor(train_data[:, 0, ...]).to(args.device).float()
        mask = torch.tensor(train_data[:, 1:, ...]).reshape(-1, args.segments, args.img_side * args.img_side).to(args.device).float()

        op.zero_grad()
        phase_list_train, coupling_train, omega_train = model(batch.unsqueeze(1))

        tavg_loss = criterion(phase_list_train, mask, args.transform, args.device)
        tavg_loss = tavg_loss.mean() / norm
        tavg_loss += args.sparsity_weight * torch.abs(coupling_train).mean()
        l+=tavg_loss.data.cpu().numpy()
        tavg_loss.backward()
        op.step()
       
        # visualize training
        if step % args.show_every == 0:
            #train_ind = np.random.randint(args.batch_size)
            #train_image = batch[train_ind].cpu().data.numpy()
            #train_mask = mask[train_ind].cpu().unsqueeze(0).data.numpy()
            #coupling_train_show = \
            #    torch.zeros(1, args.img_side ** 2, args.img_side ** 2).to('cpu').scatter_(dim=2, index=connectivity.cpu(),
            #                                        src=coupling_train[train_ind].unsqueeze(0).cpu()).data.numpy()[0]
            #train_phase_list = np.array([phase.cpu().data.numpy()[train_ind, :] for phase in phase_list_train])
            #omega_train_show = omega_train[train_ind,...].reshape(args.img_side, args.img_side).detach().cpu().numpy()

            display(displayer, phase_list_train, batch, mask, coupling_train, omega_train, args.img_side, args.segments, save_dir,
                'train{}_{}'.format(epoch,step), args.rp_field)
    loss_history.append(l / step)
    l=0
    for step, (test_data, _) in tqdm(enumerate(testing_loader)):
        # cross-validation
        batch = test_data[:,  0, ...].float().to(args.device)
        mask = test_data[:, 1:, ...].reshape(-1, args.segments, args.img_side * args.img_side).float().to(args.device)

        phase_list_test, coupling_test, omega_test = model(batch.unsqueeze(1))

        # phase.shape=(time, batch, N)
        # mask.shape=(time, batch, group, N)
        # return loss.shape=(time * batch
        tavg_loss_test = criterion(phase_list_test, mask, args.transform, args.device, True)
        tavg_loss_test = tavg_loss_test.mean() / norm
        tavg_loss_test += args.sparsity_weight * torch.abs(coupling_test).mean()
        l+=tavg_loss_test.data.cpu().numpy()
		
        if step % args.show_every == 0:
            # visualize validation and save
            # validation example, save its coupling matrix
            test_ind = np.random.randint(args.batch_size)
            test_image = batch[test_ind].cpu().data.numpy()
            test_mask = mask[test_ind].cpu().unsqueeze(0).data.numpy()
            coupling_test_show = \
                torch.zeros(1, args.img_side ** 2, args.img_side ** 2).to('cpu').scatter_(dim=2, index=connectivity.cpu(),
                                                                     src=coupling_test[test_ind].cpu().unsqueeze(
                                                                         0)).data.numpy()[0]
            omega_test_show = omega_test[test_ind,...].reshape(args.img_side, args.img_side).detach().cpu().numpy()
            test_phase_list = np.array([phase.cpu().data.numpy()[test_ind, :] for phase in phase_list_test])
            show(displayer, test_phase_list, test_image, test_mask, coupling_test_show, omega_test_show, save_dir, 'test{}_{}'.format(epoch,step), args.segments, args.img_side)
	if step*args.batch_size > num_test:
	    break
    loss_history_test.append(l /step)


    # save file s
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': op.state_dict(),
        'initial_phase': args.phase_initialization,
        'connectivity': connectivity}, model_dir + '/model{}.pt'.format(epoch))

    plt.plot(np.array(loss_history))
    plt.plot(np.array(loss_history_test))
    plt.title('Time Averaged Loss')
    plt.legend(['train', 'valid'])
    plt.savefig(save_dir + '/loss' + '.png')
    plt.close()
