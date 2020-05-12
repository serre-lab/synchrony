import os, subprocess
#import nest_asyncio
#nest_asyncio.apply()
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
from distutils.util import strtobool
from losses import calc_sbd, calc_pq
from utils import *
import ipdb
import warnings
warnings.filterwarnings('ignore')
from NetProp import NetProp 
import glob
"""
DataParallel
kuramoto and loss_func_ex are integrated into an nn.Module to operate parallel calculation
"""
parser = argparse.ArgumentParser()

# Cosmetic parameters
parser.add_argument('--data_name', type=str, default='composite_textures_mini')
parser.add_argument('--exp_name', type=str, default='four_texture_segment')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--interactive', type=lambda x:bool(strtobool(x)), default=False)
parser.add_argument('--pretrained', type=lambda x:bool(strtobool(x)), default=False)
parser.add_argument('--show_every', type=int,default=50)
parser.add_argument('--eval_interval', type=int,default=1)
parser.add_argument('--verbosity', type=int,default=1)
# Model parameters
parser.add_argument('--model_name', type=str, default='simple_conv')
parser.add_argument('--in_channels', type=int, default=1)
parser.add_argument('--start_filts', type=int, default=32)
parser.add_argument('--depth', type=int, default=2)
parser.add_argument('--out_channels', type=int, default=32)
parser.add_argument('--split', type=int, default=4)
parser.add_argument('--kernel_size', type=str, default=5)
parser.add_argument('--dropout_p', type=float, default=0.0)
parser.add_argument('--activation', type=str, default='tanh')
parser.add_argument('--num_cn', type=int, default=8)
parser.add_argument('--num_global_control', type=int, default=0)
parser.add_argument('--p_rewire', type=float, default=0.0)
parser.add_argument('--rf_type', type=str, default='arange')
parser.add_argument('--phase_initialization', type=str, default='random')
parser.add_argument('--intrinsic_frequencies', type=str, default='zero')
parser.add_argument('--update_rate', type=float, default=1.6)
parser.add_argument('--sw', type=lambda x:bool(strtobool(x)),default=False)
parser.add_argument('--time_steps', type=int, default=8)
parser.add_argument('--record_steps', type=int, default=8)
parser.add_argument('--walk_step', type=float, default=.1)

#graph stats
parser.add_argument('--graph_stats', type=lambda x:bool(strtobool(x)), default=False)
parser.add_argument('--path_length', type=lambda x:bool(strtobool(x)), default=False)
parser.add_argument('--cluster',  type=lambda x:bool(strtobool(x)), default = False)
parser.add_argument('--laplacian', type=lambda x:bool(strtobool(x)), default = False)
parser.add_argument('--glob_order_parameter', type=lambda x:bool(strtobool(x)), default = False)
parser.add_argument('--bassett', type=lambda x:bool(strtobool(x)), default = False)
parser.add_argument('--one_image', type=lambda x:bool(strtobool(x)), default = False)
parser.add_argument('--multi_image', type=lambda x:bool(strtobool(x)), default = False )
# Data parameters
parser.add_argument('--img_side', type=int, default=32)
parser.add_argument('--segments', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=32)

# Learning parameters
parser.add_argument('--train_epochs', type=int, default=200)
parser.add_argument('--time_weight', type=int, default=0)
parser.add_argument('--anneal', type=float, default=0)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--sparsity_weight', type=float, default=1e-5)
parser.add_argument('--clustering_algorithm',type=str,default='km')
# loss parameters
parser.add_argument('--transform', type=str, default='linear')
parser.add_argument('--classify',type=lambda x:bool(strtobool(x)), default=False)
parser.add_argument('--recurrent_classifier',type=lambda x:bool(strtobool(x)), default=False)
parser.add_argument('--segmentation_metric', type=str, default='pq')

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
    num_devices = torch.cuda.device_count()
    print("Assigned {} GPUs".format(num_devices))
else:
    num_devices=1
 ######################
# parameters
######################
# path
load_dir = os.path.join('/media/data_cifs/yuwei/osci_save/data', args.data_name, str(args.segments))
save_dir = os.path.join('/media/data_cifs/yuwei/osci_save/results', args.exp_name)
model_dir = os.path.join('/media/data_cifs/yuwei/osci_save/models', args.exp_name)
train_path = load_dir + '/training'
test_path = load_dir + '/test'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
#subprocess.call('rm -rf {}'.format(os.path.join(save_dir, '*')), shell=True)

if args.pretrained is True:
    model_files = glob.glob(os.path.join(model_dir, '*.pt'))
    latest_file = max(model_files, key=os.path.getctime)
    pretrained_model = torch.load(latest_file)
		
######################
# Load data
# testing set
testing_set = datasets.DatasetFolder(test_path, np.load, extensions=('npy',))
testing_loader = DataLoader(testing_set, batch_size=args.batch_size, shuffle=True,
	drop_last=True)

#####################
# Connectivity
if args.pretrained is False:
    local_connectivity, global_connectivity = \
        generate_connectivity(args.num_cn, args.img_side, sw=args.sw, 
                          num_global_control=args.num_global_control,
                          p_rewire=args.p_rewire, rf_type=args.rf_type)

    local_connectivity = torch.tensor(local_connectivity).long()

    if global_connectivity is not None:
        global_connectivity = torch.tensor(global_connectivity).long()
        connectivity = [local_connectivity, global_connectivity]
    else:
        connectivity = local_connectivity
else:
    connectivity = pretrained_model['connectivity']
######################
# initialization
if torch.cuda.device_count() > 1 and args.device=='cuda':
    model = nn.DataParallel(nets.load_net(args, connectivity, args.num_global_control)).to(args.device)
    if args.pretrained:
        model.load_state_dict(pretrained_model['model_state_dict'])
        if args.phase_initialization == 'fixed':  model.osci.phase_init('fixed', fixed_phase=pretrained_model['initial_phase'])
    freq_params = model.module.osci.freq_net.parameters() if args.intrinsic_frequencies=='learned' else []
    criterion = nn.DataParallel(nets.criterion(args.time_weight, args.img_side**2, classify=args.classify, recurrent_classifier=args.recurrent_classifier, device=args.device)).to(args.device)
    classifier_params = criterion.module.classifier.parameters() if args.classify is True else []
else:
    model = nets.load_net(args, connectivity, args.num_global_control).to(args.device)
    if args.pretrained:
        model.load_state_dict(pretrained_model['model_state_dict'])
        if args.phase_initialization == 'fixed' and not 'noKura' in args.model_name:  model.osci.phase_init('fixed', fixed_phase=pretrained_model['initial_phase'])
    freq_params = model.osci.freq_net.parameters() if args.intrinsic_frequencies=='learned' else []
    criterion = nets.criterion(args.time_weight, args.img_side**2, classify=args.classify, recurrent_classifier=args.recurrent_classifier, device=args.device).to(args.device)
    classifier_params = criterion.classifier.parameters() if args.classify is True else []
    print('network contains {} parameters'.format(nets.count_parameters(model))) # parameter number

displayer = disp.displayer(args.segments, interactive=args.interactive)
if args.intrinsic_frequencies == 'learned':
    params = [q1 for q1 in model.parameters()] + [q2 for q2 in freq_params]
else:
    params = list(model.parameters())
if args.classify is True:
    params += [q3 for q3 in classifier_params]

params = tuple(params)

if args.segmentation_metric == 'sbd':
    accuracy = calc_sbd
elif args.segmentation_metric == 'pq':
    accuracy = calc_pq
else:
    raise NotImplemented("The chosen segmentation metric is not recognized.")

######################
# training pipeline
#norm = np.sum(np.arange(1, args.time_steps + 1) ** 2)
# Testing 
l=0
fl=0
acc = 0
ns = 0

model.eval()
with torch.no_grad():
    for step, (test_data, _) in tqdm(enumerate(testing_loader)):
        # cross-validation
        batch = test_data[:,  0, ...].float().to(args.device)
        mask = test_data[:, 1:, ...].reshape(-1, args.segments, args.img_side * args.img_side).float().to(args.device)

        num_segments = ((mask.sum(2) > 0)*1).sum(1)
        label_inds = (num_segments == args.segments - 1)*1
        labels = torch.zeros((args.batch_size,2)).to(args.device).scatter_(1,label_inds.unsqueeze(1),1.0)

        phase_list_test, coupling_test, omega_test = model(batch.unsqueeze(1))
                
        last_phase = phase_list_test[-1].cpu().detach().numpy()
        colored_mask = (np.expand_dims(np.expand_dims(np.arange(args.segments), axis=0), axis=-1) * mask.cpu().detach().numpy()).sum(1)

        clustered_batch = []
        predicted_segs_batch = []
        for idx, (sample_phase, sample_mask) in enumerate(zip(last_phase, colored_mask)):
            clustered_img, n_clusters = clustering(sample_phase, algorithm = args.clustering_algorithm, max_clusters=args.segments)
            clustered_batch.append(clustered_img)
            predicted_segs_batch.append(n_clusters)
            acc += accuracy(clustered_batch[idx]+1, sample_mask+1)
        ns += (1.*(np.array(predicted_segs_batch) == num_segments.cpu().numpy())).mean()
        print('Running accuracy: {}'.format(acc / ((step+1) * args.batch_size)))
        print('Running # seg accuracy: {}'.format(ns / ((step+1))))

        tavg_loss_test, final_loss_test = criterion(phase_list_test[-1*args.record_steps:], mask, args.transform, valid=True, targets=labels)
        if coupling_test is not None:
            tavg_loss_test += args.sparsity_weight * torch.abs(coupling_test).mean()
        if omega_test is not None:
           tavg_loss_test += args.sparsity_weight * torch.abs(omega_test).mean()
        l+=tavg_loss_test.data.cpu().numpy()
        fl+= final_loss_test.data.cpu().numpy()
        if step % args.show_every == 0 and args.show_every > 0:
            # visualize validation and save
            # validation example, save its coupling matrix
            display(displayer, phase_list_test, batch, mask, clustered_batch, coupling_test, omega_test, args.img_side, args.segments, save_dir, 
            'test{}'.format(step), args.rf_type)

print('Test loss: {}'.format(l / (step + 1)))
print('Test final loss: {}'.format(fl / (step + 1)))
print('Test accuracy: {}'.format(acc / ((step+1) * args.batch_size)))

np.save(os.path.join(save_dir, 'test_loss.npy'), l / (step + 1))
np.save(os.path.join(save_dir, 'test_final_loss.npy'), fl / (step + 1))
np.save(os.path.join(save_dir, 'test_acc.npy'), acc / ((step + 1) * args.batch_size))
