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
from losses import calc_sbd
from utils import *
import ipdb
import warnings
warnings.filterwarnings('ignore')
from NetProp import NetProp 
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
parser.add_argument('--show_every', type=int,default=50)
parser.add_argument('--eval_interval', type=int,default=1)
# Model parameters
parser.add_argument('--model_name', type=str, default='simple_conv')
parser.add_argument('--in_channels', type=int, default=1)
parser.add_argument('--start_filts', type=int, default=32)
parser.add_argument('--depth', type=int, default=2)
parser.add_argument('--out_channels', type=int, default=32)
parser.add_argument('--split', type=int, default=4)
parser.add_argument('--kernel_size', type=str, default='3,3')
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
parser.add_argument('--segmentation', type=lambda x:bool(strtobool(x)), default=True)
parser.add_argument('--pretrained', type=lambda x:bool(strtobool(x)), default=False)

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
parser.add_argument('--batch_size', type=int, default=16)

# Learning parameters
parser.add_argument('--train_epochs', type=int, default=200)
parser.add_argument('--time_weight', type=int, default=2)
parser.add_argument('--anneal', type=float, default=0)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--sparsity_weight', type=float, default=1e-5)

# loss parameters
parser.add_argument('--transform', type=str, default='linear')
parser.add_argument('--classify',type=lambda x:bool(strtobool(x)), default=False)
parser.add_argument('--recurrent_classifier',type=lambda x:bool(strtobool(x)), default=False)

args = parser.parse_args()
args.kernel_size = [int(k) for k in args.kernel_size.split(',')]
args.exp_name = '_'.join(['SR_polyominoes_8_free_rotation_test_2_gpus',str(args.model_name),str(args.classify),str(args.pretrained),str(args.time_steps),str(args.update_rate),str(args.record_steps),str(args.phase_initialization),str(args.intrinsic_frequencies),str(args.sparsity_weight)])

if args.interactive:
    import matplotlib.pyplot as plt
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
######################
# device
if args.device is not 'cpu':
    num_devices = torch.cuda.device_count()
    print("Assigned {} GPUs".format(num_devices))
else:
    num_devices=1
 ######################
# parameters
num_test = 1000

######################
def is_valid(path):
    if path[:3]=='img' and path[-3:]=='npy':
        return True
    else:
        return False
# path
#load_dir = os.path.join('/media/data_cifs/yuwei/osci_save/data', args.data_name)
load_dir = os.path.join('/media/data/mchalvid/osci_save_v3_masks/data', args.data_name)
save_dir = os.path.join('/media/data/mchalvid/osci_save_v3_masks/results', args.exp_name)
model_dir = os.path.join('/media/data/mchalvid/osci_save_v3_masks/models', args.exp_name)
train_path = load_dir + '/train'
test_path = load_dir + '/test'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
subprocess.call('rm -rf {}'.format(os.path.join(save_dir, '*')), shell=True)
subprocess.call('rm -rf {}'.format(os.path.join(model_dir, '*')), shell=True)

with open(save_dir + '/commandline_args.txt', 'w') as f:
    f.write('\n'.join(sorted([str(k)+' = '+ str(v) for k,v in vars(args).items()])))
		
######################
# Load data
#training set
training_set = datasets.DatasetFolder(train_path, np.load, extensions=('npy',))
training_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
# testing set
testing_set = datasets.DatasetFolder(test_path, np.load, extensions=('npy',))
testing_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True,
	drop_last=True)

#####################
# Connectivity
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

###############
if args.pretrained:
    exp = torch.load(os.path.join('/media/data/mchalvid/osci_save_v3_masks/models','SR_polyominoes_seg_50ts_fixed_init','model98.pt'))
    connectivity = exp['connectivity']

######################
# initialization
if torch.cuda.device_count() > 1 and args.device=='cuda':
    model = nn.DataParallel(nets.load_net(args, connectivity, args.num_global_control)).to(args.device)
    if not args.intrinsic_frequencies=='zero':
        model.module.osci.freq_net = model.module.osci.freq_net.to(args.device)
    freq_params = model.module.osci.freq_net.parameters() if args.intrinsic_frequencies=='learned' else []
    criterion = nn.DataParallel(nets.criterion(args.time_weight, args.img_side**2, classify=args.classify, recurrent_classifier=args.recurrent_classifier, device=args.device)).to(args.device)
    classifier_params = criterion.module.classifier.parameters() if args.classify is True else []
else:
    model = nets.load_net(args, connectivity, args.num_global_control)
    if args.pretrained:

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in exp['model_state_dict'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        model_osci_freq = model.osci.freq_net.state_dict()
        pretrained_dict = {k: v for k, v in exp['model_state_dict'].items() if k in model_osci_freq}
        model_osci_freq.update(pretrained_dict)
        model.osci.freq_net.load_state_dict(model_osci_freq)

    model = model.to(args.device)
    if not args.intrinsic_frequencies=='zero':
        model.osci.freq_net = model.osci.freq_net.to(args.device)
    freq_params = model.osci.freq_net.parameters() if args.intrinsic_frequencies=='learned' else []
    criterion = nets.criterion(args.time_weight, args.img_side**2, classify=args.classify, recurrent_classifier=args.recurrent_classifier, device=args.device).to(args.device)
    classifier_params = criterion.classifier.parameters() if args.classify is True else []
    print('network contains {} parameters'.format(nets.count_parameters(model))) # parameter number

loss_history = []
accuracy_history = []
accuracy_history_test = []
sbd_history=[]
loss_history_test = []
sbd_history_test=[]
epoch_history_test=[]
coupling_history = []

displayer = disp.displayer(args.segments, interactive=args.interactive)
if not args.pretrained:
    if args.intrinsic_frequencies == 'learned':
        params = [q1 for q1 in model.parameters()] + [q2 for q2 in freq_params]
    else:
        params = list(model.parameters())
else:
    params = []
    if args.intrinsic_frequencies == 'learned':
        for q1 in model.parameters():
            q1.requires_grad = False
        for q2 in model.osci.freq_net.parameters():
            q2.requires_grad = False

if args.classify is True:
    params += [q3 for q3 in classifier_params]

params = tuple(params)

op = torch.optim.Adam(params, lr=args.learning_rate)

######################
# training pipeline
norm = np.sum(np.arange(1, args.record_steps + 1) ** 2)
counter = 0
PL_train = []
PL_val = []
clustering_train = []
clustering_val = []
for epoch in range(args.train_epochs):
    print('Epoch: {}'.format(epoch))

    l=0
    correct=0
    sbd = 0
    cont_epoch = True
    PL_epoch = []
    clustering_epoch = []

    for step, (train_data, _) in tqdm(enumerate(training_loader)):
        batch = torch.tensor(train_data[:, 0, ...]).to(args.device).float()
        if args.segmentation  == True:
            mask = torch.tensor(train_data[:, 1:-1, ...]).reshape(args.batch_size, -1, args.img_side * args.img_side).to(args.device).float()
            label_nb = ((mask.sum(2) > 0) * 1).sum(1)
            #label_inds = (((mask.sum(2) > 0)*1).sum(1) == args.segments - 1)*1
            labels = one_hot_label(train_data[:, -1, ...]).to(args.device).float()
        else:
            mask=None
            labels = one_hot_label(train_data[:, -1, ...]).to(args.device).float()

        op.zero_grad()
        phase_list_train, coupling_train, omega_train = model(batch.unsqueeze(1))
        last_phase = phase_list_train[-1].cpu().detach().numpy()

        if args.segmentation == True:
            colored_mask = (np.expand_dims(np.expand_dims(np.arange(args.segments), axis=0), axis=-1) * mask.cpu().detach().numpy()).sum(1)
        else:
            colored_mask = None
        
        if args.one_image:
            if np.logical_and(epoch==0,step==0):
                ex_image = torch.tensor(train_data[0, 0, ...]).to(args.device).float()
                #ex_mask = torch.tensor(train_data[0, 1:, ...]).reshape(-1, args.segments, args.img_side * args.img_side).to(args.device).float()
                if args.num_global_control > 0:
                    ex_connectivity = [connectivity[0],connectivity[1]]
                else:
                    ex_connectivity = connectivity

        if args.multi_image:
            if np.logical_and(epoch==0,step==0):
                ex_batch = torch.tensor(train_data[:, 0, ...]).to(args.device).float()
                if args.num_global_control > 0:
                    ex_connectivity = [connectivity[0],connectivity[1]]
                else:
                    ex_connectivity = connectivity

        tavg_loss, correct_batch = criterion(phase_list_train[-1*args.record_steps:], mask, args.transform, valid=False ,targets=labels)
        tavg_loss = tavg_loss.mean() / norm

        if coupling_train is not None:
            tavg_loss += args.sparsity_weight * torch.abs(coupling_train).mean()
        if omega_train is not None:
            tavg_loss += args.sparsity_weight * torch.abs(omega_train).mean()
        l=tavg_loss.data.cpu().numpy()
        print('loss : ', l)
        tavg_loss.backward()
        op.step()

        #compute accuracy
        if args.classify == True:
            correct += correct_batch

        # visualize training
        clustered_batch = []
        if step % args.show_every == 0:
            if args.segmentation == True:
                for idx, (sample_phase, sample_mask) in enumerate(zip(last_phase, colored_mask)):
                    clustered_batch.append(clustering(sample_phase, n_clusters=args.segments))
                    sbd += calc_sbd(clustered_batch[idx]+1, sample_mask+1)
                display(displayer, phase_list_train, batch, mask, clustered_batch, coupling_train, omega_train, args.img_side, args.segments, save_dir,
                    'train{}_{}'.format(epoch,step), args.rf_type, args.segmentation)
            else:
                for idx, sample_phase in enumerate(last_phase):
                    clustered_batch.append(clustering(sample_phase, n_clusters=args.segments))
                    #sbd += calc_sbd(clustered_batch[idx]+1, sample_mask+1)
                display(displayer, phase_list_train, batch, mask, clustered_batch, coupling_train, omega_train, args.img_side, args.segments, save_dir,
                    'train{}_{}'.format(epoch,step), args.rf_type, args.segmentation)


        if step % 600 == 0:
            NP_initialized = False
            if args.one_image:
                #import pdb;pdb.set_trace()
                phase_list_train, coupling_train, omega_train = model(ex_image.unsqueeze(0).unsqueeze(0).repeat(args.batch_size,1,1,1)) #was getting error with 
                #one not matching the original batch size
                connectivity = ex_connectivity 
                coupling_train = coupling_train[0:1,:]
                np.save(save_dir+'/coupling_train_epoch'+str(epoch)+'step'+str(step),coupling_train.cpu().detach().numpy())
            
            if args.multi_image:
                phase_list_train, coupling_train, omega_train = model(batch.unsqueeze(1))
                connectivity = ex_connectivity 
                np.save(save_dir+'/multi_coupling_train_epoch'+str(epoch)+'step'+str(step),coupling_train.cpu().detach().numpy())

                
            
            if args.cluster == True:
                #pass in whole coupling train and batch_connectivity and then deal with it in NP?
                #import pdb;pdb.set_trace()
                if NP_initialized==False:
                    NP = NetProp(coupling_train,connectivity, args.num_global_control>0)
                    NP_initialized = True
                
                clustering_epoch.append(NP.cluster_coefficient())
            if args.laplacian == True:
                if NP_initialized==False:
                    NP = NetProp(coupling_train,connectivity, args.num_global_control>0)
                    NP_initialized = True
                NP.plot_laplacian(save_dir,epoch,step,'train',args.num_global_control)
                
            if np.logical_and(cont_epoch==True,args.path_length == True):
                if NP_initialized==False:
                    NP = NetProp(coupling_train,connectivity, args.num_global_control>0)
                    NP_initialized = True
                this_path_length = NP.path_length()
                if this_path_length is None:
                    PL_epoch = []
                    cont_epoch = False
                else:
                    PL_epoch.append(this_path_length)
                
    if args.cluster == True:
        clustering_train.append(np.mean(np.array(clustering_epoch))) 
        
    if np.logical_and(args.path_length == True, cont_epoch == True):
        PL_train.append(np.mean(np.array(PL_epoch)))

    elif args.path_length==True:
        PL_train.append(-1)
        sbd_history.append(sbd / ((1+ (step // args.show_every)) * args.batch_size))

    loss_history.append(l)

    if args.classify == True:
        print('accuracy : {}'.format(100*correct/(args.batch_size*step)))
        accuracy_history.append(100*correct/(args.batch_size*step))
    
    if (epoch+1) % args.eval_interval == 0:
        torch.cuda.empty_cache()
        l=0
        correct=0
        sbd = 0
        PL_epoch = []
        clustering_epoch = []
        # Testing
        with torch.no_grad():
            for step, (test_data, _) in tqdm(enumerate(testing_loader)):
                # cross-validation
                batch = torch.tensor(test_data[:, 0, ...]).to(args.device).float()
                if args.segmentation == True:
                    mask = torch.tensor(test_data[:, 1:-1, ...]).reshape(args.batch_size, -1, args.img_side * args.img_side).to(args.device).float()
                    label_nb = ((mask.sum(2) > 0) * 1).sum(1)
                    #label_inds = (((mask.sum(2) > 0) * 1).sum(1) == args.segments - 1) * 1
                    labels = one_hot_label(train_data[:, -1, ...]).to(args.device).float()
                else:
                    mask = None
                    labels = one_hot_label(test_data[:, -1, ...]).to(args.device).float()


                phase_list_test, coupling_test, omega_test = model(batch.unsqueeze(1))
                last_phase = phase_list_test[-1].cpu().detach().numpy()

                if args.segmentation == True:
                    if mask is not None:
                        colored_mask = (np.expand_dims(np.expand_dims(np.arange(args.segments), axis=0),
                                                       axis=-1) * mask.cpu().detach().numpy()).sum(1)
                        clustered_batch = []
                        for idx, (sample_phase, sample_mask) in enumerate(zip(last_phase, colored_mask)):
                            clustered_batch.append(clustering(sample_phase, n_clusters=args.segments))
                            sbd += calc_sbd(clustered_batch[idx] + 1, sample_mask + 1)


                tavg_loss_test, correct_batch = criterion(phase_list_test[-1*args.record_steps:], mask, args.transform, valid=True, targets=labels)
                tavg_loss_test = tavg_loss_test.mean() / norm

                if coupling_test is not None:
                    tavg_loss_test += args.sparsity_weight * torch.abs(coupling_test).mean()
                if omega_test is not None:
                    tavg_loss_test += args.sparsity_weight * torch.abs(omega_test).mean()
                l=tavg_loss_test.data.cpu().numpy()

                # compute accuracy
                if args.classify == True:
                    correct += correct_batch
                
                if step % args.show_every == 0:
                    # visualize validation and save
                    # validation example, save its coupling matrix
                    display(displayer, phase_list_test, batch, mask, clustered_batch, coupling_test, omega_test, args.img_side, args.segments, save_dir, 
                    'test{}_{}'.format(epoch, step), args.rf_type, args.segmentation)
                #if step*args.batch_size > num_test:
                #    break
                if step % 600 == 0:
                    NP_initialized = False
                    if args.one_image:
                        phase_list_test, coupling_test, omega_test = model(ex_image.unsqueeze(0).unsqueeze(0).repeat(args.batch_size,1,1,1))
                        connectivity = ex_connectivity
                        coupling_test = coupling_test[0:1,:]
                        np.save(save_dir+'/coupling_test_epoch'+str(epoch)+'step'+str(step),coupling_test.cpu().detach().numpy())
                    
                    if args.multi_image:
                        phase_list_train, coupling_train, omega_train = model(batch.unsqueeze(1))
                        connectivity = ex_connectivity
                        np.save(save_dir+'/multi_coupling_test_epoch'+str(epoch)+'step'+str(step),coupling_train.cpu().detach().numpy())

                    if args.cluster == True:
                        if NP_initialized ==False:
                            NP = NetProp(coupling_test,connectivity,args.num_global_control>0)
                            NP_initialized = True
                        clustering_epoch.append(NP.cluster_coefficient())
                    if args.laplacian == True:
                        if NP_initialized ==False:
                            NP = NetProp(coupling_test,connectivity,args.num_global_control>0)
                            NP_initialized = True
                        NP.plot_laplacian(save_dir,epoch,step,'val',args.num_global_control)
                    
                    if np.logical_and(cont_epoch==True,args.path_length == True):
                        if NP_initialized==False:
                            NP = NetProp(coupling_test,connectivity,args.num_global_control>0)
                            NP_initialized = True
                        this_path_length = NP.path_length()
                        if this_path_length is None:
                            PL_epoch = []
                            cont_epoch = False
                        else:
                            PL_epoch.append(this_path_length)
                            
            
        if args.cluster == True:
            clustering_val.append(np.mean(np.array(clustering_epoch))) 
        if np.logical_and(args.path_length == True, cont_epoch == True):
            PL_val.append(np.mean(np.array(PL_epoch)))
        elif args.path_length==True: 
            PL_val.append(-1)
        loss_history_test.append(l)
        sbd_history_test.append(sbd / ((step+1) * args.batch_size))
        epoch_history_test.append(epoch)
        accuracy_history_test.append(100 * correct / (args.batch_size * step))

    # save file s
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': op.state_dict(),
        'initial_phase': args.phase_initialization,
        'connectivity': connectivity}, model_dir + '/model{}.pt'.format(epoch))

    plt.plot(loss_history)
    plt.plot(epoch_history_test, loss_history_test)
    plt.title('Time Averaged Loss')
    plt.legend(['train', 'valid'])
    plt.savefig(save_dir + '/loss' + '.png')
    plt.close()

    plt.plot(sbd_history)
    plt.plot(epoch_history_test, sbd_history_test)
    plt.title('Symmetric Best Dice')
    plt.legend(['train', 'valid'])
    plt.savefig(save_dir + '/sbd' + '.png')
    plt.close()

    plt.plot(accuracy_history)
    plt.plot(epoch_history_test, accuracy_history_test)
    plt.title('Accuracy')
    plt.legend(['train', 'valid'])
    plt.savefig(save_dir + '/accuracy' + '.png')
    plt.close()

    np.save(os.path.join(save_dir, 'train_accuracy.npy'), np.array(accuracy_history))
    np.save(os.path.join(save_dir, 'test_accuracy.npy'), np.array(accuracy_history_test))
    np.save(os.path.join(save_dir, 'train_loss.npy'), np.array(loss_history))
    np.save(os.path.join(save_dir, 'valid_loss.npy'), np.array(loss_history_test))
    np.save(os.path.join(save_dir, 'train_sbd.npy'), np.array(sbd_history))
    np.save(os.path.join(save_dir, 'valid_sbd.npy'), np.array(sbd_history_test))
    np.save(os.path.join(save_dir, 'valid_epoch.npy'), np.array(epoch_history_test))
   
    # ipdb.set_trace() 
    #sbd_diff = np.abs(np.diff(sbd_history_test[-5:])).mean()
    #if args.early_stopping and sbd_history_test[-1] > .98 or sbd_diff < .05:
    #    break

    if args.path_length == True:
        plt.plot(np.array(PL_train))
        plt.plot(np.array(PL_val))
        plt.title('Path Length')
        plt.legend(['Train', 'Validation'])
        plt.savefig(save_dir +'/pathlength.png')
        plt.close()
    if args.cluster == True:
        plt.plot(np.array(clustering_train))
        plt.plot(np.array(clustering_val))
        plt.title('Clustering Coefficient')
        plt.legend(['Train', 'Validation'])
        plt.savefig(save_dir +'/clusteringcoefficient.png')
        plt.close()
