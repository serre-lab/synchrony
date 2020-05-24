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
from losses import calc_sbd, calc_pq, just_kuramoto
from utils import *
import ipdb
import warnings
warnings.filterwarnings('ignore')
from NetProp import NetProp 
import glob
from torch.autograd import Variable
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
save_dir = os.path.join('/media/data_cifs/yuwei/osci_save/results/aneri/', args.exp_name)
model_dir = os.path.join('/media/data_cifs/yuwei/osci_save/models/aneri/', args.exp_name)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
#subprocess.call('rm -rf {}'.format(os.path.join(save_dir, '*')), shell=True)

if args.pretrained is True:
    model_files = glob.glob(os.path.join(model_dir, '*.pt'))
    latest_file = max(model_files, key=os.path.getctime)
    pretrained_model = torch.load(latest_file)

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

#####################
# Initialize random coupling
######################
# initialization
if torch.cuda.device_count() > 1 and args.device=='cuda':
    model = nn.DataParallel(nets.load_net(args, connectivity, args.num_global_control)).to(args.device)
    if args.pretrained:
        model.load_state_dict(pretrained_model['model_state_dict'])
        if args.phase_initialization == 'fixed':  model.osci.phase_init('fixed', fixed_phase=pretrained_model['initial_phase'])
    freq_params = model.module.osci.freq_net.parameters() if args.intrinsic_frequencies=='learned' else []
    criterion = nn.DataParallel(nets.kura_criterion(device=args.device)).to(args.device)
else:
    model = nets.load_net(args, connectivity, args.num_global_control).to(args.device)
    if args.pretrained:
        model.load_state_dict(pretrained_model['model_state_dict'])
        if args.phase_initialization == 'fixed':  model.osci.phase_init('fixed', fixed_phase=pretrained_model['initial_phase'])
    freq_params = model.osci.freq_net.parameters() if args.intrinsic_frequencies=='learned' else []
    criterion = nets.kura_criterion(device=args.device).to(args.device)
    print('network contains {} parameters'.format(nets.count_parameters(model))) # parameter number

loss_history = []
final_loss_history = []
loss_history_test = []
final_loss_history_test = []
#long_loss_history_test = []
epoch_history_test=[]
coupling_history = []

if args.intrinsic_frequencies == 'learned':
    params = [q1 for q1 in model.parameters()] + [q2 for q2 in freq_params]
else:
    params = list(model.parameters())
if args.classify is True:
    params += [q3 for q3 in classifier_params]
    
K = torch.tensor(np.random.uniform(0,0.01,size = [args.img_side**2,args.img_side**2])).to(args.device)
#K = torch.tensor(np.random.normal(loc = 0, scale = 0.01,size = [args.img_side**2,args.img_side**2]))
K = Variable(K,requires_grad = True)
print(K)
#params = K

# params = tuple(params)
# op = torch.optim.Adam({K}, lr=1.0)
# def fake_phase(K):
#     return (K[0]**2).sum()
# for epoch in range(10):
#     op.zero_grad()
#     phase = fake_phase(K)
#     loss=phase*2
#     loss.backward()
#     op.step()
#     print(K._grad)
    



op = torch.optim.Adam({K}, lr=args.learning_rate)

######################
# training pipeline
counter = 0
PL_train = []
PL_val = []
clustering_train = []
clustering_val = []
#omega = 2*np.pi * torch.normal(mean = 0, std = 10, size = (args.batch_size,args.img_side**2))
omega = torch.distributions.Uniform(0,10).rsample(sample_shape = (args.batch_size,args.img_side**2))

for epoch in range(args.train_epochs):
    print('Epoch: {}'.format(epoch))
    l=0
    fl=0
    cont_epoch = True
    PL_epoch = []
    clustering_epoch = []
    model.train()
    
    #omega = 2*np.pi * torch.normal(mean = 0.0, std = 10.0, size = (args.batch_size,args.img_side**2))

    for step in range(2):
        #batch = torch.tensor(np.random.uniform(-1,1, size=[args.batch_size,args.img_side**2,args.img_side**2])).to(args.device).float()
        #batch = torch.tensor(K).to(args.device).float().repeat(args.batch_size,1,1)
        batch = K
        op.zero_grad()
        phase_list_train, coupling_train, omega_train = model(batch.unsqueeze(0), omega)

#         if args.batch_size == 1:
#             phase_list_train, coupling_train, omega_train = model(batch.unsqueeze(0), omega)
#         else:        
#             phase_list_train, coupling_train, omega_train = model(batch, omega)
        tavg_loss, final_loss = criterion(phase_list_train, args.record_steps)
        if final_loss<0.5:
            import pdb; pdb.set_trace()
            tavg_loss, final_loss = criterion(phase_list_train, args.record_steps)
        #tavg_loss = torch.sum(phase_list_train)
        #tavg_loss.backward()
        tavg_loss.backward()
        op.step()
        #print(K._grad)

        
        l+=tavg_loss.data.cpu().numpy()

        fl+= final_loss.data.cpu().numpy()
        torch.save(K, save_dir+'/K_epoch{}.pt'.format(epoch))
        torch.save(phase_list_train[-1], save_dir+'/phase_epoch{}.pt'.format(epoch))

    
            
    if step > 0:
        loss_history.append(l / (step+1))
        final_loss_history.append(fl / (step+1))
   
     # Testing 
#     if epoch  % args.eval_interval == 0:
#         l=0
#         fl=0
#         ll=0
#         PL_epoch = []
#         clustering_epoch = []

#         model.eval()
#         with torch.no_grad():
#             for step in range(10):
#                 # cross-validation
#                 batch = torch.tensor().to(args.device).float()

#                 phase_list_test, coupling_test, omega_test = model(batch.unsqueeze(0))
                
               
#                 tavg_loss_test, final_loss_test = criterion(phase_list_test[-1*args.record_steps:], mask, args.transform, valid=True, targets=labels)
#                 if coupling_test is not None:
#                     tavg_loss_test += args.sparsity_weight * torch.abs(coupling_test).mean()
#                 if omega_test is not None:
#                     tavg_loss_test += args.sparsity_weight * torch.abs(omega_test).mean()
#                 l+=tavg_loss_test.data.cpu().numpy()
#                 fl+= final_loss_test.data.cpu().numpy()
#                 #ll+= long_loss_test.data.cpu().numpy()
                
#                 if step % 600 == 0 and args.graph_stats:
#                     NP_initialized = False
#                     if args.one_image:
#                         phase_list_test, coupling_test, omega_test = model(ex_image.unsqueeze(0).unsqueeze(0).repeat(args.batch_size,1,1,1))
#                         connectivity = ex_connectivity
#                         coupling_test = coupling_test[0:1,:]
#                         np.save(save_dir+'/coupling_test_epoch'+str(epoch)+'step'+str(step),coupling_test.cpu().detach().numpy())
                    
#                     if args.multi_image:
#                         phase_list_train, coupling_train, omega_train = model(batch.unsqueeze(1))
#                         connectivity = ex_connectivity
#                         np.save(save_dir+'/multi_coupling_test_epoch'+str(epoch)+'step'+str(step),coupling_train.cpu().detach().numpy())

#                     if args.cluster == True:
#                         if NP_initialized ==False:
#                             NP = NetProp(coupling_test,connectivity,args.num_global_control>0)
#                             NP_initialized = True
#                         clustering_epoch.append(NP.cluster_coefficient())
#                     if args.laplacian == True:
#                         if NP_initialized ==False:
#                             NP = NetProp(coupling_test,connectivity,args.num_global_control>0)
#                             NP_initialized = True
#                         NP.plot_laplacian(save_dir,epoch,step,'val',args.num_global_control)
                    
#                     if np.logical_and(cont_epoch==True,args.path_length == True):
#                         if NP_initialized==False:
#                             NP = NetProp(coupling_test,connectivity,args.num_global_control>0)
#                             NP_initialized = True
#                         this_path_length = NP.path_length()
#                         if this_path_length is None:
#                             PL_epoch = []
#                             cont_epoch = False
#                         else:
#                             PL_epoch.append(this_path_length)
            
#         if args.cluster == True and args.graph_stats:
#             clustering_val.append(np.mean(np.array(clustering_epoch))) 
#         if args.path_length == True and cont_epoch == True and args.graph_stats == True:
#             PL_val.append(np.mean(np.array(PL_epoch)))
#         elif args.path_length==True and args.graph_stats: 
#             PL_val.append(-1)
#         loss_history_test.append(l / (step+1))
#         final_loss_history_test.append(fl / (step + 1))

#         epoch_history_test.append(epoch)

    # save file s

    if not 'noKura' in args.model_name:
        save_phase_init = args.phase_initialization if args.phase_initialization != 'fixed' else model.osci.current_phase
    else:
        save_phase_init = None
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': op.state_dict(),
        'initial_phase': save_phase_init,
        'connectivity': connectivity}, model_dir + '/model.pt')

    plt.plot(loss_history)
    plt.plot(epoch_history_test, loss_history_test)
    plt.title('Time Averaged Loss')
    plt.legend(['train', 'valid'])
    plt.savefig(save_dir + '/loss' + '.png')
    plt.close()

    plt.plot(final_loss_history)
    plt.plot(epoch_history_test, final_loss_history_test)
    plt.title('Time Averaged Loss')
    plt.legend(['final_train', 'final_valid'])
    plt.savefig(save_dir + '/final_loss' + '.png')
    plt.close()


    np.save(os.path.join(save_dir, 'train_loss.npy'), np.array(loss_history))
    np.save(os.path.join(save_dir, 'valid_loss.npy'), np.array(loss_history_test))
    np.save(os.path.join(save_dir, 'valid_epoch.npy'), np.array(epoch_history_test))
    np.save(os.path.join(save_dir, 'train_final_loss.npy'), np.array(final_loss_history))
    np.save(os.path.join(save_dir, 'valid_final_loss.npy'), np.array(final_loss_history_test))

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

print(K)


def load_files(directory):
    files = os.listdir(directory)
    phase_files = []
    phases = []
    for f in files:
        if 'phase' in f:
            phase_files.append(f)
            phases.append(torch.load(directory+f).cpu().detach().numpy())
    return phases

def plot_phase_line(directory,num_osc):
    phases = load_files(directory)
    osc = {}
    for i in range(num_osc):
        this_osc = []
        for j in range(len(phases)):
            this_osc.append(phases[j][0][i])
        osc[i] = this_osc
    for i in osc:
        plt.plot(osc[i])
    plt.xlabel('Epochs')
    plt.yticks([0,0.5*np.pi,np.pi,1.5*np.pi,2*np.pi])
    plt.ylabel('Phase')
    plt.savefig(directory+'PhaseAcrossTraining.png')
def load_files_coupling(directory):
    files = os.listdir(directory)
    K_files = []
    K = []
    for f in files:
        if 'K' in f:
            K_files.append(f)
            K.append(torch.load(directory+f).cpu().detach().numpy())
    return K
#will plot the phases and make gif
def plot_coupling(directory, num_osc,num_vis = 10):
    couplings = load_files_coupling(directory)

    im_side = int(num_osc**0.5)
    update = int(len(couplings)/num_vis)
    p = []
    for i in range(num_vis):
        p.append(update*i)
    p.append(len(couplings)-1)
    for i in range(len(p)):
    #for i in range(5):
        plt.subplot(2,(num_vis+2)/2,i+1)
        coupling = couplings[p[i]]
        fig = plt.imshow(coupling)
        #plt.plot(phases[p[i]][0]+p[i])
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title(str(p[i]))
    plt.subplot(2,(num_vis+2)/2,num_vis)
    #fig.colorbar()
    plt.suptitle('Coupling Across Training Epochs') 
    plt.savefig(directory+'CouplingAcrossTraining.png')
print('Plotting Phase Course')
plot_phase_line(save_dir+'/',args.img_side**2)

print('Plotting Coupling Course')
plot_coupling(save_dir+'/', args.img_side**2)
    
