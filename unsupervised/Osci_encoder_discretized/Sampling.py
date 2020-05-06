from __future__ import print_function
import argparse
import os
import torch
import torch.utils.data
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from Kura_module import Kuramoto as km
from Kura_module import ODEDynamic, ODEDynamic_linear
from utils import phase_evol, plot_evolution
import subprocess
from Kura_module import Osci_AE
from sklearn.mixture.gaussian_mixture import GaussianMixture
import matplotlib.pyplot as plt
from supervised.kuramoto.utils import generate_connectivity

os.environ["CUDA_VISIBLE_DEVICES"]='0'

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--epochs', type=int, default=10, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')


# Cosmetic parameters
parser.add_argument('--data_name', type=str, default='MNIST')
parser.add_argument('--exp_name', type=str, default='Osci_MNIST_linear')
parser.add_argument('--device', type=str, default='cuda')
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
parser.add_argument('--kernel_size', type=str, default=5)
parser.add_argument('--num_cn', type=int, default=16)
parser.add_argument('--num_global_control', type=int, default=0)
parser.add_argument('--p_rewire', type=float, default=0.0)
parser.add_argument('--rf_type', type=str, default='arange')
parser.add_argument('--phase_initialization', type=str, default='fixed')
parser.add_argument('--intrinsic_frequencies', type=str, default='zero')
parser.add_argument('--update_rate', type=float, default=1.6)
parser.add_argument('--sw', type=lambda x:bool(strtobool(x)),default=False)
parser.add_argument('--time_steps', type=int, default=8)
parser.add_argument('--record_steps', type=int, default=1)
parser.add_argument('--walk_step', type=float, default=.1)
parser.add_argument('--ode_train', type=lambda x:bool(strtobool(x)),default=False)
parser.add_argument('--n_osci', type=int, default=16)
parser.add_argument('--max_time', type=float, default=20)
parser.add_argument('--dropout_p', type=float, default=0.4)
parser.add_argument('--solver', type=str, default='euler')
parser.add_argument('--dynamic_type', type=str, default='linear')

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
parser.add_argument('--batch_size', type=int, default=64)

# Learning parameters
parser.add_argument('--time_weight', type=int, default=2)
parser.add_argument('--anneal', type=float, default=0)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--sparsity_weight', type=float, default=1e-5)

# loss parameters
parser.add_argument('--transform', type=str, default='linear')
parser.add_argument('--classify',type=lambda x:bool(strtobool(x)), default=False)
parser.add_argument('--recurrent_classifier',type=lambda x:bool(strtobool(x)), default=False)
parser.add_argument('--entropy_reg',type=lambda x:bool(strtobool(x)), default=False)


parser.add_argument('--path',type=str, default='/media/data_cifs/mchalvid/Project_synchrony/MNIST')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


load_dir = os.path.join(args.path, args.data_name)
save_dir = os.path.join(args.path,'results', args.exp_name)
model_dir = os.path.join(args.path,'models', args.exp_name)


args.load_dir = load_dir
args.save_dir = save_dir
args.model_dir = model_dir



model_logs = torch.load('/media/data_cifs/mchalvid/Project_synchrony/MNIST/models/Osci_MNIST_linear/model10.pt')
model = Osci_AE(args=args, connectivity=model_logs['connectivity'], device=args.device, update_rate=.1, anneal=0, time_steps=10, phase_initialization=model_logs['initial_phase'], walk_step=.1, intrinsic_frequencies=args.intrinsic_frequencies).to(device)
model.load_state_dict(model_logs['model_state_dict'])
model.eval()


if True:
    path = '/media/data_cifs/mchalvid/Project_synchrony/MNIST/results/Osci_MNIST_linear/phases_10.npy'
    codes = np.load(path,allow_pickle=True)
    GMM = GaussianMixture(n_components=100, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=1000, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=1, verbose_interval=10)
    GMM.fit(codes)
    z = GMM.sample(100)
    z_ = torch.tensor(z[0]).to(args.device).float()
    samples = model.decode(z_)
    samples = samples.view(-1, 1, 28, 28)
    save_dir = os.path.join(args.save_dir, 'samples_GMM.png')
    save_image(samples.cpu(), save_dir, nrow=10)
else:
    path = '/media/data_cifs/mchalvid/Project_synchrony/MNIST/results/Osci_MNIST_linear/couplings_10.npy'
    codes = np.load(path,allow_pickle=True).reshape(10000,-1)
    GMM = GaussianMixture(n_components=10, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=1000, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=1, verbose_interval=10)
    GMM.fit(codes)
    couplings = GMM.sample(100)
    #couplings = torch.tensor(codes[:100].reshape(100,16,16)).to(args.device).float()
    couplings = torch.tensor(couplings[0].reshape(100,16,16)).to(args.device).float()
    samples, phase_list, couplings = model.sample_couplings(couplings)
    samples = samples[-1].view(-1, 1, 28, 28)
    save_dir = os.path.join(args.save_dir, 'samples_GMM_couplings.png')
    save_image(samples.cpu(), save_dir, nrow=10)