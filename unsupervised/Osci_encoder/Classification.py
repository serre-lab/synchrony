import argparse
import os
import torch
import torch.utils.data
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
#from Kura_module import Kuramoto as km
#from Kura_module import ODEDynamic, ODEDynamic_linear
from utils import phase_evol, plot_evolution
import subprocess
from sklearn.mixture.gaussian_mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
from itertools import permutations
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

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



#model_logs = torch.load('/media/data_cifs/mchalvid/Project_synchrony/MNIST/models/Osci_MNIST_linear/model10.pt')
#model = Osci_AE(args=args, connectivity=model_logs['connectivity'], device=args.device, update_rate=.1, anneal=0, time_steps=10, phase_initialization=model_logs['initial_phase'], walk_step=.1, intrinsic_frequencies=args.intrinsic_frequencies).to(device)
#model.load_state_dict(model_logs['model_state_dict'])
#model.eval()

Couplings = False
if not Couplings:
    path = '/media/data_cifs/mchalvid/Project_synchrony/MNIST/results/Osci_MNIST_general_dim20/phases_10.npy'
    path_targets = '/media/data_cifs/mchalvid/Project_synchrony/MNIST/results/Osci_MNIST_general_dim20/targets_10.npy'
    #path = '/media/data_cifs/mchalvid/Project_synchrony/MNIST/results/AE_encoder_/z_10.npy'
    #path_targets = '/media/data_cifs/mchalvid/Project_synchrony/MNIST/results/AE_encoder_/targets_10.npy'
    #path = '/media/data_cifs/mchalvid/Project_synchrony/MNIST/results/VAE_vanilla/z_10.npy'
    #path_targets = '/media/data_cifs/mchalvid/Project_synchrony/MNIST/results/VAE_vanilla/targets_10.npy'
    codes = np.load(path,allow_pickle=True)
    targets = np.load(path_targets,allow_pickle=True)

    tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                          n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='random',
                          verbose=1, random_state=None, angle=0.5, n_jobs=None)
    comp = tsne.fit_transform(codes)

    #KM = KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=1000, tol=0.0001, precompute_distances='auto', verbose=1, random_state=None, copy_x=True, n_jobs=None, algorithm='auto')
    GMM = GaussianMixture(n_components=10, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=1000, n_init=10, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=1, verbose_interval=10)
    #DB = DBSCAN(eps=10, min_samples=25, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
    #OP = OPTICS(min_samples=5, max_eps=np.inf, metric='minkowski', p=2, metric_params=None, cluster_method='xi',
                           #eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto',
                           #leaf_size=30, n_jobs=None)
    Labels_KM = GMM.fit_predict(comp)

    print(Labels_KM)
    permutation = np.zeros(10)
    scores = np.zeros(10)
    for i in range(10):
        max = 0
        for j in range(10):
            score = np.sum((Labels_KM == j)*(targets == i))
            if score > max:
                permutation[i]=j
                scores[i]=score
                max=score

    score = np.sum(scores)/len(targets)
    print(permutation,score)
    predicted = np.array([permutation[i] for i in targets])
    """
    sns.set_style("white")
    current_palette = sns.color_palette("hls", 10)
    f, ax = plt.subplots(figsize=(10, 10))
    sns.despine(f, left=True, bottom=True)
    sns.scatterplot(x=comp[:,0], y=comp[:,1], palette=current_palette,
                    hue=Labels_KM, alpha=.6, ax=ax)
    plt.savefig('/media/data_cifs/mchalvid/Project_synchrony/MNIST/results/VAE_vanilla/' + 'clustering.png', dpi=200)
    plt.show()
    """