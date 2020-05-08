from __future__ import print_function
import argparse
import os, sys
import torch
import torch.utils.data
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from Kura_module import Kuramoto as km
from Kura_module import ODEDynamic, Osci_AE
from utils import phase_evol, plot_evolution, generate_connectivity
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"]='2'

parser = argparse.ArgumentParser(description='OAE_CIFAR')
parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')



# Cosmetic parameters
parser.add_argument('--data_name', type=str, default='CIFAR')
parser.add_argument('--exp_name', type=str, default='Osci_CIFAR_debug')
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
parser.add_argument('--kernel_size', type=str, default=5)
parser.add_argument('--num_cn', type=int, default=60)
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
parser.add_argument('--n_osci', type=int, default=121)
parser.add_argument('--max_time', type=float, default=20)
parser.add_argument('--dropout_p', type=float, default=0.4)
parser.add_argument('--solver', type=str, default='euler')
parser.add_argument('--dynamic_type', type=str, default='general')

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


parser.add_argument('--path',type=str, default='/media/data_cifs/mchalvid/Project_synchrony/CIFAR')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


load_dir = os.path.join(args.path, args.data_name)
save_dir = os.path.join(args.path,'results', args.exp_name)
model_dir = os.path.join(args.path,'models', args.exp_name)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
subprocess.call('rm -rf {}'.format(os.path.join(save_dir, '*')), shell=True)

args.load_dir = load_dir
args.save_dir = save_dir
args.model_dir = model_dir


train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.path,train=True,
                                            transform=transforms.Compose([transforms.ToTensor()])),
                                           batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.path,train=False,
                                                                    transform = transforms.Compose([transforms.ToTensor()])),
                                          batch_size=args.batch_size, shuffle=False, **kwargs)


######## Connectivity ########
local_connectivity, global_connectivity = \
    generate_connectivity(args.num_cn, 11, sw=args.sw,
                          num_global_control=args.num_global_control,
                          p_rewire=args.p_rewire, rf_type=args.rf_type,all_to_all=False)

local_connectivity = torch.tensor(local_connectivity).long()

if global_connectivity is not None:
    global_connectivity = torch.tensor(global_connectivity).long()
    connectivity = [local_connectivity, global_connectivity]
else:
    connectivity = local_connectivity


######### Model ##########
model = Osci_AE(args=args, connectivity=connectivity, device=args.device, update_rate=.1, anneal=0, time_steps=10, phase_initialization=args.phase_initialization, walk_step=.1, intrinsic_frequencies=args.intrinsic_frequencies).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1,3,args.img_side,args.img_side), reduction='sum')
    #MSE = F.mse_loss(recon_x, x.view(-1,3,args.img_side,args.img_side), size_average=None, reduce=None, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE #BCE #+ KLD


def train(epoch,args):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, phase_list, couplings = model(data)
        loss=0
        for reco in recon_batch[-1*args.record_steps:]:
            loss += loss_function(reco, data)
        loss.backward(retain_graph=True)
        couplings.backward(gradient=model.osci.ODEDynamic.couplings.grad)
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            #phase_evol(phase_list, data[0], epoch, args, masks=None, save_name=None)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch[-1].view(args.batch_size, 3, 32, 32)[:n]])
            save_dir = os.path.join(args.save_dir, 'reconstruction_' + str(epoch) + '.png')
            save_image(comparison.cpu(), save_dir, nrow=n)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    phases = []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, phase_list, couplings = model(data)
            for reco in recon_batch[-1 * args.record_steps:]:
                test_loss += loss_function(reco, data).item()/args.record_steps
            #test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i % args.show_every == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch[-1].view(args.batch_size, 3, 32, 32)[:n]])
                save_dir = os.path.join(args.save_dir,'reconstruction_' + str(epoch) + '.png')
                save_image(comparison.cpu(),save_dir, nrow=n)
                #plot_evolution(recon_batch, data, epoch,args)
                #phase_evol(phase_list,data[0].cpu().numpy(),epoch,args,masks=None, save_name=None)
            phases.append(phase_list[-1].cpu().numpy())

    phases = np.concatenate(phases)
    #targets = test_loader.dataset.targets.cpu().numpy()
    np.save(os.path.join(args.save_dir, 'phases_{}.npy'.format(epoch)), phases)
    #np.save(os.path.join(args.save_dir, 'targets_{}.npy'.format(epoch)), targets)
    test_loss /= len(test_loader.dataset)

    print('====> Test set loss: {:.4f}'.format(test_loss))



if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch,args)
        test(epoch)
        #with torch.no_grad():
            #sample = torch.randn(64, 20).to(device)
            #sample = model.decode(sample).cpu()
            #save_dir = os.path.join(args.save_dir,'sample_' + str(epoch) + '.png')
            #save_image(sample.view(64, 1, 28, 28),save_dir)