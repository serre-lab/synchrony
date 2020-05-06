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
from Kura_module_v2 import Kuramoto as km
from Kura_module_v2 import ODEDynamic
from utils_v2 import phase_evol, plot_evolution
import subprocess
from supervised.kuramoto.utils import generate_connectivity


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--epochs', type=int, default=50, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')



# Cosmetic parameters
parser.add_argument('--data_name', type=str, default='MNIST')
parser.add_argument('--exp_name', type=str, default='Osci_MNIST_debug_v2')
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
parser.add_argument('--num_cn1', type=int, default=25)
parser.add_argument('--num_cn2', type=int, default=25)
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
parser.add_argument('--n_osci1', type=int, default=25)
parser.add_argument('--n_osci2', type=int, default=784)
parser.add_argument('--max_time', type=float, default=20)
parser.add_argument('--dropout_p', type=float, default=0.4)
parser.add_argument('--solver', type=str, default='euler')

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

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
subprocess.call('rm -rf {}'.format(os.path.join(save_dir, '*')), shell=True)

args.load_dir = load_dir
args.save_dir = save_dir
args.model_dir = model_dir



train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.path,train=True, download=True,transform=transforms.ToTensor()),batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(datasets.MNIST(args.path,train=False, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=False, **kwargs)


class Osci_AE_v2(nn.Module):
    def __init__(self, args, connectivity1, connectivity2,
                 phase_initialization='random',
                 intrinsic_frequencies='zero', device='cpu'):
        super(Osci_AE_v2, self).__init__()
        """
        nn.module object for passing to odeint module for various image size, feature maps are all in the same shape as input
        """
        self.args = args
        self.n_osci1 = args.n_osci1
        self.n_osci2 = args.n_osci2
        self.num_global = args.num_global_control
        self.connectivity1 = connectivity1
        self.connectivity2 = connectivity2
        self.rank = 0
        self.osci1 = km(self.n_osci1, batch_size=args.batch_size,
                       connectivity0=self.connectivity1, num_global=self.num_global,
                       phase_initialization=phase_initialization,
                        device=device, max_time=args.max_time,
                       intrinsic_frequencies=intrinsic_frequencies)

        self.osci2 = km(self.n_osci2, batch_size=args.batch_size,
                       connectivity0=self.connectivity2, num_global=self.num_global,
                       phase_initialization=phase_initialization,
                        device=device, max_time=args.max_time,
                       intrinsic_frequencies=intrinsic_frequencies)

        self.ODE_evolution1 = self.osci1.ODE_evolution
        self.ODE_evolution2 = self.osci2.ODE_evolution

        self.num_cn1 = args.num_cn1
        self.num_cn2 = args.num_cn2
        self.img_side = args.img_side
        if self.num_global > 0:
            self.out_channels += 1
        self.split = args.split
        self.depth = args.depth
        self.args = args
        self.sigma = nn.Sigmoid()
        self.dropout = nn.Dropout(args.dropout_p)

        self.couplings1 = nn.Linear(784, args.n_osci1*args.num_cn1)
        self.couplings2 = nn.Linear(args.n_osci1, 784*args.num_cn2)
        self.read_out = nn.Linear(784, 784)

        self.osci1.ODEDynamic = ODEDynamic(args)
        self.osci2.ODEDynamic = ODEDynamic(args)

    def Kura1(self, x):
        couplings = self.couplings1(x.view(-1, 784)).reshape(-1, self.n_osci1, self.num_cn1)
        couplings / couplings.norm(p=2, dim=2).unsqueeze(2)
        omega = None
        phase_list, couplings = self.ODE_evolution1(couplings, omega=omega, method=self.args.solver)
        return phase_list, couplings

    def Kura2(self, z):
        couplings = self.couplings2(z.view(args.batch_size, -1)).reshape(-1, self.n_osci2, self.num_cn2)
        couplings / couplings.norm(p=2, dim=2).unsqueeze(2)
        omega = None
        phase_list, couplings = self.ODE_evolution2(couplings, omega=omega, method=self.args.solver)
        phase_list = [torch.sigmoid(self.read_out(phase)) for phase in phase_list]
        #phase_list = [torch.sigmoid(phase) for phase in phase_list]
        return phase_list, couplings

    def forward(self, x):
        z_list, couplings_1 = self.Kura1(x)
        reco=[]
        for z in z_list:
            x_rec, couplings_2 = self.Kura2(z)
            reco.append(x_rec[-1])
        return reco, z_list, couplings_1, couplings_2



######## Connectivity ########
# Number 1
local_connectivity, global_connectivity = generate_connectivity(args.num_cn1, 5, sw=args.sw,
                          num_global_control=args.num_global_control,
                          p_rewire=args.p_rewire, rf_type=args.rf_type)

local_connectivity = torch.tensor(local_connectivity).long()

if global_connectivity is not None:
    global_connectivity = torch.tensor(global_connectivity).long()
    connectivity = [local_connectivity, global_connectivity]
else:
    connectivity1 = local_connectivity

# Number 2
local_connectivity, global_connectivity = generate_connectivity(args.num_cn2, 28, sw=args.sw,
                          num_global_control=args.num_global_control,
                          p_rewire=args.p_rewire, rf_type=args.rf_type)

local_connectivity = torch.tensor(local_connectivity).long()

if global_connectivity is not None:
    global_connectivity = torch.tensor(global_connectivity).long()
    connectivity = [local_connectivity, global_connectivity]
else:
    connectivity2 = local_connectivity


######### Model ##########
model = Osci_AE_v2(args=args, connectivity1=connectivity1, connectivity2=connectivity2, device=args.device, phase_initialization=args.phase_initialization, intrinsic_frequencies=args.intrinsic_frequencies).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE #+ KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, z_list, couplings1, couplings2 = model(data)
        loss=0
        for reco in recon_batch[-1*args.record_steps:]:
            loss += loss_function(reco, data)
        loss.backward(retain_graph=True)
        couplings2.backward(gradient=model.osci2.ODEDynamic.couplings.grad,retain_graph=True)
        #couplings1.backward(gradient=model.osci1.ODEDynamic.couplings.grad, retain_graph=True)
        train_loss += loss.item()
        optimizer.step()
        #model.osci2.ODEDynamic.zero_grad()
        #model.osci1.ODEDynamic.zero_grad()
        if batch_idx % args.log_interval == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],recon_batch[-1].view(args.batch_size, 1, 28, 28)[:n]])
            save_dir = os.path.join(args.save_dir,'reconstruction_'+ str(epoch) + '_' +str(batch_idx) + '.png')
            save_image(comparison.cpu(),save_dir, nrow=n)
            #plot_evolution(recon_batch, data, epoch, args)
            #phase_evol(phase_list, data[0], epoch, args, masks=None, save_name=None)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))


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
                #comparison = torch.cat([data[:n],
                #                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                #save_dir = os.path.join(args.save_dir,'reconstruction_' + str(epoch) + '.png')
                #save_image(comparison.cpu(),save_dir, nrow=n)
                plot_evolution(recon_batch, data, epoch,args)
                phase_evol(phase_list,data[0].cpu().numpy(),epoch,args,masks=None, save_name=None)
            phases.append(phase_list[-1].cpu().numpy())

    phases = np.concatenate(phases)
    targets = test_loader.dataset.targets.cpu().numpy()
    np.save(os.path.join(args.save_dir, 'phases_{}.npy'.format(epoch)), phases)
    np.save(os.path.join(args.save_dir, 'targets_{}.npy'.format(epoch)), targets)
    test_loss /= len(test_loader.dataset)

    print('====> Test set loss: {:.4f}'.format(test_loss))



if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        #test(epoch)
        #with torch.no_grad():
            #sample = torch.randn(64, 20).to(device)
            #sample = model.decode(sample).cpu()
            #save_dir = os.path.join(args.save_dir,'sample_' + str(epoch) + '.png')
            #save_image(sample.view(64, 1, 28, 28),save_dir)