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
from Kura_module import ODEDynamic
from utils import phase_evol, plot_evolution
import subprocess
from supervised.kuramoto.utils import generate_connectivity


parser = argparse.ArgumentParser(description='VAE BouncingBall Example')
parser.add_argument('--epochs', type=int, default=50, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')


# Cosmetic parameters
parser.add_argument('--data_name', type=str, default='Bouncing_data')
parser.add_argument('--exp_name', type=str, default='Osci_BB')
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
parser.add_argument('--num_cn', type=int, default=64)
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
parser.add_argument('--n_osci', type=int, default=64)
parser.add_argument('--max_time', type=float, default=20)
parser.add_argument('--dropout_p', type=float, default=0.4)
parser.add_argument('--solver', type=str, default='euler')
parser.add_argument('--n_pres', type=int, default=4)


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
parser.add_argument('--batch_size', type=int, default=50)

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


parser.add_argument('--path',type=str, default='/media/data_cifs/mchalvid/Project_synchrony')

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

args.train_path = load_dir + '/train'
args.test_path = load_dir + '/test'

train_loader = torch.utils.data.DataLoader(datasets.DatasetFolder(args.train_path, np.load, extensions=('npy',),transform=transforms.ToTensor()),batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(datasets.DatasetFolder(args.test_path, np.load, extensions=('npy',),transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True, **kwargs)

class Osci_AE(nn.Module):
    def __init__(self, args, connectivity,
                 update_rate=1, anneal=0, time_steps=20,
                 phase_initialization='random', walk_step=20,
                 intrinsic_frequencies='zero', device='cpu'):
        super(Osci_AE, self).__init__()
        """
        nn.module object for passing to odeint module for various image size, feature maps are all in the same shape as input
        """
        super(Osci_AE, self).__init__()
        self.args = args
        self.n_osci = args.n_osci
        self.num_global = args.num_global_control
        self.connectivity = connectivity
        self.rank = 0
        self.osci = km(self.n_osci, update_rate=update_rate, batch_size=args.batch_size,
                       anneal=anneal, time_steps=time_steps,
                       connectivity0=connectivity, num_global=self.num_global,
                       phase_initialization=phase_initialization,
                       walk_step=walk_step, device=device, max_time=args.max_time,
                       intrinsic_frequencies=intrinsic_frequencies)
        self.evolution = self.osci.evolution
        self.ODE_evolution = self.osci.ODE_evolution

        self.num_cn = args.num_cn
        self.img_side = args.img_side
        if self.num_global > 0:
            self.out_channels += 1
        self.split = args.split
        self.depth = args.depth
        self.args = args
        self.sigma = nn.Sigmoid()
        self.dropout = nn.Dropout(args.dropout_p)


        self.fc1 = nn.Linear(784*args.n_pres, 400)
        self.fc21 = nn.Linear(400, args.n_osci*args.num_cn)
        #self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(args.n_osci, 400)
        self.fc4 = nn.Linear(400, 784)
        self.osci.ODEDynamic = ODEDynamic(args)

    def encode(self, x):

        h1 = F.relu(self.fc1(x))
        return self.fc21(h1)  #, self.fc22(h1)

    #def reparameterize(self, mu, logvar):
    #    std = torch.exp(0.5*logvar)
    #    eps = torch.randn_like(std)
    #    return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):

        couplings = self.encode(x.view(-1, 784*args.n_pres)).reshape(-1, self.n_osci, self.num_cn)
        couplings = couplings / couplings.norm(p=2, dim=2).unsqueeze(2)
        omega=None
        phase_list , couplings = self.ODE_evolution(couplings, omega=omega, method=self.args.solver)
        reco=[]
        for phase in phase_list:
            reco.append(self.decode(phase))
        return reco, phase_list, couplings

    def sample(self, couplings):
        with torch.no_grad():
            couplings = couplings / couplings.norm(p=2, dim=2).unsqueeze(2)
            omega=None
            phase_list , couplings = self.ODE_evolution(couplings, omega=omega, method=self.args.solver)
            reco=[]
            for phase in phase_list:
                reco.append(self.decode(phase))
            return reco, phase_list, couplings




######## Connectivity ########
local_connectivity, global_connectivity = \
    generate_connectivity(args.num_cn, 8, sw=args.sw,
                          num_global_control=args.num_global_control,
                          p_rewire=args.p_rewire, rf_type=args.rf_type)

local_connectivity = torch.tensor(local_connectivity).long()

if global_connectivity is not None:
    global_connectivity = torch.tensor(global_connectivity).long()
    connectivity = [local_connectivity, global_connectivity]
else:
    connectivity = local_connectivity

print(connectivity.shape)

######### Model ##########
model = Osci_AE(args=args, connectivity=connectivity, device=args.device, update_rate=.1, anneal=0, time_steps=10, phase_initialization=args.phase_initialization, walk_step=.1, intrinsic_frequencies=args.intrinsic_frequencies).to(device)
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
        data = data.to(device).float()
        pres = torch.cat([data[int(i)] for i in [0,2,4,6]],dim=0)
        optimizer.zero_grad()
        recon_batch, phase_list, couplings = model(pres)
        loss=0
        for i in [8,10,18,24]:
            loss += loss_function(recon_batch[i*2], data[i])
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


    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    phases = []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device).float()
            print(data.shape)
            pres = torch.cat([data[int(i)] for i in [0, 2, 4, 6]], dim=0)
            recon_batch, phase_list, couplings = model(pres)
            loss = 0
            for i in [8, 10, 12, 14]:
                loss += loss_function(recon_batch[i], data[i])
            #test_loss += loss_function(recon_batch, data, mu, logvar).item()
            #if i % args.show_every == 0:
                #n = min(data.size(0), 50)
                #comparison = torch.cat([data[:n],
                #                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                #save_dir = os.path.join(args.save_dir,'reconstruction_' + str(epoch) + '.png')
                #save_image(comparison.cpu(),save_dir, nrow=n)
                #plot_evolution(recon_batch, data, epoch,args)
                #phase_evol(phase_list,data[0].cpu().numpy(),epoch,args,masks=None, save_name=None)
            plot_evolution(recon_batch, data, epoch, args)
            phases.append(phase_list[-1].cpu().numpy())

    phases = np.concatenate(phases)
    #targets = test_loader.dataset.targets.cpu().numpy()
    np.save(os.path.join(args.save_dir, 'phases_{}.npy'.format(epoch)), phases)
    #np.save(os.path.join(args.save_dir, 'targets_{}.npy'.format(epoch)), targets)
    test_loss /= len(test_loader.dataset)
    torch.save(model.state_dict(), args.model_dir+'/OAE_{}'.format(epoch))
    print('====> Test set loss: {:.4f}'.format(test_loss))



if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        #with torch.no_grad():
        #    sample = torch.randn(64, args.n_osci,args.num_cn).to(device)
        #    sample_rec, phase_list, couplings = model.sample(sample)
        #    save_dir = os.path.join(args.save_dir,'sample_' + str(epoch) + '.png')
        #    save_image(sample_rec[-1].view(64, 1, 28, 28).cpu(),save_dir)