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
from modules import AutoencODE, AutoencODE_CIFAR
from New_modules import AutoencODE_stack
from utils import phase_evol, plot_evolution
import subprocess
#from supervised.kuramoto.utils import generate_connectivity

os.environ["CUDA_VISIBLE_DEVICES"]='0'

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')


# Cosmetic parameters
parser.add_argument('--data_name', type=str, default='MNIST')
parser.add_argument('--exp_name', type=str, default='MNIST_t=10')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--interactive', type=lambda x:bool(strtobool(x)), default=False)

# Model parameters
parser.add_argument('--num_global_control', type=int, default=0)
parser.add_argument('--p_rewire', type=float, default=0.0)
parser.add_argument('--rf_type', type=str, default='arange')
parser.add_argument('--phase_initialization', type=str, default='fixed_unif')
parser.add_argument('--intrinsic_frequencies', type=str, default='zero')
parser.add_argument('--update_rate', type=float, default=1.6)
parser.add_argument('--sw', type=lambda x:bool(strtobool(x)),default=False)
parser.add_argument('--record_steps', type=int, default=1)
parser.add_argument('--ode_train', type=lambda x:bool(strtobool(x)),default=False)
parser.add_argument('--n_osci', type=int, default=20)
parser.add_argument('--num_cn', type=int, default=20)
parser.add_argument('--max_time', type=float, default=10)
parser.add_argument('--time_steps', type=int, default=20)
parser.add_argument('--dropout_p', type=float, default=0.4)
parser.add_argument('--solver', type=str, default='euler')
parser.add_argument('--dynamic_type', type=str, default='general')
parser.add_argument('--model_type', type=str, default='regular')

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


args = parser.parse_args()


torch.manual_seed(args.seed)
args.path = '/media/data_cifs/mchalvid/Project_synchrony/{}/stacked_exp'.format(args.data_name)
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}




#DATA loading
args.load_dir = os.path.join(args.path, args.data_name)
args.save_dir = os.path.join(args.path,'results', args.exp_name)
args.model_dir = os.path.join(args.path,'models', args.exp_name)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
subprocess.call('rm -rf {}'.format(os.path.join(args.save_dir, '*')), shell=True)

if args.data_name == 'MNIST':
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.path,train=True, download=True,transform=transforms.ToTensor()),batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST(args.path,train=False, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=False, **kwargs)
elif args.data_name == 'CIFAR':
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.path, train=True,
                                                                transform=transforms.Compose([transforms.ToTensor()])),
                                               batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.path, train=False,
                                                               transform=transforms.Compose([transforms.ToTensor()])),
                                              batch_size=args.batch_size, shuffle=False, **kwargs)


######## Connectivity ########
#local_connectivity, global_connectivity = \
#    generate_connectivity(args.num_cn, 4, sw=args.sw,
#                          num_global_control=args.num_global_control,
#                          p_rewire=args.p_rewire, rf_type=args.rf_type)

#local_connectivity = torch.tensor(local_connectivity).long()

#if global_connectivity is not None:
    #global_connectivity = torch.tensor(global_connectivity).long()
    #connectivity = [local_connectivity, global_connectivity]
#else:
    #connectivity = local_connectivity

connectivity = (torch.ones(args.n_osci,args.n_osci) - torch.eye(args.n_osci,args.n_osci)).long()
#connectivity = (torch.triu(torch.ones(args.n_osci,args.n_osci))).long()
#connectivity = (torch.distributions.bernoulli.Bernoulli(0.5).sample((20,20))).long()
print(connectivity.shape)



######### Model ##########
if args.model_type == 'stacked':
    connectivity2 = (torch.ones(100, 100) - torch.eye(100, 100)).long()
    if args.data_name == 'MNIST':
        model = AutoencODE_stack(args=args, connectivity1=connectivity2,connectivity2=connectivity, device=args.device, phase_initialization=args.phase_initialization).to(device)
        def loss_function(recon_x, x):
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
            return BCE
        args.img_side = 28
        args.channels_im = 1
else:
    if args.data_name == 'MNIST':
        model = AutoencODE(args=args, connectivity=connectivity, device=args.device, update_rate=.1, anneal=0, time_steps=10, phase_initialization=args.phase_initialization, walk_step=.1, intrinsic_frequencies=args.intrinsic_frequencies).to(device)
        def loss_function(recon_x, x):
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
            return BCE
        args.img_side= 28
        args.channels_im = 1

    elif args.data_name == 'CIFAR':
        model = AutoencODE_CIFAR(args=args, connectivity=connectivity, device=args.device, update_rate=.1, anneal=0, time_steps=10, phase_initialization=args.phase_initialization, walk_step=.1, intrinsic_frequencies=args.intrinsic_frequencies).to(device)
        def loss_function(recon_x, x):
            #BCE = F.binary_cross_entropy(recon_x, x.view(-1,3,args.img_side,args.img_side), reduction='sum')
            MSE = F.mse_loss(recon_x, x.view(-1,3,args.img_side,args.img_side), size_average=None, reduce=None, reduction='sum')
            return MSE #BCE
        args.img_side = 32
        args.channels_im=3


optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
    loss_history = []

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, phase_list, couplings = model(data)
        loss=0
        for reco in recon_batch[-1*args.record_steps:]:
            loss += loss_function(reco, data)
        loss.backward(retain_graph=True)
        couplings.backward(gradient=model.osci.ODEDynamic.couplings.grad)
        #couplings2.backward(gradient=model.osci2.ODEDynamic.couplings.grad,retain_graph=True)
        #couplings1.backward(gradient=model.osci1.ODEDynamic.couplings.grad, retain_graph=True)
        train_loss += loss.item()
        optimizer.step()
        loss_history.append(loss.item()/len(data))

        if batch_idx % args.log_interval == 0:
            phase_evol(phase_list, epoch, args, masks=None, save_name=None)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    return loss_history


def test(epoch):
    model.eval()
    test_loss = 0
    loss_history = []
    phases = []
    couplings_list = []

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, phase_list, couplings = model(data)
            for reco in recon_batch[-1 * args.record_steps:]:
                l = loss_function(reco, data).item()/args.record_steps
                test_loss += l/len(data)
            #test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i % args.log_interval == 0:
                n = min(data.size(0), 20)
                comparison = torch.cat([data[:n],reco.view(args.batch_size, args.channels_im, args.img_side, args.img_side)[:n]])
                save_dir = os.path.join(args.save_dir,'reconstruction_' + str(epoch) + '.png')
                save_image(comparison.cpu(),save_dir, nrow=10)
                plot_evolution(recon_batch, data, epoch,args)
                phase_evol(phase_list,epoch,args,masks=None, save_name=None)
            phases.append(phase_list[-1].cpu().numpy())
            couplings_list.append(couplings.cpu().numpy())

    phases = np.concatenate(phases)
    couplings_list = np.concatenate(couplings_list)
    targets = np.array(test_loader.dataset.targets)
    np.save(os.path.join(args.save_dir, 'phases_{}.npy'.format(epoch)), phases)
    np.save(os.path.join(args.save_dir, 'couplings_{}.npy'.format(epoch)), couplings_list)
    np.save(os.path.join(args.save_dir, 'targets_{}.npy'.format(epoch)), targets)
    #test_loss /= len(test_loader.dataset)
    torch.save(model.state_dict(), args.model_dir+'/OAE_{}'.format(epoch))
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

if __name__ == "__main__":
    loss_history = []
    loss_history_test = []
    for epoch in range(1, args.epochs + 1):
        loss_hist = train(epoch)
        loss_history += loss_hist
        np.save(os.path.join(args.save_dir,'loss'+'.npy'),np.array(loss_history))
        loss_hist_test = test(epoch)
        loss_history_test.append(loss_hist_test)
        np.save(os.path.join(args.save_dir,'loss_test' + '.npy'),np.array(loss_history_test))
        #with torch.no_grad():
        #    sample = torch.randn(64, args.n_osci,args.num_cn).to(device)
        #    sample_rec, phase_list, couplings = model.sample(sample)
        #    save_dir = os.path.join(args.save_dir,'sample_' + str(epoch) + '.png')
        #   save_image(sample_rec[-1].view(64, 1, 28, 28).cpu(),save_dir)
        # save file s
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'initial_phase': args.phase_initialization,
                    'connectivity': connectivity}, args.model_dir + '/model{}.pt'.format(epoch))