from __future__ import print_function
import argparse
import os, subprocess
import torch
import numpy as np
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils import phase_evol, plot_evolution

os.environ["CUDA_VISIBLE_DEVICES"]='2'


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

path = '/media/data_cifs/mchalvid/Project_synchrony/MNIST'
args.exp_name = 'VAE_vanilla'
args.path = path
args.show_every = 50

args.load_dir = args.path
args.save_dir = os.path.join(args.path,'results', args.exp_name)
args.model_dir = os.path.join(args.path,'models', args.exp_name)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
subprocess.call('rm -rf {}'.format(os.path.join(args.save_dir, '*')), shell=True)


train_loader = torch.utils.data.DataLoader(datasets.MNIST(path , train=True, download=True,transform=transforms.ToTensor()),batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(datasets.MNIST(path , train=False, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=False, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 25)
        self.fc22 = nn.Linear(400, 25)
        self.fc3 = nn.Linear(25, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, z = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))


    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    Z_ = []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar, z = model(data)
            Z_.append(z.cpu().numpy())
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i % args.show_every == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_dir = os.path.join(args.save_dir,'reconstruction_' + str(epoch) + '.png')
                save_image(comparison.cpu(),save_dir, nrow=n)
                #plot_evolution(recon_batch, data, epoch,args)
                #phase_evol([z],data[0].cpu().numpy(),epoch,args,masks=None, save_name=None)

    Z_ = np.concatenate(Z_)
    targets = test_loader.dataset.targets.cpu().numpy()
    np.save(os.path.join(args.save_dir, 'z_{}.npy'.format(epoch)), Z_)
    np.save(os.path.join(args.save_dir, 'targets_{}.npy'.format(epoch)), targets)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 25).to(device)
            sample = model.decode(sample).cpu()
            save_dir = os.path.join('/media/data_cifs/mchalvid/Project_synchrony/MNIST',
                                    'results/sample_' + str(epoch) + '.png')
            save_image(sample.view(64, 1, 28, 28),save_dir)