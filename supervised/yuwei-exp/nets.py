import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import torch
import torch.nn as nn
import numpy as np
from kuramoto import Kuramoto as km
import losses as ls
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import ipdb

class KuraNet(nn.Module):
    def __init__(self, img_side, connectivity, num_global, batch_size=32, device='cpu',
                 update_rate=.1, anneal=0, time_steps=10, record_steps=10, phase_initialization='random', walk_step=.1, intrinsic_frequencies='zero'):
        super(KuraNet, self).__init__()

        self.img_side = img_side
        self.num_global = num_global
        self.connectivity = connectivity
        self.osci = km(img_side**2, update_rate=update_rate, batch_size=batch_size,
                       anneal=anneal, time_steps=time_steps,
                       connectivity0=connectivity, num_global=num_global,
                       record_steps=record_steps,
                       phase_initialization=phase_initialization,
                       walk_step=walk_step, device=device,
                       intrinsic_frequencies=intrinsic_frequencies)
        self.evolution = self.osci.evolution

def load_net(args, connectivity, num_global):
    if args.model_name == 'simple_conv':
        return simple_conv(args, connectivity, num_global)
    elif args.model_name == 'Unet':
        return Unet(args, connectivity, num_global)
    else:
        raise ValueError('Network not included so far')

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling):
        super(DownConv, self).__init__()

        self.pooling = pooling
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1,
                               padding=int((kernel_size - 1) / 2))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1,
                               padding=int((kernel_size - 1) / 2))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(2 * out_channels, out_channels, kernel_size=kernel_size, stride=1,
                               padding=int((kernel_size - 1) / 2))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        x = torch.cat((from_up, from_down), 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x


class Unet(KuraNet):
    def __init__(self, args, connectivity, num_global):
        """
        Unet for semantic segmentation
        """
        super(Unet, self).__init__(args.img_side, connectivity, num_global, batch_size=args.batch_size, update_rate=args.update_rate, anneal=args.anneal, time_steps=args.time_steps, record_steps=args.record_steps, phase_initialization=args.phase_initialization, intrinsic_frequencies=args.intrinsic_frequencies, device=args.device)

        self.connections = args.num_cn
        self.img_side = args.img_side
        self.out_channels = args.out_channels
        if num_global > 0: self.out_channels += 1
        self.split = args.split
        self.num_global = num_global
        self.down_convs = []
        self.up_convs = []

        # create a encoder pathway
        for i in range(args.depth):
            ins = args.in_channels if i == 0 else outs
            outs = args.start_filts * (2 ** i)
            pooling = True if i < args.depth - 1 else False

            down_conv = DownConv(ins, outs, args.kernel_size[0], pooling)
            self.down_convs.append(down_conv)

        # create a decoder pathway
        for i in range(args.depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, args.kernel_size[1])
            self.up_convs.append(up_conv)

        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.conv_final = nn.Conv2d(outs, self.out_channels, kernel_size=1, stride=1, padding=0)
        if num_global == 0:
            self.linear = nn.Linear(int((self.out_channels/args.split) * (args.img_side ** 2)),
                                    int(((args.img_side ** 2)/args.split) * args.num_cn))
        else:
            self.linear1 = nn.Linear(int(((self.out_channels)/args.split) * (args.img_side ** 2)),
                                     int(((args.img_side ** 2)/args.split) * (args.num_cn + 1)))
            self.linear2 = nn.Linear((args.img_side ** 2), args.img_side ** 2 + num_global ** 2 - num_global)
        self.reset_params()

    def forward(self, x):
        x_in = x
        encoder_outs = []

        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)
        if self.num_global == 0:
            x = self.linear(x.reshape(-1, int((self.out_channels / self.split) *
                                              (self.img_side ** 2)))).reshape(-1, self.img_side ** 2, self.connections)
            x = x / x.norm(p=2, dim=2).unsqueeze(2)
            phase_list, coupling, omega = self.evolution(x, batch=x_in, hierarchical=False)
            return phase_list, coupling, omega
        else:
            x1 = self.linear1(x[:, :-1,
                              ...].reshape(-1, int(((self.out_channels -1)/self.split) *
                                           (self.img_side ** 2)))).reshape(-1, self.img_side ** 2,
                                                                           self.connections + 1)
            x2 = self.linear2(x[:, -1,
                              ...].reshape(-1, self.img_side ** 2)).reshape(-1, self.num_global,
                                                                            int(self.img_side ** 2/self.num_global) +
                                                                            self.num_global - 1)
            x1 = x1 / x1.norm(p=2, dim=2).unsqueeze(2)
            x2 = x2 / x2.norm(p=2, dim=2).unsqueeze(2)
            x=[x1,x2]
            phase_list, coupling, omega = self.evolution(x, batch=x_in, hierarchical = True)
            phase_list = [phase[:, :-self.num_global] for phase in phase_list]
            return phase_list, coupling, omega

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weights_init(m)


class simple_conv(KuraNet):
    def __init__(self, args, connectivity, num_global):
        """
        For various image size, feature maps are all in the same shape as input
        """
        super(simple_conv, self).__init__(args.img_side, connectivity, num_global, batch_size=args.batch_size, update_rate=args.update_rate, anneal=args.anneal, time_steps=args.time_steps, record_steps=args.record_steps, phase_initialization=args.phase_initialization, walk_step=args.walk_step, intrinsic_frequencies=args.intrinsic_frequencies, device=args.device)

        self.num_cn = args.num_cn
        self.num_global = num_global
        self.img_side = args.img_side
        self.out_channels = args.out_channels
        if num_global > 0: self.out_channels += 1
        self.split = args.split
        self.convs1 = []
        self.convs2 = []
        self.depth = args.depth

        start_filts = int(args.start_filts / 2)
        for i in range(self.depth):
            ins = args.in_channels if i == 0 else outs
            outs = start_filts * (2 ** i)
            conv = nn.Conv2d(ins, outs, kernel_size=args.kernel_size[0], stride=1, padding=int((args.kernel_size[0] - 1) / 2))
            self.convs1.append(conv)
            conv = nn.Conv2d(ins, outs, kernel_size=args.kernel_size[1], stride=1, padding=int((args.kernel_size[1] - 1) / 2))
            self.convs2.append(conv)

        self.convs1 = nn.ModuleList(self.convs1)
        self.convs2 = nn.ModuleList(self.convs2)

        self.conv_final = nn.Conv2d(outs * 2, self.out_channels, kernel_size=1, stride=1, padding=0)

        if num_global == 0:
            self.linear = nn.Linear(int((self.out_channels / self.split) * (self.img_side ** 2)),
                                    int(((self.img_side ** 2) / self.split) * self.num_cn))
        else:
            self.linear1 = nn.Linear(int(((self.out_channels - 1)/self.split) * (self.img_side ** 2)),
                                     int(((self.img_side ** 2)/self.split) * (self.num_cn + 1)))
            self.linear2 = nn.Linear((self.img_side ** 2), self.img_side ** 2 + self.num_global ** 2 - self.num_global)
        self.reset_params()

    def forward(self, x):
        x_in = x
        x1 = x
        x2 = x
        for i, module in enumerate(self.convs1):
            x1 = torch.relu(module(x1)) if i < self.depth - 1 else torch.sigmoid(module(x1))
        for i, module in enumerate(self.convs2):
            x2 = torch.relu(module(x2)) if i < self.depth - 1 else torch.sigmoid(module(x2))

        x = torch.cat([x1, x2], dim=1)
        for c in range(x1.shape[1]):
            x[:, c * 2, ...] = x1[:, c, ...]
            x[:, c * 2 + 1, ...] = x2[:, c, ...]

        x = self.conv_final(x)

        if self.num_global == 0:
            x = self.linear(x.reshape(-1, int((self.out_channels / self.split) *
                                              (self.img_side ** 2)))).reshape(-1, self.img_side ** 2, self.num_cn)

            x = x / x.norm(p=2, dim=2).unsqueeze(2)

            phase_list, coupling, omega = self.evolution(x, batch=x_in, hierarchical=False)
            return phase_list, coupling, omega
        else:
            x1 = self.linear1(x[:, :-1,
                              ...].reshape(-1, int(((self.out_channels - 1)/self.split) *
                                           (self.img_side ** 2)))).reshape(-1, self.img_side ** 2,
                                                                           self.num_cn + 1)
            x2 = self.linear2(x[:, -1:,
                              ...].reshape(-1, self.img_side ** 2)).reshape(-1, self.num_global,
                                                                            int(self.img_side ** 2/self.num_global) +
                                                                            self.num_global - 1)
            x1 = x1 / x1.norm(p=2, dim=2).unsqueeze(2)
            x2 = x2 / x2.norm(p=2, dim=2).unsqueeze(2)
            x=[x1,x2]
            phase_list, coupling, omega = self.evolution(x, batch=x_in, hierarchical=True)
            phase_list = [phase[:, :-self.num_global] for phase in phase_list]
            return phase_list, coupling, omega

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weights_init(m)


class criterion(nn.Module):
    def __init__(self, degree):
        super(criterion, self).__init__()

        self.degree = degree

    def forward(self, phase_list, mask, transform, device, valid=False):
        # losses will be 1d, with its length = episode length
        if valid:
            losses = \
                ls.exinp_integrate_torch2(torch.cat(phase_list, dim=0).detach(),
                                          mask.repeat(len(phase_list), 1, 1).detach(),
                                          transform,
                                          device).reshape(len(phase_list), mask.shape[0]).mean(1)
        else:
            losses = \
                ls.exinp_integrate_torch2(torch.cat(phase_list, dim=0),
                                          mask.repeat(len(phase_list), 1, 1),
                                          transform,
                                          device).reshape(len(phase_list), mask.shape[0]).mean(1)
        return torch.matmul(losses,
                            torch.pow(torch.arange(len(phase_list)) + 1, self.degree).unsqueeze(1).float().to(device))


class regularizer(nn.Module):
    def __init__(self):
        super(regularizer, self).__init__()

    def forward(self, coupling, mask, device, valid=False):
        if valid:
            return lx.coupling_regularizer(coupling.detach(), mask.detach(), device).mean()
        else:
            return lx.coupling_regularizer(coupling, mask, device).mean()


def plot_grad_flow(named_parameters, save_name):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("biasqq" not in n) and (p.grad is not None):

            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            if p.grad.abs().mean() == 0:
                layers.append(n + "ZERO")
            elif p.grad.abs().mean() < 0.00001:
                layers.append(n + "SMALL")
            else:
                layers.append(n)

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.2, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.2, lw=1, color="b")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical", fontsize=5)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=10)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([lines.Line2D([0], [0], color="c", lw=4, alpha=0.2),
                lines.Line2D([0], [0], color="b", lw=4, alpha=0.2)], ['max-gradient', 'mean-gradient'])
    # plt.draw()
    # plt.pause(0.001)
    plt.savefig(save_name + '.png')
    plt.close()
    return 0


def plot_weight_flow(named_parameters, save_name):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave = []
    max = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("biasqq" not in n) and (p.grad is not None):

            ave.append(p.data.abs().mean())
            max.append(p.data.abs().max())
            layers.append(n)

    plt.bar(np.arange(len(max)), max, alpha=0.2, lw=1, color="y")
    plt.bar(np.arange(len(max)), ave, alpha=0.2, lw=1, color="r")
    plt.xticks(range(0, len(ave), 1), layers, rotation="vertical", fontsize=5)
    plt.xlim(left=0, right=len(ave))
    plt.ylim(bottom=-0.001, top=0.5)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average weights")
    plt.title("Weights flow")
    plt.grid(True)
    plt.legend([lines.Line2D([0], [0], color="y", lw=4, alpha=0.2),
                lines.Line2D([0], [0], color="r", lw=4, alpha=0.2)], ['max-value', 'mean-value'])
    # plt.draw()
    # plt.pause(0.001)
    plt.savefig(save_name + '.png')
    plt.close()
    return 0


def plot_variable_flow(named_parameters, save_name):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    weights = []
    grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("biasqq" not in n) and (p.grad is not None):
            weights.append(p.data.abs().mean())
            grads.append(p.grad.abs().mean())
            layers.append(n)

    plt.bar(np.arange(len(weights)), weights, alpha=0.2, lw=1, color="r")
    plt.bar(np.arange(len(grads)), grads, alpha=0.2, lw=1, color="y")
    plt.xticks(range(0, len(weights), 1), layers, rotation="vertical", fontsize=6)
    plt.xlim(left=0, right=len(weights))
    plt.ylim(bottom=-0.001, top=1.0)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Value")
    plt.title("Weights and Gradients flow")
    plt.grid(True)
    plt.legend([lines.Line2D([0], [0], color="r", lw=4, alpha=0.2),
                lines.Line2D([0], [0], color="y", lw=4, alpha=0.2)], ['weights', 'gradients'])
    # plt.draw()
    # plt.pause(0.001)
    plt.savefig(save_name + '.png')
    plt.close()
    return 0


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class autoencoder(nn.Module):
    def __init__(self, img_side, num_global_control=0):
        super(autoencoder,self).__init__()
        self.num_global_control = num_global_control
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(16,8,kernel_size=5),
            nn.ReLU(True))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8,16,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(16,1,kernel_size=5),
            nn.ReLU(True))
        if num_global_control > 0:
            self.h_channel = nn.Linear(8*(img_side - 8)**2, num_global_control)

    def forward(self,x):
        x = self.encoder(x)
        if self.num_global_control > 0:
            y = self.h_channel(x.reshape(x.shape[0], -1))
            x = self.decoder(x)
            return torch.cat((x.reshape(x.shape[0],-1),y),dim=1)
        else:
            x = self.decoder(x)
            return x.reshape(x.shape[0], -1)
