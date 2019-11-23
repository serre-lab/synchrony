import torch
import torch.nn as nn
import kuramoto as km
import losses as ls


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
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        x = torch.cat((from_up, from_down), 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, start_filts, depth, img_side, connections, kernel_size, split):
        """
        Unet for semantic segmentation
        """
        super(Unet, self).__init__()
        self.connections = connections
        self.img_side = img_side
        self.out_channels = out_channels
        self.split = split
        self.down_convs = []
        self.up_convs = []

        # create a encoder pathway
        for i in range(depth):
            ins = in_channels if i == 0 else outs
            outs = start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, kernel_size, pooling)
            self.down_convs.append(down_conv)

        # create a decoder pathway
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs)
            self.up_convs.append(up_conv)

        self.conv_final = nn.Conv2d(outs, out_channels, kernel_size=1, stride=1, padding=0)

        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.linear = nn.Linear(int((out_channels/split) * (img_side ** 2)), int(((img_side ** 2)/split) * connections))

        self.reset_params()

    def forward(self, x, device,
                kura_update_rate,
                anneal,
                episodes,
                initial_phase,
                connectivity,
                record_step,
                test):
        x = x
        osci = km.kura_torch(self.img_side ** 2, device=device)
        osci.set_ep(kura_update_rate)
        osci.phase_init(initial_phase)

        encoder_outs = []

        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)

        x = self.linear(x.reshape(-1, int((self.out_channels / self.split) *
                                          (self.img_side ** 2)))).reshape(-1, self.img_side ** 2, self.connections)

        x = x / x.norm(p=2, dim=2).unsqueeze(2)

        phase_list = osci.evolution(x, connectivity, anneal=anneal,
                                     steps=episodes, initial_state=test, record_step=record_step)
        return phase_list, x

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weights_init(m)


class simple_conv(nn.Module):
    def __init__(self, depth, connections, out_channels, img_side, in_channels, kernel_size, start_filts, split):
        """
        For various image size, feature maps are all in the same shape as input
        """
        super(simple_conv, self).__init__()
        self.connections = connections
        self.img_side = img_side
        self.out_channels = out_channels
        self.split = split
        self.convs1 = []
        self.convs2 = []
        self.depth = depth

        start_filts = int(start_filts / 2)
        for i in range(depth):
            ins = in_channels if i == 0 else outs
            outs = start_filts * (2 ** i)
            conv = nn.Conv2d(ins, outs, kernel_size=kernel_size[0], stride=1, padding=int((kernel_size[0] - 1) / 2))
            self.convs1.append(conv)
            conv = nn.Conv2d(ins, outs, kernel_size=kernel_size[1], stride=1, padding=int((kernel_size[1] - 1) / 2))
            self.convs2.append(conv)

        self.convs1 = nn.ModuleList(self.convs1)
        self.convs2 = nn.ModuleList(self.convs2)

        self.conv_final = nn.Conv2d(outs * 2, out_channels, kernel_size=1, stride=1, padding=0)

        self.linear = nn.Linear(int((self.out_channels / split) * (img_side ** 2)),
                                int(((img_side ** 2) / split) * connections))
        self.reset_params()

    def forward(self, x, device,
                kura_update_rate,
                anneal,
                episodes,
                initial_phase,
                connectivity,
                record_step,
                test):
        osci = km.kura_torch(self.img_side ** 2, device=device)
        osci.set_ep(kura_update_rate)
        osci.phase_init(initial_phase)

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

        x = self.linear(x.reshape(-1, int((self.out_channels / self.split) *
                                          (self.img_side ** 2)))).reshape(-1, self.img_side ** 2, self.connections)

        x = x / x.norm(p=2, dim=2).unsqueeze(2)

        phase_list = osci.evolution(x, connectivity, anneal=anneal,
                                     steps=episodes, initial_state=test, record_step=record_step)
        return phase_list, x

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

    def forward(self, phase_list, mask, device, valid=False):
        # losses will be 1d, with its length = episode length
        if valid:
            losses = \
                ls.exinp_integrate_torch(torch.cat(phase_list, dim=0).detach(),
                                         mask.repeat(len(phase_list), 1, 1).detach(),
                                         device).reshape(len(phase_list), mask.shape[0]).mean(1)
        else:
            losses = \
                ls.exinp_integrate_torch(torch.cat(phase_list, dim=0),
                                         mask.repeat(len(phase_list), 1, 1),
                                         device).reshape(len(phase_list), mask.shape[0]).mean(1)
        return torch.matmul(losses,
                            torch.pow(torch.arange(len(phase_list)) + 1, self.degree).unsqueeze(1).float().to(device))


def count_parameters(model):
    """counting parameter number of the assigned model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_net(name, **kwargs):
    if name == 'simple_conv': return simple_conv(in_channels=kwargs['in_channels'], start_filts=kwargs['start_filts'],
                                                 depth=kwargs['depth'], img_side=kwargs['img_side'],
                                                 connections=kwargs['num_cn'], out_channels=kwargs['out_channels'],
                                                 split=kwargs['split'], kernel_size=kwargs['kernel_size'])
    elif name == 'Unet': return Unet(in_channels=kwargs['in_channels'], start_filts=kwargs['start_filts'],
                                     depth=kwargs['depth'], img_side=kwargs['img_side'],
                                     connections=kwargs['num_cn'], out_channels=kwargs['out_channels'],
                                     split=kwargs['split'], kernel_size=kwargs['kernel_size'])
    else: raise ValueError('Network not included so far')
