import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import toeplitz
import ipdb

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv = nn.Conv1d(1, 4, kernel_size=2, padding=0)
        # conv.shape=(batch, 4, N)

    def forward(self, input):
        output = torch.sigmoid(self.conv(input))

        # avg over channel
        output = output - torch.mean(output, dim=1, keepdim=True)

        # numerator -> covariance
        # do not have to avg as the denominator is not avg too
        # output.shape=(batch, channel, N)
        # !!! pay attention, channel different from TF
        cov = torch.matmul(torch.transpose(output, 1, 2), output)

        # std for every column
        mid = torch.sum(output ** 2, dim=1, keepdim=True)

        # mid.shape=(batch, 1, N)
        std = torch.sqrt(torch.matmul(torch.transpose(mid, 1, 2), mid))

        # denominator -> correlation
        corr = cov / std

        return corr


class net2(nn.Module):
    def __init__(self):
        super(net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, padding=0)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=2, padding=0)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=2, padding=0)

    def forward(self, input, effect=1):
        input = torch.nn.functional.pad(convert2twod(input), pad=(0, 1, 0, 1))

        hidden = torch.nn.functional.pad(self.conv1(input), pad=(0, 1, 0, 1))
        hidden = torch.nn.functional.pad(self.conv2(hidden), pad=(0, 1, 0, 1))
        output = torch.sigmoid(self.conv3(hidden))

        initial_shape = output.shape

        output = output.reshape(initial_shape[0], initial_shape[1], -1)

        # avg over channel
        output = output - torch.mean(output, dim=1, keepdim=True)

        # numerator -> covariance
        # do not have to avg as the denominator is not avg too
        # output.shape=(batch, channel, N)
        # !!! pay attention, channel different from TF
        cov = torch.matmul(torch.transpose(output, 1, 2), output)

        # std for every column
        mid = torch.sum(output ** 2, dim=1, keepdim=True)

        # mid.shape=(batch, 1, N)
        std = torch.sqrt(torch.matmul(torch.transpose(mid, 1, 2), mid))

        # denominator -> correlation
        corr = cov / std

        return corr * effect

class big_net(nn.Module):
    def __init__(self, img_side, num_layers, kernel_size=5, num_features=16, return_coupling=False, normalize_output=True, out_kernel_side=None):
        super(big_net, self).__init__()
        self.img_side = img_side
        self.return_coupling = return_coupling
        self.normalize_output = normalize_output
        self.kernel_mask = self.make_kernel_mask(out_kernel_side)
        num_features = [1] + (num_layers + 1) * [num_features]
        minus_one = 0 if kernel_size % 2 == 0 else 1
        pad_size = int((kernel_size - minus_one) / 2.0)
        self.layers = torch.nn.ModuleList([torch.nn.Conv2d(num_features[i], num_features[i+1], kernel_size, padding=pad_size) for i in range(num_layers)])

    def make_kernel_mask(self, out_kernel_side):
        if out_kernel_side is None:
            return torch.ones((self.img_side**2, self.img_side**2)).float()
        else:
            toeplitz_row = np.zeros((self.img_side**2,))
            for k in range(out_kernel_side - 1):
                toeplitz_row[(k*self.img_side):(k * self.img_side) + out_kernel_side] = 1.0
            return torch.tensor(toeplitz(toeplitz_row).T).float()
                

    def forward(self, x):
        batch_size = x.shape[0]
        for l, layer in enumerate(self.layers):
            x = layer(x)
        if not self.return_coupling: 
           return x.mean(1).view(batch_size, -1)
        else:
            if self.normalize_output:
                means = x.mean(1)
                stds = x.std(1)
                x = x - means.unsqueeze(1)
                x = x.view(batch_size, x.shape[1], -1)
                stds = stds.view(batch_size, -1)
                return torch.einsum('bci, bcj->bcij', x, x).mean(1) / torch.einsum('bi,bj->bij',stds,stds) * self.kernel_mask.to(x.device)
            else:
                x = x.view(batch_size, x.shape[1], -1)
                return torch.einsum('bci, bcj->bcij', x, x).mean(1) * self.kernel_mask.to(x.device)

class net_linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(net_linear, self).__init__()
        self.fc = torch.nn.Linear(in_features, out_features)

    def forward(self, input):
        # return self.fc(input) * torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
        return self.fc(input)


def weights_init(m, w_mean=0.0, w_std=0.1, b_mean=0.0, b_std=0.01):
    if isinstance(m, nn.Conv1d) | isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=w_mean, std=w_std)
        nn.init.normal_(m.bias, mean=b_mean, std=b_std)
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, w_mean, w_std)
        nn.init.uniform_(m.bias, b_mean, b_std)


def convert2twod(img_batch):
    img_side = img_batch.shape[-1]
    output = torch.reshape(torch.tensor(img_batch), [-1, 1, img_side, img_side]).float()
    return output


def mask_pro(group_dict):
    # create masks for training
    masks = []
    group_names = group_dict.keys()
    N = 0
    for group_name in group_names:
        N += len(group_dict[group_name])
    for group_name in group_names:
        group_idxes = group_dict[group_name]
        mask = [1. if i in group_idxes else 0. for i in range(N)]
        masks.append(mask)
    return np.expand_dims(masks, axis=0)


if __name__ == '__main__':
    x = torch.rand(1, 1, 4)
    network = net()
    network.parameters()
    corr = network(x)
    print(corr)
