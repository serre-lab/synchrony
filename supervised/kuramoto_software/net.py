import torch
import torch.nn as nn
import numpy as np


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


class net_linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(net_linear, self).__init__()
        self.fc = torch.nn.Linear(in_features, out_features)

    def forward(self, input):
        # return self.fc(input) * torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
        return self.fc(input)


def weights_init(m):
    if isinstance(m, nn.Conv1d) | isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0., std=.1)
        nn.init.normal_(m.bias, mean=.0, std=.01)
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, 0., .1)
        nn.init.uniform_(m.bias, 0., .5)


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
