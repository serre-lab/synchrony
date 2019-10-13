import torch
import torch.nn as nn
import ipdb
import numpy as np

class small_net(nn.Module):
    def __init__(self, kernel_size, num_features):
        super(net, self).__init__()
        padding = int(np.floor(kernel_size / 2.0))
        self.conv = nn.Conv1d(1, num_features, kernel_size=kernel_size, padding=padding)
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
class big_net(nn.Module):
    def __init__(self, img_side, num_layers, kernel_size):
        super(big_net, self).__init__()
        num_features = [1] + (num_layers + 1) * [4]
        pad_size     = int(np.ceil((kernel_size -1)/ 2.0))
        self.layers  = torch.nn.ModuleList([torch.nn.Conv2d(num_features[i], num_features[i+1], kernel_size, padding=pad_size) for i in range(num_layers)])

    def forward(self,x):
        batch_size = x.shape[0]
        for l, layer in enumerate(self.layers):
            x = layer(x)
            x=torch.sigmoid(x) if l < len(self.layers) - 1 else x
        means = x.mean(1)
        stds  = x.std(1)
        x     = x-means.unsqueeze(1)
        x     = x.view(batch_size, x.shape[1], -1)
        stds  = stds.view(batch_size, -1)
        return torch.einsum('bci,bcj->bcij',x,x).mean(1) / torch.einsum('bi,bj->bij',stds,stds)
           

class net2(nn.Module):
    def __init__(self):
        super(net2, self).__init__()


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight, mean=0., std=.1)
        nn.init.normal_(m.bias, mean=.0, std=.01)


if __name__ == '__main__':
    x = torch.rand(1, 1, 4)
    network = big_net()
    network.parameters()
    corr = network(x)
    print(corr)
