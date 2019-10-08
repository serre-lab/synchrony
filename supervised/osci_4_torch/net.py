import torch
import torch.nn as nn


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


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight, mean=0., std=.1)
        nn.init.normal_(m.bias, mean=.0, std=.01)


if __name__ == '__main__':
    x = torch.rand(1, 1, 4)
    network = net()
    network.parameters()
    corr = network(x)
    print(corr)
