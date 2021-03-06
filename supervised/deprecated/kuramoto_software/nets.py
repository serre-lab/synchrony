import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import toeplitz
from skimage.filters import gabor_kernel as gk
from itertools import product
import ipdb

def gabor_filters(num_orientations=8):
    gabor_list = [gk(frequency=.3, theta=k*2.*np.pi / num_orientations, sigma_x=2, sigma_y=2, n_stds=1).real for k in range(num_orientations)]
    m = np.max([gabor.shape for gabor in gabor_list])
    for g, gabor in enumerate(gabor_list):
        if gabor.shape[0] == m: continue
        else:
            gabor_list[g] = np.pad(gabor, m-gabor.shape[0], m-gabor.shape[1])
    return gabor_list
        

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
    def __init__(self, img_side, num_conv_layers, kernel_side=5, num_conv_features=16, return_coupling=False, normalize_output=True, out_kernel_side=None, pretrained=False):
        super(big_net, self).__init__()
        self.img_side = img_side

        if pretrained:
            assert num_conv_layers==1

        self.return_coupling = return_coupling
        self.normalize_output = normalize_output
        self.kernel_mask = self.make_kernel_mask(out_kernel_side)
        num_conv_features = [1] + (num_conv_layers + 1) * [num_conv_features]
        minus_one = 0 if kernel_side % 2 == 0 else 1
        pad_size = int((kernel_side - minus_one) / 2.0)
        self.conv_layers = torch.nn.ModuleList([torch.nn.Conv2d(num_conv_features[i], num_conv_features[i+1], kernel_side, padding=pad_size) for i in range(num_conv_layers)])
        if pretrained:
            self.conv_layers[0].weight.data = torch.tensor(gabor_filters(num_conv_features[1])).unsqueeze(1).float()
            self.conv_layers[0].requires_grad = False


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
        for l, layer in enumerate(self.conv_layers):
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

class deep_net(nn.Module):
    def __init__(self, img_side, num_conv_layers, num_fc_layers, kernel_side=5, num_conv_features=16, num_fc_features=128, pretrained=False, bias=True):
        super(deep_net, self).__init__()

        if pretrained:
            assert num_conv_layers==1
            assert kernel_side==5
        self.img_side = img_side

        num_conv_features = [1] + (num_conv_layers + 1) * [num_conv_features]
        num_fc_features = [self.img_side**2 * num_conv_features[-1]] + (num_fc_layers) * [num_fc_features] + [self.img_side**4]
        pad_size = int((kernel_side / 2.0))
        self.conv_layers = torch.nn.ModuleList([torch.nn.Conv2d(num_conv_features[i], num_conv_features[i+1], kernel_side, padding=pad_size, bias=bias) for i in range(num_conv_layers)])
        if pretrained:
            self.conv_layers[0].weight.data = torch.tensor(gabor_filters(num_conv_features[1])).unsqueeze(1).float()
            self.conv_layers[0].requires_grad = False
            
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(num_fc_features[i], num_fc_features[i+1]) for i in range(num_fc_layers + 1)])
 
    def forward(self, x):
        batch_size = x.shape[0]
        for layer in self.conv_layers:
            x = layer(x)
            x = torch.relu(x)
        x = x.reshape(batch_size, -1)
        for l, layer in enumerate(self.fc_layers):
            x = layer(x)
            x = torch.relu(x) if l < len(self.fc_layers) - 1 else x
        return x.reshape(batch_size, self.img_side**2, self.img_side**2)

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


class simple_conv(nn.Module):
    def __init__(self):
        super(simple_conv, self).__init__()
        self.conv1_k3 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2_k3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv3_k3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv1_k5 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2_k5 = nn.Conv2d(32, 16, kernel_size=5, padding=2)
        self.conv3_k5 = nn.Conv2d(16, 8, kernel_size=5, padding=2)
        self.fc = nn.Linear(1024, 16384)

    def forward(self, input):
        conv1_k3 = self.conv1_k3(input.reshape(-1, 1, 16, 16))
        conv1_k5 = self.conv1_k5(input.reshape(-1, 1, 16, 16))
        conv2_k3 = self.conv2_k3(conv1_k3)
        conv2_k5 = self.conv2_k5(conv1_k5)
        conv3_k3 = self.conv3_k3(conv2_k3)
        conv3_k5 = self.conv3_k5(conv2_k5)
        # conv.shape=(batch, channels, H, W)
        conv = torch.empty(input.shape[0], 16, 16, 16).to(device=input.device)
        for c in range(8):
            conv[:, c * 2, ...] = conv3_k3[:, c, ...]
            conv[:, c * 2 + 1, ...] = conv3_k5[:, c, ...]
        fc = self.fc(torch.sigmoid(conv.reshape(-1, 1024)))
        return fc.reshape(-1, 256, 256)

class simple_meta_net(nn.Module):
    def __init__(self, input_shape, batch_size, num_conv_layers=1, num_fc_layers=1, num_conv_features=[32], num_fc_features=[], conv_kernel_sides=[5], bias=True, coupling_params={}):
        super(simple_meta_net, self).__init__()

        # OUTPUT PARAMETERS
        if coupling_params == {}:
            coupling_params['num_layers'] = 2
            coupling_params['vertical'] = {}
            coupling_params['horizontal'] = {}
            coupling_params['vertical']['stride'] = [(2,2)]
            coupling_params['vertical']['num_features'] = [4]
            coupling_params['vertical']['kernel_side'] = [3]
            coupling_params['horizontal']['stride'] = [(2,2), (2,2)]
            coupling_params['horizontal']['kernel_side'] = [3,3]
        self.coupling_params = coupling_params
        self.num_layers = coupling_params['num_layers']
        self.v_stride = coupling_params['vertical']['stride']
        self.v_num_features = coupling_params['vertical']['num_features']
        self.v_kernel_side  = coupling_params['vertical']['kernel_side']
        self.h_stride = coupling_params['horizontal']['stride']
        self.h_kernel_side  = coupling_params['horizontal']['kernel_side']
        
        self.shape_dict = {}
        self.batch_size = batch_size
        layer_side = input_shape[1]
        self.shape_dict['layer_shape'] = [input_shape]
        for l in range(self.num_layers - 1):
            k = self.v_kernel_side[l]
            layer_side = int(np.floor( (layer_side - k + 1) / float(self.v_stride[0][0])))
            self.shape_dict['layer_shape'].append((self.v_num_features[l], layer_side, layer_side))

        # Vertical couplings
        total_couplings = 0
        for l in range(self.num_layers - 1):
            key = 'coupling_shape_{}{}'.format(l, l+1) 
            k = self.v_kernel_side[l]
            sb = self.shape_dict['layer_shape'][l]
            st = self.shape_dict['layer_shape'][l+1]
            # Vertical couplings are shaped like num_units_l+1 X num_units_in_RF
            fan_out = st[0]*st[1]*st[2] 
            fan_in  = sb[0]*k**2
            self.shape_dict[key] = (fan_out, fan_in) 
            total_couplings += fan_out * fan_in

        # Horizontal couplings
        for l in range(self.num_layers):
            key = 'coupling_shape_{}{}'.format(l,l)
            k = self.h_kernel_side[l]
            s = self.shape_dict['layer_shape'][l]
            fan_out = s[0]*s[1]*s[2]
            fan_in = s[0]*k**2
            self.shape_dict[key] = (fan_out, fan_in)
            total_couplings += fan_out * fan_in
        self.total_couplings = total_couplings

        # METANET PARAMETERS
        num_conv_features = [1] + num_conv_features
        num_fc_features = [input_shape[1]*input_shape[2] * num_conv_features[-1]] + num_fc_features + [self.total_couplings]
        pad_sizes = [int((k / 2.0)) for k in conv_kernel_sides]
        self.conv_layers = torch.nn.ModuleList([torch.nn.Conv2d(num_conv_features[i], num_conv_features[i+1], conv_kernel_sides[i], padding=pad_sizes[i], bias=bias) for i in range(num_conv_layers)])
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(num_fc_features[i], num_fc_features[i+1]) for i in range(num_fc_layers)])

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
            x = torch.relu(x)
        x = x.reshape(self.batch_size, -1)

        for l, layer in enumerate(self.fc_layers):
            x = layer(x)
            x = torch.relu(x) if l < len(self.fc_layers) - 1 else x

        x = x.reshape(self.batch_size,-1)
        # Vertical couplings
        last_coupling_num = 0
        coupling_dict = {}
        for (i,j) in list(product(range(self.num_layers), range(self.num_layers))):
            if j < i: continue
            shape_key = 'coupling_shape_{}{}'.format(i,j)
            key = 'couplings_{}{}'.format(i, j)
            coupling_shape = self.shape_dict[shape_key]
            coupling_num = coupling_shape[0]*coupling_shape[1]
            current_couplings =  x[:,last_coupling_num:last_coupling_num + coupling_num]
            coupling_dict[key] = current_couplings.reshape(-1, coupling_shape[0], coupling_shape[1])
            last_coupling_num += coupling_num
        return coupling_dict

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
    my_net = simple_meta_net([1,16,16],32)
    x = torch.rand([32,1,16,16])
    out = my_net.forward(x)
    ipdb.set_trace()
    #x = torch.rand(1, 1, 4)
    #network = net()
    #network.parameters()
    #corr = network(x)
    print(corr)
