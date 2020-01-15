import os
import torch as tc
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import permutations


"""
1. Read a batch of data (specified by a list of indices) through numpy.load
2. Display real-time performance through kura_visual
3. Generate connectivity, 2 modes:
    1) local (sw=False, p_rewire=0)
    2) small-world (sw=True, p_rewire>0)
   Add hierarchical (num_global_control>0 and ==n^2 and (img_side%~)==0)
"""


def read_data(data_inds, path, img_side, group_size, device='cuda', valid=False):
    images = np.zeros((len(data_inds), img_side, img_side))
    masks = np.zeros((len(data_inds), group_size, img_side ** 2))
    b = 0
    for data_ind in data_inds:
        data = np.load(os.path.join(path, 'img_' + str(data_ind).rjust(4, '0') + '.npy'))
        image = data[0, ...].reshape(1, img_side, img_side)
        mask = data[1:, ...].reshape(1, group_size, img_side ** 2)

        images[b, ...] = image
        masks[b, ...] = mask
        b += 1
    if not valid:
        return tc.tensor(images).float().to(device), tc.tensor(masks).float().to(device)
    else:
        return tc.tensor(images).float().detach().to(device), tc.tensor(masks).float().detach().to(device)


def display(displayer, phase_list, images, masks, coupling, omega, img_side, group_size, path, name, rp_field):
    # randomly select images to display
    ind = np.random.randint(images.shape[0])
    image = images[ind].cpu().data.numpy()
    mask = masks[ind].cpu().unsqueeze(0).data.numpy()
    coupling = coupling[ind].cpu().data.numpy()
    omega = omega[ind].cpu().data.numpy()
    np_phase_list = np.array([phase.cpu().data.numpy()[ind, :] for phase in phase_list])

    colored_mask = (np.expand_dims(np.expand_dims(np.arange(group_size), axis=0), axis=-1) * mask / group_size).sum(1)
    displayer.set_phases(np_phase_list)
    displayer.set_masks(mask)
    if rp_field == 'arange':
        kura_param_show(coupling, omega, img_side, path, name)

    if len(phase_list) > 4:
        displayer.static_evol(img_side, img_side, image, path + '/static_' + name, colored_mask)
        displayer.phase_evol2(path + '/phase_' + name)
    else:
        displayer.static_evol2(img_side, img_side, image, path + '/static_' + name, colored_mask)
    return True


def kura_param_show(coupling, omega, img_side, path, name):

    # Coupling
    if coupling.shape[0] != img_side ** 2:
        num_global = coupling.shape[0] - img_side ** 2
        s = int(np.sqrt(num_global))

        # first layer
        im = plt.imshow(coupling[:-num_global, :-num_global].reshape(img_side ** 2, img_side ** 2))
        plt.colorbar(im)
        plt.gca().grid(False)
        plt.axis('off')
        plt.savefig(path + '/first_layer' + name + '.png')
        plt.close()

        # second layer
        im = plt.imshow(coupling[-num_global:, -num_global:].reshape(num_global, num_global))
        plt.colorbar(im)
        plt.gca().grid(False)
        plt.axis('off')
        plt.savefig(path + '/second_layer' + name + '.png')
        plt.close()

        # top down
        top_down = coupling[:-num_global, -num_global:].sum(1)
        top_down = np.concatenate(list(map(lambda a: np.hstack(a),
                                           np.split(top_down.reshape(num_global, int(img_side/s),
                                                                     int(img_side/s)), s, axis=0))), axis=0)
        im = plt.imshow(top_down)
        plt.colorbar(im)
        plt.gca().grid(False)
        plt.axis('off')
        plt.savefig(path + '/top_down' + name + '.png')
        plt.close()

        # bottom up
        bottom_up = coupling[-num_global:, :-num_global].sum(0)
        bottom_up = np.concatenate(list(map(lambda a: np.hstack(a),
                                            np.split(bottom_up.reshape(num_global, int(img_side / s),
                                                                       int(img_side / s)), s, axis=0))), axis=0)
        im = plt.imshow(bottom_up)
        plt.colorbar(im)
        plt.gca().grid(False)
        plt.axis('off')
        plt.savefig(path + '/bottom_up' + name + '.png')
        plt.close()
    else:
        im = plt.imshow(coupling.reshape(img_side ** 2, img_side ** 2))
        plt.colorbar(im)
        plt.gca().grid(False)
        plt.axis('off')
        plt.savefig(path + '/coupling' + name + '.png')
        plt.close()

    # Frequencies
    
    layer1_omega = omega[:img_side**2].reshape(img_side, img_side)
    layer2_omega = omega[img_side**2:]
    layer_name = ('layer_1', 'layer_2')
    for o, omegas in enumerate([layer1_omega, layer2_omega]):
        im = plt.imshow(layer1_omega)
        plt.title('{} Intrinsic Frequencies'.format(names[o]))
        plt. gca().grid(False)
        plt.axis('off')
        plt.colorbar(im)
        plt.savefig(path + '{}_frequencies_{}'.format(layer_name[o], name))
        plt.close()
    
    return True


def generate_connectivity(num_cn, img_side,
                          sw=False, num_global_control=0,
                          p_rewire=0.5, rp_field='arange'):
    # Generate local coupling
    if num_global_control > 0:
        s = np.sqrt(num_global_control)
        if (int(s) - s) != 0.:
            raise ValueError('Number of global oscillator should be a square number')
        else:
            s = int(s)
        if (img_side % s) != 0.:
            raise ValueError('Number of global oscillator should be a dividend of img_side')

        connectivity = np.zeros((img_side ** 2, num_cn + 1))
        seq = np.arange(img_side ** 2)
        if rp_field == 'random':
            np.random.shuffle(seq)
        elif rp_field == 'arange':
            seq = np.concatenate(np.split(np.stack(np.split(seq.reshape(img_side, img_side), s, axis=1),
                                                   axis=0), s, axis=1), axis=0).reshape(-1)
        else:
            raise ValueError('Receptive field type not understood')
    else:
        connectivity = np.zeros((img_side ** 2, num_cn))

    # Generate local connectivity
    for i in tqdm(range(img_side ** 2)):
        count = 0
        x_1 = int(i % img_side)
        y_1 = int(i // img_side)
        connectivity[i, :num_cn] += get_cn(num_cn, (x_1, y_1), img_side, sw)

        # rewire when small world
        if sw:
            change = sorted(range(num_cn), key=lambda k: np.random.random())[:int(num_cn*p_rewire)]
            connectivity[i, ...][change] = \
                sorted(np.delete(range(img_side ** 2), connectivity[i, :-int(num_global_control > 1)]),
                       key=lambda k: np.random.random())[:int(num_cn*p_rewire)]
        # add one when hierarchy
        if num_global_control > 0:
            connectivity[i, -1] = np.argwhere(seq == i)[0][0] // int(img_side ** 2 / num_global_control) + img_side ** 2

    if num_global_control > 0:
        global_connectivity = np.stack(np.split(seq, num_global_control, axis=0), axis=0)
        inner = np.tile(np.expand_dims(np.arange(num_global_control), axis=0),
                        (num_global_control, 1))[np.where(np.eye(num_global_control) == 0)].reshape(num_global_control,
                                                                                                    -1) + img_side ** 2
        global_connectivity = np.concatenate([global_connectivity, inner], axis=1)
    else:
        global_connectivity = None
    return connectivity, global_connectivity


def critic_dist(num_cn):
    if num_cn <= 0:
        raise ValueError("Number of connected neighbor should be positive")
    elif num_cn <= 8:
        return 2
    elif num_cn <= 24:
        return 3
    elif num_cn <= 45:
        return 4
    elif num_cn <= 68:
        return 5
    else:
        raise ValueError("Number is not included, try smaller one")


def get_cn(num_cn, coord, img_side, sw):
    dist = critic_dist(num_cn)
    dist_list = np.concatenate([np.tile(np.arange(1, dist).reshape(dist - 1, -1), 2),
                                np.stack(list(permutations(np.arange(dist).tolist(), 2)), axis=0)], axis=0)
    dist_list = np.concatenate([dist_list, np.expand_dims((dist_list ** 2).sum(1), axis=1)], axis=1)
    dist_list = np.sort(dist_list.view('i8, i8, i8'), order=['f2'], axis=0).view(np.int).tolist()
    cn = []
    for p in dist_list:
        if 0 in p:
            direction_list = [(p[0], p[1]), (-p[0], -p[1])]
        else:
            direction_list = [(p[0], p[1]), (-p[0], -p[1]), (-p[0], p[1]), (p[0], -p[1])]
        for d in direction_list:
            if len(cn) < num_cn:
                new_coord = np.array(d) + np.array(coord)
                if (new_coord[0] < img_side) & (new_coord[0] >= 0) & (new_coord[1] < img_side) & (new_coord[1] >= 0):
                    cn.append(new_coord[0] + new_coord[1] * img_side)
            else:
                break
    cn = np.array(cn)
    if sw:
        extra = sorted(np.delete(range(img_side ** 2), cn), key=lambda k: np.random.random())[:(num_cn - cn.shape[0])]
    else:
        extra = np.ones((num_cn - cn.shape[0])) * (coord[0] + coord[1] * img_side)
    return np.concatenate([cn, extra], axis=0).astype(np.int)
