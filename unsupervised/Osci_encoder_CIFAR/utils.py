import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from tqdm import tqdm
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import permutations

sns.set(style="whitegrid")
import sys


def plot_evolution(recon_batch,data,epoch,args):
    n = min(data.size(0), 10)
    recon_batch = torch.cat(recon_batch,dim=1)[:n].view(n,20,28,28)
    comparison = torch.cat([recon_batch,data[:n]],dim=1).view(-1,1,28,28)
    save_dir = os.path.join(args.save_dir, 'evolution_' + str(epoch) + '.png')
    save_image(comparison.cpu(), save_dir, nrow=20+1)

def phase_evol(phase_list,data,epoch,args,masks=None, save_name=None):
    plt.style.use('seaborn-darkgrid')
    if masks is not None:
        group_num = self.masks.shape[1]
        annot_bool = [True] * group_num
        colors = {}
        for num in range(group_num):
            mean_angle = np.arctan2((np.sin(self.phases[-1]) * self.masks[0][num]).sum(),
                                    (np.cos(self.phases[-1]) * self.masks[0][num]).sum() + 1e-6)
            colors[num] = (mean_angle + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
        for i in range(self.phases.shape[1]):
            for num in range(group_num):
                if self.masks[0][num][i]:
                    plt.plot(self.phases[:, i], color=plt.cm.hsv(colors[num]), alpha=0.5)
                    if annot_bool[num]:
                        plt.text(self.phases.shape[0] + 0.01, self.phases[-1, i], 'group{}'.format(num),
                                 horizontalalignment='left', size='x-small', color=plt.cm.hsv(colors[num]))
                        annot_bool[num] = False
        plt.xlabel('Time')
        plt.ylabel('Phase')
        plt.title('Phase Evolution')
    else:
        phases = torch.stack(phase_list,2)[0].detach().cpu().numpy()
        for i in range(phases.shape[0]):
            plt.plot(torch.linspace(0., 100, 20),phases[i])
        plt.xlabel('Time')
        plt.ylabel('Phase')
        plt.title('Phase Evolution')
    plt.tight_layout()
    save_name = os.path.join(args.save_dir, 'Phase_evolution_' + str(epoch) + '.png')
    if save_name is not None:
        plt.savefig(save_name + '.png')
    else:
        plt.show()
    plt.close()


def PCA_phase(phase_path, targets_path):
    phases = np.load(phase_path)
    targets = np.load(targets_path)+1
    pca = PCA(n_components=2)
    comp = pca.fit_transform(phases)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    sns.set_style("white")
    current_palette = sns.color_palette("hls", 10)
    f, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.despine(f, left=True, bottom=True)
    sns.scatterplot(x=comp[:2000,0], y=comp[:2000,1],palette=current_palette,
                    hue=targets[:2000],alpha=.6, ax=ax)

def tSNE_phase(phase_path, targets_path):
    phases = np.load(phase_path)
    targets = np.load(targets_path)
    classes = dict(zip(range(10),['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']))
    targets = np.array([classes[i] for i in targets])
    tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                          n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='random',
                          verbose=1, random_state=None, method='barnes_hut', angle=0.5, n_jobs=None)
    comp = tsne.fit_transform(phases)
    sns.set_style("white")
    current_palette = sns.color_palette("hls", 10)
    f, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.despine(f, left=True, bottom=True)
    sns.scatterplot(x=comp[:2000, 0], y=comp[:2000, 1], palette=current_palette,
                    hue=targets[:2000], alpha=.6, ax=ax)


def phase_distribution(phase_path, targets_path):
    phases = np.load(phase_path)
    targets = np.load(targets_path)
    f, axes = plt.subplots(5, 5, figsize=(15, 15), sharex=True)
    current_palette = sns.color_palette("hls", 10)
    for number in np.arange(10):
        phase_number = []
        print(number)
        for i, phase in enumerate(phases):
            if targets[i]==number:
                phase_number.append(phase%(2*np.pi))
        phase_number = np.stack(phase_number)
        for i in range(phase_number.shape[1]):
            axes[i % 5, i // 5].set_xlim(0,2*np.pi)
            if i == 24:
                sns.distplot(phase_number[:,i], hist=False, rug=False, color=current_palette[number], ax=axes[i%5,i//5],label=number)
            else:
                sns.distplot(phase_number[:,i], hist=False, rug=False, color=current_palette[number], ax=axes[i%5,i//5])
    axes[4,4].legend()
    plt.show()

#tSNE_phase('/media/data_cifs/mchalvid/Project_synchrony/MNIST/results/Osci_MNIST_debug/phases_5.npy',
               #'/media/data_cifs/mchalvid/Project_synchrony/MNIST/results/Osci_MNIST_debug/targets_5.npy')

def generate_connectivity(num_cn, img_side,
                          sw=False, num_global_control=0,
                          p_rewire=0.5, rf_type='arange', all_to_all=False):
    if all_to_all == True:
        connectivity = np.ones((img_side, num_cn))
        global_connectivity = None
    else:
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
            if rf_type == 'random':
                np.random.shuffle(seq)
            elif rf_type == 'arange':
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
                change = sorted(range(num_cn), key=lambda k: np.random.random())[:int(num_cn * p_rewire)]
                connectivity[i, ...][change] = \
                    sorted(np.delete(range(img_side ** 2), connectivity[i, :-int(num_global_control > 1)]),
                           key=lambda k: np.random.random())[:int(num_cn * p_rewire)]
            # add one when hierarchy
            if num_global_control > 0:
                connectivity[i, -1] = np.argwhere(seq == i)[0][0] // int(
                    img_side ** 2 / num_global_control) + img_side ** 2

        if num_global_control > 0:
            global_connectivity = np.stack(np.split(seq, num_global_control, axis=0), axis=0)
            inner = np.tile(np.expand_dims(np.arange(num_global_control), axis=0),
                            (num_global_control, 1))[np.where(np.eye(num_global_control) == 0)].reshape(
                num_global_control,
                -1) + img_side ** 2
            global_connectivity = np.concatenate([global_connectivity, inner], axis=1)
        else:
            global_connectivity = None
    return connectivity, global_connectivity

def get_cn(num_cn, coord, img_side, sw):
    dist = critic_dist(num_cn)
    dist_list = np.concatenate([np.tile(np.arange(1, dist).reshape(dist - 1, -1), 2),
                                np.stack(list(permutations(np.arange(dist).tolist(), 2)), axis=0)], axis=0)
    dist_list = np.concatenate([dist_list, np.expand_dims((dist_list ** 2).sum(1), axis=1)], axis=1)
    # dist_list = np.sort(dist_list.view('i8, i8, i8'), order=['f2'], axis=0).view(np.int).tolist()
    dist_list = dist_list[dist_list[:, 1].argsort()]
    cn = []
    for p in dist_list:
        if 0 in p:
            direction_list = [(-p[0], -p[1]), (p[0], p[1])]
        else:
            direction_list = [(-p[0], -p[1]), (p[0], -p[1]), (-p[0], p[1]), (p[0], p[1])]
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