import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from torchvision.utils import save_image
import numpy as np
import torch
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
import sys


def plot_evolution(recon_batch,data,epoch,args):
    n = min(data.size(0), 5)
    recon_batch = torch.cat(recon_batch,dim=1)[:n].view(n,50,28,28)
    comparison = torch.cat([data[:n].view(n,50,28,28),recon_batch],dim=1).view(-1,1,28,28)
    save_dir = os.path.join(args.save_dir, 'evolution_' + str(epoch) + '.png')
    save_image(comparison.cpu(), save_dir, nrow=50)

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
    tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                          n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='random',
                          verbose=0, random_state=None, method='barnes_hut', angle=0.5, n_jobs=None)
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