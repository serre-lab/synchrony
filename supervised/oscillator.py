DISPLAY=False
if not DISPLAY:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
else:
    import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import copy
from numpy import linalg as LA
import matplotlib.animation as animation
import os
from make_data import triominos
import ipdb

class oscillator(object):
    """
    oscillation model toolkit
    goals:
    1. define a vector of oscillators with their phases and initial frequencies
    2. define a update policy, default is mean field kuramoto dynamics
    2. available using other coupling matrix
    3. available using other dynamics
    params:
    epsilon: update rate
    N: oscillator number, an int number
    K: mean field factor, a number
    coupling_mats: an n-d array, define coupling matrix
    initial_frq: an array, define initial frequency
    initial_phase: an array as its name
    classes: designed for frequency update using method in meir's paper. an int number showing classes number
    dynamic func in self.update: dynamic for only one oscillator
    """
    def __init__(self, images, sameness_dicts, epsilon=0.01, K=None, img_side=None, coupling_mats=None,
                 initial_frq=None, initial_phase=None, classes=None):
        # important parameters
        # coupling_mats: shape=[N, N](should be symmetric in kuramoto)
        home_dir = os.path.expanduser('~')
        self.save_dir = os.path.join(home_dir, 'oscillators')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # parameters for kuramoto
        if epsilon:
            self.ep = epsilon
        if K:
            self.K = K
        if img_side:
            self.img_side = img_side
            N = img_side**2
            self.N = N
        self.classes = classes
        self.batch_size = images.shape[0]

        # image parameters
        self.image = images
        self.sds             = sameness_dicts
        if classes:
            # prepare initial alpha for frequency update
            #self.freq_alpha = np.array([np.random.randint(low=1, high=classes + 1) for _ in range(N)])
            self.freq_alpha = torch.randint(1,classes+1, (self.batch_size,N,)).float()
            self.in_frq = self.freq_alpha * 0.1
        else:
            # intrinsic frequency
            if id(initial_frq) != id(None):
                self.in_frq = initial_frq
            else:
                #self.in_frq = np.random.normal(0, 1, N)
                self.in_frq = torch.normal(torch.zeros(self.batch_size, N),torch.ones(self.batch_size, N))
        if id(coupling_mats) != id(None):
            self.c_mats = coupling_mats.float()
        else:
            #self.c_mat = np.ones((N, N))
            self.c_mat = torch.ones((N,N))
        if id(initial_phase) != id(None):
            self.phase = Variable(initial_phase, requrires_grad=True)
        else:
            #self.phase = np.random.rand(N) * 2 * np.pi
            self.phase = Variable(2* np.pi*torch.rand(self.batch_size, N), requires_grad=True)
        # real frequency to measure if phase locking
        self.freq = self.in_frq

    def kura_dy(self):
        # kuramoto dynamic phase update for each oscillator
        # save sum of 'sin' difference
        #sin_sum = 0
        diffs = self.phase.unsqueeze(1) - self.phase.unsqueeze(2)
        delta = self.in_frq + (self.K * .5) * (self.c_mats * torch.sin(diffs)).mean(2)
        self.phase = (self.phase + self.ep * delta) % (2 * np.pi)
        self.freq = delta
        #for idx_n in range(self.N):
        #    # exclude itself
        #    if idx_n == idx_m:
        #        continue
        #    else:
        #        # sum up all 'sin' differences
        #        sin_sum += self.c_mat[idx_m, idx_n] * np.sin(phase_copy[idx_n] - phase_copy[idx_m])
        # update phase
        #delta = self.ep * (self.in_frq[idx_m] + self.K * sin_sum / self.N)
        #self.phase[idx_m] += delta
        # print(self.in_frq[idx_m], self.K * sin_sum / self.N)
        # print(self.ep * (self.in_frq[idx_m] + self.K * sin_sum / self.N))
        # scale phases to (0, 2*pi)
        #self.phase[idx_m] = self.phase[idx_m] % (2 * np.pi)
        #self.freq[idx_m] = delta

    def coherence(self):
        # compute coherence order
        # return a float32 number

        if self.classes == None:
            #real = sum(np.cos(self.phase))
            real=sum(torch.cos(self.phase))
            #imagnery = sum(np.sin(self.phase))
            imaginary = sum(torch.sin(self.phase))
            return (1 / self.N) * (real**2 + imaginary**2)
            #return LA.norm([real, imagnery]) / self.N
        else:
            r = []
            for alpha in range(1, self.classes + 1):
                group = [i for i in range(self.N) if self.freq_alpha[i] == alpha]
                # print(group)
                if len(group) != 0:
                    #real = sum([np.cos(self.phase[idx]) for idx in group])
                    real = sum([torch.cos(self.phase[idx]) for idx in group])
                    #imagnery = sum([np.sin(self.phase[idx]) for idx in group])
                    imaginary = sum([torch.sin(self.phase[idx]) for idx in group])
                    #r.append(LA.norm([real, imagnery]))
                    r.append(real**2 + imag**2)
            return round(sum(r) / self.N, 4)

    def update(self, dynamic=None, *args):
        # *args could be other parameters in specific dynamic rule
        if dynamic:
            self.phase = dynamic(*args)
        else:
            self.kura_dy()
        #phase_copy = copy.deepcopy(self.phase)
        #for idx in range(self.N):
        #    if dynamic:
        #        # assume new dynamics only depend on current frequency and phase
        #        new_phase = dynamic(idx, self.in_frq, phase_copy, self.c_mat, *args)
        #        self.phase[idx] = new_phase
        #    else:
        #        self.kura_dy(idx, phase_copy)
        if self.classes:
            self.freq_update()
            self.in_frq = self.freq_alpha * 0.1
        # print(self.freq_alpha)

    def freq_update(self):
        # support from group alpha
        #support = np.zeros(self.classes)
        #support = torch.zeros((self.batch_size, self.N, self.classes))
        support = []
        phase_diffs = torch.cos(self.phase.unsqueeze(1) - self.phase.unsqueeze(2))
        for alpha in range(1, self.classes + 1):
            group_mask = self.freq_alpha == alpha
            coupling_mask = torch.einsum('bi,bj->bij', group_mask, group_mask).float()
            support.append(.5*((self.c_mats * (phase_diffs + 1)) * coupling_mask).sum(1))
            
        self.freq_alpha = (torch.argmax(torch.stack(support),dim=0) + 1).float()
    def synch_loss(self, oscillators):
        #diffs = np.expand_dims(oscillators,0) - np.expand_dims(oscillators,1)
        diffs = oscillators.unsqueeze(0) - oscillators.unsqueeze(1)
        #return torch.abs(torch.sin(.5*diffs)).mean()
        return .5*torch.where(diffs < np.pi, torch.abs(diffs), 2*np.pi - torch.abs(diffs)).mean()
    def desynch_loss(self, unit_vectors):
        loss = 0
        for i, v1 in enumerate(unit_vectors):
            for j, v2 in enumerate(unit_vectors):
                if i==j: continue
                #loss+=np.abs(v1[0]*v2[0] + v1[1]*v2[1])**2
                loss+=torch.abs(v1[0]*v2[0] + v1[1]*v2[1])**2
        return loss
    def seg_loss(self):
        sl = 0
        dl = 0
        for b in range(self.batch_size):
            unit_vectors = []
            sd = self.sds[b]
            phs = self.phase[b,:]
            for key in sd.keys():
                group_coords = np.ravel_multi_index(sd[key], (self.img_side,self.img_side)) 
                group_oscillators = phs[group_coords]
                sl += self.synch_loss(group_oscillators)
                mean_vector = [torch.cos(group_oscillators).mean(), torch.sin(group_oscillators).mean()]
                norm = torch.sqrt(mean_vector[0]**2 + mean_vector[1]**2) + 1e-5
                unit_vectors.append([mean_vector[0]/norm, mean_vector[1]/norm])

            dl+=self.desynch_loss(unit_vectors)
        return (sl+dl) / self.batch_size, sl / self.batch_size, dl / self.batch_size


class run_osci:
    """
    goal:
    1. run a oscillation process
    2. visualize evolution with coherence shown in each step
    requirment:
    oscillator object
    """
    def __init__(self, oscillator, episodes, save=False):
        self.osci = oscillator
        self.episodes = episodes
        self.save  = save
        self.total_loss = []
        self.synch_loss = []
        self.desynch_loss = []

        home_dir = os.path.expanduser('~')
        self.save_dir = os.path.join(home_dir, 'oscillators')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def _convert2square(self, nrows, ncols):
        return self.osci.phase.reshape((nrows, ncols))

    def _update(self, d, sc1, sc2, im, ti, tx, nrows, ncols, dynamic=None, loss_every=10):
        self.osci.update(dynamic=dynamic)
        if d%loss_every==0:
            total, sl, dl = self.osci.seg_loss()
            self.total_loss.append(total) 
            self.synch_loss.append(sl) 
            self.desynch_loss.append(dl) 
            print('Loss at step {}: {}'.format(d,total))
            # Loss fig
            plt.plot(self.total_loss)
            plt.plot(self.synch_loss)
            plt.plot(self.desynch_loss)
            plt.ylim([0,np.max(self.total_loss)])
            plt.title('Losses')
            plt.legend(('Total', 'Synch', 'Desynch'))
            plt.savefig(os.path.join(self.save_dir, 'loss.png'))
            plt.close()

        # osci.update()
        data1 = np.concatenate((self.osci.phase.reshape(-1, 1),
                               self.osci.freq_alpha.reshape(-1, 1)), axis=1)
        data2 = np.concatenate((self.osci.phase.reshape(-1, 1),
                               self.osci.freq.reshape(-1, 1)), axis=1)
        sc1.set_offsets(data1)
        sc1.set_cmap('hsv')
        sc1.set_color(sc1.to_rgba(self.osci.phase))

        sc2.set_offsets(data2)
        sc2.set_cmap('hsv')
        sc2.set_color(sc1.to_rgba(self.osci.phase))

        ti.set_text('coherence: ' + str(self.osci.coherence()))
        im.set_array(self._convert2square(nrows, ncols))
        s = 'steps: ' + str(d)
        tx.set_text(s)
        return im, ti,

    def show_evol(self, nrows, ncols, time_interval=10, dynamic=None):
      
        fig = plt.figure(figsize=(6.5, 6.5))
        ax1 = fig.add_subplot(224, polar=True)
        sc1 = ax1.scatter(self.osci.phase,
                          self.osci.freq_alpha,
                          c=self.osci.phase,
                          vmin=0, vmax=2 * np.pi,
                          cmap='hsv')
        plt.title('frequency alpha', fontsize='x-small', verticalalignment='bottom')
        plt.setp(ax1.xaxis.get_majorticklabels(), va="top", rotation_mode="anchor", fontsize=8)

        ax2 = fig.add_subplot(223, polar=True)
        ax2.set_ylim([0, 0.4])
        sc2 = ax2.scatter(self.osci.phase,
                          np.abs(self.osci.freq),
                          c=self.osci.phase,
                          vmin=0, vmax=2 * np.pi,
                          cmap='hsv')
        ax2.set_yticks(np.arange(1, 5) * 0.1)
        ax2.set_yticklabels(('0.1', '0.2', '0.3', '0.4'))
        plt.title('real frequency', fontsize='x-small', verticalalignment='bottom')
        plt.setp(ax2.xaxis.get_majorticklabels(), va="top", rotation_mode="anchor", fontsize=8)

        ax3 = fig.add_subplot(222)
        im1 = ax3.imshow(self._convert2square(nrows, ncols), cmap='hsv', animated=True)
        ti3 = plt.title('coherence: ' + str(self.osci.coherence()), fontsize='small')
        cbar = fig.colorbar(im1, ax=ax3,
                     ticks=[np.pi / 2, np.pi, np.pi * 3 / 2],
                     fraction=0.05)
        cbar.ax.set_yticklabels(('0.5$\pi$', '$\pi$', '1.5$\pi$'))

        ax3.axis('off')

        ax4 = fig.add_subplot(221)
        #area = triominos()
        area = self.osci.images[0]
        im2 = ax4.imshow(area)
        ti4 = plt.title('original image', fontsize='small')
        ax4.axis('off')
        
        s = 'steps: 0'
        tx = fig.suptitle(s, verticalalignment='bottom', fontsize=10)

        plt.tight_layout(pad=0, w_pad=0.1, h_pad=1.0)
        ani = animation.FuncAnimation(
            fig, self._update, interval=time_interval,
            frames=self.episodes, repeat=False, fargs=(sc1, sc2, im1, ti3, tx, nrows, ncols, dynamic))
        plt.show()
        plt.close()
        if self.save:
            print('Saving animation.')
            ani.save(os.path.join(self.save_dir, 'oscillation.gif'), writer='imagemagick', fps=24)
            print('Done')
        plt.close()
