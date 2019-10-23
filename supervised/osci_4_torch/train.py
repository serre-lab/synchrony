import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')
import torch
from nets import small_net, big_net, weights_init
from osci_tools import matt_loss, coherence_loss, evolution, IP_loss
from make_data_new import generate_small, polyomino_scenes, generate_test_img
import numpy as np
import os
from torch.nn.utils import clip_grad_norm_
import ipdb

class TrainPipeline(object):
    def __init__(self,
                 n=1,
                 num=2,
                 lr=1e-2,
                 report=25,
                 batch_size=16,
                 train_step=100,
                 evolution_step=20,
                 img_side=2):

        # omino type
        self.n = n
        # number of objects
        self.num = num
        # learning rate
        self.lr = lr
        # report steps
        self.report = report
        # batch size
        self.batch = batch_size
        # image size
        self.img_side = img_side
        # save dir
        self.fig_dir = os.path.join(os.path.expanduser('~'), 'oscillators')
        # model dir
        self.data_dir = os.path.join(os.path.expanduser('~'),'osc_models')
        # Min and max number of groups
        self.min_groups = 2
        self.max_groups = self.num 

        for dr in [self.fig_dir, self.data_dir]:
            if not os.path.exists(dr):
                os.makedirs(dr)

        # initialize a generator
        if self.img_side > 2:
            self.generator = polyomino_scenes(self.n, self.img_side, self.num, self.batch, True)
            self.types = len(self.generator.n_ominoes)
        else:
            self.types = 2

        # training steps
        self.steps = train_step

        # iterate steps
        self.episodes = evolution_step

        # start to build the graph

        # initialize a network
        #self.network = small_net(5,16)
        self.network = big_net(img_side, 3, 5, num_features=16)
        self.network.apply(weights_init)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        # if you want to see which param is gonna require grads print it with its value
        # by uncommenting following
        """
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                print(name, param.data)
                """

        # save map generated
        self.couplings = []
        # one test image could be used
        if self.img_side > 2:
            test_batch, test_mask = generate_test_img(n=self.n, img_side=self.img_side, num=self.num)
            #self.test_img = self.convert2oned(test_data[0], test=True)
            self.test_batch = torch.tensor(test_batch).unsqueeze(1).float()
            self.test_mask = torch.tensor(list(map(self._mask_pro, test_mask))).float()
        else:
            self.test_img = np.reshape(np.concatenate([np.zeros((1, 2)), np.ones((1, 2))], axis=0), newshape=[1, 2, 2, 1])
            self.test_img = self.convert2oned(self.test_img, test=True)

            self.test_mask = np.reshape(np.concatenate([np.array([[1, 1, 0, 0]]), np.array([[0, 0, 1, 1]])]), [1, 2, 4])
            self.test_mask = torch.tensor(self.test_mask).float()

        self.loss_history = []

    def opt(self):
        for t in range(self.steps):
            if self.img_side > 2:
                batch, sd = self.generator.generate_batch()
            else:
                batch, sd = generate_small(self.batch, self.num, self.img_side)
            #print('data generated')
            # give data a channel and pad with tensorflow strategy 'SAME'
            #batch = self.convert2oned(batch, test=False)
            #print(batch)
            # batch.shape=(batch, channel, N + 1)

            self.optimizer.zero_grad()
            coupling_mats = self.network.forward(torch.tensor(batch).unsqueeze(1).float())
            final_phase = evolution(self.episodes, coupling_mats, self.batch, self.img_side ** 2, eps_init=.01,anneal=True)
            mask = torch.tensor(list(map(self._mask_pro, sd))).float()

            #print(final_phase.shape, mask.shape)
            #total_loss, synch_loss, desynch_loss = IP_loss(final_phase, mask)
            total_loss, synch_loss, desynch_loss = matt_loss(final_phase, mask, eta=.5)
            total_loss.backward()
            clip_grad_norm_(self.network.parameters(), .1)

            # if you want to see which gradients print it with its value
            # by uncommenting following
            """
            for name, param in self.network.named_parameters():
                if param.requires_grad:
                    print(name, param.grad.data)
                    """
            if t % self.report == 0:
                np.set_printoptions(precision=4)
                test_coupling = self.network(self.test_batch)
                print('Coupling', test_coupling)
                self.couplings.append(test_coupling.data.numpy())
                #print(final_phase)
                loss = np.array([total_loss.data.numpy(), synch_loss.data.numpy(), desynch_loss.data.numpy()])
                print('STEPS: ', str(t), '|LOSS: ', str(loss))
                np.save(os.path.join(self.data_dir,'./couplings25.npy'), self.couplings)
                torch.save(self.network.state_dict(), os.path.join(self.data_dir, 'IP_loss.pt'))
                self.loss_history.append(
                    np.array([total_loss.data.numpy(), synch_loss.data.numpy(), desynch_loss.data.numpy()]))
                plt.plot(np.array(self.loss_history))
                plt.legend(('Total', 'Synch', 'Desynch'))
                plt.savefig(os.path.join(self.fig_dir, 'coupling_net_loss25.png'))
                plt.close()

    def _mask_pro(self, group_dict):
        # create masks for training
        masks = []
        group_names = group_dict.keys()
        for group_name in group_names:
            group_idxes = group_dict[group_name]
            mask = np.zeros((self.img_side**2,))
            #mask = [1. if i in group_idxes else 0. for i in range(self.img_side ** 2)]
            mask[group_idxes] = 1.0
            masks.append(mask)
        return masks

    def convert2oned(self, img_batch, test=False):
        if not test:
            output = torch.reshape(torch.tensor(img_batch), [self.batch, 1, self.img_side ** 2]).float()
            # This function will pad only the last dim
            return torch.nn.functional.pad(output, pad=(0, 1))
        else:
            output = torch.reshape(torch.tensor(img_batch), [1, 1, self.img_side ** 2]).float()
            return torch.nn.functional.pad(output, pad=(0, 1))


if __name__ == '__main__':
    training = TrainPipeline(n=5, num=4, img_side=16, evolution_step=25, train_step=10000, lr=1e-4)
    print('start optimizing')
    training.opt()
