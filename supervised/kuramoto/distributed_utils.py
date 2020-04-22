import os
import torch
import torch.distributed as dist
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms
from tqdm import tqdm
from losses import calc_sbd
import display as disp
from utils import *

import nets

""" Dataset partitioning helper """
class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index
    def __len__(self):
        return len(self.index)
    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    def __init__(self, data, size, seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)
        sizes = [float(1/size) for i in range(0, size)]
        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]
    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset(dataset, batch_size, shuffle=True, drop_last=True):
    size = dist.get_world_size()
    bsz = int(batch_size/size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, size)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                            batch_size=bsz,
                                            shuffle=shuffle,drop_last=drop_last)
    return train_set, bsz


def run(rank, args, connectivity):

    loss_history = []
    sbd_history = []
    loss_history_test = []
    sbd_history_test = []
    epoch_history_test = []
    coupling_history = []

    displayer = disp.displayer(args.segments, interactive=args.interactive)

    args.rank = rank
    # training_sets
    training_set = datasets.DatasetFolder(args.train_path, np.load, extensions=('npy',))
    training_loader, train_bsz = partition_dataset(dataset=training_set, batch_size=args.batch_size, shuffle=True,
                                                   drop_last=True)
    # testing_sets
    testing_set = datasets.DatasetFolder(args.test_path, np.load, extensions=('npy',))
    testing_loader, test_bsz = partition_dataset(dataset=testing_set, batch_size=args.batch_size, shuffle=True,
                                                 drop_last=True)

    #loading model and syncing
    model = nets.load_net(args, connectivity, rank, args.num_global_control)
    criterion = nets.criterion(args.time_weight, args.img_side ** 2, classify=args.classify,
                               recurrent_classifier=args.recurrent_classifier, rank=rank)

    sync_weights_2(model)

    #if args.intrinsic_frequencies == 'learned':
    #    params = [q1 for q1 in model.parameters()] + [q2 for q2 in freq_params]
    #else:
        #params = list(model.parameters())
    #if args.classify is True:
    #    params += [q3 for q3 in classifier_params]

    #Params declaration
    params = list(model.parameters())
    params = tuple(params)
    op = torch.optim.Adam(params, lr=args.learning_rate)


    #TRAINING LOOP#
    num_batches = ceil(len(training_loader.dataset)/float(train_bsz))
    for epoch in range(args.train_epochs):
        print('Epoch: {}'.format(epoch))

        l = 0
        sbd = 0
        cont_epoch = True
        PL_epoch = []
        clustering_epoch = []

        for step, (train_data, _) in tqdm(enumerate(training_loader)):
            batch = torch.tensor(train_data[:, 0, ...]).cuda(rank).float()
            mask = torch.tensor(train_data[:, 1:, ...]).reshape(-1, args.segments, args.img_side * args.img_side).to(rank).float()
            #label_inds = (((mask.sum(2) > 0) * 1).sum(1) == args.segments - 1) * 1
            #labels = torch.zeros((args.batch_size, 2)).cuda(rank).scatter_(1, label_inds.unsqueeze(1), 1.0)

            op.zero_grad()

            phase_list_train, coupling_train, omega_train = model(batch.unsqueeze(1))
            last_phase = phase_list_train[-1].cpu().detach().numpy()
            colored_mask = (np.expand_dims(np.expand_dims(np.arange(args.segments), axis=0),
                                           axis=-1) * mask.cpu().detach().numpy()).sum(1)

            #Loss computation
            tavg_loss = criterion(phase_list_train[-1 * args.record_steps:], mask, args.transform, valid=False)
            tavg_loss = tavg_loss.mean() #/ norm
            print('loss', tavg_loss)

            if coupling_train is not None:
                tavg_loss += args.sparsity_weight * torch.abs(coupling_train).mean()
            if omega_train is not None:
                tavg_loss += args.sparsity_weight * torch.abs(omega_train).mean()
            l += tavg_loss.data.cpu().numpy()


            # Backpropagation and workaround to propagate gradients through the conv model
            tavg_loss.backward(retain_graph=True)
            nfe_forward = model.osci.nfe
            model.osci.nfe = 0
            coupling_train.backward(gradient=model.osci.ODEDynamic.couplings.grad)

            # Send and average gradients across models
            average_gradients(model)
            op.step()
            sync_weights_2(model)

            print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', tavg_loss.data.cpu().numpy() )

            if step > 0:
                loss_history.append(l / (step + 1))
                sbd_history.append(sbd / ((1 + (step // args.show_every)) * args.batch_size))

            if step % args.show_every == 0:
                sbd += display_master(last_phase,
                               phase_list_train,batch,
                               mask,
                               coupling_train,
                               omega_train,
                               colored_mask,
                               clustering,
                               args,
                               display,
                               epoch,
                               step,
                               displayer)

                save_plot(args, epoch,
                          model, op,
                          connectivity,
                          loss_history,
                          loss_history_test,
                          sbd_history,
                          epoch_history_test,
                          sbd_history_test,
                          PL_train=None,
                          PL_val=None,
                          clustering_train=None,
                          clustering_val=None)











def display_master(last_phase,
                   phase_list_train,
                   batch, mask,
                   coupling_train,
                   omega_train,
                   colored_mask,
                   clustering,
                   args, display,
                   epoch,step,
                   displayer):
    clustered_batch = []
    sbd=[]
    if args.rank == 0:
        for idx, (sample_phase, sample_mask) in enumerate(zip(last_phase, colored_mask)):
            clustered_batch.append(clustering(sample_phase, n_clusters=args.segments))
            sbd += calc_sbd(clustered_batch[idx] + 1, sample_mask + 1)
        display(displayer, phase_list_train, batch, mask, clustered_batch, coupling_train, omega_train,
                args.img_side, args.segments, args.save_dir,
                'train{}_{}'.format(epoch, step), args.rf_type)
        return sbd

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def sync_weights(model, rank, world_size):
    for param in model.parameters():
        if rank == 0:
            # Rank 0 is sending it's own weight
            # to all it's siblings (1 to world_size)
            for sibling in range(1, world_size):
                dist.send(param.data, dst=sibling)
        else:
            # Siblings must recieve the parameters
            dist.recv(param.data, src=0)

def sync_weights_2(model):
    for param in model.parameters():
        dist.broadcast(param.data, 0, async_op = False)

def validate(model, rank, world_size):
    for param in model.parameters():
        if rank == 0:
            # Rank 0 is sending it's own weight
            # to all it's siblings (1 to world_size)
            for sibling in range(1, world_size):
                dist.send(param.data, dst=sibling)
        else:
            # Siblings must recieve the parameters
            dist.recv(param.data, src=0)


def init_processes(rank, args, connectivity, func, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=args.world_size)
    torch.manual_seed(42)
    func(rank, args, connectivity)



def Launch(args, connectivity):
    #Launch every process and join
    size = args.world_size
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, args, connectivity, run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def testing(epoch, args, testing_loader):
    if (epoch + 1) % args.eval_interval == 0:
        l = 0
        sbd = 0
        PL_epoch = []
        clustering_epoch = []

        with torch.no_grad():
            for step, (test_data, _) in tqdm(enumerate(testing_loader)):
                # cross-validation
                batch = test_data[:, 0, ...].float().to(args.device)
                mask = test_data[:, 1:, ...].reshape(-1, args.segments, args.img_side * args.img_side).float().to(
                    args.device)
                label_inds = (((mask.sum(2) > 0) * 1).sum(1) == args.segments - 1) * 1
                labels = torch.zeros((args.batch_size, 2)).to(args.device).scatter_(1, label_inds.unsqueeze(1), 1.0)

                phase_list_test, coupling_test, omega_test = model(batch.unsqueeze(1))

                last_phase = phase_list_test[-1].cpu().detach().numpy()
                colored_mask = (np.expand_dims(np.expand_dims(np.arange(args.segments), axis=0),
                                               axis=-1) * mask.cpu().detach().numpy()).sum(1)

                clustered_batch = []
                for idx, (sample_phase, sample_mask) in enumerate(zip(last_phase, colored_mask)):
                    clustered_batch.append(clustering(sample_phase, n_clusters=args.segments))
                    sbd += calc_sbd(clustered_batch[idx] + 1, sample_mask + 1)

                tavg_loss_test = criterion(phase_list_test[-1 * args.record_steps:], mask, args.transform, valid=True,
                                           targets=labels)
                tavg_loss_test = tavg_loss_test.mean() / norm
                if coupling_test is not None:
                    tavg_loss_test += args.sparsity_weight * torch.abs(coupling_test).mean()
                if omega_test is not None:
                    tavg_loss_test += args.sparsity_weight * torch.abs(omega_test).mean()
                l += tavg_loss_test.data.cpu().numpy()

                if step % args.show_every == 0:
                    # visualize validation and save
                    # validation example, save its coupling matrix
                    display(displayer, phase_list_test, batch, mask, clustered_batch, coupling_test, omega_test,
                            args.img_side, args.segments, save_dir,
                            'test{}_{}'.format(epoch, step), args.rf_type)
                # if step*args.batch_size > num_test:
                #    break

        loss_history_test.append(l / (step + 1))
        sbd_history_test.append(sbd / ((step + 1) * args.batch_size))
        epoch_history_test.append(epoch)


def save_plot(args,epoch,
              model,op,
              connectivity,
              loss_history,
              loss_history_test,
              sbd_history,
              epoch_history_test,
              sbd_history_test,
              PL_train=None,
              PL_val=None,
              clustering_train=None,
              clustering_val=None):

    if args.rank == 0:
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': op.state_dict(),
                    'initial_phase': args.phase_initialization,
                    'connectivity': connectivity}, args.model_dir + '/model{}.pt'.format(epoch))

        plt.plot(loss_history)
        plt.plot(epoch_history_test, loss_history_test)
        plt.title('Time Averaged Loss')
        plt.legend(['train', 'valid'])
        plt.savefig(args.save_dir + '/loss' + '.png')
        plt.close()

        plt.plot(sbd_history)
        plt.plot(epoch_history_test, sbd_history_test)
        plt.title('Symmetric Best Dice')
        plt.legend(['train', 'valid'])
        plt.savefig(args.save_dir + '/sbd' + '.png')
        plt.close()

        np.save(os.path.join(args.save_dir, 'train_loss.npy'), np.array(loss_history))
        np.save(os.path.join(args.save_dir, 'valid_loss.npy'), np.array(loss_history_test))
        np.save(os.path.join(args.save_dir, 'train_sbd.npy'), np.array(sbd_history))
        np.save(os.path.join(args.save_dir, 'valid_sbd.npy'), np.array(sbd_history_test))
        np.save(os.path.join(args.save_dir, 'valid_epoch.npy'), np.array(epoch_history_test))

        if args.path_length == True:
            plt.plot(np.array(PL_train))
            plt.plot(np.array(PL_val))
            plt.title('Path Length')
            plt.legend(['Train', 'Validation'])
            plt.savefig(args.save_dir + '/pathlength.png')
            plt.close()
        if args.cluster == True:
            plt.plot(np.array(clustering_train))
            plt.plot(np.array(clustering_val))
            plt.title('Clustering Coefficient')
            plt.legend(['Train', 'Validation'])
            plt.savefig(args.save_dir + '/clusteringcoefficient.png')
            plt.close()