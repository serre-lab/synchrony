import os
import sys
import numpy as np
import glob
import torch
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str, default='/users/mchalvid/scratch/Kura_ODE/results')
parser.add_argument('--value', type=str,default='loss')
parser.add_argument('--target_dir', type=str, default='/users/mchalvid/scratch/Kura_ODE/results')

args = parser.parse_args()

experiments = subfolders = [ f.path for f in os.scandir(args.result_dir) if f.is_dir() ]

fig,ax = plt.subplots(figsize=(8,8))
colors = plt.get_cmap('hsv')([20*i for i in range(len(experiments))])
for i, experiment in enumerate(sorted(experiments)):
    print(experiment.split('/')[-1])
    if experiment.split('/')[-1] in ['2_digit_mnist_ODE_100_debug','2_digit_mnist_ODE_exp_3']:
        continue
    files = glob.glob(experiment + '/*.npy')
    for file in files:
        print(file)
        if 'valid_epoch' in file:
            valid_epoch = np.load(file)
        if 'train_{}'.format(args.value) in file:
            train = np.load(file)
        if 'valid_{}'.format(args.value) in file:
            valid = np.load(file)
    N=30
    train = np.convolve(train, np.ones(N)/N, mode='full')[30:-30]
    valid_epoch = valid_epoch+1
    if args.value == 'loss':
        if experiment.split('/')[-1] == '2_digit_mnist_ODE_exp_record_steps':
            train = train/10
            valid = valid/10
        elif experiment.split('/')[-1] == '2_digit_mnist_ODE_exp_fulltime':
            train = train/20
            valid = valid/20
        else:
            train = train/5
            valid = valid/5
    if args.value == 'sbd':
        ax.plot(np.arange(len(train)) * 100,train, c=colors[i], label=experiment.split('/')[-1])
        if experiment.split('/')[-1] in ['2_digit_mnist_ODE_linear', '2_digit_mnist_ODE_linear_1_conv', '2_digit_mnist_ODE_linear_random_connected','2_digit_mnist_ODE_linear_explicit_adams']:
            ax.plot(valid_epoch * 300, valid, c=colors[i], linestyle=":")
        else:
            ax.plot(valid_epoch * 700, valid, c=colors[i], linestyle=":")
    else:
        ax.plot(train,c=colors[i],label = experiment.split('/')[-1])
        if experiment.split('/')[-1] in ['2_digit_mnist_ODE_linear', '2_digit_mnist_ODE_linear_1_conv', '2_digit_mnist_ODE_linear_random_connected','2_digit_mnist_ODE_linear_explicit_adams']:
            ax.plot(valid_epoch * 250, valid, c=colors[i], linestyle=":")
        else:
            ax.plot(valid_epoch * 600, valid, c=colors[i], linestyle=":")
ax.set_xlim([-5000,90000])
ax.set_ylim([0.6,1.2])
plt.legend(loc=0)
plt.savefig(args.target_dir+'/'+args.value+'_general_linear_exp.png',dpi=100)
plt.show()