import argparse
from ConfigParser import ConfigParser 
import os
import subprocess
import ipdb

meta_parser = argparse.ArgumentParser()
meta_parser.add_argument('--name', type=str, default='Multi_MNIST')
meta_args = meta_parser.parse_args()

config = ConfigParser()
config.read('experiments.cfg')

if not config.has_section(meta_args.name):
    ValueError('This is not a recognized experiment.')

#TODO Finish this
if meta_args.name[-6:] == 'search':
    sub_experiment = config.get(meta_args.name, 'sub_experiment')
    sub_args = config.items(sub_experiment)
arg_strings = ['--{} {}'.format(key, value) + ' ' for (key, value) in config.items(meta_args.name)] 
process_string = 'python train.py '
for st in arg_strings: process_string += st
subprocess.call(process_string, shell=True)
