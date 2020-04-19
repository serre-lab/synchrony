import argparse
from configparser import ConfigParser 
import os
import torch
import subprocess
import json, ast
import sys
import ipdb

os.environ["CUDA_VISIBLE_DEVICES"]='0'

meta_parser = argparse.ArgumentParser()
meta_parser.add_argument('--name', type=str, default='DEFAULT')
meta_args = meta_parser.parse_args()

print(meta_args)

config = ConfigParser()
config.read('experiments.cfg')

if not config.has_section(meta_args.name):
    ValueError('This is not a recognized experiment.')

if meta_args.name[-6:] == 'search':
    sub_experiment = config.get(meta_args.name, 'sub_experiment')
    sub_args = {}
    for item in config.items(sub_experiment): sub_args[item[0]] = item[1]
    search_parameter = config.get(meta_args.name, 'search_parameter')
    search_values = ast.literal_eval(config.get(meta_args.name, 'search_values'))
    device_indices = json.loads(config.get(meta_args.name, 'device_indices'))
    if len(search_values) != len(device_indices):
        ValueError('Your search has not allocated the correct number of devices.') 
    for v, value in enumerate(search_values):
        sub_args[search_parameter] = value 
        sub_args['exp_name'] = '{}_{}_{}'.format(sub_experiment,search_parameter, value)
        print('Running {} experiment with value {} on device {}'.format(search_parameter,value,device_indices[v]))
        #os.environ["CUDA_VISIBLE_DEVICES"]=str(device_indices[v])
        arg_strings = ['--{} {}'.format(key, value) + ' ' for (key, value) in sub_args.items()] 
        process_string = 'CUDA_VISIBLE_DEVICES={}, python train.py '.format(device_indices[v])
        for st in arg_strings: process_string += st
        subprocess.call(process_string + '&', shell=True)    
    
else:
    arg_strings = ['--{} {}'.format(key, value) + ' ' for (key, value) in config.items(meta_args.name)] 
    process_string = ' train.py '
    for st in arg_strings:
        process_string += st
    print(sys.executable + process_string)
    subprocess.call(sys.executable+process_string, shell=True)