import subprocess
import argparse
from distutils.util import strtobool
import os, csv
from configparser import ConfigParser
import ipdb

argparser = argparse.ArgumentParser()
argparser.add_argument('--seed_search', type=lambda x:bool(strtobool(x)), default=True)
argparser.add_argument('--load_dir', type=str, default='/media/data_cifs/projects/prj_synchrony/results/models/mixture')
args = argparser.parse_args()

num_devices=8
experiments = ['DEFAULT'] + ['E{}'.format(i+1) for i in range(23)]
config = ConfigParser()
config.read('experiments.cfg')

for e, exp in enumerate(experiments):
    if exp == 'E7': continue    
    device = e % num_devices
    if not args.seed_search:
        exp_name = config.get(exp,'exp_name')
        result_csv = os.path.join(args.load_dir,exp_name,'result.csv')
        with open(result_csv, mode='r') as fn:
            reader = csv.reader(fn)
            exp_dict = {row[0]:row[1] for row in reader}
            best_seed = exp_dict['seed']
        num_seeds = 0
    else:
        best_seed = -1
        num_seeds = 10
    subprocess.call('CUDA_VISIBLE_DEVICES={} python train_model.py --name {} --num_seeds {} --best_seed {}&'.format(device,exp, num_seeds, best_seed),shell=True)
