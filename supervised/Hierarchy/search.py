import argparse
import nvidia_smi
import subprocess

"""
This file is used for running multiple set up, you need to specify which experiment you want to run and how many
gpus you want to use per experiment. Then the program will find available gpus automatically (under certain threshold),
and allocating experiment on those gpus.
It hasn't been tested yet, but the structure of the code is fairly simple, should be easy to debug
"""

###################
# parser
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, help='training file name')
parser.add_argument('-g', '--num-gpu', type=int, help='number of gpus per experiment')
parser.add_argument('-n', '--names', type=str, help='names of multiple experiment, separate by space')
args = parser.parse_args()

###################
# available gpus
gpu_avail = []
nvidia_smi.nvmlInit()
for i in range(8):
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
    mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    use_res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
    # calculate memory percentage, using threshold 1%
    # get utility percentage, using threshold 10%
    if (100 * (mem_res.used / mem_res.total) < 100) and (use_res.gpu < 100):
        gpu_avail.append(str(i))
print("Available gpu:{}".format(gpu_avail))

exps = args.names.split(",")
print("Running experiment:{}".format(exps))

if len(exps) * args.num_gpu > len(gpu_avail):
    raise RuntimeError("Available gpu number is not enough, need {} gpus,"
                       "but {} gpus are available".format(len(exps) * args.num_gpu, len(gpu_avail)))

####################
# allocating
for ind, name in enumerate(exps):
    gpu_name = gpu_avail[ind*args.num_gpu:(ind+1)*args.num_gpu]
    # print('export CUDA_VISIBLE_DEVICES={}'.format(",".join(gpu_name)))
    # print('python {} -n {}'.format(args.file, name))
    subprocess.call('CUDA_VISIBLE_DEVICES={} python {} -n {}'.format(",".join(gpu_name), args.file, name),
                    shell=True) # shell=True, security hazard
