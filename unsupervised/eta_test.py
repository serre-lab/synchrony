import subprocess
import numpy as np

competition_weight= np.linspace(0,10,8) 
for e, eta in enumerate(competition_weight):
    subprocess.call('CUDA_VISIBLE_DEVICES={} python synch_learn.py {} {}&'.format(e,eta,e), shell=True)
