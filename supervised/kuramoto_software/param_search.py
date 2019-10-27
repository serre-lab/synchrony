import subprocess
import numpy as np

sparsity = [1e-2, 1e-3]
etas = np.linspace(0,.25,4)

device = 0
for e, eta in enumerate(etas):
    for s in sparsity:
    process_string = 'python demo4.py --device {} --eta {} --sparsity {}&'.format(device, eta, s)
    subprocess.call(process_string, shell=True)
    device+=1
