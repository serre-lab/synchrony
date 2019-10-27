import subprocess
import numpy as np

etas = np.linspace(0,.5,8)

for e, eta in enumerate(etas):
    process_string = 'python demo4.py {} {}&'.format(e, eta)
    subprocess.call(process_string, shell=True)
