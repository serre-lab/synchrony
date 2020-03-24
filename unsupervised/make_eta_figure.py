import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')
import numpy as np
import os

competition_weights = np.linspace(0,2,8)
num_runs = len(competition_weights)
save_dir = '/home/matt/figs/synch_learn'
energies = []
for run in range(num_runs):
    energies.append(np.load(os.path.join(save_dir, 'energy_{}.npy'.format(run))))

plt.plot(np.array(energies).T)
plt.legend(tuple(['eta={}'.format(eta) for eta in competition_weights]))
plt.savefig(os.path.join(save_dir, 'energies_per_eta.png'))
