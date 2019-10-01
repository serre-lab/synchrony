import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from oscillator import oscillator, run_osci
import numpy as np
from make_data import polyomino_scenes
# import matplotlib.pyplot as plt
from itertools import product
import torch
from tqdm import tqdm
import ipdb

def get_coupling(batch, sameness_dicts):
    batch_size = batch.shape[0]
    img_side   = batch.shape[1]
    # Default is weak repulsion
    coupling_mats = np.ones((batch_size, img_side**2,img_side**2)) * .01
    for b in range(batch_size):
        sd = sameness_dicts[b]
        for key1 in sd.keys():
            for key2 in sd.keys():
                if key1 == 'bgrnd' and key2 == 'bgrnd': continue
                group1_coords = sd[key1] 
                num1 = len(group1_coords[0])
                flat_coords1 = np.ravel_multi_index(group1_coords, (img_side, img_side))
                group2_coords = sd[key2] 
                num2 = len(group1_coords[1])
                flat_coords2 = np.ravel_multi_index(group2_coords, (img_side, img_side))
                for coords in list(product(flat_coords1, flat_coords2)):
                    if key1 == 'bgrnd' or key2 == 'bgrnd':
                        #c = -0.01
                        c = -2.0 / (num1 + num2)
                    elif key1==key2:
                        c = 2.0 / (num1 + num2)
                    else:
                        c = -2.0 / (num1 + num2)
                    # Strong within-group attraction; moderate between-group repulsion
                    coupling_mats[b,coords[0], coords[1]] = c
    return torch.tensor(coupling_mats)

# [42, 43, 59, 188, 204, 205], [100, 116, 132]
coupling_mat = np.ones((256, 256)) * 0.01
group1 = [42, 43, 59, 188, 204, 205]
group2 = [100, 116, 132]

# group1 & group2 sightly decouple from population
for elem in group1:
    coupling_mat[elem] = np.full((256,), -0.01)
for elem in group2:
    coupling_mat[elem] = np.full((256,), -0.01)

# group1 & group2 sightly decouple from population
for elem in group1:
    coupling_mat[:, elem] = np.full((256,), -0.01)
for elem in group2:
    coupling_mat[:, elem] = np.full((256,), -0.01)

# group1 couple with each other and decouple with group2
for elem in group1:
    for other in group1:
        coupling_mat[elem, other] = 1
    for another in group2:
        coupling_mat[elem, another] = -0.4

# group2 couple with each other and decouple with group1
for elem in group2:
    for other in group2:
        coupling_mat[elem, other] = 2
    for another in group1:
        coupling_mat[elem, another] = -0.4

# plt.imshow(coupling_mat)
# plt.show()

color = [0] * 256
for elem in group1:
    color[elem] = 10
for elem in group2:
    color[elem] = 6
n=4
img_side= 32
num_objects=5
batch_size=128
osc_steps = 256
show_every = 16
classes = None # Setting classes = int adds int freq dynamics, but is very slow

losses=[]
generator = polyomino_scenes(n, img_side, num_objects, batch_size=batch_size,rotations=True)
batch, sameness_dict = generator.generate_batch()
print('Generating couplings')
coupling_mats = get_coupling(batch,sameness_dict)

print('Initializing oscillators')
osci = oscillator(batch, sameness_dict, epsilon=1e-1, K=256, img_side=img_side, coupling_mats=coupling_mats, classes=classes, initial_frq=torch.zeros((batch_size, img_side**2)))
print('Running oscillation')
for i in tqdm(range(osc_steps)):
    osci.update()
    if i%show_every == 0:
        tl, sl, dl = osci.seg_loss() 
        losses.append(np.array([tl.data.numpy(), sl.data.numpy(), dl.data.numpy()]))

min_avg_fp = (1. / 12) * (2*(num_objects + 1)**2 - 3*(num_objects + 1) + 1)

plt.plot(losses)
plt.hlines(min_avg_fp*np.ones((len(losses),)), 0, len(losses,),linestyles='dashed')
plt.ylim([0, np.array(losses).max()])
plt.legend(('Total', 'Synch', 'Desynch'))
plt.savefig('/home/matt/loss_test.png')
plt.close()
