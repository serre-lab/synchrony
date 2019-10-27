DISPLAY=False
if DISPLAY:
    import matplotlib.pyplot as plt
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
import numpy as np
import kura_visual as kv
import kuramoto as km
import nets
from make_data_new import polyomino_scenes
import os
import ipdb
from itertools import product

save_name = os.path.join(os.path.expanduser('~'), 'oscillators', 'hard-wired')

img_side = 12
generator = polyomino_scenes(n=5, num_objects=3, img_side=img_side, batch_size=1, rotation=True)
train_data = generator.generate_batch()

# I just want to get rid of the zeros in mask to save memory
group_dict = {}
for group_name in train_data[1][0].keys():
    if train_data[1][0][group_name]:
        group_dict[group_name] = train_data[1][0][group_name]
mask = nets.mask_pro(group_dict).astype(np.float32)
counter = 0
def get_coupling(gd, img_side, K=1.0):
    coupling_mat = np.ones((img_side**2, img_side**2)) * .01
    for key1 in gd.keys():
        for key2 in gd.keys():
            if key1 == 'bgrnd' and key2 == 'bgrnd': continue
            group1_coords = gd[key1]
            num1 = len(group1_coords)
            group2_coords = gd[key2]
            num2 = len(group1_coords)
            for coords in list(product(group1_coords, group2_coords)):
                if key1 == 'bgrnd' or key2 == 'bgrnd':
                    c = -2.0 / (num1 + num2)
                elif key1==key2:
                    c = 2.0 / (num1 + num2) 
                else:
                    c = -2.0 / (num1 + num2)
                coupling_mat[coords[0],coords[1]] = c*K
    return coupling_mat.astype(np.float32)

coupling = get_coupling(group_dict, img_side, K=10.0)
osci = km.kura_np(img_side**2, updating_rate=0.1)
osci.phase_init()
osci.set_coupling(coupling)
#int_freq = np.random.rand(img_side**2)
int_freq=None

phase_list, _ = osci.evolution(steps=500, record=True, show=True, anneal=False, int_freq=int_freq)

displayer = kv.displayer()
displayer.set_phases(phase_list)
displayer.set_masks(mask)
displayer.compute_properties()
displayer.animate_evol_compare(12, 12, compare=train_data[0][0], save_name=save_name)
