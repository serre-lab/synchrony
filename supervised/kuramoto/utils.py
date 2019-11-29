import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

def show(displayer, phase_list, image, mask, coupling, path, name, group_size, img_side):
    colored_mask = (np.expand_dims(np.expand_dims(np.arange(group_size), axis=0), axis=-1) * mask / group_size).sum(1)
    displayer.set_phases(phase_list)
    displayer.set_masks(mask)
    displayer.compute_properties()

    displayer.animate_evol_compare(img_side, img_side, compare=image, save_name=path + '/ani_' + name)
    displayer.static_evol(img_side, img_side, image, path + '/static_' + name, colored_mask)
    displayer.phase_evol2(path + '/phase_' + name)

    im = plt.imshow(coupling)
    plt.title('Coupling Matrix')
    plt.gca().grid(False)
    plt.axis('off')
    plt.colorbar(im)
    plt.savefig(path + '/coupling_' + name)
    plt.close()
	
def sw_connectivity(img_side, num_cn, critic_dist):
    connectivity = np.zeros((img_side ** 2, num_cn))
    for i in tqdm(range(img_side ** 2)):
        count = 0
        for j in range(img_side ** 2):
	    x_1 = int(i % img_side)
            y_1 = int(i // img_side)
	    x_2 = int(j % img_side)
	    y_2 = int(j // img_side)
	    dist = np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)
	    if (dist < critic_dist) and (dist > 0):
	        connectivity[i, count] = j
		count += 1
        while count < num_cn:
	    connectivity[i, count:] = \
	    sorted(np.delete(range(img_side ** 2), connectivity[i, ...]),
            key=lambda k: np.random.random())[:int(num_cn - count)]
            count += 1
        change = sorted(range(num_cn), key=lambda k: np.random.random())[:int(num_cn / 2 + 1)]
	connectivity[i, ...][change] = \
        sorted(np.delete(range(img_side ** 2), connectivity[i, ...]),
	    key=lambda k: np.random.random())[:int(num_cn / 2 + 1)]
    return connectivity
