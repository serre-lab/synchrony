import numpy as np
import kura_visual as kv
import kuramoto as km
import net
from make_data_new import polyomino_scenes

generator = polyomino_scenes(n=5, num_objects=3, img_side=12, batch_size=1, rotation=True)
train_data = generator.generate_batch()

# I just want to get rid of the zeros in mask to save memory
group_dict = {}
for group_name in train_data[1][0].keys():
    if train_data[1][0][group_name]:
        group_dict[group_name] = train_data[1][0][group_name]
mask = net.mask_pro(group_dict)

coupling = np.zeros((144, 144))
counter = 0
for group_name in group_dict.keys():
    for other in group_dict.keys():
        if other == group_name:
            continue
        else:
            for m in group_dict[group_name]:
                if len(group_dict[other]) * len(group_dict[group_name]) > 60:
                    for n in group_dict[other]:
                        coupling[m][n] = -1
                else:
                    for n in group_dict[other]:
                        coupling[m][n] = -10

                if len(group_dict[group_name]) > 10:
                    for m_ in group_dict[group_name]:
                        coupling[m][m_] = 1
                else:
                    for m_ in group_dict[group_name]:
                        coupling[m][m_] = .1

osci = km.kura_np(144)
osci.phase_init()
osci.set_coupling(coupling)
osci.set_ep(1)

phase_list, _ = osci.evolution(steps=80, record=True, show=True)

displayer = kv.displayer()
displayer.set_phases(phase_list)
displayer.set_masks(mask)
displayer.compute_properties()
displayer.animate_evol_compare(12, 12, compare=train_data[0][0], save_name='hard-wired')
