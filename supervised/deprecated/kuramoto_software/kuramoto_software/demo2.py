import numpy as np
import kuramoto as km
import kura_visual as kv

PATH = './three/4'

oscis = km.kura_np(4)
displayer = kv.displayer()

po_ones = np.ones((2, 2))
ne_ones = - np.ones((2, 2))
de_ones = np.array([[1, -1], [-1, 1]]) * 2
col1 = np.concatenate((de_ones, ne_ones), axis=0)
col2 = np.concatenate((ne_ones, po_ones), axis=0)

coupling = np.concatenate((col1, col2), axis=1)
oscis.set_coupling(coupling)
oscis.frequency_init(np.random.rand(4) * 0.1)

two_oscis = np.ones((1, 2))
de_two_oscis = np.array([[1, 0.6]])
compare = np.concatenate((de_two_oscis * 0.8, two_oscis * 0.2), axis=0)
mask = np.concatenate((np.array([[1, 0, 0, 0]]),
                       np.array([[0, 1, 0, 0]]),
                       np.array([[0, 0, 1, 1]])),
                      axis=0)
mask = np.expand_dims(mask, axis=0)

phases, freqs = oscis.evolution(steps=500, record=True, show=True)
# print(np.where(loss.coh_btw_groups_np(np.squeeze(phases), mask) == 0))

displayer.set_phases(phases)
displayer.set_masks(mask)
displayer.compute_properties()
displayer.animate_evol_compare(2, 2, compare, save_name=PATH)
displayer.phase_evol(save_name=PATH + '_phase')
displayer.properties_evol(save_name=PATH + '_property')
