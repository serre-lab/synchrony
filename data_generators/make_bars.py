import numpy as np
from tqdm import tqdm
import ipdb
import os

def generate_bars(num_imgs, 
                  img_side=32,
		  num_bars=1,
                  bgrnd_phase=True,
                  save_dir=None):

    for n in tqdm(range(num_imgs)):
        img = np.zeros((img_side, img_side)) if not bgrnd_phase else np.exp(1j*2*np.pi*np.random.rand())*np.ones((img_side,img_side))
        rand_v_locs = np.random.permutation(img_side)[:num_bars]
        rand_h_locs = np.random.permutation(img_side)[:num_bars]
	rand_locs = [rand_v_locs, rand_h_locs]

        for b in range(num_bars):
            v_or_h = np.random.rand() > .5
            #rand_phase = 2*np.pi*np.random.rand(2)
            rand_phase = np.random.choice(np.linspace(0,2*np.pi,8)[:-1], size=2)
            if v_or_h:
                img[:,rand_v_locs[b]] = np.exp(1j*rand_phase[0])
                img[rand_h_locs[b],:] = np.exp(1j*rand_phase[1])
            else:
	        img[rand_h_locs[b],:] = np.exp(1j*rand_phase[0])
	        img[:,rand_v_locs[b]] = np.exp(1j*rand_phase[1])
        mod   = np.abs(img)
        phase = np.angle(img)
        cplx_img = np.concatenate((np.expand_dims(mod*np.cos(phase),0), np.expand_dims(mod*np.sin(phase),0)),axis=0) 
        np.save(os.path.join(save_dir,'img_%04d.npy'.format(num_bars) % n), cplx_img)

if __name__=='__main__':
    bgrnd_phase = False
    bgrnd = 'BGRND' if bgrnd_phase is True else 'NO_BGRND'
    for nb in [1,2,3,4,5,6]:
        save_dir = os.path.join(os.path.expanduser('~'), 'data', 'synch_data','PHASE_BARS_{}'.format(bgrnd), str(nb), '0')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print ('Generating {} bar images'.format(nb))
        generate_bars(10000, num_bars=nb, bgrnd_phase=bgrnd_phase, save_dir=save_dir)
