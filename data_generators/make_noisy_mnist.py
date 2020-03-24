import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import torch
import numpy as np
import os
import ipdb
from tqdm import tqdm
from test_utils import cplx_imshow

def noisy_mnist(max_noise_patches,
               cplx=True,
               patch_side=8,
               num_images=60000,
               save_dir = None,
               cplx_bgrnd = True,
               display=True):

    clean_mnist = torch.load(os.path.join(os.path.expanduser('~'), 'data', 'synch_data', 'MNIST/processed/training.pt'))
    num_mnist   = clean_mnist[0].shape[0]
    im_side = 40

    imgs = []
    labels=[]

    counter = 0
    pbar = tqdm(total=num_images)
    while counter < num_images:
        all_patches=[]
        all_phases = []
        canvas    = torch.zeros(im_side, im_side)
        clean_img = clean_mnist[0][counter]
        label     = clean_mnist[1][counter]
        digit_y   = torch.randint(40 - 28 - 1, (1,))
        digit_x   = torch.randint(40 - 28 - 1, (1,))
        canvas[digit_y:digit_y + 28, digit_x:digit_x + 28] = clean_img
        all_phases.append(2*np.pi*torch.rand((1,)))

        # Add 'structured' noise
        patch_inds = torch.randperm(num_mnist)[:max_noise_patches]
        for ind in patch_inds:
            patch_canvas = torch.zeros(im_side, im_side)
    
            # Acquire patch
            rand_y = torch.randint(28 - patch_side - 1, (1,))
            rand_x = torch.randint(28 - patch_side - 1, (1,))
            patch = clean_mnist[0][ind][rand_y:rand_y+patch_side, rand_x:rand_x+patch_side]
        
            # Place patch 
            patch_y = torch.randint(40-patch_side-1, (1,))
            patch_x = torch.randint(40-patch_side-1, (1,))
            patch_canvas[patch_y:patch_y+patch_side, patch_x:patch_x+patch_side] = patch
            all_patches.append(patch_canvas)
            all_phases.append(2*np.pi*torch.rand((1,)))
    
        if cplx:
            canvas = canvas.unsqueeze(0)
            cplx_canvas = torch.cat([canvas*torch.cos(all_phases[0]), canvas*torch.sin(all_phases[0])],dim=0)
            for (pch, phs) in zip(all_patches[1:], all_phases[1:]):
                pch = pch.unsqueeze(0)
                cplx_pch = torch.cat([pch*torch.cos(phs), pch*torch.sin(phs)],dim=0)
                cplx_canvas[cplx_pch != 0] = cplx_pch[cplx_pch != 0]
            canvas = cplx_canvas
        else:
            for pch in all_patches[1:]:
                canvas+=pch
                canvas = torch.clamp(canvas, 0, 255)
        if cplx_bgrnd:
            bgrnd_phase = 2*np.pi*torch.rand((1,))
            canvas[0,...] = torch.where(canvas[0,...] == 0.0, torch.cos(bgrnd_phase)*torch.ones_like(canvas[0,...]), canvas[0,...])
            canvas[1,...] = torch.where(canvas[1,...] == 0.0, torch.sin(bgrnd_phase)*torch.ones_like(canvas[1,...]), canvas[1,...])
        imgs.append(canvas)
        labels.append(label)
        if display and counter == 0:
            if cplx:
                np_img = canvas.numpy() / 255.
                cplx_img = np_img[0,...] + 1j*np_img[1,...]
                fig, ax = plt.subplots()
                cplx_imshow(ax, cplx_img) 
                plt.savefig(os.path.join(os.path.expanduser('~'),'tmp_{}.png'.format(max_noise_patches)))
                plt.close()
            else:
                np_img = canvas.numpy()
                plt.imshow(np_img)
                plt.savefig(os.path.join(os.path.expanduser('~'),'tmp.png'))
                plt.close()
        counter+=1
        pbar.update(1)
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)
    torch.save((imgs, labels), os.path.join(save_dir, 'processed.pt'.format(max_noise_patches)))

if __name__=='__main__':
    cplx_bgrnd=True
    for mn in [3,4,5,6,7,8]:
        save_dir = os.path.join(os.path.expanduser('~'), 'data', 'synch_data', 'NOISY_MNIST', str(mn))
        print('Generating {} patch images'.format(mn))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        noisy_mnist(mn,num_images=60000, save_dir=save_dir, cplx_bgrnd=cplx_bgrnd)
