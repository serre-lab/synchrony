import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import torch
import numpy as np
import ipdb
from tqdm import tqdm
from test_utils import cplx_imshow
import os

def multi_mnist(num_digits, 
                cplx=True,
                num_images=60000,
                cplx_bgrnd=True,
                save_dir=None,
                display=True):

    if num_images > 60000:
       raise('Number of images must be less than 60000')

    clean_mnist = torch.load(os.path.join(os.path.expanduser('~'), 'data/synch_data/MNIST/processed/training.pt'))
    num_mnist   = clean_mnist[0].shape[0]
    im_side     = 28 + (num_digits - 1)*10

    imgs        = []
    labels      = []

    counter = 0
    pbar = tqdm(total=num_images)
    while counter < num_images:
        lb         = []
        all_digits  = []
        all_phases = []
        canvas     = torch.zeros(im_side, im_side)
        clean_img  = clean_mnist[0][counter]
        if num_digits > 1:
            digit_y1   = torch.randint(im_side - 28, (1,))
            digit_x1   = torch.randint(im_side - 28, (1,))
            canvas[digit_y1:digit_y1 + 28, digit_x1:digit_x1 + 28] = clean_img
        else:
            canvas = clean_img.float()
        lb.append(clean_mnist[1][counter])

        all_digits.append(canvas)
        all_phases.append(2*np.pi*torch.rand((1,)))

        # Add digits
        digit_inds = torch.randperm(num_mnist)[:num_digits-1]
        for ind in digit_inds:
            digit_canvas    = torch.zeros(im_side, im_side)
            new_digit = clean_mnist[0][ind]
            lb.append(clean_mnist[1][ind])
        
            # Place patch 
            if num_digits > 1:
                digit_yn  = torch.randint(im_side - 28, (1,))
                digit_xn  = torch.randint(im_side - 28, (1,))
                digit_canvas[digit_yn:digit_yn+28, digit_xn:digit_xn+28] = new_digit
            else:
                digit_canvas = new_digit.float()

            all_digits.append(digit_canvas)
            all_phases.append(2*np.pi*torch.rand((1,)))

        if cplx:
            canvas = canvas.unsqueeze(0)
            cplx_canvas= torch.cat([canvas*torch.cos(all_phases[0]), canvas*torch.sin(all_phases[0])], dim=0)
            for (dig,phs) in zip(all_digits[1:], all_phases[1:]):
                dig = dig.unsqueeze(0)
                cplx_dig= torch.cat([dig*torch.cos(phs), dig*torch.sin(phs)], dim=0)
                cplx_canvas[cplx_dig != 0] = cplx_dig[cplx_dig != 0]
            canvas=cplx_canvas
        else:
            for dig in all_digits[1:]:
                canvas+=dig
                canvas = torch.clamp(canvas, 0,255)
        if cplx_bgrnd:
            bgrnd_phase = 2*np.pi*torch.rand((1,))
            canvas[0,...] = torch.where(canvas[0,...] == 0.0, 128*torch.cos(bgrnd_phase*torch.ones_like(canvas[0,...])), canvas[0,...])
            canvas[1,...] = torch.where(canvas[1,...] == 0.0, 128*torch.sin(bgrnd_phase*torch.ones_like(canvas[0,...])), canvas[1,...])
        imgs.append(canvas)
        labels.append(torch.tensor(lb)) 
        if display and counter == 0:
            if cplx:
                np_img = canvas.numpy() / 255.
                cplx_img = np_img[0,...] + 1j*np_img[1,...]
                fig, ax = plt.subplots()
                cplx_imshow(ax,cplx_img)
                plt.savefig(os.path.join(os.path.expanduser('~'),'tmp{}.png'.format(num_digits)))
                #ipdb.set_trace()
            else:
                np_img = canvas.numpy()
                plt.imshow(np_img)
                plt.savefig(os.path.join(os.path.expanduser('~'),'tmp{}.png'.format(num_digits)))
                plt.close()
                #ipdb.set_trace()
        
        counter+=1
        pbar.update(1)
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)
    torch.save((imgs, labels), os.path.join(save_dir, 'processed.pt'.format(num_digits)))

if __name__=='__main__':
    cplx_bgrnd = True
    for nd in [1,2,3,4]:
        save_dir = os.path.join(os.path.expanduser('~'), 'data', 'synch_data', 'MULTI_MNIST', str(nd))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print('Making {} digit images'.format(nd))
        multi_mnist(nd, num_images=60000, cplx_bgrnd=cplx_bgrnd, save_dir=save_dir, display=False)
