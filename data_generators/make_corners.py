import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')
import numpy as np
from test_utils import cplx_imshow
import ipdb
from tqdm import tqdm
import os

def corners(num_imgs, num_corners, height=20, width=20, cplx_bgrnd=True, save_dir=None,hid_square=False):

    corner = np.array([[1,1,1,1],
                       [1,0,0,0],
                       [1,0,0,0],
                       [1,0,0,0]])

    if hid_square:
        square = np.array([[1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1],
                           [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                           [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                           [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                           [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                           [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                           [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                           [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                           [1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1]])

    corners = [corner, corner[::-1,:].copy(), corner[:,::-1].copy(), corner[::-1,::-1].copy()]
    for i in tqdm(range(num_imgs)):
        img = np.zeros((height, width)) + 0j if not cplx_bgrnd else np.exp(1j*2*np.pi*np.random.rand())*np.ones((height, width))
        if hid_square:
            ul_y = np.random.randint(0,height-4,(1,))
            ul_x = np.random.randint(0,width-4, (1,))
            
        for s in range(num_corners):
            phase = 2*np.pi*np.random.rand()
            crn = corners[np.random.randint(0,len(corners))]
            sy, sx = crn.shape
            x = np.random.randint(0, width - sx + 1)
            y = np.random.randint(0, height - sy + 1)
            region = (slice(y,y+sy), slice(x,x+sx))
            img[region][crn != 0] = np.exp(1j*phase)
        if hid_square:
            x = np.random.randint(0,width - 16)
            y = np.random.randint(0,height - 16)
            phase = 2*np.pi*np.random.rand()
            region = (slice(y,y+16), slice(x,x+16))
            img[region][square != 0] = np.exp(1j*phase)
       
        #fig, ax = plt.subplots()
        #im = cplx_imshow(ax, img)
        #plt.savefig(os.path.join(os.path.expanduser('~'),'tmp_{}.png'.format(i)))
        #plt.close()
        cplx_img = np.concatenate([np.expand_dims(np.abs(img)*np.cos(np.angle(img)),0), np.expand_dims(np.abs(img)*np.sin(np.angle(img)),0)],axis=0)
        np.save(os.path.join(save_dir,'img_%04d.npy' % i), cplx_img)
            
if __name__=='__main__' :
    #TODO Finish
    cplx_bgrnd = False
    bgrnd_name = 'BGRND' if cplx_bgrnd else 'NO_BGRND'
    for nc in [3,4,5,6]:
        save_dir = os.path.join(os.path.expanduser('~'), 'data', 'synch_data', 'PHASE_CORNERS_{}'.format(bgrnd_name), str(nc), str(0))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print('Generating {} corner images'.format(nc))   
        corners(5000, nc, save_dir=save_dir, cplx_bgrnd=cplx_bgrnd, hid_square=True)
