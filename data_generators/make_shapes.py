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
def generate_shapes(num_imgs, num_shapes=3, bgrnd_phase=False, save_dir=None):
    height = (num_shapes - 1)*10 if num_shapes > 2 else 20
    width  = (num_shapes - 1)*10 if num_shapes > 2 else 20

    square = np.array([[1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1],
                        [1,1,0,0,0,1,1],
                        [1,1,0,0,0,1,1],
                        [1,1,0,0,0,1,1],
                        [1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1]])
    
    triangle = np.array([[0,0,0,0,0,1,1,0,0,0,0,0],
                         [0,0,0,0,1,1,1,1,0,0,0,0],
                         [0,0,0,1,1,1,1,1,1,0,0,0],
                         [0,0,1,1,1,0,0,1,1,1,0,0],
                         [0,1,1,1,0,0,0,0,1,1,1,0],
                         [1,1,1,1,1,1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1,1,1,1,1,1]])

    shapes = [square, triangle, triangle[::-1,:].copy()]
    for i in tqdm(range(num_imgs)):
        img = np.zeros((height, width)) + 0j if bgrnd_phase is False else np.exp(1j*2*np.pi*np.random.rand())*np.ones((height, width))
        for s in range(num_shapes):
            phase = 2*np.pi*np.random.rand()
            shape = shapes[np.random.randint(0,len(shapes))]
            sy, sx = shape.shape
            x = np.random.randint(0, width - sx + 1)
            y = np.random.randint(0, height - sy + 1)
            region = (slice(y,y+sy), slice(x,x+sx))
            img[region][shape != 0] = np.exp(1j*phase)
        #fig, ax = plt.subplots()
        #im = cplx_imshow(ax, img)
        #plt.savefig('./tmp_{}.png'.format(i))
        #plt.close()
        #ipdb.set_trace()
        cplx_img = np.concatenate([np.expand_dims(np.abs(img)*np.cos(np.angle(img)),0), np.expand_dims(np.abs(img)*np.sin(np.angle(img)),0)],axis=0)
        np.save(os.path.join(save_dir,'img_%04d.npy'.format(num_shapes) % i), cplx_img)
            
if __name__=='__main__' :
    bgrnd_phase = True
    bgrnd_name = 'BGRND' if bgrnd_phase else 'NO_BGRND'
    for ns in [1,2,3,4,5]:
        print('Making {} shape images.'.format(ns))
        save_dir = os.path.join(os.path.expanduser('~'), 'data/synch_data/PHASE_SHAPES_{}/{}/0'.format(bgrnd_name, ns))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        generate_shapes(10000,num_shapes=ns,bgrnd_phase=bgrnd_phase,save_dir=save_dir)
