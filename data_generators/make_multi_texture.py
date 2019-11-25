import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from PIL import Image
import numpy as np
import os
import ipdb
from tqdm import tqdm

def rand_crop(img, side=256):
    sz = img.shape[:-1]
    crop_corner = [np.random.randint(dim - side) for dim in sz]
    return img[crop_corner[0]:crop_corner[0] + side, crop_corner[1]:crop_corner[1] + side, :]

def generate(number, data_kind='train', img_side=256, texture_number=4, gray=True, display=False, save_dir='/media/data_cifs/yuwei/osci_save/data/composite_textures_mini'):
    data_dir = os.path.join(os.path.expanduser('~'), 'data', 'dtd', 'images')
    save_dir = os.path.join(save_dir, str(texture_number), data_kind, '0')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    text_type = os.listdir(os.path.join(data_dir))
    images_per_type = [len(os.listdir(os.path.join(data_dir, text_type[i]))) for i in range(len(text_type))]
    channels = 1 if gray else 3
    for n in tqdm(range(number)):
        rand_type_ind = np.random.randint(0,len(text_type), size=texture_number)
        rand_type_name = [text_type[i] for i in rand_type_ind]
        img_names = [os.listdir(os.path.join(data_dir, name))[np.random.randint(images_per_type[i])] for (name, i) in zip(rand_type_name, rand_type_ind)]
        imgs = [np.asarray(Image.open(os.path.join(data_dir, type_name, file_name))) for (type_name, file_name) in zip(rand_type_name, img_names)]
        imgs = [rand_crop(img,side=img_side) for img in imgs]
        # Voronoi mask for image
        yy = np.linspace(0,img_side-1, img_side).astype(int)
        xx = np.linspace(0,img_side-1, img_side).astype(int)
        grid = np.meshgrid(yy,xx)

        points = np.random.randint(0, img_side, size=(texture_number, 2)) 
        dists = []
        for p in points: 
            dists.append(np.sqrt((p[0] - grid[0])**2 + (p[1] - grid[1])**2))
        dists = np.array(dists) 
        masks = [np.argmin(dists,axis=0) == i for i in range(len(points))]
        composite = np.array([np.expand_dims(mask,-1)*image for (mask,image) in zip(masks, imgs)]).sum(0)
        composite = np.expand_dims(composite.mean(-1), axis=0) if gray else composite
        if display:
            plt.imshow(composite[0,:,:])
            plt.savefig(os.path.join(os.path.expanduser('~'), 'texture.png'))
            ipdb.set_trace()
        img_and_mask = np.concatenate((composite, np.array(masks)), axis=0) 
        ipdb.set_trace()
        file_name = os.path.join(save_dir, 'img_%04d.npy' % n)
        np.save(file_name, img_and_mask)
if __name__=='__main__':
    num_textures = [2,3,4,5]
    num_images = 10000
    for num in num_textures:
        for kind in ['train', 'test']:
            print('Generating {} texture {} images'.format(str(num), kind))
            generate(num_images,data_kind=kind,img_side=16, texture_number=num, display=False)
