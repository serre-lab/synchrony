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


def generate(number, data_kind='train', img_side=256, texture_number=4, gray=True, display=False,
             save_dir='/media/data_cifs/yuwei/osci_save/data/textures'):
    data_dir = os.path.join('/media', 'data_cifs', 'dtd', 'images')
    save_dir = os.path.join(save_dir, str(texture_number), data_kind, '0')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    text_type = os.listdir(os.path.join(data_dir))
    images_per_type = [os.listdir(os.path.join(data_dir, text_type[i])) for i in range(len(text_type))]
    for images in images_per_type:
        if './directory' in images:
            images.remove('./directory')

    n = 0
    pbar = tqdm(total=number + 1)
    while n < number:
        rand_type_ind = np.random.randint(0, len(text_type), size=texture_number)
        img_names = [images_per_type[i][np.random.randint(len(images_per_type[i]))] for i in rand_type_ind]
        imgs = [np.asarray(Image.open(os.path.join(data_dir, text_type[i], file_name))) for (i, file_name) in zip(rand_type_ind, img_names)]
        imgs = [rand_crop(img, side=img_side) for img in imgs]
        # Voronoi mask for image
        yy = np.linspace(0, img_side-1, img_side).astype(int)
        xx = np.linspace(0, img_side-1, img_side).astype(int)
        grid = np.meshgrid(yy, xx)

        points = np.random.randint(0, img_side, size=(texture_number, 2))
        dists = []
        for p in points:
            dists.append(np.sqrt((p[0] - grid[0])**2 + (p[1] - grid[1])**2))
        dists = np.array(dists)
        masks = [np.argmin(dists, axis=0) == i for i in range(len(points))]

        areas = np.array([np.where(mask)[0].shape[0] for mask in masks])
        if ((np.abs(areas - areas.mean()) / areas.mean()) > 0.5).any():
            continue

        composite = np.array([np.expand_dims(mask, -1)*image for (mask, image) in zip(masks, imgs)]).sum(0)
        composite = np.expand_dims(composite.mean(-1), axis=0) if gray else composite
        if display:
            plt.imshow(composite[0, :, :])
            plt.savefig(os.path.join(save_dir, 'texture_%04d.png' % n))
            plt.close()
        img_and_mask = np.concatenate((composite, np.array(masks)), axis=0)
        file_name = os.path.join(save_dir, 'img_%04d.npy' % n)
        np.save(file_name, img_and_mask)
        n += 1
        pbar.update()
    pbar.close()


if __name__ == '__main__':
    generate(128, data_kind='test', img_side=64, texture_number=5, display=True)
