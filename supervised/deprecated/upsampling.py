import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm


def upscale(image, mask, zoom):
    # image.shape=(img_side, img_side)
    # mask.shape=(group, img_side ** 2)
    # pass in one image at one time
    old_size = image.shape[-1]
    new_size = old_size * zoom

    image = data['image'].reshape(-1)

    new_image = np.zeros((new_size ** 2,))
    new_mask = np.zeros((mask.shape[0], new_size ** 2))

    for ind in range(old_size ** 2):
        x_ = int(ind % old_size)
        y_ = int(ind // old_size)
        new_inds = [x_ * zoom + y_ * zoom * new_size,
                    x_ * zoom + 1 + y_ * zoom * new_size,
                    x_ * zoom + (y_ * zoom + 1) * new_size,
                    x_ * zoom + 1 + (y_ * zoom + 1) * new_size]
        for new_ind in new_inds:
            new_image[new_ind] = image[ind]
        for group in range(data['mask'].shape[2]):
            if data['mask'][0][0][group][ind] == 1:
                for new_ind in new_inds:
                    new_mask[group][new_ind] = 1
    return new_image.reshape(new_size, new_size), new_mask


load_path = '/media/data_cifs/yuwei/osci_save/data/328'
save_path = '/media/data_cifs/yuwei/osci_save/data/up328'

if not os.path.exists(save_path):
    os.mkdir(save_path)

zoom = 2

for i in tqdm.tqdm(range(100)):
    data = np.load(load_path + '/train_' + str(i) + '.npz')

    image = data['image'][0]
    mask = data['mask'][0][0]

    new_image, new_mask = upscale(image, mask, zoom)

    new_image = np.expand_dims(new_image, axis=0)
    new_mask = np.expand_dims(np.expand_dims(new_mask, axis=0), axis=0)

    np.savez(save_path + '/train_' + str(i), image=new_image, mask=new_mask)
    im = plt.imshow(new_image[0], cmap='gray')
    plt.savefig(save_path + '/canvas_' + str(i))
