import gzip
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm


def crop_digit(image):
    new_size = 20
    y_ = np.where(image != 0)[0]
    x_ = np.where(image != 0)[1]
    x_pad = new_size - (x_.max() - x_.min() + 1)
    y_pad = new_size - (y_.max() - y_.min() + 1)
    crop = image[y_.min():y_.max() + 1, x_.min():x_.max() + 1]
    crop = np.pad(crop,
                  pad_width=((int((y_pad + 1) / 2), int(y_pad / 2)), (int((x_pad + 1) / 2), int(x_pad / 2))),
                  mode='constant')
    # crop = cv2.resize(image, dsize=(new_size, new_size), interpolation=cv2.INTER_NEAREST)
    mask = np.array(crop != 0).astype('uint8')
    return crop, mask, (new_size, new_size)


def non_overlap(image1, image2, label1, label2, outsize):
    canvas = np.zeros((outsize, outsize))

    # crop digits and reshape
    crop1, local_mask1, size1 = crop_digit(image1)
    crop2, local_mask2, size2 = crop_digit(image2)

    # find the overlapping area of smaller than 64
    flag = True
    while flag:
        top_left1 = (np.random.randint(1, outsize - 20), np.random.randint(1, outsize - 20))
        top_left2 = (np.random.randint(1, outsize - 20), np.random.randint(1, outsize - 20))
        y_overlap = 20 - np.abs(top_left2[0] - top_left1[0])
        x_overlap = 20 - np.abs(top_left2[1] - top_left1[1])
        if (y_overlap * x_overlap < overlap_area[1] * (outsize ** 2)) \
                and (y_overlap * x_overlap > overlap_area[0] * (outsize ** 2)):
            flag = False

    bottom_right1 = (outsize - top_left1[0] - size1[0], outsize - top_left1[1] - size1[1])
    bottom_right2 = (outsize - top_left2[0] - size2[0], outsize - top_left2[1] - size2[1])

    # pad local masks to generate global masks
    if label1 != label2:
        global_mask1 = np.pad(local_mask1, ((top_left1[0], bottom_right1[0]), (top_left1[1], bottom_right1[1])),
                              'constant')
        global_mask2 = np.pad(local_mask2, ((top_left2[0], bottom_right2[0]), (top_left2[1], bottom_right2[1])),
                              'constant')
    else:
        pad_sum = np.pad(local_mask1, ((top_left1[0], bottom_right1[0]), (top_left1[1], bottom_right1[1])),
               'constant') + np.pad(local_mask2, ((top_left2[0], bottom_right2[0]), (top_left2[1], bottom_right2[1])),
                              'constant')
        global_mask1 = np.array(pad_sum != 0).astype('uint8')
        global_mask2 = np.zeros((outsize, outsize))

    # decide which digit at deeper depth
    deep = np.random.randint(2)
    joint = np.array(global_mask1 + global_mask2 > 1).astype('uint8')
    if deep == 0:
        global_mask1 -= joint
    else:
        global_mask2 -= joint

    canvas[top_left1[0]:top_left1[0] + size1[0], top_left1[1]:top_left1[1] + size1[1]] += crop1
    canvas[top_left2[0]:top_left2[0] + size2[0], top_left2[1]:top_left2[1] + size2[1]] += crop2
    return np.clip(canvas, 0, 255), global_mask1, global_mask2


f_image = gzip.open('/media/data_cifs/yuwei/osci_save/data/mnist/train-images-idx3-ubyte.gz', 'r')
save_dir = '/media/data_cifs/yuwei/osci_save/data/multi-mnist3/'
save_path = save_dir + '3/train/0'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    os.mkdir(save_dir + '3/')
if not os.path.exists(save_path):
    os.mkdir(save_path[:-1])
    os.mkdir(save_path)

image_size = 28
num_digits = 3
num_images = 5000
num_types = 20
overlap_area = [0.6, 0.8]

f_image.read(16)
buf = f_image.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32).reshape(num_images, image_size, image_size)

# inds = [0, 1, 2, 3, 4, 5, 7, 13, 15, 17]

f_label = gzip.open('/media/data_cifs/yuwei/osci_save/data/mnist/train-labels-idx1-ubyte.gz', 'r')
f_label.read(8)
labels = []
for j in range(num_images):
    buf_label = f_label.read(1)
    labels.append(np.frombuffer(buf_label, dtype=np.uint8)[0])


count = np.zeros(num_digits)
save_images = []
save_labels = []
for m in range(num_images):
    numbers = np.arange(num_digits)
    if labels[m] in numbers:
        if count[labels[m]] < num_types:
            count[labels[m]] += 1
            save_images.append(data[m])
            save_labels.append(labels[m])
if (count != num_types).any():
    raise ValueError('Buffer should be larger')

for i in tqdm.tqdm(range(10000)):
    num = (np.random.randint(len(save_images)), np.random.randint(len(save_images)))
    # np.random.shuffle(inds)
    # num = inds[:2]
    canvas, mask1, mask2 = \
        non_overlap(save_images[num[0]], save_images[num[1]], save_labels[num[0]], save_labels[num[1]], 36)
    extra_mask = np.array(mask1 + mask2 == 0).astype('uint8')
    mask1_ = mask1 * 255
    mask2_ = mask2 * 255
    extra_mask_ = extra_mask * 255
    mask_sum = np.clip(mask1 + mask2, 0, 1)
    if all((mask_sum.reshape(-1) != 0) == (canvas.reshape(-1) != 0)):
        plt.imshow(np.concatenate([canvas, mask1_, mask2_, extra_mask_], axis=1), cmap='gray')
        plt.title('({}, {})'.format(save_labels[num[0]], save_labels[num[1]]))
        plt.savefig(save_path + '/img_{}'.format(str(i).rjust(4, '0')) + '.png')
        plt.close()

        save_data = np.concatenate([np.expand_dims(canvas, axis=0),
                                    np.expand_dims(extra_mask, axis=0),
                                    np.expand_dims(mask1, axis=0),
                                    np.expand_dims(mask2, axis=0)],
                                   axis=0)
        np.save(save_path + '/img_{}'.format(str(i).rjust(4, '0')) + '.npy', save_data)
