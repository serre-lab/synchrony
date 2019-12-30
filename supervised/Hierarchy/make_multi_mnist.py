import gzip
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm


"""
1. This is a 2-mnist data generator which includes digits overlapping, digits are cropped from the original dataset with
their bounding box in the size of 20*20. They will be created along with their masks which indicate one specific digits
and background at a time.
2. The overlapping area proportion is calculated with respect to the total size of bounding box size of 400,
eg. [0., 0.8] means overlapping areas are taken with equal chance to be within the range of 0~320 pixels.
3. Validation set uses the same data as training, but they will be randomly generated respectively. However, test set
will use the data from the original test set of mnist data, and allow any overlapping range to keep the diversity
4. Data are saved as both npy file and png file, the png files will show both the input images and masks and can be
turned off by set the bool argument 'save_as_png' False
"""


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


where_is_zip_file = '/media/data_cifs/yuwei/osci_save/data/mnist'
where_to_save = '/media/data_cifs/yuwei/osci_save/data'
save_as_png = True

data_mode = 'test'
if (data_mode == 'valid') | (data_mode == 'test'):
    data_num = 1000
elif data_mode == 'train':
    data_num = 60000
else:
    raise ValueError

if (data_mode == 'valid') | (data_mode == 'train'):
    file_name = 'train'
elif data_mode == 'test':
    file_name = 't10k'
else:
    raise ValueError

image_size = 28  # dont change
num_digits = 10  # how many digits to include, eg. 3: 0,1,2 or 10: all digits
overlap_area = [0., 0.1]
if data_mode == 'test':
    buffer_size = 10000
    num_types = 800
else:
    buffer_size = 60000
    num_types = 800  # how many types for each image
start = 0 # set this for the start number of file name when you want to create more data into a specific dataset
same_diff = True # if include same difference task

if same_diff:
    prob = 0.5 # probability of rejecting a pair of different digits
else:
    prob = 0.

# organize the file name into a pytorch loader style
f_image = gzip.open(where_is_zip_file + '/{}-images-idx3-ubyte.gz'.format(file_name), 'r')

save_dir = where_to_save + '/1000-types-mnist/multi-mnist10-0.1-sd0.5/'
save_path = save_dir + '3/{}/0'.format(data_mode)
if not os.path.exists(save_path):
    os.makedirs(save_path)

f_image.read(16)
buf = f_image.read(image_size * image_size * buffer_size)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32).reshape(buffer_size, image_size, image_size)

f_label = gzip.open(where_is_zip_file + '/{}-labels-idx1-ubyte.gz'.format(file_name), 'r')
f_label.read(8)
labels = []
for j in range(buffer_size):
    buf_label = f_label.read(1)
    labels.append(np.frombuffer(buf_label, dtype=np.uint8)[0])

count = np.zeros(num_digits)
save_images = []
save_labels = []
for m in range(buffer_size):
    numbers = np.arange(num_digits)
    if labels[m] in numbers:
        if count[labels[m]] < num_types:
            count[labels[m]] += 1
            save_images.append(data[m])
            save_labels.append(labels[m])
if (count != num_types).any():
    raise ValueError('Buffer should be larger')

n = 0
pbar = tqdm.tqdm(total=data_num + 1)
while n < data_num:
    num = (np.random.randint(len(save_images)), np.random.randint(len(save_images)))
    if same_diff:
        if save_labels[num[0]] != save_labels[num[1]]:
            if np.random.rand() > prob:
                canvas, mask1, mask2 = \
                    non_overlap(save_images[num[0]], save_images[num[1]], save_labels[num[0]], save_labels[num[1]], 36)
            else:
                continue
        else:
            canvas, mask1, mask2 = \
                non_overlap(save_images[num[0]], save_images[num[1]], save_labels[num[0]], save_labels[num[1]], 36)
    else:
        if save_labels[num[0]] != save_labels[num[1]]:
            canvas, mask1, mask2 = \
                non_overlap(save_images[num[0]], save_images[num[1]], save_labels[num[0]], save_labels[num[1]], 36)
        else:
            continue
    extra_mask = np.array(mask1 + mask2 == 0).astype('uint8')
    mask1_ = mask1 * 255
    mask2_ = mask2 * 255
    extra_mask_ = extra_mask * 255
    mask_sum = np.clip(mask1 + mask2, 0, 1)
    if all((mask_sum.reshape(-1) != 0) == (canvas.reshape(-1) != 0)):
        if save_as_png:
            plt.imshow(np.concatenate([canvas, mask1_, mask2_, extra_mask_], axis=1), cmap='gray')
            plt.title('({}, {})'.format(save_labels[num[0]], save_labels[num[1]]))
            plt.savefig(save_path + '/img_{}'.format(str(n + start).rjust(4, '0')) + '.png')
            plt.close()

        save_data = np.concatenate([np.expand_dims(canvas, axis=0),
                                    np.expand_dims(extra_mask, axis=0),
                                    np.expand_dims(mask1, axis=0),
                                    np.expand_dims(mask2, axis=0)],
                                   axis=0)
        np.save(save_path + '/img_{}'.format(str(n + start).rjust(4, '0')) + '.npy', save_data)
        n += 1
        pbar.update(1)
pbar.close()
