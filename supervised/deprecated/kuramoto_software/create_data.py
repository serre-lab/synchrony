from make_data_new import polyomino_scenes
import numpy as np
import net
import matplotlib.pyplot as plt

saved = True
while saved:
    generator = polyomino_scenes(n=3, num_objects=2, img_side=8, batch_size=1, rotation=True)
    train_data = generator.generate_batch()
    train_img = net.convert2twod(train_data[0])
    train_mask = net.mask_pro(train_data[1][0])
    if np.min(np.sum(train_mask, axis=2), axis=1) != 0:
        np.savez('/media/data_cifs/yuwei/osci_save/data/train_data328.npz', image=train_data[0], mask=train_mask)
        plt.imshow(train_data[0][0])
        plt.savefig('/media/data_cifs/yuwei/osci_save/data/train_data328.png')
        saved = False
