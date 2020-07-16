import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import torch
import numpy as np
import ipdb
from tqdm import tqdm
import os
from random import choice

def partition(number):
    answer = set()
    answer.add((number, ))
    for x in range(1, number):
        for y in partition(number - x):
            answer.add(tuple(sorted((x, ) + y)))
    return answer

def multi_mnist(num_digits, 
                num_images=60000,
                grouping='object',
                kind='training',
                max_overlap=.25,				
                display=True):

    save_dir = '/media/data_cifs/yuwei/osci_save/data/nonoverlapping_masks_Multi_MNIST/{}_grouping/overlap_{}/{}/{}/0'.format(grouping,max_overlap,num_digits+1,kind)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    mnist_imgs, mnist_labels = torch.load(os.path.join(os.path.expanduser('~'), 'data/synch_data/MNIST/processed/{}.pt'.format(kind)))
    num_mnist   = mnist_imgs.shape[0]
    im_side     = 28 + (num_digits - 1)*8

    data        = []
    labels      = []
	
    pts = list(partition(num_digits))

    counter = 0
    pbar = tqdm(total=num_images)
    while counter < num_images:
        all_labels=[]
        all_masks = []
          
        canvas     = torch.zeros(im_side, im_side)
        current_num_digits = 0
        current_pt = choice(pts) if grouping is not 'object' else (1,)*num_digits
        timeout = 0		
        classes = [i for i in range(10)]
        instances = [i for i in range(num_mnist)]
        for group_num in current_pt:
            group_labels = []
            group_digits = []
            if grouping == 'object':
                ind = choice(instances)
                group_digits.append(mnist_imgs[ind])
                group_labels.append(mnist_labels[ind])
            elif grouping == 'instance':
                ind = choice(instances)
                instances.remove(ind)
                group_digits += [mnist_imgs[ind]]*group_num
                group_labels += [mnist_labels[ind]]
            elif grouping == 'class':
                rand_class = choice(classes)
                classes.remove(rand_class)
                mnist_class = mnist_imgs[mnist_labels==rand_class] 
                inds = np.random.randint(0,len(mnist_class), (group_num,))
                group_digits += [mnist_class[ind] for ind in inds]
                group_labels += [torch.tensor(rand_class)]
            # Add digits
            mask = torch.zeros(im_side, im_side)			
            for digit in group_digits:

                overlap=True
                timeout = 0
                while overlap:
                    digit_canvas = torch.zeros(im_side, im_side)
                    timeout +=1
                    if timeout > 100:
                        break	                			
                    digit_yn  = torch.randint(0,im_side - 28, (1,))
                    digit_xn  = torch.randint(0,im_side - 28, (1,))
                    digit_canvas[digit_yn:digit_yn+28, digit_xn:digit_xn+28] = digit
                    if current_num_digits == 0:
                        overlap = False
                    else:
                        mass = ((canvas > 0)*1.).sum() / current_num_digits
                        overlap = (((digit_canvas > 0)*1 + (canvas > 0)*1) > 1).float().sum() > max_overlap * mass
                if timeout > 100:
                    continue
                canvas += digit_canvas
                canvas = torch.clamp(canvas, 0, 255)		   
                current_num_digits +=1
                mask+= 1.*(digit_canvas > 0)
                mask = torch.clamp(mask, 0, 1)
                
            if timeout > 100:
                break			
            all_masks.append(mask)
            all_labels += group_labels

        if timeout > 100:
            #print('Generation timed out. Consider using larger max overlap.')            
            continue

        if len(all_masks) > 1:
            overlap = all_masks[0]*all_masks[1] > 0
            all_masks[0][overlap] = 0.0

        background = 1. - 1*(canvas > 0)
        all_masks.append(background)
        all_labels.append(torch.tensor(-1))
		
        all_masks += [torch.zeros(im_side, im_side)]*(num_digits - len(current_pt))
        all_labels += [torch.tensor(-2)]*(num_digits - len(current_pt))
        final_datum = torch.cat((canvas.unsqueeze(0), torch.stack(all_masks)))
        if final_datum.shape[0] < 4:
            ipdb.set_trace()
        #data.append(final_datum)
        labels.append(torch.stack(all_labels))
      
        if display and counter == 0:
            
            fig, axes = plt.subplots(num_digits + 2)
            for a, ax in enumerate(axes.reshape(-1)):
                ax.imshow(final_datum.numpy()[a])
            plt.savefig(os.path.join(os.path.expanduser('~'),'tmp{}.png'.format(num_digits)))
            plt.close()
            ipdb.set_trace()
        file_name = os.path.join(save_dir, 'img_%06d.npy' % counter)
        
        np.save(file_name, final_datum.numpy())
        counter+=1
        pbar.update(1)
    #data = torch.stack(data)
    #labels = torch.stack(labels)
    #torch.save((data, labels), os.path.join(save_dir, 'processed.pt'.format(num_digits)))

if __name__=='__main__':
    # TODO: CURRENTLY ONLY WORKS WITH 2 DIGITS!!!
    display = False
    for nd in [2]:
        for grouping in ['class', 'instance', 'object']:
            for kind in ['test', 'training']:
                print('Making {} digit {}-grouped {} images'.format(nd, grouping, kind))
                multi_mnist(nd, num_images=250000, grouping=grouping, kind=kind, max_overlap=0.25, display=display)
