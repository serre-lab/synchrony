def read_data(data_inds, path, valid=False):
    images = np.zeros((len(data_inds), img_side, img_side))
    masks = np.zeros((len(data_inds), group_size, img_side ** 2))
    b = 0
    for data_ind in data_inds:
        data = np.load(path + '/img_' + str(data_ind).rjust(4, '0') + '.npy')
        image = data[0, ...].reshape(1, img_side, img_side)
        mask = data[1:, ...].reshape(1, group_size, img_side ** 2)

        images[b, ...] = image
        masks[b, ...] = mask
        b += 1
    if not valid:
        return tc.tensor(images).float().to(device), tc.tensor(masks).float().to(device)
    else:
        return tc.tensor(images).float().detach().to(device), tc.tensor(masks).float().detach().to(device)


def show(displayer, phase_list, image, mask, coupling, path, name):
    colored_mask = (np.expand_dims(np.expand_dims(np.arange(group_size), axis=0), axis=-1) * mask / group_size).sum(1)
    displayer.set_phases(phase_list)
    displayer.set_masks(mask)
    displayer.compute_properties()

    displayer.animate_evol_compare(img_side, img_side, compare=image, save_name=path + '/ani_' + name)
    displayer.static_evol(img_side, img_side, image, path + '/static_' + name, colored_mask)
    displayer.phase_evol2(path + '/phase_' + name)

    im = plt.imshow(coupling)
    plt.title('Coupling Matrix')
    plt.gca().grid(False)
    plt.axis('off')
    plt.colorbar(im)
    plt.savefig(path + '/coupling_' + name)
    plt.close()
	
def sw_connectivity():
    connectivity = np.zeros((img_side ** 2, num_cn))
	for i in tqdm(range(img_side ** 2)):
		count = 0
		for j in range(img_side ** 2):
			x_1 = int(i % img_side)
			y_1 = int(i // img_side)
			x_2 = int(j % img_side)
			y_2 = int(j // img_side)
			dist = np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)
			if (dist < critic_dist) and (dist > 0):
				connectivity[i, count] = j
				count += 1
		while count < num_cn:
			connectivity[i, count:] = \
				sorted(np.delete(range(img_side ** 2), connectivity[i, ...]),
					   key=lambda k: np.random.random())[:int(num_cn - count)]
			count += 1
		change = sorted(range(num_cn), key=lambda k: np.random.random())[:int(num_cn / 2 + 1)]
		connectivity[i, ...][change] = \
			sorted(np.delete(range(img_side ** 2), connectivity[i, ...]),
				   key=lambda k: np.random.random())[:int(num_cn / 2 + 1)]
    return connectivity