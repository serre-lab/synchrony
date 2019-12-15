import os
import net
import time
import argparse
import torch as tc
import numpy as np
from torch import nn
from tqdm import tqdm
import display as disp
import ipdb
import warnings
warnings.filterwarnings('ignore')
"""
DataParallel
kuramoto and loss_func_ex are integrated into an nn.Module to operate parallel calculation
"""

def Train(**kwargs):

    if interactive:
        import matplotlib.pyplot as plt
    else:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff()

    ######################
    # device
    if not disable_cuda:
        gpu_num = tc.cuda.device_count()
        print("Assigned {} GPUs".format(gpu_num))
        device = 'cuda'
    else:
        device = 'cpu'

    ######################
    # parameters
    num_cn = 8 # connected neighbours
    critic_dist = 2 # distance from the center pixel
	num_test = 1000
    real_train_data_num = train_data_num - int(train_data_num % batch_size)
    train_epochs = 200
    show_every = 20

    anneal = 0.5 # waiting to explore

   ######################
    # path
    load_dir = os.path.join('/media/data_cifs/yuwei/osci_save/data/', data_name, data_param_string)
    save_dir = os.path.join('/media/data_cifs/yuwei/osci_save/results/', exp_name)

    train_path = load_dir + '{}/train/0'.format(group_size)
    test_path = load_dir + '{}/test/0'.format(group_size)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
		
    ######################
    # Load data
	#training set
    training_set = datasets.DatasetFolder(train_path, np.load, extensions=('.npy',), transform=transform)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, drop_last=True)
    # testing set
    testing_set = datasets.DatasetFolder(test_path, np.load, extensions=('.npy',), transform=    transform)
    testing_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True,
	drop_last=True)
    
    ######################
    # save parameters
    file = open(save_path + "/params.txt", "w")
    L = [key + '= {}'.format(value) for (key, value) in kwargs.items]
    file.writelines(L)
    file.close()

    ######################
    # initialization
    if tc.cuda.device_count() > 1:
        model = nn.DataParallel(nets.load_net(model_name)).to(device)
        criterion = nn.DataParallel(nets.criterion()).to(device)
    else:
        model = nets.load_net(model_name).to(device)
        criterion = nets.criterion().to(device)
    print('network contains {} parameters'.format(nets.count_parameters(model))) # parameter number
    time.sleep(2)

    if not rand_initial_phase:
        initial_phase = np.random.rand(1, img_side ** 2) * 2 * np.pi
        rand_phase = tc.tensor(initial_phase).to('cpu')
        batch_initial_phase = rand_phase.repeat(batch_size, 1).to(device)
        test_initial_phase = rand_phase.repeat(batch_size, 1).detach().to(device)

    loss_history = []
    loss_test_history = []
    coupling_history = []

    displayer = disp.displayer(interactive=interactive)
    op = tc.optim.Adam(model.parameters(), lr=learning_rate)

    ######################
    # connectivity
    print('Generating fixed adjancency matrix.')
    if small_world:
        connectivity = sw_connectivity()
    else:
        connectivity = np.ones((img_side ** 2, num_cn))
    connectivity = tc.tensor(connectivity).long().unsqueeze(0).to('cpu')
    batch_connectivity = connectivity.repeat(batch_size, 1, 1).to(device)
    test_connectivity = connectivity.repeat(batch_size, 1, 1).detach().to(device)

    ######################
    # training pipeline
    #if not shuffle:
    #    training_order = np.arange(train_data_num)
    #    np.random.shuffle(training_order)
    #    training_order = training_order[:real_train_data_num].reshape(-1, batch_size)
    norm = np.sum(np.arange(1, time_steps + 1) ** 2)
	
counter = 0
for epoch in range(train_epochs):
    print('Epoch: {}'.format(epoch))

    for step, train_data in tqdm(enumerate(training_loader)):
        batch = train_data[0][:, 0, ...]
        mask = train_data[0][:, 1:, ...].reshape(-1, group_size, img_side * img_side)

        op.zero_grad()

        phase_list_train, coupling_train = model(batch.unsqueeze(1), device,
                                                 update_rate, anneal, time_steps,
                                                 batch_initial_phase, batch_connectivity)

        tavg_loss = criterion(phase_list_train, mask_train, device)
        tavg_loss = tavg_loss.mean() / norm
        tavg_loss += sparsity_weight * tc.abs(coupling_train).mean()

        tavg_loss.backward()
        op.step()
		
		# visualize training
        if counter % show_every == 0:
            train_ind = np.random.randint(batch_size)
            train_image = batch[train_ind].cpu().data.numpy()
            train_mask = mask_train[train_ind].cpu().unsqueeze(0).data.numpy()
            coupling_train_show = \
                tc.zeros(1, img_side ** 2, img_side ** 2).to('cpu').scatter_(dim=2, index=connectivity,
                                                        src=coupling_train[train_ind].unsqueeze(0).cpu()).data.numpy()[0]
            coupling_history.append(coupling_train_show)
            train_phase_list = np.array([phase.cpu().data.numpy()[train_ind, :] for phase in phase_list_train])
            show(displayer, train_phase_list, train_image, train_mask, coupling_train_show, save_path,
                    'train{}'.format(epoch))

	for step, test_data in tqdm(enumerate(testing_loader)):
        # cross-validation
		batch = test_data[0][:, 0, ...]
        mask = test_data[0][:, 1:, ...].reshape(-1, group_size, img_side * img_side)


        phase_list_test, coupling_test = model(test_image.unsqueeze(1), device,
                                               update_rate, anneal, time_steps,
                                               test_initial_phase, test_connectivity)

        # phase.shape=(time, batch, N)
        # mask.shape=(time, batch, group, N)
        # return loss.shape=(time * batch
		tavg_loss_test = criterion(phase_list_test, test_mask, device, True)
        tavg_loss_test = tavg_loss_test.mean() / norm
        tavg_loss_test += sparsity_weight * tc.abs(coupling_test).mean()
        loss_test_history.append(tavg_loss_test.cpu().data.numpy()
		loss_history.append(tavg_loss.cpu().data.numpy())
		
		if step*batch_size > num_test:
		    break

    # # visualize validation and save
    # if (epoch == 0) | ((epoch + 1) % show_every == 0):
        # # validation example, save its coupling matrix
        # valid_ind = np.random.randint(batch_size)
        # valid_image = cv_image[valid_ind].cpu().data.numpy()
        # valid_mask = cv_mask[valid_ind].cpu().unsqueeze(0).data.numpy()
        # coupling_valid_show = \
            # tc.zeros(1, img_side ** 2, img_side ** 2).to('cpu').scatter_(dim=2, index=connectivity,
                                                                         # src=coupling_cv[valid_ind].cpu().unsqueeze(
                                                                             # 0)).data.numpy()[0]
        # valid_phase_list = np.array([phase.cpu().data.numpy()[valid_ind, :] for phase in phase_list_cv])
        # show(displayer, valid_phase_list, valid_image, valid_mask, coupling_valid_show, save_path,
            # 'valid{}'.format(epoch))

    # save files
    tc.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': op.state_dict(),
                 'initial_phase': rand_phase,
                 'connectivity': connectivity}, save_path + '/model{}.pt'.format(epoch))
    np.save('coupling.npy', np.array(coupling_history))

    plt.plot(np.array(loss_history))
    plt.plot(np.array(loss_test_history))
    plt.title('Time Averaged Loss')
    plt.legend(['train', 'valid'])
    plt.savefig(save_path + '/loss' + '.png')
    plt.close()
