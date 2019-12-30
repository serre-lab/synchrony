import net
import time
import argparse
import matplotlib
from torch import nn
import kura_visual as kv
import configparser
from utils import *
matplotlib.use('Agg')
plt.style.use('seaborn-darkgrid')

"""
Clean redundancy
Add configparser and load model,
now you can stop the training and pick it up with the saved the weights and parameters
"""

######################
# argument parser: if_cuda, data path, save path
parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-n', '--name', type=str, help='name of experiment')
parser.add_argument('-m', '--model', type=str, default=None, help='name of model')
args = parser.parse_args()

args.device = tc.device('cuda')
gpu_num = tc.cuda.device_count()

######################
# parameters
cfg = configparser.ConfigParser()
cfg.read('./params.cfg')
if not cfg.has_section(args.name):
    raise ValueError('Experiment has not been configured')
experiment = cfg[args.name]

img_side = int(experiment['img_side'])
group_size = int(experiment['group_size'])
num_cn = int(experiment['num_cn'])
num_global_control = int(experiment['num_global_control'])
sw = experiment.getboolean('sw')
p_rewire = float(experiment['p_rewire'])
rp_field = experiment['rp_field']
train_data_num = int(experiment['train_data_num'])
train_batch_size = int(experiment['train_batch_size'])
cv_batch_size = int(train_batch_size / 8)
real_train_data_num = train_data_num - int(train_data_num % train_batch_size)
train_epochs = int(experiment['train_epochs'])
show_every = int(experiment['show_every'])

kura_update_rate = float(experiment['kura_update_rate'])

model_name = experiment['model']
depth = int(experiment['depth'])
start_filts = int(experiment['start_filts'])
out_channels = int(experiment['out_channels'])
split = int(experiment['split'])
kernel_size = experiment['kernel_size']
kernel_size = (int(kernel_size[0]), int(kernel_size[2]))

episodes = int(experiment['episodes'])
record_step = int(experiment['record_step'])
anneal = float(experiment['anneal'])
perturb = float(experiment['perturb'])

learning_rate = float(experiment['learning_rate'])
regular_weight = float(experiment['regular_weight'])
degree = int(experiment['degree'])
shuffle = experiment.getboolean('shuffle')
keep_valid_data = experiment.getboolean('keep_valid_data')

main_path = experiment['main_path']
data_path = experiment['data_path']

load_dir = data_path
print("Data from: {}".format(load_dir))

load_path = load_dir + '/{}/train/0'.format(group_size)
save_path = main_path + '/{}'.format(args.name)

if not os.path.exists(save_path):
    os.makedirs(save_path)

######################
# initialization
if gpu_num > 1:
    model = nn.DataParallel(net.load_net(model_name, 1, out_channels, start_filts, depth, img_side,
                                         num_cn, split, kernel_size, num_global_control)).to(args.device)
    criterion = nn.DataParallel(net.criterion(degree=degree)).to(args.device)
    regularizer = nn.DataParallel(net.regularizer()).to(args.device)
    print("Assigned {} GPUs".format(gpu_num))
else:
    model = net.load_net(model_name, 1, out_channels, start_filts, depth, img_side,
                         num_cn, split, kernel_size, num_global_control).to(args.device)
    criterion = net.criterion(degree=degree).to(args.device)
    regularizer = net.regularizer().to(args.device)
    print("Assigned on {}".format(args.device))

print('network contains {} parameters'.format(net.count_parameters(model))) # parameter number
time.sleep(2)

displayer = kv.displayer()
op = tc.optim.Adam(model.parameters(), lr=learning_rate)

if args.model is not None:
    checkpoint = tc.load(save_path + args.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    op.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss_history = checkpoint['train_loss']
    loss_cv_history = checkpoint['valid_loss']
    model.train()

    rand_phase = checkpoint['initial_phase']
    connectivity = checkpoint['connectivity']
    global_connectivity = checkpoint['gconnectivity']
else:
    start_epoch = 0
    loss_history = []
    loss_cv_history = []
    initial_phase = np.random.rand(1, img_side ** 2 + num_global_control) * 2 * np.pi
    rand_phase = tc.tensor(initial_phase).to('cpu')
    connectivity, global_connectivity = \
        generate_connectivity(num_cn, img_side, sw=sw,
                              num_global_control=num_global_control, p_rewire=p_rewire, rp_field=rp_field)

connectivity = tc.tensor(connectivity).long().unsqueeze(0).to('cpu')
batch_connectivity = connectivity.repeat(train_batch_size, 1, 1).to(args.device)
cv_connectivity = connectivity.repeat(cv_batch_size, 1, 1).detach().to(args.device)

if global_connectivity is None:
    batch_gconnectivity = None
    cv_gconnectivity = None
else:
    global_connectivity = tc.tensor(global_connectivity).long().unsqueeze(0).to('cpu')
    batch_gconnectivity = global_connectivity.repeat(train_batch_size, 1, 1).to(args.device)
    cv_gconnectivity = global_connectivity.repeat(cv_batch_size, 1, 1).detach().to(args.device)

# generate initial phases
batch_initial_phase = rand_phase.repeat(train_batch_size, 1).to(args.device)
cv_initial_phase = rand_phase.repeat(cv_batch_size, 1).detach().to(args.device)

######################
# training pipeline
if not shuffle:
    training_order = np.arange(train_data_num)
    np.random.shuffle(training_order)
    training_order = training_order[:real_train_data_num].reshape(-1, train_batch_size)

tavg_norm = np.sum(np.arange(1, record_step + 1) ** 2)

if keep_valid_data:
    cv_ordering = np.arange(128)
    np.random.shuffle(cv_ordering)
    cv_image, cv_mask = read_data(cv_ordering[:cv_batch_size], load_dir + '/{}/valid/0'.format(group_size),
                                  img_side, group_size, valid=True)

for epoch in tqdm(range(train_epochs)):
    epoch += start_epoch + 1
    if shuffle:
        training_order = np.arange(train_data_num)
        np.random.shuffle(training_order)
        training_order = training_order[:real_train_data_num].reshape(-1, train_batch_size)

    for step in range(training_order.shape[0]):
        # just add a small perturbation
        rand_phase += perturb * tc.rand(1, img_side ** 2 + num_global_control) * 2 * np.pi
        rand_phase = rand_phase % (2 * np.pi)
        batch_initial_phase = rand_phase.repeat(train_batch_size, 1).to(args.device)
        cv_initial_phase = rand_phase.repeat(cv_batch_size, 1).detach().to(args.device)

        batch, mask_train = read_data(training_order[step], load_path,
                                      img_side, group_size, valid=False)

        op.zero_grad()
        phase_list_train, coupling_train = model(batch.unsqueeze(1), kura_update_rate, anneal, episodes,
                                                 batch_initial_phase, batch_connectivity,
                                                 record_step=record_step, test=False, device=args.device,
                                                 global_connectivity=batch_gconnectivity)

        tavg_loss = criterion(phase_list_train, mask_train, args.device).mean() / tavg_norm
        if num_global_control == 0:
            tavg_loss += regular_weight * regularizer(coupling_train, mask_train, args.device).mean()

        tavg_loss.backward()
        op.step()

        if step == training_order.shape[0] - 1:
            # cross-validation
            if not keep_valid_data:
                cv_ordering = np.arange(128)
                np.random.shuffle(cv_ordering)
                cv_image, cv_mask = read_data(cv_ordering[:cv_batch_size], load_dir + '{}/valid/0'.format(group_size),
                                              img_side, group_size, valid=True)

            phase_list_cv, coupling_cv = model(cv_image.unsqueeze(1), kura_update_rate, anneal, episodes,
                                               cv_initial_phase, cv_connectivity,
                                               record_step=record_step, test=False, device=args.device,
                                               global_connectivity=cv_gconnectivity)

            tavg_loss_cv = criterion(phase_list_cv, cv_mask, args.device, True).mean() / tavg_norm
            if num_global_control == 0:
                tavg_loss_cv += regular_weight * regularizer(coupling_cv, cv_mask, args.device).mean()

            # record and plot loss history
            loss_cv_history.append(tavg_loss_cv.cpu().data.numpy())
            loss_history.append(tavg_loss.cpu().data.numpy())

            plt.plot(np.array(loss_history))
            plt.plot(np.array(loss_cv_history))
            plt.title('Time Averaged Loss')
            plt.legend(['train', 'valid'])
            plt.savefig(save_path + '/loss' + '.png')
            plt.close()

        # visualize training
        if (step == 0) & ((epoch == 0) | ((epoch + 1) % show_every == 0)):
            display(displayer, phase_list_train, batch, mask_train, coupling_train,
                    img_side, group_size, save_path, 'train{}'.format(epoch))

    # visualize validation and save
    if (epoch == 0) | ((epoch + 1) % show_every == 0):
        # validation example, save its coupling matrix
        display(displayer, phase_list_cv, cv_image, cv_mask, coupling_cv,
                img_side, group_size, save_path, 'valid{}'.format(epoch))

        # save files
        if gpu_num > 1:
            tc.save({'epoch': epoch, 'train_loss:': loss_history, 'valid_loss': loss_cv_history,
                     'model_state_dict': model.module.state_dict(),
                     'optimizer_state_dict': op.state_dict(),
                     'initial_phase': rand_phase,
                     'connectivity': connectivity,
                     'gconnectivity': global_connectivity},
                    save_path + '/model{}.pt'.format(epoch))
        else:
            tc.save({'epoch': epoch, 'train_loss:': loss_history, 'valid_loss': loss_cv_history,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': op.state_dict(),
                     'initial_phase': rand_phase,
                     'connectivity': connectivity,
                     'gconnectivity': global_connectivity},
                    save_path + '/model{}.pt'.format(epoch))
