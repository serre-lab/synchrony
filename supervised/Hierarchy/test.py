import argparse
import configparser
import kura_visual as kv
from utils import *
import net
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.style.use('seaborn-darkgrid')


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, help='name of experiment')
parser.add_argument('-m', '--model', type=str, help='name of model')
args = parser.parse_args()

cfg = configparser.ConfigParser()
cfg.read('./params.cfg')
experiment = cfg[args.name]

test_data_num = 1000
batch_size = 20
show_every = 5 # show after every several batch
episodes = 15
record_step = 15
# manually set data path
data_path = '/media/data_cifs/yuwei/osci_save/data/1000-types-mnist/multi-mnist10-0.1-sd0.5/3/test/0'
test_order = np.arange(test_data_num).reshape(-1, batch_size)

img_side = int(experiment['img_side'])
group_size = int(experiment['group_size'])
model_name = experiment['model']
out_channels = int(experiment['out_channels'])
start_filts = int(experiment['start_filts'])
depth = int(experiment['depth'])
num_cn = int(experiment['num_cn'])
split = int(experiment['split'])
kernel_size = experiment['kernel_size']
kernel_size = (int(kernel_size[0]), int(kernel_size[2]))
num_global_control = int(experiment['num_global_control'])
degree = int(experiment['degree'])
kura_update_rate = float(experiment['kura_update_rate'])
anneal = float(experiment['anneal'])
rp_field = experiment['rp_field']
main_path = experiment['main_path']
save_path = os.path.join(main_path, args.name)
checkpoint = tc.load(os.path.join(save_path, args.model), map_location=tc.device('cuda'))

test_save_path = os.path.join(save_path, 'test-{}'.format(args.model))
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

model = net.load_net(model_name, 1, out_channels, start_filts, depth, img_side,
                     num_cn, split, kernel_size, num_global_control).to('cuda')
criterion = net.criterion(degree=degree)
regularizer = net.regularizer()

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

rand_phase = checkpoint['initial_phase'].squeeze(0) # Change this line to random initialization when test random walk
connectivity = checkpoint['connectivity'].squeeze(0)
global_connectivity = checkpoint['gconnectivity']

batch_connectivity = connectivity.repeat(batch_size, 1, 1).to('cuda')
batch_initial_phase = rand_phase.repeat(batch_size, 1).to('cuda')

displayer = kv.displayer()

if num_global_control == 0:
    global_connectivity = None
    batch_gconnectivity = None
else:
    batch_gconnectivity = global_connectivity.repeat(batch_size, 1, 1).to('cuda')


displayer = kv.displayer()

tavg_norm = np.sum(np.arange(1, int(experiment['record_step']) + 1) ** 2)
loss_history = 0
regular_history = 0

for i in tqdm(range(test_order.shape[0])):
    images, masks = read_data(test_order[i], data_path, img_side, group_size, valid=True)
    phase_list, coupling = model(images.unsqueeze(1), kura_update_rate, anneal, episodes,
                                 batch_initial_phase, batch_connectivity, record_step,
                                 test=True, device='cuda', global_connectivity=batch_gconnectivity)
    # Only estimate the last frame
    loss_history += (criterion([phase_list[-1]], masks, 'cuda', valid=True).mean() / tavg_norm).cpu().data.numpy()
    if num_global_control == 0:
        regular_history += regularizer(coupling, masks, 'cuda').mean()
    if (i % show_every) == 0:
        display(displayer, phase_list, images, masks, coupling, img_side, group_size,
                test_save_path, str(int(i / show_every)))

loss_history = loss_history/test_order.shape[0]
regular_history = regular_history/test_order.shape[0]

if num_global_control > 0:
    regular_history = None

file = open(test_save_path + "/perform.txt", "w")
L = ['{} data test performance:{}\n'.format(test_data_num, loss_history),
     'Basset regularizer:{}'.format(regular_history)]
file.writelines(L)
file.close()
