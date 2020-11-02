import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')
from matplotlib import cm
import torch
import numpy as np
from data_generators import GMM, Spirals, Circles, Moons
from train_model import Kuramoto_features, KuraTrainer, KuraODE, load_data, laplacian
import networkx as nx
from networkx.algorithms import community
import argparse
import os, csv
import ipdb

def string_to_array(string):
    array = []
    float_inds = [i for i in range(len(string)) if string.startswith('.', i)] 
    for fi in float_inds:
        array.append(float(string[fi - 1 : fi + 7]))
    return np.array(array)

def phase_evol(phases, masks, save_name=None):
        phases = np.squeeze(np.array(phases, dtype=np.float32))
        phases = (phases % ((2*np.pi) + 2*np.pi)) % (2*np.pi)
        if masks is not None:
            group_num = masks.shape[1]
            annot_bool = [True] * group_num
            colors = {}
            for num in range(group_num):
                mean_angle = np.arctan2((np.sin(phases[-1]) * masks[0][num]).sum(),
                                        (np.cos(phases[-1]) * masks[0][num]).sum() + 1e-6)
                colors[num] = num / float(group_num)#(mean_angle + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
            for i in range(phases.shape[1]):
                for num in range(group_num):
                    if masks[0][num][i]:
                        plt.plot(phases[:, i], color=plt.cm.hsv(colors[num]), alpha=0.5)
                        if annot_bool[num]:
                            plt.text(phases.shape[0] + 0.01, phases[-1, i], 'group{}'.format(num+1),
                                     horizontalalignment='left', size='small', color=plt.cm.hsv(colors[num]))
                            annot_bool[num] = False
            plt.xlabel('Time')
            plt.ylabel('Phase')
            plt.title('Phase Evolution')
            plt.ylim([0,2*np.pi])
        else:
            for i in range(phases.shape[1]):
                plt.plot(phases[:, i])
            plt.xlabel('Time')
            plt.ylabel('Phase')
            plt.title('Phase Evolution')
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(save_name + '.png')
        else:
            plt.show()
        plt.close()

def phase_evol_ax(ax, phases, masks, save_name=None):
    phases = np.squeeze(np.array(phases, dtype=np.float32))
    phases = (phases % ((2*np.pi) + 2*np.pi)) % (2*np.pi)
    if masks is not None:
        group_num = masks.shape[1]
        annot_bool = [True] * group_num
        colors = {}
        for num in range(group_num):
            mean_angle = np.arctan2((np.sin(phases[-1]) * masks[0][num]).sum(),
                                    (np.cos(phases[-1]) * masks[0][num]).sum() + 1e-6)
            colors[num] = num / float(group_num)#(mean_angle + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
        for i in range(phases.shape[1]):
            for num in range(group_num):
                if masks[0][num][i]:
                    ax.plot(phases[:, i], color=plt.cm.hsv(colors[num]), alpha=0.5)
                    if annot_bool[num]:
                        ax.text(phases.shape[0] + 0.01, phases[-1, i], 'group{}'.format(num+1),
                                 horizontalalignment='left', size='small', color=plt.cm.hsv(colors[num]))
                        annot_bool[num] = False
        ax.set_xlabel('Time')
        ax.set_ylabel('Phase')
        ax.set_title('Phase Evolution')
        ax.set_ylim([0,2*np.pi])
    else:
        for i in range(phases.shape[1]):
            ax.set_plot(phases[:, i])
        ax.set_xlabel('Time')
        ax.set_ylabel('Phase')
        ax.set_title('Phase Evolution')

if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--name',type=str, default='asymmetric_GMM5')
    argparser.add_argument('--seed',type=int, default=0)
    args = argparser.parse_args()

    load_dir = '/media/data_cifs/projects/prj_synchrony/results/models/mixture'
    fig_dir = '/media/data_cifs/projects/prj_synchrony/results/matt_results/mixture/figures'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    result_fn = os.path.join(load_dir, args.name, 'result.csv') 

    with open(result_fn, mode='r') as infile:
        reader = csv.reader(infile)
        exp_dict = {rows[0]:rows[1] for rows in reader}

    # Fix data types
    loss = string_to_array(exp_dict['loss'])
    conflict = string_to_array(exp_dict['conflict'])
    #loss = np.zeros(2000)
    batch_size = int(exp_dict['batch_size'])
    m = int(exp_dict['num_components']) if 'num_components' in exp_dict.keys() else 2
    num_hid = int(exp_dict['num_hid'])
    #device = exp_dict['device']
    device='cpu'
    adjoint = exp_dict['adjoint']
    symmetric = True if exp_dict['symmetric'][0] == 'T' else False

    Model = Kuramoto_features(batch_size=batch_size, num_hid=num_hid, symmetric=symmetric,
                              input_size=2, device=device, adjoint=adjoint).to(device)
    state_dict = torch.load(os.path.join(load_dir, args.name, 'model.pt'))
    del state_dict['ODEDynamic.couplings']
    Model.load_state_dict(state_dict)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = load_data(**exp_dict)

    ## PLOTTING ##
    a, idx = generator.sample(batch_size)
    a = a.float().to(device)
    sorted_idx = np.argsort(idx)

    triu_ind = np.triu_indices(batch_size,k=1)
    tril_ind = np.tril_indices(batch_size,k=-1)

    masks = torch.FloatTensor(batch_size,m).zero_()
    masks.scatter_(1,idx.unsqueeze(1).long(),1).transpose_(1,0).to(device)
    wb_inds = np.einsum('gi,gj->ij', masks,masks)
    in_wb = wb_inds[triu_ind]
    out_wb = wb_inds[tril_ind]

    ecolors = [cm.jet(x.item()*10) for x in idx]

    init_phase = torch.normal(np.pi, 1.0, (batch_size,)).float().unsqueeze(0).to(device)
    phase, couplings  = Model(init_phase,a)
    a = a.to('cpu')
    in_cpl = couplings[triu_ind].detach().cpu().numpy()
    out_cpl = couplings[tril_ind].detach().cpu().numpy()

    fig = plt.figure(figsize=(20,10),constrained_layout=True)
    spec4 = fig.add_gridspec(ncols=3, nrows=2)
    ax = fig.add_subplot(spec4[0, 0], polar=True)
    ax.set_title('Phases at t_final')
    ax.scatter(phase[-1].squeeze(0).detach().cpu().numpy(), np.ones(len(idx)), c=idx, alpha=0.75)
    ax = fig.add_subplot(spec4[0, 1])
    ax.set_title('Classification')
    ecolors = [cm.jet(x.item()*10) for x in idx]
    colors = [cm.hsv(x) for x in (phase[-1].squeeze().detach().cpu().numpy()%(2*np.pi))/(2*np.pi)]
    ax.scatter(a[:,0].cpu().numpy(),a[:,1].cpu().numpy(),c=colors,edgecolors=ecolors,linewidth=2,s=200)
    ax = fig.add_subplot(spec4[1,0])
    ax.set_title('Couplings matrix')
    sorted_couplings = couplings[sorted_idx,:]
    sorted_couplings = sorted_couplings[:,sorted_idx].detach().cpu().numpy()
    ax.imshow(sorted_couplings)
    #ax.colorbar()
    plt.axis('off')
    plt.grid(b=None)
    ax = fig.add_subplot(spec4[1,1])
    ax.set_title('Histogram of couplings')
    ax.hist(couplings.reshape(-1).detach().cpu().numpy(), 100, density=True)
    ax = fig.add_subplot(spec4[0,2])
    phase_evol_ax(ax,phase.detach().cpu().numpy(),masks.unsqueeze(0).data.cpu().numpy())
    ax = fig.add_subplot(spec4[1,2])
    ax.set_title('Loss function')
    ax.plot(loss)
    plt.savefig(os.path.join(fig_dir, args.name +  '_summary.png'))
    plt.close()

    fig, axes = plt.subplots(2,3, figsize=(20,12))

    pairwise_dist = np.sqrt(((a.reshape(batch_size,1,2) - a.reshape(1,batch_size,2))**2).sum(-1))
    in_dist = pairwise_dist[triu_ind]
    out_dist = pairwise_dist[tril_ind]

    pairwise_disp = (a.reshape(batch_size,1,2) - a.reshape(1,batch_size,2))

    pairwise_x_disp = pairwise_disp[...,0]
    in_x = pairwise_x_disp[triu_ind]
    out_x = pairwise_x_disp[tril_ind]

    pairwise_y_disp = pairwise_disp[...,1]
    in_y = pairwise_y_disp[triu_ind]
    out_y = pairwise_y_disp[tril_ind]

    ind_vars = [in_dist, in_x, in_y, out_dist, out_x, out_y]
    dep_vars = [in_cpl, out_cpl]
    all_wb = [in_wb, out_wb]

    colors = ['r','b']
    wb_labels = ['Between', 'Within']
    x_labels = ['Euclidean distance', 'X displacement', 'Y displacement']

    for a, ax in enumerate(axes.reshape(-1)):
        ind_var = ind_vars[a]
        dep_var = dep_vars[0] if a < 3 else dep_vars[1]
        wb_ind = all_wb[0] if a < 3 else all_wb[1]
        direction = 'In' if a < 3 else 'Out'
        for w, wb in enumerate(np.unique(wb_inds)):
            wb_ind_var = ind_var[wb_ind==wb]
            wb_cpl = dep_var[wb_ind==wb]
            ax.scatter(wb_ind_var, wb_cpl, label=wb_labels[w], color=colors[w], alpha=.05)
        ax.set_xlabel(x_labels[a % 3])
        ax.set_ylabel('{}-Coupling Strength'.format(direction))
        ax.legend()

    plt.savefig(os.path.join(fig_dir, args.name +  '_scatters.png'))
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(conflict)
    ax.set_ylabel('Conflict')
    ax.set_xlabel('Iterations')
    plt.savefig(os.path.join(fig_dir, args.name + '_conflict.png'))
    plt.close()

    if symmetric:
        num_batches = 100
        evals = []
        evals_control = []
        min_evals = []
        min_evals_control = []
        max_evals = []
        max_evals_control = []
        unit_evals = []
        unit_evals_control = []
        for _ in range(num_batches):
            print('Spectrum for batch: {}'.format(_))
            batch, idx = generator.sample(batch_size)
            phase, couplings  = Model(init_phase,batch.float())
            np_couplings = couplings.detach().cpu().numpy()
            control_coupling = np.random.uniform(np_couplings.min(), np_couplings.max(), size=np_couplings.shape)
            control_coupling = .5*(control_coupling + control_coupling.transpose(1,0))
    
            L = laplacian(couplings, batch=False, sym_norm=True).detach().cpu().numpy()
            L_control = laplacian(torch.tensor(control_coupling), sym_norm=True, batch=False).cpu().numpy()

            batch_evals = np.sort(np.linalg.eigvals(L))
            batch_evals_control = np.sort(np.linalg.eigvals(L_control))
            evals.append(batch_evals)
            evals_control.append(batch_evals_control)

            min_evals.append(batch_evals[0])
            min_evals_control.append(batch_evals_control[0])

            max_evals.append(batch_evals[-1])
            max_evals_control.append(batch_evals_control[-1])

            unit_evals.append(((np.abs(batch_evals - 1) < 1e-3)*1).sum())
            unit_evals_control.append(((np.abs(batch_evals_control - 1) < 1e-3)*1).sum())


        evals = np.array(evals).reshape(-1)
        evals_control = np.array(evals_control).reshape(-1)
        min_evals = np.array(min_evals).mean()
        min_evals_control = np.array(min_evals_control).mean()
        max_evals = np.array(max_evals).mean()
        max_evals_control = np.array(max_evals_control).mean()
        unit_evals = np.array(unit_evals).mean()
        unit_evals_control = np.array(unit_evals_control).mean()

        bins = np.linspace(0,2,500)
        plt.hist(evals, color='b', bins=bins, alpha=.4, density=True)
        plt.hist(evals_control, color='r',bins=bins, alpha=.4, density=True)

        textstr = '\n'.join((
                       r'$\lambda_1=%.2f$' % (min_evals, ),
                       r'$\bar{\lambda}_1=%.2f$' % (min_evals_control, ),
                       r'$\lambda_n=%.2f$' % (max_evals, ),
                       r'$\bar{\lambda}_n=%.2f$' % (max_evals_control, ),
                       r'$ \langle\{\lambda=1\}\rangle=%.2f$' % (unit_evals,),
                       r'$\langle\{\bar{\lambda} = 1\}\rangle=%.2f$' % (unit_evals_control,)))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(.2, 80, textstr, bbox=props)
        plt.legend(('Optimized', 'Random'))
        plt.savefig(os.path.join(fig_dir, args.name + '_spectra.png'))
  
        plt.close()

