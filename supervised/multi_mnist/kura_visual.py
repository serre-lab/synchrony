import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import loss_func_ex
import copy


class displayer(object):
    """
    matplotlib 3.1.1
    """
    def __init__(self, phases=None, freqs=None, masks=None):
        if id(phases) != id(None) & id(freqs) != id(None):
            self.phases = np.squeeze(np.array(phases, dtype=np.float32))
            self.phases = (self.phases % (2 * np.pi) + 2 * np.pi) % (2 * np.pi)
            self.freqs = np.squeeze(np.array(freqs, dtype=np.float32))

            if masks is not None:
                self.fps = loss_func_ex.fpt_btw_groups_np(self.phases, masks)
                self.inps = loss_func_ex.inp_btw_groups_np2(self.phases, masks)
                self.chs = loss_func_ex.coh_btw_groups_np(self.phases, masks)
                self.abs = loss_func_ex.abs_angle_diffs_np(self.phases)
            else:
                self.fps = loss_func_ex.frame_pt_np(self.phases)
                self.inps = loss_func_ex.inp_np(self.phases)
                self.chs = loss_func_ex.coherence_np(self.phases)
                self.abs = loss_func_ex.abs_angle_diffs_np(self.phases)

            self.masks = masks

            assert len(self.phases.shape) == 2
            assert len(self.freqs.shape) == 2
            assert len(self.fps.shape) == 1
            assert len(self.inps.shape) == 1
            assert len(self.chs.shape) == 1
            assert len(self.abs.shape) == 1
        else:
            self.phases = None
            self.freqs = None
            self.fps = None
            self.inps = None
            self.abs = None
            self.chs = None
            self.masks = masks

    def animate_evol_compare(self, nrows, ncols, compare, save_name=None):
        if self.masks is None:
            raise ValueError('There are no masks provided')

        compare = np.squeeze(compare)

        fig = plt.figure()

        gs1 = gridspec.GridSpec(2, 1)
        gs2 = gridspec.GridSpec(2, 1)

        ax_cp = fig.add_subplot(gs1[0])
        cp_im = ax_cp.imshow(compare, cmap='gray')
        ax_cp.axis('off')

        ax_im = fig.add_subplot(gs1[1])
        im = ax_im.imshow(self._convert2square(self.phases[0], ncols=ncols, nrows=nrows),
                          cmap='hsv', animated=True)
        cbar = plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=2 * np.pi),
                                              cmap='hsv'),
                            ax=ax_im,
                            ticks=[np.pi / 2, np.pi, np.pi * 3 / 2],
                            fraction=0.05)
        cbar.ax.set_yticklabels(('0.5$\pi$', '$\pi$', '1.5$\pi$'))
        ax_im.axis('off')
        gs1.tight_layout(fig, rect=[0, 0, 0.5, 1])

        ax_sc = fig.add_subplot(gs2[0], polar=True)
        sc = ax_sc.scatter(self.phases[0],
                           np.ones_like(self.phases[0]),
                           vmin=0,
                           vmax=2 * np.pi,
                           cmap='hsv')

        plt.setp(ax_sc.xaxis.get_majorticklabels(), va="top",
                 rotation_mode="anchor", fontsize=8)
        ax_sc.grid(False)
        ax_sc.set_yticks([])
        ax_sc.set_ylim(0, 1.05)

        ax_tx = fig.add_subplot(gs2[1])
        tx1 = ax_tx.text(0.2, 0.2, 'frame potential:  ' + str(self.fps[0]), verticalalignment='center')
        tx2 = ax_tx.text(0.2, 0.4, 'coherence:  ' + str(self.chs[0]), verticalalignment='center')
        tx3 = ax_tx.text(0.2, 0.6, 'inner product:  ' + str(self.inps[0]), verticalalignment='center')
        tx4 = ax_tx.text(0.2, 0.8, 'synchrony loss:  ' + str(self.abs[0]), verticalalignment='center')
        ax_tx.axis('off')
        ax_tx.grid(False)
        gs2.tight_layout(fig, rect=[0.5, 0, 1, 1], h_pad=0.2)

        s = 'steps: 0'
        ti = fig.suptitle(s, verticalalignment='bottom',
                          fontsize=10, horizontalalignment='right')

        # plt.tight_layout()

        print('Making GIF')
        ani = animation.FuncAnimation(
            fig, self._ani_update, interval=100,
            frames=self.phases.shape[0], repeat=False,
            fargs=(im, sc, tx1, tx2, tx3, tx4, ti, nrows, ncols)
        )

        if save_name is not None:
            writer = animation.PillowWriter(fps=8)
            ani.save(save_name + '.gif', writer=writer)
            writer.finish()
            print('\nGIF saved')
        else:
            plt.show()
        plt.close()

    def animate_evol(self, nrows, ncols, save_name=None):
        fig = plt.figure()

        gs1 = gridspec.GridSpec(1, 1)
        gs2 = gridspec.GridSpec(1, 1)
        gs3 = gridspec.GridSpec(1, 1)
        gs4 = gridspec.GridSpec(1, 2)

        ax_im = fig.add_subplot(gs1[0])
        im = ax_im.imshow(self._convert2square(self.phases[0], ncols=ncols, nrows=nrows),
                          cmap='hsv', animated=True)
        s = 'steps: 0'
        ti1 = plt.title(s, verticalalignment='bottom',
                        fontsize=10)
        cbar = plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=2 * np.pi),
                                              cmap='hsv'),
                            ax=ax_im,
                            ticks=[np.pi / 2, np.pi, np.pi * 3 / 2],
                            fraction=0.05)
        cbar.ax.set_yticklabels(('0.5$\pi$', '$\pi$', '1.5$\pi$'))
        ax_im.axis('off')
        gs1.tight_layout(fig, rect=[0, 0.5, 0.5, 1])

        ax_sc = fig.add_subplot(gs2[0], polar=True)
        sc = ax_sc.scatter(self.phases[0],
                           np.ones_like(self.phases[0]),
                           vmin=0,
                           vmax=2 * np.pi,
                           cmap='hsv')

        plt.setp(ax_sc.xaxis.get_majorticklabels(), va="top",
                 rotation_mode="anchor", fontsize=6)
        ax_sc.grid(False)
        ax_sc.set_yticks([])
        ax_sc.set_ylim(0, 1.05)
        gs2.tight_layout(fig, rect=[0.5, 0.6, 1, 1])

        ax_tx = fig.add_subplot(gs3[0])
        tx1 = ax_tx.text(0.25, 0.05, 'frame potential:  ' + str(self.fps[0]), verticalalignment='center', fontsize=8)
        tx2 = ax_tx.text(0.25, 0.2, 'coherence:  ' + str(self.chs[0]), verticalalignment='center', fontsize=8)
        tx3 = ax_tx.text(0.25, 0.35, 'inner product:  ' + str(self.inps[0]), verticalalignment='center', fontsize=8)
        tx4 = ax_tx.text(0.25, 0.5, 'synchrony loss:  ' + str(self.abs[0]), verticalalignment='center', fontsize=8)
        ax_tx.axis('off')
        ax_tx.grid(False)
        gs3.tight_layout(fig, rect=[0.5, 0.5, 1, 0.7])

        ax_ph = fig.add_subplot(gs4[0])
        for i in range(self.phases.shape[1]):
            plt.plot(self.phases[:, i], 'red')
        plt.title('Phase Evolution', fontsize=8)

        ax_ls = fig.add_subplot(gs4[1])
        plt.plot(self.inps, 'orange')
        plt.plot(self.fps, 'b')
        plt.plot(self.chs, 'g')
        plt.plot(self.abs, 'c')
        plt.legend(['inner product', 'frame potential', 'coherence', 'synchrony'], prop={'size': 6})
        for tick in ax_ph.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        for tick in ax_ph.yaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        for tick in ax_ls.yaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        for tick in ax_ls.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        plt.title('properties', fontsize=8)
        gs4.tight_layout(fig, rect=[0, 0, 1, 0.5], w_pad=0.5)

        #plt.tight_layout()

        print('Making GIF')
        ani = animation.FuncAnimation(
            fig, self._ani_update, interval=100,
            frames=self.phases.shape[0], repeat=False,
            fargs=(im, sc, tx1, tx2, tx3, tx4, ti1, nrows, ncols)
        )

        if save_name is not None:
            writer = animation.PillowWriter(fps=8)
            ani.save(save_name + '.gif', writer=writer)
            writer.finish()
            print('\nGIF saved')
        else:
            plt.show()
        plt.close()

    def _convert2square(self, phase, nrows, ncols):
        return phase.reshape((nrows, ncols))

    def _ani_update(self, d, im, sc, tx1, tx2, tx3, tx4, ti, nrows, ncols):
        data = np.concatenate((self.phases[d].reshape(-1, 1),
                               np.ones_like(self.phases[d].reshape(-1, 1))),
                              axis=1)

        sc.set_offsets(data)
        sc.set_cmap('hsv')
        sc.set_color(sc.to_rgba(self.phases[d]))

        tx1.set_text('frame potential:  ' + str(self.fps[d]))
        tx2.set_text('coherence:  ' + str(self.chs[d]))
        tx3.set_text('inner product:  ' + str(self.inps[d]))
        tx4.set_text('synchrony loss:  ' + str(self.abs[d]))

        ti.set_text('steps: ' + str(d))

        im.set_array(self._convert2square(self.phases[d], nrows, ncols))

        sys.stdout.write('\r' + 'Step: ' + str(d))
        sys.stdout.flush()

        return im, sc, tx1, tx2, tx3, tx4, ti, nrows, ncols,

    def phase_evol(self, save_name=None):
        plt.style.use('seaborn-darkgrid')
        if self.masks is not None:
            group_num = self.masks.shape[1]
            annot_bool = [True] * group_num
            colorlib = ['orange', 'c', 'g', 'm', 'r', 'y', 'b']

            color_counter = 0
            colors = []
            for num in range(group_num):
                try:
                    colors.append('C{}'.format(num))
                    color_counter += 1
                except ValueError:
                    print('Need more colors!')
            for i in range(self.phases.shape[1]):
                for num in range(group_num):
                    if self.masks[0][num][i]:
                        plt.plot(self.phases[:, i], color=colors[num], alpha=0.5)
                        if annot_bool[num]:
                            plt.text(self.phases.shape[0] * (1 + 0.05), self.phases[-1, i], 'group{}'.format(num),
                                     horizontalalignment='left', size='small', color=colors[num])
                            annot_bool[num] = False

            color_counter = 0
            annot_bool = [True] * group_num
            plt.xlabel('Time')
            plt.ylabel('Phase')
            plt.title('Phase Evolution')
        else:
            for i in range(self.phases.shape[1]):
                plt.plot(self.phases[:, i])
            plt.xlabel('Time')
            plt.ylabel('Phase')
            plt.title('Phase Evolution')
        if save_name is not None:
            plt.savefig(save_name + '.png')
        else:
            plt.show()
        plt.close()

    def phase_evol2(self, save_name=None):
        plt.style.use('seaborn-darkgrid')
        if self.masks is not None:
            group_num = self.masks.shape[1]
            annot_bool = [True] * group_num
            colors = {}
            for num in range(group_num):
                mean_angle = np.arctan2((np.sin(self.phases[-1]) * self.masks[0][num]).sum(),
                                        (np.cos(self.phases[-1]) * self.masks[0][num]).sum() + 1e-6)
                colors[num] = (mean_angle + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
            for i in range(self.phases.shape[1]):
                for num in range(group_num):
                    if self.masks[0][num][i]:
                        plt.plot(self.phases[:, i], color=plt.cm.hsv(colors[num]), alpha=0.5)
                        if annot_bool[num]:
                            plt.text(self.phases.shape[0] + 0.01, self.phases[-1, i], 'group{}'.format(num),
                                     horizontalalignment='left', size='x-small', color=plt.cm.hsv(colors[num]))
                            annot_bool[num] = False
            plt.xlabel('Time')
            plt.ylabel('Phase')
            plt.title('Phase Evolution')
        else:
            for i in range(self.phases.shape[1]):
                plt.plot(self.phases[:, i])
            plt.xlabel('Time')
            plt.ylabel('Phase')
            plt.title('Phase Evolution')
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(save_name + '.png')
        else:
            plt.show()
        plt.close()

    def properties_evol(self, save_name=None):
        plt.style.use('seaborn-darkgrid')
        plt.plot(self.inps, 'orange')
        plt.plot(self.fps, 'b')
        plt.plot(self.chs, 'g')
        plt.plot(self.abs, 'c')
        plt.legend(['inner product', 'frame potential', 'coherence', 'synchrony'])
        if save_name is not None:
            plt.savefig(save_name + '.png')
        else:
            plt.show()

    def set_phases(self, phases=None):
        if phases is not None:
            self.phases = np.squeeze(np.array(phases, dtype=np.float32))
        else:
            self.phases = None
        self.phases = (self.phases % (2 * np.pi) + 2 * np.pi) % (2 * np.pi)
        return True

    def set_freqs(self, freqs=None):
        if freqs is not None:
            self.freqs = np.squeeze(np.array(freqs, dtype=np.float32))
        else:
            self.freqs = None
        return copy.deepcopy(self.freqs)

    def set_masks(self, masks=None):
        if masks is not None:
            self.masks = masks
            assert len(self.masks.shape) == 3
        else:
            self.masks = None
        return copy.deepcopy(self.masks)

    def compute_properties(self):
        if self.phases is not None:
            if self.masks is None:
                self.fps = loss_func_ex.frame_pt_np(self.phases)
                self.inps = loss_func_ex.inp_np(self.phases)
                self.chs = loss_func_ex.coherence_np(self.phases)
                self.abs = loss_func_ex.abs_angle_diffs_np(self.phases)
            else:
                self.fps = loss_func_ex.fpt_btw_groups_np(self.phases, self.masks)
                self.inps = loss_func_ex.inp_btw_groups_np2(self.phases, self.masks)
                self.chs = loss_func_ex.coh_btw_groups_np(self.phases, self.masks)
                self.abs = loss_func_ex.abs_angle_diffs_np(self.phases)

            assert len(self.fps.shape) == 1
            assert len(self.inps.shape) == 1
            assert len(self.chs.shape) == 1
            assert len(self.abs.shape) == 1
        else:
            raise ValueError('No phases provided')

    def static_evol(self, nrows, ncols, input, save_name, mask):
        interval = int(self.phases.shape[0] / 5)

        phase_list = [self.phases[i * interval] for i in range(5)]
        phase_list[-1] = self.phases[-1]

        fig, axes = plt.subplots(1, 7, figsize=(8, 4))
        axes.reshape(-1)
        axes[0].imshow(np.reshape(input, (nrows, ncols)), cmap='gray')
        axes[0].axis('off')
        axes[0].title.set_text('input')
        for i in range(5):
            axes[i + 1].imshow(np.reshape(phase_list[i], (nrows, ncols)), cmap='hsv')
            axes[i + 1].axis('off')
            if i == 4:
                axes[i + 1].title.set_text('last_step')
            else:
                axes[i + 1].title.set_text('step' + str(int(i * interval)))
        axes[6].imshow(np.reshape(mask, (nrows, ncols)), cmap='gray')
        axes[6].axis('off')
        axes[6].title.set_text('mask')
        cbar_ax = fig.add_axes([0.95, 0.31, 0.01, 0.38])
        plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=2 * np.pi),
                                       cmap='hsv'), cax=cbar_ax)
        ax2 = fig.add_axes([0.07, 0.15, 0.88, 0.1])
        xmin, xmax = ax2.get_xlim()
        ax2.arrow(xmin, 0, xmax - xmin, 0., fc='k', ec='k',
                  head_width=0.1, head_length=0.01,
                  length_includes_head=True, clip_on=False)
        ax2.text(0.46, 0.2, 'iterations', fontsize=12)
        ax2.set_yticks([])
        ax2.axis('off')
        plt.tight_layout(pad=3.5, w_pad=0.5, h_pad=0.6)
        plt.savefig(save_name + '.png')
        plt.close()

    def single_img(self, nrows, ncols, input, epoch, save_name):
        interval = int(self.phases.shape[0] / 5)

        phase_list = [self.phases[i * interval] for i in range(5)]
        for i in range(len(phase_list)):
            im = plt.imshow(np.reshape(phase_list[i], (nrows, ncols)), 'hsv')
            plt.colorbar(im)
            plt.title('step:' + str(int(i * interval)))
            plt.axis('off')
            plt.savefig(save_name + 's' + str(int(i * interval)) + '_e' + str(epoch) + '.png')
            plt.close()
        plt.imshow(np.reshape(input, (8, 8)), cmap='gray')
        plt.title('input')
        plt.axis('off')
        plt.savefig(save_name + 'e' + str(epoch) + '.png')
        plt.close()
