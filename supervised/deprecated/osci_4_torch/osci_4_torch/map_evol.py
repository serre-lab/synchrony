import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sys


test_img = np.concatenate([np.zeros((1, 2)), np.ones((1, 2))], axis=0)
maps = np.squeeze(np.load('./maps.npy', allow_pickle=True))
# maps = np.array((np.clip(maps, -1, 1) + 1) * 255 / 2, np.int32)
print(maps.shape)
"""
plt.imshow(maps[0])
plt.colorbar()
plt.show()
raise ValueError
"""

fig = plt.figure(figsize=(7, 4))
ax1 = fig.add_subplot(121)
ax1.axis('off')
t_im = plt.imshow(test_img, cmap='gray')
ti1 = ax1.title.set_text('test image')

ax2 = fig.add_subplot(122)
im = plt.imshow(maps[0, :, :], vmin=-1, vmax=1)
cbar = plt.colorbar(ax=ax2, ticks=[-1, 0, 1])
cbar.ax.set_yticklabels(['-1', '0', '1'])
ti2 = ax2.title.set_text('couplings')

ti3 = plt.suptitle('training steps: ' + str(0))


def _update(d, image, title):
    image.set_array(maps[d + 1, :, :])

    title.set_text('training steps: ' + str((d + 1)))
    sys.stdout.write('\r' + 'Step: ' + str(d))
    sys.stdout.flush()
    return image, title,


ani = animation.FuncAnimation(fig, _update, frames=99, repeat=False, fargs=(im, ti3))

writer = animation.PillowWriter(fps=8)
ani.save('./map_evolution.gif', writer=writer)
writer.finish()
