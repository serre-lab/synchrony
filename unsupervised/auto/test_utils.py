from colorsys import hls_to_rgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from matplotlib import animation as anim
import numpy as np
import ipdb
def colorize(z):
    if len(z.shape) > 2:
	z = np.squeeze(z)
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + np.pi) / (2* np.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb)(h,l,s)
    c = np.stack(c, axis=-1)
    return c
	
def cplx_imshow(ax, z, cm=None):
    if cm == None:
        cm = plt.cm.hsv
    return ax.imshow(colorize(z), aspect='equal', interpolation='nearest', cmap=cm)
def real_imshow(ax, z, cm):
    return ax.imshow(z, aspect='equal', interpolation='nearest', cmap=cm)
def save_cplx_anim(filename, batch, fps=5, number=1, cplx=True, type='mp4'):
    ''' 
    anim should be the module matplotlib.animation
    z has shape NHW or NHW1 with complex values.
    '''
    for i in range(number):
	fn = filename + str(i) + '.' + type
	z = np.squeeze(batch[i,:,:,:])
        fig = plt.figure(figsize=(5, 5))
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = cplx_imshow(ax, z[0], plt.cm.hsv) if cplx else real_imshow(ax, z[0], plt.cm.gist_gray)
        def init(): return [im]
        def animate(i):
            im.set_array(colorize(z[i])) if cplx else im.set_array(z[i])
            return [im]
        a = anim.FuncAnimation(fig, animate, init_func=init, frames=len(z), interval=1, blit=True)
        if type == 'gif':
            a.save(fn, fps=fps, writer='imagemagick')
        elif type == 'mp4':
            a.save(fn, fps=fps)
        plt.close('all')

def generate_shape_img(width, height, nr_shapes=3, pad_width = 0):

    square = np.array(
    [[1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1]])

    triangle = np.array(
    [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
     [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    shapes = [square, triangle, triangle[::-1, :].copy()]
    img = np.zeros((height, width))

    for i in range(nr_shapes):
        shape = shapes[np.random.randint(0, len(shapes))]
        sy  , sx = shape.shape
        x = np.random.randint(0, width-sx+1)
        y = np.random.randint(0, height-sy+1)
        region = (slice(y,y+sy), slice(x,x+sx))
        img[region][shape != 0] += 1
        
    return np.expand_dims(np.pad(img, pad_width, 'constant'), -1)
 
def generate_bars_img(width, height, num_bars=6, pad_width=0,broken=False):

    img = np.zeros((height, width))

    rand_locs_h = np.random.permutation(height)[0:num_bars]
    rand_locs_v = np.random.permutation(width)[0:num_bars]

    img[rand_locs_h,:] = 1
    img[:,rand_locs_v] = 1

    if broken:
        img[0::4,:] = 0
	img[:,0::4] = 0

    return np.expand_dims(np.pad(img, pad_width, 'constant'), -1)
