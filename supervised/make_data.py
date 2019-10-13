DISPLAY=False
if not DISPLAY:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
else:
    import matplotlib.pyplot as plt
import numpy as np
from itertools import imap, imap, groupby, chain, imap
from operator import itemgetter
from sys import argv
from array import array
import os
import ipdb

class polyomino_scenes(object):
    def __init__(self, n, image_side, num_objects, batch_size=32, rotations=True):
        self.n = n
        self.image_side = image_side
        self.num_objects = num_objects
        self.n_ominoes = self.generate_free_polyominoes(n)
        self.rotations = rotations
        self.batch_size = batch_size
       
        if rotations:
            num_ominoes  = [1,1,2,7,18,60,60,196,704,2500,9189,33896,126759]
        else:
            num_ominoes = [1,1,2,5,12,35,108,369,1285,4655,17073,63600]

        self.num_n_ominoes = num_ominoes[n - 1]

    def generate_free_polyominoes(self,n):
        fp = []
        for poly in rank(n):
            bgrnd=np.zeros((max([i for (i,j) in poly]) + 1, max([j for (i,j) in poly]) + 1))
            for coord in poly:
                bgrnd[coord] = 1.0
            fp+=[bgrnd]
        return fp

    def generate_batch(self):
        batch = []
        dicts = []
        for b in range(self.batch_size):
            sameness_dict = {}
            diff_inds = []
            bgrnd = np.zeros((self.image_side, self.image_side))
            canvas = bgrnd.copy()
            num_same =  int(np.ceil( self.num_objects * np.random.rand()) + 1) if self.n > 1 else self.num_objects

            # place first
            same_ind = np.random.randint(len(self.n_ominoes))
            same_poly = self.n_ominoes[same_ind]
            coord = [np.random.randint(self.image_side - same_poly.shape[0]), np.random.randint(self.image_side - same_poly.shape[1])]
            template = bgrnd.copy()
            template[coord[0]:coord[0]+same_poly.shape[0], coord[1]:coord[1]+same_poly.shape[1]] = same_poly
            canvas+=template

            local_coords = np.where(same_poly==1.0)
            global_coords = [local_coords[0] + coord[0], local_coords[1] + coord[1]]
            sameness_dict['same_group'] = global_coords
            for m in range(self.num_objects - 1):
                
                if m < num_same - 1:
                    poly = same_poly
                else:
                    ind = np.random.randint(len(self.n_ominoes))
                    ind = ind-1 if ind==same_ind or ind in diff_inds else ind
                    diff_inds.append(ind)
                    poly = self.n_ominoes[ind]
                overlap = True
                while overlap:
                    template = bgrnd.copy()
                    if self.rotations:
                        poly = np.rot90(poly,np.random.randint(4))
                    coord = [np.random.randint(self.image_side - poly.shape[0]), np.random.randint(self.image_side - poly.shape[1])]
                    template[coord[0]:coord[0]+poly.shape[0], coord[1]:coord[1]+poly.shape[1]] = poly
                    overlap = np.max(template + canvas) > 1.0
                    if not overlap:
                        local_coords = np.where(poly==1.0)
                        global_coords = [local_coords[0] + coord[0], local_coords[1] + coord[1]]
                        key = 'diff_group_{}'.format(m-num_same+1) if m >= num_same - 1 else 'same_group'
                        if key == 'same_group':
                            sameness_dict[key][0] = np.concatenate((sameness_dict[key][0], global_coords[0]),0)
                            sameness_dict[key][1] = np.concatenate((sameness_dict[key][1], global_coords[1]),0)
                        else:
                            sameness_dict[key] = global_coords
                canvas+=template
            batch.append(canvas)
            bgrnd_coords = np.where(canvas == 0.0)
            sameness_dict['bgrnd'] = bgrnd_coords
            dicts.append(sameness_dict)
        return np.array(batch), dicts

def concat_map(func, it):
    return list(chain.from_iterable(imap(func, it)))

def minima(poly):
    """Finds the min x and y coordiate of a Polyomino."""
    return (min(pt[0] for pt in poly), min(pt[1] for pt in poly))

def translate_to_origin(poly):
    (minx, miny) = minima(poly)
    return [(x - minx, y - miny) for (x, y) in poly]

rotate90   = lambda (x, y): ( y, -x)
rotate180  = lambda (x, y): (-x, -y)
rotate270  = lambda (x, y): (-y,  x)
reflect    = lambda (x, y): (-x,  y)

def rotations_and_reflections(poly):
    """All the plane symmetries of a rectangular region."""
    return (poly,
            map(rotate90, poly),
            map(rotate180, poly),
            map(rotate270, poly),
            map(reflect, poly),
            [reflect(rotate90(pt)) for pt in poly],
            [reflect(rotate180(pt)) for pt in poly],
            [reflect(rotate270(pt)) for pt in poly])

def canonical(poly):
    return min(sorted(translate_to_origin(pl)) for pl in rotations_and_reflections(poly))

def unique(lst):
    lst.sort()
    return map(next, imap(itemgetter(1), groupby(lst)))

# All four points in Von Neumann neighborhood.
contiguous = lambda (x, y): [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

def new_points(poly):
    """Finds all distinct points that can be added to a Polyomino."""
    return unique([pt for pt in concat_map(contiguous, poly) if pt not in poly])

def new_polys(poly):
    return unique([canonical(poly + [pt]) for pt in new_points(poly)])

monomino = [(0, 0)]
monominoes = [monomino]

def rank(n):
    """Generates polyominoes of rank n recursively."""
    assert n >= 0
    if n == 0: return []
    if n == 1: return monominoes
    return unique(concat_map(new_polys, rank(n - 1)))

def triominos(number=3, width=16):
    # create 3 random orientated triominos random place on 16*16 area
    area = np.zeros((width, width))

    elem1 = np.zeros((3, 3))
    elem1[0, 1] = 1
    elem1[1, 1] = 1
    elem1[2, 1] = 1
    # print(elem1)

    elem2 = np.zeros((3, 3))
    elem2[0, 1] = 1
    elem2[1, 1] = 1
    elem2[1, 2] = 1
    # print(elem2)

    elem3 = np.zeros((3, 3))
    elem3[1, 0] = 1
    elem3[1, 1] = 1
    elem3[2, 1] = 1
    # print(elem3)

    area[6:9, 3:6] = elem1
    area[11:14, 11:14] = elem2
    area[1:4, 10:13] = elem3
    return area

def tetrominos(width=32):
    area = np.zeros((width, width))

    elem1 = np.zeros((4, 4))
    elem1[1, 1] = 1
    elem1[1, 2] = 1
    elem1[2, 1] = 1
    elem1[2, 2] = 1

    elem2 = np.zeros((4, 4))
    elem2[1, 0] = 1
    elem2[2, 0] = 1
    elem2[2, 1] = 1
    elem2[2, 2] = 1

    elem3 = np.zeros((4, 4))
    elem3[0, 1] = 1
    elem3[1, 1] = 1
    elem3[1, 2] = 1
    elem3[2, 2] = 1

    elem4 = np.zeros((4, 4))
    elem4[1, 0] = 1
    elem4[1, 1] = 1
    elem4[1, 2] = 1
    elem4[2, 2] = 1

    elem5 = np.zeros((4, 4))
    elem5[1, 0] = 1
    elem5[1, 1] = 1
    elem5[2, 1] = 1
    elem5[2, 2] = 1

    area[3:7, 1:5] = elem1
    area[13:17, 15:19] = elem2
    area[1:5, 25:29] = elem3
    area[17:21, 22:26] = elem4
    area[24:28, 3:7] = elem5

    return area

def save_as_np(area):
    np.save('./area', area)

if __name__ == '__main__':
    #area = triominos()
    #plt.imshow(area)
    #plt.show(a
    generator = polyomino_scenes(1,4,2,rotations=False)
    batch, dicts = generator.generate_batch()
    fig,axes = plt.subplots(2,2)
    for a, ax in enumerate(axes.reshape(-1)):
        ax.imshow(batch[a])
    plt.savefig(os.path.join(os.path.expanduser('~'), 'polyominoes.png'))
