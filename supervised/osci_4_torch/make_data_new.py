DISPLAY=False
if not DISPLAY:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
else:
    import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from itertools import imap, groupby, chain, imap
from sys import argv
from array import array
import ipdb

def concat_map(func, it):
    return list(chain.from_iterable(imap(func, it)))


def minima(poly):
    """Finds the min x and y coordiate of a Polyomino."""
    # print('poly' + str(poly))
    return (min([pt[0] for pt in poly]), min([pt[1] for pt in poly]))


def translate_to_origin(poly):
    (minx, miny) = minima(poly)
    return [(x - minx, y - miny) for (x, y) in poly]


rotate90   = lambda z: (z[1], -z[0])
rotate180  = lambda z: (-z[0], -z[1])
rotate270  = lambda z: (-z[1],  z[0])
reflect    = lambda z: (-z[0],  z[1])


def rotations_and_reflections(poly):
    """All the plane symmetries of a rectangular region."""
    output = (poly,
              list(map(rotate90, poly)),
              list(map(rotate180, poly)),
              list(map(rotate270, poly)),
              list(map(reflect, poly)),
              [reflect(rotate90(pt)) for pt in poly],
              [reflect(rotate180(pt)) for pt in poly],
              [reflect(rotate270(pt)) for pt in poly])
    # print('output:' + str(output))
    return output


def canonical(poly):
    return min(sorted(translate_to_origin(pl)) for pl in rotations_and_reflections(poly))


def unique(lst):
    lst.sort()
    return map(next, imap(itemgetter(1), groupby(lst)))


# All four points in Von Neumann neighborhood.
contiguous = lambda z: [(z[0] - 1, z[1]), (z[0] + 1, z[1]), (z[0], z[1] - 1), (z[0], z[1] + 1)]

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

class polyomino_scenes(object):
    def __init__(self, n, img_side, num_objects, batch_size, rotation=True):
        self.n = n
        self.img_side = img_side
        self.num_objects = num_objects
        self.batch = batch_size
        self.rotation = rotation
        self.n_ominoes = self.generate_free_polyominoes(self.n)
        self.num_n_ominoes = len(self.n_ominoes)

    def generate_free_polyominoes(self, n):
        fp = []
        for poly in rank(n):
            # pad with zeros
            bgrnd = np.zeros((max([i for (i, j) in poly]) + 1, max([j for (i, j) in poly]) + 1))
            for coord in poly:
                bgrnd[coord] = 1.0
            fp += [bgrnd]
        return fp

    def generate_batch(self):

        # record data
        batch = []
        # record group dict
        grouping = []

        for b in range(self.batch):
            dicts = {}
            for i in range(len(self.n_ominoes)):
                dicts[str(i)] = []
            dicts['bgrnd'] = []

            bgrnd = np.zeros((self.img_side, self.img_side))
            canvas = bgrnd.copy()

            # place first
            same_ind = np.random.randint(len(self.n_ominoes))
            same_poly = self.n_ominoes[same_ind]
            coord = [np.random.randint(self.img_side - same_poly.shape[0]),
                     np.random.randint(self.img_side - same_poly.shape[1])]
            template = bgrnd.copy()
            template[coord[0]:coord[0] + same_poly.shape[0], coord[1]:coord[1] + same_poly.shape[1]] = same_poly
            canvas += template

            local_coords = np.where(same_poly == 1.0)
            global_coords = [local_coords[0] + coord[0], local_coords[1] + coord[1]]

            dicts[str(same_ind)].extend(list(map(self._coord2idx, self.convert_coords(global_coords))))
            # sameness_dict['same_group'].append(global_coords)
            for n in range(self.num_objects - 1):
                ind = np.random.randint(len(self.n_ominoes))
                poly = self.n_ominoes[ind]
                overlap = True
                while overlap:
                    template = bgrnd.copy()
                    if self.rotation:
                        poly = np.rot90(poly, np.random.randint(4))
                    coord = [np.random.randint(self.img_side - poly.shape[0]),
                             np.random.randint(self.img_side - poly.shape[1])]
                    template[coord[0]:coord[0] + poly.shape[0], coord[1]:coord[1] + poly.shape[1]] = poly
                    overlap = np.max(template + canvas) > 1.0
                    if not overlap:
                        local_coords = np.where(poly == 1.0)
                        global_coords = [local_coords[0] + coord[0], local_coords[1] + coord[1]]
                        # key = 'diff_group_{}'.format(n - num_same + 1) if n >= num_same - 1 else 'same_group'
                        dicts[str(ind)].extend(list(map(self._coord2idx, self.convert_coords(global_coords))))
                canvas += template
            batch.append(canvas)
            bgrnd_coords = np.where(canvas == 0.0)
            dicts['bgrnd'].extend(list(map(self._coord2idx, np.transpose(bgrnd_coords))))
            grouping.append(dicts)
        return np.array(batch), grouping

    def _coord2idx(self, coord):
        return int(coord[0] * self.img_side + coord[1])

    def convert_coords(self, coords):
        new_coords = []
        for n in range(len(coords[0])):
            new_coords.append((coords[0][n], coords[1][n]))
        return new_coords


def generate_small(batch, num, img_side):
    """
    only image sides 2, monominoes, two objects
    :return:
    """
    batch_list = []
    grouping = []
    for _ in range(batch):
        dicts = {}
        dicts['0'] = []
        dicts['bgrnd'] = []

        canvas = np.zeros((img_side, img_side))
        while(True):
            ind = np.random.randint(img_side ** 2)
            if ind in dicts['0']:
                continue
            else:
                canvas[int(ind // img_side), int(ind % img_side)] = 1
                dicts['0'].append(ind)
                if len(dicts['0']) == num:
                    break
        bg_inds = np.where(np.reshape(canvas, newshape=(img_side ** 2,)) == 0)
        dicts['bgrnd'].extend(list(bg_inds[0]))
        batch_list.append(canvas)
        grouping.append(dicts)
    return batch_list, grouping


def generate_test_img(n, num, img_side):
    generator = polyomino_scenes(n, img_side, num, 1, True)
    return generator.generate_batch()


if __name__ == '__main__':
    generator = polyomino_scenes(5, 16, 4, 1, True)
    canvas, dicts = generator.generate_batch()

    print(canvas)
    print('\n'.join(map(str, dicts)))
