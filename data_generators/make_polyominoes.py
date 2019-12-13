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
import ipdb
from tqdm import tqdm
from itertools import groupby, chain
import os
try:
    from itertools import imap
except ImportError:
    imap = map

def partition(n, I=1):
    yield(n,)
    for i in range(I, n//2 + 1):
        for p in partition(n-i, i):
            yield(i,) + p

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
    def __init__(self, n, img_side, num_objects, free=True):
        self.n = n
        self.img_side = img_side
        self.num_objects = num_objects
        self.free = free
        self.n_ominoes = self.generate_polyominoes(self.n)
        self.num_n_ominoes = len(self.n_ominoes)

    def generate_polyominoes(self, n, free=False):
        fp = []
        for poly in rank(n):
            # pad with zeros
            bgrnd = np.zeros((max([i for (i, j) in poly]) + 1, max([j for (i, j) in poly]) + 1))
            for coord in poly:
                bgrnd[coord] = 1.0
            if not self.free:
                for n in range(4): 
                    rotated = np.rot90(bgrnd,n)
                    try:
                        fpl = [f.tolist() for f in fp]
                        ind = fpl.index(rotated.tolist())
                    except:
                        fp += [rotated]
            else:
                fp += [bgrnd]
        return fp

def generate(img_side=32, num_imgs=10000, n=4, num_objects=4, rotation=False, data_kind='train',save_dir='/media/data_cifs/yuwei/osci_save/data/polyominoes', display=False):
        rot_string = 'fixed' if rotation else 'free' 
        size_string = 'large' if img_side >= 32 else 'small'
        # Specify save dir
        save_dir = os.path.join(save_dir, str(n), str(num_objects), rot_string, size_string, data_kind)

        # Object which contains actual polyominoes
        generator = polyomino_scenes(n, img_side, num_objects, free=rotation)
       
        # List containing ways to group the `num_object` objects
        possible_groupings = list(partition(num_objects))
        # Possible number of groups in an image
        possible_group_nums = list(set([min(len(grouping), generator.num_n_ominoes) for grouping in possible_groupings]))

        # Organize groupings according to number of groupings
        sized_groupings = [[] for _ in possible_group_nums]
        for grouping in possible_groupings:
            l = len(grouping)
            try:
                sized_groupings[l -1].append(grouping)
            except:
                continue
        # Make file paths according to the number of groupings in an image
        file_paths = [os.path.join(save_dir, str(n + 1)) for n in possible_group_nums]
        for fp in file_paths:
            if not os.path.exists(fp):
                os.makedirs(fp)
        # For each possible group size and grouping with that size 
        for num_groups, groupings in enumerate(sized_groupings):

            # Generate `num_imgs` images
            for i in tqdm(range(num_imgs)):

                # Index of the individual polyomino types for given n
                inds = range(generator.num_n_ominoes)
                masks = [] # where maasks are stored
                bgrnd = np.zeros((img_side, img_side)) #image background
                canvas = bgrnd.copy()

                # Select this image's particular grouping from among the available grou[pings with this size
                grouping = groupings[np.random.randint(len(groupings))]

                # For each group
                for group in grouping:
                    # Get polyomino
                    ind = np.random.randint(len(inds))
                    poly = generator.n_ominoes[inds[ind]]
                    # Remove this index from the index list so that it is not selected by the other groups
                    inds.pop(ind)
                    mask = np.zeros((img_side, img_side))
                    # For each object in the group
                    for g in range(group):
                        # Rotate the polyomino of rotational congruence is declared
                        if rotation: poly = np.rot90(poly, np.random.randint(4))
                        overlap = True
                        overlap_counter = 0
                        while overlap:
                            template = bgrnd.copy()
                            # Sample location
                            coord = [np.random.randint(img_side - poly.shape[0]),
                                     np.random.randint(img_side - poly.shape[1])]
                            # Place polyomino
                            template[coord[0]:coord[0] + poly.shape[0], coord[1]:coord[1] + poly.shape[1]] = poly
                            # Check for overlap
                            overlap = np.max(template + canvas) > 1.0
                            overlap_counter+=overlap
                        canvas+= template
                        mask += template
                    # if no overlap, add object to canvas and add mask to the masks list
                    masks  += [mask]
                masks.append(1 - canvas) # the background is the final group
                #masks += [np.zeros_like(canvas) for _ in range(max_masks - len(masks))]
                data = np.array([canvas] + masks)
                if display:
                    fig, axes = plt.subplots(1, data.shape[0])
                    for a, ax in enumerate(axes):
                        ax.imshow(data[a])
                    plt.savefig(os.path.join(os.path.expanduser('~'), 'polyominoes.png'))
                    plt.close()
                fp = file_paths[num_groups]
                np.save(os.path.join(fp, 'img_%04d.npy' % i), data)
            
if __name__ == '__main__':
    ns = [3,4,5,6]
    num_objects = [2,3,4,5]
    num_imgs = 2500
    rotations = [True, False]
    data_kind = ['train', 'test']
    img_side = 16
    display = False
    for n in ns:
        for o in num_objects:
            for r in rotations:
                for d in data_kind:
                    print('Generating images: {}-ominoes, {} objects, {} rotations, {} set'.format(n, o, r, d))
                    generate(img_side=img_side, num_imgs=num_imgs, n=n, num_objects=o, rotation=r, data_kind=d, display=display)  
