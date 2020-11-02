import subprocess

exp_names = []
exp_names += ['asymmetric_GMM{}'.format(i + 1) for i in range(10)]
exp_names += ['symmetric_GMM{}'.format(i + 1) for i in range(10)]
exp_names += ['asymmetric_{}'.format(str) for str in ['moons', 'spirals', 'circles']]
exp_names += ['symmetric_{}'.format(str) for str in ['moons', 'spirals', 'circles']]


for name in exp_names:
    try:
        subprocess.call('python visualize.py --name {}&'.format(name), shell=True)
    except:
        print('Oops! There is something wrong with {}!'.format(name))
