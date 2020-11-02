import torch
import numpy as np
from sklearn import datasets
from sklearn.mixture import GaussianMixture

class Moons(object):
    def __init__(self, noise = 0.05, random_state=1):
        self.noise = noise
        self.random_state = random_state
        self.n_components = 2

    def sample(self, n_samples=400):
        x, y = datasets.make_moons(n_samples,
                            noise = self.noise,
                             random_state=self.random_state)
        return [torch.tensor(x), torch.tensor(y)]

    def train_test(selfn_samples_trainn_samples_test):
        return self.sample(n_samples_train), self.sample(n_samples_test)
   
class Circles(object):
    def __init__(self, noise=.1, factor = 0.5, random_state=1):
        self.factor = factor
        self.random_state = random_state
        self.n_components = 2
        self.noise = noise

    def sample(self, n_samples=400):
        x, y = datasets.make_circles(n_samples,
                            factor = self.factor,
                            noise = self.noise,
                            random_state=self.random_state)
        return [torch.tensor(x), torch.tensor(y)]

class GMM(object):
    def __init__(self, n_components, centroids, cov_matrices):
        self.n_components = n_components
        self.centroids = centroids
        self.cov_matrices = cov_matrices

    def one_sampling(self,statistics):
        return np.random.multivariate_normal(statistics[0], statistics[1])

    def sample(self, n_samples):
        idx = np.random.randint(0, self.n_components,n_samples)
        list_stat = [[self.centroids[i],self.cov_matrices[i]] for i in idx]
        return [torch.tensor((np.vstack(list(map(self.one_sampling,list_stat))))), torch.tensor(idx)]

    def train_test(selfn_samples_trainn_samples_test):
        return self.sample(n_samples_train), self.sample(n_samples_test)

class Spirals(object):
    def __init__(self, noise):
        self.noise = noise
        self.n_components = 2


    def sample(self, n_samples):
        n_samples=int(n_samples/2)
        theta = np.sqrt(np.random.rand(n_samples))*2*np.pi # np.linspace(0,2*pi,100)
        r_a = 2*theta + np.pi
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        x_a = data_a + np.random.randn(n_samples,2)*self.noise

        r_b = -2*theta - np.pi
        data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
        x_b = data_b + np.random.randn(n_samples,2)*self.noise
        
        res_a = np.append(x_a, np.zeros((n_samples,1)), axis=1)
        res_b = np.append(x_b, np.ones((n_samples,1)), axis=1)

        res = torch.tensor(np.append(res_a, res_b, axis=0)).float()
        return [res[:,:2],res[:,2].long()]
