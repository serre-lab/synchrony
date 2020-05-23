#!/usr/bin/env python
# coding: utf-8

# # Exploring the Laplacian for Spectral Analysis and Graph Partitioning 
# 
# L = D - A 

# ### Definitions to Calculate Laplacian, Eigenvalue+Eigenvectors, Partitions, Spectral Gap

# In[49]:


#define a W matrix that gives the couplings
#these are the definitions in NetProp.py 
import numpy as np
from numpy import linalg as LA
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
import ipdb

class Laplacian():
    def __init__(self,W, weighted=False):
        self.W = W
        self.weighted = weighted
        self.nodes = len(W)
        
    def get_L(self, signed=False, normalized=False):
        A = self.calc_adj()
        D = self.get_D(signed=signed)
        Lin = D[0]-A
        Lout = D[1]-A
        Lboth = D[2]-A
        if normalized:
            N0 = np.diag(np.diag(D[0])**(-.5))
            N1 = np.diag(np.diag(D[1])**(-.5))
            N2 = np.diag(np.diag(D[2])**(-.5))
 
            Lin = np.matmul(N0,np.matmul(Lin,N0))
            Lout = np.matmul(N1,np.matmul(Lout,N1))
            Lboth = np.matmul(N2,np.matmul(Lboth,N2))

        self.Lin = Lin
        self.Lout = Lout
        self.Lboth = Lboth
        return([Lin,Lout,Lboth])
    def calc_adj(self):
        W = self.W
        A = np.where(W>0, 1,0) if not self.weighted else W
        self.A = A
        return A
    def get_D(self, signed=False):
        A = self.A
        if signed: A = np.abs(A)
        W = self.W
        D = []
        Din = np.zeros(len(A))
        Dout=np.zeros(len(A))
        Dboth=np.zeros(len(A))
        for i in range(len(A)):
            Din[i] = sum(A[i,:])
            Dout[i] = sum(A[:,i])
            if A[i,i]>0:
                Dboth[i] = Din[i]+Dout[i]-1
            else:
                Dboth[i] = Din[i]+Dout[i]
        D.append(np.diag(Din))
        D.append(np.diag(Dout))
        D.append(np.diag(Dboth))
        self.Din = Din
        self.Dout = Dout
        self.Dboth = Dboth
        return(D)

    def get_eigen(self, signed=False, normalized=False): # modified from laplacian partition from Netprop.py
            #using just Lboth for now
            Lin,Lout, Lboth = self.get_L(signed=signed, normalized=normalized)
            val, vec = LA.eig(Lout) #Lboth in Netprop.py
            #print(Lin)
            self.val = val
            self.vec = vec
            return val,vec

    def partition(self,part):
        part1 = []
        part2 = []
        for i in range(self.nodes):
            if part[i]>0: 
                part1.append(i)
            else:
                part2.append(i)
        return([part1,part2])

    def laplacian_partition(self): #adapted from Netprop.py
        W = self.W
        val,vec = self.get_eigen()
        #if np.logical_and(np.sort(val)[0]==0,np.sort(val)[1]>0):
        #    part = vec[np.where(val==np.sort(val)[1])]
        #else:
        #    part = vec[np.min(np.where(np.sort(val)>0))]
        #sorted_vec = vecs[:,np.argsort(L.val)]
        sorted_vec = vec[:,np.argsort(L.val)]
        part = vec[:,1]
        part1,part2 = self.partition(part)
        self.part1 = part1
        self.part2 = part2
        return([part1,part2]) #returns list of two lists (one for each group)
        #should it be return([part1,part2])


    def plot_partition(self): #adapted from Netprop.py #need to check this definition
        W = self.W
        parts = self.laplacian_partition()
        side = int(len(W)**0.5)
        Limg = np.zeros(shape = side**2)
        for p in parts[0]:
            Limg[p] = 0
        for p in parts[1]:
            Limg[p] = 255
        Limg = np.reshape(Limg,(side,side))
        Limg = np.repeat(Limg[:,:,np.newaxis],repeats = 3,axis = 2)
        Limg = Limg.astype('uint8')
        plt.imshow(Limg)
        #plt.savefig('{}/LaplacianPartioning_epoch{}_image{}_{}'.format(save_dir,epoch,image_num,trial_type))
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.title('Paritions')
        plt.savefig('/home/matt/laplacian/partition.png')
        plt.close()
    
    def plot_graph(self):
        G = nx.MultiDiGraph()
        W = self.W
        for i in range(len(W)):
            for j in range(len(W)):
            #print(i,j)
                if W[i,j]!= 0: 
                    G.add_weighted_edges_from([(i, j, W[i,j])])

        nx.drawing.nx_pylab.draw_networkx(G)
        #plt.savefig('/home/matt/laplacian/graph.png')
        #plt.close()

    def plot_eigenvalues(self):
        val,vec = self.get_eigen()
        plt.plot(np.sort(val),'o')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalues')
        plt.xticks(range(self.nodes))


def animate_signed(i):
    A = np.array([
    [0, 1, 1, 0, 0,-1. + (2*i / float(frames))  , 0, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [-1. + (2*i /float(frames)), 0, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
    L = Laplacian(A, weighted=True)
    val, vec = L.get_eigen()
    scat.set_offsets(np.array([np.arange(A.shape[0]), np.sort(val)]).T)
    return scat,

def animate_directed(i):
    A[5,0] += (2*i / 100.)
    L = Laplacian(A, weighted=True)
    val, vec = L.get_eigen()
    scat_r.set_offsets(np.array([np.arange(A.shape[0]), np.sort(np.real(val))]).T)
    scat_i.set_offsets(np.array([np.arange(A.shape[0]), np.sort(np.imag(val))]).T)
    plt.ylim([-10, 10])
    return scat_r,scat_i

#fig, ax = plt.subplots()
#A = np.array([
#  [0, 1, 1, 0, 0, -1., 0, 0, 1, 1],
#  [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#  [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
#  [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
#  [-1., 0, 0, 1, 1, 0, 1, 1, 0, 0],
#  [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
#  [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
#  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#  [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

#L = Laplacian(A, weighted=True)
#val, vec = L.get_eigen()
#scat = ax.scatter(np.arange(A.shape[0]), np.sort(val))

#frames=100
#ani = animation.FuncAnimation(fig, animate_signed, frames=frames)
#writer = animation.FFMpegWriter(fps=15)
#ani.save('/home/matt/laplacian/signed_spectrum.gif', fps=30, writer='imagemagick')
#plt.close()

#fig, ax = plt.subplots()

#A = np.array([
#  [0, 1, 1, 0, 0, 1., 0, 0, 1, 1],
#  [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#  [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
#  [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
#  [1., 0, 0, 1, 1, 0, 1, 1, 0, 0],
#  [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
#  [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
#  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#  [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

#L = Laplacian(A, weighted=True)
#val, vec = L.get_eigen()
#scat_r = ax.scatter(np.arange(A.shape[0]), np.sort(np.real(val)),c='b')
#scat_i = ax.scatter(np.arange(A.shape[0]), np.sort(np.imag(val)),c='r')

#frames=100
#ani = animation.FuncAnimation(fig, animate_directed, frames=frames)
#writer = animation.FFMpegWriter(fps=15)
#ani.save('/home/matt/laplacian/directed_spectrum.gif', fps=30, writer='imagemagick')

