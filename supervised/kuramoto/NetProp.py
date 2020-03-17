import numpy as np 
import matplotlib.pyplot as plt
import itertools
import platform
from numpy import linalg as LA
if platform.python_version()[0] == '3':
    #usenx = True
    usenx = False
    import networkx as nx
else: 
    usenx = False
    
    #compare pixel space to PL

#right now this is only with the the pixels defining everything - not taking into account batch_connectivity

class NetProp():
    
    def __init__(self,coupling_net, connectivity, hierarchical = False):

        self.hierarchical = hierarchical
        if hierarchical:
            self.img_side_squared = coupling_net.shape[1]
            #self.num_cn = connectivity[0].shape[2]+connectivity[1].shape[2] for when one_iamge is off
            self.num_cn = connectivity[0].shape[1]+connectivity[1].shape[1]
            
            self.coupling_net = coupling_net[0,:].unsqueeze(0).cpu().data.numpy()
            self.connectivity_low = connectivity[0].unsqueeze(0).cpu().data.numpy()
            self.connectivity_high = connectivity[1].unsqueeze(0).cpu().data.numpy()  
            #self.connectivity_low = connectivity[0][0,:].unsqueeze(0).cpu().data.numpy()
            #self.connectivity_high = connectivity[1][0,:].unsqueeze(0).cpu().data.numpy() #for when one_iamge is off
            
        else:
            
            self.img_side_squared = coupling_net.shape[1]
            self.num_cn = connectivity.shape[2]
            self.coupling_net = coupling_net[0,:].unsqueeze(0).cpu().data.numpy()#selecting just first couplin in batch
            self.connectivity = connectivity[0,:].unsqueeze(0).cpu().data.numpy()
        if usenx:
            G = self.convert_to_G()
            self.G = G
            print('Using networkx')
        else:
            print('Using self-written code')
            
            actual_coupling = np.zeros([self.img_side_squared,self.img_side_squared])

            if self.hierarchical:
                #need to loop through the connectivity matrix instead of img_side_swared
                #import pdb;pdb.set_trace()
                for i in range(self.connectivity_low.shape[1]):
                    for j in range(self.connectivity_low.shape[2]):
                        actual_coupling[self.connectivity_low[0,i,j],i] = self.coupling_net[0,i,self.connectivity_low[0,i,j]]
                #import pdb;pdb.set_trace()
                for i in range(self.connectivity_high.shape[1]):
                    for j in range(self.connectivity_high.shape[2]):
                        actual_coupling[self.connectivity_high[0,i,j],i+self.connectivity_low.shape[1]-1] = self.coupling_net[0,(i+self.connectivity_low.shape[1]-1),self.connectivity_high[0,i,j]]
                        
                
            else:
                for i in range(self.img_side_squared):
                    for j in range(self.num_cn):
                        actual_coupling[self.connectivity[0,i,j], i] =  self.coupling_net[0,i,self.connectivity[0,i,j]]
                    #again confirm direction of this --> was told that it is projections to pixel i
        
            self.actual_coupling=actual_coupling
    
        

    def convert_to_G(self):
        if self.hierarchical:
            G = nx.MultiDiGraph()
            for i in range(self.connectivity_low.shape[1]):
                for j in range(self.connectivity_low.shape[2]):
                    G.add_weighted_edges_from([(self.connectivity_low[0,i,j], i, self.coupling_net[0,i,self.connectivity_low[0,i,j]])])
            for i in range(self.connectivity_high.shape[1]):
                for j in range(self.connectivity_high.shape[2]):
                    G.add_weighted_edges_from([(self.connectivity_high[0,i,j], i+self.connectivity_low.shape[1], self.coupling_net[0,i+self.connectivity_low.shape[1],self.connectivity_high[0,i,j]])])
            
            return G
        else:
            G = nx.MultiDiGraph()
            for i in range(self.img_side_squared):
                for j in range(self.num_cn):
                    G.add_weighted_edges_from([(self.connectivity[0,i,j], i, self.coupling_net[0,i,self.connectivity[0,i,j]])])
                    #again confirm direction of this --> was told that it is projections to pixel i
            return G
    
    
        
        

    def plot_graph(self):
        
        CM = self.coupling_net
        N = self.img_side_squared#number of nodes 
    
        x = np.linspace(0,np.pi*2)
        y = np.sin(x)
        
        z = np.cos(x)
        nodes = np.linspace(0,np.pi*2,N+1)
        nodes = nodes[:-1]
        points = np.array([np.sin(nodes), np.cos(nodes)])
        plt.plot(y,z)
        plt.plot(points[0],points[1],'o')
        plt.axis('equal')



        for con in itertools.combinations(range(N),2):
            i = con[0]
            j = con[1]
            x_i=points[:,i]
            x_j=points[:,j]
        #plt.plot([x_i[0],x_j[0]],[x_i[1],x_j[1]])
            plt.annotate("", xy=(x_i[0], x_i[1]), xytext=(x_j[0], x_j[1]) ,
                         arrowprops=dict(arrowstyle="->"))
            plt.annotate("", xy=(x_j[0]+0.03, x_j[1]+0.03), xytext=(x_i[0]+0.03, x_i[1]+0.03) ,
                         arrowprops=dict(arrowstyle="->"))

            #plt.plot([x_j[0]+0.03,x_i[0]+0.03],[x_j[1]+0.03,x_i[1]+0.03])

            #corroborate the the direction of arrow matches intended direction 


            #last thing to do: use color to reflect weights
            #i.e. generate color bar

            #outline: 
            min_weight = np.min(CM)  #or hard code these 
            max_weight = np.max(CM)  #or hard code these





        plt.show()



    def path_length(self):
        CN = self.actual_coupling
        if usenx: 
            distance = dict(nx.algorithms.shortest_paths.weighted.all_pairs_bellman_ford_path_length(self.G))
            d = []
            for node1 in range(img_side**2):
                for node2 in range(num_cn):
                    if node1 != node2:
                        print('{} - {}: {}'.format(node1,node2, distance[node1][node2]))
                        d.append(distance[node1][node2])
            avg_shortest_PL = np.mean(d)
            self.AVGpl = avg_shortest_PL
            return avg_shortest_PL
            
        else: 
    
            #unweighted graphs: BFS
            #non-negtive edges (weighted): Dijkstra
            #neg or postive edges: Bellman-Ford 

            #1) check which algorithm is appropriate

            #2) apply algorithm 

            #3) alternatively - always use Bellman Ford

            #find the shortest path length for each 
            
            #this is not friendly to a non-square matrix yet. 
            #need to adapt this - 
            d_all= {}
            p_all={}
            for i in range(len(CN)):
                d,p = bellman_ford(CN,i)
                if d is None:
                    return None 
                d_all[i] = d
                p_all[i] = p

            #find the average
            sum_d = 0
            for d in d_all:
                for i in d_all[d]:
                    if d != i:
                        #print(d_all[d][i])
                        sum_d+=d_all[d][i]
            avg_shortest_PL = sum_d/(self.img_side_squared*(self.img_side_squared-1))



            self.d_all = d_all
            self.p_all = p_all
            self.AVGpl = avg_shortest_PL
            return avg_shortest_PL


    def cluster_coefficient(self):
        #used the formula from this paper: https://arxiv.org/pdf/physics/0612169v3.pdf
        if usenx:
            print('not supported')
        else:
            W = self.coupling_net
            W = W[0,:,:]
            clustering = []
            for i in range(len(W)):
                clustering.append(calc_clustering_i(W,i))
            avg_clustering = np.mean(np.array(clustering))
            self.avg_clustering = avg_clustering
            return avg_clustering
        
    def laplacian_partition(self):
        W = self.coupling_net
        W = W[0,:,:]
        
        Lin,Lout, Lboth = get_L(W)
        
        #using just Lboth for now
        val, vec = LA.eig(Lboth)
        part = vec[np.where(val==np.sort(val)[1])]
        part1,part2 = partition(part[0])
        self.val = val
        self.vec = vec
        self.partition = [part1,part2]
        return partition([part1,part2]) #returns list of two lists (one for each group)
    
    def plot_laplacian(self,exp_name,save_dir, epoch, image_num, trial_type,num_glob):
        parts = self.laplacian_partition()
        if self.hierarchical:
            side = int((len(self.coupling_net)-num_glob)**5+num_glob)
        else:
            side = int(len(self.coupling_net)**0.5)
        Limg = np.zeros(shape = side**2)
        for p in parts[0]:
            Limg[p] = 0
        for p in parts[1]:
            Limg[p] = 255
        Limg = np.reshape(Limg,(side,side))
        Limg = np.repeat(Limg[:,:,np.newaxis],repeats = 3,axis = 2)
        Limg = Limg.astype('uint8')
        plt.imshow(Limg)
        plt.savefig('{}/{}/LaplacianPartioning_epoch{}_image{}_{}.png'.format(exp_name,save_dir,epoch,image_num,trial_type))
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.title('Paritions')
        plt.close()
        plt.plot(np.sort(self.val))
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalues')
        plt.xticks(range(side**2))
        plt.savefig('{}/{}/LaplacianEigenvalues_epoch{}_image{}_{}.png'.format(exp_name,save_dir,epoch,image_num,trial_type))
        plt.close()
        
            
        
    def inh_exc_ratio(self):
        inh = np.sum([i<0 for i in np.nditer(self.coupling_net)])
        exc = np.sum([i>0 for i in np.nditer(self.coupling_net)]) 
        #exc = self.coupling_net.size - inh #use this if no 0 couplings?
       
    #def inh_exc_calculation(self):






#supporting definitions

#def for global order parameter: 

def phase_coherence(phases):
    num_osc = phases.size
    phase_sums = 0 
    for p in np.nditer(phases):
        phase_sums += complex(math.cos(p),math.sin(p))
    PC = phase_sums/num_osc
    return PC

#def for laplacian, sources - to be added 

def partition(part):
    part1 = []
    part2 = []
    for i in range(len(part)):
        if part[i]>0: 
            part1.append(i)
        else:
            part2.append(i)
    return([part1,part2])

def get_D(W,A):
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
    return(D)
#def calc_adj(W):  #this definition already defined in cluster
#    A = np.where(W>0, 1,0)
#    return A
def get_L(W):
    A = calc_adj(W)
    D = get_D(W,A)
    Lin = D[0]-A
    Lout = D[1]-A
    Lboth = D[2]-A
    return([Lin,Lout,Lboth])

"""Clustering definition 
written based on the paper: Directed_clustering_in_weighted_networks_A_new_perspective
"""
#our i is reverse i from paper - Ai1 gives stuff out in the paper but for us it gives 
#number of connects in
def calc_adj(W):
    A = np.where(W>0, 1,0)
    return A
def calc_si(W,i):
    #from equation 7 (WT+W) indexed at i *1 
    B = W+W.transpose()
    si = np.dot(B[i],np.ones(len(W)))
    return si

def calc_di(A,i):
    #from equation 3 - di_tot = di_in +di_out = (AT + A)i1 for us
    
    B = A+A.transpose()
    di = np.dot(B[i],np.ones(len(A)))
    
    return di

def calc_sbi(A,W,i):
    #using equation 8 strength between bilateral arcs between adjuacent ndoes
    # sum(all j not equal to i) (aij*aji)(wij+wji)/2)
    sbi = 0 
    for j in range(len(A)):
        if j != i: 
            sbi += (A[i,j]*A[j,i])*(W[i,j]+W[j,i])/2
    return sbi

def calc_clustering_i(W,i):
    
    #given by equation 14
    #calculate for one node i
    
    A = calc_adj(W)
    si = calc_si(W,i)
    di = calc_di(A,i)
    sbi = calc_sbi(A,W,i)
    
    Ci = 0.5*(((W+W.transpose())*((A+A.transpose())**2))[i,i])/(si*(di-1)-2*sbi)
    return Ci


"""
The Bellman-Ford algorithm
re-written to use the coupling matrix 

"""




def initialize(CM,source):
    d = {}  #destination (ends up)
    p = {} #stands for predecessor 

    for node in range(len(CM)):
        d[node] = float('Inf')
        p[node] = None
    d[source] = 0 #this is where we are starting

    return d,p

def relax(node, neighbor, CM, d, p):

# If the distance between the node and the neighbour is lower than the one I have now
    if d[neighbor] > d[node] + CM[node,neighbor]:
        # Record this lower distance
        d[neighbor]  = d[node] + CM[node,neighbor]
        p[neighbor] = node


def bellman_ford(CM,source):
    d, p = initialize(CM, source)

    for i in range(len(CM)-1): #Run this until is converges
        for u in range(len(CM)): #all of the nodes 
            for v in range(len(CM)): #all of the nodes, 
                if u != v:  #now all of the neigbors only 
                    relax(u, v, CM, d, p) #Lets relax it 

    # Step 3: check for negative-weight cycles
    for u in range(len(CM)):
        for v in range(len(CM)):
            if u != v:
                if d[v] > d[u] + CM[u,v]:
                    print('Negative cycle')
                    return None,None
                #assert d[v] <= d[u] + CM[u,v]
                    
    return d,p

