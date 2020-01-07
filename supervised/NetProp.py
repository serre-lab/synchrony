import numpy as np 
import matplotlib.pyplot as plt
import itertools

class NetProp():
    
    def __init__(self,coupling_net):
        self.coupling_net = coupling_net
    
    #def make_graph(self):
        #not going to use anymore?
        

    def plot_graph(self):
        CM = self.coupling_net
        N = len(CM)#number of nodes 
    
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
        CN = self.coupling_net
    
        #unweighted graphs: BFS
        #non-negtive edges (weighted): Dijkstra
        #neg or postive edges: Bellman-Ford 
        
        #1) check which algorithm is appropriate
        
        #2) apply algorithm 
        
        #3) alternatively - always use Bellman Ford

        #find the shortest path length for each 
        d_all= {}
        p_all={}
        for i in range(len(CN)):
            d,p = bellman_ford(CN,i)
            d_all[i] = d
            p_all[i] = p
            
        #find the average
        sum_d = 0
        for d in d_all:
            for i in d_all[d]:
                if d != i:
                    #print(d_all[d][i])
                    sum_d+=d_all[d][i]
        avg_shortest_PL = sum_d/(len(CM)*(len(CM)-1))
        
        
        
        self.d_all = d_all
        self.p_all = p_all
        self.AVGpl = avg_shortest_PL


    #def cluster_coefficient(self,threshold):
        
    def inh_exc_ratio(self):
        inh = np.sum([i<0 for i in np.nditer(self.coupling_net)])
        exc = np.sum([i>0 for i in np.nditer(self.coupling_net)]) 
        #exc = self.coupling_net.size - inh #use this if no 0 couplings?
       
    #def inh_exc_calculation(self):






#supporting definitions

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
                assert d[v] <= d[u] + CM[u,v]
    return d,p

