import numpy as np 

class NetProp():
    
    def __init__(self,coupling_net):
        self.coupling_net = coupling_net
        

    def create_graph(self):
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



    def path_length(self, threshold):
        CN = self.coupling_net
    
        #unweighted graphs: BFS
        #non-negtive edges (weighted): Dijkstra
        #neg or postive edges: Bellman-Ford 
        
        #1) check which algorithm is appropriate
        
        #2) apply algorithm 
        
        #3) alternatively - always use Bellman Ford
        

    def cluster_coefficient(self,threshold):
        
    def inh_exc_ratio(self):
        inh = np.sum([i<0 for i in np.nditer(self.coupling_net)])
        exc = np.sum([i>0 for i in np.nditer(self.coupling_net)]) 
        #exc = self.coupling_net.size - inh #use this if no 0 couplings?
       
    def inh_exc_calculation 








#supporting definitions

import pdb
"""
The Bellman-Ford algorithm
Graph API:
    iter(graph) gives all nodes
    iter(graph[u]) gives neighbours of u
    graph[u][v] gives weight of edge (u, v)
"""

# Step 1: For each node prepare the destination and predecessor
def initialize(graph, source):
    d = {} # Stands for destination
    p = {} # Stands for predecessor
    for node in graph:
        d[node] = float('Inf') # We start admiting that the rest of nodes are very very far
        p[node] = None
    d[source] = 0 # For the source we know how to reach
    return d, p

def relax(node, neighbour, graph, d, p):
    # If the distance between the node and the neighbour is lower than the one I have now
    if d[neighbour] > d[node] + graph[node][neighbour]:
        # Record this lower distance
        d[neighbour]  = d[node] + graph[node][neighbour]
        p[neighbour] = node

def bellman_ford(graph, source):
    d, p = initialize(graph, source)
    for i in range(len(graph)-1): #Run this until is converges
        for u in graph:
            for v in graph[u]: #For each neighbour of u
                relax(u, v, graph, d, p) #Lets relax it

    # Step 3: check for negative-weight cycles
    for u in graph:
        for v in graph[u]:
            assert d[v] <= d[u] + graph[u][v]

    return d, p