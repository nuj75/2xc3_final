import math
import random 
import matplotlib.pyplot as plt
import timeit
class Node:
    def __init__(self, data, priority):
        self.data = data
        self.priority = priority
    
    def __lt__(self, other):
        if isinstance(other, Node):
            return self.priority < other.priority  
        return NotImplemented  

    def __le__(self, other):
        if isinstance(other, Node):
            return self.priority <= other.priority
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.priority == other.priority and self.priority == other.priority 
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Node):
            return not self.__eq__(other) 
        return NotImplemented

    def __gt__(self, other):
       if isinstance(other, Node):
           return self.priority > other.priority
       return NotImplemented

    def __ge__(self, other):
       if isinstance(other, Node):
           return self.priority >= other.priority
       return NotImplemented
    
    def __str__(self):
        return str(self.data)

class Heap:
    def __init__(self, list=None):
        self.items = [Node(-1, float("inf"))]
        self.map = {}
        if list:
            for i in range(len(list)):
                self.map[list[i].data] = i + 1
            self.items += list
            self.heapify()

    def sink(self, index):
        curr_index = index
        while True:
            max_index = curr_index

            if 2 * curr_index < len(self.items) and self.items[2 * curr_index] > self.items[max_index]:
                max_index = 2 * curr_index
            if 2 * curr_index + 1 < len(self.items) and self.items[2 * curr_index + 1] > self.items[max_index]:
                max_index = 2 * curr_index + 1

            if max_index == curr_index:
                return

            self.items[curr_index], self.items[max_index] = self.items[max_index], self.items[curr_index]
            self.map[self.items[curr_index].data], self.map[self.items[max_index].data] = self.map[self.items[max_index].data], self.map[self.items[curr_index].data]

            curr_index = max_index

            
    def swim(self, index):
        curr_index = index
        while curr_index // 2 > 0 and self.items[curr_index // 2] < self.items[curr_index]:
            self.items[curr_index], self.items[curr_index//2] = self.items[curr_index//2], self.items[curr_index]
            self.map[self.items[curr_index].data], self.map[self.items[curr_index // 2].data] = self.map[self.items[curr_index // 2].data], self.map[self.items[curr_index].data]
            curr_index = curr_index // 2
    
    def heapify(self):
        for i in range(len(self.items) - 1, 0, -1):
            self.sink(i)
 
    def add(self, node):
        self.items.append(node)
        self.map[node.data] = len(self.items) - 1
        self.swim(len(self.items) - 1)

    def extract_max(self):
        if len(self.items) == 1:
            return -1
        
        self.items[1], self.items[len(self.items) - 1] = self.items[len(self.items) - 1], self.items[1]
        self.map[self.items[1].data], self.map[self.items[len(self.items) - 1].data] = self.map[self.items[len(self.items) - 1].data], self.map[self.items[1].data]

        popped_max = self.items[-1]
        self.items = self.items[:-1]
        self.map.pop(popped_max.data, None)


        self.sink(1)

        return popped_max
    
    def change_priority(self, data, new_priority):
        self.items[self.map[data]].priority = new_priority


        self.swim(self.map[data])

        self.sink(self.map[data])

    
    def __str__(self):
        print_str = '['

        for i in range (1, len(self.items)):
            print_str += str(self.items[i]) + ", "
        
        return print_str[:-2] + ']'

class WeightedGraph:
    def __init__(self, size):
        self.adj_list = [[] for i in range(size)]
        self.map = {}
    
    def add_node(self):
        self.adj_list.append([])

    def max_node(self):
        return len(self.adj_list) - 1
    
    def edge(self, from_node, to_node, weight):
        if not self.connected(from_node, to_node) and not self.connected(to_node, from_node):
            self.adj_list[from_node].append(to_node)
            self.adj_list[to_node].append(from_node)
            
            self.map[(from_node, to_node)] = weight
            self.map[(to_node, from_node)] = weight


    def w(self, from_node, to_node):
        if self.connected(from_node, to_node):
            return self.map[(from_node, to_node)]

        return -1
        
    def connected(self, from_node, to_node):
        return (from_node in self.adj_list[to_node]) or (to_node in self.adj_list[from_node])
    


def dijkstras(G, source, k):
    relaxed = {}

    for node in range(len(G.adj_list)):
        relaxed[node] = k
    
    heap = Heap()
    heap.add(Node(source, 0))

    curr_shortest_distance = {}
    for node in range(len(G.adj_list)):
        if node == source:
            curr_shortest_distance[node] = 0
        else:
            curr_shortest_distance[node] = float("inf")

    edge_to = [i for i in range(len(G.adj_list))]

    while len(heap.items) > 1:
        popped_node = heap.extract_max()
        popped_node = popped_node.data

        for adj_node in G.adj_list[popped_node]:
            if relaxed[adj_node] > 0:
                relax_distance = curr_shortest_distance[popped_node] + G.w(popped_node, adj_node)
                curr_distance = curr_shortest_distance[adj_node]

                if curr_distance > relax_distance and adj_node not in heap.map.keys():
                    edge_to[adj_node] = popped_node
                    curr_shortest_distance[adj_node] = relax_distance
                    heap.add(Node(adj_node, relax_distance))
                
                if curr_distance > relax_distance and adj_node in heap.map.keys():
                    edge_to[adj_node] = popped_node
                    curr_shortest_distance[adj_node] = relax_distance
                    heap.change_priority(adj_node, relax_distance)
                
                relaxed[adj_node] += -1
    
    return_map = {}
    for i in range(len(G.adj_list)):
        curr_node = i
        path = ""
        while edge_to[curr_node] != curr_node:
            path = str(edge_to[curr_node]) + path
            curr_node = edge_to[curr_node]
        
        return_map[i] = (curr_shortest_distance[i], path)
    
    return return_map

def bellmanford(G, source, k):
    relaxed = {}

    for node in range(len(G.adj_list)):
        relaxed[node] = k
    
    edge_to = [i for i in range(len(G.adj_list))]
    curr_shortest_distance = {}
    for node in range(len(G.adj_list)):
        if node == source:
            curr_shortest_distance[node] = 0
        else:
            curr_shortest_distance[node] = float("inf")

    queue = [source]

    for i in range(len(G.adj_list) * len(G.adj_list)):
        if len(queue) == 0:
            break

        curr_node = queue.pop(0)

        for adj_node in G.adj_list[curr_node]:
            if curr_shortest_distance[adj_node] > curr_shortest_distance[curr_node] + G.w(curr_node, adj_node) and relaxed[adj_node] > 0:
                queue.append(adj_node)

                edge_to[adj_node] = curr_node
                curr_shortest_distance[adj_node] = curr_shortest_distance[curr_node] + G.w(curr_node, adj_node)

                relaxed[adj_node] += -1
    
    for i in range(len(G.adj_list) * len(G.adj_list)):
        if len(queue) == 0:
            break

        curr_node = queue.pop(0)

        for adj_node in G.adj_list[curr_node]:
            if curr_shortest_distance[adj_node] > curr_shortest_distance[curr_node] + G.w(curr_node, adj_node):
                return {}
    
    return_map = {}
    for i in range(len(G.adj_list)):
        curr_node = i
        path = ""
        while edge_to[curr_node] != curr_node:
            path = str(edge_to[curr_node]) + path
            curr_node = edge_to[curr_node]
        
        return_map[i] = (curr_shortest_distance[i], path)
    
    return return_map




graph = WeightedGraph(10)
graph.edge(0, 1, 1)
graph.edge(0, 2, 10)
graph.edge(2, 1, 0)
graph.edge(2, 3, 0)
graph.edge(0, 3, 100)
graph.edge(3,4, 10)




def create_random_graph(n, e):
    G = WeightedGraph(n)
    edges_left = min(e, math.comb(n, 2))

    #break out of cycle if all possible edges are taken
    while edges_left > 0:
        src, dst = random.randint(0, n - 1), random.randint(0, n - 1)

        if src == dst:
            continue
        # only add the generated pair into the graph if it isn't already in the graph
        if dst not in G.adj_list[src] and src not in G.adj_list[dst]: 
            G.edge(src, dst, random.randint(0, 100))
            edges_left -= 1
    
    return G


def dijkstras_experiment():
    #2d array. first value represent number of edges, next represents k
    results = [[0 for j in range(10)] for i in range(10)] 

    for i in range(0, len(results)):
        for j in range(0, len(results[0])):
            #todo: add logic 
            #for the given edge and k values, time how long it takes to perform dijkstras
            averaging_array = []
            for k in range(100):
                G = create_random_graph(100, (i + 1) * 20)

                start = timeit.default_timer()
                dijkstras(G, 0, (k + 1) * 20)
                stop = timeit.default_timer()
                averaging_array.append(stop-start)

            results[i][j] = sum(averaging_array)/len(averaging_array)

    
    # start of x for each bar
    x = []
    for i in range(10):
        for j in range(10):
            x += [(i + 1) * 20]

    # start of y for each bar
    y = []
    for i in range(10):
        for j in range(10):
            y += [(j + 1) * 20]

    # start of z for each bar
    z = []
    for i in range(10):
        for j in range(10):
            z += [0]

    # size of x for each bar
    dx = []
    for i in range(10):
        for j in range(10):
            dx += [1]

    # size of y for each bar
    dy = []
    for i in range(10):
        for j in range(10):
            dy += [1]

    # size of z for each bar
    dz = []
    for i in range(10):
        for j in range(10):
            dz += [(results[i][j])]

    plt.clf()
    ax = plt.axes(projection="3d")
    ax.bar3d(x, y, z, dx, dy, dz, color = "#E02050")

    ax.set_xlabel('# edges')
    ax.set_ylabel('k value')
    ax.set_zlabel('Time Taken in s')
    plt.title("Time taken for Dijkstra's by Density and K value")
    plt.savefig("dijkstras.png")


def bellmanford_experiment():
    #2d array. first value represent number of edges, next represents k
    results = [[0 for j in range(10)] for i in range(10)] 

    for i in range(0, len(results)):
        for j in range(0, len(results[0])):
            #todo: add logic 
            #for the given edge and k values, time how long it takes to perform dijkstras
            averaging_array = []
            for k in range(100):
                G = create_random_graph(100, (i + 1) * 20)

                start = timeit.default_timer()
                bellmanford(G, 0, (k + 1) * 20)
                stop = timeit.default_timer()
                averaging_array.append(stop-start)

            results[i][j] = sum(averaging_array)/len(averaging_array)

    
    # start of x for each bar
    x = []
    for i in range(10):
        for j in range(10):
            x += [(i + 1) * 20]

    # start of y for each bar
    y = []
    for i in range(10):
        for j in range(10):
            y += [(j + 1) * 20]

    # start of z for each bar
    z = []
    for i in range(10):
        for j in range(10):
            z += [0]

    # size of x for each bar
    dx = []
    for i in range(10):
        for j in range(10):
            dx += [1]

    # size of y for each bar
    dy = []
    for i in range(10):
        for j in range(10):
            dy += [1]

    # size of z for each bar
    dz = []
    for i in range(10):
        for j in range(10):
            dz += [(results[i][j])]

    plt.clf()
    ax = plt.axes(projection="3d")
    ax.bar3d(x, y, z, dx, dy, dz, color = "#E02050")

    ax.set_xlabel('# edges')
    ax.set_ylabel('k value')
    ax.set_zlabel('Time Taken in s')
    plt.title("Time taken for BellmanFord's by Density and K value") 
    plt.savefig("BellmanFord.png")

dijkstras_experiment()
bellmanford_experiment()

