# read information from both files
# make a weighted graph where each node corresponds to a station in the london stations file
# make a hashmap that links every station id to its location
# when reading the connections file, find the weight by extracting the locations of both
# stations and calculating the distance between the two
# calculate the heuristic function in the a* algorithm. take the location of the source and each other node
# and calculate the distance
import numpy as np
import matplotlib.pyplot as plt
import timeit

class WeightedGraph:
    def __init__(self, size):
        self.adj_list = [[] for i in range(size)]
        self.locations = {}
        self.map = {}
        self.line = {}
    
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
    

def calculate_distance(graph, node_one, node_two):
    return ((graph.locations[node_one]["long"] - graph.locations[node_two]["long"]) ** 2 + (graph.locations[node_one]["lat"] - graph.locations[node_two]["lat"]) ** 2) ** 0.5

def file_parser():
    with open("london_stations.csv", "r") as f:
        file = f.readlines()

        return_graph = WeightedGraph(len(file) + 1)


        for i in range(1, len(file)):
            line_split = file[i].split(",")

            return_graph.locations[int(line_split[0])] = {"lat": float(line_split[1]), "long": float(line_split[2])}
            return_graph.line[int(line_split[0])] = line_split


    with open("london_connections.csv", "r") as f:
        file = f.readlines()

        for i in range(1, len(file)):
            line_split = file[i].split(",")

            start_node = int(line_split[0])
            end_node = int(line_split[1])
            
            return_graph.edge(start_node, end_node, calculate_distance(return_graph, start_node, end_node))
            return_graph.line[(start_node, end_node)] = int(line_split[2])
            return_graph.line[(end_node, start_node)] = int(line_split[2])


    return return_graph




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



def astar(G, source, destination, heuristic = None):
    if heuristic == None:
        heuristic = {}

        for i in G.locations.keys():
            heuristic[i] = calculate_distance(G, destination,i)

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


        if popped_node == destination:
            break

        for adj_node in G.adj_list[popped_node]:
            relax_distance = curr_shortest_distance[popped_node] + G.w(popped_node, adj_node)
            curr_distance = curr_shortest_distance[adj_node]

            if curr_distance > relax_distance and adj_node not in heap.map.keys():
                edge_to[adj_node] = popped_node
                curr_shortest_distance[adj_node] = relax_distance
                heap.add(Node(adj_node, -1 * (relax_distance + heuristic[adj_node])))

            
            if curr_distance > relax_distance and adj_node in heap.map.keys():
                edge_to[adj_node] = popped_node
                curr_shortest_distance[adj_node] = relax_distance
                heap.change_priority(adj_node, -1 * (relax_distance + heuristic[adj_node]))

                
    
    
    curr_node = destination
    path = str(destination)
    linecount = 0
    curr_line = -1
    while edge_to[curr_node] != curr_node:
        path = str(edge_to[curr_node]) + "," + path


        linecount += (1 if curr_line != G.line[(curr_node, edge_to[curr_node])] else 0)
        curr_line = G.line[(curr_node, edge_to[curr_node])]

        curr_node = edge_to[curr_node]
        
    return path, linecount



def dijkstras(G, source, destination):

    
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

        if popped_node == destination:
            break


        for adj_node in G.adj_list[popped_node]:
            relax_distance = curr_shortest_distance[popped_node] + G.w(popped_node, adj_node)
            curr_distance = curr_shortest_distance[adj_node]

            if curr_distance > relax_distance and adj_node not in heap.map.keys():
                edge_to[adj_node] = popped_node
                curr_shortest_distance[adj_node] = relax_distance

                heap.add(Node(adj_node, -1 * relax_distance))
            
            if curr_distance > relax_distance and adj_node in heap.map.keys():
                edge_to[adj_node] = popped_node
                curr_shortest_distance[adj_node] = relax_distance

                heap.change_priority(adj_node, -1 * relax_distance)
                
    
    curr_node = destination
    path = str(destination)
    linecount = 0
    curr_line = -1
    while edge_to[curr_node] != curr_node:
        path = str(edge_to[curr_node]) + "," + path


        linecount += (1 if curr_line != G.line[(curr_node, edge_to[curr_node])] else 0)
        curr_line = G.line[(curr_node, edge_to[curr_node])]

        curr_node = edge_to[curr_node]
        
    return path, linecount

# how to test results

def draw_plot(run_arr, mean, sort_name, file_name):
    x = np.arange(0, len(run_arr),1)
    fig=plt.figure(figsize=(20,8))

    plt.axhline(mean,color="red",linestyle="--",label="Avg")
    plt.xlabel("Experiment")
    plt.xticks([0, 1,2], ["same line", "adjacent line", "multiline"])
    plt.ylabel("Run time in s")
    plt.title("Run time for " + sort_name)

    
    plt.bar(x,run_arr)
    plt.savefig(file_name + ".png")

def algorithm_testers():
    G = file_parser()

    dijkstras_same_line = []
    dijkstras_adjacent_line = []
    dijkstras_multi_line = []
    for i in range(len(G.adj_list)):
        if i == 0 or i == 189:
            continue
        for j in range(len(G.adj_list)):
            if j == 0 or j == 189:
                continue

            start_time = timeit.default_timer()
            path, lines = dijkstras(G, i, j)
            stop_time = timeit.default_timer()

            if lines == 0:
                dijkstras_same_line.append(stop_time - start_time)
            elif lines == 1:
                dijkstras_adjacent_line.append(stop_time - start_time)
            else:
                dijkstras_multi_line.append(stop_time - start_time)
    
    a = sum(dijkstras_same_line)/(len(dijkstras_same_line) if len(dijkstras_same_line) != 0 else 1 )
    b = sum(dijkstras_adjacent_line)/(len(dijkstras_adjacent_line) if len(dijkstras_adjacent_line) != 0 else 1 )
    c = sum(dijkstras_multi_line)/(len(dijkstras_multi_line) if len(dijkstras_multi_line) != 0 else 1 )

    avgs_arr = [a,b,c]

    draw_plot(avgs_arr, sum(avgs_arr)/len(avgs_arr), "Dijkstra's", "dijkstra")
        
    
    astar_same_line = []
    astar_adjacent_line = []
    astar_multi_line = []
    
    for i in range(len(G.adj_list)):
        if i == 0 or i == 189:
            continue

        for j in range(len(G.adj_list)):
            if j == 0 or j == 189:
                continue
            heuristic = {}

            for k in G.locations.keys():
                heuristic[k] = calculate_distance(G, j,k)


            start_time = timeit.default_timer()
            path, lines = astar(G, i, j, heuristic)
            stop_time = timeit.default_timer()

            if lines == 0:
                astar_same_line.append(stop_time - start_time)
            elif lines == 1:
                astar_adjacent_line.append(stop_time - start_time)

            else:
                astar_multi_line.append(stop_time - start_time)

    a = sum(astar_same_line)/(len(astar_same_line) if len(astar_same_line) != 0 else 1 )
    b = sum(astar_adjacent_line)/(len(astar_adjacent_line) if len(astar_adjacent_line) != 0 else 1 )
    c = sum(astar_multi_line)/(len(astar_multi_line) if len(astar_multi_line) != 0 else 1 )

    avgs_arr = [a,b,c]

    draw_plot(avgs_arr, sum(avgs_arr)/len(avgs_arr), "A star", "astarnoh")


    astar_same_line = []
    astar_adjacent_line = []
    astar_multi_line = []
    
    for i in range(len(G.adj_list)):
        if i == 0 or i == 189:
            continue

        for j in range(len(G.adj_list)):
            if j == 0 or j == 189:
                continue


            start_time = timeit.default_timer()
            path, lines = astar(G, i, j)
            stop_time = timeit.default_timer()

            if lines == 0:
                astar_same_line.append(stop_time - start_time)
            elif lines == 1:
                astar_adjacent_line.append(stop_time - start_time)
            else:
                astar_multi_line.append(stop_time - start_time)

    a = sum(astar_same_line)/(len(astar_same_line) if len(astar_same_line) != 0 else 1 )
    b = sum(astar_adjacent_line)/(len(astar_adjacent_line) if len(astar_adjacent_line) != 0 else 1 )
    c = sum(astar_multi_line)/(len(astar_multi_line) if len(astar_multi_line) != 0 else 1 )

    avgs_arr = [a,b,c]  
    draw_plot(avgs_arr, sum(avgs_arr)/len(avgs_arr), "A star", "astar")

algorithm_testers()
    

