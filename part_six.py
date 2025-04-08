class ShortPathFinder:
    def __init__(self, graph, algorithm = None):
        self.graph = graph
        self.algorithm = algorithm

    def set_graph(self, graph):
        self.graph = graph

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def calc_short_path(self, source, dest = None):
        if self.graph and self.algorithm:
            return self.algorithm.calc_sp(self.graph, source, dest)
        return None
    

class SPAlgorithm:
    def calc_sp(self, graph, source, dest = None):
        print("Subclasses must implement method")

class dijkstras(SPAlgorithm):
    def calc_sp(self, G, source, dest = None, k=float("inf")):
        relaxed = {}
        # print(k)
        for node in range(len(G.adj_list)):
            relaxed[node] = k
        # print(relaxed)
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
                        heap.add(Node(adj_node, -1 * relax_distance))
                    
                    if curr_distance > relax_distance and adj_node in heap.map.keys():
                        edge_to[adj_node] = popped_node
                        curr_shortest_distance[adj_node] = relax_distance
                        heap.change_priority(adj_node, -1 * relax_distance)
                    
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

class bellmanford(SPAlgorithm):
    def calc_sp(self, G, source, dest = None, k=float("inf")):
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

class astar(SPAlgorithm):
    def calc_sp(self, G, source, destination):
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
                    heap.add(Node(adj_node, -1 * (relax_distance + G.get_heuristic(adj_node))))
                
                if curr_distance > relax_distance and adj_node in heap.map.keys():
                    edge_to[adj_node] = popped_node
                    curr_shortest_distance[adj_node] = relax_distance
                    heap.change_priority(adj_node, -1 * (relax_distance + G.get_heuristic(adj_node)))

        curr_node = destination
        path = ""
        while edge_to[curr_node] != curr_node:
            path = str(edge_to[curr_node]) + path
            curr_node = edge_to[curr_node]
            
        return path

#                   TESTING                         #

def create_sample_graph():
    graph = WeightedGraph(6)
    
    graph.add_edge(0, 1, 4)
    graph.add_edge(0, 2, 2)
    graph.add_edge(1, 3, 3)
    graph.add_edge(2, 3, 1)
    graph.add_edge(2, 4, 5)
    graph.add_edge(3, 4, 2)
    graph.add_edge(3, 5, 6)
    graph.add_edge(4, 5, 4)
    
    return graph

def create_heuristic_graph():
    graph = HeuristicGraph(5)
    
    # same as anujan test in p4
    graph.add_edge(0, 1, 12)
    graph.add_edge(0, 2, 10)
    graph.add_edge(3, 1, 0)
    graph.add_edge(2, 3, 0)
    graph.add_edge(0, 3, 100)
    graph.add_edge(3, 4, 10)
    
    heuristic = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for node, h_value in heuristic.items():
        graph.set_heuristic(node, h_value)
    
    return graph

def main():
    graph = create_sample_graph()

    print("Dijkstra's Algorithm Test:")
    path_finder = ShortPathFinder(graph, dijkstras())
    shortest_paths = path_finder.calc_short_path(source=0)
    print("Shortest paths from node 0:")
    for node, (distance, path) in shortest_paths.items():
        print(f"To node {node}: Distance = {distance}, Path = {path}")

    print("\nBellman-Ford Algorithm Test:")
    path_finder.set_algorithm(bellmanford())
    shortest_paths = path_finder.calc_short_path(source=0)
    print("Shortest paths from node 0:")
    for node, (distance, path) in shortest_paths.items():
        print(f"To node {node}: Distance = {distance}, Path = {path}")

    hGrapgh = create_heuristic_graph()

    print("\nAStar Algorithm Test:")
    path_finder.set_graph(hGrapgh)
    path_finder.set_algorithm(astar())
    destination_path = path_finder.calc_short_path(source=0, dest=3)
    print(f"To node 3: {destination_path}")

#                   GRAPH                           #

class Graph:
    def __init__(self, size=0):
        self.adj_list = [[] for _ in range(size)]

    def get_adj_nodes(self, node):
        return self.adj_list[node]

    def add_node(self):
        self.adj_list.append([])

    def add_edge(self, from_node, to_node):
        if not self.connected(from_node, to_node) and not self.connected(to_node, from_node):
            self.adj_list[from_node].append(to_node)
            self.adj_list[to_node].append(from_node)

    def get_num_of_nodes(self):
        return len(self.adj_list)
    
    def connected(self, from_node, to_node):
        return (from_node in self.adj_list[to_node]) or (to_node in self.adj_list[from_node])

class WeightedGraph(Graph):
    def __init__(self, size = 0):
        super().__init__(size)
        self.map = {}

    def max_node(self):
        return len(self.adj_list) - 1
    
    def add_edge(self, from_node, to_node, weight):
        if not self.connected(from_node, to_node) and not self.connected(to_node, from_node):
            self.adj_list[from_node].append(to_node)
            self.adj_list[to_node].append(from_node)
            
            self.map[(from_node, to_node)] = weight
            self.map[(to_node, from_node)] = weight


    def w(self, from_node, to_node):
        if self.connected(from_node, to_node):
            return self.map[(from_node, to_node)]

        return -1

class HeuristicGraph(WeightedGraph):
    def __init__(self, size = 0):
        super().__init__(size)
        self.heuristic = {}
    
    def set_heuristic(self, node, hValue):
        self.heuristic[node] = hValue
    
    def get_heuristic(self, node):
        return self.heuristic.get(node, float('inf'))

#                   NODE and HEAP                   #

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


main()