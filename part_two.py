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
        for adj_node in G.adj_list[popped_node]:
            if relaxed[node] > 0:
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
    
    return_map = {}
    for i in range(len(G.adj_list)):
        curr_node = i
        path = ""
        while curr_node != source:
            path = str(edge_to[curr_node]) + path
            curr_node = edge_to[curr_node]
        
        return_map[i] = (curr_shortest_distance[i], path)
    
    return return_map
    

