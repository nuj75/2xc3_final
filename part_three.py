class DirectedWeightedGraph:
    def __init__(self):
        self.adj = {}
        self.weights = {}

    def are_connected(self, node1, node2):
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return True
        return False

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def number_of_nodes(self):
        return len(self.adj)
    
    def get_node(self):
        return self.adj
    
    
def all_pairs_shorest_path(G):
    distances = {}
    predecessors = {}
    for vertex in G.get_node():
        source = vertex
        distances[source] = {}
        predecessors[source] = {}

        # initializing
        for v in G.get_node():
            distances[source][v] = float('inf')
            predecessors[source][v] = None
        distances[source][source] = 0

        # relax all edges for V-1 times
        for i in range(len(G.get_node())):
            for v1 in G.get_node():
                for v2 in G.adj[v1]:
                    if distances[source][v1] + G.w(v1,v2) < distances[source][v2]:
                        distances[source][v2] = distances[source][v1] + G.w(v1,v2)
                        predecessors[source][v2] = v1
        
    return distances,predecessors
