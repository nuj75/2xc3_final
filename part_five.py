# read information from both files
# make a weighted graph where each node corresponds to a station in the london stations file
# make a hashmap that links every station id to its location
# when reading the connections file, find the weight by extracting the locations of both
# stations and calculating the distance between the two
# calculate the heuristic function in the a* algorithm. take the location of the source and each other node
# and calculate the distance


class WeightedGraph:
    def __init__(self, size):
        self.adj_list = [[] for i in range(size)]
        self.locations = {}
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
    

def calculate_distance(graph, node_one, node_two):
    return ((graph.locations[node_one]["long"] - graph.locations[node_two]["long"]) ** 2 + (graph.locations[node_one]["lat"] - graph.locations[node_two]["lat"]) ** 2) ** 0.5

def file_parser():
    with open("london_stations.csv", "r") as f:
        file = f.readlines()

        return_graph = WeightedGraph(len(file) + 1)


        for i in range(1, len(file)):
            line_split = file[i].split(",")

            return_graph.locations[int(line_split[0])] = {"lat": float(line_split[1]), "long": float(line_split[2])}


    with open("london_connections.csv", "r") as f:
        file = f.readlines()

        for i in range(1, len(file)):
            line_split = file[i].split(",")

            start_node = int(line_split[0])
            end_node = int(line_split[1])
            
            return_graph.edge(start_node, end_node, calculate_distance(return_graph, start_node, end_node))

    return return_graph

graph = file_parser()

