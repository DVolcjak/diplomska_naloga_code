import generate_data
import numpy as np
import sys
import bisect
import networkx as nx
import matplotlib.pyplot as plt

class Node:
    def __init__(self, name, num, cost):
        self.name = name
        self.num = num
        self.cost = cost

class Arc:
    capacity = 1
    flow = 0
    def __init__(self, from_, to_, cost):
        self.from_ = from_
        self.to_ = to_
        self.cost = cost

    def is_arc_from_to(self, from_node, to_node):
        if self.from_ == from_node and self.to_ == to_node:
            return True
        else:
            return False


def bellman_ford(nodes_1, edges_1):

    """ nodes_1 = nodes.copy()
    edges_1 = edges.copy() """

    for key in nodes_1:
        if key != "source":
            nodes_1[key].cost = sys.maxsize

    path_dict = {key: None for key in nodes_1}

    for _ in range(len(nodes_1)):
        changes = False
        
        #for node in next_nodes:
        for edge in edges_1:

            from_cost = nodes_1[edge.from_].cost
            cost = edge.cost
            to_cost = nodes_1[edge.to_].cost
            
            if (from_cost + cost) < to_cost:
                nodes_1[edge.to_].cost = from_cost + cost
                path_dict[edge.to_] = edge.from_
                changes = True

        if not changes:
            break

    changes = False
    for edge in edges_1:

        from_cost = nodes_1[edge.from_].cost
        cost = edge.cost
        to_cost = nodes_1[edge.to_].cost

        if (from_cost + cost) < to_cost:
            nodes_1[edge.to_].cost = from_cost + cost
            path_dict[edge.to_] = edge.from_
            changes = True

    if changes:
        print("Graph contains negative cycles")


    return nodes_1, path_dict


def find_shortest_path(path_dict):
    previous_node = "sink"
    actual_path = [previous_node]
    for _ in range(len(path_dict)):
        current_node = path_dict[previous_node]
        if current_node == None:
            break
        actual_path.append(current_node)
        previous_node = current_node

    actual_path.reverse()

    return actual_path


def residual_graph_creation(arcs):

    residual_graph = []
    for arc in arcs:
        if arc.flow == 1:
            residual_graph.append(Arc(arc.to_, arc.from_, -arc.cost))
        else:
            residual_graph.append(arc)

    return residual_graph


def cost_of_path(path, L, edges):
    cost_of_path = 0
    for i in range(len(path) - 1):
        path_from = path[i]
        path_to = path[i+1]
        for arc in edges:
            if arc.is_arc_from_to(path_from, path_to):
                if arc.cost != L and arc.cost != -L:
                    cost_of_path += (arc.cost)
                break
    
    return cost_of_path


def successive_shortest_path(nodes, edges, num_of_servers):
    
    L = 10000
    path = []
    cost = 0

    for _ in range(num_of_servers):

        """ print("Iteration " + str(_))
        for arc in graph_arcs:
            print("ARC [From " + arc.from_ + " to " + arc.to_ + "] Cost: " + str(arc.cost) + " Flow: " + str(arc.flow)) """

        nodes, path_dict = bellman_ford(nodes, edges)
        path = find_shortest_path(path_dict)
        cost += cost_of_path(path, L, edges)

        """ for i in range(len(path) - 1):
            path_from = path[i]
            path_to = path[i+1]
            for arc in edges:
                if arc.is_arc_from_to(path_from, path_to):
                    if arc.cost != L and arc.cost != -L:
                        cost += (arc.cost)
                    break """

        edges = flow_augmentation(path, edges)
        edges = residual_graph_creation(edges)

    #print("Final cost is " + str(cost))

    return cost


def flow_augmentation(path, edges):
    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i+1]
        for arc in edges:
            if arc.is_arc_from_to(from_node, to_node):
                arc.flow = 1

    for arc in edges:
        check = False
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i+1]

            if arc.is_arc_from_to(from_node, to_node):
                check = True
                break

        if not check:
            arc.flow = 0 

    return edges



def graph_creation(server_loc, requests, distance_matrix):
    L = 10000

    G = nx.DiGraph()
    G.add_node('source')
    G.add_node('sink')

    nodes = {}

    nodes['source'] = Node("source", 0, 0)
    nodes['sink'] = Node("sink", 0, sys.maxsize)

    server_nodes = []
    for i_serv, server in enumerate(server_loc):
        server_nodes.append(Node("server" + str(i_serv), server, sys.maxsize))
        nodes["server" + str(i_serv)] = server_nodes[i_serv]
        G.add_node("server" + str(i_serv))

    request_nodes = []
    _request_nodes = []
    for i_req, req in enumerate(requests):
        request_nodes.append(Node("request" + str(i_req), req, sys.maxsize))
        _request_nodes.append(Node("_request" + str(i_req), req, sys.maxsize))
        nodes["request" + str(i_req)] = request_nodes[i_req]
        nodes["_request" + str(i_req)] = _request_nodes[i_req]
        G.add_node("request" + str(i_req))
        G.add_node("_request" + str(i_req))

    arcs = []
    for i_sn, sn in enumerate(server_nodes):
        arcs.append(Arc('source', sn.name, 0))
        arcs.append(Arc(sn.name, 'sink', 0))
        G.add_edge('source', 'server' + str(i_sn), capacity=1, weight=0)
        G.add_edge('server' + str(i_sn), 'sink', capacity=1, weight=0)
        for i_rn, rn in enumerate(request_nodes):
            rn_ = _request_nodes[i_rn]
            arcs.append(Arc(sn.name, rn.name, distance_matrix[sn.num][rn.num]))
            G.add_edge('server' + str(i_sn), 'request' + str(i_rn), capacity=1, weight=distance_matrix[sn.num][rn.num])


    for i_rn, rn in enumerate(request_nodes):
        rn_ = _request_nodes[i_rn]
        arcs.append(Arc(rn.name, rn_.name, -L))
        G.add_edge( 'request' + str(i_rn), '_request' + str(i_rn), capacity=1, weight=-L)    
        for j in range(i_rn + 1, len(request_nodes)):
            rn = request_nodes[j]
            arcs.append(Arc(rn_.name, rn.name, distance_matrix[rn_.num][rn.num]))
            G.add_edge( '_request' + str(i_rn), 'request' + str(j), capacity=1, weight=distance_matrix[rn_.num][rn.num])
        arcs.append(Arc(rn_.name, 'sink', 0))
        G.add_edge( '_request' + str(i_rn), 'sink', capacity=1, weight=0)

    return nodes, arcs, G


def recursive_search(req_ind, serv_config, requests, distance_matrix):

    if req_ind >= len(requests):
        return 0

    min_dist = sys.maxsize
    req = requests[req_ind]
    
    for serv_ind, serv in enumerate(serv_config):
        old_serv = serv
        serv_config[serv_ind] = req
        dist = distance_matrix[old_serv][req] + recursive_search(req_ind + 1, serv_config, requests, distance_matrix)
        serv_config[serv_ind] = old_serv
        if dist < min_dist:
            min_dist = dist

    return min_dist


#if __name__=="__main__" :

    """ num_nodes = 5
    num_of_servers = 3

    distance_matrix = [[0, 1, 1, 1, 2],
                    [1, 0, 1, 1, 2],
                    [1, 1, 0, 1, 2],
                    [1, 1, 1, 0, 2],
                    [2, 2, 2, 2, 0]]

    requests = [4, 3, 0, 1, 2, 0, 1, 0, 2, 4]

    server_loc = [i for i in range(num_of_servers)] # the indexes of nodes array where the servers are located """


    """ min_value = 0
    max_value = 100
    dimension = 2
    num_nodes = 20
    num_of_servers = 2

    # Generate nodes
    locations = generate_data.generate_nodes(min_value, max_value, num_nodes, dimension)
    #print("Nodes:\n", nodes)

    # Calculate distance matrix
    distance_matrix = generate_data.calculate_distance_matrix(locations)
    #print("Distance Matrix:\n", distance_matrix)

    server_loc = [i for i in range(num_of_servers)] # the indexes of nodes array where the servers are located
    total_dist = 0
    avg_total_dist = 0

    num_of_requests = 10
    num_of_iters = 1
    old_serv_loc_ind = 0

    requests = []

    for i in range(num_of_requests):
        request_loc = np.random.randint(0, num_nodes)
        requests.append(request_loc)


    #transform to graph
    graph_nodes, graph_arcs, G = graph_creation(server_loc, requests)

    ssp_cost = successive_shortest_path(graph_nodes, graph_arcs)
    print("Successive shortest path: " + str(ssp_cost))

    mincostFlow = nx.max_flow_min_cost(G, 'source', 'sink', capacity='capacity', weight='weight')
    mincost = nx.cost_of_flow(G, mincostFlow)
    print("Max flow min cost: " + str(mincost + len(requests) * 1000)) """

    """ total_cost = recursive_search(0, [i for i in range(num_of_servers)])
    print("Recursive: " + str(total_cost)) """

    """ server_path = {}
    for key in mincostFlow['source']:
        server_path[key] = 'Move ' + key
        currentKey = key

        while mincostFlow[currentKey]:

            currentDict = mincostFlow[currentKey]

            for key1 in currentDict:

                if currentDict[key1] == 1:
                    server_path[key] += ' to ' + key1
                    currentKey = key1

    for key in server_path:
        print(server_path[key]) """

    """ for key in graph_nodes:
        print("Node: " + key + " Cost: " + str(graph_nodes[key].cost))

    for arc in residual_graph:
        print("ARC [From " + arc.from_ + " to " + arc.to_ + "] Cost: " + str(arc.cost)) """

    """ print("Bellman-ford: " + str(new_nodes["sink"].cost + len(requests) * 1000))

    mincostFlow = nx.max_flow_min_cost(G, 'source', 'sink', capacity='capacity', weight='weight')
    mincost = nx.cost_of_flow(G, mincostFlow)

    print("Max flow min cost: " + str(mincost + len(requests) * 1000)) """

    """ server_path = {}
    for key in mincostFlow['source']:
        server_path[key] = 'Move ' + key
        currentKey = key

        while mincostFlow[currentKey]:

            currentDict = mincostFlow[currentKey]

            for key1 in currentDict:

                if currentDict[key1] == 1:
                    server_path[key] += ' to ' + key1
                    currentKey = key1

        
    for key in server_path:
        print(server_path[key]) """

    """ total_cost = recursive_search(0, [i for i in range(num_of_servers)])
    print("Recursive: " + str(total_cost)) """

    """ for arc in arcs:
        print("ARC [From " + arc.from_ + " to " + arc.to_ + "] Cost: " + str(arc.cost)) """
    