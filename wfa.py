import generate_data
import numpy as np
import sys
import itertools
import networkx as nx
from matplotlib import pyplot as plt
import re


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


def generate_configurations(k, n):
    configurations = list(itertools.combinations(range(0, n), k))
    return configurations

def generate_permutations(nums):
    permutations = list(itertools.permutations(nums))
    return permutations


def work_function_algorithm(requests, server_loc, distance_matrix):

    configurations = generate_configurations(len(server_loc), len(distance_matrix))

    permutations = generate_permutations(server_loc)

    server_loc = tuple([i for i in range(len(server_loc))])
    w_matrix = []
    ws_matrix = []
    for config in configurations:
        w0 = sys.maxsize
        for perm in permutations:
            cost = 0
            cost = sum(distance_matrix[i][j] for i, j in zip(perm, config))
        
            if cost < w0:
                w0 = cost

        w_matrix.append(np.round(w0))
        #w_matrix.append(w0)

    ws_matrix.append(w_matrix)
    #print(ws_matrix)


    move_matrix = []
    #requests = [4, 3, 0, 1, 2, 0, 1, 0, 2, 4]
    for i in range(len(requests)):
        """ request_loc = np.random.randint(0, num_nodes)
        requests.append(request_loc) """

        request_loc = requests[i]
        w_i = []
        move_i = []

        for j, config in enumerate(configurations):

            if request_loc in config:
                w_i.append(ws_matrix[i][j])
                move_i.append(request_loc)

            else:
                wi_min = sys.maxsize
                move_j = 0
                for conf_ind, serv_ind in enumerate(config):
                    cost = 0
                    w_prev = config[:conf_ind] + (request_loc,) + config[conf_ind+1:]
                    w_sort = tuple(sorted(w_prev))
                    ind = configurations.index(w_sort)

                    cost = ws_matrix[i][ind] + distance_matrix[serv_ind][request_loc]

                    if cost < wi_min:
                        wi_min = cost
                        move_j = serv_ind

                w_i.append(np.round(wi_min))
                move_i.append(move_j)
                #w_i.append(wi_min)

        #print(w_i)

        ws_matrix.append(w_i)
        move_matrix.append(move_i)
                    
    #print(ws_matrix)
    #print(move_matrix)

    total_dist = 0

    for i in range(len(move_matrix)):
        j = configurations.index(server_loc)
        old_loc = move_matrix[i][j]
        new_loc = requests[i]
        total_dist += distance_matrix[old_loc][new_loc]
        old_loc_ind = server_loc.index(old_loc)
    
        #print("Move server " + str(old_loc_ind) + " from " + str(old_loc) + " to " + str(new_loc) + " with cost " + str(distance_matrix[old_loc][new_loc]) + ".")

        new_server_loc = list(server_loc)
        new_server_loc[old_loc_ind] = new_loc
        server_loc = tuple(sorted(new_server_loc))


    return np.round(total_dist)


def graph_creation(G, initial_config, current_config, requests, distance_matrix):

    L = 1000

    G.add_node('source')
    G.add_node('sink')

    nodes = {}

    nodes['source'] = Node("source", 0, 0)
    nodes['sink'] = Node("sink", 0, sys.maxsize)

    initial_server_nodes = []
    for i_serv, server in enumerate(initial_config):
        initial_server_nodes.append(Node("server" + str(i_serv), server, sys.maxsize))
        nodes["server" + str(i_serv)] = initial_server_nodes[i_serv]
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

    current_server_nodes = []
    for i_serv, server in enumerate(current_config):
        current_server_nodes.append(Node("_server" + str(i_serv), server, sys.maxsize))
        nodes["_server" + str(i_serv)] = current_server_nodes[i_serv]
        G.add_node("_server" + str(i_serv))

    arcs = []
    for i_sn, sn in enumerate(initial_server_nodes):
        arcs.append(Arc('source', sn.name, 0))
        G.add_edge('source', 'server' + str(i_sn), capacity=1, weight=0)

        for i_rn, rn in enumerate(request_nodes):
            rn_ = _request_nodes[i_rn]
            arcs.append(Arc(sn.name, rn.name, distance_matrix[sn.num][rn.num]))
            G.add_edge('server' + str(i_sn), 'request' + str(i_rn), capacity=1, weight=distance_matrix[sn.num][rn.num])

        for i_csn, csn in enumerate(current_server_nodes):
            arcs.append(Arc(sn.name, csn.name, distance_matrix[sn.num][csn.num]))
            G.add_edge('server' + str(i_sn), '_server' + str(i_csn), capacity=1, weight=distance_matrix[sn.num][csn.num])


    for i_rn, rn in enumerate(request_nodes):
        rn_ = _request_nodes[i_rn]
        arcs.append(Arc(rn.name, rn_.name, -L))
        G.add_edge( 'request' + str(i_rn), '_request' + str(i_rn), capacity=1, weight=-L)

        for j in range(i_rn + 1, len(request_nodes)):
            rn = request_nodes[j]
            arcs.append(Arc(rn_.name, rn.name, distance_matrix[rn_.num][rn.num]))
            G.add_edge( '_request' + str(i_rn), 'request' + str(j), capacity=1, weight=distance_matrix[rn_.num][rn.num])

        for i_csn, csn in enumerate(current_server_nodes):
            arcs.append(Arc(rn_.name, csn.name, distance_matrix[rn_.num][csn.num]))
            G.add_edge( '_request' + str(i_rn), '_server' + str(i_csn), capacity=1, weight=distance_matrix[rn_.num][csn.num])
        
    for i_csn, csn in enumerate(current_server_nodes):
        arcs.append(Arc(csn.name, 'sink', 0))
        G.add_edge('_server' + str(i_csn), 'sink', capacity=1, weight=0)

    return G

def graph_update(G, initial_config, current_config, requests, distance_matrix):

    L = 1000
    num_of_req = 0
    current_req = 0

    if len(requests) > 0:
        num_of_req = len(requests) - 1
        current_req = requests[num_of_req]
        G.add_node('request' + str(num_of_req))
        G.add_node('_request' + str(num_of_req))
        G.add_edge('request' + str(num_of_req), '_request' + str(num_of_req), capacity=1, weight=-L)

    for i_serv, serv in enumerate(initial_config):

        if len(requests) > 0:
            serv_loc = current_config[i_serv]
            G.add_edge('server' + str(i_serv), 'request' + str(num_of_req), capacity=1, weight=distance_matrix[serv][current_req])
            G.add_edge('_request' + str(num_of_req), '_server' + str(i_serv), capacity=1, weight=distance_matrix[current_req][serv_loc])

        for i_cserv, cserv in enumerate(current_config):
            G['server' + str(i_serv)]['_server' + str(i_cserv)]['weight'] = distance_matrix[serv][cserv]

    if len(requests) > 0:
        for i_req in range(num_of_req):
            req = requests[i_req]
            G.add_edge('_request' + str(i_req), 'request' + str(num_of_req), capacity=1, weight=distance_matrix[req][current_req])

            for i_cserv, cserv in enumerate(current_config):
                G['_request' + str(i_req)]['_server' + str(i_cserv)]['weight'] = distance_matrix[req][cserv]

    return G

def graph_update_window(G, initial_config, current_config, requests, distance_matrix):

    for i_serv, serv in enumerate(initial_config):

        for i_req, req in enumerate(requests):
            G['server' + str(i_serv)]['request' + str(i_req)]['weight'] = distance_matrix[serv][req]

        for i_cserv, cserv in enumerate(current_config):
            G['server' + str(i_serv)]['_server' + str(i_cserv)]['weight'] = distance_matrix[serv][cserv]

    for _i_req, _req in enumerate(requests):

        for i_req in range(_i_req + 1, len(requests)):
            req = requests[i_req]
            G['_request' + str(_i_req)]['request' + str(i_req)]['weight'] = distance_matrix[_req][req]

        for i_cserv, cserv in enumerate(current_config):
            G['_request' + str(_i_req)]['_server' + str(i_cserv)]['weight'] = distance_matrix[_req][cserv]

    return G

def modified_wfa(requests, server_loc, distance_matrix, w):

    L = 1000
    initial_config = server_loc.copy()
    server_config = server_loc.copy()
    req_window = []
    server_movement = []
    current_cost = 0
    mod_wfa_cost = 0

    #print(requests)

    G = nx.DiGraph()
    G = graph_creation(G, server_loc, server_config, req_window, distance_matrix)

    for i, req in enumerate(requests):
        min_cost = sys.maxsize
        serv_to_move = 0

        if len(req_window) >= w:
            s_to_move = server_movement[i-w]
            server_loc[s_to_move] = req_window[0]
            req_window.pop(0)

        #print(str(i) + ". request")

        if req not in server_config:
            for j in range(len(server_config)):

                old_serv_loc = server_config[j]
                server_config[j] = req

                """ print("Starting server locations: " + str(server_loc))
                print("Moving server from " + str(old_serv_loc) + " to " + str(req)) """
                if i >= w:
                    G = graph_update_window(G, server_loc, server_config, req_window, distance_matrix)
                    """ print("Moving " + str(j) + ". server")
                    for arc in G.edges.data():
                        print("ARC [From " + arc[0] + " to " + arc[1] + "] Cost: " + str(arc[2]["weight"])) """
                else:
                    G = graph_update(G, server_loc, server_config, req_window, distance_matrix)
                    """ print("Moving " + str(j) + ". server")
                    for arc in G.edges.data():
                        print("ARC [From " + arc[0] + " to " + arc[1] + "] Cost: " + str(arc[2]["weight"])) """


                """ negative_cycle = nx.negative_edge_cycle(G)
                if negative_cycle is not None:
                    print("The graph contains a negative cycle.")
                    print("Nodes in negative cycle:", negative_cycle)
                else:
                    print("The graph does not contain any negative cycles.") """


                mincostFlow = nx.max_flow_min_cost(G, 'source', 'sink', capacity='capacity', weight='weight')
                mincost = nx.cost_of_flow(G, mincostFlow)

                """ print("Max flow min cost: " + str(mincost + len(req_window) * 1000))
                print() """

                current_cost = mincost + len(req_window) * L
                if current_cost < min_cost:
                    min_cost = current_cost
                    serv_to_move = j

                """server_path = {}
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
                    print(server_path[key])
                """

                server_config[j] = old_serv_loc
        else:
            serv_to_move = server_config.index(req)
            server_config[serv_to_move] = req
            if i >= w:
                G = graph_update_window(G, server_loc, server_config, req_window, distance_matrix)
            else:
                G = graph_update(G, server_loc, server_config, req_window, distance_matrix)

        #print("Moving server " + str(serv_to_move) + " price is " + str(distance_matrix[server_config[serv_to_move]][req]))

        mod_wfa_cost += distance_matrix[server_config[serv_to_move]][req]
        server_config[serv_to_move] = req
        server_movement.append(serv_to_move)

        #print(server_config)
        req_window.append(req)

    return mod_wfa_cost



def fast_graph_creation(G, initial_config, current_config, requests, distance_matrix):

    L = 10000
    X = 0   # USED TO CALCULATE THE COST OF CONNECTIONS FROM CURRENT SERVER CONFIGUARTION NODES TO THE SINK
    current_req = requests[len(requests) - 1]

    G.add_node('source')
    G.add_node('sink')

    nodes = {}

    nodes['source'] = Node("source", 0, 0)
    nodes['sink'] = Node("sink", 0, sys.maxsize)

    initial_server_nodes = []
    for i_serv, server in enumerate(initial_config):
        initial_server_nodes.append(Node("server" + str(i_serv), server, sys.maxsize))
        nodes["server" + str(i_serv)] = initial_server_nodes[i_serv]
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

    current_server_nodes = []
    for i_serv, server in enumerate(current_config):
        current_server_nodes.append(Node("_server" + str(i_serv), server, sys.maxsize))
        nodes["_server" + str(i_serv)] = current_server_nodes[i_serv]
        G.add_node("_server" + str(i_serv))
        X += distance_matrix[server][current_req]

    X = (1/(len(initial_config) - 1)) * X

    arcs = []
    for i_sn, sn in enumerate(initial_server_nodes):
        arcs.append(Arc('source', sn.name, 0))
        G.add_edge('source', 'server' + str(i_sn), capacity=1, weight=0)

        for i_rn, rn in enumerate(request_nodes):
            rn_ = _request_nodes[i_rn]
            arcs.append(Arc(sn.name, rn.name, distance_matrix[sn.num][rn.num]))
            G.add_edge('server' + str(i_sn), 'request' + str(i_rn), capacity=1, weight=distance_matrix[sn.num][rn.num])

        for i_csn, csn in enumerate(current_server_nodes):
            arcs.append(Arc(sn.name, csn.name, distance_matrix[sn.num][csn.num]))
            G.add_edge('server' + str(i_sn), '_server' + str(i_csn), capacity=1, weight=distance_matrix[sn.num][csn.num])


    for i_rn, rn in enumerate(request_nodes):
        rn_ = _request_nodes[i_rn]
        arcs.append(Arc(rn.name, rn_.name, -L))
        G.add_edge( 'request' + str(i_rn), '_request' + str(i_rn), capacity=1, weight=-L)

        """ for j in range(i_rn + 1, len(request_nodes)):
            rn = request_nodes[j]
            arcs.append(Arc(rn_.name, rn.name, distance_matrix[rn_.num][rn.num]))
            G.add_edge( '_request' + str(i_rn), 'request' + str(j), capacity=1, weight=distance_matrix[rn_.num][rn.num])

        for i_csn, csn in enumerate(current_server_nodes):
            arcs.append(Arc(rn_.name, csn.name, distance_matrix[rn_.num][csn.num]))
            G.add_edge( '_request' + str(i_rn), '_server' + str(i_csn), capacity=1, weight=distance_matrix[rn_.num][csn.num]) """
        
    for i_csn, csn in enumerate(current_server_nodes):
        edge_weight = X - distance_matrix[csn.num][current_req]
        arcs.append(Arc(csn.name, 'sink', 0))
        G.add_edge('_server' + str(i_csn), 'sink', capacity=1, weight=edge_weight)

    G.add_edge('_request' + str(len(requests)-1), 'sink', capacity=1, weight=0)

    return G

def fast_graph_update(G, initial_config, current_config, requests, distance_matrix):

    L = 10000
    num_of_req = 0
    current_req = 0
    previous_req = 0

    if len(requests) > 0:
        num_of_req = len(requests) - 1
        current_req = requests[num_of_req]
        previous_req = requests[num_of_req-1]
        G.add_node('request' + str(num_of_req))
        G.add_node('_request' + str(num_of_req))    
        G.add_edge('request' + str(num_of_req), '_request' + str(num_of_req), capacity=1, weight=-L)
        G.add_edge('_request' + str(num_of_req), 'sink', capacity=1, weight=0)
        G.remove_edge('_request' + str(num_of_req-1), 'sink')

    for i_serv, serv in enumerate(initial_config):

        if len(requests) > 0:
            serv_loc = current_config[i_serv]
            # serv is the location of the server in initial config
            G.add_edge('server' + str(i_serv), 'request' + str(num_of_req), capacity=1, weight=distance_matrix[serv][current_req])
            # serv_loc is the location of the server in current config
            G.add_edge('_request' + str(num_of_req-1), '_server' + str(i_serv), capacity=1, weight=distance_matrix[previous_req][serv_loc])

        for i_cserv, cserv in enumerate(current_config):
            G['server' + str(i_serv)]['_server' + str(i_cserv)]['weight'] = distance_matrix[serv][cserv]

    if len(requests) > 0:
        for i_req in range(num_of_req):
            req = requests[i_req]
            G.add_edge('_request' + str(i_req), 'request' + str(num_of_req), capacity=1, weight=distance_matrix[req][current_req])

            for i_cserv, cserv in enumerate(current_config):
                G['_request' + str(i_req)]['_server' + str(i_cserv)]['weight'] = distance_matrix[req][cserv]

    # CALCULATION OF EDGE COSTS FROM CURRENT CONFIGURATION NODES TO SINK    
    X = (1/(len(initial_config) - 1)) * sum([distance_matrix[cserv][current_req] for cserv in current_config])
    for i_cserv, cserv in enumerate(current_config):
        G['_server' + str(i_cserv)]['sink']['weight'] = X - distance_matrix[cserv][current_req]
        #print("Cost from " + str(cserv) + " to sink is " + str((X - distance_matrix[cserv][current_req])))

    return G


def fast_graph_update_window(G, initial_config, current_config, requests, distance_matrix):

    current_req = requests[len(requests) - 1]

    for i_serv, serv in enumerate(initial_config):

        for i_req, req in enumerate(requests):
            G['server' + str(i_serv)]['request' + str(i_req)]['weight'] = distance_matrix[serv][req]

        for i_cserv, cserv in enumerate(current_config):
            G['server' + str(i_serv)]['_server' + str(i_cserv)]['weight'] = distance_matrix[serv][cserv]

    for _i_req in range(len(requests) - 1):

        _req = requests[_i_req]
        for i_req in range(_i_req + 1, len(requests)):
            req = requests[i_req]
            G['_request' + str(_i_req)]['request' + str(i_req)]['weight'] = distance_matrix[_req][req]

        for i_cserv, cserv in enumerate(current_config):
            G['_request' + str(_i_req)]['_server' + str(i_cserv)]['weight'] = distance_matrix[_req][cserv]

    # CALCULATION OF EDGE COSTS FROM CURRENT CONFIGURATION NODES TO SINK
    X = (1/(len(initial_config) - 1)) * sum([distance_matrix[cserv][current_req] for cserv in current_config])
    for i_cserv, cserv in enumerate(current_config):
        G['_server' + str(i_cserv)]['sink']['weight'] = X - distance_matrix[cserv][current_req]

    return G


def find_path_with_current_request(mincostFlowDict, num_of_req):
    server_path = {}
    server_key = ''
    serv_to_move = ''

    for key in mincostFlowDict['source']:
        server_path[key] = 'Move ' + key
        server_key = key
        currentKey = key

        while mincostFlowDict[currentKey]:

            currentDict = mincostFlowDict[currentKey]

            for key1 in currentDict:

                if currentDict[key1] == 1:
                    server_path[key] += ' to ' + key1
                    currentKey = key1

                    if key1 == 'request' + str(num_of_req - 1):
                        serv_to_move = server_key


    serv_ind = int(re.search(r'\d+$', serv_to_move).group())

    return serv_ind

def fast_wfa(requests, server_loc, distance_matrix, w):

    L = 10000
    initial_config = server_loc.copy()
    current_config = server_loc.copy()
    req_window = []
    server_movement = []
    current_cost = 0
    fast_wfa_cost = 0

    #print(requests)

    G = nx.DiGraph()

    for i, req in enumerate(requests):
        #min_cost = sys.maxsize

        req_window.append(req)

        if len(req_window) > w:
            s_to_move = server_movement[i-w]
            initial_config[s_to_move] = req_window[0]
            req_window.pop(0)

        #print("Iteration " + str(i))
        #print("Request window: " + str(req_window))
        #print("Initial config: " + str(initial_config))
        #print("Current config: " + str(current_config))

        """ old_serv_loc = server_config[j]
        server_config[j] = req """


        if i == 0:
            G = fast_graph_creation(G, initial_config, current_config, req_window, distance_matrix)
        else:
            G = fast_graph_update(G, initial_config, current_config, req_window, distance_matrix)

        """ print("Serving request " + str(i) + ".")
        for arc in G.edges.data():
            print("ARC [From " + arc[0] + " to " + arc[1] + "] Cost: " + str(arc[2]["weight"])) """


        """ mincostFlow = nx.max_flow_min_cost(G, 'source', 'sink', capacity='capacity', weight='weight')
        #mincost = nx.cost_of_flow(G, mincostFlow)

        serv_ind = find_path_with_current_request(mincostFlow, len(req_window))
        server = current_config[serv_ind] """

        if req not in current_config:
            #print("Graph info:", nx.info(G))
            mincostFlow = nx.max_flow_min_cost(G, 'source', 'sink', capacity='capacity', weight='weight')
            #mincost = nx.cost_of_flow(G, mincostFlow)

            serv_ind = find_path_with_current_request(mincostFlow, len(req_window))
            server = current_config[serv_ind]
        else:
            serv_ind = current_config.index(req)
            server = req        

        #print("Moving server " + str(serv_ind) + " to location " + str(req) + " | Cost: " + str(distance_matrix[server][req]))

        fast_wfa_cost += distance_matrix[server][req]
        current_config[serv_ind] = req
        server_movement.append(serv_ind)

    return fast_wfa_cost