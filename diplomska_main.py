import generate_data
import optimal_offline
import greedy_search
import balanced_alg
import wfa
import numpy as np
import networkx as nx
import time


def generate_non_uniform_probabilities(locations, percentage_high_weight, high_weight):
    total_locations = len(locations)
    high_weight_count = int(total_locations * percentage_high_weight / 100)
    low_weight_count = total_locations - high_weight_count

    weights = [high_weight] * high_weight_count + [1] * low_weight_count

    np.random.shuffle(weights)

    probabilities = [w / sum(weights) for w in weights]

    return probabilities


if __name__=="__main__" :

    slo_distribution = np.array([0.414, 0.141, 0.055, 0.054, 0.038, 0.037, 0.035, 0.026, 0.020, 0.020, 0.019, 0.019, 0.019, 0.017, 0.017, 0.016, 0.014, 0.014, 0.013, 0.012])

    distance_matrix = np.round([
                    [0, 130.0, 29.6, 78.3, 106.0, 76.0, 71.2, 138.0, 27.0, 61.2, 62.3, 13.8, 106.0, 110.0, 28.1, 179.0, 31.3, 52.3, 20.7, 106.0],
                    [130.0, 0, 155.0, 53.4, 234.0, 73.5, 128.0, 30.2, 108.0, 87.1, 187.0, 114.0, 234.0, 238.0, 146.0, 57.9, 159.0, 180.0, 150.0, 22.3],
                    [29.6, 155.0, 0, 107.0, 131.0, 105.0, 101.0, 167.0, 25.2, 90.3, 35.2, 27.0, 131.0, 135.0, 12.5, 208.0, 55.9, 76.9, 45.3, 135.0],
                    [78.3, 53.4, 107.0, 0, 186.0, 25.5, 82.9, 64.9, 60.4, 39.2, 139.0, 64.3, 186.0, 190.0, 103.0, 105.0, 111.0, 132.0, 100.0, 32.2],
                    [106.0, 234.0, 131.0, 186.0, 0, 183.0, 171.0, 246.0, 129.0, 169.0, 164.0, 120.0, 90.6, 7.8, 125.0, 286.0, 81.0, 59.2, 87.1, 213.0],
                    [76.0, 73.5, 105.0, 25.5, 183.0, 0, 135.0, 85.4, 56.8, 36.5, 137.0, 60.7, 182.0, 186.0, 95.0, 126.0, 107.0, 128.0, 96.7, 52.8],
                    [71.2, 128.0, 101.0, 82.9, 171.0, 135.0, 0, 140.0, 88.4, 74.9, 135.0, 79.8, 172.0, 176.0, 95.5, 188.0, 97.1, 118.0, 86.6, 107.0],
                    [138.0, 30.2, 167.0, 64.9, 246.0, 85.4, 140.0, 0, 112.0, 90.2, 190.0, 115.0, 246.0, 241.0, 149.0, 48.3, 162.0, 192.0, 151.0, 25.1],
                    [27.0, 108.0, 25.2, 60.4, 129.0, 56.8, 88.4, 112.0, 0, 52.7, 57.4, 9.3, 129.0, 133.0, 27.2, 161.0, 54.3, 75.3, 43.7, 87.9],
                    [61.2, 87.1, 90.3, 39.2, 169.0, 36.5, 74.9, 90.2, 52.7, 0, 122.0, 47.5, 169.0, 173.0, 80.8, 139.0, 94.1, 115.0, 83.5, 66.1],
                    [62.3, 187.0, 35.2, 139.0, 164.0, 137.0, 135.0, 190.0, 57.4, 122.0, 0, 60.4, 164.0, 168.0, 46.2, 241.0, 89.2, 110.0, 78.6, 168.0],
                    [13.8, 114.0, 27.0, 64.3, 120.0, 60.7, 79.8, 115.0, 9.3, 47.5, 60.4, 0, 120.0, 124.0, 29.1, 165.0, 44.8, 65.8, 34.3, 91.8],
                    [106.0, 234.0, 131.0, 186.0, 90.6, 182.0, 172.0, 246.0, 129.0, 169.0, 164.0, 120.0, 0, 80.8, 126.0, 288.0, 62.2, 60.8, 71.2, 215.0],
                    [110.0, 238.0, 135.0, 190.0, 7.8, 186.0, 176.0, 241.0, 133.0, 173.0, 168.0, 124.0, 80.8, 0, 129.0, 290, 85.0, 63.2, 91.1, 217.0],
                    [28.1, 146.0, 12.5, 103.0, 125.0, 95.0, 95.5, 149.0, 27.2, 80.8, 46.2, 29.1, 126.0, 129.0, 0, 197.0, 49.0, 70.0, 38.4, 125.0],
                    [179.0, 57.9, 208.0, 105.0, 286.0, 126.0, 188.0, 48.3, 161.0, 139.0, 241.0, 165.0, 288.0, 290.0, 197.0, 0, 210.0, 231.0, 200.0, 73.7],
                    [31.3, 159.0, 55.9, 111.0, 81.0, 107.0, 97.1, 162.0, 54.3, 94.1, 89.2, 44.8, 62.2, 85.0, 49.0, 210.0, 0, 26.9, 10.1, 139.0],
                    [52.3, 180.0, 76.9, 132.0, 59.2, 128.0, 118.0, 192.0, 75.3, 115.0, 110.0, 65.8, 60.8, 63.2, 70.0, 231.0, 26.9, 0, 33.8, 160.0],
                    [20.7, 150.0, 45.3, 100.0, 87.1, 96.7, 86.6, 151.0, 43.7, 83.5, 78.6, 34.3, 71.2, 91.1, 38.4, 200.0, 10.1, 33.8, 0, 128.0],
                    [106.0, 22.3, 135.0, 32.2, 213.0, 52.8, 107.0, 25.1, 87.9, 66.1, 168.0, 91.8, 215.0, 217.0, 125.0, 73.7, 139.0, 160.0, 128.0, 0]
                    ])
    
    #np.random.seed(42)

    min_value = 0
    max_value = 1000
    dimension = 2
    num_nodes = 50
    num_of_servers = 2
    num_of_requests = 100

    locations = generate_data.generate_nodes(min_value, max_value, num_nodes, dimension)
    #print("Nodes:\n", nodes)

    # Calculate distance matrix
    distance_matrix = generate_data.calculate_distance_matrix(locations)
    #print("Distance Matrix:\n", distance_matrix)

    greedy_comp_ratio = 0
    bal_comp_ratio = 0
    wfa_comp_ratio = 0
    mwfa_comp_ratio = 0
    better_mwfa_comp_ratio = 0

    ssp_time = 0
    greedy_time = 0
    bal_time = 0
    WFA_time = 0
    MWFA_time = 0
    fastWFA_time = 0

    num_of_iterations = 10
    print("k = " + str(num_of_servers) + ", m = " + str(num_nodes) + ", n = " + str(num_of_requests))
    for _ in range(num_of_iterations):

        server_locations = [i for i in range(num_of_servers)] # the indexes of nodes array where the servers are located

        requests = []

        percentage_high_weight = 100
        high_weight = 1
        uniform = generate_non_uniform_probabilities(locations, percentage_high_weight, high_weight)
        #print(uniform)

        percentage_high_weight = 10
        high_weight = 10
        moderately_non_uniform = generate_non_uniform_probabilities(locations, percentage_high_weight, high_weight)
        #print(moderately_non_uniform)

        percentage_high_weight = 10
        high_weight = 50
        highly_non_uniform = generate_non_uniform_probabilities(locations, percentage_high_weight, high_weight)
        #print(highly_non_uniform)

        requests = np.random.choice(num_nodes, num_of_requests, p=uniform)

        graph_nodes, graph_arcs, G = optimal_offline.graph_creation([i for i in range(num_of_servers)], requests, distance_matrix)

        start = time.time()
        ssp_cost = optimal_offline.successive_shortest_path(graph_nodes, graph_arcs, num_of_servers)
        end = time.time()
        ssp_time += (end - start)
        #print("Optimal solution: Successive shortest path: " + str(ssp_cost)) 
        #print()

        """ start = time.time()
        mincostFlow = nx.max_flow_min_cost(G, 'source', 'sink', capacity='capacity', weight='weight')
        mincost = nx.cost_of_flow(G, mincostFlow)
        ssp_cost = mincost + len(requests) * 10000
        end = time.time()
        mincost_time = (end - start)
        print("Mincost time elapsed: " + str(mincost_time) + "s") """
        

        """ recursive_optimal_cost = optimal_offline.recursive_search(0, [i for i in range(num_of_servers)], requests, distance_matrix)
        print("Recursive optimal cost: " + str(recursive_optimal_cost)) """

        start = time.time()
        greedy_cost = greedy_search.greedy_algorithm(requests, [i for i in range(num_of_servers)], distance_matrix)
        greedy_comp_ratio += (greedy_cost / ssp_cost)
        #print("Greedy cost: " + str(greedy_cost))
        end = time.time()
        greedy_time += (end - start)

        start = time.time()
        balanced_cost = balanced_alg.balanced_algorithm(requests, [i for i in range(num_of_servers)], distance_matrix)
        bal_comp_ratio += (balanced_cost / ssp_cost)
        end = time.time()
        bal_time += (end - start)

        start = time.time()
        wfa_cost = wfa.work_function_algorithm(requests, [i for i in range(num_of_servers)], distance_matrix)
        wfa_comp_ratio += (wfa_cost / ssp_cost)
        #print("WFA cost: " + str(wfa_cost))
        end = time.time()
        WFA_time += (end - start)
        #print("WFA time elapsed: " + str(end - start) + "s")

        start = time.time()
        window = 20
        modified_wfa_cost = wfa.modified_wfa(requests, [i for i in range(num_of_servers)], distance_matrix, window)
        mwfa_comp_ratio += (modified_wfa_cost / ssp_cost)
        #print("w-WFA cost: " + str(modified_wfa_cost))
        end = time.time()
        MWFA_time += (end - start)
        #print("MWFA time elapsed: " + str(end - start) + "s")

        start = time.time()
        window = num_of_requests
        better_modified_wfa_cost = wfa.fast_wfa(requests, [i for i in range(num_of_servers)], distance_matrix, window)
        better_mwfa_comp_ratio += (better_modified_wfa_cost / ssp_cost)
        #print("Better w-WFA cost: " + str(better_modified_wfa_cost))
        end = time.time()
        fastWFA_time += (end - start)
        #print("Fast WFA Time elapsed: " + str(end - start))


    ssp_time /= num_of_iterations
    print("SSP average time: " + str(round(ssp_time, 3)))
    print()

    greedy_comp_ratio /= num_of_iterations
    greedy_time /= num_of_iterations
    print("Greedy Competitive ratio: " + str(round(greedy_comp_ratio, 3)))
    print("Greedy average time: " + str(round(greedy_time, 6)))
    print()

    bal_comp_ratio /= num_of_iterations
    bal_time /= num_of_iterations
    print("Balanced Competitive ratio: " + str(round(bal_comp_ratio, 3)))
    print("Balanced average time: " + str(round(bal_time, 6)))
    print()

    wfa_comp_ratio /= num_of_iterations
    WFA_time /= num_of_iterations
    print("WFA Competitive ratio: " + str(round(wfa_comp_ratio, 3)))
    print("WFA average time: " + str(round(WFA_time, 3)))
    print()

    mwfa_comp_ratio /= num_of_iterations
    MWFA_time /= num_of_iterations
    print("MWFA Competitive ratio: " + str(round(mwfa_comp_ratio, 3)))
    print("MWFA average time: " + str(round(MWFA_time, 3)))
    print()

    better_mwfa_comp_ratio /= num_of_iterations
    fastWFA_time /= num_of_iterations
    print("Fast WFA Competitive ratio: " + str(round(better_mwfa_comp_ratio, 3)))
    print("Fast WFA average time: " + str(round(fastWFA_time, 3)))
    print()