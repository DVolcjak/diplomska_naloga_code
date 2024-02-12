import generate_data
import numpy as np
import sys

def balanced_algorithm(requests, server_loc, distance_matrix):

    cost = 0
    distances = [0 for _ in server_loc]
    old_serv_loc_ind = 0

    for i in range(len(requests)):
        #request_loc = np.random.randint(0, num_nodes)
        request_loc = requests[i]
        min_distance = sys.maxsize
        dist_is_zero = False
        dist = 0

        for serv_ind, serv in enumerate(server_loc):
            dist = distance_matrix[serv][request_loc]

            if dist == 0:
                dist_is_zero = True
                break

            cum_dist = dist + distances[serv_ind]

            if cum_dist < min_distance:
                min_distance = cum_dist
                old_serv_loc_ind = serv_ind

        """ print(min_distance, old_serv_loc_ind)

        min_distance, old_serv_loc_ind = min(
            enumerate(server_loc),
            key=lambda x: (distance_matrix[x[1]][request_loc] + distances[x[0]], x[0]) if distance_matrix[x[1]][request_loc] != 0 else (distances[x[0]], x[0])
        )

        min_distance, old_serv_loc_ind = min(
            enumerate(server_loc),
            key=lambda x: (distances[x[0]] if distance_matrix[x[1]][request_loc] == 0 else distances[x[0]] + distance_matrix[x[1]][request_loc], x[0])
        )

        print(min_distance, old_serv_loc_ind)

        min_distance, old_serv_loc_ind = min(
            ((distances[serv_ind] + distance_matrix[serv][request_loc], serv_ind) for serv_ind, serv in enumerate(server_loc)),
            key=lambda x: (x[0] if x[0] != 0 else float('inf'), x[1])
        )
        """

        if not dist_is_zero:
            server_loc[old_serv_loc_ind] = request_loc
            distances[old_serv_loc_ind] = min_distance

        """ server_loc[old_serv_loc_ind] = request_loc
        distances[old_serv_loc_ind] = min_distance """

    cost += sum(distances)

    return np.round(cost)