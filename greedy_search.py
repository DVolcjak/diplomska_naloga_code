import generate_data
import numpy as np
import sys
import bisect

def greedy_algorithm(requests, server_loc, distance_matrix):

    total_dist = 0

    for i in range(len(requests)):
        request_loc = requests[i]
        min_distance = sys.maxsize

        for serv_ind, serv in enumerate(server_loc):
            dist = distance_matrix[serv][request_loc]

            #print("The cost of moving server " + str(serv_ind) + " from " + str(serv) + " to " + str(request_loc) + " is " + str(dist))

            if dist < min_distance:
                min_distance = dist
                old_serv_loc_ind = serv_ind

        total_dist += min_distance

        """ print("Current server config: " + str(server_loc))
        print("Moving server " + str(old_serv_loc_ind) + " from " + str(server_loc[old_serv_loc_ind]) + " to " + str(request_loc)) """

        """ server_loc.pop(old_serv_loc_ind)
        new_serv_loc_ind = bisect.bisect_left(server_loc, request_loc)
        server_loc.insert(new_serv_loc_ind, request_loc) """
        server_loc[old_serv_loc_ind] = request_loc

        """ print("New server config: " + str(server_loc)) """

    #print(np.round(total_dist, 3))
    return np.round(total_dist, 3)
