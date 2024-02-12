import generate_data
import numpy as np
import sys


min_value = 0
max_value = 100
num_nodes = 10
dimension = 2
num_of_servers = 2

# Generate nodes
nodes = generate_data.generate_nodes(min_value, max_value, num_nodes, dimension)
#print("Nodes:\n", nodes)

# Calculate distance matrix
distance_matrix = generate_data.calculate_distance_matrix(nodes)
#print("Distance Matrix:\n", distance_matrix)

server_loc = [i for i in range(num_of_servers)]  # the indexes of nodes array where the servers are located
total_dist = 0
avg_total_dist = 0

num_of_requests = 10
num_of_iters = 1
old_serv_loc = 0


for iters in range(num_of_iters):
    total_dist = 0

    for i in range(num_of_requests):
        serv_at_request = False
        request_loc = np.random.randint(0, num_nodes)

        for serv_ind, serv in enumerate(server_loc):
            dist = distance_matrix[serv][request_loc]

            if dist == 0:
                serv_at_request = True
                break

        if not serv_at_request:
            random_index = np.random.randint(0, num_of_servers)  # select's a random index of the array with server locations 
            old_serv_loc = server_loc[random_index]  # get's the location at which the random server is
            total_dist += distance_matrix[old_serv_loc][request_loc]  # add's the distance from the server old location to the location of the request
            server_loc[random_index] = request_loc  # moves the server

    avg_total_dist += total_dist

avg_total_dist /= num_of_iters 

print("The total distance travelled by the servers is: " + str(np.round(avg_total_dist)))