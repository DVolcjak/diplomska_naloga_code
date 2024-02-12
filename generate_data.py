import numpy as np

def generate_nodes(min_value, max_value, num_nodes, dimension):
    #np.random.seed(42)

    # Generate random coordinates for the nodes
    nodes = np.random.randint(min_value, max_value, size=(num_nodes, dimension))
    return nodes

def calculate_distance_matrix(nodes):
    num_nodes = nodes.shape[0]
    distances = np.zeros((num_nodes, num_nodes))

    # Calculate the distances between each pair of nodes
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            dist = np.linalg.norm(nodes[i] - nodes[j])
            distances[i][j] = np.round(dist)
            distances[j][i] = np.round(dist)

    return distances

""" # Example usage
min_value = 0
max_value = 100
num_nodes = 10
dimension = 2

# Generate nodes
nodes = generate_nodes(min_value, max_value, num_nodes, dimension)
print("Nodes:\n", nodes)

# Calculate distance matrix
distance_matrix = calculate_distance_matrix(nodes)
print("Distance Matrix:\n", distance_matrix)  """