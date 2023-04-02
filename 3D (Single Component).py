import random
import numpy as np
from collections import defaultdict
import time

# For timing purposes
start_time = time.time()
def update(string, last_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_rel = end_time - last_time
    print(string + "Total Time elapsed: {:.2f} seconds".format(elapsed_time) +  " Time elapsed since last update: {:.2f} seconds".format(elapsed_rel) )
    return end_time

# Set the size of the grid
grid_size = 20
probability = 0.7

# Define the dimensions of the grid
total_rows = 2 * grid_size + 1
total_cols = 2 * grid_size + 1
total_layers = 2 * grid_size + 1

# Initialize a grid of tuples with indices and perform coin-flips
grid = [[[(1 if random.random() < probability else 0) for k in range(total_layers)] for j in range(total_cols)] for i in range(total_rows)]

# Initialize a grid to hold the distance to the nearest zero
dist_grid = [[[None for _ in range(total_layers)] for _ in range(total_cols)] for _ in range(total_rows)]

# Updates for initialization
time_init = update("Initialization done", start_time)

# Calculate the distance to the nearest zero for each point in the grid
for i in range(total_rows):
    for j in range(total_cols):
        for k in range(total_layers):
            min_distance = float("inf")
            for x in range(total_rows):
                for y in range(total_cols):
                    for z in range(total_layers):
                        if grid[x][y][z] == 0:
                            distance = abs(i - x) + abs(j - y) + abs(k - z)
                            min_distance = min(min_distance, distance)
            dist_grid[i][j][k] = min_distance


# Updates fpr distance
time_dist = update("Distance calculations done", time_init)

# Initialize a grid to hold next point
pointer_grid = [[[None for _ in range(total_layers)] for _ in range(total_cols)] for _ in range(total_rows)]

# Calculate the pointers to the nearest point of height one more for each point in the grid
for i in range(total_rows):
    for j in range(total_cols):
        for k in range(total_layers):
            current_height = dist_grid[i][j][k]
            min_distance = float("inf")
            next_point = None
            for x in range(total_rows):
                for y in range(total_cols):
                    for z in range(total_layers):
                        if dist_grid[x][y][z] == current_height + 1:
                            distance = abs(i - x) + abs(j - y) + abs(k - z)
                            if distance < min_distance:
                                min_distance = distance
                                next_point = (x, y, z)
            pointer_grid[i][j][k] = next_point


# Updates for pointers
time_point = update("Pointer calculations done", time_dist)

# Find the root of the tree.
def find_root(node, pointer_grid):
    i, j, k = node
    while pointer_grid[i][j][k] is not None:
        i, j, k = pointer_grid[i][j][k]
    return (i, j, k)

# Find connected components
max_height_points = {}
for i in range(total_rows):
    for j in range(total_cols):
        for k in range(total_layers):
            if grid[i][j][k] == 1:
                max_height_points[(i, j, k)] = find_root((i, j, k), pointer_grid)

connected_components = defaultdict(list)
for point, max_height_point in max_height_points.items():
    connected_components[max_height_point].append(point)

connected_components = list(connected_components.values())

# Print time elapsed for computation
time_comp = update("Computations done", time_point)


import plotly.graph_objs as go

# Generates colors
def generate_color_dict(connected_components):
    random_colors = np.random.rand(len(connected_components), 3)
    color_dict = {}
    for idx, component in enumerate(connected_components):
        for point in component:
            color_dict[point] = random_colors[idx]
    return color_dict

color_dict = generate_color_dict(connected_components)

# Find the connected component containing the point (n, n)
n = grid_size
root_nn = find_root((n, n, n), pointer_grid)
component_nn = None
for component in connected_components:
    if root_nn in component:
        component_nn = component
        break

if component_nn is None:
    print("No component found at (n, n).")
else:
    scatter_colors = [f'rgb({int(color_dict[(i, j, k)][0] * 255)}, {int(color_dict[(i, j, k)][1] * 255)}, {int(color_dict[(i, j, k)][2] * 255)})' for i, j, k in component_nn]

    scatter = go.Scatter3d(x=[i for i, j, k in component_nn],
                           y=[j for i, j, k in component_nn],
                           z=[k for i, j, k in component_nn],
                           mode='markers',
                           marker=dict(size=4, color=scatter_colors))

    fig = go.Figure(data=[scatter])
    fig.update_layout(scene=dict(xaxis_title='X',
                                 yaxis_title='Y',
                                 zaxis_title='Z'),
                      margin=dict(r=20, b=10, l=10, t=10))
    fig.show()
