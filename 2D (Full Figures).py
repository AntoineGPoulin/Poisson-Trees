import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# Set the size of the grid
grid_size = 200
probability = 0.9

# For timing purposes
start_time = time.time()
def update(string, last_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_rel = end_time - last_time
    print(string + " Total Time elapsed: {:.2f} seconds".format(elapsed_time) +  " Time elapsed since last update: {:.2f} seconds".format(elapsed_rel) )
    return end_time

# Define the dimensions of the grid
total_rows = 2 * grid_size + 1
total_cols = 2 * grid_size + 1

# Initialize a grid of tuples with indices and perform coin-flips
grid = [[(1 if random.random() < probability else 0) for j in range(total_cols)] for i in range(total_rows)]

# Initialize a grid to hold the distance to the nearest zero
dist_grid = [[None for _ in range(total_cols)] for _ in range(total_rows)]
pointer_grid = [[None for _ in range(total_cols)] for _ in range(total_rows)]

# Updates for initialization
time_init = update("Initialization done", start_time)

# Calculate the distance to the nearest zero for each point in the grid
for i in range(total_rows):
    for j in range(total_cols):
        distance = 0
        found_zero = False
        while not found_zero:
            # Check all points in a diamond pattern around the current point
            diamond_points = [(dx, dy) for k in range(distance + 1) for dx, dy in [(i + k, j + distance - k),
                               (i - k, j + distance - k),
                               (i + k, j - distance + k),
                               (i - k, j - distance + k)] if 0 <= dx < total_rows and 0 <= dy < total_cols]
            # If there is a zero in the diamond, stop searching
            found_zero = any(grid[x][y] == 0 for x, y in diamond_points)
            if not found_zero:
                distance += 1
            # Break the loop if the distance is greater than the maximum possible distance in the grid
            if distance > max(total_rows, total_cols):
                break
        dist_grid[i][j] = distance

# Updates fpr distance
time_dist = update("Distance calculations done", time_init)


# Initialize a grid to hold next point
pointer_grid = [[None for _ in range(total_cols)] for _ in range(total_rows)]

# Calculate the pointers to the nearest point of height one more for each point in the grid
for i in range(total_rows):
    for j in range(total_cols):
        distance = 0
        current_height = dist_grid[i][j]
        found_point = False
        while not found_point:
            # Check all points in a diamond pattern around the current point
            diamond_points = [(dx, dy) for k in range(distance + 1) for dx, dy in [(i + k, j + distance - k),
                               (i - k, j + distance - k),
                               (i + k, j - distance + k),
                               (i - k, j - distance + k)] if 0 <= dx < total_rows and 0 <= dy < total_cols]
            # Find a point with height one more
            next_point = next(((x, y) for x, y in diamond_points if dist_grid[x][y] == current_height + 1), None)
            if next_point is not None:
                pointer_grid[i][j] = next_point
                found_point = True
            distance += 1
            if distance > total_cols:
                found_point = True

# Updates for pointers
time_point = update("Pointer calculations done", time_dist)

# Find the root of the tree.
def find_root(node, pointer_grid):
    i, j = node
    while pointer_grid[i][j] is not None:
        i, j = pointer_grid[i][j]
    return (i, j)

# Find connected components
max_height_points = {}
for i in range(total_rows):
    for j in range(total_cols):
        if grid[i][j] == 1:
            max_height_points[(i, j)] = find_root((i, j), pointer_grid)

connected_components = defaultdict(list)
for point, max_height_point in max_height_points.items():
    connected_components[max_height_point].append(point)

connected_components = list(connected_components.values())
random_colors = np.random.rand(len(connected_components), 3)
color_grid = np.zeros((total_rows, total_cols, 3))

for i in range(total_rows):
    for j in range(total_cols):
        root_ij = find_root((i, j), pointer_grid)
        root_idx = -1
        for idx, component in enumerate(connected_components):
            if root_ij in component:
                root_idx = idx
                break
        color_grid[i, j] = random_colors[root_idx]

# Print time elapsed for computation
time_comp = update("Computations done", time_point)

# Create a 2x2 grid of subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))


# Plot the i.i.d coin-flips
im1 = ax1.imshow([[cell for cell in row] for row in grid], cmap='Greys', vmin=0, vmax=1)
ax1.set_title('i.i.d coin-flips')
fig.colorbar(im1, ax=ax1)

# Plot the distances to nearest 0
im2 = ax2.imshow(dist_grid, cmap='jet')
ax2.set_title('distances to nearest 0')
fig.colorbar(im2, ax=ax2)

# Set limits and aspect ratio for ax3
ax3.set_xlim(0, total_cols)
ax3.set_ylim(total_rows, 0)
ax3.set_aspect('equal')

# Draw connections between centroids on ax3
for i in range(total_rows):
    for j in range(total_cols):
        if pointer_grid[i][j] is not None:
            k, l = pointer_grid[i][j]
            centroid_ij = (j + 0.5, i + 0.5)
            centroid_kl = (l + 0.5, k + 0.5)

            # Find the root for both points
            root_ij = find_root((i, j), pointer_grid)
            root_kl = find_root((k, l), pointer_grid)

            if root_ij == root_kl:
                # Find the index of the root in the connected_components
                root_idx = -1
                for idx, component in enumerate(connected_components):
                    if root_ij in component:
                        root_idx = idx
                        break

                # Use the index to generate the line color
                line_color = random_colors[root_idx]
                ax3.plot(*zip(centroid_ij, centroid_kl), color=line_color, linewidth=1)

ax3.set_title('centroid connections')

# Make the bottom-right subplot (ax4) blank
ax4.imshow(color_grid)
ax4.set_title('colored grid')

# Give time to plot everything
update("Graph done", time_comp)

# Show the plot
plt.show()
