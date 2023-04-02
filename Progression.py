import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

def run_code():
    # For timing purposes
    start_time = time.time()
    def update(string, last_time):
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_rel = end_time - last_time
        print(string + " Total Time elapsed: {:.2f} seconds".format(elapsed_time) +  " Time elapsed since last update: {:.2f} seconds".format(elapsed_rel) )
        return end_time

    # Set the size of the grid
    grid_size = 500
    probability = 0.5

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

    def apply_transformation(point, pointer_grid, times):
        i, j = point
        for _ in range(times):
            if pointer_grid[i][j] is not None:
                i, j = pointer_grid[i][j]
            else:
                break
        return i, j

    def ith_level_root(point, dist_grid, pointer_grid, level):
        i, j = point
        distance = dist_grid[i][j]
        if level <= distance:
            return point
        return apply_transformation(point, pointer_grid, level - distance)

    # Calculate the maximal distance to the nearest zero in the grid
    max_distance = max(max(row) for row in dist_grid)

    # Compute the ith-level connected components for each level
    # Initialize the connected components for level 0 with empty lists
    ith_level_components = defaultdict(dict)

    # Calculate the connected components for each level from 1 to max_distance
    for level in range(1, max_distance + 1):
        # Initialize the connected components for the current level
        connected_components = defaultdict(set)

        # Check for points that become leaves at the current level
        for i in range(total_rows):
            for j in range(total_cols):
                if dist_grid[i][j] == level:
                    root_point = ith_level_root((i, j), dist_grid, pointer_grid, level)
                    connected_components[root_point].add((i, j))

        # Merge the connected components from the previous level
        for prev_root, prev_component in ith_level_components[level - 1].items():
            new_root = apply_transformation(prev_root, pointer_grid, 1)
            connected_components[new_root] |= prev_component

        # Save the connected components for the current level
        ith_level_components[level] = connected_components

    time_comp = update("Components found", time_point)

    # Calculate the starting level for the last 4 levels
    start_level = max(1, max_distance - 3)

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    axes = axes.flatten()

    for idx, level in enumerate(range(start_level, max_distance + 1)):
        # Create the color grid for the current level
        color_grid = np.zeros((total_rows, total_cols, 3))
        connected_components = ith_level_components[level]
        
        # Generate random colors for the current level
        random_colors = np.random.rand(len(connected_components), 3)

        for i in range(total_rows):
            for j in range(total_cols):
                root_ij = ith_level_root((i, j), dist_grid, pointer_grid, level)
                root_idx = -1
                for idx_color, component in enumerate(connected_components.values()):
                    if root_ij in component:
                        root_idx = idx_color
                        break
                color_grid[i, j] = random_colors[root_idx]

        # Plot the colored grid for the current level on the corresponding axis
        ax = axes[idx]
        ax.imshow(color_grid)
        ax.set_title(f'Colored grid for level {level}')

    update("Graph done", time_comp)

    return fig
