import numpy as np
from scipy.spatial import Voronoi, cKDTree
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
import networkx as nx
import time
import dipvt_sec

# Example usage# Complexity is approximated at O(h * g^2 * log g), where h is height, g is gridsize.
rate = 0.5
dimensions = 2
region_sizes = (200, 200)
height = 5

# Seeding
seeds = None
# Start the timer
start_time = time.time()

def get_color_map(point_layers, G):
    color_map = {}
    
    # Use the tab20 color map to get distinct colors
    color_idx = 0
    unique_colors = cm.get_cmap('tab20', 20)

    for node in G:
        if node in point_layers[0]:
            component = frozenset(nx.node_connected_component(G, node))
            if component not in color_map:
                color_map[component] = unique_colors(color_idx % 20)
                color_idx += 1
    return color_map

def plot_colored_voronoi_first_layer(point_layers, color_map, G, ax):
    first_layer_points = point_layers[0]
    vor = compute_voronoi(first_layer_points)

    patches = []
    colors = []
    for region, index in zip(vor.regions, vor.point_region):
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            patches.append(Polygon(polygon))

            point = first_layer_points[index % len(first_layer_points)]
            for component, color in color_map.items():
                if tuple(point) in component:
                    colors.append(color)
                    break

    p = PatchCollection(patches, alpha=0.4)
    p.set_color(colors)
    ax.add_collection(p)
    ax.set_xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    ax.set_ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)


# Generate the multilayer Poisson point process
print("Generating multilayer Poisson point process...")
point_layers = generate_multilayer_poisson(rate, dimensions, region_sizes, height, seeds=seeds)
print(f"Multilayer Poisson point process generated. Time elapsed: {time.time() - start_time:.2f} seconds")

# Construct the graph
print("Constructing the graph...")
G = construct_graph(point_layers)
print(f"Graph constructed. Time elapsed: {time.time() - start_time:.2f} seconds")

# Get the color map for connected components
print("Getting the color map...")
color_map = get_color_map(point_layers, G)
print(f"Color map obtained. Time elapsed: {time.time() - start_time:.2f} seconds")

# Plot the colored Voronoi diagram
print("Plotting the colored Voronoi diagram...")
fig, ax = plt.subplots()
plot_colored_voronoi_first_layer(point_layers, color_map, G, ax)
plt.show()
print(f"Colored Voronoi diagram plotted. Total time elapsed: {time.time() - start_time:.2f} seconds")
