import numpy as np
from scipy.spatial import Voronoi, cKDTree
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib
import networkx as nx
import time

# Parameters
rate = 0.3
dimensions = 2
region_sizes = (100, 100)
height = 5
seeds = None
show_tree_edges = True  # Set to False if you don't want to display tree edges

# Start the timer
start_time = time.time()


# Generate Poisson Point Process samples in arbitrary dimensions
def poisson_point_process(rate, dimensions, region_sizes, seed=None):
    np.random.seed(seed)
    volume = np.prod(region_sizes)
    num_points = np.random.poisson(rate * volume)
    points = np.random.rand(num_points, dimensions) * region_sizes
    return points


# Compute Voronoi cells for the given points
def compute_voronoi(points):
    vor = Voronoi(points)
    return vor


# Generate the multilayer Poisson point process
def generate_multilayer_poisson(rate, dimensions, region_sizes, height, seeds=None):
    if seeds is None:
        seeds = [None] * (height + 1)

    point_layers = [poisson_point_process(rate ** i, dimensions, region_sizes, seed=seeds[i]) for i in range(1, height + 1)]
    return point_layers


# Construct the graph
def construct_graph(point_layers):
    G = nx.Graph()

    # Flatten all points from layers above the first one
    points_above_first_layer = np.vstack(point_layers[1:])

    # Build a single cKDTree for all points above the first layer
    tree = cKDTree(points_above_first_layer)

    for point in point_layers[0]:
        # Find the nearest neighbor among all points above the first layer
        dist, idx = tree.query(point)

        # Get the actual nearest neighbor point
        nearest_neighbor = tuple(points_above_first_layer[idx])

        G.add_edge(tuple(point), nearest_neighbor)

    return G


# Get the color map for connected components
def get_component_colors(point_layers, G):
    color_map = {}

    # Use the tab20 color map to get distinct colors
    color_idx = 0
    unique_colors = matplotlib.colors.ListedColormap(matplotlib.colors.TABLEAU_COLORS, N=20)

    for node in G:
        if node in point_layers[0]:
            component = frozenset(nx.node_connected_component(G, node))
            if component not in color_map:
                color_map[component] = unique_colors(color_idx % 20)
                color_idx += 1
    return color_map

# Plot the trees and Voronoi cells with the option to show or hide tree edges
def plot_trees_and_voronoi_cells(point_layers, G, component_colors, ax, show_tree_edges=True):
    first_layer_points = point_layers[0]
    vor = compute_voronoi(first_layer_points)

    for node, edges in G.adj.items():
        if node in first_layer_points:
            component = frozenset(nx.node_connected_component(G, node))
            color = component_colors[component]
        else:
            # Find the connected component color for higher layer nodes
            for edge in edges:
                if edge in first_layer_points:
                    component = frozenset(nx.node_connected_component(G, edge))
                    color = component_colors[component]
                    break

        # Plot edges with the color of the connected component
        if show_tree_edges:
            for edge in edges:
                p1, p2 = np.array(node), np.array(edge)
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=2)

    # Plot the colored Voronoi cells
    patches = []
    colors = []

    for region, index in zip(vor.regions, vor.point_region):
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            patches.append(Polygon(polygon))

            point = tuple(first_layer_points[index % len(first_layer_points)])
            component = frozenset(nx.node_connected_component(G, point))
            colors.append(component_colors[component])

    p = PatchCollection(patches, alpha=0.6)
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
print("Getting the component colors...")
component_colors = get_component_colors(point_layers, G)
print(f"Component colors obtained. Time elapsed: {time.time() - start_time:.2f} seconds")

# Plot the colored Voronoi diagram
print("Plotting the colored Voronoi diagram...")
fig, ax = plt.subplots()

# Plot the trees and Voronoi cells with the option to show or hide tree edges
plot_trees_and_voronoi_cells(point_layers, G, component_colors, ax, show_tree_edges=show_tree_edges)

ax.set_xlim(0, region_sizes[0])
ax.set_ylim(0, region_sizes[1])
plt.title('Picking next layer')
plt.show()
print(f"Colored Voronoi diagram plotted. Total time elapsed: {time.time() - start_time:.2f} seconds")
