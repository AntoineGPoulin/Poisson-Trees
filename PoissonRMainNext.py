import numpy as np
from scipy.spatial import Voronoi, cKDTree
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib
import networkx as nx
import time

# Example usage# Complexity is approximated at O(h * g^2 * log g), where h is height, g is gridsize.
rate = 0.3
dimensions = 2
region_sizes = (100, 100)
height = 5

# Seeding
seeds = None
# Start the timer
start_time = time.time()

def poisson_point_process(rate, dimensions, region_sizes, seed=None):
    """
    Generate Poisson Point Process samples in arbitrary dimensions.
    
    Parameters:
    rate (float): Intensity of the Poisson point process.
    dimensions (int): Number of dimensions.
    region_sizes (tuple): Size of the region in each dimension.
    seed (int, optional): Seed for random number generation. Defaults to None.
    
    Returns:
    numpy.ndarray: Array of points generated from the Poisson Point Process.
    """
    np.random.seed(seed)
    volume = np.prod(region_sizes)
    num_points = np.random.poisson(rate * volume)
    points = np.random.rand(num_points, dimensions) * region_sizes
    return points

def compute_voronoi(points):
    """
    Compute Voronoi cells for the given points.
    
    Parameters:
    points (numpy.ndarray): Array of points.
    
    Returns:
    scipy.spatial.Voronoi: Voronoi tessellation.
    """
    vor = Voronoi(points)
    return vor

def generate_multilayer_poisson(rate, dimensions, region_sizes, height, seeds=None):
    if seeds is None:
        seeds = [None] * (height + 1)

    point_layers = [poisson_point_process(rate ** i, dimensions, region_sizes, seed=seeds[i]) for i in range(1, height + 1)]
    return point_layers

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

def plot_trees(point_layers, color_map, G, ax):
    for node in G:
        if node in point_layers[0]:
            component = frozenset(nx.node_connected_component(G, node))
            color = color_map[component]
            for edge in nx.dfs_edges(G, source=node):
                p1, p2 = np.array(edge[0]), np.array(edge[1])
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=2)

def get_color_map(point_layers, G):
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

def plot_colored_voronoi_first_layer(point_layers, color_map, ax):
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
print("Getting the color map...")
color_map = get_color_map(point_layers, G)
print(f"Color map obtained. Time elapsed: {time.time() - start_time:.2f} seconds")

# Plot the colored Voronoi diagram
print("Plotting the colored Voronoi diagram...")
fig, ax = plt.subplots()

# Plot the colored Voronoi cells
plot_colored_voronoi_first_layer(point_layers, color_map, ax)

# Plot the trees with the same colors as their associated components
plot_trees(point_layers, color_map, G, ax)

ax.set_xlim(0, region_sizes[0])
ax.set_ylim(0, region_sizes[1])
plt.title('Picking next layer')
plt.show()
print(f"Colored Voronoi diagram plotted. Total time elapsed: {time.time() - start_time:.2f} seconds")
