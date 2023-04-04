import numpy as np
from scipy.spatial import Voronoi, cKDTree
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib
import networkx as nx
import time

# Parameters
rate = 0.3
dimensions = 2
region_sizes = (100, 100)
height = 5

def poisson_point_process(rate, dimensions, region_sizes, seed=None):
    np.random.seed(seed)
    volume = np.prod(region_sizes)
    num_points = np.random.poisson(rate * volume)
    points = np.random.rand(num_points, dimensions) * region_sizes
    return points

def compute_voronoi(points):
    vor = Voronoi(points)
    return vor

def construct_graph(point_layers):
    G = nx.Graph()

    for i, layer in enumerate(point_layers[:-1]):
        points_above = np.vstack(point_layers[i+1:])
        tree = cKDTree(points_above)

        for point in layer:
            dist, idx = tree.query(point)
            nearest_neighbor = tuple(points_above[idx])
            G.add_edge(tuple(point), nearest_neighbor)

    return G

def get_component_colors(point_layers, G):
    component_colors = {}
    color_idx = 0
    unique_colors = matplotlib.colors.ListedColormap(matplotlib.colors.TABLEAU_COLORS, N=20)

    for node in G:
        if node in point_layers[0]:
            component = frozenset(nx.node_connected_component(G, node))
            if component not in component_colors:
                component_colors[component] = unique_colors(color_idx % 20)
                color_idx += 1

    return component_colors

def plot_trees_and_voronoi_cells(point_layers, G, component_colors, ax):
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
        for edge in edges:
            p1, p2 = np.array(node), np.array(edge)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=2)

        if node in first_layer_points:
            region = vor.regions[vor.point_region[np.where(vor.points == node)[0][0]]]
            if not -1 in region and len(region) > 0:
                polygon = [vor.vertices[i] for i in region]
                ax.add_patch(Polygon(polygon, facecolor=color, alpha=0.6))


# Generate the multilayer Poisson point process
point_layers = [poisson_point_process(rate ** i, dimensions, region_sizes) for i in range(1, height + 1)]

# Construct the graph
G = construct_graph(point_layers)

# Get the component colors
component_colors = get_component_colors(point_layers, G)

# Plot the trees and Voronoi cells
fig, ax = plt.subplots()
plot_trees_and_voronoi_cells(point_layers, G, component_colors, ax)
ax.set_xlim(0, region_sizes[0])
ax.set_ylim(0, region_sizes[1])
plt.title('Poisson Trees with Colored Voronoi Cells')
plt.show()
