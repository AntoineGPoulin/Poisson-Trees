import numpy as np
from scipy.spatial import Voronoi, cKDTree
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
import networkx as nx
import time

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