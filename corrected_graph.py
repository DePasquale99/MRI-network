#This script uses the data to recreate another pickle file containing a graph in which:
#coordinates are stored in the correct SDR
#links are weighted based on the distance between the nodes

import nibabel as nib
import numpy as np
import networkx as nx
from skimage.measure import marching_cubes
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import networkx as nx
import os
import numpy as np
from glob import glob
import nibabel as nib
import nibabel.freesurfer.mghformat as mghf # read mgz images
from skimage.morphology import skeletonize_3d
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors

def rescale_coordinates(node_positions, skeleton_coords):
    graph_min = node_positions.min(axis=0)
    graph_max = node_positions.max(axis=0)
    skeleton_min = skeleton_coords.min(axis=0)
    skeleton_max = skeleton_coords.max(axis=0)

    normalized_node_positions = (
        (node_positions - graph_min) / (graph_max - graph_min)
    ) * (skeleton_max - skeleton_min) + skeleton_min

    return normalized_node_positions



main_directory = "PR-graphomics/sequences/"

# List all subfolders inside the main directory
folders = [f for f in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, f))]
total = len(folders)
#folder = folders[10]
for  i, folder in enumerate(folders):
    print(f"Accessing folder: {folder}, completion percentage: {i/total}")
    folder_path = os.path.join(main_directory, folder)
        
    #In this way I add to the address the folder inside the first one
    folder_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    #aparc.DKTatlas+aseg.deep.mgz aparc.DKTatlas+aseg.deep.nii.gz
    image_path = os.path.join(folder_path, 'aparc.DKTatlas+aseg.deep.mgz')
    graph_path = os.path.join(folder_path, 'sk-graph.pickle')


    vol = mghf.load(image_path)
    volume = vol.get_fdata()
    volume = volume != 0
    verts, faces, normals, values = marching_cubes(volume, 0)

    ptx, pty, ptz = verts.T
    skeleton = skeletonize_3d(volume)

    # Load your original graph
    original_graph = nx.read_gpickle(graph_path)  # Replace with your graph file path



        # Extract skeleton coordinates
    sx, sy, sz = np.where(skeleton)
    skeleton_coords = np.vstack((sx, sy, sz)).T

        # Extract graph node positions
    node_positions = np.array(original_graph.nodes, dtype=float)

        # Rescale graph node positions to fit the skeleton's coordinates
    normalized_node_positions = rescale_coordinates(node_positions, skeleton_coords)

    # Create a new graph
    new_graph = nx.Graph()

    # Add nodes with x, y, z attributes
    for i, node in enumerate(original_graph.nodes()):
        # Parse the coordinates from the node label (assumes format '[x, y, z]')
        x, y, z = normalized_node_positions[i]
        new_graph.add_node(node, x=x, y=y, z=z)

    # Add edges with weights based on inverse distance
    for u, v in original_graph.edges():
        # Get the coordinates for each node
        x1, y1, z1 = new_graph.nodes[u]['x'], new_graph.nodes[u]['y'], new_graph.nodes[u]['z']
        x2, y2, z2 = new_graph.nodes[v]['x'], new_graph.nodes[v]['y'], new_graph.nodes[v]['z']
        
        # Calculate Euclidean distance
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        
        # Avoid division by zero; set a very high weight if nodes are the same
        weight = 1 / distance if distance > 0 else float('inf')
        
        # Add edge with the calculated weight
        new_graph.add_edge(u, v, weight=weight)

    # Save or use the new_graph
    nx.write_gpickle(new_graph, os.path.join(folder_path, 'sk-weighted_graph.pickle'))
