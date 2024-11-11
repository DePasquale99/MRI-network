
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
from scipy.spatial import cKDTree




##FUNCTIONS NECESSARY (I WOULD LIKE TO TAKE MY LIFE FOR DOING THAT)



def assign_voxels_to_nearest_nodes(node_coords, binary_volume, communities,nodes_names):
    """
    Assign each voxel in a binary MRI volume to the community of the nearest node.

    Parameters:
    - node_coords: A list or array of 3D coordinates for each node in the network (shape: [num_nodes, 3]).
    - binary_volume: A 3D NumPy array representing the binary MRI volume (1 for voxels to include, 0 otherwise).
    - communities: A list of lists, where each sublist contains the indices of nodes belonging to a specific community.

    Returns:
    - voxel_node_map: A 3D array where each voxel contains the index of the nearest node.
    - voxel_community_map: A 3D array where each voxel contains the community index of the nearest node.
    """
    # Convert the list of lists `communities` to a dictionary mapping each node to its community index
    node_to_community = {}
    for community_index, nodes in enumerate(communities):
        for node in nodes:
            node_to_community[node] = community_index + 1
    # Get the shape of the volume
    volume_shape = binary_volume.shape

    # Get the coordinates of all active (1-valued) voxels in the binary volume
    voxel_indices = np.argwhere(binary_volume > 0)  # N x 3 array of voxel coordinates

    # Build a KD-tree for the node coordinates to enable efficient nearest-neighbor search
    tree = cKDTree(node_coords)

    # Find the index of the closest node for each voxel coordinate
    _, nearest_node_indices = tree.query(voxel_indices)

    # Create empty maps to store the nearest node and community indices for each voxel
    voxel_node_map = np.zeros(volume_shape, dtype=int)
    voxel_community_map = np.zeros(volume_shape, dtype=int)

    # Populate the voxel_node_map and voxel_community_map
    for voxel, node_index in zip(voxel_indices, nearest_node_indices):
        voxel_node_map[tuple(voxel)] = node_index
        node = nodes_names[node_index]
        voxel_community_map[tuple(voxel)] = node_to_community.get(node, -1)  # -1 if node not in any community

    return voxel_node_map, voxel_community_map



def louvain_with_target_communities(graph, target_communities, initial_resolution=5, max_iter=100, tolerance=1):
    """
    Run Louvain community detection repeatedly, adjusting the resolution and handling oscillations 
    until the target number of communities is reached.

    Parameters:
    - graph: NetworkX graph on which to perform community detection.
    - target_communities: The desired number of communities.
    - initial_resolution: Starting value for the resolution parameter (default is 1.0).
    - max_iter: Maximum number of iterations to try to reach the target (default is 100).
    - tolerance: Initial tolerance for adjusting the resolution (default is 0.1).

    Returns:
    - A tuple of (partition, num_communities, final_resolution), where:
        - partition is a dictionary mapping each node to its community.
        - num_communities is the number of communities in the final partition.
        - final_resolution is the final value of the resolution parameter used.
    """
    resolution = initial_resolution
    last_num_communities = None
    oscillation_count = 0  # Counter to track oscillations

    for i in range(max_iter):
        # Perform Louvain community detection with the current resolution
        partition = nx.community.louvain_communities(graph, resolution=resolution)
        num_communities = len(list(partition))
        #print(f"Cycle {i +1}, resolution = {resolution}, number of communities = {num_communities}")

        # Check if the target number of communities has been achieved
        if num_communities == target_communities:
            #print(f"Target achieved in {i+1} iterations with resolution={resolution}.")
            return partition, num_communities, resolution

        # Check for oscillation by comparing with the previous number of communities
        if last_num_communities is not None and (
            (num_communities > target_communities and last_num_communities < target_communities) or
            (num_communities < target_communities and last_num_communities > target_communities)
        ):
            oscillation_count += 1
            if oscillation_count >= 2:
                # Halve the tolerance if we detect oscillation
                tolerance /= 2
                oscillation_count = 0  # Reset oscillation count
                #print(f"Oscillation detected. Reducing tolerance to {tolerance}")

        # Update the resolution based on whether we need more or fewer communities
        if num_communities > target_communities:
            resolution -= tolerance  # Increase resolution to reduce the number of communities
        else:
            resolution += tolerance  # Decrease resolution to increase the number of communities

        # Update last number of communities for the next iteration
        last_num_communities = num_communities

    print("Warning: Target number of communities not achieved within max_iter.")
    return partition, num_communities, resolution



def plot_3d_matrix_general(matrix):
    """
    Plots points in 3D space for each unique entry in a 3D matrix,
    assigning a different color to each unique value, except for zero.

    Parameters:
    - matrix: 3D NumPy array with multiple unique values.
    """
    # Identify unique values in the matrix, excluding zero
    unique_values = np.unique(matrix)
    unique_values = unique_values[unique_values != 0]  # Exclude zero
    print(len(unique_values))
    
    # Set up a colormap
    colors = plt.cm.get_cmap("tab20", len(unique_values))  # Use a colormap with enough colors
    
    # Create a 3D plot
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each unique value in the matrix with a different color
    for i, value in enumerate(unique_values):
        # Get the coordinates of all entries with the current value
        voxel_indices = np.argwhere(matrix == value)
        
        # Separate the coordinates into x, y, and z components
        x, y, z = voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
        
        # Plot the points with the color corresponding to the current value
        ax.scatter(x, y, z, c=[colors(i)],   alpha =.1, s= 1)#label=f'Value {value}',marker='o',

    # Set labels, title, and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of brain areas')
    #ax.legend()
    plt.show()






main_directory = "PR-graphomics/sequences/"
save_directory = ""
# List all subfolders inside the main directory
folders = [f for f in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, f))]






for folder in folders:

    folder_path = os.path.join(main_directory, folder)
    #In this way I add to the address the folder inside the first one
    folder_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    #aparc.DKTatlas+aseg.deep.mgz aparc.DKTatlas+aseg.deep.nii.gz
    image_path = os.path.join(folder_path, 'aparc.DKTatlas+aseg.deep.mgz')
    graph_path = os.path.join(folder_path, 'sk-graph.pickle')
    weighted_graph_path =  os.path.join(folder_path, 'sk-weighted_graph.pickle')
    vol = mghf.load(image_path)


    volume = vol.get_fdata()
    bin_volume = volume != 0
    verts, faces, normals, values = marching_cubes(volume, 0)

    ptx, pty, ptz = verts.T
    skeleton = skeletonize_3d(volume != 0)
    graph = nx.read_gpickle(weighted_graph_path)


    comm_number = len(np.unique(volume))
    communities, number, res = louvain_with_target_communities(graph, comm_number, max_iter= 1000, tolerance = 1) 

    #PRINTING  nodes communities and skeleton of the image
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    # Plot the skeleton points
    sx, sy, sz = np.where(skeleton)
    ax.scatter(sx, sy, sz, color='r', marker='o', s=3, alpha=0.05)

    # Plot the nodes, using their x, y, z attributes for positions
    colors = list(mcolors.CSS4_COLORS.keys())
    np.random.shuffle(colors)

    for com, color in zip(communities, colors):
        # Extract x, y, z coordinates for the nodes in each community
        com_coords = [(graph.nodes[node]['x'], graph.nodes[node]['y'], graph.nodes[node]['z']) for node in com]
        ax.scatter(*zip(*com_coords), color=color, marker='o', s=30, alpha=0.5)

    # Plot edges based on node coordinates
    for ex, ey in graph.edges():
        x_coords = [graph.nodes[ex]['x'], graph.nodes[ey]['x']]
        y_coords = [graph.nodes[ex]['y'], graph.nodes[ey]['y']]
        z_coords = [graph.nodes[ex]['z'], graph.nodes[ey]['z']]
        ax.plot(x_coords, y_coords, z_coords, color='k', linewidth=2, alpha=0.25)

    # Set plot labels and aspect
    ax.set_box_aspect((np.ptp(sx), np.ptp(sy), np.ptp(sz)))
    ax.set_xlabel('x', fontsize=24)
    ax.set_ylabel('y', fontsize=24)
    ax.set_zlabel('z', fontsize=24)
    _ = ax.set_title('3D Brain Communities', fontsize=24)

    plt.show()

    #Obtain the nearest community and the nearest nodes for every voxel

    nodes_names = list(graph.nodes())
    nodes_coords = [(graph.nodes[node]['x'], graph.nodes[node]['y'], graph.nodes[node]['z']) for node in nodes_names]

    node_map, community_map = assign_voxels_to_nearest_nodes(nodes_coords,  volume != 0, communities, nodes_names)


    plot_3d_matrix_general(volume)

    plot_3d_matrix_general(community_map)