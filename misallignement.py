#Loads every image and its relative network, then checks if the two intervals are compatible.

import networkx as nx
import os
import numpy as np
from glob import glob
import nibabel as nib
import nibabel.freesurfer.mghformat as mghf # read mgz images
from skimage.morphology import skeletonize_3d
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt

def contains(extremes_outer, extremes_inner):
    # Check if max of outer is >= max of inner
    max_check = np.all(np.array(extremes_outer[0]) >= np.array(extremes_inner[0]))
    
    # Check if min of outer is <= min of inner
    min_check = np.all(np.array(extremes_outer[1]) <= np.array(extremes_inner[1]))
    
    return max_check and min_check


main_directory = "PR-graphomics/sequences/"

# List all subfolders inside the main directory
folders = [f for f in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, f))]

#variables in which the number 
skel_count, net_count = 0, 0


for folder in folders:
    print(f"Accessing folder: {folder}")
    
    # Construct the path to the current folder
    folder_path = os.path.join(main_directory, folder)
    
     #In this way I add to the address the folder inside the first one
    folder_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    graph_path = os.path.join(folder_path, 'sk-graph.pickle')
    image_path = os.path.join(folder_path, 'aparc.DKTatlas+aseg.deep.mgz')
    vol = mghf.load(image_path)

    # get the volume array
    volume = vol.get_fdata()
    # binarize the volume
    volume = volume != 0

    verts, faces, normals, values = marching_cubes(volume, 0)
    mesh_extremes = (np.max(verts, axis=0), np.min(verts, axis=0))


    skeleton = skeletonize_3d(volume)

    sx, sy, sz = np.where(skeleton) 

    skel_extremes = (np.max((sx, sy, sz), axis=1), np.min((sx, sy, sz), axis=1))

    graph = nx.read_gpickle(graph_path)

    nodes_coord = list(graph.nodes())
    #print(np.shape(nodes_coord))
    nodes_extremes = (np.max(nodes_coord, 0), np.min(nodes_coord, 0))
    print(nodes_extremes)
    print(mesh_extremes)
    print(skel_extremes)

    if contains(mesh_extremes, skel_extremes):
        print('The skeleton is inside the mesh!')
        skel_count +=1

    if contains(mesh_extremes, nodes_extremes):
        print('The graph is inside the mesh!')
        net_count += 1


    #break

print('The magic results are:')
print('percentage of fit for the skelly: ', str(skel_count/len(folders)))
print('percentage of fit for the grapphy: ', str(net_count/len(folders)))

