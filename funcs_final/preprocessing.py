import os
import glob
import sys
import random
import shutil
import tqdm

# data processing
import tensorflow as tf
import numpy as np
import tetgen
import SimpleITK as sitk
from skimage import measure
from numpy import random
from math import factorial
import numpy as np
from scipy.spatial import distance
from scipy import ndimage
# ploting
import seaborn as sns
import matplotlib
import matplotlib.pylab as plt
import pyvista as polyv
import networkx as nx
import itertools
from scipy.ndimage import measurements


import scipy
from scipy.optimize import fmin_cg
from scipy.stats import chi2
from scipy.stats import wasserstein_distance
import skimage
from tensorflow.keras import layers
import pyvista as polyv
# stats
import math
import smm     ## student mixture model
import vtk

#sys.path.append('/Users/paulbd/Documents/Projects/pynd-lib')
#sys.path.append('./utils/spatial_transformer_network/')
#from spatial_transformer_network import *

from tempfile import mkstemp,mkdtemp
import nibabel as nib


def find_idx_Pn(Pn, tuple_coord):
    return [i for i, Pn_i in enumerate(Pn) if Pn_i == tuple_coord][0]



def build_mesh_from_3Dmask(mask, STEP_SIZE, SPACING):
    '''update_outlier
    build a mesh from a 3D mask
    mask :
    STEP_SIZE :
    SPACING :
    surface : if true only surfacic mesh


    '''
    labels, n_labels = measurements.label(mask)

    ls_cells = []
    ls_points = []
    #for i in range(1, 3):
    print(n_labels)
    for i in range(1, n_labels + 1):
        label = (labels == i) * 1
        #print('IDX {}, n vox{}'.format(i, np.sum(label)))
        try:
            verts, faces, normals, values = measure.marching_cubes_lewiner(label, step_size=STEP_SIZE, spacing=SPACING,
                                                                           allow_degenerate=True)
            verts = np.round(verts)
            faces_1_adapted = np.ones([faces.shape[0], faces.shape[1] + 1]) * 3
            faces_1_adapted[:, 1:] = faces
            surf = polyv.PolyData(verts, faces_1_adapted)
            tet = tetgen.TetGen(surf)
            tet.make_manifold()
            tet.tetrahedralize(order=1, mindihedral=10, steinerleft=-1, minratio=1.01)
            grid = tet.grid
            #print(grid.number_of_cells)
            if i > 1:
                PREVIOUS_NUMBER_OF_VERTICES = points_concat.reshape([-1, 3]).shape[0]
                ls_points.append(grid.points)
                cells = grid.cells.reshape([-1, 5])
                cells[:, 1:] += PREVIOUS_NUMBER_OF_VERTICES
                cells = cells.reshape([-1])
                ls_cells.append(cells)
                cells_concat = np.concatenate(ls_cells, axis=0)
                points_concat = np.concatenate(ls_points, axis=0)
                n_cells = cells_concat.reshape([-1, 5]).shape[0]
            else:
                ls_cells.append(grid.cells)
                ls_points.append(grid.points)
                # fig,ax=plt.subplots()
                # ax.imshow(label[200,:,:])
                cells_concat = np.concatenate(ls_cells, axis=0)
                points_concat = np.concatenate(ls_points, axis=0)
                n_cells = cells_concat.reshape([-1, 5]).shape[0]

        except:
            print('erreur {}'.format(i))

    cell_type = np.array([vtk.VTK_TETRA] * n_cells)
    offset = np.arange(0, n_cells * 5, step=5)
    grid = polyv.UnstructuredGrid(offset, cells_concat, cell_type, points_concat)
    return grid


def generate_graph_from_mesh(grid):
    G = nx.Graph()
    cells = grid.cells.reshape(-1, 5)[:, :]
    points = np.array(grid.points)
    coords_list = list(points)
    points_edges = points[cells[:, 1:]]

    Pn = [(Pn_i[0], Pn_i[1], Pn_i[2]) for Pn_i in points]

    for num_tetraedra in range(cells.shape[0]):
        points_tetraedra = points_edges[num_tetraedra]
        ls_i = []
        for point_tetraedra in points_tetraedra:
            i = find_idx_Pn(Pn, tuple(point_tetraedra))
            ls_i.append(i)
            point_tetraedra = np.pad(point_tetraedra, (1, 1), 'constant', constant_values=0)
            G.add_node(i, Pn=point_tetraedra, Qn=point_tetraedra)
        for comb in [*itertools.combinations(ls_i, 2)]:
            # Pn[comb[0]]-Pn[comb[1]]
            G.add_edge(*comb, weight=1)
    return G


def extract_patchexact_from_coord_list(np_array, c_list, keys, patch_size, vox_size):
    '''
    c_list : a list of coordinates
    patch_size : the size of the patch
    '''
    ls_patch = []
    idx_in_batch_kept = []
    dict_patches = {}
    for i, coords_i in enumerate(c_list):
        key_batch = keys[i]
        coords_vox_space = (coords_i + 1e-9) / vox_size
        coords_min = np.int16(np.floor(coords_vox_space))
        coords_max = np.int16(np.ceil(coords_vox_space))

        patch = np_array[:, coords_min[1] - (patch_size // 2):coords_max[1] + (patch_size // 2),
                coords_min[2] - (patch_size // 2):coords_max[2] + (patch_size // 2),
                coords_min[3] - (patch_size // 2):coords_max[3] + (patch_size // 2), :]
        if (patch.shape == (1, patch_size, patch_size, patch_size, 1)) & (np.sum(patch) != 0):
            ls_patch.append(patch)
            idx_in_batch_kept.append(key_batch)
            dict_patches[key_batch] = patch
        else:
            print('ERROR PATCH EXTRACTION' + str(i))
            print(' SHAPE {}, SUM {}'.format(patch.shape, np.sum(patch)))

    # if len(idx_in_batch_kept) == 0:
    #    idx_in_batch_kept = None
    #    ls_patch = None

    return idx_in_batch_kept, dict_patches


def get_Pn_Qn_batch(Graph, list_keys_batches):
    '''
    function to get a list of Pn and Qn coordinates based on
    '''
    Pns = nx.get_node_attributes(Graph, 'Pn')
    Qns = nx.get_node_attributes(Graph, 'Qn')
    Pn_batch = [Pns[key] for key in list_keys_batches]
    Qn_batch = [Qns[key] for key in list_keys_batches]
    return Pn_batch, Qn_batch


def pad_patches_fixed(patches_fixed_to_pad, PATCH_SIZE_Qn=16, PATCH_SIZE_Pn=8):
    '''
    pad a patch to match Qn SIZE
    '''
    patches_padded = []
    for i in range(len(patches_fixed_to_pad)):
        patch = patches_fixed_to_pad[i]
        padding = ((0, 0), ((PATCH_SIZE_Qn - PATCH_SIZE_Pn) // 2, (PATCH_SIZE_Qn - PATCH_SIZE_Pn) // 2),
                   ((PATCH_SIZE_Qn - PATCH_SIZE_Pn) // 2,
                    (PATCH_SIZE_Qn - PATCH_SIZE_Pn) // 2),
                   ((PATCH_SIZE_Qn - PATCH_SIZE_Pn) // 2, (PATCH_SIZE_Qn - PATCH_SIZE_Pn) // 2), (0, 0))
        patches_padded.append(np.pad(patch, padding, 'constant', constant_values=-3000))

    return patches_padded


def mean_and_std_neighbors(key_batch, graph, t_mean, t_std, A_mean, A_std):
    idx_neighbors = [*graph.neighbors(key_batch)]
    tn_neighbors = np.array(
        [nx.get_node_attributes(graph, 'accumulated_translation')[idx_n] for idx_n in idx_neighbors])

    MSE_i = np.stack(
        [nx.get_node_attributes(graph, 'MSE')[idx_n] for idx_n in idx_neighbors])
    MSE_i[MSE_i > 1] = 1
    # t_mean_neigbors = tf.cast(tf.reduce_mean(tn_neighbors, axis=0), tf.float32)
    t_mean_neigbors = np.float32(np.array(np.matrix(-np.log(MSE_i)) * tn_neighbors / np.sum(-np.log(MSE_i))))
    tn_std = np.float32(np.std(tn_neighbors, axis=0))
    if any(np.isnan(tn_std)):
        tn_std = np.float32(t_std)
    if any(np.isnan(t_mean_neigbors[0])):
        t_mean_neigbors = np.float32(t_mean)

    '''
    An_neigbors = np.array([
        nx.get_node_attributes(graph, 'NablaQ')[idx_n]
        for idx_n in idx_neighbors])

    An_neigbors = np.reshape(An_neigbors, [An_neigbors.shape[0], -1])
    # print(An_neigbors)
    MSE_i = np.reshape(MSE_i, [MSE_i.shape[0], -1])

    An_mean = np.squeeze(np.array(np.float32(np.matrix(-np.log(MSE_i).T) * An_neigbors / np.sum(-np.log(MSE_i)))))

    # An_std = tf.cast(tf.reshape(np.std(An_neigbors, axis=1), [1, An_mean.shape[0], -1]), tf.float32)
    An_std = np.float32(np.std(An_neigbors))
    if np.isnan(An_mean):
        An_mean = np.float32(A_mean)
    if np.isnan(An_std):
        An_std = np.float32(A_std)
    '''
    return t_mean_neigbors, tn_std

