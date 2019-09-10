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
from scipy.spatial import distance
from scipy import ndimage
# ploting
import pyvista as polyv
import networkx as nx
import itertools
import scipy
from scipy.stats import chi2
import funcs_final
from funcs_final.tetraedra_tools import *
from funcs_final.preprocessing import *

# stats
import math
import funcs_final

from tempfile import mkstemp,mkdtemp
import nibabel as nib

def calc_D_mats(grid):
    ''''
    '''
    cells = grid.cells.reshape(-1, 5)[:, 1:]
    ls_Hinv=[]
    for i in range(cells.shape[0]):

        cell = cells[i]
        Q = grid.points[cell]
        v=simplex_volume(vertices=Q)
        ls_Hinv.append(ComputeGradient(Q).T/(6*v))
    return np.stack(ls_Hinv,axis=0)


def compute_D(grid_test):
    '''
    Compute the Derivative of the tetrahedron, aka the shape vectors
    :param idx_points: the idx of the points of the tetrahedron
    :param grid: the tetrahedrical mesh as a pvysta array
    :return:
    '''
    cells = grid_test.cells.reshape(-1, 5)[:, 1:]
    ls_D = []
    for i,cell in tqdm.tqdm(enumerate(cells)):
        P = grid_test.points[cell]
        _, Hinv= calc_shape_vectors(P)
        ls_D.append(Hinv)
    return np.stack(ls_D,axis=0)

def calc_bn(vertex_num, grid):
    tetrahedrons = grid.cells.reshape(-1, 5)[:, 1:]
    Pns = np.array(grid.points)

    # RETURN THE NUMBER OF THE CELLS
    # find the num of tetra
    # if idx_points in cell

    v_i = [num_cell
           for num_cell, tetra in enumerate(tetrahedrons)
           if any([idx_cell in [vertex_num] for idx_cell in [*tetra]])]
    c_n = len(v_i)

    # print('N TETRA WITH {} : {}'.format(vertex_num,c_n))

    ls_Dmat_index_s_i = []
    # ls_Bn_i = []
    list_idx_vertex = np.unique(tetrahedrons[v_i])
    list_idx_vertex_full = tetrahedrons[v_i]

    Bn_mat = np.zeros([len(list_idx_vertex), 3])

    for num_tetra, v in enumerate(v_i):
        ## LOOP WITHIN THETRAHEDRONS
        # print(num_tetra)
        index_vertex_tetra_v = tetrahedrons[v]
        # print(index_vertex_tetra_v)
        P = Pns[index_vertex_tetra_v]
        v = simplex_volume(vertices=P)
        D_mat = ComputeGradient(P).T / (6 * v)

        for index_pos_vertex, vertex_idx in enumerate(index_vertex_tetra_v):
            D_mat_index = D_mat[index_pos_vertex, :]
            vertex_pos_in_Bn = np.argwhere(vertex_idx == list_idx_vertex)[0][0]
            # print('VERTEX {} POS {}'.format(vertex_idx,vertex_pos_in_Bn))
            # print(D_mat_index)
            Bn_mat[vertex_pos_in_Bn, :] += 1 / c_n * D_mat_index

            # print(Bn_mat)
            # print(Bn_mat)
    return Bn_mat, list_idx_vertex


def outer(a):
    '''
    a 1 dimensional vector of size 3
    '''
    outer_a = [[a[0], 0, 0],
               [0, a[0], 0],
               [0, 0, a[0]],
               [a[1], 0, 0],
               [0, a[1], 0],
               [0, 0, a[1]],
               [a[2], 0, 0],
               [0, a[2], 0],
               [0, 0, a[2]]]

    return outer_a


def find_neighbors(idx_vertex, grid):
    '''
    :param idx_vertex:
    :param grid:
    :return:
    '''
    tetrahedrons = grid.cells.reshape(-1, 5)[:, 1:]
    v_i = [num_cell
           for num_cell, tetra in enumerate(tetrahedrons)
           if any([idx_cell in [idx_vertex] for idx_cell in [*tetra]])]

    list_idx_vertex = np.unique(tetrahedrons[v_i])
    neighbors = set(list_idx_vertex) - set([idx_vertex])
    return neighbors

def update_outlier(Graph, mean_glob, std_glob, z_i, update_outlier=False, z_score_lim=.99):
    '''

    :param Graph:
    :param mean_glob:
    :param std_glob:
    :param z_i:
    :param update_outlier:
    :param z_score_lim:
    :return:
    '''
    ls_outlier = []

    for idx in nx.get_node_attributes(Graph, 'translation_from_Pn').keys():
        # idx_neighbors=[*Graph.neighbors(idx)]
        idx_neighbors = np.unique(
            np.concatenate([[*Graph.neighbors(idx_n)] for idx_n in [*Graph.neighbors(idx)]]))
        ls_Pn_neigbours = np.array(
            [nx.get_node_attributes(Graph, 'Pn')[i] for i in idx_neighbors])
        ls_Qn_neigbours = np.array(
            [nx.get_node_attributes(Graph, 'Qn')[i] for i in idx_neighbors])
        ls_t_neigbours = (ls_Qn_neigbours - ls_Pn_neigbours)[:, 1:-1]
        mse_i_neighbours = np.array([nx.get_node_attributes(Graph, 'MSE')[i] for i in idx_neighbors])
        mse_i_neighbours[mse_i_neighbours > 1] = 1

        z_score = z_i[idx]
        if z_score > z_score_lim:
            ls_outlier.append(idx)

            Pn_idx = nx.get_node_attributes(Graph, 'Pn')[idx]
            Pn_idx_cop = np.copy(Pn_idx)
            test_t = np.dot(-np.log(mse_i_neighbours), ls_t_neigbours) / (-np.log(mse_i_neighbours)).sum()
            # Pn_idx_cop[1:-1] = Pn_idx_cop[1:-1] + test_t
            if np.mean(mse_i_neighbours) == 1.:
                test_t = mean_glob
            print(idx)
            print(test_t)

            Pn_idx_cop[1:-1] = Pn_idx_cop[1:-1] + test_t
            if update_outlier:
                # pass
                Graph.add_node(idx, Qn=Pn_idx_cop, accumulated_translation=-test_t)

    # except:
    #  pass
    return Graph, ls_outlier

def save_values(saving_values, translation_mean, translation_std, thetas_mean, thetas_std, Graph):
    MSE_all = [nx.get_node_attributes(Graph, 'MSE_or')[key]
               for key in nx.get_node_attributes(Graph, 'MSE').keys()]
    loss_all = [nx.get_node_attributes(Graph, 'loss_reg')[key]
                for key in nx.get_node_attributes(Graph, 'loss_reg').keys()]
    params = np.mean([nx.get_node_attributes(Graph, 'pars')[key]
                for key in nx.get_node_attributes(Graph, 'loss_reg').keys()],axis=0)
    In_var = np.mean(MSE_all)

    print('T_MEAN = {}; T_STD = {} ;An_mean = {}; AnSTD = {},  In_VAR = {}'.format(translation_mean, translation_std,
                                                                                   thetas_mean, thetas_std, In_var))

    saving_values['MSE'].append(MSE_all)
    saving_values['LOSS'].append(loss_all)
    saving_values['VAR_Tn'].append(translation_std ** 2)
    saving_values['VAR_An'].append(thetas_std ** 2)
    saving_values['VAR_In'].append(In_var)
    saving_values['PARAMETERS'].append(params)
    return saving_values

def estimate_mean_variance_as_paper(Graph, grid):
    n_vertices = Graph.number_of_nodes()
    # tn = np.stack(
    #    [nx.get_node_attributes(Graph, 'translation_from_Pn')[key] for key in range(n_vertices)])
    Qns = np.array(np.stack(
        [nx.get_node_attributes(Graph, 'Qn')[key][1:-1] for key in range(n_vertices)]))
    Pns = np.array(np.stack(
        [nx.get_node_attributes(Graph, 'Pn')[key][1:-1] for key in range(n_vertices)]))

    tn = np.array(Qns - Pns)
    n_vertices = grid.number_of_points

    std_tn = tn.ravel().std()

    t = np.mean(tn, axis=0)
    t_cov = np.cov(tn, rowvar=False)

    mahalanobis_distance = (np.diag(
        (np.dot((np.dot((tn - t), np.linalg.inv(t_cov))), (tn - t).T)))) ** 0.5
    z = chi2.cdf(mahalanobis_distance, 2)

    An = np.stack(
        [nx.get_node_attributes(Graph, 'affines_or')[key][:, :-1] for key in range(n_vertices)])

    A = np.mean(An, axis=0)

    ls_nablaQ = []
    Ra_new = np.zeros([tn.shape[0], 3])

    for idx_vertex_RA in range(n_vertices):
        An_n = An[idx_vertex_RA]
        Bn_idx_n, ls_vertex_Bn_idx = calc_bn(idx_vertex_RA, grid)
        nablaQ_n = 0
        for idx_enum_i, _ in enumerate(Bn_idx_n):
            Bn_i = Bn_idx_n[idx_enum_i, ...]
            Ra_new[ls_vertex_Bn_idx[idx_enum_i], :] += Bn_i.dot((An_n - np.eye(3)))
            nablaQ_n += np.matrix(outer(Bn_i)).dot(Qns[ls_vertex_Bn_idx[idx_enum_i], ...])
        ls_nablaQ.append(nablaQ_n.reshape([3, 3]))  ### MAYBE IT IS .T

    Ra = np.array(Ra_new).reshape([-1])
    Rt = tn.reshape([-1])

    nablaQ = np.array(ls_nablaQ)
    diff = An - nablaQ
    std_A = np.std(diff.ravel())
    std_t = np.array([std_tn] * 3)

    return t, A, std_t, std_tn, std_A, Rt, Ra, z


def Q_step(Graph, grid, K, lambda_Km, divide_variance=False):
    '''
    Graph : A graph
    Grid : a grid
    K : the np array of the three stiffness matrices in that order Km,Kt,ka
    lambda_Km :
    '''
    # estimate variance and mean
    new_graph = Graph.copy()
    t, A, std_t, std_tn, std_A, Rt, Ra, z = estimate_mean_variance_as_paper(new_graph, grid)
    new_graph, _ = update_outlier(new_graph, t, std_t, z, update_outlier=True, z_score_lim=.95)
    #
    stiff = np.copy(K)
    Km_Q = stiff[0, ...]
    Kt_Q = stiff[1, ...]
    Ka_Q = stiff[2, ...]

    if divide_variance:
        Kt_Q /= (std_tn ** 2)
        Ka_Q /= (std_A ** 2)
        Rt /= (std_tn ** 2)
        Ra /= (std_A ** 2)

    matrix_assembly = lambda_Km*Km_Q + Kt_Q + Ka_Q

    x = np.dot(np.linalg.inv(np.matrix(matrix_assembly)), (Rt + Ra))
    x_reshaped = np.array(x.reshape([-1, 3]))

    for idx_vertex in range(new_graph.number_of_nodes()):
        Pn_idx = np.copy(nx.get_node_attributes(new_graph, 'Pn')[idx_vertex])
        new_Qn = Pn_idx
        new_Qn[1:-1] += x_reshaped[idx_vertex, :]
        new_graph.add_node(idx_vertex, Qn=new_Qn, accumulated_translation=-x_reshaped[idx_vertex, :])
    return new_graph, x_reshaped


def extract_u_for_grid2(grid, graph):
    '''
    u is a matrix of size n_cells * 12

    '''
    cells = grid.cells.reshape(-1, 5)[:, 1:]
    # u = np.array([nx.get_node_attributes(graph,'accumulated_translation')[_] for _ in range(grid.number_of_points)])
    u = np.array([(nx.get_node_attributes(graph, 'Qn')[_] -
                   nx.get_node_attributes(graph, 'Pn')[_])[1:-1] for _ in range(grid.number_of_points)])

    ls_u = u[cells]

    return ls_u

def convert_grid_to_fiximage(mesh, vox_spacing, final_dims):
    n_tetrahedrons = mesh.number_of_cells
    mesh['num_tet'] = np.arange(n_tetrahedrons) + 1  ## each tetrahedron num is added 1++++

    # Dmat for lambda computation
    #Ds = compute_D(mesh)
    #Ds_lin = Ds.reshape([-1, 16])
    #mesh['Ds'] = Ds_lin

    # Displacement estimated
    #U = extract_u_for_grid2(mesh, graph)
    #U_reshape = U.reshape([-1, 12])
    #mesh['U'] = U_reshape

    new_grid = polyv.create_grid(mesh, dimensions=final_dims)
    #if adapt_origin:
    #    origin = padding * vox_spacing
    #else:
    origin = np.array([0., 0., 0.])

    new_grid.SetOrigin(origin)

    new_grid.SetSpacing(vox_spacing)
    image_resampled = new_grid.sample(mesh)
    return image_resampled, origin  # , [start_at,ends_at]