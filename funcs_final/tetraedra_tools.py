from math import factorial
import numpy as np
from scipy.spatial import distance
from scipy.optimize import fmin_cg
import scipy
import networkx as nx
import sys
import tensorflow as tf
import tqdm

try:
    sys.path.append('./utils/pyOptFEM/')
    from pyOptFEM.FEM3D.assembly import *
    from pyOptFEM import *
except:
    sys.path.append('./utils/pyOptFEM/pyOptFEM/')
    from pyOptFEM import *
    from pyOptFEM.pyOptFEM.FEM3D.assembly import *



def simplex_volume(*, vertices=None, sides=None) -> float:
    """
    Return the volume of the simplex with given vertices or sides.

    If vertices are given they must be in a NumPy array with shape (N+1, N):
    the position vectors of the N+1 vertices in N dimensions. If the sides
    are given, they must be the compressed pairwise distance matrix as
    returned from scipy.spatial.distance.pdist.

    Raises a ValueError if the vertices do not form a simplex (for example,
    because they are coplanar, colinear or coincident).

    Warning: this algorithm has not been tested for numerical stability.
    """

    # Implements http://mathworld.wolfram.com/Cayley-MengerDeterminant.html

    if (vertices is None) == (sides is None):
        raise ValueError("Exactly one of vertices and sides must be given")

    # β_ij = |v_i - v_k|²
    if sides is None:
        vertices = np.asarray(vertices, dtype=float)
        sq_dists = distance.pdist(vertices, metric='sqeuclidean')

    else:
        sides = np.asarray(sides, dtype=float)
        if not distance.is_valid_y(sides):
            raise ValueError("Invalid number or type of side lengths")

        sq_dists = sides ** 2

    # Add border while compressed
    num_verts = distance.num_obs_y(sq_dists)
    bordered = np.concatenate((np.ones(num_verts), sq_dists))

    # Make matrix and find volume
    sq_dists_mat = distance.squareform(bordered)

    coeff = - (-2) ** (num_verts-1) * factorial(num_verts-1) ** 2
    vol_square = np.linalg.det(sq_dists_mat) / coeff

    if vol_square <= 0:
        raise ValueError('Provided vertices do not form a tetrahedron')

    return np.sqrt(vol_square)

def ComputeGradient(ql):
    D12=ql[0]-ql[1];D13=ql[0]-ql[2];D14=ql[0]-ql[3]
    D23=ql[1]-ql[2];D24=ql[1]-ql[3];D34=ql[2]-ql[3]
    G=np.zeros((3,4))
    G[0,0]=-D23[1]*D24[2] + D23[2]*D24[1]
    G[1,0]= D23[0]*D24[2] - D23[2]*D24[0]
    G[2,0]=-D23[0]*D24[1] + D23[1]*D24[0]
    G[0,1]= D13[1]*D14[2] - D13[2]*D14[1]
    G[1,1]=-D13[0]*D14[2] + D13[2]*D14[0]
    G[2,1]= D13[0]*D14[1] - D13[1]*D14[0]
    G[0,2]=-D12[1]*D14[2] + D12[2]*D14[1]
    G[1,2]= D12[0]*D14[2] - D12[2]*D14[0]
    G[2,2]=-D12[0]*D14[1] + D12[1]*D14[0]
    G[0,3]= D12[1]*D13[2] - D12[2]*D13[1]
    G[1,3]=-D12[0]*D13[2] + D12[2]*D13[0]
    G[2,3]= D12[0]*D13[1] - D12[1]*D13[0]
    return G


def D_idx(idx_points, grid, Qns):
    '''
    Compute the Derivative of the tetrahedron, aka the shape vectors
    :param idx_points: the idx of the points of the tetrahedron
    :param grid: the tetrahedrical mesh as a pvysta array
    :return:
    '''
    cells = grid.cells.reshape(-1, 5)[:, 1:]

    # RETURN THE NUMBER OF THE CELLS
    # find the num of tetra
    # if idx_points in cell

    num_cells = [num_cell
                 for num_cell, cell in enumerate(np.reshape(grid.cells, [-1, 5])[:, 1:])
                 if any([idx_cell in idx_points for idx_cell in [*cell]])]
    ls_Di = []
    ls_Dmat = []
    ls_pos_idx_cell = []
    ls_cell = []
    ls_nablaQ = []
    for num_cell in num_cells:
        cell = cells[num_cell]
        idx_cell = np.argwhere(cell == idx_points[0])[0][0]
        ls_pos_idx_cell.append(idx_cell)

        P = grid.points[cell]
        Q = Qns[cell]

        ls_cell.append(cell)

        v = simplex_volume(vertices=P)
        D_mat = ComputeGradient(P).T / (6 * v)

        ls_Dmat.append(D_mat)
        # erreur sur le nablaq  il faut MODIFIER et updater grid.points[cell] pour le déplacement ette a Qn!
        ls_nablaQ.append(np.einsum('ij,jk->ik',Q.T,D_mat).T)

        Di = D_mat[idx_cell]
        ls_Di.append(Di)
    return np.stack(ls_Dmat, axis=0), np.sum(ls_Di, axis=0), ls_pos_idx_cell, np.stack(ls_cell, axis=0), np.mean(np.stack(ls_nablaQ), axis=0)


def calc_D(tet, idx_Pn, V):
    '''
    derivatives
    '''

    tetpn_without_idx = np.delete(tet, idx_Pn, axis=0)
    cross = (-1) ** (idx_Pn+1) * (np.cross(tetpn_without_idx[0], tetpn_without_idx[1]) +
                                np.cross(tetpn_without_idx[1], tetpn_without_idx[2]) +
                                np.cross(tetpn_without_idx[2], tetpn_without_idx[0])) / (6 * V)
    E_i = -np.dot(cross, tetpn_without_idx[0])
    mat = np.zeros(4)
    mat[0:3] = cross
    mat[3] = E_i
    return mat

def calc_shape_vectors(tetrahedron):
    '''
    func to extract from a tetraedra

    INPUTS :
    tetrahedra : a tetreahedra
    coord: a tupple of (x,y,z)

    OUTPUTS
    V : Volume of the tetraedra
    H : Matrix composed of D_i and -E_i
    lambda_i : shapes vector, sum = 1

    '''

    # barycenter=np.mean(tetrahedra,axis=0)
    # calculate volume
    tetrahedra = np.copy(tetrahedron)
    V = simplex_volume(vertices=tetrahedra)

    # calculate D vectors
    H_inv = [calc_D(tetrahedra, idx, V) for idx in range(4)]
    H_inv = -np.matrix(np.array(H_inv))
    if np.linalg.det(np.linalg.inv(H_inv)) < 0:
        H_inv=-H_inv

    # one can check that det(H-1)= 6V
    # deduce lambda
    #lambda_i = np.dot(H_inv, np.pad(coords, (0, 1), mode='constant', constant_values=1))

    return V, np.array(H_inv)


def pyvistamesh_global_stiffness(grid_mesh, lambda_lame, shear_modulus,  Num = 0):
    """
    :param grid_mesh:
    :param lame_Poisson: shear modulus
    :param shear_modulus:
    :return:
    """

    coors = np.array(grid_mesh.points)  #coordinates Th.q
    n_vertices = grid_mesh.n_points  #Th.nq # n_points

    # mat_ids=np.array(grid_mesh.cells)
    conn = np.reshape(grid_mesh.cells, [-1, 5])[:, 1:]  #Th.me # edges

    #Th.q # coordonnées # int
    n_cells = grid_mesh.n_cells  #Th.nme # num cells (tetrahedrons)
    V = np.array(
        [simplex_volume(vertices=coors[conn][_]) for _ in range(n_cells)])  #Th.volumes # volumes des tetraedres

    return StiffElasAssembling3DP1base(n_vertices, n_cells, coors, conn, V, shear_modulus, lambda_lame, Num)

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



