import sys
import ray
import ray.experimental.tf_utils
# data processing
import tensorflow as tf
import tetgen
from skimage import measure
from math import factorial
import numpy as np
from scipy.spatial import distance
import networkx as nx
import itertools
import nibabel as nib
from scipy import ndimage

# ploting
import scipy
from scipy.optimize import fmin_cg
from scipy.stats import chi2
from scipy.stats import wasserstein_distance
import pyvista as polyv

# stats
import math
import smm

ray.init()


try:
    sys.path.append('./utils/pyOptFEM/')
    from pyOptFEM.FEM3D.assembly import *
    from pyOptFEM import *
except:
    sys.path.append('./utils/pyOptFEM/pyOptFEM/')
    from pyOptFEM import *
    from pyOptFEM.pyOptFEM.FEM3D.assembly import *

#HYPERPARAMETERS


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'Begin Registration')
    parser.add_argument('--num_p', required = True, type = str,
                        help='NUM_P')
    parser.add_argument('--num_tetra', default = 1000, required=False,
                        help='NUM TETRAHEDRON')
    parser.add_argument('--do_Q_step', default = False, required=False,
                        help='if Q step is performed')
    args = parser.parse_args()

    NUM_P = str(args.num_p)
    MIN_NUM_TETRA = np.int16(args.num_tetra)
    Q_STEP = args.do_Q_step


#NUM_P = 01'
PATCH_SIZE_Qn = 16
PATCH_SIZE_Pn = 8
NU = 0.3                          #(0< NU <0.5)
YOUNG_MODULUS = 1
THETA_MIN=math.pi/10
SHEAR_MIN=.1
TRANSLATION_MIN=.7
NUM_EPOCH=5
NUM_ITERS=80    #
LR = 5e-2
PVAL_OUTLIER = 0.02
lame_lambda = YOUNG_MODULUS*NU /  ((1 + NU)*(1-2*NU))
shear_modulus =  YOUNG_MODULUS / (2* ( 1 +  NU))
DIS_TETRAHEDRON = 50
lr_Q_STEP=5e-3
PYRAMID_LEVELS=[4,3]
N_BATCH=12
#MIN_NUM_TETRA = 2000
MAX_ITER_Qstep = 500
#Q_STEP = False

#FUNCS
print('STARTING')

def batch_affine_warp3d(imgs, theta):
    """
    affine transforms 3d images

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, zlen, n_channel]
    theta : tf.Tensor
        parameters of affine transformation
        [n_batch, 12]

    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, zlen, n_channel]
    """
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    zlen = tf.shape(imgs)[3]
    theta = tf.reshape(theta, [-1, 3, 4])
    matrix = tf.slice(theta, [0, 0, 0], [-1, -1, 3])
    t = tf.slice(theta, [0, 0, 3], [-1, -1, -1])

    grids = batch_mgrid(n_batch, xlen, ylen, zlen)
    grids = tf.reshape(grids, [n_batch, 3, -1])

    T_g = tf.matmul(matrix, grids) + t
    T_g = tf.reshape(T_g, [n_batch, 3, xlen, ylen, zlen])
    output = batch_warp3d(imgs, T_g)
    return output

def batch_warp3d(imgs, mappings):
    """
    warp image using mapping function
    I(x) -> I(phi(x))
    phi: mapping function

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, zlen, n_channel]
    mapping : tf.Tensor
        grids representing mapping function
        [n_batch, xlen, ylen, zlen, 3]

    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, zlen, n_channel]
    """
    n_batch = tf.shape(imgs)[0]
    coords = tf.reshape(mappings, [n_batch, 3, -1])
    x_coords = tf.slice(coords, [0, 0, 0], [-1, 1, -1])
    y_coords = tf.slice(coords, [0, 1, 0], [-1, 1, -1])
    z_coords = tf.slice(coords, [0, 2, 0], [-1, 1, -1])
    x_coords_flat = tf.reshape(x_coords, [-1])
    y_coords_flat = tf.reshape(y_coords, [-1])
    z_coords_flat = tf.reshape(z_coords, [-1])

    output = _interpolate3d(imgs, x_coords_flat, y_coords_flat, z_coords_flat)
    return output

def _repeat(base_indices, n_repeats):
    base_indices = tf.matmul(
        tf.reshape(base_indices, [-1, 1]),
        tf.ones([1, n_repeats], dtype='int32'))
    return tf.reshape(base_indices, [-1])

def _interpolate2d(imgs, x, y):
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    n_channel = tf.shape(imgs)[3]

    x = tf.to_float(x)
    y = tf.to_float(y)
    xlen_f = tf.to_float(xlen)
    ylen_f = tf.to_float(ylen)
    zero = tf.zeros([], dtype='int32')
    max_x = tf.cast(xlen - 1, 'int32')
    max_y = tf.cast(ylen - 1, 'int32')

    # scale indices from [-1, 1] to [0, xlen/ylen]
    x = (x + 1.) * (xlen_f - 1.) * 0.5
    y = (y + 1.) * (ylen_f - 1.) * 0.5

    # do sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    base = _repeat(tf.range(n_batch) * xlen * ylen, ylen * xlen)
    base_x0 = base + x0 * ylen
    base_x1 = base + x1 * ylen
    index00 = base_x0 + y0
    index01 = base_x0 + y1
    index10 = base_x1 + y0
    index11 = base_x1 + y1

    # use indices to lookup pixels in the flat image and restore
    # n_channel dim
    imgs_flat = tf.reshape(imgs, [-1, n_channel])
    imgs_flat = tf.to_float(imgs_flat)
    I00 = tf.gather(imgs_flat, index00)
    I01 = tf.gather(imgs_flat, index01)
    I10 = tf.gather(imgs_flat, index10)
    I11 = tf.gather(imgs_flat, index11)

    # and finally calculate interpolated values
    dx = x - tf.to_float(x0)
    dy = y - tf.to_float(y0)
    w00 = tf.expand_dims((1. - dx) * (1. - dy), 1)
    w01 = tf.expand_dims((1. - dx) * dy, 1)
    w10 = tf.expand_dims(dx * (1. - dy), 1)
    w11 = tf.expand_dims(dx * dy, 1)
    output = tf.add_n([w00*I00, w01*I01, w10*I10, w11*I11])

    # reshape
    output = tf.reshape(output, [n_batch, xlen, ylen, n_channel])

    return output


def _interpolate3d(imgs, x, y, z):
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    zlen = tf.shape(imgs)[3]
    n_channel = tf.shape(imgs)[4]

    x = tf.cast(x,tf.float32)
    y = tf.cast(y,tf.float32)
    z = tf.cast(z,tf.float32)
    xlen_f = tf.cast(xlen,tf.float32)
    ylen_f = tf.cast(ylen,tf.float32)
    zlen_f = tf.cast(zlen,tf.float32)
    zero = tf.zeros([], tf.int32)
    max_x = tf.cast(xlen - 1, tf.int32)
    max_y = tf.cast(ylen - 1, tf.int32)
    max_z = tf.cast(zlen - 1, tf.int32)

    # scale indices from [-1, 1] to [0, xlen/ylen]
    x = (x + 1.) * (xlen_f - 1.) * 0.5
    y = (y + 1.) * (ylen_f - 1.) * 0.5
    z = (z + 1.) * (zlen_f - 1.) * 0.5

    # do sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    z0 = tf.cast(tf.floor(z), 'int32')
    z1 = z0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    z0 = tf.clip_by_value(z0, zero, max_z)
    z1 = tf.clip_by_value(z1, zero, max_z)
    base = _repeat(tf.range(n_batch) * xlen * ylen * zlen,
                   xlen * ylen * zlen)
    base_x0 = base + x0 * ylen * zlen
    base_x1 = base + x1 * ylen * zlen
    base00 = base_x0 + y0 * zlen
    base01 = base_x0 + y1 * zlen
    base10 = base_x1 + y0 * zlen
    base11 = base_x1 + y1 * zlen
    index000 = base00 + z0
    index001 = base00 + z1
    index010 = base01 + z0
    index011 = base01 + z1
    index100 = base10 + z0
    index101 = base10 + z1
    index110 = base11 + z0
    index111 = base11 + z1

    # use indices to lookup pixels in the flat image and restore
    # n_channel dim
    imgs_flat = tf.reshape(imgs, [-1, n_channel])
    imgs_flat = tf.cast(imgs_flat,tf.float32)
    I000 = tf.gather(imgs_flat, index000)
    I001 = tf.gather(imgs_flat, index001)
    I010 = tf.gather(imgs_flat, index010)
    I011 = tf.gather(imgs_flat, index011)
    I100 = tf.gather(imgs_flat, index100)
    I101 = tf.gather(imgs_flat, index101)
    I110 = tf.gather(imgs_flat, index110)
    I111 = tf.gather(imgs_flat, index111)

    # and finally calculate interpolated values
    dx = x - tf.cast(x0,tf.float32)
    dy = y - tf.cast(y0,tf.float32)
    dz = z - tf.cast(z0,tf.float32)
    w000 = tf.expand_dims((1. - dx) * (1. - dy) * (1. - dz), 1)
    w001 = tf.expand_dims((1. - dx) * (1. - dy) * dz, 1)
    w010 = tf.expand_dims((1. - dx) * dy * (1. - dz), 1)
    w011 = tf.expand_dims((1. - dx) * dy * dz, 1)
    w100 = tf.expand_dims(dx * (1. - dy) * (1. - dz), 1)
    w101 = tf.expand_dims(dx * (1. - dy) * dz, 1)
    w110 = tf.expand_dims(dx * dy * (1. - dz), 1)
    w111 = tf.expand_dims(dx * dy * dz, 1)
    output = tf.add_n([w000 * I000, w001 * I001, w010 * I010, w011 * I011,
                       w100 * I100, w101 * I101, w110 * I110, w111 * I111])

    # reshape
    output = tf.reshape(output, [n_batch, xlen, ylen, zlen, n_channel])

    return output

def mgrid(*args, **kwargs):
    """
    create orthogonal grid
    similar to np.mgrid

    Parameters
    ----------
    args : int
        number of points on each axis
    low : float
        minimum coordinate value
    high : float
        maximum coordinate value

    Returns
    -------
    grid : tf.Tensor [len(args), args[0], ...]
        orthogonal grid
    """
    low = kwargs.pop("low", -1)
    high = kwargs.pop("high", 1)
    low = tf.cast(low,tf.float32)
    high = tf.cast(high,tf.float32)
    coords = (tf.linspace(low, high, arg) for arg in args)
    grid = tf.stack(tf.meshgrid(*coords, indexing='ij'))
    return grid

def batch_mgrid(n_batch, *args, **kwargs):
    """
    create batch of orthogonal grids
    similar to np.mgrid

    Parameters
    ----------
    n_batch : int
        number of grids to create
    args : int
        number of points on each axis
    low : float
        minimum coordinate value
    high : float
        maximum coordinate value

    Returns
    -------
    grids : tf.Tensor [n_batch, len(args), args[0], ...]
        batch of orthogonal grids
    """
    grid = mgrid(*args, **kwargs)
    grid = tf.expand_dims(grid, 0)
    grids = tf.tile(grid, [n_batch] + [1 for _ in range(len(args) + 1)])
    return grids

# Preprocessing
def find_idx_Pn(Pn, tuple_coord):
    return [i for i, Pn_i in enumerate(Pn) if Pn_i == tuple_coord][0]


def build_mesh_from_3Dmask(mask, STEP_SIZE, SPACING, surface=True):
    '''update_outlier
    build a mesh from a 3D mask
    mask :
    STEP_SIZE :
    SPACING :
    surface : if true only surfacic mesh


    '''

    com = ndimage.center_of_mass(mask)
    com_xminus_1 = np.int16(com[2])
    com_xplus_1 = np.int16(com[2])

    SPAGING_DIL = 20
    while any([mask[np.int16(com[0]), np.int16(com[1]), com_xminus_1 - 1] == 0,
               mask[np.int16(com[0]), np.int16(com[1]), com_xminus_1 - 2] == 0]):
        mask[np.int16(com[0]) - SPAGING_DIL:np.int16(com[0]) + SPAGING_DIL,
        np.int16(com[1]) - SPAGING_DIL:np.int16(com[1]) + SPAGING_DIL, com_xminus_1 - 1] = 1
        com_xminus_1 -= 1
    while any([mask[np.int16(com[0]), np.int16(com[1]), com_xplus_1 + 1] == 0,
               mask[np.int16(com[0]), np.int16(com[1]), com_xplus_1 + 2] == 0]):
        mask[np.int16(com[0]) - SPAGING_DIL:np.int16(com[0]) + SPAGING_DIL,
        np.int16(com[1]) - SPAGING_DIL:np.int16(com[1]) + SPAGING_DIL, com_xplus_1 + 1] = 1
        com_xplus_1 += 1
    mask[np.int16(com[0]) - SPAGING_DIL:np.int16(com[0]) + SPAGING_DIL,
    np.int16(com[1]) - SPAGING_DIL:np.int16(com[1]) + SPAGING_DIL, np.int16(com[2])] = 1

    verts, faces, normals, values = measure.marching_cubes_lewiner(mask, step_size=STEP_SIZE, spacing=SPACING,
                                                                   allow_degenerate=True)
    if surface:

        coords_space = np.append(np.insert(verts, 0, values=0, axis=1), np.zeros([verts.shape[0], 1]), axis=1)
        # coords= np.append(np.insert(verts*spacing_orig, 0, values=0, axis=1),np.zeros([verts.shape[0],1]),axis=1)
        # coords=coords_space
        coords_list = list(coords_space)
        G = build_GRAPH(coords_list, faces)
    else:
        verts = np.round(verts)
        faces_1_adapted = np.ones([faces.shape[0], faces.shape[1] + 1]) * 3
        faces_1_adapted[:, 1:] = faces
        surf = polyv.PolyData(verts, faces_1_adapted)
        tet = tetgen.TetGen(surf)
        tet.make_manifold()
        tet.tetrahedralize(order=1, mindihedral=10, steinerleft=-1, minratio=1.01)
        grid = tet.grid
        cells = grid.cells.reshape(-1, 5)[:, :]
        points = np.array(grid.points)
        coords_list = list(points)

        points_edges = points[cells[:, 1:]]

        Pn = [(Pn_i[0], Pn_i[1], Pn_i[2]) for Pn_i in points]

        G = nx.Graph()
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

    print('Number of Vertex : {}'.format(G.number_of_nodes()))

    return coords_list, G, grid


def extract_patch_from_coord_list(np_array, c_list, patch_size):
    '''
    c_list : a list of coordinates
    patch_size : the size of the patch
    '''
    ls_patch = []
    for i, coords_i in enumerate(c_list):

        patch = np_array[:, coords_i[1] - patch_size // 2:coords_i[1] + patch_size // 2,
                coords_i[2] - patch_size // 2:coords_i[2] + patch_size // 2
        , coords_i[3] - patch_size // 2:coords_i[3] + patch_size // 2, :]
        if patch.shape == (1, patch_size, patch_size, patch_size, 1):
            ls_patch.append(patch)
        else:
            print('ERROR ' + str(i))

    return ls_patch

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

# Tetrahedron Tools
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

    coeff = - (-2) ** (num_verts - 1) * factorial(num_verts - 1) ** 2
    vol_square = np.linalg.det(sq_dists_mat) / coeff

    if vol_square <= 0:
        raise ValueError('Provided vertices do not form a tetrahedron')

    return np.sqrt(vol_square)

def ComputeGradient(ql):
    D12 = ql[0] - ql[1];
    D13 = ql[0] - ql[2];
    D14 = ql[0] - ql[3]
    D23 = ql[1] - ql[2];
    D24 = ql[1] - ql[3];
    D34 = ql[2] - ql[3]
    G = np.zeros((3, 4))
    G[0, 0] = -D23[1] * D24[2] + D23[2] * D24[1]
    G[1, 0] = D23[0] * D24[2] - D23[2] * D24[0]
    G[2, 0] = -D23[0] * D24[1] + D23[1] * D24[0]
    G[0, 1] = D13[1] * D14[2] - D13[2] * D14[1]
    G[1, 1] = -D13[0] * D14[2] + D13[2] * D14[0]
    G[2, 1] = D13[0] * D14[1] - D13[1] * D14[0]
    G[0, 2] = -D12[1] * D14[2] + D12[2] * D14[1]
    G[1, 2] = D12[0] * D14[2] - D12[2] * D14[0]
    G[2, 2] = -D12[0] * D14[1] + D12[1] * D14[0]
    G[0, 3] = D12[1] * D13[2] - D12[2] * D13[1]
    G[1, 3] = -D12[0] * D13[2] + D12[2] * D13[0]
    G[2, 3] = D12[0] * D13[1] - D12[1] * D13[0]
    return G

def D_idx(idx_points, grid):
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
    for num_cell in num_cells:
        cell = cells[num_cell]
        idx_cell = np.argwhere(cell == idx_points[0])[0][0]
        ls_pos_idx_cell.append(idx_cell)

        Q = grid.points[cell]
        ls_cell.append(cell)

        v = simplex_volume(vertices=Q)

        D_mat = ComputeGradient(Q).T / (6 * v)
        ls_Dmat.append(D_mat)

        Di = D_mat[idx_cell]
        ls_Di.append(Di)
    return np.stack(ls_Dmat, axis=0), np.sum(ls_Di, axis=0), ls_pos_idx_cell, np.stack(ls_cell, axis=0)

def calc_sp(trans, Dmat_D_idx):
    '''
    calculate the angle between D and the translation
    '''
    mean_Di = tf.constant(Dmat_D_idx, tf.float32)
    eps_d = tf.cast(1e-3, tf.float32)
    mean_Di = tf.reshape(mean_Di, [3])
    scalar_product = tf.tensordot(mean_Di + eps_d, tf.reshape(trans + eps_d, [3]), axes=1)
    norm_t = tf.norm(trans + eps_d)
    norm_Di = tf.norm(mean_Di + eps_d)
    costhet = tf.divide(scalar_product, norm_t * norm_Di)
    dir_scalar = (1 - tf.math.abs(costhet))  # * norm_t #norm_t small n are not
    return dir_scalar

# Model Funcs

def rodrigues_batch(rvecs):
    """
    Convert a batch of axis-angle rotations in rotation vector form shaped
    (batch, 3) to a batch of rotation matrices shaped (batch, 3, 3).
    See
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

    FROM :
    https://github.com/blzq/tf_rodrigues

    """
    batch_size = tf.shape(rvecs)[0]
    #tf.assert_equal(tf.shape(rvecs)[1], 3)

    thetas = tf.norm(rvecs, axis=1, keepdims=True)
    is_zero = tf.equal(tf.squeeze(thetas), 0.0)
    u = rvecs / thetas

    # Each K is the cross product matrix of unit axis vectors
    # pyformat: disable
    zero = tf.zeros([batch_size])  # for broadcasting
    Ks_1 = tf.stack([  zero   , -u[:, 2],  u[:, 1] ], axis=1)  # row 1
    Ks_2 = tf.stack([  u[:, 2],  zero   , -u[:, 0] ], axis=1)  # row 2
    Ks_3 = tf.stack([ -u[:, 1],  u[:, 0],  zero    ], axis=1)  # row 3
    # pyformat: enable
    Ks = tf.stack([Ks_1, Ks_2, Ks_3], axis=1)                  # stack rows

    Rs = tf.eye(3, batch_shape=[batch_size]) +          tf.sin(thetas)[..., tf.newaxis] * Ks +          (1 - tf.cos(thetas)[..., tf.newaxis]) * tf.matmul(Ks, Ks)

    # Avoid returning NaNs where division by zero happened
    mat=tf.where(is_zero,
        tf.eye(3, batch_shape=[batch_size]), Rs)
    mat=tf.reshape(mat,[3,3])
    return mat

def extract_par_from_outputs(outputs):
    '''
    :param outputs: outputs of network
    :return: translation, affine
    '''
    outputs_tf=tf.reshape(outputs,[-1,4,3])
    thetas=tf.reshape(tf.transpose(tf.slice(outputs_tf,[0,0,0],[-1,1,-1])),[3,-1])
    thetas=tf.reshape(thetas,[1,3,-1])
    shears=tf.transpose(tf.reshape(tf.slice(outputs_tf,[0,1,0],[-1,2,-1]),[-1,6]))
    shears=tf.reshape(shears,[1,6,-1])
    translations=tf.transpose(tf.reshape(tf.slice(outputs_tf,[0,3,0],[-1,1,-1]),[-1,3]))
    translations=tf.reshape(translations,[1,3,-1])
    pars = tf.concat([thetas,shears],axis=1)
    return translations, pars

def Affine_Matrix_from_outputs(matrix, with_translation=True):
    '''
    transform output of Neural Network to an affine matrix of
    dimension 3x4

    '''
    theta0 = tf.reshape(matrix[0], [1])
    theta1 = tf.reshape(matrix[1], [1])
    theta2 = tf.reshape(matrix[2], [1])

    # shear
    shear0 = tf.reshape(matrix[3], [1])
    shear1 = tf.reshape(matrix[4], [1])
    shear2 = tf.reshape(matrix[5], [1])
    shear3 = tf.reshape(matrix[6], [1])
    shear4 = tf.reshape(matrix[7], [1])
    shear5 = tf.reshape(matrix[8], [1])

    if with_translation:
        t1 = tf.reshape(matrix[9], [1])
        t2 = tf.reshape(matrix[10], [1])
        t3 = tf.reshape(matrix[11], [1])
    else:
        t1 = tf.cast(tf.reshape(0., [1]), tf.float32)
        t2 = tf.cast(tf.reshape(0., [1]), tf.float32)
        t3 = tf.cast(tf.reshape(0., [1]), tf.float32)

    theta = tf.concat([theta0, theta1, theta2], axis=0)
    thetas = tf.expand_dims(theta, axis=0)
    rot = rodrigues_batch(
        thetas)
    # shearing_matrix=tf.cast([[1,shear0[0],shear1[0]],[shear2[0],1.,shear3[0]],[shear4[0],shear5[0],1.]],tf.float32)
    shearing_matrix = tf.cast([[shear0[0], shear1[0], shear2[0]],
                               [shear1[0], shear3[0], shear4[0]],
                               [shear2[0], shear4[0], shear5[0]]
                               ], tf.float32)
    shearing_matrix = tf.linalg.expm(shearing_matrix)
    rot = tf.cast(rot, tf.float32)

    #COMPOSE MATRIX
    rot = tf.tensordot(rot, shearing_matrix, axes=1)
    t = tf.reshape(tf.concat([t1, t2, t3], axis=0), [3, 1])
    affine = tf.concat([rot, t], 1)
    return affine

def losses_per_batch_RMSE_orig(y_true_batch,y_pred_batch):


    y_pred_flatten=tf.reshape(y_pred_batch,[-1,1])
    y_true_flatten=tf.reshape(y_true_batch,[-1,1])
    # ONLY PIXELS WHERE
    indices_nn_null_pred=tf.where(tf.math.greater(y_pred_flatten,-3))

    y_pred_filtered1=tf.gather_nd(y_pred_flatten,indices_nn_null_pred)
    y_true_filtered1=tf.gather_nd(y_true_flatten,indices_nn_null_pred)

    #
    indices_neg_pred=tf.where(tf.math.greater(y_true_filtered1,-3))
    RMSE=tf.reduce_sum((y_true_filtered1-y_pred_filtered1)**2)


    return RMSE/(8**3)

# Postprocessing

def mean_and_std_neighbors(key_batch, graph, t_mean, t_std, A_mean, A_std):
    idx_neighbors = [*graph.neighbors(key_batch)]
    tn_neighbors = np.array(
        [nx.get_node_attributes(graph, 'translation_from_Pn')[idx_n] for idx_n in idx_neighbors])

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
    #    An_mean = None
    #    An_std = None
    return t_mean_neigbors, tn_std, An_mean, An_std

def save_values(saving_values, translation_mean, translation_std, thetas_mean, thetas_std,  Output_GRAPH):
    MSE_all = [nx.get_node_attributes(Output_GRAPH, 'MSE_or')[key]
               for key in nx.get_node_attributes(Output_GRAPH, 'MSE').keys()]
    loss_all = [nx.get_node_attributes(Output_GRAPH, 'loss_reg')[key]
                for key in nx.get_node_attributes(Output_GRAPH, 'loss_reg').keys()]
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


def update_outlier(Graph, mean_glob, std_glob, z_i, update_outlier=False, z_score_lim=.99):
    ls_outlier = []

    for idx in nx.get_node_attributes(Graph, 'translation_from_Pn').keys():
        # idx_neighbors=[*Graph.neighbors(idx)]
        idx_neighbors = np.unique(
            np.concatenate([[*Output_GRAPH.neighbors(idx_n)] for idx_n in [*Graph.neighbors(idx)]]))
        ls_t_neigbours = np.array(
            [nx.get_node_attributes(Graph, 'translation_from_Pn')[i] for i in idx_neighbors])
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
                Graph.add_node(idx, Qn=Pn_idx_cop, translation_from_Pn=test_t)

    # except:
    #  pass
    return Graph, ls_outlier


def estimate_variances_and_means_multivariateStudent_Unique_NablaQ(Graph):
    '''
    Bayesian estimation of the distributions of :
     - T : along z,y and x using a multivariate t-student distribution
     - An : For the 9 parameters using a multivariate t-student distribution
    Weighting based on the mahalanobis distance (MD) : small MD should be lessed penalized

    '''
    # for translation
    dist = np.reshape(np.stack(
        [nx.get_node_attributes(Graph, 'translation_from_Pn')[key] for key in range(Graph.number_of_nodes())]), [-1, 3])
    pars = np.reshape(np.stack(
        [nx.get_node_attributes(Graph, 'NablaQ')[key] for key in range(Graph.number_of_nodes())]), [-1, 1])

    global_pars = np.concatenate([dist, pars], axis=1)
    mix_t = smm.SMM(n_components=1, covariance_type='full', n_iter=1000, n_init=1, params='wmcd', init_params='wmcd')
    mix_t.fit(global_pars)
    global_pars_mean, global_pars_cov, global_pars_dof = mix_t.means, mix_t.covariances[0], mix_t.degrees[0]

    mahalanobis_distance = (np.diag(
        (np.dot((np.dot((global_pars - global_pars_mean), np.linalg.inv(global_pars_cov))),
                (global_pars - global_pars_mean).T)))) ** 0.5

    z = chi2.cdf(mahalanobis_distance, 3)  # or global_pars_dof voir avec hervé

    tn_mean = global_pars_mean[0][0:3]
    tn_std = global_pars_cov.diagonal()[0:3] ** 0.5
    An_mean = global_pars_mean[0][-1]
    # print(global_pars_mean)
    An_std = global_pars_cov.diagonal()[-1] ** 0.5
    in_var = np.mean([nx.get_node_attributes(Graph, 'MSE_or')[_] for _ in range(Graph.number_of_nodes())])
    return tn_mean, tn_std, An_mean, An_std, z, global_pars_cov, in_var, mahalanobis_distance

def estimate_variances_and_means_multivariateGaussian_Unique_nablaQ(Graph):
    '''
    Bayesian estimation of the distributions of :
     - T : along z,y and x using a multivariate t-student distribution
     - An : For the 9 parameters using a multivariate t-student distribution
    Weighting based on the mahalanobis distance (MD) : small MD should be lessed penalized

    '''
    # for translation
    dist = np.stack(
        [nx.get_node_attributes(Graph, 'translation_from_Pn')[key] for key in range(Graph.number_of_nodes())])
    pars = np.reshape(np.stack(
        [nx.get_node_attributes(Graph, 'NablaQ')[key] for key in range(Graph.number_of_nodes())]), [-1, 1])

    global_pars = np.concatenate([dist, pars], axis=1)

    pars_means = np.mean(global_pars, axis=0)
    pars_cov = np.cov(global_pars, rowvar=False)

    mahalanobis_distance = (np.diag(
        (np.dot((np.dot((global_pars - pars_means), np.linalg.inv(pars_cov))), (global_pars - pars_means).T)))) ** 0.5
    z = chi2.cdf(mahalanobis_distance, 3)

    tn_mean = pars_means[0:3]

    tn_std = pars_cov.diagonal()[0:3] ** 0.5

    An_mean = global_pars_cov[0][3:]
    An_std = pars_cov.diagonal()[3:] ** 0.5

    return tn_mean, tn_std, An_mean, An_std, z, pars_cov, In_var, mahalanobis_distance

# Q STEP
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
    # desc='3_4'
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

def minimize_energy(graph, grid, lambda_lame, shear_modulus, sigma_tn, lambda_bulk, An_mean, sigma_An,max_iter=500):
    '''
    Graph :
    grid :


    return Graph : with Qn updated
    '''
    S_sparse = pyvistamesh_global_stiffness(grid, lambda_lame, shear_modulus)
    # S_dense = scipy.sparse.csr_matrix.todense(S_sparse)
    Dmat = calc_D_mats(grid)  # shape N_tetrahedron x 4 x 3

    def f_to_minimize(x, *args):
        '''
        x = vector [ux1,ux2... uxn, vx1....]
        tn = translation vector predicted
        sigma = standard deviation

        '''
        # global_stif, tn, sigm, lambd = args[0], args[1], args[2], args[3]
        global_stif, tn, sigm_tn, An, sigm_An, lambd, grid, ls_hinv = args[0], args[1], args[2], args[3], args[4], args[
            5], args[6], args[7]
        cells = grid.cells.reshape(-1, 5)[:, 1:]

        P = grid.points

        lambdaW = 1 / 2 * np.sum((global_stif.dot(x)).dot(np.matrix(x).T))
        # lambdaW = 1 / 2 * np.sum(np.dot(x, global_stif) * (np.matrix(x).T))
        tn_reshape = tn.reshape([3, -1]).T
        x_reshape = x.reshape([3, -1]).T
        Qn_new = P + x_reshape
        Qn = P + tn_reshape
        #
        tet_new = Qn_new[cells]
        tet_old = Qn[cells]
        #
        reg_an = np.sum(
            (np.sum(np.multiply(tet_new, ls_hinv), axis=0) - np.sum(np.multiply(tet_old, ls_hinv), axis=0)) ** 2) / (
                             sigm_An ** 2)

        reg_tn = np.sum((tn_reshape - x_reshape) ** 2) / (np.array(sigm_tn) ** 2)
        lambdaW = lambd * lambdaW
        #print('W : {} ; tn : {} ; An : {} '.format(lambdaW,reg_tn,reg_an) )
        min_func = lambdaW + reg_tn + reg_an

        return min_func

    u_Pns = np.array([nx.get_node_attributes(graph, 'translation_from_Pn')[_]
                      for _ in range(grid.points.shape[0])])
    Pns = np.array([nx.get_node_attributes(graph, 'Pn')[_]
                    for _ in range(grid.points.shape[0])])

    u_Pns_lin = (u_Pns.T).reshape([-1])
    # Pns_lin = (Pns.T).reshape([-1])
    x_to_optimize = np.copy(u_Pns_lin)

    # res = fmin_cg(f_to_minimize, x_to_optimize,
    #              args=(S_sparse, u_Pns_lin, sigma_tn, lambda_bulk), maxiter=300, retall=False)
    res = fmin_cg(f_to_minimize, x_to_optimize,
                  args=(S_sparse, u_Pns_lin, sigma_tn, An_mean, sigma_An, lambda_bulk, grid, Dmat), maxiter=max_iter,
                  retall=False)

    new_d = res.reshape([3, -1]).T
    new_Qn = Pns
    new_Qn[:, 1:-1] += new_d

    diff = new_d - u_Pns

    for i, idx in enumerate(nx.get_node_attributes(graph, 'Pn').keys()):
        graph.add_node(i, Qn=new_Qn[i, :], translation_from_Pn=new_d[i, :])

    return graph, new_d

# RAY NETWORK MULTIPROCESS
class Network(object):

    def __init__(self, x, y, dict_start=None, dict_reg=None):

        # Seed TensorFlow to make the script deterministic.
        tf.set_random_seed(0)

        self.idx_vertex = tf.constant(dict_reg['idx_vertex'], tf.int16)

        self.pn = tf.constant(dict_reg['Pn'], tf.float32)
        self.qn = tf.constant(dict_reg['Qn'], tf.float32)
        self.Wass = tf.constant(dict_reg['Wass'], tf.float32)

        self.val_init_thetas = dict_start['rotation_init']
        self.val_init_shears = dict_start['shearing_init']
        self.val_init_t = dict_start['translation_init']

        # Define the inputs.
        self.x_data = tf.constant(x, dtype=tf.float32)
        self.y_data = tf.constant(y, dtype=tf.float32)

        # Define the weights and computation.

        theta0 = tf.Variable(initial_value=tf.reshape(self.val_init_thetas[0], [1]), dtype='float32', trainable=True,
                             constraint=tf.keras.constraints.MinMaxNorm(-THETA_MIN, +THETA_MIN),
                             name='theta0')

        theta1 = tf.Variable(initial_value=tf.reshape(self.val_init_thetas[1], [1]), dtype='float32', trainable=True,
                             constraint=tf.keras.constraints.MinMaxNorm(-THETA_MIN, +THETA_MIN),
                             name='theta1')

        theta2 = tf.Variable(initial_value=tf.reshape(self.val_init_thetas[2], [1]), dtype='float32', trainable=True,
                             constraint=tf.keras.constraints.MinMaxNorm(-THETA_MIN, +THETA_MIN),
                             name='theta2')

        shear0 = tf.Variable(initial_value=tf.reshape(self.val_init_shears[0], [1]), dtype='float32', trainable=True,
                             constraint=tf.keras.constraints.MinMaxNorm(-SHEAR_MIN, SHEAR_MIN),
                             name='shear0')
        shear1 = tf.Variable(initial_value=tf.reshape(self.val_init_shears[1], [1]), dtype='float32', trainable=True,
                             constraint=tf.keras.constraints.MinMaxNorm(-SHEAR_MIN, SHEAR_MIN),
                             name='shear1')
        shear2 = tf.Variable(initial_value=tf.reshape(self.val_init_shears[2], [1]), dtype='float32', trainable=True,
                             constraint=tf.keras.constraints.MinMaxNorm(-SHEAR_MIN, SHEAR_MIN),
                             name='shear2')
        shear3 = tf.Variable(initial_value=tf.reshape(self.val_init_shears[3], [1]), dtype='float32', trainable=True,
                             constraint=tf.keras.constraints.MinMaxNorm(-SHEAR_MIN, SHEAR_MIN),
                             name='shear3')
        shear4 = tf.Variable(initial_value=tf.reshape(self.val_init_shears[4], [1]), dtype='float32', trainable=True,
                             constraint=tf.keras.constraints.MinMaxNorm(-SHEAR_MIN, SHEAR_MIN),
                             name='shear4')
        shear5 = tf.Variable(initial_value=tf.reshape(self.val_init_shears[5], [1]), dtype='float32', trainable=True,
                             constraint=tf.keras.constraints.MinMaxNorm(-SHEAR_MIN, SHEAR_MIN),
                             name='shear5')

        translation1 = tf.Variable(initial_value=tf.reshape(self.val_init_t[0], [1]), dtype='float32', trainable=True,
                                   constraint=tf.keras.constraints.MinMaxNorm(-TRANSLATION_MIN, TRANSLATION_MIN),
                                   name='translation1')
        translation2 = tf.Variable(initial_value=tf.reshape(self.val_init_t[1], [1]), dtype='float32', trainable=True,
                                   constraint=tf.keras.constraints.MinMaxNorm(-TRANSLATION_MIN, TRANSLATION_MIN),
                                   name='translation2')
        translation3 = tf.Variable(initial_value=tf.reshape(self.val_init_t[2], [1]), dtype='float32', trainable=True,
                                   constraint=tf.keras.constraints.MinMaxNorm(-TRANSLATION_MIN, TRANSLATION_MIN),
                                   name='translation3')

        self.matrice = Affine_Matrix_from_outputs(
            [theta0, theta1, theta2, shear0, shear1, shear2, shear3, shear4, shear5,
             translation1, translation2, translation3])
        translations_from_Qn, self.pars_batch = extract_par_from_outputs(
            [theta0, theta1, theta2, shear0, shear1, shear2, shear3, shear4, shear5,
             translation1, translation2, translation3])
        y_pred = batch_affine_warp3d(self.x_data, self.matrice)

        # Define the loss.

        self.MSE = tf.reduce_sum(losses_per_batch_RMSE_orig(y_true_batch=self.y_data, y_pred_batch=y_pred))
        self.mse_std = tf.divide(self.MSE, tf.pow(dict_reg['In_var'], 1))

        # REGULARIZATION
        T_Qn = tf.constant((np.array(dict_reg['Qn']) - np.array(dict_reg['Pn']))[1:-1], tf.float32)
        self.qn_pn = tf.reshape(T_Qn, [1, 3, -1]) * tf.reshape(tf.constant(dict_reg['spacing_pyramid'], tf.float32),
                                                               [1, 3, -1])

        self.translation_from_Qn = tf.multiply(translations_from_Qn,
                                               tf.constant((dict_reg['PATCH_SIZE_QN'] / 2), tf.float32)) * tf.reshape(
            tf.constant(dict_reg['spacing_pyramid'], tf.float32), [1, 3, -1])

        self.translation_from_Pn = self.qn_pn - self.translation_from_Qn

        # NablaQ
        dmat = tf.constant(dict_reg['Dmat'], tf.float32)
        Qns = tf.constant(dict_reg['Qns'], tf.float64)

        updates = []
        for idx, pos in enumerate(dict_reg['cell_i']):
            for iter_coord in range(3):
                updates.append([idx, pos, iter_coord])
        self.updates_tf = tf.constant(np.array(updates), tf.int64)

        zero_tf = tf.zeros_like(Qns)

        translation_from_Qn_repeated = tf.tile(tf.reshape(self.translation_from_Qn, [3]), [idx + 1])

        self.translation_from_Qn_repeated = tf.cast(translation_from_Qn_repeated, tf.float64)

        zero_updated = tf.tensor_scatter_update(zero_tf, self.updates_tf, self.translation_from_Qn_repeated)
        Qns_updated = tf.cast(Qns - zero_updated, tf.float32)
        self.nablaQ = tf.reduce_mean(tf.einsum('ijk,ijk->ij', Qns_updated, dmat))
        self.sp = calc_sp(self.translation_from_Pn, dict_reg['Di']) / 3
        if (dict_reg['epoch'] > 0) | (dict_reg['step'] > 0):
            tmean = tf.reshape(tf.constant(dict_reg['t_mean_n']), [1, 3, -1])
            tstd = tf.reshape(tf.constant(dict_reg['t_std_n']), [1, 3, -1])
            self.reg_tn = tf.reduce_mean(tf.divide(tf.pow((self.translation_from_Pn - tmean), 2), tf.pow(tstd, 2)))
            Amean = tf.constant(dict_reg['t_mean_n'])
            Astd = tf.constant(dict_reg['t_std_n'])
            self.reg_An = tf.reduce_mean(tf.divide(tf.pow((self.nablaQ - Amean), 2), tf.pow(Astd, 2)))

        else:
            self.reg_tn = tf.constant(0., tf.float32)
            self.reg_An = tf.constant(0., tf.float32)

        self.loss = self.mse_std + self.sp + self.reg_tn + self.reg_An  # + self.sp #+ sp #+ reg_tn + reg_An

        # optimizer = tf.train.GradientDescentOptimizer(LR)
        optimizer = tf.train.MomentumOptimizer(learning_rate=LR, momentum=0.1, use_nesterov=True)
        self.grads = optimizer.compute_gradients(self.loss)

        clipping_value = .3
        capped_gvs = [(tf.clip_by_value(grad, -clipping_value, clipping_value), var) for grad, var in self.grads]
        self.train = optimizer.apply_gradients(capped_gvs)
        self.grad = capped_gvs

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.variables = ray.experimental.tf_utils.TensorFlowVariables(self.loss, self.sess)
        self.sess.run(init)

    # Define a remote function that trains the network for one step and returns the
    # new weights.
    def step(self, iterations):
        # Set the weights in the network.

        # Do one step of training.
        ls_loss = []
        for _ in range(iterations):
            self.sess.run(self.train)
            weights = self.variables.get_weights()
            self.variables.set_weights(weights)
            result_to_update = self.sess.run([self.train,
                                              self.loss,
                                              self.MSE,
                                              self.translation_from_Pn,
                                              self.translation_from_Qn,
                                              self.qn_pn,
                                              self.pars_batch,
                                              self.matrice,
                                              self.pn,
                                              self.qn,
                                              self.Wass,
                                              self.sp,
                                              self.reg_tn,
                                              self.idx_vertex,
                                              self.grad,
                                              self.nablaQ,
                                              self.reg_An,
                                              self.mse_std,
                                              self.x_data,
                                              self.y_data])
            ls_loss.append(result_to_update[1])
            # print(str(_) + 'idx {} ;  MSE = {:.4f} ; MSE_std = {:.4f}; ScalP = {:.2f}  reg_tn={:.4f} reg_An={:.4f}'.format(l[-5], l[2], l[-3],l[-9],l[-8],l[-4]))
            # print(str(_) +' PARS = ' + str(l[6]))

        return weights, result_to_update, ls_loss

    def get_weights(self):
        return self.variables.get_weights()


remote_network = ray.remote(Network)

# DATA LOADING AND PLOTTING

fixed_image_n_pad_nii = nib.load('data_preprocess/fixed_image_' + str(NUM_P) + '.nii.gz')
moving_image_n_pad_nii = nib.load('data_preprocess/moving_image_' + str(NUM_P) + '.nii.gz')
mask_padded_nii = nib.load('data_preprocess/moving_mask_' + str(NUM_P) + '.nii.gz')
fixed_image_n_pad = fixed_image_n_pad_nii.get_data()
mask_padded = mask_padded_nii.get_data()
moving_image_n_pad = moving_image_n_pad_nii.get_data()
SPACING_ORIG = np.diag(moving_image_n_pad_nii.affine)[0:-1]
coord_list, G, grid_mesh = build_mesh_from_3Dmask(mask_padded, DIS_TETRAHEDRON, SPACING_ORIG, surface=False)

while G.number_of_nodes() < MIN_NUM_TETRA:
    DIS_TETRAHEDRON-=5
    coord_list, G, grid_mesh = build_mesh_from_3Dmask(mask_padded, DIS_TETRAHEDRON, SPACING_ORIG, surface=False)

NAME_DIR_SAVE = 'EMPIRE10PATIENT' + str(NUM_P) + 'PYR' + str(PYRAMID_LEVELS[-1]) + 'NODES' + str(G.number_of_nodes()) +\
                'P' + str(PATCH_SIZE_Pn) + "Q" + str(PATCH_SIZE_Qn) + 'E' + str(NUM_EPOCH) + "S" + str(NUM_ITERS) +\
                'regtns_regAn_noQstep_conditionedROT/'

try:
    os.makedirs('./results/' + NAME_DIR_SAVE)
except:
    pass
grid_mesh.save('./results/' + NAME_DIR_SAVE + 'grid.vtk')


# a reference grid for substracting later
grid_patch_coordinates=batch_mgrid(1,PATCH_SIZE_Qn,PATCH_SIZE_Qn,PATCH_SIZE_Qn,low=-PATCH_SIZE_Qn//2+.5,high=PATCH_SIZE_Qn//2-.5)
grid_patch_coordinates_reshaped= tf.reshape(grid_patch_coordinates, [1, 3, -1])

saving_values={'MSE':[],"VAR_Tn":[],"VAR_An":[],"VAR_In":[], "LOSS":[],"PARAMETERS":[]}
translation_mean, translation_std, translation_cov, z_tn, An_mean, An_cov, z_An, params, In_var, z, An_std = None, None, None, None, None, None, None, None, 1, None, None
Output_GRAPH=G.copy()
list_nodes=list(np.arange(0,G.number_of_nodes()))

# TRAINING_LOOP

for idx_pyramid, pyramid_level in enumerate(PYRAMID_LEVELS):
    try:
        os.makedirs('./results/' + NAME_DIR_SAVE + str(pyramid_level))
    except:
        pass
    """
    ESTIMATE DIFFERENT PARAMETERS AT EACH CHANGE OF PYRAMID RESOLUTION
    
    """
    reducing_factor = 2**(pyramid_level - 1)

# Upsampling of the moving and fixed pyramid
    moving_pyramid_img_i = measure.block_reduce(moving_image_n_pad, (1, reducing_factor, reducing_factor, reducing_factor, 1), np.mean)
    fixed_pyramid_img_i = measure.block_reduce(fixed_image_n_pad, (1, reducing_factor, reducing_factor, reducing_factor, 1), np.mean)
    
# Pyramid Smoothing
    moving_pyramid_img = np.float32(scipy.ndimage.gaussian_filter(moving_pyramid_img_i, 1))
    fixed_pyramid_img = np.float32(scipy.ndimage.gaussian_filter(fixed_pyramid_img_i, 1))
    spacing_pyramid = (np.array(fixed_image_n_pad.shape) / moving_pyramid_img.shape)[1:-1] * SPACING_ORIG
    ones_spacing = np.ones(5)
    ones_spacing[1:-1] = np.float32(spacing_pyramid)
    xlen, ylen, zlen = np.int16(moving_pyramid_img.shape[1:-1])
    coords = (tf.linspace(0., _-1, _) for _ in (xlen, ylen, zlen))
    grid_indexes_no_tf = tf.stack(tf.meshgrid(*coords, indexing='ij'))

    for step in range(0, NUM_EPOCH):
        sparse_dict_deformation_field = {'Coords': [], 'Values': []}
        for iter_batch in range(len(list_nodes) // N_BATCH + np.int16(np.ceil((len(list_nodes) % N_BATCH) / N_BATCH))):
            print(iter_batch)
            list_keys_batches = list_nodes[iter_batch * N_BATCH:(iter_batch + 1) * N_BATCH]

            len_batch = len(list_keys_batches)
            ls_dict_init = []
            for idx_p_in_batch in range(len_batch):
                if (step == 0) & (idx_pyramid == 0):
                    val_init_rotation = [1e-9, 1e-9, 1e-9]  # Rodrigues errors at 0.
                    val_init_shears = [0., 0., 0., 0., 0., 0.]
                else:
                    val_init_rotation = [par for par in nx.get_node_attributes(Output_GRAPH, 'pars')[
                        list_keys_batches[idx_p_in_batch]]][0:3]
                    val_init_shears = [par for par in
                                       nx.get_node_attributes(Output_GRAPH, 'pars')[list_keys_batches[idx_p_in_batch]]][
                                      3:]
                ls_dict_init.append({'translation_init': np.float32([.0, .0, .0]),
                                     'rotation_init': np.reshape(np.float32(val_init_rotation), [-1]),
                                     'shearing_init': np.reshape(np.float32(val_init_shears), [-1])})

            Pn_batch_s, Qn_batch_s = get_Pn_Qn_batch(Output_GRAPH, list_keys_batches)
            Pn_batch, Qn_batch = np.int16(np.round(Pn_batch_s / ones_spacing)), np.int16(
                np.round(Qn_batch_s / ones_spacing))  # get Pn_batch & Qn_batch in pixel
            # Extract Patches around Pn and Qn
            patches_moving = extract_patch_from_coord_list(moving_pyramid_img, Pn_batch, patch_size=PATCH_SIZE_Pn)
            patches_fixed_larger = extract_patch_from_coord_list(fixed_pyramid_img, Qn_batch, patch_size=PATCH_SIZE_Qn)

            patches_moving_padded = pad_patches_fixed(patches_moving, PATCH_SIZE_Qn=PATCH_SIZE_Qn,
                                                      PATCH_SIZE_Pn=PATCH_SIZE_Pn)

            Wass_batch = [wasserstein_distance(np.ravel(patches_moving[_]),
                                               np.ravel(patches_fixed_larger[_][0, 4:-4, 4:-4, 4:-4, 0])) for _ in
                          range(len_batch)]
            Qns = np.array([nx.get_node_attributes(Output_GRAPH, 'Qn')[key][1:-1] for key in
                            range(Output_GRAPH.number_of_nodes())])

            ls_Dmat, ls_Di, ls_cell, ls_Qns_i = [], [], [], []
            ls_mean_t, ls_std_t, ls_mean_A, ls_std_A = [], [], [], []
            for i in range(len_batch):
                # Shape Vectors
                Dmat, Di, cell_i, cells = D_idx([list_keys_batches[i]], grid_mesh)
                Qns_i = Qns[cells]
                ls_Dmat.append(Dmat), ls_Di.append(Di), ls_cell.append(cell_i), ls_Qns_i.append(Qns_i)

                if step > 0:
                    ls_mean_t_i, ls_std_t_i, ls_mean_A_i, ls_std_A_i = mean_and_std_neighbors(list_keys_batches[i],
                                                                                              Output_GRAPH, tn_mean,
                                                                                              tn_std, An_mean, An_std)
                else:
                    ls_mean_t_i, ls_std_t_i, ls_mean_A_i, ls_std_A_i = None, None, None, None
                ls_mean_t.append(ls_mean_t_i), ls_std_t.append(ls_std_t_i)
                ls_mean_A.append(ls_mean_A_i), ls_std_A.append(ls_std_A_i)

            ls_dict_regularization = [{'idx_vertex': np.int16(list_keys_batches[_]),
                                       'step': np.int16(step),
                                       "epoch" : np.int16(idx_pyramid),
                                       'PATCH_SIZE_QN': np.int16(PATCH_SIZE_Qn),
                                       'SPACING_ORIG': np.float32(SPACING_ORIG),
                                       'Pn': np.float32(Pn_batch[_]),
                                       'Qn': np.float32(Qn_batch[_]),
                                       'spacing_pyramid': np.float32(spacing_pyramid),
                                       'Wass': np.float32(Wass_batch[_]), 'In_var': np.float32(In_var),
                                       'Dmat': np.float32(ls_Dmat[_]),
                                       'Di': np.float32(ls_Di[_]),
                                       'cell_i': np.int16(ls_cell[_]),
                                       't_mean_n': ls_mean_t[_], 't_std_n': ls_std_t[_],
                                       'A_mean_n': ls_mean_A[_], 'A_std_n': ls_std_A[_],
                                       'Qns': ls_Qns_i[_]}
                                      for _ in range(len_batch)]

            # put variables on ray
            y_ids = [ray.put(patches_fixed_larger[i]) for i in range(len_batch)]
            x_ids = [ray.put(patches_moving_padded[i]) for i in range(len_batch)]
            network_inits_ids = [ray.put(ls_dict_init[i]) for i in range(len_batch)]
            regularization_ids = [ray.put(ls_dict_regularization[i]) for i in range(len_batch)]

            # Create actors to store the networks.
            actor_list = [remote_network.remote(x_ids[i], y_ids[i], network_inits_ids[i], regularization_ids[i]) for i
                          in range(len_batch)]

            new_weights_ids = [actor.step.remote(iterations=NUM_ITERS) for actor in actor_list]
            results = ray.get(new_weights_ids)
            #   weights=[result[0] for result in results]

            # clear_output()
            for result in results:
                weights, ls_res, ls_loss = result[0], result[1], result[2]

                train_idx, loss_idx, MSE_idx, translation_from_Pn_idx, \
                translation_from_Qn_idx, qn_pn_idx, \
                pars_batch_idx, matrice_idx, Pn_batch_idx, Qn_batch_idx, \
                wass_idx, sp_idx, regtn_idx, idx_v, grads_idx, NablaQ, regAn_idx, mse_std_idx, x_idx, y_idx = ls_res

                Qn_updated = np.pad(Qn_batch_idx[1:4] * spacing_pyramid - translation_from_Qn_idx[0, :, 0],
                                    pad_width=[1, 1], mode='constant', constant_values=0)
                matrice_updated = np.copy(matrice_idx)
                matrice_updated[:, -1] = 0
                Output_GRAPH.add_node(idx_v,
                                      # Pn = Pn_updated,
                                      nn_outpouts=weights,
                                      affines=matrice_updated,
                                      affines_or=matrice_idx,
                                      NablaQ=NablaQ,
                                      MSE=MSE_idx,
                                      MSE_or=MSE_idx,
                                      Qn=Qn_updated,
                                      # An=nabla_Q.numpy(),
                                      loss_reg=loss_idx,
                                      ls_loss=ls_loss,
                                      translation_from_Qn=translation_from_Qn_idx[0, :, 0],
                                      Qn_Pn=qn_pn_idx,
                                      translation_from_Pn=translation_from_Pn_idx[0, :, 0],
                                      sp=sp_idx,
                                      Wass=wass_idx,
                                      pars=pars_batch_idx[0, :, 0])
                print(('EPOCH = {}, idx = {} , MSE = {:.4f} ,MSE_std = {:.4f}, regTn = {:.3f}' + ', regAn= ' + str(
                    regAn_idx) + 'regSp = ' + str(sp_idx)).format(step, idx_v, MSE_idx, mse_std_idx, regtn_idx))

        tn_mean, tn_std, An_mean, An_std, z, pars_cov, In_var, Maha = estimate_variances_and_means_multivariateStudent_Unique_NablaQ(
            Output_GRAPH)

        if step != (NUM_EPOCH-1):
            outliers = np.where(1 - z > 1- PVAL_OUTLIER)[0]
            [Output_GRAPH.add_node(idx_v, MSE=1) for idx_v in outliers]
            Output_GRAPH,outliers = update_outlier(Output_GRAPH, tn_mean, tn_std, z, update_outlier=True,z_score_lim=1-PVAL_OUTLIER)
            nx.write_gpickle(Output_GRAPH,'./results/' + NAME_DIR_SAVE+str(pyramid_level) + '/Output_GRAPH'+str(step))

            print(outliers)
            if Q_STEP:
                Output_GRAPH,u_opt=minimize_energy(Output_GRAPH, grid_mesh, lame_lambda, shear_modulus, np.mean(tn_std),
                    lr_Q_STEP, An_mean, An_std)
                nx.write_gpickle(Output_GRAPH, './results/' +'Output_GRAPH_after_Qstep')

        tn_mean, tn_std, An_mean, An_std, z, pars_cov, _, Maha = estimate_variances_and_means_multivariateStudent_Unique_NablaQ(Output_GRAPH)

        saving_values=save_values(saving_values, tn_mean, tn_std,
                              An_mean, An_std, Output_GRAPH)
        np.save('./results/' + NAME_DIR_SAVE + str(pyramid_level) + '/saving_values.npy', saving_values)


