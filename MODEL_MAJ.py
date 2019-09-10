import os
import glob
import sys
import random
import shutil
import ray
ray.init()
import ray.experimental.tf_utils
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


import scipy
from scipy.optimize import fmin_cg 
from scipy.stats import chi2
from scipy.stats import wasserstein_distance
import skimage
from tensorflow.keras import layers
import pyvista as polyv
from IPython.display import clear_output

# stats
import math
import smm     ## student mixture model
import nibabel as nib

sys.path.append('./funcs_final/')
import funcs_final
from funcs_final.preprocessing import *
from funcs_final.tetraedra_tools import *
from funcs_final.postprocessing import *

#sys.path.append('./utils/pyOptFEM/pyOptFEM/')

#from pyOptFEM import *
#from pyOptFEM.FEM3D.assembly import *
#from tetraedra_tools import *
#from funcs import *
#from model_funcs import *
try:
    sys.path.append('./utils/pyOptFEM/')
    from pyOptFEM.FEM3D.assembly import *
    from pyOptFEM import *
except:
    sys.path.append('./utils/pyOptFEM/pyOptFEM/')
    from pyOptFEM import *
    from pyOptFEM.pyOptFEM.FEM3D.assembly import *


# # HYPERPARAMETERS


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'Begin Registration')
    parser.add_argument('--num_p', required = True, type = str,
                        help='NUM_P')
    parser.add_argument('--NUM_EPOCH', default = 4, required=False)
    parser.add_argument('--lambdaW', default = 1e-3, required=False)

    args = parser.parse_args()

    NUM_P = np.int16(args.num_p)
    NUM_P = str('{:02d}'.format(NUM_P))
    LAMBDA_W = np.float32(args.lambdaW)
    NUM_EPOCH = np.int16(args.NUM_EPOCH)

#PATH_PATIENTS = '/home/pblancdu/projets/lung_registration/patients/'
PATH_PATIENTS = './patients/'
#NUM_P = '21'
PATCH_SIZE_Qn = 15   ## must be less than 16
PATCH_SIZE_Pn = 7
NU = 0.3                          #(0< NU <0.5)
YOUNG_MODULUS = 1
N_BATCH=12
THETA_MIN=math.pi/10
SHEAR_MIN=.1
TRANSLATION_MIN=.7
#NUM_EPOCH=10
NUM_ITERS=30    ###
PYRAMID_LEVELS=[4,3,2,1]
LR = 1e-1
PVAL_OUTLIER = 0.05

lame_lambda = YOUNG_MODULUS*NU /  ((1 + NU)*(1-2*NU))
shear_modulus =  YOUNG_MODULUS / (2* ( 1 +  NU))



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

# ## Model Funcs

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

    ##COMPOSE MATRIX
    rot = tf.tensordot(rot, shearing_matrix, axes=1)
    t = tf.reshape(tf.concat([t1, t2, t3], axis=0), [3, 1])
    affine = tf.concat([rot, t], 1)
    return affine

def losses_per_batch_RMSE_orig(y_true_batch,y_pred_batch):
    y_pred_flatten=tf.reshape(y_pred_batch,[-1,1])
    y_true_flatten=tf.reshape(y_true_batch,[-1,1])
    indices_nn_null_pred=tf.where(tf.math.greater(y_pred_flatten,-3))
    y_pred_filtered1=tf.gather_nd(y_pred_flatten,indices_nn_null_pred)
    y_true_filtered1=tf.gather_nd(y_true_flatten,indices_nn_null_pred)
    #indices_neg_pred=tf.where(tf.math.greater(y_true_filtered1,-3))
    RMSE=tf.reduce_sum((y_true_filtered1-y_pred_filtered1)**2)
    return RMSE


# # RAY NETWORK MULTIPROCESS


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
        self.affine = tf.slice(self.matrice, [0, 0], [3, 3])
        
        ## modified Qn by Pn
        T_Pn, self.pars_batch = extract_par_from_outputs(
            [theta0, theta1, theta2, shear0, shear1, shear2, shear3, shear4, shear5,
             translation1, translation2, translation3])
        y_pred = batch_affine_warp3d(self.x_data, self.matrice)

        # Define the loss.

        self.MSE = tf.reduce_sum(losses_per_batch_RMSE_orig(y_true_batch=self.y_data, y_pred_batch=y_pred))
        self.mse_std = self.MSE / tf.constant(dict_reg['var_in'],tf.float32)

        # REGULARIZATION
        # takes into account the earlier displacement
        Qn_Pn_pixel_space = tf.constant(np.array(dict_reg['Qn'][1:-1]),tf.float32) - tf.constant(np.array(dict_reg['Pn'])[1:-1], tf.float32)
        self.QN_PN_image_space = tf.reshape(Qn_Pn_pixel_space, [1, 3, -1]) * tf.reshape(tf.constant(dict_reg['spacing_pyramid'], tf.float32),
                                                               [1, 3, -1])

        
        # the one estimated here -> the next Qn
        self.translation_from_Pn = tf.multiply(T_Pn,
                                               tf.constant((dict_reg['PATCH_SIZE_QN'] / 2), tf.float32)) * \
                                               tf.reshape(tf.constant(dict_reg['spacing_pyramid'], tf.float32), [1, 3, -1])

        # translation from Qn to next Qn
        self.ACCUMULATED_TRANSLATION = self.QN_PN_image_space  - self.translation_from_Pn

        # NablaQ
        self.nablaQ = tf.constant(dict_reg['NablaQ'], tf.float32)


        #self.sp = calc_sp(self.translation_from_Pn, dict_reg['Di']) / 3
        self.sp = tf.constant(0.,tf.float32)
        if (dict_reg['epoch'] > 0) | (dict_reg['step'] > 0):
            tmean = tf.reshape(tf.constant(dict_reg['t_mean_n'],tf.float32), [1, 3, -1])
            tstd = tf.reshape(tf.constant(dict_reg['t_std_n'],tf.float32), [1, 3, -1])
            self.reg_tn = tf.reduce_sum(tf.divide(tf.pow((self.ACCUMULATED_TRANSLATION - tmean), 2), tf.pow(tstd, 2)))

            # Amean = tf.constant(dict_reg['A_mean_n'])
            Astd = tf.constant(dict_reg['A_std_n'])
            self.reg_An = tf.reduce_sum(tf.divide(tf.pow((self.affine - self.nablaQ), 2), tf.pow(Astd, 2)))

        else:
            self.reg_tn = tf.constant(0., tf.float32)
            self.reg_An = tf.constant(0., tf.float32)

        self.loss = self.mse_std + 2*(self.reg_tn + self.reg_An)  # + self.sp #+ sp #+ reg_tn + reg_An

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
                                              self.ACCUMULATED_TRANSLATION,
                                              self.QN_PN_image_space,
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
            print(str(_) + 'idx {} ;  MSE = {:.4f} ; MSE_std = {:.4f}; ScalP = {:.2f}  reg_tn={:.4f} reg_An={:.4f}'.format(result_to_update[-7], result_to_update[2], result_to_update[-3],result_to_update[-9],result_to_update[-8],result_to_update[-4]))
            # print(str(_) +' PARS = ' + str(l[6]))

        return weights, result_to_update, ls_loss

    def get_weights(self):
        return self.variables.get_weights()
remote_network = ray.remote(Network)

# # DATA LOADING AND PLOTTING
# LOAD IMAGE, stiffs
fixed_image_n_pad_nii = nib.load('data_preprocess/fixed_image_'+str(NUM_P)+'.nii.gz')
moving_image_n_pad_nii = nib.load('data_preprocess/moving_image_'+str(NUM_P)+'.nii.gz')
moving_mask_padded_nii = nib.load('data_preprocess/moving_mask_'+str(NUM_P)+'.nii.gz')
fixed_mask_padded_nii = nib.load('data_preprocess/fixed_mask_'+str(NUM_P)+'.nii.gz')
mask_padded_I = fixed_mask_padded_nii.get_data()
mask_padded_J = moving_mask_padded_nii.get_data()
image_I_n_pad = fixed_image_n_pad_nii.get_data()
image_J_n_pad = moving_image_n_pad_nii.get_data()
SPACING_ORIG = np.diag(fixed_image_n_pad_nii.affine)[:-1]
image_I_n_pad[mask_padded_I[np.newaxis,...][...,np.newaxis]==0] = 0
image_J_n_pad[mask_padded_J[np.newaxis,...][...,np.newaxis]!=1] = 0
grid_mesh = polyv.read(PATH_PATIENTS+NUM_P+'/grid.vtk')
Output_GRAPH = generate_graph_from_mesh(grid_mesh)
n_vertices = Output_GRAPH.number_of_nodes()
nx.write_gpickle(Output_GRAPH,'./patients/'+NUM_P+'/Output_GRAPH_initial')
K = np.load(PATH_PATIENTS+NUM_P+'/stiffs_stacked.npy')
Km = K[0,...]
Kt = K[1,...]
Ka = K[2,...]

print('NUM_P {},N_VERTICES {}'.format(NUM_P, grid_mesh.number_of_points ))
#
saving_values={'MSE':[],"VAR_Tn":[],"VAR_An":[],"VAR_In":[], "LOSS":[],"PARAMETERS":[]}
list_nodes=list(np.arange(0,Output_GRAPH.number_of_nodes()))
t, std_t = None, 1
A, std_A = None, 1

# TRAINING_LOOP

def extract_patchexact_from_coord_list(np_array, c_list, keys, patch_size,vox_size):
    '''
    c_list : a list of coordinates
    patch_size : the size of the patch
    '''
    ls_patch = []
    idx_in_batch_kept = []
    dict_patches = {}
    for i,coords_i in enumerate(c_list):
        key_batch = keys[i]
        coords_vox_space = (coords_i+1e-9) / vox_size
        coords_min = np.int16(np.floor(coords_vox_space))
        coords_max = np.int16(np.ceil(coords_vox_space))

        patch = np_array[:, coords_min[1] - (patch_size // 2 ) :coords_max[1] + (patch_size // 2 ),
                        coords_min[2] - (patch_size // 2 ):coords_max[2] + (patch_size // 2 ),
                        coords_min[3] - (patch_size // 2 ):coords_max[3] + (patch_size // 2), :]
        if (patch.shape == (1, patch_size, patch_size, patch_size, 1)) & (np.sum(patch)!=0):
            ls_patch.append(patch)
            idx_in_batch_kept.append(key_batch)
            dict_patches[key_batch] = patch
        else:
            print('ERROR PATCH EXTRACTION' +str(i))
            print(' SHAPE {}, SUM {}'.format(patch.shape,np.sum(patch)))
        
    #if len(idx_in_batch_kept) == 0:
    #    idx_in_batch_kept = None
    #    ls_patch = None
        
    return idx_in_batch_kept,dict_patches

for idx_pyramid, pyramid_level in enumerate(PYRAMID_LEVELS):
    try:
        os.makedirs('./patients/'+NUM_P+'/'+ str(pyramid_level))
    except:
        pass
    """
    ESTIMATE DIFFERENT PARAMETERS AT EACH CHANGE OF PYRAMID RESOLUTION

    """
    reducing_factor = 2 ** (pyramid_level - 1)

    # Upsampling of the moving and fixed pyramid
    image_I_pyramid = measure.block_reduce(image_I_n_pad,
                                                (1, reducing_factor, reducing_factor, reducing_factor, 1), np.mean)
    image_J_pyramid = measure.block_reduce(image_J_n_pad,
                                               (1, reducing_factor, reducing_factor, reducing_factor, 1), np.mean)

    # Pyramid Smoothing
    image_I_pyramid = np.float32(scipy.ndimage.gaussian_filter(image_I_pyramid, 1))
    image_J_pyramid = np.float32(scipy.ndimage.gaussian_filter(image_J_pyramid, 1))
    spacing_pyramid = (np.array(image_J_n_pad.shape) / image_I_pyramid.shape)[1:-1] * SPACING_ORIG
    ones_spacing = np.ones(5)
    ones_spacing[1:-1] = np.float32(spacing_pyramid)


    Pn_coords = np.array([nx.get_node_attributes(Output_GRAPH,'Pn')[key] for key in range(n_vertices)])
    Pn_coords_vox_space = Pn_coords / ones_spacing
    list_nodes,dict_patch = extract_patchexact_from_coord_list(image_I_pyramid, Pn_coords, np.arange(n_vertices), patch_size=PATCH_SIZE_Pn,vox_size = ones_spacing)
    ls_errors_batch = list(set(np.arange(n_vertices)) - set(list_nodes))


    for step in range(0,2):
        Qns = np.array([nx.get_node_attributes(Output_GRAPH, 'Qn')[key][1:-1] for key in
                        range(Output_GRAPH.number_of_nodes())])
        for iter_batch in tqdm.tqdm(range(0,len(list_nodes) // N_BATCH + np.int16(np.ceil((len(list_nodes) % N_BATCH) / N_BATCH)))):
            print(iter_batch)
            list_keys_batches = list_nodes[iter_batch * N_BATCH:(iter_batch + 1) * N_BATCH]

            _, Qn_batch_s = get_Pn_Qn_batch(Output_GRAPH, list_keys_batches)
            Pn_batch = [Pn_coords_vox_space[_] for _ in list_keys_batches]
            Qn_batch =  Qn_batch_s / ones_spacing # get Pn_batch & Qn_batch in pixel
            # Extract Patches around Pn and Qn
            patches_Pn = [dict_patch[_] for _ in list_keys_batches]

            list_idx_Qn, dict_patch_Qn = extract_patchexact_from_coord_list(image_J_pyramid, Qn_batch_s, list_keys_batches,patch_size=PATCH_SIZE_Qn, vox_size = ones_spacing)
            patches_Qn = [dict_patch_Qn[_] for _ in list_idx_Qn]
            list_keys_batches_without_bugs = list_idx_Qn
            len_batch = len(list_keys_batches_without_bugs)
            errors_batch = list(set(list_keys_batches) - set(list_keys_batches_without_bugs))

            if len_batch>0:
                patches_Pn_padded = pad_patches_fixed(patches_Pn, PATCH_SIZE_Qn=PATCH_SIZE_Qn,
                                                          PATCH_SIZE_Pn=PATCH_SIZE_Pn)


                len_batch = len(list_keys_batches)
                ls_dict_init = []
                for idx_p_in_batch in range(len_batch):
                    if (step == 0) & (pyramid_level == 4):
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

                std_In = [np.std((np.ravel(patches_Pn[_])- np.ravel(
                    patches_Qn[_][0, (PATCH_SIZE_Qn - PATCH_SIZE_Pn) // 2:-(PATCH_SIZE_Qn - PATCH_SIZE_Pn) // 2,
                    (PATCH_SIZE_Qn - PATCH_SIZE_Pn) // 2:-(PATCH_SIZE_Qn - PATCH_SIZE_Pn) // 2,
                    (PATCH_SIZE_Qn - PATCH_SIZE_Pn) // 2:-(PATCH_SIZE_Qn - PATCH_SIZE_Pn) // 2, 0]))) for _ in range(len_batch)]

                ls_Dmat, ls_Di, ls_cell, ls_Qns_i, ls_NablaQ_i = [], [], [], [], []
                ls_mean_t, ls_std_t, ls_mean_A, ls_std_A = [], [], [], []
                for i in range(len_batch):
                    # Shape Vectors
                    Dmat, Di, cell_i, tetrahedrons_batch, nablaq_i = D_idx([list_keys_batches[i]], grid_mesh, Qns)
                    Qns_i = Qns[tetrahedrons_batch]
                    ls_Dmat.append(Dmat), ls_Di.append(Di), ls_cell.append(cell_i), ls_Qns_i.append(
                        Qns_i), ls_NablaQ_i.append(nablaq_i)

                    if (step > 0) | (idx_pyramid > 0):
                        ls_mean_t_i, _ = mean_and_std_neighbors(list_keys_batches[i],
                                                                         Output_GRAPH, t,
                                                                         std_t, A, std_A)
                        ls_mean_t.append(ls_mean_t_i), ls_std_t.append(std_t)
                        ls_mean_A.append(None), ls_std_A.append(None)
                    else:
                        ls_mean_A.append(None), ls_std_A.append(1)
                        ls_mean_t.append(None), ls_std_t.append(1)

                ls_dict_regularization = [{'idx_vertex': np.int16(list_keys_batches[_]),
                                           'step': np.int16(step),
                                           "epoch": np.int16(idx_pyramid),
                                           'PATCH_SIZE_QN': np.int16(PATCH_SIZE_Qn),
                                           'SPACING_ORIG': np.float32(SPACING_ORIG),
                                           'Pn': np.float32(Pn_batch[_]),
                                           'Qn': np.float32(Qn_batch[_]),
                                           'spacing_pyramid': np.float32(spacing_pyramid),
                                           'Wass': np.float32(0),
                                           'std_in': np.float32(std_In[_]**0.5),
                                           'var_in': std_In[_],
                                           #'In_var': np.float32(In_var),
                                           'Dmat': np.float32(ls_Dmat[_]),
                                           'Di': np.float32(ls_Di[_]),
                                           'NablaQ': np.float32(ls_NablaQ_i[_]),
                                           'cell_i': np.int16(ls_cell[_]),

                                           #'t_mean_n': np.float32(t),
                                           #'t_std_n': np.float32(std_t),
                                            't_mean_n': np.float32(ls_mean_t[_]),
                                            't_std_n': ls_std_t[_],
                                           #'A_mean_n': np.float32(A),
                                           'A_std_n': np.float32(std_A),
                                           'Qns': ls_Qns_i[_]} for _ in range(len_batch)]

                # put variables on ray
                x_ids = [ray.put(patches_Pn_padded[i]) for i in range(len_batch)]
                y_ids = [ray.put(patches_Qn[i]) for i in range(len_batch)]

                network_inits_ids = [ray.put(ls_dict_init[i]) for i in range(len_batch)]
                regularization_ids = [ray.put(ls_dict_regularization[i]) for i in range(len_batch)]

                # Create actors to store the networks.
                actor_list = [remote_network.remote(x_ids[i], y_ids[i], network_inits_ids[i], regularization_ids[i]) for i
                              in range(len_batch)]

                new_weights_ids = [actor.step.remote(iterations=NUM_ITERS) for actor in actor_list]
                results = ray.get(new_weights_ids)
                #   weights=[result[0] for result in results]

                #clear_output()
                for i,result in enumerate(results):
                    weights, ls_res, ls_loss = result[0], result[1], result[2]

                    train_idx, loss_idx, MSE_idx, translation_from_Pn_idx,                 accumulated_translation_idx, qn_pn_idx,                 pars_batch_idx, matrice_idx, Pn_batch_idx, Qn_batch_idx,                 wass_idx, sp_idx, regtn_idx, idx_v, grads_idx, NablaQ, regAn_idx, mse_std_idx, x_idx, y_idx = ls_res

                    Qn_updated = np.pad(Qn_batch_idx[1:4] * spacing_pyramid - translation_from_Pn_idx[0, :, 0],
                                        pad_width=[1, 1], mode='constant', constant_values=0)
                    matrice_updated = np.copy(matrice_idx)
                    matrice_updated[:, -1] = 0
                    Output_GRAPH.add_node(idx_v,
                                          # Pn = Pn_updated,
                                          nn_outpouts=weights,
                                          affines=matrice_updated,
                                          affines_or=matrice_idx,
                                          NablaQ=NablaQ,
                                          MSE=MSE_idx/PATCH_SIZE_Pn**3,
                                          MSE_or=MSE_idx/PATCH_SIZE_Pn**3,
                                          Qn=Qn_updated,
                                          # An=nabla_Q.numpy(),
                                          loss_reg=loss_idx,
                                          ls_loss=ls_loss,
                                          accumulated_translation = accumulated_translation_idx[0, :, 0],
                                          Qn_Pn =  qn_pn_idx,
                                          is_ERROR = False,
                                          translation_from_Pn=translation_from_Pn_idx[0, :, 0],
                                          sp=sp_idx,
                                          Wass=wass_idx,
                                          pars=pars_batch_idx[0, :, 0])
                    print(('EPOCH = {}, idx = {} , MSE = {:.4f} ,MSE_std = {:.4f}, regTn = {:.3f}' + ', regAn= ' + str(
                        regAn_idx) + 'regSp = ' + str(sp_idx)).format(step, idx_v, MSE_idx, mse_std_idx, regtn_idx))
            [ls_errors_batch.append(batch_error_num) for batch_error_num  in errors_batch]

        for errors_num in ls_errors_batch:
            Output_GRAPH.add_node(batch_error_num, is_ERROR = True, MSE = 1)



        nx.write_gpickle(Output_GRAPH, PATH_PATIENTS + NUM_P + '/' + str(pyramid_level) + '/Output_GRAPH' + str(step))
        Output_GRAPH, x = Q_step(Output_GRAPH, grid_mesh, K, LAMBDA_W, divide_variance=True)
        t, A, std_t, std_tn, std_A, Rt, Ra, z = estimate_mean_variance_as_paper(Output_GRAPH, grid_mesh)
        nx.write_gpickle(Output_GRAPH,
                         PATH_PATIENTS + NUM_P + '/' + str(pyramid_level) + '/Output_GRAPHQ' + str(step))

        saving_values = save_values(saving_values, t, std_t,
                                    A, std_A, Output_GRAPH)
        np.save(PATH_PATIENTS + NUM_P + '/' + str(pyramid_level) + '/saving_values.npy', saving_values)
