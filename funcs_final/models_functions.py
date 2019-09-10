import tensorflow as tf
import networkx as nx
from scipy import *
import sys
#sys.path.append('../utils/spatial_transformer_network')
#from spatial_transformer_network import *

## VAR
THETA_MIN = math.pi / 10
SHEAR_MIN = 0.05
TRANSLATION_MIN = 1


## MODEL



class estimate_affine(tf.keras.layers.Layer):
    def __init__(self, val_init_t=[0., 0., 0.], val_init_thetas=[0., 0., 0.], val_init_shears=[0., 0., 0., 0., 0., 0.],
                 val_init_all=None):
        '''
        Input dim is the dimension in which it is performed (3D affine matrix of size 3x3)
        val_init_t= list of three translations
        '''
        super(estimate_affine, self).__init__()
        # w_init = tf.zeros_initializer()

        self.theta0 = tf.Variable(initial_value=tf.reshape(val_init_thetas[0], [1]), dtype='float32', trainable=True,
                                  constraint=tf.keras.constraints.MinMaxNorm(-THETA_MIN, +THETA_MIN),
                                  name='theta0')

        self.theta1 = tf.Variable(initial_value=tf.reshape(val_init_thetas[1], [1]), dtype='float32', trainable=True,
                                  constraint=tf.keras.constraints.MinMaxNorm(-THETA_MIN, +THETA_MIN),
                                  name='theta1')

        self.theta2 = tf.Variable(initial_value=tf.reshape(val_init_thetas[2], [1]), dtype='float32', trainable=True,
                                  constraint=tf.keras.constraints.MinMaxNorm(-THETA_MIN, +THETA_MIN),
                                  name='theta2')

        self.shear0 = tf.Variable(initial_value=tf.reshape(val_init_shears[0], [1]), dtype='float32', trainable=True,
                                  constraint=tf.keras.constraints.MinMaxNorm(-SHEAR_MIN, SHEAR_MIN),
                                  name='shear0')
        self.shear1 = tf.Variable(initial_value=tf.reshape(val_init_shears[1], [1]), dtype='float32', trainable=True,
                                  constraint=tf.keras.constraints.MinMaxNorm(-SHEAR_MIN, SHEAR_MIN),
                                  name='shear1')
        self.shear2 = tf.Variable(initial_value=tf.reshape(val_init_shears[2], [1]), dtype='float32', trainable=True,
                                  constraint=tf.keras.constraints.MinMaxNorm(-SHEAR_MIN, SHEAR_MIN),
                                  name='shear2')
        self.shear3 = tf.Variable(initial_value=tf.reshape(val_init_shears[3], [1]), dtype='float32', trainable=True,
                                  constraint=tf.keras.constraints.MinMaxNorm(-SHEAR_MIN, SHEAR_MIN),
                                  name='shear3')
        self.shear4 = tf.Variable(initial_value=tf.reshape(val_init_shears[4], [1]), dtype='float32', trainable=True,
                                  constraint=tf.keras.constraints.MinMaxNorm(-SHEAR_MIN, SHEAR_MIN),
                                  name='shear4')
        self.shear5 = tf.Variable(initial_value=tf.reshape(val_init_shears[5], [1]), dtype='float32', trainable=True,
                                  constraint=tf.keras.constraints.MinMaxNorm(-SHEAR_MIN, SHEAR_MIN),
                                  name='shear5')

        self.translation1 = tf.Variable(initial_value=tf.reshape(val_init_t[0], [1]), dtype='float32', trainable=True,
                                        constraint=tf.keras.constraints.MinMaxNorm(-TRANSLATION_MIN, TRANSLATION_MIN),
                                        name='translation1')
        self.translation2 = tf.Variable(initial_value=tf.reshape(val_init_t[1], [1]), dtype='float32', trainable=True,
                                        constraint=tf.keras.constraints.MinMaxNorm(-TRANSLATION_MIN, TRANSLATION_MIN),
                                        name='translation2')
        self.translation3 = tf.Variable(initial_value=tf.reshape(val_init_t[2], [1]), dtype='float32', trainable=True,
                                        constraint=tf.keras.constraints.MinMaxNorm(-TRANSLATION_MIN, TRANSLATION_MIN),
                                        name='translation3')

    def call(self, inputs):
        return (self.theta0, self.theta1, self.theta2,
                self.shear0, self.shear1, self.shear2, self.shear3, self.shear4, self.shear5,
                self.translation1, self.translation2, self.translation3)


class Models(tf.keras.Model):

    def __init__(self, val_init_t,val_init_thetas,val_init_shears,N_MODELS=1):
        super(Models, self).__init__()

        self.block_i = [estimate_affine(val_init_t=val_init_t[node],
                                        val_init_thetas=val_init_thetas[node],
                                        val_init_shears=val_init_shears[node])
                        for node in range(N_MODELS)]

    def call(self, inputs):
        x = [self.block_i[i](input_i) for i, input_i in enumerate(inputs)]

        return x


def losses_per_batch_RMSE_orig(y_true,y_pred):
    ls_losses=[]

    for iter_loss in range(len(y_true)):
        y_true_batch=y_true[iter_loss]
        y_pred_batch=y_pred[iter_loss]
        y_pred_flatten=tf.reshape(y_pred_batch,[-1,1])
        y_true_flatten=tf.reshape(y_true_batch,[-1,1])
        ### ONLY PIXELS WHERE
        indices_nn_null_pred=tf.where(tf.math.greater(y_pred_flatten,-3))

        indices_nn_null_pred=tf.where(tf.math.greater(y_pred_flatten,-3))
        y_pred_filtered1=tf.gather_nd(y_pred_flatten,indices_nn_null_pred)
        y_true_filtered1=tf.gather_nd(y_true_flatten,indices_nn_null_pred)

        ##
        indices_neg_pred=tf.where(tf.math.greater(y_true_filtered1,-3))
        y_pred_filtered=tf.gather_nd(y_pred_filtered1,indices_neg_pred)
        y_true_filtered=tf.gather_nd(y_true_filtered1,indices_neg_pred)
        #RMSE=tf.reduce_sum((y_true_filtered-y_pred_filtered)**2)
        AE_square= tf.reduce_sum((tf.sort(y_pred_filtered)-tf.sort(y_true_filtered))**2)
        RMSE=tf.reduce_sum((y_true_filtered1-y_pred_filtered1)**2)

        ### BACKGROUND REGULARIZATION
        indices_neg_pred=tf.where(tf.math.less(y_pred_filtered1,-3))
        n_vox=indices_nn_null_pred.shape[0]
        ls_losses.append(RMSE/(8**3)) #+reg_pred_in_BG/n_vox+AE_square/n_vox
    return ls_losses

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
    tf.assert_equal(tf.shape(rvecs)[1], 3)

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

    Rs = tf.eye(3, batch_shape=[batch_size]) + \
         tf.sin(thetas)[..., tf.newaxis] * Ks + \
         (1 - tf.cos(thetas)[..., tf.newaxis]) * tf.matmul(Ks, Ks)

    # Avoid returning NaNs where division by zero happened
    mat=tf.where(is_zero,
        tf.eye(3, batch_shape=[batch_size]), Rs)
    mat=tf.reshape(mat,[3,3])
    return mat


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


def calc_nabla_Q(idx_points, grid, trans):
    '''

    '''
    cells = grid.cells.reshape(-1, 5)[:, 1:]

    ## RETURN THE NUMBER OF THE CELLS
    ## find the num of tetra
    ## if idx_points in cell

    num_cells = [num_cell
                 for num_cell, cell in enumerate(np.reshape(grid.cells, [-1, 5])[:, 1:])
                 if any([idx_cell in idx_points for idx_cell in [*cell]])]

    ls_DjQj = []
    for num_cell in num_cells:
        cell = cells[num_cell]
        idx_cell = np.argwhere(cell == idx_points[0])[0][0]  ## not multibatch compatible
        Q = grid.points[cell]
        V, Hinv, _ = calc_shape_vectors(Q, coords=(0, 0, 0))

        Hinv_tf = tf.cast(Hinv, tf.float32)
        D_mat_tf = tf.slice(Hinv_tf, [0, 0], [4, 3])
        Pns_tf = tf.cast(Q, tf.float32)

        indices = tf.constant([[idx_cell, 0], [idx_cell, 1], [idx_cell, 2]])
        values = [tf.slice(tf.reshape(trans, [-1]), [0], [1]),
                              tf.slice(tf.reshape(trans, [-1]), [1], [1]),
                              tf.slice(tf.reshape(trans, [-1]), [2], [1])]
        values = tf.reshape(values, [-1])
        Pns_tf = tf.tensor_scatter_nd_add(Pns_tf, indices, values)
        djqj = tf.math.multiply(D_mat_tf , Pns_tf)
        ls_DjQj.append(tf.reduce_sum(djqj,axis=0))
        #ls_DjQj.append(tf.reduce_sum(djqj))

    nablaQ = tf.reshape(tf.reduce_mean(ls_DjQj, axis=[0]), [1, 3, -1])
    #nablaQ = tf.reduce_mean(ls_DjQj, axis=[0])

    return nablaQ


def calc_scalP(idx_points, grid, trans):
    cells = grid.cells.reshape(-1, 5)[:, 1:]

    ## RETURN THE NUMBER OF THE CELLS
    ## find the num of tetra
    ## if idx_points in cell

    num_cells = [num_cell
                 for num_cell, cell in enumerate(np.reshape(grid.cells, [-1, 5])[:, 1:])
                 if any([idx_cell in idx_points for idx_cell in [*cell]])]
    ls_Di = []
    for num_cell in num_cells:
        cell = cells[num_cell]
        idx_cell = np.argwhere(cell == idx_points[0])[0][0]  ## not multibatch compatible
        Q = grid.points[cell]

        v = simplex_volume(vertices=Q)
        # V, _, _ = calc_shape_vectors(Q, coords=(0, 0, 0))
        D_mat = ComputeGradient(Q).T
        D_mat_tf = tf.cast(D_mat, tf.float32) / (6 * v)
        Di = tf.slice(D_mat_tf, (idx_cell, 0), (1, 3))

        ls_Di.append(Di)
        # Pi=tf.cast(tf.slice(Q,(idx_cell,0),(1,3)),tf.float32)
        # Qi= Pi + tf.reshape(trans,[1,3])
    # norm_t=tf.math.sqrt(tf.reduce_sum(tf.math.pow(trans,2)))+tf.cast(1e-3,tf.float32)

    mean_Di = tf.reshape(tf.reduce_mean(ls_Di, axis=0), [3])
    scalar_product = tf.tensordot(mean_Di, tf.reshape(trans, [3]), axes=1)
    norm_t = tf.norm(trans + tf.cast(1e-3, tf.float32))
    norm_Di = tf.norm(mean_Di + tf.cast(1e-3, tf.float32))
    costhet = tf.divide(scalar_product, norm_t * norm_Di)

    return 1 - tf.math.abs(costhet)

def calc_nabla_Q_pyopt_tensordot(idx_points, grid, trans):
    '''

    '''
    cells = grid.cells.reshape(-1, 5)[:, 1:]

    ## RETURN THE NUMBER OF THE CELLS
    ## find the num of tetra
    ## if idx_points in cell

    num_cells = [num_cell
                 for num_cell, cell in enumerate(np.reshape(grid.cells, [-1, 5])[:, 1:])
                 if any([idx_cell in idx_points for idx_cell in [*cell]])]

    ls_DjQj = []
    for num_cell in num_cells:
        cell = cells[num_cell]
        idx_cell = np.argwhere(cell == idx_points[0])[0][0]  ## not multibatch compatible
        Q = grid.points[cell]
        v = simplex_volume(vertices=Q)
        #V, _, _ = calc_shape_vectors(Q, coords=(0, 0, 0))
        D_mat=ComputeGradient(Q).T
        D_mat_tf = tf.cast(D_mat, tf.float32)/(6*v)

        Pns_tf = tf.cast(Q, tf.float32)

        indices = tf.constant([[idx_cell, 0], [idx_cell, 1], [idx_cell, 2]])
        values = [tf.slice(tf.reshape(trans, [-1]), [0], [1]),
                              tf.slice(tf.reshape(trans, [-1]), [1], [1]),
                              tf.slice(tf.reshape(trans, [-1]), [2], [1])]
        values = tf.reshape(values, [-1])
        Pns_tf = tf.tensor_scatter_nd_add(Pns_tf, indices, values)
        #djqj = tf.tensordot(D_mat_tf , Pns_tf,axes=0)
        #djqj = tf.reduce_sum([tf.reduce_sum(tf.tensordot(tf.reshape(tf.slice(D_mat_tf,[i,0],[1,3]),[3]),
        #    tf.reshape(tf.slice(Pns_tf,[i,0],[1,3]),[3]),axes=0),axis=0) for i in range(4)],axis=0)
        #print(djqj)
        djqj = tf.math.multiply(D_mat_tf , Pns_tf)
        ls_DjQj.append(tf.reduce_sum(djqj,axis=0))
        #ls_DjQj.append(tf.reduce_sum(djqj))

    nablaQ = tf.reshape(tf.reduce_mean(ls_DjQj, axis=[0]),[1,3,-1])
    return nablaQ



def calc_regularization(step, idx_patient, mode, predicted_t, predicted_A, t, t_cov, z_t, A, A_cov, z_A, graph,
                        normalize_on_confidence=False):
    if step == 0:
        reg_tn = tf.cast(0., tf.float32)
        reg_An = tf.cast(0., tf.float32)
    else:
        if mode == "N":

            idx_neighbors = [*graph.neighbors(idx_patient)]
            tn_neighbors = np.array(
                [nx.get_node_attributes(graph, 'translation_from_Pn')[idx_n] for idx_n in idx_neighbors])
            MSE_i = np.stack(
                [nx.get_node_attributes(graph, 'MSE')[idx_n] for idx_n in idx_neighbors])
            MSE_i[MSE_i>1]=1
            # t_mean_neigbors = tf.cast(tf.reduce_mean(tn_neighbors, axis=0), tf.float32)
            t_mean_neigbors = tf.cast(np.matrix(-np.log(MSE_i)) * tn_neighbors / np.sum(-np.log(MSE_i)), tf.float32)
            t_mean = tf.reshape(t_mean_neigbors, [1, 3, -1])
            tn_std = tf.cast(tf.reshape(np.std(tn_neighbors, axis=0), [1, 3, -1]), tf.float32)

            An_neigbors = np.array([
                nx.get_node_attributes(graph, 'An')[idx_n]
                for idx_n in idx_neighbors])
            An_neigbors=np.reshape(An_neigbors,[-1,3])

            An_mean = tf.cast(np.matrix(-np.log(MSE_i)) * An_neigbors / np.sum(-np.log(MSE_i)), tf.float32)
            An_std = tf.cast(tf.reshape(np.std(An_neigbors, axis=0), [1, An_mean.shape[0], -1]), tf.float32)
            # An_mean_neigbors=tf.cast(An_mean_neigbors,tf.float32)

            # reg_An=tf.reduce_sum(tf.divide(diff_An,tf.cast(An_cov,tf.float32)))

        elif mode == "G":
            t_mean = tf.reshape(t, [1, t.shape[1], -1])
            An_mean = tf.reshape(A, [1, A.shape[1], -1])


        else:
            print('MODE MUST BE N for neigbors or G for Global')
        diff_tn = (predicted_t - t_mean) ** 2
        # reg_tn = tf.reduce_mean(
        #    tf.divide(tf.reshape(diff_tn, [-1, t_mean.shape[0]]), tf.cast(t_cov.diagonal(), tf.float32)))
        reg_tn = tf.reduce_mean(tf.divide(tf.reshape(diff_tn, [-1, t_mean.shape[1]]), tn_std**2))

        diff_An = (predicted_A - An_mean) ** 2
        # reg_An = tf.reduce_mean(
        #    tf.divide(tf.reshape(diff_An, [-1, An_mean.shape[0]]), tf.cast(A_cov.diagonal(), tf.float32)))
        reg_An = tf.reduce_mean(tf.divide(tf.reshape(diff_An, [1, An_mean.shape[1], -1]), An_std**2)) / 3

        if normalize_on_confidence:
            reg_An *= z_A[idx_patient]
            reg_tn *= z_t[idx_patient]

    return reg_tn, reg_An


def calc_regularization_unique_old(step, idx_patient, mode, predicted_t, predicted_A, t, t_cov, z_t, A, A_cov, z_A, graph,
                        normalize_on_confidence=False):
    if step == 0:
        reg_tn = tf.cast(0., tf.float32)
        reg_An = tf.cast(0., tf.float32)
    else:
        if mode == "N":

            idx_neighbors = [*graph.neighbors(idx_patient)]
            tn_neighbors = np.array(
                [nx.get_node_attributes(graph, 'translation_from_Pn')[idx_n] for idx_n in idx_neighbors])
            MSE_i = np.stack(
                [nx.get_node_attributes(graph, 'MSE')[idx_n] for idx_n in idx_neighbors])
            MSE_i[MSE_i > 1] = 1
            # t_mean_neigbors = tf.cast(tf.reduce_mean(tn_neighbors, axis=0), tf.float32)
            t_mean_neigbors = tf.cast(np.matrix(-np.log(MSE_i)) * tn_neighbors / np.sum(-np.log(MSE_i)), tf.float32)
            t_mean = tf.reshape(t_mean_neigbors, [1, 3, -1])
            tn_std = tf.cast(tf.reshape(np.std(tn_neighbors, axis=0), [1, 3, -1]), tf.float32)

            An_neigbors = np.array([
                nx.get_node_attributes(graph, 'An')[idx_n]
                for idx_n in idx_neighbors])
            An_neigbors = np.reshape(An_neigbors, [-1, 3])

            An_mean = tf.cast(np.matrix(-np.log(MSE_i)) * An_neigbors / np.sum(-np.log(MSE_i)), tf.float32)
            An_std = tf.cast(tf.reshape(np.std(An_neigbors, axis=0), [1, An_mean.shape[0], -1]), tf.float32)
            # An_mean_neigbors=tf.cast(An_mean_neigbors,tf.float32)

            # reg_An=tf.reduce_sum(tf.divide(diff_An,tf.cast(An_cov,tf.float32)))

        elif mode == "G":
            t_mean = tf.reshape(t, [1, t.shape[1], -1])
            An_mean = tf.reshape(A, [1, A.shape[1], -1])
            tn_std = t_cov
            An_std = A_cov

            idx_neighbors = [*graph.neighbors(idx_patient)]
            tn_neighbors = np.array(
                [nx.get_node_attributes(graph, 'translation_from_Pn')[idx_n] for idx_n in idx_neighbors])
            MSE_i = np.stack(
                [nx.get_node_attributes(graph, 'MSE')[idx_n] for idx_n in idx_neighbors])
            MSE_i[MSE_i > 1] = 1
            # t_mean_neigbors = tf.cast(tf.reduce_mean(tn_neighbors, axis=0), tf.float32)
            t_mean_neigbors = tf.cast(np.matrix(-np.log(MSE_i)) * tn_neighbors / np.sum(-np.log(MSE_i)), tf.float32)
            t_mean = tf.reshape(t_mean_neigbors, [1, 3, -1])
            tn_std_n = tf.cast(tf.reshape(np.std(tn_neighbors, axis=0), [1, 3, -1]), tf.float32)

            An_neigbors = np.array([
                nx.get_node_attributes(graph, 'An')[idx_n]
                for idx_n in idx_neighbors])
            An_neigbors = np.reshape(An_neigbors, [-1, 3])

            An_mean = tf.cast(np.matrix(-np.log(MSE_i)) * An_neigbors / np.sum(-np.log(MSE_i)), tf.float32)
            An_std_n = tf.cast(tf.reshape(np.std(An_neigbors, axis=0), [1, An_mean.shape[0], -1]), tf.float32)

            # An_std=An_std_n/An_std
            # tn_std=tn_std_n/tn_std


        else:
            print('MODE MUST BE N for neigbors or G for Global')
        diff_tn = (predicted_t - t_mean) ** 2
        # reg_tn = tf.reduce_mean(
        #    tf.divide(tf.reshape(diff_tn, [-1, t_mean.shape[0]]), tf.cast(t_cov.diagonal(), tf.float32)))
        reg_tn = tf.reduce_mean(tf.divide(tf.reshape(diff_tn, [-1, t_mean.shape[1]]), tn_std ** 2))

        diff_An = (predicted_A - An_mean) ** 2
        # reg_An = tf.reduce_mean(
        #    tf.divide(tf.reshape(diff_An, [-1, An_mean.shape[0]]), tf.cast(A_cov.diagonal(), tf.float32)))
        reg_An = tf.reduce_mean(tf.divide(tf.reshape(diff_An, [1, An_mean.shape[1], -1]), An_std ** 2))

        if normalize_on_confidence:
            reg_An *= z_A[idx_patient]
            reg_tn *= z_t[idx_patient]

    return reg_tn, reg_An


def calc_regularization_unique(step, idx_patient, mode, predicted_t, predicted_A, t, t_cov, z_t, A, A_cov, z_A, graph,
                               normalize_on_confidence=False):
    if step == 0:
        reg_tn = tf.cast(0., tf.float32)
        reg_An = tf.cast(0., tf.float32)
    else:
        if mode == "N":

            idx_neighbors = [*graph.neighbors(idx_patient)]
            tn_neighbors = np.array(
                [nx.get_node_attributes(graph, 'translation_from_Pn')[idx_n] for idx_n in idx_neighbors])
            MSE_i = np.stack(
                [nx.get_node_attributes(graph, 'MSE')[idx_n] for idx_n in idx_neighbors])
            MSE_i[MSE_i > 1] = 1
            # t_mean_neigbors = tf.cast(tf.reduce_mean(tn_neighbors, axis=0), tf.float32)
            t_mean_neigbors = tf.cast(np.matrix(-np.log(MSE_i)) * tn_neighbors / np.sum(-np.log(MSE_i)), tf.float32)
            t_mean = tf.reshape(t_mean_neigbors, [1, 3, -1])
            tn_std = tf.cast(tf.reshape(np.std(tn_neighbors, axis=0), [1, 3, -1]), tf.float32)

            An_neigbors = np.array([
                nx.get_node_attributes(graph, 'An')[idx_n]
                for idx_n in idx_neighbors])

            An_neigbors = np.reshape(An_neigbors, [An_neigbors.shape[0], -1])
            # print(An_neigbors)
            MSE_i = np.reshape(MSE_i, [MSE_i.shape[0], -1])

            An_mean = tf.cast(np.matrix(-np.log(MSE_i).T) * An_neigbors / np.sum(-np.log(MSE_i)), tf.float32)
            # An_std = tf.cast(tf.reshape(np.std(An_neigbors, axis=1), [1, An_mean.shape[0], -1]), tf.float32)
            An_std = tf.cast(np.std(An_neigbors, axis=0), tf.float32)

            # An_mean_neigbors=tf.cast(An_mean_neigbors,tf.float32)

            # reg_An=tf.reduce_sum(tf.divide(diff_An,tf.cast(An_cov,tf.float32)))

        elif mode == "G":
            t_mean = tf.reshape(t, [1, t.shape[1], -1])
            An_mean = tf.reshape(A, [1, A.shape[1], -1])
            tn_std = t_cov
            An_std = A_cov

            idx_neighbors = [*graph.neighbors(idx_patient)]
            tn_neighbors = np.array(
                [nx.get_node_attributes(graph, 'translation_from_Pn')[idx_n] for idx_n in idx_neighbors])
            MSE_i = np.stack(
                [nx.get_node_attributes(graph, 'MSE')[idx_n] for idx_n in idx_neighbors])
            MSE_i[MSE_i > 1] = 1
            # t_mean_neigbors = tf.cast(tf.reduce_mean(tn_neighbors, axis=0), tf.float32)
            t_mean_neigbors = tf.cast(np.matrix(-np.log(MSE_i)) * tn_neighbors / np.sum(-np.log(MSE_i)), tf.float32)
            t_mean = tf.reshape(t_mean_neigbors, [1, 3, -1])
            tn_std_n = tf.cast(tf.reshape(np.std(tn_neighbors, axis=0), [1, 3, -1]), tf.float32)

            An_neigbors = np.array([
                nx.get_node_attributes(graph, 'An')[idx_n]
                for idx_n in idx_neighbors])
            # An_neigbors = np.reshape(An_neigbors, [-1, nabla_Q])

            An_mean = tf.cast(np.matrix(-np.log(MSE_i)) * An_neigbors / np.sum(-np.log(MSE_i)), tf.float32)
            An_std_n = tf.cast(np.std(An_neigbors, axis=0), tf.float32)

            # An_std=An_std_n/An_std
            # tn_std=tn_std_n/tn_std


        else:
            print('MODE MUST BE N for neigbors or G for Global')
        diff_tn = (predicted_t - t_mean) ** 2
        # reg_tn = tf.reduce_mean(
        #    tf.divide(tf.reshape(diff_tn, [-1, t_mean.shape[0]]), tf.cast(t_cov.diagonal(), tf.float32)))
        reg_tn = tf.reduce_mean(tf.divide(tf.reshape(diff_tn, [-1, t_mean.shape[1]]), tn_std ** 2))

        diff_An = (predicted_A - An_mean) ** 2
        # reg_An = tf.reduce_mean(
        #    tf.divide(tf.reshape(diff_An, [-1, An_mean.shape[0]]), tf.cast(A_cov.diagonal(), tf.float32)))
        reg_An = tf.reduce_mean(tf.divide(diff_An, An_std ** 2))

        if normalize_on_confidence:
            reg_An *= z_A[idx_patient]
            reg_tn *= z_t[idx_patient]

    return reg_tn, reg_An