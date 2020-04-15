# _author_ = "jiali cui"
# Email : cuijiali961224@gmail.com
# Data:
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf
import numpy as np
import cv2
def deconv2d(input, weight, bias, output_shape, strides=(2, 2), padding='SAME', activate_func=None, name='deconv2d'):

    if type(strides) == list or type(strides) == tuple:
        [s_h, s_w] = list(strides)
    output_shape = list(output_shape)
    output_shape[0] = tf.shape(input)[0]
    with tf.variable_scope(name):
        w = weight
        deconv = tf.nn.conv2d_transpose(input, filter=w, output_shape=tf.stack(output_shape, axis=0),
                                        strides=[1, s_h, s_w, 1], padding=padding)
        biases = bias
        deconv = tf.nn.bias_add(deconv, biases)
        if activate_func:
            deconv = activate_func(deconv)
        return deconv

def show_in_one(path, images, column, row, show_size=[300, 300], blank_size=5):
    small_h, small_w = images[0].shape[:2]
    # column = int(show_size[1] / (small_w + blank_size))

    show_size[0] = small_h * row + row * blank_size
    show_size[1] = small_w * column + column * blank_size

    # row = int(show_size[0] / (small_h + blank_size))
    shape = [show_size[0], show_size[1]]
    for i in range(2, len(images[0].shape)):
        shape.append(images[0].shape[i])

    merge_img = np.zeros(tuple(shape), images[0].dtype)

    max_count = len(images)
    count = 0
    for i in range(row):
        if count >= max_count:
            break
        for j in range(column):
            if count < max_count:
                im = images[count]
                t_h_start = i * (small_h + blank_size)
                t_w_start = j * (small_w + blank_size)
                t_h_end = t_h_start + im.shape[0]
                t_w_end = t_w_start + im.shape[1]

                merge_img[t_h_start:t_h_end, t_w_start:t_w_end] = im
                count = count + 1
            else:
                break
    cv2.imwrite(path, merge_img)
    # cv2.namedWindow(window_name)
    # cv2.imshow(window_name, merge_img)

def getOutputShape(output_shape):
    output_shape = list(output_shape)
    output_shape = tf.stack(output_shape, axis=0)
    return output_shape

def map_conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], x_shapes[3], y_shapes[4]])], 4)

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

def conv2d(input_, output_dim,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""

    init_range = tf.sqrt(6.0/(tf.to_float(shape[0])+tf.to_float(shape[1])))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx