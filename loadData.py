import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

class DataSet():
    def __init__(self, category='cora'):
        self.category = category

    def Load(self):
        """
          Loads input data from gcn/data directory

          ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
          ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
          ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
              (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
          ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
          ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
          ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
          ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
              object;
          ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

          All objects above must be saved using python pickle module.

          :param dataset_str: Dataset name
          :return: All data input files loaded (as well the training/test data).
          """
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(self.category, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = self.parse_index_file("data/ind.{}.test.index".format(self.category))
        test_idx_range = np.sort(test_idx_reorder)

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        features = self.preprocess_features(features)

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = self.preprocess_adj_bias(adj)

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        train_mask = self.sample_mask(idx_train, labels.shape[0])
        val_mask = self.sample_mask(idx_val, labels.shape[0])
        test_mask = self.sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

    def parse_index_file(self, filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    def sample_mask(self, idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

    def preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features.todense()

    def sparse_to_tuple(self, sparse_mx):
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

    def preprocess_adj_bias(self, adj_mat):
        num_nodes = adj_mat.shape[0]
        adj_mat = adj_mat + sp.eye(num_nodes)  # self-loop
        adj_mat[adj_mat > 0.0] = 1.0
        if not sp.isspmatrix_coo(adj_mat):
            adj_mat = adj_mat.tocoo()
        adj_mat = adj_mat.astype(np.float32)
        indices = np.vstack(
            (adj_mat.col, adj_mat.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
        # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
        return indices, adj_mat.data, adj_mat.shape