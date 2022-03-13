from __future__ import print_function

import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence

import os
import pickle as pkl
import sys

import networkx as nx

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """Load data."""
    FILE_PATH = os.path.abspath(__file__)
    DIR_PATH = os.path.dirname(FILE_PATH)
    DATA_PATH = os.path.join(DIR_PATH, 'data/')

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}ind.{}.{}".format(DATA_PATH, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}ind.{}.test.index".format(DATA_PATH, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder),
                                    max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
    
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    s_val = labels[np.random.choice(labels.shape[0], 500, replace=False)]
    
    idx_train = range(len(y))
    idx_val = range(len(y), len(s_val) + 500)
    idx_test = range(len(y) + 500, len(s_val) + 1500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask, val_mask, test_mask, labels

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# def normalize_adj(adj, symmetric=True):
#     if symmetric:
#         d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
#         a_norm = adj.dot(d).transpose().dot(d).tocsr()
#     else:
#         d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
#         a_norm = d.dot(adj).tocsr()
#     return a_norm


# def normalize_adj_numpy(adj, symmetric=True):
#     if symmetric:
#         d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
#         a_norm = adj.dot(d).transpose().dot(d)
#     else:
#         d = np.diag(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
#         a_norm = d.dot(adj)
#     return a_norm


# def preprocess_adj(adj, symmetric=True):
#     adj = adj + sp.eye(adj.shape[0])
#     adj = normalize_adj(adj, symmetric)
#     return adj


# def preprocess_adj_numpy(adj, symmetric=True):
#     adj = adj + np.eye(adj.shape[0])
#     adj = normalize_adj_numpy(adj, symmetric)
#     return adj


# def preprocess_adj_tensor(adj_tensor, symmetric=True):
#     adj_out_tensor = []
#     for i in range(adj_tensor.shape[0]):
#         adj = adj_tensor[i]
#         adj = adj + np.eye(adj.shape[0])
#         adj = normalize_adj_numpy(adj, symmetric)
#         adj_out_tensor.append(adj)
#     adj_out_tensor = np.array(adj_out_tensor)
#     return adj_out_tensor


# def preprocess_adj_tensor_with_identity(adj_tensor, symmetric=True):
#     adj_out_tensor = []
#     for i in range(adj_tensor.shape[0]):
#         adj = adj_tensor[i]
#         adj = adj + np.eye(adj.shape[0])
#         adj = normalize_adj_numpy(adj, symmetric)
#         adj = np.concatenate([np.eye(adj.shape[0]), adj], axis=0)
#         adj_out_tensor.append(adj)
#     adj_out_tensor = np.array(adj_out_tensor)
#     return adj_out_tensor


# def preprocess_adj_tensor_with_identity_concat(adj_tensor, symmetric=True):
#     adj_out_tensor = []
#     for i in range(adj_tensor.shape[0]):
#         adj = adj_tensor[i]
#         adj = adj + np.eye(adj.shape[0])
#         adj = normalize_adj_numpy(adj, symmetric)
#         adj = np.concatenate([np.eye(adj.shape[0]), adj], axis=0)
#         adj_out_tensor.append(adj)
#     adj_out_tensor = np.concatenate(adj_out_tensor, axis=0)
#     return adj_out_tensor

# def preprocess_adj_tensor_concat(adj_tensor, symmetric=True):
#     adj_out_tensor = []
#     for i in range(adj_tensor.shape[0]):
#         adj = adj_tensor[i]
#         adj = adj + np.eye(adj.shape[0])
#         adj = normalize_adj_numpy(adj, symmetric)
#         adj_out_tensor.append(adj)
#     adj_out_tensor = np.concatenate(adj_out_tensor, axis=0)
#     return adj_out_tensor

# def preprocess_edge_adj_tensor(edge_adj_tensor, symmetric=True):
#     edge_adj_out_tensor = []
#     num_edge_features = int(edge_adj_tensor.shape[1]/edge_adj_tensor.shape[2])

#     for i in range(edge_adj_tensor.shape[0]):
#         edge_adj = edge_adj_tensor[i]
#         edge_adj = np.split(edge_adj, num_edge_features, axis=0)
#         edge_adj = np.array(edge_adj)
#         edge_adj = preprocess_adj_tensor_concat(edge_adj, symmetric)
#         edge_adj_out_tensor.append(edge_adj)

#     edge_adj_out_tensor = np.array(edge_adj_out_tensor)
#     return edge_adj_out_tensor


def get_splits(y, s):
    idx_train = range(len(s))
    idx_val = range(len(s), 500)
    idx_test = range(500, 1500)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):
    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


# def normalized_laplacian(adj, symmetric=True):
#     adj_normalized = normalize_adj(adj, symmetric)
#     laplacian = sp.eye(adj.shape[0]) - adj_normalized
#     return laplacian


# def rescale_laplacian(laplacian):
#     try:
#         print('Calculating largest eigenvalue of normalized graph Laplacian...')
#         largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
#     except ArpackNoConvergence:
#         print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
#         largest_eigval = 2

#     scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
#     return scaled_laplacian


# def chebyshev_polynomial(X, k):
#     """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
#     print("Calculating Chebyshev polynomials up to order {}...".format(k))

#     T_k = list()
#     T_k.append(sp.eye(X.shape[0]).tocsr())
#     T_k.append(X)

#     def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
#         X_ = sp.csr_matrix(X, copy=True)
#         return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

#     for i in range(2, k + 1):
#         T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

#     return T_k


# def sparse_to_tuple(sparse_mx):
#     if not sp.isspmatrix_coo(sparse_mx):
#         sparse_mx = sparse_mx.tocoo()
#     coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
#     values = sparse_mx.data
#     shape = sparse_mx.shape
#     return coords, values, shape

# def lmax(L, normalized=True):
#     """Upper-bound on the spectrum."""
#     if normalized:
#         return 2
#     else:
#         return scipy.sparse.linalg.eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]
