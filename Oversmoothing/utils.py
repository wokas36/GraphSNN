import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch

import os
import pickle as pkl
import sys

import random
import math
import numexpr as ne
from scipy.sparse import csgraph, coo_matrix, issparse, spdiags
from scipy.linalg import cholesky, eigh, lu, qr, svd, norm, solve
# from IPython.core.debugger import Tracer


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
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

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    #features = normalize(features)   
    adj_norm = normalize(adj + sp.eye(adj.shape[0]))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.where(labels)[1]
    #s_val = labels[np.random.choice(labels.shape[0], 500, replace=False)]
    
    #Tracer()()
    
    #idx_test = test_idx_range.tolist()
    #idx_train = range(1500)
    #idx_val = range(1500, 1500 + 500)

    idx_test = range(len(y)+500, len(y)+1500) #test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    return adj_norm, adj, features, labels, idx_train, idx_val, idx_test


def full_load_data(dataset_name, splits_file_path=None):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj_norm, adj, features, labels, _, _, _ = load_data(dataset_name)
        #labels = np.argmax(labels, axis=-1)
        #features = features.todense()
        G = nx.DiGraph(adj)
    else:
        graph_adjacency_list_file_path = os.path.join('new_data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name,
                                                                'out1_node_feature_label.txt')

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}


        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        adj_norm = normalize(adj + sp.eye(adj.shape[0]))
        
        features = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        features = sp.lil_matrix(features)
        
        labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    #features = normalize(features)

    g = adj
  
    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']
    
    num_features = features.shape[1]
    num_labels = len(np.unique(labels))

    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    return adj_norm, g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def calc_f1_micro(predictions, labels):
    """
    Calculate f1 micro.

    Args:
        predictions: tensor with predictions
        labels: tensor with original labels

    Returns:
        f1 score
    """
    preds = predictions.max(1)[1].type_as(labels)
    true_positive = torch.eq(labels, preds).sum().float()
    f1_score = torch.div(true_positive, len(labels))
    return f1_score


def f1(predicted, actual, label, epsilon=1e-7):
    """ A helper function to calculate f1-score for the given `label` """
    # F1 = 2 * (precision * recall) / (precision + recall)
    tp = ((actual.eq(label)) & (predicted.eq(label))).double().sum(dim=0)
    fp = ((actual.ne(label)) & (predicted.eq(label))).double().sum(dim=0)
    fn = ((predicted.ne(label)) & (actual.eq(label))).double().sum(dim=0)
    
    precision = tp/(tp+fp+epsilon)
    recall = tp/(tp+fn+epsilon)
    f1 = 2 * (precision * recall) / (precision + recall+epsilon)
    
    return f1

def f1_macro(predicted, actual):
    # `macro` f1- unweighted mean of f1 per label
    preds = predicted.max(1)[1].type_as(actual)
    return np.mean([f1(preds, actual, label) for label in np.unique(actual)])