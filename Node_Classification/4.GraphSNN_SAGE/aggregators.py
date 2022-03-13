import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import random

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        
        self.eps = nn.Parameter(torch.FloatTensor(1))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv_eps = 0.0397 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)
        
    def forward(self, nodes, to_neighs, normalized_adj, feat_data, full_nodes, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        structural_coeff = [normalized_adj[nodes[j]][list(samp_neighs[j])].tolist() for j in np.arange(len(samp_neighs))]
        
        
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(full_nodes).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(full_nodes))
            
        total_emb = []
        for i in np.arange(len(nodes)):
            sample_neigh_list = list(samp_neighs[i])
            h_v = torch.zeros(embed_matrix.shape[1])
            h_u = torch.zeros(embed_matrix.shape[1])
            if(nodes[i] in sample_neigh_list):
                index = sample_neigh_list.index(nodes[i])
                h_v = self.eps*structural_coeff[i][index]*embed_matrix[nodes[i]]
                sample_neigh_list.pop(index)
                structural_coeff[i].pop(index)
            
            num_neigh = len(structural_coeff[i])
            for count, j in enumerate(sample_neigh_list):
                h_u += structural_coeff[i][count]*embed_matrix[j]
            h_u /= num_neigh
            emb = torch.cat((h_u, h_v), 0)
            total_emb.append(emb)
           
        embeded_matrix = torch.stack(total_emb)
        
        return embeded_matrix 