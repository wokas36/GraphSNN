import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists, normalized_adj, feat_data, aggregator,
            num_sample=10,
            base_model=None, cuda=False, 
            feature_transform=False): 
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.normalized_adj = normalized_adj
        self.feat_data = feat_data
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, 2 * self.feat_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes, full_nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """
        neigh_list = [self.adj_lists[int(node)] for node in nodes]
        
        combined_neigh_feats = self.aggregator.forward(nodes, neigh_list, self.normalized_adj, 
                                                       self.feat_data, full_nodes, self.num_sample)
        
        combined = F.relu(self.weight.mm(combined_neigh_feats.t()))
        
        return combined