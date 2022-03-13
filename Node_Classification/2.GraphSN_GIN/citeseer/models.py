import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from torch.nn import Sequential, Linear, ReLU, Parameter, Sequential, BatchNorm1d, Dropout

class Graphsn_GIN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Graphsn_GIN, self).__init__()
        
        self.nn = Linear(nfeat, nhid)
        self.fc = Linear(nhid, nclass)
        self.dropout = dropout
        
        self.eps = nn.Parameter(torch.FloatTensor(1))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv_eps = 0.60 / math.sqrt(self.eps.size(0)) #68.10 -> 0.44 | 68.70 -> 0.60
        nn.init.constant_(self.eps, stdv_eps)

    def forward(self, x, adj):
        
        v = self.eps*torch.diag(adj)
        mask = torch.diag(torch.ones_like(v))
        adj = mask*torch.diag(v) + (1. - mask)*adj
        
        x = torch.mm(adj, x)
        x = F.relu(self.nn(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=-1)
    

# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# from layers import Graphsn_GCN

# class GNN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GNN, self).__init__()
        
#         self.gc1 = Graphsn_GCN(nfeat, nhid)
#         self.gc2 = Graphsn_GCN(nhid, nclass)
#         self.dropout = dropout

#     def forward(self, x, adj):
        
#         x = F.relu(self.gc1(x, adj)) #F.celu(self.gc1(x, adj), alpha=2e-5)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
        
#         return F.log_softmax(x, dim=-1)