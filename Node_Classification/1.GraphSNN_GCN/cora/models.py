import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import Graphsn_GCN

class GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN, self).__init__()
        
        self.gc1 = Graphsn_GCN(nfeat, nhid)
        self.gc2 = Graphsn_GCN(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        
        x = F.relu(self.gc1(x, adj)) #F.celu(self.gc1(x, adj), alpha=2e-5)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        
        return F.log_softmax(x, dim=-1)