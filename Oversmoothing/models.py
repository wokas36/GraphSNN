import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import Graphsn_GCN

class GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN, self).__init__()
        
        self.gc1 = Graphsn_GCN(nfeat, nhid)
        self.gc3 = Graphsn_GCN(nhid, nhid)
        self.gc4 = Graphsn_GCN(nhid, nhid)
        self.gc5 = Graphsn_GCN(nhid, nhid)
        self.gc6 = Graphsn_GCN(nhid, nhid)
        self.gc7 = Graphsn_GCN(nhid, nhid)
        self.gc8 = Graphsn_GCN(nhid, nhid)
        self.gc2 = Graphsn_GCN(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, 0.2, training=self.training)
        x = F.relu(self.gc4(x, adj))
        x = F.dropout(x, 0.2, training=self.training)
        x = F.relu(self.gc5(x, adj))
        x = F.dropout(x, 0.2, training=self.training)
        x = F.relu(self.gc6(x, adj))
        x = F.dropout(x, 0.2, training=self.training)
        x = F.relu(self.gc7(x, adj))
        x = F.dropout(x, 0.2, training=self.training)
        x = F.relu(self.gc8(x, adj))
        x = F.dropout(x, 0.2, training=self.training)
        x = self.gc2(x, adj)
        
        return F.log_softmax(x, dim=-1)