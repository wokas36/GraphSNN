import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import numpy as np
from IPython.core.debugger import Tracer

class Graphsn_GCN(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(Graphsn_GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.eps = nn.Parameter(torch.FloatTensor(1))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.95 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
        stdv_eps = 0.21 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)
        
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        
        v = (self.eps)*torch.diag(adj)
        mask = torch.diag(torch.ones_like(v))
        adj = mask*torch.diag(v) + (1. - mask)*adj

        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'