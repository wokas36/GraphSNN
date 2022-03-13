import torch.nn as nn
import torch.nn.functional as F
from layers import GraphSN
import torch

class GNN(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, batchnorm_dim, dropout_1, dropout_2):
        super().__init__()
        
        self.dropout = dropout_1
        
        self.convs = nn.ModuleList()
        
        self.convs.append(GraphSN(input_dim, hidden_dim, batchnorm_dim, dropout_2))
        
        for _ in range(n_layers-1):
            self.convs.append(GraphSN(hidden_dim, hidden_dim, batchnorm_dim, dropout_2))
        
        # In order to perform graph classification, each hidden state
        # [batch x nodes x hidden_dim] is concatenated, resulting in
        # [batch x nodes x input_dim+hidden_dim*(n_layers)], then aggregated
        # along nodes dimension, without keeping that dimension:
        # [batch x input_dim+hidden_dim*(n_layers)].
        #self.out_proj = nn.Linear(input_dim+hidden_dim*(n_layers), output_dim)
        self.out_proj = nn.Linear((input_dim+hidden_dim*(n_layers)), output_dim)

    def forward(self, data):
        X, A = data[:2]

        hidden_states = [X]
        
        for layer in self.convs:
            X = F.dropout(layer(A, X), self.dropout)
            hidden_states.append(X)

        X = torch.cat(hidden_states, dim=2).sum(dim=1)
        X = self.out_proj(X)

        return X