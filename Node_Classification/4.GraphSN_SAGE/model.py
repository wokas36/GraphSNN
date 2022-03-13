import torch
import torch.nn as nn
from torch.nn import init

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes, full_nodes):
        embeds = self.enc(nodes, full_nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, full_nodes, labels):
        scores = self.forward(nodes, full_nodes)
        return self.xent(scores, labels.squeeze())