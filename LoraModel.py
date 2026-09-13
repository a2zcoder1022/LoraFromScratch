

import torch.nn as nn
import torch

import torch.nn.functional as F

class LinearLayer(nn.Module):
    def __init__(self, original_layer, rank, alpha, dropout=0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)

        self.A = nn.Parameter(torch.randn(self.rank, self.original_layer.in_features), requires_grad=True)
        self.B = nn.Parameter(torch.zeros(self.original_layer.out_features, self.rank), requires_grad=True)


    def forward(self, x):
        output = self.original_layer(x)

        a_output = F.linear(self.dropout(x), self.A)
        b_output = F.linear(a_output, self.B)

        return output + (self.alpha / self.rank) * b_output




