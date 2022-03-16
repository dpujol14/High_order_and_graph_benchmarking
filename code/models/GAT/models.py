import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphAttentionLayer

# This model are obtained from the PyTorch implementation of (https://github.com/Diego999/pyGAT)
class GAT(nn.Module):
    def __init__(self, hpars):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.hpars = hpars
        self.n_heads = self.hpars.experiment.n_heads
        self.in_feats = self.hpars.experiment.in_feats
        self.n_classes = self.hpars.experiment.n_classes
        self.hidden_dim = self.hpars.experiment.hidden_dim
        self.model_depth = self.hpars.model.model_depth

        self.dropout = nn.Dropout(self.hpars.experiment.dropout)

        self.attentions = [GraphAttentionLayer(hpars=self.hpars, in_feats = self.in_feats, out_feats=self.hidden_dim, final=False) for _ in range(self.n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(hpars = hpars, in_feats=self.in_feats * self.n_heads, out_feats=self.n_classes, final=False)


    def forward(self, x, adj, training=True):
        x = self.dropout(x)
        x = torch.cat([att(x, adj, training) for att in self.attentions], dim=1)
        x = self.dropout(x)
        x = F.elu(self.out_att(x, adj, training))

        # We return the output of dimension (N x n_classes). This still requires to do proper pooling or so depening on the type of class (e.g., graph)
        return x