import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, hpars, in_feats, out_feats, final=False):
        super(GraphAttentionLayer, self).__init__()
        self.hpars = hpars
        self.dropout = self.hpars.experiment.dropout

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.alpha = self.hpars.experiment.alpha
        self.final = final

        self.W = nn.Parameter(torch.empty(size=(self.in_feats, self.out_feats)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=( 2 * self.out_feats, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, training=True):
        Wh = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 *torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=training)
        h_prime = torch.matmul(attention, Wh)

        if not self.final:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_feats, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_feats:, :])

        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_feats) + ' -> ' + str(self.out_feats) + ')'
