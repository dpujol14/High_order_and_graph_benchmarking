from torch_geometric.nn import GINEConv
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, dropout = 0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.dropout(F.gelu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


# This model is the vanilla GAT model but substituting the additive interaction of the vanilla self-attention for multiplicative interactions
class GINE(nn.Module):
    def __init__(self, hpars):
        super(GINE, self).__init__()
        self.hpars = hpars
        self.n_heads = self.hpars.experiment.n_heads
        self.in_feats = self.hpars.experiment.in_feats

        self.edge_feats_dim = self.hpars.experiment.edge_feats_dim
        self.n_classes = self.hpars.experiment.n_classes
        self.hidden_dim = self.hpars.experiment.hidden_dim
        self.model_depth = self.hpars.model.model_depth

        self.dropout = nn.Dropout(self.hpars.experiment.dropout)

        self.layers = []
        for i in range(self.model_depth):
            in_dim = self.hidden_dim
            out_dim = self.hidden_dim

            if i==0:
                in_dim = self.in_feats

            net = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, out_dim)
            )
            layer = GINEConv(nn = net,
                            eps=0,
                            train_eps = True,
                            edge_dim=self.edge_feats_dim)
            self.layers.append(layer)
            self.add_module('conv_layer_{}'.format(i), layer)


    def forward(self, x, edge_attr, edge_idx, training=True):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_idx, edge_attr)
            #x += x_hat

        return x