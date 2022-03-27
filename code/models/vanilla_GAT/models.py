import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# This model is the vanilla GAT model but substituting the additive interaction of the vanilla self-attention for multiplicative interactions
class GAT(nn.Module):
    def __init__(self, hpars):
        super(GAT, self).__init__()
        self.hpars = hpars
        self.n_heads = self.hpars.experiment.n_heads
        self.in_feats = self.hpars.experiment.in_feats

        self.edge_feats_dim = self.hpars.experiment.edge_feats_dim
        self.n_classes = self.hpars.experiment.n_classes
        self.hidden_dim = self.hpars.experiment.hidden_dim
        self.model_depth = self.hpars.model.model_depth

        self.dropout_rate = self.hpars.experiment.dropout
        self.dropout = nn.Dropout()

        self.layers = []
        for i in range(self.model_depth):
            #in_dim = self.hidden_dim if i != 0 else self.in_feats
            out_dim = self.hidden_dim
            concat = True if i != (self.model_depth -1) else False

            layer = GATConv(in_channels=-1,
                            out_channels=out_dim,
                            heads = self.n_heads,
                            concat = concat,
                            dropout=self.dropout_rate,
                            add_self_loops=True,
                            edge_dim = self.edge_feats_dim)

            self.layers.append(layer)
            self.add_module('gat_conv_layer_{}'.format(i), layer)


    def forward(self, x, edge_attr, edge_idx, training=True):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_idx, edge_attr)
            #x += x_hat  # residual

        return x