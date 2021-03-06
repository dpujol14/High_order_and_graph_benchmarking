import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import FullNeighborhoodMultGraphLayer

# This model is the vanilla GAT model but substituting the additive interaction of the vanilla self-attention for multiplicative interactions
class FullNeighborhoodMultGAT(nn.Module):
    def __init__(self, hpars):
        super(FullNeighborhoodMultGAT, self).__init__()
        self.hpars = hpars
        self.n_heads = self.hpars.experiment.n_heads
        self.in_feats = self.hpars.experiment.in_feats

        self.edge_feats_dim = self.hpars.experiment.edge_feats_dim
        self.n_classes = self.hpars.experiment.n_classes
        self.hidden_dim = self.hpars.experiment.hidden_dim
        self.model_depth = self.hpars.model.model_depth

        self.dropout = nn.Dropout(self.hpars.experiment.dropout)

        self.attention_layers = []
        # Create the hidden layers that operate in latten F' dimension
        for i in range(self.model_depth):
            if i==0:
                # First layer that maps the input from dimensionality F to F'
                self.attentions = [FullNeighborhoodMultGraphLayer(hpars=self.hpars,
                                                               in_feats=self.in_feats,
                                                               out_feats= int(self.hidden_dim / self.n_heads),
                                                               edge_feat_dim = self.edge_feats_dim,
                                                               final=False) for _ in range(self.n_heads)]
                for j, attention in enumerate(self.attentions):
                     self.add_module('attention_{}_{}'.format(i,j), attention)
            elif i == self.model_depth-1:
                # If this is the last layer, then we need to map it from F' to the output class
                # This last layer takes as input the concatenation of all the heads already
                self.attentions = [FullNeighborhoodMultGraphLayer(hpars=self.hpars,
                                                               in_feats=self.hidden_dim,
                                                               out_feats=self.n_classes,
                                                               edge_feat_dim = self.edge_feats_dim,
                                                               final=False) for _ in range(self.n_heads)]
                for j, attention in enumerate(self.attentions):
                    self.add_module('attention_{}_{}'.format(i, j), attention)
            else:
                # If this is a hidden layer, then we just do a mapping F' -> F'
                self.attentions = [
                    FullNeighborhoodMultGraphLayer(hpars=self.hpars,
                                                in_feats= self.hidden_dim,
                                                out_feats= int(self.hidden_dim / self.n_heads),
                                                edge_feat_dim = self.edge_feats_dim,
                                                final=False) for _ in range(self.n_heads)]
                for j, attention in enumerate(self.attentions):
                    self.add_module('attention_{}_{}'.format(i, j), attention)

            self.attention_layers.append(self.attentions)


    def forward(self, node_feats, edge_feats, edge_indices, adj, training=True):
        x_hat = node_feats
        for i, single_att_layer in enumerate(self.attention_layers):
            #x = self.dropout(x)

            if i == self.model_depth - 1:
                # If it is the last layer, then we perform average aggregation
                x_hat = torch.mean(torch.stack([att(x_hat, edge_feats, edge_indices, adj, training) for att in single_att_layer]), dim=1)
            else:
                x_hat = torch.cat([att(x_hat, edge_feats, edge_indices, adj, training) for att in single_att_layer], dim=1)

            # x = self.dropout(x)
            # x = F.elu(self.out_att(x, adj, training))

        x = F.elu(x_hat)
        return x