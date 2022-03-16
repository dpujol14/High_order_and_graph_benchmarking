import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix

class GraphAttentionLayer(nn.Module):
    """
    Extension of the vanilla_GAT layer (based on 3rd order interactions)
    """
    def __init__(self, hpars, in_feats, out_feats, edge_feat_dim, final=False):
        super(GraphAttentionLayer, self).__init__()
        self.hpars = hpars
        self.dropout = self.hpars.experiment.dropout

        self.out_feats = out_feats
        self.final = final
        self.edge_feat_dim = edge_feat_dim

        # Learning parameters of the node features
        self.W = nn.Parameter(torch.empty(size=(in_feats, self.out_feats)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Learning parameters of the edge features
        self.W_edge = nn.Parameter(torch.empty(size=(self.edge_feat_dim, self.edge_feat_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Learning parameters to map (Wh_i || Wh_j || edge_feats_ij)
        self.a = nn.Parameter(torch.empty(size=(2 * self.out_feats + self.edge_feat_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.hpars.experiment.alpha)

    def forward(self, h, edge_feats, edge_indices, adj, training=True):
        # Process the edges
        We = torch.matmul(edge_feats, self.W_edge)  # edge_feats.shape: (M, edge_feat_dim), Wh.shape: (M, edge_feats_dim)

        # Process the nodes
        Wh = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)

        e = self._prepare_attentional_mechanism_input(Wh, We, edge_indices)

        zero_vec = -9e15 *torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=training)
        h_prime = torch.matmul(attention, Wh)

        if not self.final:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh, We, edge_indices):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature + edge_feats_dim, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_feats, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_feats: -self.edge_feat_dim, :])

        # broadcast add
        e = Wh1 + Wh2.T

        # If necessary, consider also the edges (check if in this case that this is 0, it wont do anything anyway)
        if self.edge_feat_dim > 0:
            # Transformer the edges to We: (M,1)
            We = torch.matmul(We, self.a[-self.edge_feat_dim:, :])

            # Transform the edge embeddings into We: (N x N)
            N = Wh.shape[0]
            edge_indices = torch.stack([edge_indices[0], edge_indices[1]])
            edge_weights = torch.sparse_coo_tensor(edge_indices, torch.squeeze(We), (N, N))

            # add the edge embeddings
            e += edge_weights.to_dense()

        return self.leakyrelu(e)
