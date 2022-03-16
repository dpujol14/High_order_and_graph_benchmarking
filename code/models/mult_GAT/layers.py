import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix

class MultGraphAttentionLayer(nn.Module):
    """
    Simple vanilla_GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, hpars, in_feats, out_feats, edge_feat_dim, final=False):
        super(MultGraphAttentionLayer, self).__init__()
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
        nn.init.xavier_uniform_(self.W_edge.data, gain=1.414)

        # Learning parameters to map edge_feats_ij \in R^F -> R
        self.a = nn.Parameter(torch.empty(size=(self.edge_feat_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.hpars.experiment.alpha)

    def forward(self, h, edge_feats, edge_indices, adj, training=True):
        # Process the edges
        We = torch.matmul(edge_feats, self.W_edge)  # edge_feats.shape: (M, edge_feat_dim), Wh.shape: (M, edge_feats_dim)

        # Process the nodes (compute (HW_Q)(HW_K)') to form the full (N x N) matrix
        HW_q = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        HW_k = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        HW_v = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)

        # IMPORTANT: For now we do parameter sharing, but we may want to define self.W_Q and self.W_K
        attention = self._prepare_attentional_mechanism_input(HW_q, HW_k, We, edge_indices, adj)

        #zero_vec = -9e15 *torch.ones_like(e)
        #attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=training)
        h_prime = torch.matmul(attention, HW_v)

        if not self.final:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, HW_q, HW_k, We, edge_indices, adj):
        att_scores = torch.matmul(HW_q, HW_k.T)  # att_scores: (N x N)

        att_scores *= adj   # att_scores: (N x N)

        # How do we add the edges?? (For now we just add this information
        # If necessary, consider also the edges (check if in this case that this is 0, it wont do anything anyway)
        if self.edge_feat_dim > 0:
             # Transformer the edges to We: (M,1)
             We = torch.matmul(We, self.a)

             # Transform the edge embeddings into We: (N x N)
             N = att_scores.shape[0]
             edge_indices = torch.stack([edge_indices[0], edge_indices[1]])
             edge_weights = torch.sparse_coo_tensor(edge_indices, torch.squeeze(We), (N, N))

             # add the edge embeddings
             att_scores += edge_weights.to_dense()

        return self.leakyrelu(att_scores)
