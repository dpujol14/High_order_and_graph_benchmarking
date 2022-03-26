import os
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