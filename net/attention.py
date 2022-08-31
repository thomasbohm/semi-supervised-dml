import math

import torch
from torch import nn
from torch_scatter import scatter_add, scatter_max, scatter_mean


class MultiHeadDotProduct(nn.Module):
    """
    Multi head attention like in transformers
    embed_dim: dimension of input embedding
    nhead: number of attention heads
    aggr: available are ['add', 'mean', 'max']
    dropout: use 0 if no dropout is needed
    """
    def __init__(self, embed_dim: int, nhead: int, aggr: str, dropout: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.hdim = embed_dim // nhead
        self.nhead = nhead
        
        if aggr == "add":
            self.aggr = lambda out, row, dim, x_size: scatter_add(out, row, dim=dim, dim_size=x_size)
        if aggr == "mean":
            self.aggr = lambda out, row, dim, x_size: scatter_mean(out, row, dim=dim, dim_size=x_size)
        else: #if aggr == "max":
            self.aggr = lambda out, row, dim, x_size: scatter_max(out, row, dim=dim, dim_size=x_size)

        # FC Layers for input
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # fc layer for concatenated output
        self.out = nn.Linear(embed_dim, embed_dim)

        self.reset_parameters()

    def forward(self, feats: torch.Tensor, edge_index: torch.Tensor):
        q = k = v = feats
        bs = q.size(0)

        # FC layer and split into heads --> h * bs * embed_dim
        k = self.k_linear(k).view(bs, self.nhead, self.hdim).transpose(0, 1)
        q = self.q_linear(q).view(bs, self.nhead, self.hdim).transpose(0, 1)
        v = self.v_linear(v).view(bs, self.nhead, self.hdim).transpose(0, 1)
        
        # perform multi-head attention
        feats = self._attention(q, k, v, edge_index, bs)
        # concatenate heads and put through final linear layer
        feats = feats.transpose(0, 1).contiguous().view(bs, self.nhead * self.hdim)
        feats = self.out(feats)

        return feats #, edge_index, edge_attr

    def _attention(self, q, k, v, edge_index, bs=None):
        r, c, e = edge_index[:, 0], edge_index[:, 1], edge_index.shape[0]

        scores = torch.matmul(
            q.index_select(1, c).unsqueeze(dim=-2),
            k.index_select(1, r).unsqueeze(dim=-1)
        )
        scores = scores.view(self.nhead, e, 1) / math.sqrt(self.hdim)
        scores = softmax(scores, c, 1, bs)
        scores = self.dropout(scores)
        
        out = scores * v.index_select(1, r)  # H x e x hdim
        out = self.aggr(out, c, 1, bs)  # H x bs x hdim
        if type(out) == tuple:
            out = out[0]
        return out

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.constant_(self.q_linear.bias, 0.)

        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.constant_(self.v_linear.bias, 0.)

        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.constant_(self.k_linear.bias, 0.)
        
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.)


def softmax(src, index, dim, dim_size, margin: float = 0.):
    src_max = torch.clamp(scatter_max(src.float(), index, dim=dim, dim_size=dim_size)[0], min=0.)
    src = (src - src_max.index_select(dim=dim, index=index)).exp()
    denom = scatter_add(src, index, dim=dim, dim_size=dim_size)
    out = src / (denom + (margin - src_max).exp()).index_select(dim, index)

    return out
