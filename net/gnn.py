import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn

from net.attention import MultiHeadDotProduct


class GNNModel(nn.Module):
    def __init__(self, embed_dim, output_dim, num_proxies, num_heads, device):
        super().__init__()

        self.att = geom_nn.GATConv(embed_dim, embed_dim, heads=num_heads) 
        #self.att = MultiHeadDotProduct(embed_dim, nhead=num_heads, aggr='add', dropout=0.1)

        self.act = nn.ReLU()

        #self.linear1 = nn.Linear(embed_dim, 4*embed_dim)
        #self.linear2 = nn.Linear(4*embed_dim, embed_dim)
        #self.dropout_mlp = nn.Dropout(0.1)
        #self.act_mlp = nn.ReLU()

        #self.norm1 = nn.LayerNorm(embed_dim)
        #self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        #self.dropout2 = nn.Dropout(0.1)

        self.fc = nn.Linear(num_heads * embed_dim, output_dim)

        self.proxies = nn.parameter.Parameter(torch.randn((num_proxies, embed_dim))).to(device)
        self.num_proxies = num_proxies
        self.device = device


    def forward(self, x):
        # nodes = proxies + x
        feats = torch.cat([self.proxies, x])
        # connect every sample with every proxy
        edge_index = self.get_edge_index(feats).to(self.device)

        feats = self.att(feats, edge_index)
        feats = self.dropout1(feats)
        feats = self.act(feats)
        
        #feats = self.norm1(feats + feats2)

        #feats2 = self.linear2(self.dropout_mlp(self.act_mlp(self.linear1(feats))))
        #feats2 = self.dropout2(feats2)
        #feats = self.norm2(feats + feats2)

        preds = self.fc(feats)

        # do not return proxy predictions and features
        return preds[self.num_proxies:], feats[self.num_proxies:]
    

    def get_edge_index(self, nodes):
        edges = []
        num_samples = nodes.shape[0] - self.num_proxies
        for p in range(self.num_proxies):
            for i in range(num_samples):
                edges.append([p, i + self.num_proxies])
                edges.append([i + self.num_proxies, p])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()

        return edge_index
