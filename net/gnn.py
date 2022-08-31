import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn

from net.attention import MultiHeadDotProduct


class GNNModel(nn.Module):
    def __init__(self, embed_dim, output_dim, num_proxies, device):
        super().__init__()

        self.layers = nn.ModuleList([
            MultiHeadDotProduct(embed_dim, nhead=2, aggr='add', dropout=0.0),
            #geom_nn.GATConv(embed_dim, embed_dim, heads=2),
            nn.ReLU(),
            nn.Dropout(),                                 
        ])

        self.fc = nn.Linear(embed_dim, output_dim)

        self.proxies = nn.parameter.Parameter(torch.randn((num_proxies, embed_dim))).to(device)
        self.num_proxies = num_proxies
        self.device = device


    def forward(self, x):
        # nodes = proxies + x
        x = torch.cat([self.proxies, x])
        # connect every sample with every proxy
        edge_index = self.get_edge_index(x).to(self.device)
        
        for layer in self.layers:
            if isinstance(layer, (geom_nn.MessagePassing, MultiHeadDotProduct)):
                x = layer(x, edge_index)
            else:
                x = layer(x)

        preds = self.fc(x)

        # do not return proxy predictions and features
        return preds[self.num_proxies:], x[self.num_proxies:]
    

    def get_edge_index(self, nodes):
        edges = []
        num_samples = nodes.shape[0] - self.num_proxies
        for p in range(self.num_proxies):
            for i in range(num_samples):
                edges.append([p, i + self.num_proxies])
                edges.append([i + self.num_proxies, p])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()

        return edge_index
