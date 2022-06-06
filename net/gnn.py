import torch
import torch.nn as nn

import torch_geometric.nn as geom_nn


class GNNModel(nn.Module):
    def __init__(self, num_proxies, device):
        super().__init__()

        self.layers = nn.ModuleList([
            geom_nn.GATConv(512, 512),
            nn.ReLU(),
            nn.Dropout(),    
            geom_nn.GATConv(512, 512),
            nn.ReLU(),
            nn.Dropout(),                             
        ])

        self.fc = nn.Linear(512, 100)

        self.proxies = nn.parameter.Parameter(torch.randn((num_proxies, 512))).to(device)
        self.num_proxies = num_proxies

        self.device = device
    

    def forward(self, x):
        # nodes = proxies + x
        x = torch.cat([self.proxies, x])
        # connect every sample with every proxy
        edge_index = self.get_edge_index(x).to(self.device)
        
        for layer in self.layers:
            if isinstance(layer, geom_nn.MessagePassing):
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
        print(edge_index)
        print()
        return edge_index
