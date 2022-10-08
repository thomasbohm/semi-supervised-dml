import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn

from net.attention import MultiHeadDotProduct


class GNNModel(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()

        embed_dim = kwargs['embed_dim']
        output_dim = kwargs['output_dim']
        num_heads = kwargs['num_heads']
        num_proxies = kwargs['num_proxies']

        in_channels = embed_dim
        layers = []
        if kwargs['gnn_conv'] == 'GAT':
            for _ in range(kwargs['num_layers']):
                layers += [
                    geom_nn.GATConv(in_channels=in_channels, out_channels=embed_dim, heads=num_heads),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ]
                in_channels = num_heads * embed_dim
        else:
            for _ in range(kwargs['num_layers']):
                layers += [
                    MultiHeadDotProduct(embed_dim, num_heads, aggr='add', dropout=0.1),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ]

        if kwargs['add_mlp']:
            layers += [
                nn.Linear(in_channels, 4 * embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(4 * embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ]
        elif in_channels != embed_dim:
            layers += [
                nn.Linear(in_channels, embed_dim)
            ]

        self.layers = nn.ModuleList(layers)
        if kwargs['gnn_fc']:
            self.fc = geom_nn.GATConv(embed_dim, output_dim, heads=num_heads, concat=False)
        else:
            self.fc = nn.Linear(embed_dim, output_dim)

        self.proxies = nn.parameter.Parameter(torch.randn((num_proxies, embed_dim))).to(device)
        self.num_proxies = num_proxies
        self.device = device


    def forward(self, x, return_proxies=False):
        # nodes = proxies + x
        feats = torch.cat([self.proxies, x])
        # connect every sample with every proxy
        edge_index = self.get_edge_index(feats).to(self.device)

        for l in self.layers:
            if isinstance(l, (geom_nn.MessagePassing, MultiHeadDotProduct)):
                feats = l(feats, edge_index)
            else:
                feats = l(feats)

        if isinstance(self.fc, geom_nn.MessagePassing):
            preds = self.fc(feats, edge_index)
        else:
            preds = self.fc(feats)

        if not return_proxies:
            # do not return proxy predictions and features
            return preds[self.num_proxies:], feats[self.num_proxies:]
        else:
            return preds[self.num_proxies:], feats[self.num_proxies:], preds[:self.num_proxies], feats[:self.num_proxies]
    

    def get_edge_index(self, nodes):
        edges = []
        num_samples = nodes.shape[0] - self.num_proxies
        for p in range(self.num_proxies):
            for i in range(num_samples):
                edges.append([p, i + self.num_proxies])
                edges.append([i + self.num_proxies, p])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()

        return edge_index
