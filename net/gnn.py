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
        if kwargs['reduction_layer']:
            hidden_dim = 128
            layers += [
                nn.Linear(in_channels, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ]
            in_channels = 128
        else:
            hidden_dim = 512 // num_heads

        if kwargs['gnn_conv'] == 'GAT':
            for _ in range(kwargs['num_layers']):
                layers += [
                    geom_nn.GATConv(in_channels=in_channels, out_channels=hidden_dim, heads=num_heads),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ]
                in_channels = num_heads * hidden_dim
        else:
            for _ in range(kwargs['num_layers']):
                layers += [
                    MultiHeadDotProduct(in_channels, num_heads, aggr='add', dropout=0.1),
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
                nn.Linear(in_channels, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ]

        self.layers = nn.ModuleList(layers)
        if kwargs['gnn_fc']:
            self.fc = geom_nn.GATConv(embed_dim, output_dim, heads=num_heads, concat=False)
        else:
            self.fc = nn.Linear(embed_dim, output_dim)

        self.proxies = nn.parameter.Parameter(torch.randn((num_proxies, embed_dim))).to(device)
        self.device = device


    def forward(self, x, proxy_idx=None, return_proxies=False, kclosest=0, true_proxies=None):
        if proxy_idx is not None:
            proxies = self.proxies[proxy_idx]
        else:
            proxies = self.proxies
        num_proxies = proxies.shape[0]
        
        if kclosest > 0 and true_proxies is not None:
            # connect every sample with k closest proxies
            edge_index = self.get_kclosest_edge_index(x, proxies, kclosest, true_proxies).to(self.device)
        else:
            # connect every sample with every proxies
            edge_index = self.get_fully_connected_edge_index(x, proxies.shape[0]).to(self.device)
        
        # feats = proxies + x
        feats = torch.cat([proxies, x])

        for l in self.layers:
            if isinstance(l, (geom_nn.MessagePassing, MultiHeadDotProduct)):
                torch.use_deterministic_algorithms(False)
                feats = l(feats, edge_index)
                torch.use_deterministic_algorithms(True)
            else:
                feats = l(feats)

        if isinstance(self.fc, geom_nn.MessagePassing):
            preds = self.fc(feats, edge_index)
        else:
            preds = self.fc(feats)

        if not return_proxies:
            # do not return proxy predictions and features
            return preds[num_proxies:], feats[num_proxies:]
        else:
            return preds[num_proxies:], feats[num_proxies:], preds[:num_proxies], feats[:num_proxies]
    

    def get_kclosest_edge_index(self, nodes, proxies, kclosest, true_proxies):
        dist = torch.sqrt(((nodes[:, None, :] - proxies[None, :, :]) ** 2).sum(dim=2)) # (B, P)
        _, closest_idx = torch.topk(dist, kclosest, dim=1, largest=False) # (B, k)

        edges = []
        for i in range(nodes.shape[0]):
            for p in closest_idx[i]:
                edges.append([p, i])
                edges.append([i, p])
            if true_proxies[i] not in closest_idx[i]:
                edges.append([i, true_proxies[i]])
                edges.append([true_proxies[i], i])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return edge_index
    
    def get_fully_connected_edge_index(self, nodes, num_proxies):
        edges = []
        for i in range(num_proxies, num_proxies + nodes.shape[0]):
            for p in range(num_proxies):
                edges.append([p, i])
                edges.append([i, p])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return edge_index
