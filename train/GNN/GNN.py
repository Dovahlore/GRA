from typing import Union

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.data import Data

class LargeGCN(nn.Module):
    def __init__(self,args, num_node_features ,num_out_features,dropout=0.3):
        super(LargeGCN, self).__init__()
        self.num_layers = args.num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First GCN layer
        self.convs.append(GCNConv(num_node_features, args.hidden_dim))
        self.bns.append(BatchNorm(args.hidden_dim))

        # Hidden GCN layers
        for _ in range(args.num_layers - 2):
            self.convs.append(GCNConv(args.hidden_dim, args.hidden_dim))
            self.bns.append(BatchNorm(args.hidden_dim))

        # Final GCN layer
        self.convs.append(GCNConv(args.hidden_dim, args.output_dim))
        self.dropout = dropout
        self.fc = nn.Linear(args.output_dim, num_out_features)

    def forward(self,data: Union[Data]):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer (no activation, as it may be used in different tasks)
        x = self.convs[-1](x, edge_index)

        x = self.fc(x).squeeze(1)
        return x