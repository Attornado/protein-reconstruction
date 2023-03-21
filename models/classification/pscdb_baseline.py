from typing import Optional, final
import torch
import torch_geometric.nn as nn
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.nn.conv import GCNConv, GATConv, SAGEConv
from models.classification.classifiers import GraphClassifier
import torch.nn.functional as F


GCN_MODEL_TYPE: final = 'GCN'
GAT_MODEL_TYPE: final = 'GAT'
SAGE_MODEL_TYPE: final = 'SAGE'


class GNNBaselinePSCDB(GraphClassifier):
    def __init__(self, dim_features: int, dim_target: int, config: dict):
        super().__init__(dim_features=dim_features, dim_target=dim_target, config=config)

        model_name: str = str(config['model_name'])
        hidden_dim: int = int(config['hidden_dim'])
        out_dim: int = int(config['out_dim'])
        n_heads = int(config['n_heads'])

        if model_name == GCN_MODEL_TYPE:
            self.layer1 = GCNConv(in_channels=dim_features, out_channels=hidden_dim)
            self.layer2 = GCNConv(in_channels=hidden_dim, out_channels=out_dim)

        elif model_name == GAT_MODEL_TYPE:
            self.layer1 = GATConv(in_channels=dim_features, out_channels=hidden_dim, heads=n_heads,
                                  dropout=self.dropout)
            self.layer2 = GATConv(in_channels=hidden_dim * n_heads, out_channels=out_dim, heads=n_heads, concat=False,
                                  dropout=self.dropout)

        elif model_name == SAGE_MODEL_TYPE:
            self.layer1 = SAGEConv(dim_features, hidden_dim)
            self.layer2 = SAGEConv(hidden_dim, out_dim)

        self.decoder = nn.Linear(out_dim, dim_target)

    @property
    def dropout(self) -> float:
        return self.config_dict['dropout']

    @property
    def hidden_dim(self) -> int:
        return self.config_dict['hidden_dim']

    @property
    def out_dim(self) -> int:
        return self.config_dict['out_dim']

    @property
    def model_type(self) -> str:
        return self.config_dict['model_name']

    @property
    def n_heads(self) -> Optional[int]:
        if self.model_type == GAT_MODEL_TYPE:
            return self.config_dict['n_heads']
        return None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):

        x = x.float()
        x = F.elu(self.layer1(x.float(), edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_add_pool(x, batch=batch)
        x = self.decoder(x)
        return x
