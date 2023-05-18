from typing import final
import torch
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.nn.aggr import LSTMAggregation
from torch_geometric.nn.dense import Linear
from models.pretraining.encoders import RevGCNEncoder
from models.pretraining.gunet import GraphRevUNet
from models.classification.classifiers import GraphClassifier


ADD_POOL_AGGREGATION: final = "add_pool"
LSTM_AGGREGATION: final = "lstm"


class GraphRevUNetClassifier(GraphClassifier):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 num_convs: list[int],
                 dim_target: int,
                 aggregation: str = "add_pool",
                 dropout: float = 0.0,
                 pool_ratio: float = 0.5,
                 model_type: str = RevGCNEncoder.MODEL_TYPE,
                 **block_parameters):
        config = {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "num_convs": num_convs,
            "dropout": dropout,
            "pool_ratio": pool_ratio,
            "model_type": model_type,
            "aggregation": aggregation
        }
        config.update(**block_parameters)

        super().__init__(dim_features=in_channels, dim_target=dim_target, config=config)
        self._gunet = GraphRevUNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_convs=num_convs
        )
        if aggregation == LSTM_AGGREGATION:
            self._aggr = LSTMAggregation(in_channels=hidden_channels, out_channels=hidden_channels)
        else:
            self._aggr = global_add_pool
        self._lin = Linear(hidden_channels, dim_target)

    @property
    def aggregation(self) -> str:
        return self.config_dict["aggregation"]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self._gunet(
            x=x,
            edge_index=edge_index,
            batch=batch
        )
        x = self._aggr(x=x, batch=batch)
        return self._lin(x)
