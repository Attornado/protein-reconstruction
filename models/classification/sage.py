#
# Copyright (C)  2020  University of Pisa
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.pool import global_max_pool
from models.classification.classifiers import GraphClassifier


class SAGEClassifier(GraphClassifier):
    def __init__(self, dim_features: int, dim_target: int, config: dict):
        super().__init__(dim_features=dim_features, dim_target=dim_target, config=config)

        num_layers = config['num_layers']
        dim_embedding = config['dim_embedding']
        return_embeddings: bool = config["return_embeddings"]

        self.aggregation = config['aggregation']  # can be mean or max
        if self.aggregation == 'max':
            self.fc_max = nn.Linear(dim_embedding, dim_embedding)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_features if i == 0 else dim_embedding

            conv = SAGEConv(dim_input, dim_embedding)
            # Overwrite aggregation method (default is set to mean
            conv.aggr = self.aggregation

            self.layers.append(conv)

        self.fc1 = None
        self.fc2 = None
        if not return_embeddings:
            # For graph classification
            self.fc1 = nn.Linear(num_layers * dim_embedding, dim_embedding)
            self.fc2 = nn.Linear(dim_embedding, dim_target)

    @property
    def return_embeddings(self) -> bool:
        return self.config_dict["return_embeddings"]

    @return_embeddings.setter
    def return_embeddings(self, return_embeddings: bool):
        self.config_dict["return_embeddings"] = return_embeddings

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor,
                return_embeddings: bool = False) -> torch.Tensor:
        # x, edge_index, batch = data.x, data.edge_index, data.batch

        # Get predictions from each SAGE layer
        x_all = []
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.aggregation == 'max':
                x = torch.relu(self.fc_max(x))
            x_all.append(x)

        # Concat all predictions
        x = torch.cat(x_all, dim=1)

        # Return embeddings if required
        if return_embeddings or self.return_embeddings:
            return x

        # Apply readout and fully-connected layers
        x = global_max_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
