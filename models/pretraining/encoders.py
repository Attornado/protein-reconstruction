from typing import Optional, Union, Any
import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear
from torch_geometric.nn.models import GroupAddRev
from models.layers import SAGEConvBlock, GATConvBlock, GCNConvBlock, GCN2ConvBlock, SerializableModule


class RevSAGEConvEncoder(SerializableModule):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_convs: int = 1,
                 dropout: float = 0.0, project: bool = False, root_weight: bool = True,
                 aggr: Optional[Union[str, list[str]]] = "mean", num_groups: int = 2, normalize_hidden: bool = True):
        super().__init__()

        self.dropout = dropout
        self.__in_channels = in_channels
        self.__hidden_channels = hidden_channels
        self.__out_channels = out_channels
        self.__normalize_hidden = normalize_hidden
        self.__project = project
        self.__root_weight = root_weight
        self.__aggr = aggr
        self.__num_groups = num_groups
        self.lin1 = None
        self.lin2 = None
        self.norm = None

        if in_channels != hidden_channels:
            self.lin1 = Linear(in_channels, hidden_channels)

        if hidden_channels != out_channels:
            self.lin2 = Linear(hidden_channels, out_channels)

        if normalize_hidden:
            self.norm = LayerNorm(hidden_channels, elementwise_affine=True)

        if hidden_channels % num_groups != 0:
            raise ValueError(
                f"hidden_channels must be divisible by num_groups, given {hidden_channels} and {num_groups}"
            )

        self.convs = torch.nn.ModuleList()
        for _ in range(num_convs):
            conv = SAGEConvBlock(
                in_channels=hidden_channels // num_groups,
                out_channels=hidden_channels // num_groups,
                project=project,
                bias=True,
                aggr=aggr,
                root_weight=root_weight
            )
            self.convs.append(GroupAddRev(conv, num_groups=num_groups))

    @property
    def in_channels(self) -> int:
        return self.__in_channels

    @property
    def out_channels(self) -> int:
        return self.__out_channels

    @property
    def hidden_channels(self) -> int:
        return self.__hidden_channels

    @property
    def normalize_hidden(self) -> bool:
        return self.__normalize_hidden

    @property
    def project(self) -> bool:
        return self.__project

    @property
    def root_weight(self) -> bool:
        return self.__root_weight

    @property
    def aggr(self) -> Union[str, list[str]]:
        return self.__aggr

    @property
    def num_groups(self) -> int:
        return self.__num_groups

    def reset_parameters(self):
        if self.lin1 is not None:
            self.lin1.reset_parameters()

        if self.lin2 is not None:
            self.lin2.reset_parameters()

        if self.norm is not None:
            self.norm.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):

        # Apply first projection if required
        if self.lin1 is not None:
            x = self.lin1(x)

        # Generate a dropout mask which will be shared across GNN blocks
        mask = None
        if self.training and self.dropout > 0:
            mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            mask = mask.requires_grad_(False)
            mask = mask / (1 - self.dropout)

        # Apply conv layers
        for conv in self.convs:
            x = conv(x, edge_index, mask)

        # Normalize if required
        if self.norm is not None:
            x = self.norm(x).relu()
        else:
            x = F.relu(x)

        # Apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply second projection if required
        if self.lin2 is not None:
            x = self.lin2(x)

        return x

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        params_dict = {
            "in_channels": self.__in_channels,
            "hidden_channels": self.__hidden_channels,
            "out_channels": self.__out_channels,
            "num_convs": len(self.convs),
            "dropout": self.dropout,
            "project": self.__project,
            "root_weight": self.__root_weight,
            "aggr": self.__aggr,
            "num_groups": self.__num_groups,
            "normalize_hidden": self.__normalize_hidden
        }

        return params_dict


class RevGATConvEncoder(SerializableModule):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_convs: int = 1,
                 dropout: float = 0.0, version: str = "v2", edge_dim: Optional[int] = None, heads: int = 1,
                 num_groups: int = 2, normalize_hidden: bool = True):
        super().__init__()

        self.dropout = dropout
        self.__in_channels = in_channels
        self.__hidden_channels = hidden_channels
        self.__out_channels = out_channels
        self.__normalize_hidden = normalize_hidden
        self.__version = version
        self.__edge_dim = edge_dim
        self.__heads = heads
        self.__num_groups = num_groups
        self.lin1 = None
        self.lin2 = None
        self.norm = None

        if in_channels != hidden_channels:
            self.lin1 = Linear(in_channels, hidden_channels)

        if hidden_channels != out_channels:
            self.lin2 = Linear(hidden_channels, out_channels)

        if normalize_hidden:
            self.norm = LayerNorm(hidden_channels, elementwise_affine=True)

        # We can use num_groups = 1 to get the residual connection but not the groups
        if hidden_channels % num_groups != 0:
            raise ValueError(
                f"hidden_channels must be divisible by num_groups, given {hidden_channels} and {num_groups}"
            )

        self.convs = torch.nn.ModuleList()
        for _ in range(num_convs):
            conv = GATConvBlock(
                in_channels=hidden_channels // num_groups,
                out_channels=hidden_channels // num_groups,
                version=version,
                heads=heads,
                edge_dim=edge_dim,
                bias=True,
                add_self_loops=True,
                negative_slope=0.2,
                concat=True,
                fill_value='mean'
            )
            self.convs.append(GroupAddRev(conv, num_groups=num_groups))

    @property
    def in_channels(self) -> int:
        return self.__in_channels

    @property
    def out_channels(self) -> int:
        return self.__out_channels

    @property
    def hidden_channels(self) -> int:
        return self.__hidden_channels

    @property
    def normalize_hidden(self) -> bool:
        return self.__normalize_hidden

    @property
    def version(self) -> str:
        return self.__version

    @property
    def edge_dim(self) -> Optional[int]:
        return self.__edge_dim

    @property
    def heads(self) -> int:
        return self.__heads

    @property
    def num_groups(self) -> int:
        return self.__num_groups

    def reset_parameters(self):
        if self.lin1 is not None:
            self.lin1.reset_parameters()

        if self.lin2 is not None:
            self.lin2.reset_parameters()

        if self.norm is not None:
            self.norm.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_attr):

        # Apply first projection if required
        if self.lin1 is not None:
            x = self.lin1(x)

        # Generate a dropout mask which will be shared across GNN blocks
        mask = None
        if self.training and self.dropout > 0:
            mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            mask = mask.requires_grad_(False)
            mask = mask / (1 - self.dropout)

        # Apply conv layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr, dropout_mask=mask)

        # Normalize if required
        if self.norm is not None:
            x = self.norm(x).relu()
        else:
            x = F.relu(x)

        # Apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply second projection if required
        if self.lin2 is not None:
            x = self.lin2(x)

        return x

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        params_dict = {
            "in_channels": self.__in_channels,
            "hidden_channels": self.__hidden_channels,
            "out_channels": self.__out_channels,
            "num_convs": len(self.convs),
            "dropout": self.dropout,
            "version": self.__version,
            "edge_dim": self.__edge_dim,
            "heads": self.__heads,
            "num_groups": self.__num_groups,
            "normalize_hidden": self.__normalize_hidden
        }
        return params_dict


class SimpleGCNEncoder(SerializableModule):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, conv_dims: list[int],
                 dropout: float = 0.0, improved: bool = False, cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, bias: bool = True, normalize_hidden: bool = True):
        super().__init__()

        self.dropout = dropout
        self.__in_channels = in_channels
        self.__hidden_channels = hidden_channels
        self.__out_channels = out_channels
        self.__normalize_hidden = normalize_hidden
        self.__conv_dims = conv_dims
        self.__improved = improved
        self.__cached = cached
        self.__add_self_loops = add_self_loops
        self.__normalize = normalize
        self.__bias = bias
        self.lin1 = None
        self.lin2 = None
        self.norm = None

        if in_channels != hidden_channels:
            self.lin1 = Linear(in_channels, hidden_channels)

        if hidden_channels != out_channels:
            self.lin2 = Linear(conv_dims[-1], out_channels)

        if normalize_hidden:
            self.norm = LayerNorm(conv_dims[-1], elementwise_affine=True)

        self.convs = torch.nn.ModuleList()
        prev_dim = hidden_channels
        for dim in conv_dims:
            conv = GCNConvBlock(
                in_channels=prev_dim,
                out_channels=dim,
                improved=improved,
                cached=cached,
                add_self_loops=add_self_loops,
                normalize=normalize,
                bias=bias
            )
            self.convs.append(conv)

    @property
    def in_channels(self) -> int:
        return self.__in_channels

    @property
    def out_channels(self) -> int:
        return self.__out_channels

    @property
    def hidden_channels(self) -> int:
        return self.__hidden_channels

    @property
    def normalize_hidden(self) -> bool:
        return self.__normalize_hidden

    @property
    def conv_dims(self) -> list[int]:
        return self.__conv_dims

    @property
    def improved(self) -> bool:
        return self.__improved

    @property
    def cached(self) -> bool:
        return self.__cached

    @property
    def add_self_loops(self) -> bool:
        return self.__add_self_loops

    @property
    def normalize(self) -> bool:
        return self.__normalize

    @property
    def bias(self) -> bool:
        return self.__bias

    def reset_parameters(self):
        if self.lin1 is not None:
            self.lin1.reset_parameters()

        if self.lin2 is not None:
            self.lin2.reset_parameters()

        if self.norm is not None:
            self.norm.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):

        # Apply first projection if required
        if self.lin1 is not None:
            x = self.lin1(x)

        # Generate a dropout mask which will be shared across GNN blocks
        mask = None
        if self.training and self.dropout > 0:
            mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            mask = mask.requires_grad_(False)
            mask = mask / (1 - self.dropout)

        # Apply conv layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight, dropout_mask=mask)

        # Normalize if required
        if self.norm is not None:
            x = self.norm(x).relu()
        else:
            x = F.relu(x)

        # Apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply second projection if required
        if self.lin2 is not None:
            x = self.lin2(x)

        return x

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        params_dict = {
            "in_channels": self.__in_channels,
            "hidden_channels": self.__hidden_channels,
            "out_channels": self.__out_channels,
            "conv_dims": self.__conv_dims,
            "dropout": self.dropout,
            "improved": self.__improved,
            "cached": self.__cached,
            "add_self_loops": self.__add_self_loops,
            "normalize": self.__normalize,
            "bias": self.__bias,
            "normalize_hidden": self.__normalize_hidden
        }
        return params_dict


class GCN2ConvResEncoder(SerializableModule):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, alpha: float, num_convs: int = 1,
                 dropout: float = 0.0, shared_weights: bool = True, cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, normalize_hidden: bool = True):
        super().__init__()

        self.__in_channels = in_channels
        self.__hidden_channels = hidden_channels
        self.__out_channels = out_channels
        self.__alpha = alpha
        self.__num_convs = num_convs
        self.dropout = dropout
        self.__shared_weights = shared_weights
        self.__cached = cached
        self.__add_self_loops = add_self_loops
        self.__normalize = normalize
        self.__normalize_hidden = normalize_hidden
        self.lin1 = None
        self.lin2 = None
        self.norm = None

        if in_channels != hidden_channels:
            self.lin1 = Linear(in_channels, hidden_channels)

        if hidden_channels != out_channels:
            self.lin2 = Linear(hidden_channels, out_channels)

        if normalize_hidden:
            self.norm = LayerNorm(hidden_channels, elementwise_affine=True)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_convs):
            conv = GCN2ConvBlock(
                channels=hidden_channels,
                alpha=alpha,
                layer=None,
                theta=None,
                shared_weights=shared_weights,
                cached=cached,
                add_self_loops=add_self_loops,
                normalize=normalize
            )
            self.convs.append(conv)

    @property
    def in_channels(self) -> int:
        return self.__in_channels

    @property
    def out_channels(self) -> int:
        return self.__out_channels

    @property
    def hidden_channels(self) -> int:
        return self.__hidden_channels

    @property
    def normalize_hidden(self) -> bool:
        return self.__normalize_hidden

    @property
    def num_convs(self) -> int:
        return self.__num_convs

    @property
    def alpha(self) -> float:
        return self.__alpha

    @property
    def shared_weights(self) -> bool:
        return self.__shared_weights

    @property
    def cached(self) -> bool:
        return self.__cached

    @property
    def add_self_loops(self) -> bool:
        return self.__add_self_loops

    @property
    def normalize(self) -> bool:
        return self.__normalize

    def reset_parameters(self):
        if self.lin1 is not None:
            self.lin1.reset_parameters()

        if self.lin2 is not None:
            self.lin2.reset_parameters()

        if self.norm is not None:
            self.norm.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight):

        # Apply first projection if required
        if self.lin1 is not None:
            x = self.lin1(x)

        # Generate a dropout mask which will be shared across GNN blocks
        mask = None
        if self.training and self.dropout > 0:
            mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            mask = mask.requires_grad_(False)
            mask = mask / (1 - self.dropout)

        # Apply conv layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight, dropout_mask=mask)

        # Normalize if required
        if self.norm is not None:
            x = self.norm(x).relu()
        else:
            x = F.relu(x)

        # Apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply second projection if required
        if self.lin2 is not None:
            x = self.lin2(x)

        return x

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        params_dict = {
            "in_channels": self.__in_channels,
            "hidden_channels": self.__hidden_channels,
            "out_channels": self.__out_channels,
            "alpha": self.__alpha,
            "num_convs": self.__num_convs,
            "dropout": self.dropout,
            "shared_weights": self.__shared_weights,
            "cached": self.__cached,
            "add_self_loops": self.__add_self_loops,
            "normalize": self.__normalize,
            "normalize_hidden": self.__normalize_hidden
        }
        return params_dict
