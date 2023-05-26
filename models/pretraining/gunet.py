from functools import partial
from typing import Optional, Union, Tuple, final, Callable, List
import torch
import torch.nn.functional as F
from torch_geometric.nn.models import GraphUNet
from torch_geometric.nn.pool import TopKPooling
from torch_geometric.typing import PairTensor, OptTensor
from torch_geometric.utils import to_torch_coo_tensor, remove_self_loops, add_self_loops
from models.layers import SerializableModule
from models.pretraining.encoders import RevSAGEConvEncoder, RevGATConvEncoder, ResGCN2ConvEncoderV2, RevGCNEncoder


def to_torch_csr_tensor(edge_index: torch.Tensor,
                        edge_attr: Optional[torch.Tensor] = None,
                        size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    r"""Converts a sparse adjacency matrix defined by edge indices and edge
    attributes to a :class:`torch.sparse.Tensor` with layout
    `torch.sparse_csr`.
    See :meth:`~torch_geometric.utils.to_edge_index` for the reverse operation.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): The edge attributes.
            (default: :obj:`None`)
        size (int or (int, int), optional): The size of the sparse matrix.
            If given as an integer, will create a quadratic sparse matrix.
            If set to :obj:`None`, will infer a quadratic sparse matrix based
            on :obj:`edge_index.max() + 1`. (default: :obj:`None`)
        is_coalesced (bool): If set to :obj:`True`, will assume that
            :obj:`edge_index` is already coalesced and thus avoids expensive
            computation. (default: :obj:`False`)

    :rtype: :class:`torch.sparse.Tensor`

    Example:

        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])

        to_torch_csr_tensor(edge_index)

        tensor(crow_indices=tensor([0, 1, 3, 5, 6]),
               col_indices=tensor([1, 0, 2, 1, 3, 2]),
               values=tensor([1., 1., 1., 1., 1., 1.]),
               size=(4, 4), nnz=6, layout=torch.sparse_csr)

    """
    adj = to_torch_coo_tensor(edge_index=edge_index, edge_attr=edge_attr, size=size)
    return adj.to_sparse_csr()


class HierarchicalTopKRevEncoder(SerializableModule):
    MODEL_TYPES: final = frozenset([
        RevSAGEConvEncoder.MODEL_TYPE,
        RevGCNEncoder.MODEL_TYPE,
        RevGATConvEncoder.MODEL_TYPE
    ])

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_convs: list[int],
                 dropout: float = 0.0,
                 pool_ratios: Union[List[float], float] = 0.5,
                 model_type: str = RevGCNEncoder.MODEL_TYPE,
                 num_groups: int = 2,
                 **block_parameters):
        super().__init__()

        if dropout < 0 or dropout >= 1:
            raise ValueError(f"Dropout rate must be between 0 and 1 (last excluded). {dropout} given.")
        if isinstance(pool_ratios, float):
            pool_ratios = [pool_ratios for _ in range(0, len(num_convs))]
        for pool_ratio in pool_ratios:
            if pool_ratio < 0 or pool_ratio >= 1:
                raise ValueError(f"Pool ratios must be between 0 and 1 (last excluded). {pool_ratio} given.")
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"model_type must be one of {self.MODEL_TYPES}. {model_type} given.")

        self.__in_channels: int = in_channels
        self.__hidden_channels: int = hidden_channels
        self.__out_channels: int = out_channels
        self.__num_convs: list[int] = num_convs
        self.__dropout: float = dropout
        self.__pool_ratios: List[float] = pool_ratios
        self.__model_type: str = model_type
        self.__block_params: dict = block_parameters
        self.__num_groups: int = num_groups
        self._encoder_convs = torch.nn.ModuleList()
        self._pools = torch.nn.ModuleList()

        for i, n in enumerate(num_convs):
            encoder_block = None
            input_channels = in_channels if i == 0 else hidden_channels
            output_channels = out_channels if i == len(num_convs) - 1 else hidden_channels

            if model_type == RevSAGEConvEncoder.MODEL_TYPE:
                encoder_block = RevSAGEConvEncoder(in_channels=input_channels,
                                                   hidden_channels=hidden_channels,
                                                   out_channels=output_channels,
                                                   num_convs=n,
                                                   dropout=dropout,
                                                   num_groups=num_groups,
                                                   **block_parameters)
            elif model_type == RevGCNEncoder.MODEL_TYPE:
                encoder_block = RevGCNEncoder(in_channels=input_channels,
                                              hidden_channels=hidden_channels,
                                              out_channels=output_channels,
                                              num_convs=n,
                                              dropout=dropout,
                                              num_groups=num_groups,
                                              **block_parameters)
            elif model_type == RevGATConvEncoder.MODEL_TYPE:
                encoder_block = RevGATConvEncoder(in_channels=input_channels,
                                                  hidden_channels=hidden_channels,
                                                  out_channels=output_channels,
                                                  num_convs=n,
                                                  dropout=dropout,
                                                  edge_dim=1,
                                                  num_groups=num_groups,
                                                  **block_parameters)
            if i != 0:
                top_k_pooling = TopKPooling(in_channels=hidden_channels, ratio=pool_ratios[i - 1])
                self._pools.append(top_k_pooling)
            self._encoder_convs.append(encoder_block)

        self.reset_parameters()

    @property
    def depth(self):
        return len(self.__num_convs)

    @property
    def in_channels(self) -> int:
        return self.__in_channels

    @property
    def hidden_channels(self) -> int:
        return self.__hidden_channels

    @property
    def out_channels(self) -> int:
        return self.__out_channels

    @property
    def num_convs(self) -> list[int]:
        return self.__num_convs

    @property
    def dropout(self) -> float:
        return self.__dropout

    @property
    def pool_ratios(self) -> List[float]:
        return self.__pool_ratios

    @property
    def model_type(self) -> str:
        return self.__model_type

    @property
    def num_groups(self) -> int:
        return self.__num_groups

    @property
    def block_params(self) -> dict:
        return self.__block_params

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self._encoder_convs:
            conv.reset_parameters()
        for pool in self._pools:
            pool.reset_parameters()

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None,
                return_all: bool = False) -> \
            Union[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                  tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]]:

        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))

        if self.model_type == RevGCNEncoder.MODEL_TYPE or self.model_type == ResGCN2ConvEncoderV2.MODEL_TYPE:
            x = self._encoder_convs[0](x, edge_index, edge_weight=edge_weight)
        elif self.model_type == RevGATConvEncoder.MODEL_TYPE:
            x = self._encoder_convs[0](x, edge_index, edge_attr=edge_weight)
        elif self.model_type == RevSAGEConvEncoder.MODEL_TYPE:
            x = self._encoder_convs[0](x, edge_index)
        x = torch.relu(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth):

            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
            x, edge_index, edge_weight, batch_index, perm, _ = self._pools[i - 1](x, edge_index, edge_weight,
                                                                                  batch_index)

            if self.model_type == RevGCNEncoder.MODEL_TYPE or self.model_type == ResGCN2ConvEncoderV2.MODEL_TYPE:
                x = self._encoder_convs[i](x, edge_index, edge_weight=edge_weight)
            elif self.model_type == RevGATConvEncoder.MODEL_TYPE:
                x = self._encoder_convs[i](x, edge_index, edge_attr=edge_weight)
            elif self.model_type == RevSAGEConvEncoder.MODEL_TYPE:
                x = self._encoder_convs[i](x, edge_index)

            x = torch.relu(x)

            if torch.cuda.is_available():
                del _
                torch.cuda.empty_cache()

            if i < self.depth - 1:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        if return_all:
            return x, edge_index, edge_weight, batch_index, xs, edge_indices, edge_weights, perms

        # Empty cuda cache
        if torch.cuda.is_available():
            del xs
            del edge_indices
            del edge_weights
            del perms
            torch.cuda.empty_cache()

        return x, edge_index, edge_weight, batch_index

    @classmethod
    def augment_adj(cls,
                    edge_index: torch.Tensor,
                    edge_weight: torch.Tensor,
                    num_nodes: int) -> PairTensor:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        adj = to_torch_csr_tensor(edge_index, edge_weight,
                                  size=(num_nodes, num_nodes))
        adj = (adj @ adj).to_sparse_coo()
        edge_index, edge_weight = adj.indices(), adj.values()
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        param_dict = {
            "in_channels": self.in_channels,
            "hidden_channels": self.hidden_channels,
            "out_channels": self.out_channels,
            "num_convs": self.num_convs,
            "dropout": self.dropout,
            "pool_ratios": self.pool_ratios,
            "model_type": self.model_type,
            "num_groups": self.num_groups
        }
        param_dict.update(**self.block_params)
        return param_dict


class GraphRevUNet(SerializableModule):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_convs: list[int],
                 dropout: float = 0.0,
                 pool_ratio: float = 0.5,
                 model_type: str = RevGCNEncoder.MODEL_TYPE,
                 num_groups: int = 2,
                 **block_parameters):
        super().__init__()
        self._encoder = HierarchicalTopKRevEncoder(in_channels=in_channels,
                                                   hidden_channels=hidden_channels,
                                                   out_channels=hidden_channels,
                                                   num_convs=num_convs,
                                                   dropout=dropout,
                                                   pool_ratio=pool_ratio,
                                                   model_type=model_type,
                                                   num_groups=num_groups,
                                                   **block_parameters)

        # Create decoder convs
        self._up_convs = torch.nn.ModuleList()
        decoder_block = None
        for i, n in enumerate(num_convs):
            output_channels = out_channels if i == len(num_convs) - 1 else hidden_channels
            if model_type == RevSAGEConvEncoder.MODEL_TYPE:
                decoder_block = RevSAGEConvEncoder(in_channels=hidden_channels,
                                                   hidden_channels=hidden_channels,
                                                   out_channels=output_channels,
                                                   num_convs=n,
                                                   dropout=dropout,
                                                   num_groups=num_groups,
                                                   **block_parameters)
            elif model_type == RevGCNEncoder.MODEL_TYPE:
                decoder_block = RevGCNEncoder(in_channels=hidden_channels,
                                              hidden_channels=hidden_channels,
                                              out_channels=output_channels,
                                              num_convs=n,
                                              dropout=dropout,
                                              num_groups=num_groups,
                                              **block_parameters)
            elif model_type == RevGATConvEncoder.MODEL_TYPE:
                decoder_block = RevGATConvEncoder(in_channels=hidden_channels,
                                                  hidden_channels=hidden_channels,
                                                  out_channels=output_channels,
                                                  num_convs=n,
                                                  dropout=dropout,
                                                  edge_dim=1,
                                                  num_groups=num_groups,
                                                  **block_parameters)
            self._up_convs.append(decoder_block)

        self.reset_parameters()

    @property
    def depth(self):
        return self._encoder.depth

    @property
    def in_channels(self) -> int:
        return self._encoder.in_channels

    @property
    def hidden_channels(self) -> int:
        return self._encoder.hidden_channels

    @property
    def out_channels(self) -> int:
        return self._encoder.out_channels

    @property
    def num_convs(self) -> list[int]:
        return self._encoder.num_convs

    @property
    def dropout(self) -> float:
        return self._encoder.dropout

    @property
    def pool_ratio(self) -> float:
        return self._encoder.pool_ratio

    @property
    def model_type(self) -> str:
        return self._encoder.model_type

    @property
    def num_groups(self) -> int:
        return self._encoder.num_groups

    @property
    def block_params(self) -> dict:
        return self._encoder.block_params

    def reset_parameters(self):
        self._encoder.reset_parameters()
        for conv in self._up_convs:
            conv.reset_parameters()

    def hierarchical_encode(self,
                            x: torch.Tensor,
                            edge_index: torch.Tensor,
                            batch_index: torch.Tensor,
                            edge_weight: Optional[torch.Tensor] = None,
                            return_all: bool = False) -> \
            Union[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                  tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]]:
        return self._encoder(x=x,
                             edge_index=edge_index,
                             batch_index=batch_index,
                             edge_weight=edge_weight,
                             return_all=return_all)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        x, edge_index, edge_weight, batch_index, xs, edge_indices, edge_weights, perms = self.hierarchical_encode(
            x=x,
            edge_index=edge_index,
            batch_index=batch_index,
            edge_weight=edge_weight,
            return_all=True
        )

        for i in range(self.depth - 1):

            # Take corresponding encoder output
            j = self.depth - 2 - i
            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            # Perform skip-connection
            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up  # if self.sum_res else torch.cat((res, up), dim=-1)

            # Apply conv layer
            if self.model_type == RevGCNEncoder.MODEL_TYPE or self.model_type == ResGCN2ConvEncoderV2.MODEL_TYPE:
                x = self._up_convs[i](x, edge_index, edge_weight=edge_weight)
            elif self.model_type == RevGATConvEncoder.MODEL_TYPE:
                x = self._up_convs[i](x, edge_index, edge_attr=edge_weight)
            elif self.model_type == RevSAGEConvEncoder.MODEL_TYPE:
                x = self._up_convs[i](x, edge_index)
            x = torch.relu(x) if i < self.depth - 2 else x

        return x

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        return self._encoder.serialize_constructor_params()


class GraphUNetV2(SerializableModule):

    _ACTIVATIONS: final = {
        "linear": torch.nn.Identity(),
        "relu": F.relu,
        "leaky_relu": F.leaky_relu,
        "rrelu": F.rrelu,
        "relu6": F.relu6,
        "gelu": partial(F.gelu, approximate='none'),
        "elu": F.elu,
        "celu": F.celu,
        "glu": F.glu,
        "selu": F.selu,
        "prelu": F.prelu,
        "silu": F.silu,
        "hardswish": F.hardswish,
        "tanh": F.tanh,
        "sigmoid": torch.sigmoid
    }
    ACTIVATIONS: final = frozenset(_ACTIVATIONS.keys())

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 depth: int,
                 pool_ratios: Union[float, List[float]] = 0.5,
                 sum_res: bool = True,
                 act: str = "relu"):
        r"""The Graph U-Net model from the `"Graph U-Nets"
        <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
        architecture with graph pooling and unpooling operations.

        Args:
            in_channels (int): Size of each input sample.
            hidden_channels (int): Size of each hidden sample.
            out_channels (int): Size of each output sample.
            depth (int): The depth of the U-Net architecture.
            pool_ratios (float or [float], optional): Graph pooling ratio for each
                depth. (default: :obj:`0.5`)
            sum_res (bool, optional): If set to :obj:`False`, will use
                concatenation for integration of skip connections instead
                summation. (default: :obj:`True`)
            act (torch.nn.functional, optional): The nonlinearity to use.
                (default: :obj:`torch.nn.functional.relu`)
        """
        if act not in self.ACTIVATIONS:
            raise ValueError(f"act must be in {self.ACTIVATIONS}, {act} given.")
        super().__init__()

        # Store attributes
        self.__in_channels: int = in_channels
        self.__hidden_channels: int = hidden_channels
        self.__out_channels: int = out_channels
        self.__depth: int = depth
        self.__pool_ratios: Union[float, List[float]] = pool_ratios
        self.__sum_res: bool = sum_res
        self.__act: str = act

        # Initialize GraphUNet
        activation_fn = self._ACTIVATIONS[act]
        self._gunet = GraphUNet(in_channels=in_channels,
                                hidden_channels=hidden_channels,
                                out_channels=out_channels,
                                depth=depth,
                                pool_ratios=pool_ratios,
                                sum_res=sum_res,
                                act=activation_fn)

    @property
    def in_channels(self) -> int:
        return self.__in_channels

    @property
    def hidden_channels(self) -> int:
        return self.__hidden_channels

    @property
    def out_channels(self) -> int:
        return self.__out_channels

    @property
    def depth(self) -> int:
        return self.__depth

    @property
    def pool_ratios(self) -> Union[float, List[float]]:
        return self.__pool_ratios

    @property
    def sum_res(self) -> bool:
        return self.__sum_res

    @property
    def act(self) -> str:
        return self.__act

    @property
    def activation(self) -> Callable:
        return self._ACTIVATIONS[self.act]

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        return {
            "in_channels": self.in_channels,
            "hidden_channels": self.hidden_channels,
            "out_channels": self.out_channels,
            "depth": self.depth,
            "pool_ratios": self.pool_ratios,
            "sum_res": self.sum_res,
            "act": self.act
        }

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: OptTensor = None) -> torch.Tensor:
        return self._gunet(x=x, edge_index=edge_index, batch=batch)

    def augment_adj(self,
                    edge_index: torch.Tensor,
                    edge_weight: torch.Tensor,
                    num_nodes: int) -> PairTensor:
        return self._gunet.augment_adj(edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes)

    def reset_parameters(self):
        self._gunet.reset_parameters()

    def __repr__(self) -> str:
        return f"{self._gunet.__repr__()[:-1]}, act={self.act})"



