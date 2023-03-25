from typing import final, Callable, Type, Optional, Union
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import MultiheadAttention, TransformerEncoderLayer, TransformerEncoder
from torch_geometric.nn.aggr import LSTMAggregation, SoftmaxAggregation, MaxAggregation, MeanAggregation, SumAggregation
from torch_geometric.nn.dense import Linear  # , dense_diff_pool this has to be in the "future work"
from torch_geometric.utils import to_dense_batch
from models.classification.classifiers import GraphClassifier
from models.layers import SerializableModule
from functools import partial


CLASSES: final = 7


class ProtMotionNet(GraphClassifier):
    __READOUTS: final = {
        "mean_pool": MeanAggregation(),
        "max_pool": MaxAggregation(),
        "add_pool": SumAggregation(),
        'lstm': LSTMAggregation,
        'softmax': SoftmaxAggregation(learn=True)
    }
    READOUTS: final = frozenset(__READOUTS.keys())

    _ACTIVATIONS: final = {
        "linear": F.linear,
        "relu": F.relu,
        "leaky_relu": F.leaky_relu,
        "rrelu": F.rrelu,
        "relu6": F.relu6,
        "gelu": F.gelu,
        "elu": F.elu,
        "celu": F.celu,
        "glu": F.glu,
        "selu": F.selu,
        "prelu": F.prelu,
        "silu": F.silu,
        "hardswish": F.hardswish,
        "tanh": F.tanh,
        "sigmoid": torch.sigmoid,
        "softmax": partial(F.softmax, dim=-1)
    }
    ACTIVATIONS: final = frozenset(_ACTIVATIONS.keys())

    def __init__(self, encoder: SerializableModule, encoder_out_channels: int, dense_units: list[int],
                 dense_activations: list[str], dim_features: int, dropout: float = 0.0, readout: str = 'mean_pool'):

        super(ProtMotionNet, self).__init__(dim_features=dim_features,
                                            dim_target=dense_units[-1],
                                            config={"readout": readout, "dense_units": dense_units,
                                                    "dense_activations": dense_activations, "dropout": dropout}
                                            )

        # Validate input
        if len(dense_units) != len(dense_activations):
            raise ValueError(f"len(dense_activations) must be equal to len(dense_units),"
                             f" {len(dense_activations)} and {len(dense_units)} given")

        self._encoder = encoder
        self.__encoder_out_channels = encoder_out_channels
        self._readout_aggregation = self.__resolve_readout(readout, encoder_out_channels)
        # self.__dense_units = dense_units
        # self.__dense_activations = dense_activations
        # self.__dropout = dropout
        # self.__readout = readout

        # Build dense layers
        self._dense_layers = torch.nn.ModuleList()
        prev_units = encoder_out_channels
        for i in range(0, len(dense_units)):

            # Check if activations
            if dense_activations[i] not in self._ACTIVATIONS:
                raise ValueError(f"Activation function must be one of {self.ACTIVATIONS}, {dense_activations[i]} given")

            # Build dense layer
            self._dense_layers.append(Linear(prev_units, dense_units[i]))
            prev_units = dense_units[i]

    def _apply_dense_layers(self, x: Tensor) -> Tensor:

        for i in range(0, len(self._dense_layers)):

            # Apply dense layer and activation
            dense = self._dense_layers[i]
            activation = self._ACTIVATIONS[self.__dense_activations[i]]
            x = dense(x)
            x = activation(x)

            # Apply dropout on the activation, except if it is the output one
            if i < len(self._dense_layers) - 1:
                x = F.dropout(x, p=self.__dropout, training=self.training)

        return x

    def forward(self, x, edge_index, batch_index: Tensor = None, *args, **kwargs):

        # Extract features with encoder
        x = self._encoder(x, edge_index, *args, **kwargs)

        # Apply readout aggregation, assuming batch is a single graph if batch_index is not given
        if batch_index is None:
            batch_index = torch.zeros(size=(x.shape[-2],)).type(torch.int64)
        x = self._readout_aggregation(x, index=batch_index)

        x = self._apply_dense_layers(x)

        return x

    @property
    def dropout(self) -> float:
        return self.config_dict["dropout"]

    @property
    def dense_units(self) -> list[int]:
        return self.config_dict["dense_units"]

    @property
    def dense_activations(self) -> list[str]:
        return self.config_dict["dense_activations"]

    @property
    def readout(self) -> str:
        return self.config_dict["readout"]

    @property
    def encoder_out_channels(self) -> int:
        return self.__encoder_out_channels

    def serialize_constructor_params(self, *args, **kwargs) -> dict:

        # Serialize constructor parameters
        constructor_params = {
            "encoder_out_channels": self.encoder_out_channels,
            "dense_units": self.dense_units,
            "dense_activations": self.dense_activations,
            "dropout": self.dropout,
            "readout": self.readout
        }

        # Serialize encoder
        constructor_params["encoder"] = {
            "constructor_params": self._encoder.serialize_constructor_params(),
            "state_dict": self._encoder.state_dict()
        }

        return constructor_params

    # noinspection PyMethodOverriding
    @classmethod
    def from_constructor_params(cls,
                                constructor_params: dict,
                                encoder_constructor: Type[SerializableModule],
                                *args, **kwargs):
        # Deserialize encoder
        encoder_constructor_params = constructor_params["encoder"]["constructor_params"]
        encoder_state_dict = constructor_params["encoder"]["state_dict"]
        # TODO: Could not work if from_constructor_params is overridden by encoder class, must fix this to generalize
        encoder = encoder_constructor.from_constructor_params(encoder_constructor_params)
        encoder.load_state_dict(encoder_state_dict)
        del constructor_params["encoder"]  # delete encoder params from constructor_params

        return cls(encoder=encoder, **constructor_params)

    @classmethod
    def __resolve_readout(cls, readout: str, channels: int) -> Callable:
        if readout not in cls.READOUTS:
            raise ValueError(f"Readout function must be in {cls.READOUTS}")

        if readout == 'lstm':
            return LSTMAggregation(in_channels=channels, out_channels=channels)
        return cls.__READOUTS[readout]


class PairedProtMotionNet(ProtMotionNet):
    def __init__(self, encoder: SerializableModule, encoder_out_channels: int, dense_units: list[int],
                 dense_activations: list[str], dim_features: int, num_heads: int = 8, kdim: Optional[int] = None,
                 vdim: Optional[int] = None, dropout: float = 0.0, readout: str = 'mean_pool'):
        super(PairedProtMotionNet, self).__init__(encoder=encoder, encoder_out_channels=encoder_out_channels,
                                                  dense_units=dense_units, dense_activations=dense_activations,
                                                  dim_features=dim_features, dropout=dropout, readout=readout)
        self._multi_head_attention = MultiheadAttention(embed_dim=encoder_out_channels, num_heads=num_heads,
                                                        dropout=dropout, kdim=kdim, vdim=vdim, batch_first=False)
        self.__vdim: Optional[int] = vdim
        self.__kdim: Optional[int] = kdim
        self.__num_heads: int = num_heads

    @property
    def vdim(self) -> int:
        return self.__vdim if self.__vdim is not None else self.encoder_out_channels

    @property
    def kdim(self) -> int:
        return self.__kdim if self.__kdim is not None else self.encoder_out_channels

    @property
    def num_heads(self) -> int:
        return self.__num_heads

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        constructor_params = super(PairedProtMotionNet, self).serialize_constructor_params(*args, **kwargs)
        constructor_params.update({"num_heads": self.__num_heads, "vdim": self.__vdim, "kdim": self.__kdim})
        return constructor_params

    def _get_multi_head_attention_embeddings(self,
                                             x: Union[Tensor, tuple[Tensor, Tensor]],
                                             edge_index: Union[Tensor, tuple[Tensor, Tensor]],
                                             batch_index: Tensor = None, x1: Optional[Tensor] = None,
                                             edge_index1: Optional[Tensor] = None,
                                             batch_index1: Optional[Tensor] = None,
                                             *args, **kwargs):

        # Check input
        if isinstance(x, tuple) and len(x) < 2:
            raise ValueError(f"Exactly 2 x tensors must be given to the {self.__class__} forward() method. {len(x)} "
                             f"given.")
        elif not isinstance(x, tuple) and x1 is None:
            raise ValueError(f"Exactly 2 x tensors must be given to the {self.__class__} forward() method. Just 1 "
                             f"given.")

        if isinstance(edge_index, tuple) and len(edge_index) < 2:
            raise ValueError(f"Exactly 2 edge indexes must be given to the {self.__class__} forward() method. "
                             f"{len(edge_index)} given.")
        elif not isinstance(edge_index, tuple) and edge_index1 is None:
            raise ValueError(f"Exactly 2 edge index tensors must be given to the {self.__class__} forward() method. "
                             f"Just 1 given.")
        if batch_index is not None and isinstance(batch_index, tuple) and len(batch_index) < 2:
            raise ValueError(f"Exactly 2 batch index tensor must be given to the {self.__class__} forward() method. "
                             f"{len(batch_index)} given.")
        elif batch_index is not None and not isinstance(batch_index, tuple) and batch_index1 is None:
            raise ValueError(f"Exactly 2 batch index tensors must be given to the {self.__class__} forward() method. "
                             f"Just 1 given.")

        # Setup the two graphs
        if isinstance(x, tuple):
            x1: Tensor = x[1]
            x: Tensor = x[0]
        if isinstance(edge_index, tuple):
            edge_index1: Tensor = edge_index[1]
            edge_index: Tensor = edge_index[0]
        if batch_index is not None and isinstance(batch_index, tuple):
            batch_index1: Tensor = batch_index[1]
            batch_index: Tensor = batch_index[0]

        # Assume batch is a single graph if batch_index is not given
        elif batch_index is None:
            batch_index = torch.zeros(size=(x.shape[-2],)).type(torch.int64)
            batch_index1 = torch.zeros(size=(x1.shape[-2],)).type(torch.int64)

        # Extract features with encoder
        x = self._encoder(x, edge_index, *args, **kwargs)
        x1 = self._encoder(x1, edge_index1, *args, **kwargs)

        # Convert to dense batch
        x, mask0 = to_dense_batch(x=x, batch=batch_index, fill_value=0)
        x1, mask1 = to_dense_batch(x=x1, batch=batch_index1, fill_value=0)

        # TODO: Generate key padding mask and attention mask
        attn_mask = None
        key_padding_mask = None

        # Apply cross multi-head attention
        # TODO: check the longer between x and x1, so that the longer is used as query (using x1 should be fine however)
        x = self._multi_head_attention(query=x1, key=x, value=x, key_padding_mask=key_padding_mask, need_weights=False,
                                       attn_mask=attn_mask)

        # TODO: maybe add shortcut-connection with concat/add & the normalization in some way
        del x1  # no need to further memorize this

        return x, attn_mask, key_padding_mask

    def forward(self, x: Union[Tensor, tuple[Tensor, Tensor]], edge_index: Union[Tensor, tuple[Tensor, Tensor]],
                batch_index: Tensor = None, x1: Optional[Tensor] = None, edge_index1: Optional[Tensor] = None,
                batch_index1: Optional[Tensor] = None, *args, **kwargs):

        # Get cross multi-head attention embeddings
        x, _, _ = self._get_multi_head_attention_embeddings(x=x, edge_index=edge_index, batch_index=batch_index, x1=x1,
                                                            edge_index1=edge_index1, batch_index1=batch_index1, *args,
                                                            **kwargs)

        # Apply readout aggregation
        x = self._readout_aggregation(x, index=batch_index)

        # Apply dense layers for classification
        x = self._apply_dense_layers(x)

        return x


class TransformerPairedProtMotionNet(PairedProtMotionNet):
    def __init__(self, encoder: SerializableModule, encoder_out_channels: int, dense_units: list[int],
                 dense_activations: list[str], dim_features: int, n_blocks: int, num_heads: int = 8,
                 kdim: Optional[int] = None, vdim: Optional[int] = None, dropout: float = 0.0,
                 readout: str = 'mean_pool', d_ff: Optional[int] = None, ff_activation: str = "gelu",
                 pre_norm: bool = True):
        super(TransformerPairedProtMotionNet, self).__init__(encoder=encoder, encoder_out_channels=encoder_out_channels,
                                                             dense_activations=dense_activations,
                                                             dim_features=dim_features, dense_units=dense_units,
                                                             num_heads=num_heads, kdim=kdim, vdim=vdim, dropout=dropout,
                                                             readout=readout)
        transformer_block = TransformerEncoderLayer(
            d_model=encoder_out_channels,
            nhead=num_heads,
            dim_feedforward=d_ff if d_ff is not None else encoder_out_channels,
            activation=self._ACTIVATIONS[ff_activation],
            norm_first=pre_norm,
            batch_first=False,
            device=None
        )
        self._transformer_encoder = TransformerEncoder(encoder_layer=transformer_block, num_layers=n_blocks,
                                                       norm=None, enable_nested_tensor=True)

        self.__d_ff: Optional[int] = d_ff
        self.__pre_norm: bool = pre_norm
        self.__ff_activation: str = ff_activation
        self.__n_blocks: int = n_blocks

    @property
    def d_ff(self) -> int:
        return self.__d_ff if self.__d_ff is not None else self.encoder_out_channels

    @property
    def pre_norm(self) -> bool:
        return self.__pre_norm

    @property
    def ff_activation(self) -> str:
        return self.__ff_activation

    @property
    def n_blocks(self) -> int:
        return self.__n_blocks

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        constructor_params = super(TransformerPairedProtMotionNet, self).serialize_constructor_params(*args, **kwargs)
        constructor_params.update({"d_ff": self.__d_ff, "pre_norm": self.__pre_norm,
                                   "ff_activation": self.__ff_activation, "n_blocks": self.__n_blocks})
        return constructor_params

    def forward(self, x: Union[Tensor, tuple[Tensor, Tensor]], edge_index: Union[Tensor, tuple[Tensor, Tensor]],
                batch_index: Tensor = None, x1: Optional[Tensor] = None, edge_index1: Optional[Tensor] = None,
                batch_index1: Optional[Tensor] = None, *args, **kwargs):

        # Get cross multi-head attention embeddings
        x, attn_mask, key_padding_mask = self._get_multi_head_attention_embeddings(
            x=x,
            edge_index=edge_index,
            batch_index=batch_index,
            x1=x1,
            edge_index1=edge_index1,
            batch_index1=batch_index1, 
            *args,
            **kwargs
        )

        # Apply transformer encoder to further process the embeddings
        x = self._transformer_encoder(src=x, mask=attn_mask, src_key_padding_mask=key_padding_mask)

        # Apply readout aggregation
        x = self._readout_aggregation(x, index=batch_index)

        # Apply dense layers for classification
        x = self._apply_dense_layers(x)

        return x
