import os
from typing import final, Callable, Type, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.nn.aggr import LSTMAggregation, SoftmaxAggregation, MaxAggregation, MeanAggregation, SumAggregation
from torch_geometric.nn.dense import Linear  # , dense_diff_pool this has to be in the "future work"
from models.classification.classifiers import GraphClassifier
from models.layers import SerializableModule
from functools import partial
from torchmetrics.functional import accuracy, f1_score, precision, recall
from training.training_tools import FIGURE_SIZE_DEFAULT, EarlyStopping, MetricsHistoryTracer, EARLY_STOP_PATIENCE


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

    __ACTIVATIONS: final = {
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
    ACTIVATIONS: final = frozenset(__ACTIVATIONS.keys())

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
            if dense_activations[i] not in self.__ACTIVATIONS:
                raise ValueError(f"Activation function must be one of {self.ACTIVATIONS}, {dense_activations[i]} given")

            # Build dense layer
            self._dense_layers.append(Linear(prev_units, dense_units[i]))
            prev_units = dense_units[i]

    def forward(self, x, edge_index, batch_index: Tensor = None, *args, **kwargs):

        # Extract features with encoder
        x = self._encoder(x, edge_index, *args, **kwargs)

        # Apply readout aggregation, assuming batch is a single graph if batch_index is not given
        if batch_index is None:
            batch_index = torch.zeros(size=(x.shape[-2],)).type(torch.int64)
        x = self._readout_aggregation(x, index=batch_index)

        for i in range(0, len(self._dense_layers)):

            # Apply dense layer and activation
            dense = self._dense_layers[i]
            activation = self.__ACTIVATIONS[self.__dense_activations[i]]
            x = dense(x)
            x = activation(x)

            # Apply dropout on the activation, except if it is the output one
            if i < len(self._dense_layers) - 1:
                x = F.dropout(x, p=self.__dropout, training=self.training)

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
        self._multi_head_attention = torch.nn.MultiheadAttention(embed_dim=encoder_out_channels, num_heads=num_heads,
                                                                 dropout=dropout, kdim=kdim, vdim=vdim,
                                                                 batch_first=False)
        self.__vdim: Optional[int] = vdim
        self.__kdim: Optional[int] = kdim