from typing import Optional
import torch
import torch.nn.functional as F
from torch_geometric.nn.dense import Linear
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.layers import SerializableModule
from sampled_softmax import *


DROPOUT_TRANSFORMER_BLOCK = 0.5


class UGformerV1(SerializableModule):

    def __init__(self,
                 vocab_size: int,
                 feature_dim_size: int,
                 ff_hidden_size: int,
                 sampled_num: int,
                 num_self_att_layers: int,
                 num_gnn_layers: int,
                 embed_dim: Optional[int] = None,
                 n_heads: int = 1,
                 dropout: float = 0.5,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        super(UGformerV1, self).__init__()

        # Init features
        self.__feature_dim_size: int = feature_dim_size
        self.__ff_hidden_size: int = ff_hidden_size
        self.__num_self_att_layers: int = num_self_att_layers  # a layer consists of a number of self-attention layers
        self.__num_gnn_layers: int = num_gnn_layers
        self.__vocab_size: int = vocab_size
        self.__sampled_num: int = sampled_num
        self.__device: torch.device = device
        self.__embed_dim: Optional[int] = embed_dim
        self.__n_heads: int = n_heads
        self.__dropout: float = dropout

        # Init projection layers if required
        self._projection: Linear = None if embed_dim is None else Linear(feature_dim_size, embed_dim)

        # Init transformer layers
        self._ugformer_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        for _ in range(self.num_gnn_layers):

            encoder_layers = TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=n_heads,
                dim_feedforward=self.ff_hidden_size,
                dropout=DROPOUT_TRANSFORMER_BLOCK
            )  # embed_dim must be divisible by num_heads
            self._ugformer_layers.append(TransformerEncoder(encoder_layers, self.num_self_att_layers))

        # Linear function
        self._dropout_layer = nn.Dropout(dropout)
        self._ss = SampledSoftmax(self.vocab_size, self.sampled_num, self.feature_dim_size * self.num_gnn_layers,
                                  self.device)

    @property
    def feature_dim_size(self) -> int:
        return self.__feature_dim_size

    @property
    def ff_hidden_size(self) -> int:
        return self.__ff_hidden_size

    @property
    def num_self_att_layers(self) -> int:
        return self.__num_self_att_layers

    @property
    def num_gnn_layers(self) -> int:
        return self.__num_gnn_layers

    @property
    def vocab_size(self) -> int:
        return self.__vocab_size

    @property
    def embed_dim(self) -> int:
        return self.__embed_dim if self.__embed_dim is not None else self.__feature_dim_size

    @property
    def device(self) -> torch.device:
        return self.__device

    @property
    def n_heads(self) -> int:
        return self.__n_heads

    @property
    def sampled_num(self) -> int:
        return self.__sampled_num

    @property
    def dropout(self) -> float:
        return self.__dropout

    def forward(self,
                x: torch.Tensor,
                sampled_neighbour_indices: torch.Tensor,
                input_y: torch.Tensor
                ) -> (torch.Tensor, torch.Tensor):
        output_vectors = []  # should test output_vectors = [X_concat]
        input_tr = F.embedding(sampled_neighbour_indices, x)

        # Do a projection if required
        input_tr = input_tr if self._projection is None else self._projection(input_tr)

        for layer_idx in range(self.num_gnn_layers):

            # Index 0 is to take just the node of interest
            output_tr = self._ugformer_layers[layer_idx](input_tr)[0]

            # New input for next layer
            input_tr = F.embedding(sampled_neighbour_indices, output_tr)
            output_vectors.append(output_tr)

        # Concat output vectors to get the final node embeddings
        output_vectors = torch.cat(output_vectors, dim=1)
        output_vectors = self._dropout_layer(output_vectors)

        # Get sample-softmax logit loss
        logits = self._ss(output_vectors, input_y)

        # Return both loss and output embeddings
        return logits, output_vectors

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        return {
            "feature_dim_size": self.__feature_dim_size,
            "vocab_size": self.__vocab_size,
            "ff_hidden_size": self.__ff_hidden_size,
            "sampled_num": self.__sampled_num,
            "num_self_att_layers": self.__num_self_att_layers,
            "num_gnn_layers": self.__num_gnn_layers,
            "embed_dim": self.__embed_dim,
            "n_heads": self.__n_heads,
            "dropout": self.__dropout,
            "device": self.__device
        }

