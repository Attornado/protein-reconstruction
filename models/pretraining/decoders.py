from abc import ABC, abstractmethod
from typing import final
import torch
from models.layers import SerializableModule, PositionWiseFeedForward
from torch_geometric.nn.models import InnerProductDecoder


BLOCKS_DEFAULT: final = 4


class GraphDecoder(SerializableModule, ABC):

    def __init__(self, *args, **kwargs):
        """
        An abstract class representing a generic decoder to be used with GAE/VGAE/ARGAE/ARGVAE architectures.
        """
        super(GraphDecoder, self).__init__()

    @abstractmethod
    def forward_all(self, z, sigmoid: bool = True, *args, **kwargs):
        """
        Takes the latent space representation z and reconstructs a probabilistic adjacency matrix.

        :param z: the latent space representation of the nodes.
        :param sigmoid: whether or not to apply a sigmoid function on the final decoder output, normalizing it.
        :type sigmoid: bool

        :return a probabilistic adjacency matrix for the given input.
        """
        raise NotImplementedError("Any GraphDecoder module must implement forward_all() method.")


class PointwiseFeedForwardDecoder(GraphDecoder):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, activation: str = "gelu", is_gated: bool = False,
                 n_blocks: int = BLOCKS_DEFAULT):
        """
        A decoder for GAE/VGAE/ARGAE/ARGVAE architectures composing of point-wise FFN layers (from transformer
        architecture) with skip-connections and a final inner-product decoder.

        :param d_model: The dimension of the model
        :type d_model: int
        :param d_ff: The dimension of the feed-forward layers
        :type d_ff: int
        :param dropout: The amount of dropout to use
        :type dropout: float
        :param activation: The activation function to use, defaults to GELU
        :type activation: str (optional)
        :param is_gated: Whether to use a gated point-wise FF network, defaults to False
        :type is_gated: bool (optional)
        :param n_blocks: number of blocks in the decoder
        :type n_blocks: int
        """
        # TODO: test this
        super().__init__()

        # Store parameter attributes
        self.__d_model: int = d_model
        self.__d_ff: int = d_ff
        self.__dropout: dropout = dropout
        self.__activation: str = activation
        self.__is_gated: bool = is_gated
        self.__n_blocks: int = n_blocks

        # Create decoder layers
        self.blocks: torch.nn.ModuleList = torch.nn.ModuleList()
        for _ in range(0, n_blocks):
            block = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation,
                                            is_gated=is_gated)
            self.blocks.append(block)

        self.inner_product_decoder: InnerProductDecoder = InnerProductDecoder()

    @property
    def n_blocks(self) -> int:
        return self.__n_blocks

    @property
    def dropout(self) -> float:
        return self.__dropout

    @property
    def d_model(self) -> int:
        return self.__d_model

    @property
    def d_ff(self) -> int:
        return self.__d_ff

    @property
    def activation(self) -> str:
        return self.__activation

    @property
    def is_gated(self) -> bool:
        return self.__is_gated

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        return {
            "d_model": self.__d_model,
            "d_ff": self.__d_ff,
            "dropout": self.__dropout,
            "activation": self.__activation,
            "is_gated": self.__is_gated,
            "n_blocks": self.__n_blocks
        }

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, sigmoid: bool = True):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`, applying the point-wise FF blocks and then an inner product decoder.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            edge_index (Tensor): the edge index representing to take into account.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        # TODO: test this

        # Apply point-wise FFN blocks
        for block in self.blocks:
            z = block(z, add_norm=True)

        # Apply inner product decoder
        return self.inner_product_decoder(z, edge_index, sigmoid=sigmoid)

    def forward_all(self, z, sigmoid: bool = True, *args, **kwargs):
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense adjacency matrix, applying the point-wise
        FF blocks and then an inner product decoder.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        # TODO: test this
        # Apply point-wise FFN blocks
        for block in self.blocks:
            z = block(z, add_norm=True)

        # Apply inner product decoder
        return self.inner_product_decoder.forward_all(z, sigmoid=sigmoid)
