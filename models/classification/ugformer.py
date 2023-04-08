from typing import final, Optional
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import to_dense_batch
from models.batch_utils import from_dense_batch
from models.classification.classifiers import GraphClassifier
from models.layers import GATConvBlock, GCNConvBlock, SAGEConvBlock
from torch.nn import TransformerEncoder
from torch_geometric.nn.dense import Linear
from torch.nn import TransformerEncoderLayer


GAT: final = "gat"
GCN: final = "gcn"
SAGE: final = "sage"


class UGFormerV2(GraphClassifier):
    def __init__(self,
                 hidden_size: int,
                 dim_features: int,
                 n_self_att_layers: int,
                 n_layers: int,
                 n_head: int,
                 dropout: float,
                 dim_target: int,
                 embedding_size: Optional[int] = None,
                 conv_type: str = GCN,
                 **conv_kwargs):
        config = {
            "hidden_size": hidden_size,
            "embedding_size": embedding_size,
            "n_self_att_layers": n_self_att_layers,
            "n_layers": n_layers,
            "n_head": n_head,
            "dropout": dropout,
            "conv_type": conv_type,
            **conv_kwargs
        }
        super(UGFormerV2, self).__init__(dim_features=dim_features, dim_target=dim_target, config=config)

        # Each layer consists of a number of self-attention layers
        # Attention and convolution layers
        self.ug_form_layers = torch.nn.ModuleList()
        self._projection = None if embedding_size is None else Linear(in_channels=dim_features,
                                                                      out_channels=embedding_size)
        self.layers = torch.nn.ModuleList()
        for _layer in range(self.n_layers):
            encoder_layers = TransformerEncoderLayer(
                d_model=self.embedding_size,
                nhead=self.n_head,
                dim_feedforward=self.ff_hidden_size,
                dropout=dropout,
                # batch_first=True
            )  # Default batch_first=False (seq, batch, feature), while batch_first=True means (batch, seq, feature)
            self.ug_form_layers.append(
                TransformerEncoder(
                    encoder_layers,
                    self.n_self_att_layers
                )
            )
            if conv_type == GAT:
                self.layers.append(
                    GATConvBlock(
                        in_channels=self.embedding_size,
                        out_channels=self.embedding_size,
                        concat=False,
                        **conv_kwargs
                    )
                )
            elif conv_type == GCN:
                self.layers.append(
                    GCNConvBlock(
                        in_channels=self.embedding_size,
                        out_channels=self.embedding_size,
                        **conv_kwargs
                    )
                )
            elif conv_type == SAGE:
                self.layers.append(
                    SAGEConvBlock(
                        in_channels=self.embedding_size,
                        out_channels=self.embedding_size,
                        **conv_kwargs
                    )
                )
            else:
                raise ValueError(f"conv_type must be one of {SAGE}, {GAT}, {GCN}. Got {conv_type}.")
        # Linear function1e-05

        self.predictions = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.predictions.append(
                Linear(
                    self.embedding_size,
                    self.dim_target
                )
            )
            self.dropouts.append(
                nn.Dropout(
                    dropout
                )
            )

    @property
    def embedding_size(self) -> int:
        return self.config_dict["embedding_size"] if self.config_dict["embedding_size"] is not None \
            else self.in_channels

    @property
    def hidden_size(self) -> int:
        return self.config_dict["hidden_size"]

    @property
    def ff_hidden_size(self) -> int:
        return self.config_dict["hidden_size"]

    @property
    def n_self_att_layers(self) -> int:
        return self.config_dict["n_self_att_layers"]

    @property
    def n_layers(self) -> int:
        return self.config_dict["n_layers"]

    @property
    def n_head(self) -> int:
        return self.config_dict["n_head"]

    @property
    def dropout(self) -> float:
        return self.config_dict["dropout"]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, *args, **kwargs):
        prediction_scores = 0
        input_tr = x if self._projection is None else self._projection(x)
        for layer_idx in range(self.n_layers):
            # Self-Attention over all nodes
            # input_tr = torch.unsqueeze(input_tr, 1)  # [seq_length, batch_size=1, dim] for pytorch transformer
            # [batch_size, seq_length, dim] shape
            input_tr, batch_mask = to_dense_batch(x=input_tr, batch=batch, fill_value=0)
            # Change tensor shape to match the transformer [seq_length, batch_size, dim]
            input_tr = einops.rearrange(input_tr, "b s f -> s b f")

            # Generate attention src padding mask negating the batch_mask, because it has to be True in the padding pos
            batch_mask_transformer = batch_mask == False
            input_tr = self.ug_form_layers[layer_idx](input_tr, src_key_padding_mask=batch_mask_transformer)

            # input_tr = torch.squeeze(input_tr, 1)
            # Reshape to [batch_size, seq_length, dim] and convert to PyG batch again
            input_tr = einops.rearrange(input_tr, "s b f -> b s f")
            input_tr, _ = from_dense_batch(dense_batch=input_tr, mask=batch_mask)

            # Convolution layer
            input_tr = self.layers[layer_idx](input_tr, edge_index, *args, **kwargs)
            input_tr = F.gelu(input_tr)
            # not needed dropout
            # input_tr = F.dropout(input_tr, self.dropout)

            # take a sum over all node representations to get graph representations
            graph_embedding = global_add_pool(input_tr, batch=batch)
            graph_embedding = self.dropouts[layer_idx](graph_embedding)

            # produce the final scores
            prediction_scores += self.predictions[layer_idx](graph_embedding)

        # return einops.rearrange(prediction_scores, "b s -> (b s)")
        return prediction_scores


'''def label_smoothing(true_labels: torch.Tensor, classes: int, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)

    return true_dist'''
