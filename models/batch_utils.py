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
import numpy as np
import networkx as nx
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse, to_dense_adj  # , scatter_
import einops


def construct_mask_indices(sizes):
    # num_rows, num_cols = sum(sizes), len(sizes)

    indices = []
    for i, size in enumerate(sizes):
        cum_sum = sum(sizes[:i])
        indices.append((cum_sum, cum_sum + size))
    return indices


def _make_block_diag(mats, mat_sizes):
    block_diag = torch.zeros(sum(mat_sizes), sum(mat_sizes))

    for i, (mat, size) in enumerate(zip(mats, mat_sizes)):
        cum_size = sum(mat_sizes[:i])
        block_diag[cum_size:cum_size + size, cum_size:cum_size + size] = mat

    return block_diag


def make_block_diag(data):
    data = data.to_data_list()
    adjs = [to_dense_adj(d.edge_index).squeeze(0) for d in data]
    adj_sizes = [a.size(0) for a in adjs]
    bd_mat = _make_block_diag(adjs, adj_sizes)
    mask_indices = construct_mask_indices(adj_sizes)
    return bd_mat, mask_indices


def get_adj(block_diag, index):
    from_i, to_i = index
    return block_diag[from_i:to_i, from_i:to_i]


def mock_batch(batch_size, x_bidim: bool = True):
    """construct PyG batch"""
    graphs = []
    while len(graphs) < batch_size:
        g = nx.erdos_renyi_graph(np.random.choice([300, 350, 400, 450, 500]), 0.3)
        if g.number_of_edges() > 1:
            graphs.append(g)

    adjs = [torch.from_numpy(nx.to_numpy_array(g)) for g in graphs]
    graph_data = [dense_to_sparse(a) for a in adjs]
    if x_bidim:
        data_list = [Data(x=torch.randn(len(adjs[i][0]), 10), edge_index=e) for i, (e, _) in enumerate(graph_data)]
    else:
        data_list = [Data(x=x, edge_index=e) for (e, x) in graph_data]
    return Batch.from_data_list(data_list)


def from_dense_batch(dense_batch: torch.Tensor, mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    # dense batch, B, N, F
    # mask, B, N
    # B, N, F = dense_batch.size()
    flatten_dense_batch = einops.rearrange(dense_batch, "b s f -> (b s) f")
    flatten_mask = einops.rearrange(mask, "b s -> (b s)")
    data_x = flatten_dense_batch[flatten_mask, :]
    num_nodes = torch.sum(mask, dim=1)  # B, like 3,4,3
    pr_value = torch.cumsum(num_nodes, dim=0)  # B, like 3,7,10
    indicator_vector = torch.zeros(int(torch.sum(num_nodes, dim=0)))
    indicator_vector[pr_value[:-1]] = 1  # num_of_nodes, 0,0,0,1,0,0,0,1,0,0,1
    data_batch = torch.cumsum(indicator_vector, dim=0)  # num_of_nodes, 0,0,0,1,1,1,1,1,2,2,2
    return data_x, data_batch


def generate_batch_cross_attention_mask(batch_padding_mask_query: torch.Tensor,
                                        batch_padding_mask_key: torch.Tensor,
                                        num_heads: int = 1) -> torch.BoolTensor:
    """
    This function generates a boolean mask to be used for multi-head attention (MHA) when computing the cross-attention
    between query and key tensors. The function takes in two padding masks, one for the query tensor and the other for
    the key tensor, as well as an optional argument specifying the number of heads to be used for MHA.

    :param batch_padding_mask_query: A tensor with shape (B, L), where B is the batch size and L is the maximum length
        of the query tensor sequence. This tensor contains True values in positions that should be masked, and False
        values elsewhere.
    :param batch_padding_mask_key: A tensor with shape (B, S), where B is the batch size and S is the maximum length of
        the key tensor sequence. This tensor contains True values in positions that should be masked, and False values
        elsewhere.
    :param num_heads: An optional integer specifying the number of heads to be used for multi-head attention.
        Default is 1.
    :return: A boolean tensor with shape (B*num_heads, L, S) that can be used as the cross-attention mask
        for MHA. The True positions in the mask indicate positions that should not be attended to, while the False
        positions indicate positions that can be attended to.
    """

    # Given tensor X with shape (B, L) and tensor Y with shape (B, S)
    # Reshape X to have shape (B, L, 1) and Y to have shape (B, 1, S)
    batch_padding_mask_query = batch_padding_mask_query.unsqueeze(-1)
    batch_padding_mask_key = batch_padding_mask_key.unsqueeze(1)

    # If num_heads is greater than 1
    if num_heads > 1:
        # Reshape X to have shape (B, 1, L, 1) and Y to have shape (B, 1, 1, S)
        batch_padding_mask_query = batch_padding_mask_query.unsqueeze(1)
        batch_padding_mask_key = batch_padding_mask_key.unsqueeze(1)

        # Replicate X and Y over each head, obtaining tensors with shapes (B, H, L, 1) and (B, H, 1, S)
        batch_padding_mask_query = batch_padding_mask_query.repeat(1, num_heads, 1, 1)
        batch_padding_mask_key = batch_padding_mask_key.repeat(1, num_heads, 1, 1)

    # Use broadcasting of AND to obtain the desired tensor Z with shape (B, L, S), or (B, H, L, S) if num_heads > 1
    cross_attn_mask = torch.logical_and(batch_padding_mask_query, batch_padding_mask_key)

    # Invert True and False since True positions are not allowed to attend in MHA
    cross_attn_mask = torch.logical_not(cross_attn_mask)

    # If num_heads is greater than 1
    if num_heads > 1:
        # Aggregate batch and head dimensions, creating a tensor with shape (B*H, L, S)
        cross_attn_mask = einops.rearrange(cross_attn_mask, "b h l s -> (b h) l s")

    # noinspection PyTypeChecker
    return cross_attn_mask


def generate_batch_cross_attention_mask_v2(batch_index_query: torch.Tensor,
                                           batch_index_key: torch.Tensor) -> torch.BoolTensor:
    """
    This function generates a boolean mask to be used for multi-head attention (MHA) when computing the cross-attention
    between query and key tensors. The function takes in two batch indexes, one for the query tensor and the other for
    the key tensor, and returns a boolean mask bi-dimensional mask where  the [i, j] position is True, it means that the
    i-th query node cannot attend to the j-th key node, meaning that batch_index_query[i] != batch_index_key[j], while
    if the [i, j] position is False, then batch_index_query[i] = batch_index_key[j].

    :param batch_index_query: A tensor with shape (L, ) mapping each node in the query batch to corresponding
        graph (e.g. [0, 0, 0, 1, 1, 2, 2] indicates the the first 3 nodes belong to the graph 0, the subsequent 2 belong
        to the graph 1, and the last 2 belong to the graph 2).
    :param batch_index_key: A tensor with shape (S, ) mapping each node in the key batch to corresponding graph.

    :return: A boolean tensor with shape (L, S) that can be used as the cross-attention mask for MHA. The True positions
        in the mask indicate positions that should not be attended to, while the False positions indicate positions that
        can be attended to. That is, since each position corresponds to a node, that if the [i, j] position is True, it
        means that the i-th query node cannot attend to the j-th key node, and viceversa.
    """

    # Given tensor X with shape (L, ) and tensor Y with shape (S, )
    # Reshape X to have shape (L, 1) and Y to have shape (1, S)
    batch_index_query = batch_index_query.unsqueeze(-1)
    batch_index_key = batch_index_key.unsqueeze(0)

    # Use broadcasting of == operator between X and Y to obtain the desired tensor Z with shape (L, S)
    cross_attn_mask = batch_index_query.eq(batch_index_key)

    # Invert True and False since True positions are not allowed to attend in MHA
    cross_attn_mask = torch.logical_not(cross_attn_mask)

    # noinspection PyTypeChecker
    return cross_attn_mask


def test():
    batch_size = 3
    data = mock_batch(batch_size=batch_size)

    dense_data, mask = to_dense_batch(data.x, data.batch)
    output_data, output_batch = from_dense_batch(dense_data, mask)
    print((data.x.eq(output_data)).all())
    print((data.batch == output_batch).all())
    # create block diagonal matrix of batch
    # block size: [nodes_in_batch] x [nodes_in_batch]
    block_diag, indices = make_block_diag(data)
    for i in range(batch_size):
        graph_adj = get_adj(block_diag, indices[i])
        print(graph_adj)
