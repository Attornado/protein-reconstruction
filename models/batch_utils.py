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
    num_rows, num_cols = sum(sizes), len(sizes)

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


def generate_batch_cross_attention_mask(batch_padding_mask0: torch.Tensor,
                                        batch_padding_mask1: torch.Tensor) -> torch.BoolTensor:
    # Given tensor A with shape (B, L) and tensor B with shape (B, S)
    # Reshape A to have shape (B, L, 1) and B to have shape (B, 1, S)
    batch_padding_mask0 = batch_padding_mask0.unsqueeze(-1)
    batch_padding_mask1 = batch_padding_mask1.unsqueeze(1)

    # Use broadcasting to obtain the desired tensor C with shape (B, L, S)
    cross_attn_mask = batch_padding_mask0 & batch_padding_mask1

    # Invert True and False since True positions are not allowed to attend in MHA
    cross_attn_mask = cross_attn_mask == False

    return cross_attn_mask


def test():
    batch_size = 3
    data = mock_batch(batch_size=batch_size)

    dense_data, mask = to_dense_batch(data.x, data.batch)
    output_data, output_batch = from_dense_batch(dense_data, mask)
    print((data.x == output_data).all())
    print((data.batch == output_batch).all())
    # create block diagonal matrix of batch
    # block size: [nodes_in_batch] x [nodes_in_batch]
    block_diag, indices = make_block_diag(data)
    for i in range(batch_size):
        graph_adj = get_adj(block_diag, indices[i])
        print(graph_adj)
