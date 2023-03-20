from __future__ import annotations
import itertools
from itertools import combinations, product
from typing import Dict, Iterable, List, Optional, Tuple, Union
import networkx as nx
import numpy as np
import pandas as pd
from graphein.protein import add_edge, compute_distmat, filter_distmat
from loguru import logger as log
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors, kneighbors_graph


from graphein.protein.resi_atoms import (
    AA_RING_ATOMS,
    AROMATIC_RESIS,
    BACKBONE_ATOMS,
    BOND_TYPES,
    CATION_PI_RESIS,
    CATION_RESIS,
    DISULFIDE_ATOMS,
    DISULFIDE_RESIS,
    HYDROPHOBIC_RESIS,
    IONIC_RESIS,
    NEG_AA,
    PI_RESIS,
    POS_AA,
    RING_NORMAL_ATOMS,
    SALT_BRIDGE_ANIONS,
    SALT_BRIDGE_ATOMS,
    SALT_BRIDGE_CATIONS,
    SALT_BRIDGE_RESIDUES,
    SULPHUR_RESIS,
    VDW_RADII,
)
from graphein.protein.utils import filter_dataframe

INFINITE_DIST = 10_000.0  # np.inf leads to errors in some cases


def add_k_nn_edges(
        G: nx.Graph,
        long_interaction_threshold: int = 0,
        k: int = 5,
        exclude_edges: Iterable[str] = (),
        exclude_self_loops: bool = True,
        kind_name: str = "knn",
):
    """
    Adds edges to nodes based on K nearest neighbours. Long interaction
    threshold is used to specify minimum separation in sequence to add an edge
    between networkx nodes within the distance threshold

    :param G: Protein Structure graph to add distance edges to
    :type G: nx.Graph
    :param long_interaction_threshold: minimum distance in sequence for two
        nodes to be connected
    :type long_interaction_threshold: int
    :param k: Number of neighbors for each sample.
    :type k: int
    :param exclude_edges: Types of edges to exclude. Supported values are
        `inter` and `intra`.
        - `inter` removes inter-connections between nodes of the same chain.
        - `intra` removes intra-connections between nodes of different chains.
    :type exclude_edges: Iterable[str].
    :param exclude_self_loops: Whether or not to mark each sample as the first
        nearest neighbor to itself.
    :type exclude_self_loops: Union[bool, str]
    :param kind_name: Name for kind of edges in networkx graph.
    :type kind_name: str
    :return: Graph with knn-based edges added
    :rtype: nx.Graph
    """
    # Prepare dataframe
    pdb_df = filter_dataframe(
        G.graph["pdb_df"], "node_id", list(G.nodes()), True
    )
    if (
            pdb_df["x_coord"].isna().sum()
            or pdb_df["y_coord"].isna().sum()
            or pdb_df["z_coord"].isna().sum()
    ):
        raise ValueError("Coordinates contain a NaN value.")

    # Construct distance matrix
    dist_mat = compute_distmat(pdb_df)

    # Filter edges
    dist_mat = filter_distmat(pdb_df, dist_mat, exclude_edges)

    # Add self-loops if specified
    if not exclude_self_loops:
        k -= 1
        for n1, n2 in zip(G.nodes(), G.nodes()):
            add_edge(G, n1, n2, kind_name)

    # Reduce k if number of nodes is less (to avoid sklearn error)
    # Note: - 1 because self-loops are not included
    if G.number_of_nodes() - 1 < k:
        k = G.number_of_nodes() - 1

    if k == 0:
        return

    # Run k-NN search
    neigh = NearestNeighbors(n_neighbors=k, metric="precomputed")
    neigh.fit(dist_mat)
    nn = neigh.kneighbors_graph()

    # Create iterable of node indices
    outgoing = np.repeat(np.array(range(len(G.graph["pdb_df"]))), k)
    incoming = nn.indices
    interacting_nodes = list(zip(outgoing, incoming))
    log.info(f"Found: {len(interacting_nodes)} KNN edges")
    for a1, a2 in interacting_nodes:
        if dist_mat.loc[a1, a2] == INFINITE_DIST:
            continue

        # Get nodes IDs from indices
        n1 = G.graph["pdb_df"].iloc[a1]["node_id"]
        n2 = G.graph["pdb_df"].iloc[a2]["node_id"]

        # Get chains
        n1_chain = G.graph["pdb_df"].iloc[a1]["chain_id"]
        n2_chain = G.graph["pdb_df"].iloc[a2]["chain_id"]

        # Get sequence position
        n1_position = G.graph["pdb_df"].iloc[a1]["residue_number"]
        n2_position = G.graph["pdb_df"].iloc[a2]["residue_number"]

        # Check residues are not on same chain
        condition_1 = n1_chain != n2_chain
        # Check residues are separated by long_interaction_threshold
        condition_2 = (
                abs(n1_position - n2_position) > long_interaction_threshold
        )

        # If not on same chain add edge or
        # If on same chain and separation is sufficient add edge
        if condition_1 or condition_2:
            add_edge(G, n1, n2, kind_name)
