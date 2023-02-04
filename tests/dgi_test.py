import torch
from models.pretraining.graph_infomax import DeepGraphInfomaxV2 as DGI, readout_function
from models.pretraining.encoders import RevGATConvEncoder
import torch_geometric.utils.convert as tgc
import networkx as nx


# function which returns a fixed pytorch geometric graph
def get_input_graph():
    g = nx.Graph()
    g.add_node(0, x=[1., 0., 1., 1., 0.8, 0.1])
    g.add_node(1, x=[0., 1., 0, 1.0, 0.9, 0.8])
    g.add_node(2, x=[0.4, 1., 0.3, 0.2, 1.0, 0.3])
    g.add_node(3, x=[1., 1.2, 0.8, 0.8, 0.7, 0.1])
    g.add_node(4, x=[1., 1.3, 1., 0.3, 1., 0.45])
    g.add_edge(1, 0, edge_weight=0.2)
    g.add_edge(1, 2, edge_weight=1.)
    g.add_edge(2, 0, edge_weight=3.)
    g.add_edge(3, 2, edge_weight=0.3)
    g.add_edge(4, 2, edge_weight=1.)

    pyg = tgc.from_networkx(g)
    return pyg


def corruption_proxy(x: torch.Tensor, edge_index: torch.Tensor, *args, **kwargs):
    g = nx.Graph()
    g.add_node(0, x=[1., 0., 1.1, 1.1, 0.2, 0.1])
    g.add_node(1, x=[0., 1., 0, 1.0, 1.1, 0.2])
    g.add_node(2, x=[0.4, 1., 0.1, 0.2, 0.1, 0.3])
    g.add_node(3, x=[1., 0.2, 0.9, 0.9, 0.8, 0.4])
    g.add_node(4, x=[1., 0.3, 1., 0.3, 1., 0.45])
    g.add_edge(1, 3, edge_weight=0.8)
    g.add_edge(1, 2, edge_weight=1.)
    g.add_edge(2, 0, edge_weight=3.)
    g.add_edge(1, 4, edge_weight=0.3)
    g.add_edge(4, 3, edge_weight=1.5)

    pyg = tgc.from_networkx(g)
    return pyg.x, pyg.edge_index, pyg.edge_weight


rev_gat_enc = RevGATConvEncoder(
    in_channels=6,
    hidden_channels=4,
    out_channels=3,
    num_convs=3,
    dropout=0.0,
    version="v2",
    edge_dim=1,
    heads=8,
    num_groups=2,
    concat=False,
    normalize_hidden=True
)


def main():
    # graph
    g = get_input_graph()

    # dgi based encoder
    dgi = DGI(
        hidden_channels=2,
        encoder=rev_gat_enc,
        readout=readout_function,
        corruption=corruption_proxy,
        normalize_hidden=False,
        dropout=0.1
    )

    # ... the following line raises errors!
    print(dgi(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_weight))




if __name__ == "__main__":
    main()
