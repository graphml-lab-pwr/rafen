import networkx as nx
import torch
import torch_geometric

from rafen.embeddings.keyedmodel import KeyedModel
from rafen.utils.misc import relabel


def get_tg_data_from_nx_graph(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.

    Borrowed from torch_geometric.nb_utils.convert
    """

    G, rev_mapping = relabel(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data["edge_index"] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data, rev_mapping


def multigraph2graph(multi_graph_nx):
    """Borrowed from TNE repository."""
    if type(multi_graph_nx) == nx.Graph or type(multi_graph_nx) == nx.DiGraph:
        return multi_graph_nx
    graph_nx = nx.DiGraph() if multi_graph_nx.is_directed() else nx.Graph()

    if len(multi_graph_nx.nodes()) == 0:
        return graph_nx

    # add edges + attributes
    for u, v, data in multi_graph_nx.edges(data=True):
        data["weight"] = 1.0

        if graph_nx.has_edge(u, v):
            graph_nx[u][v]["weight"] += data["weight"]
        else:
            graph_nx.add_edge(u, v, **data)

    # add node attributes
    for node, attr in multi_graph_nx.nodes(data=True):
        if node not in graph_nx:
            continue
        graph_nx.nodes[node].update(attr)

    return graph_nx


def preprocess_graph_n2v_tg(multi_graph_nx):
    graph = multigraph2graph(multi_graph_nx)
    data, node_index_mapping = get_tg_data_from_nx_graph(graph)
    return data, node_index_mapping


def convert_tg_model_to_km(model, node_index_mapping):
    embeddings = model.embedding.weight.data.cpu().numpy()

    emb_dict = {}
    for node_id, node_emb in enumerate(embeddings):
        if node_index_mapping:
            node_id = node_index_mapping[node_id]

        emb_dict[str(node_id)] = node_emb

    return KeyedModel(
        size=embeddings.shape[1],
        node_emb_vectors=emb_dict,
        fill_unknown_nodes=False,
    )
