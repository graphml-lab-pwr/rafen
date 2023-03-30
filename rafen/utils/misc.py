import functools
import importlib
from timeit import default_timer as timer
from typing import Any, Dict, Tuple

import networkx as nx
import numpy as np


def get_common_nodes(g_u, g_v):
    nodes = set(g_u.nodes()).intersection(g_v.nodes())
    nodes = sorted(nodes)
    nodes = list(map(str, nodes))
    return nodes


def get_unchanged_changed_nodes(src_graph, changed_graph):
    unchanged = []
    changed = []
    common_nodes = [
        int(it) for it in get_common_nodes(src_graph, changed_graph)
    ]
    for node in common_nodes:
        src_edges = set(src_graph.edges([node]))
        changed_edges = set(changed_graph.edges([node]))
        if src_edges == changed_edges:
            unchanged.append(node)
        else:
            changed.append(node)

    return set(unchanged), set(changed)


def check_arr_is_ndarray(arr, arr_name):
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{arr_name} is not np.ndarray")


def relabel(graph: nx.Graph) -> Tuple[nx.Graph, Dict[int, Any]]:
    old_node_labels = list(graph.nodes())
    new_node_labels = np.arange(0, len(old_node_labels))

    rev_mapping = dict(zip(new_node_labels, old_node_labels))
    graph = nx.relabel_nodes(graph, dict(zip(old_node_labels, new_node_labels)))
    return graph, rev_mapping


def get_module_from_str(module: str) -> Any:
    module, cls = module.rsplit(".", maxsplit=1)
    cls = getattr(importlib.import_module(module), cls)
    return cls


def timeit(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        start_time = timer()
        result = func(*args, **kwargs)
        elapsed_time = timer() - start_time

        return result, elapsed_time

    return new_func
