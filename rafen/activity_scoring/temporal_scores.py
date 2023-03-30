"""Implementation of time-aware node scoring methods.

Source article: https://www.nature.com/articles/s41598-020-69379-z

Implemented scores:
- temporal closeness (TC)
- temporal betweenness (TB)
- temporal k-shell (TK)
- temporal degree deviation (TDD)
"""
import os
import sys
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Type

import networkx as nx
import numpy as np
import pathpy as pp
from tqdm import tqdm

from rafen.data import snapshot_generator as dgg

pp.utils.Log.set_min_severity(pp.utils.Severity.ERROR)


def aggregate_timestamps(
    graph: nx.MultiGraph,
    timestamp_format_year: bool,
    interval: int,
    split_type: str,
) -> nx.MultiGraph:
    """Aggregates timestamps in given `graph` using `interval`."""
    generator = dgg.GraphSnapshotDateGenerator(
        graph=graph, timestamp_format_year=timestamp_format_year
    )

    # Group the timestamps in the graph into snapshots using given interval
    snapshots = list(
        generator.generate(
            split_type=split_type,
            interval=interval,
            # The link prediction dataset config doesn't matter here as we drop it,
            # but it must be set in order for the generator to work.
            lp_ds_cfg={"split_proportion": 0.5},
        )
    )

    g = nx.MultiGraph()
    g.add_nodes_from(graph.nodes)
    for s in snapshots:
        g.add_edges_from(s["graph"].edges(), timestamp=s["snapshot_id"])

    return g


def to_temporal_network(graph: nx.MultiGraph) -> pp.TemporalNetwork:
    """Converts a NetworkX graph into a PathPy TemporalNetwork."""
    tn = pp.TemporalNetwork()

    for src, dst, ts in graph.edges(data="timestamp"):
        if isinstance(ts, float):
            ts = int(ts)
        tn.add_edge(source=src, target=dst, ts=ts)

    return tn


def to_networkx(
    graph: pp.TemporalNetwork,
    create_using: Type[nx.Graph] = nx.Graph,
) -> nx.Graph:
    g = create_using()

    for src, dst, ts in graph.tedges:
        g.add_edge(src, dst, timestamp=ts)

    return g


def compute_temporal_shortest_paths(
    temporal_network: pp.TemporalNetwork,
    max_delta: int,
) -> Tuple[dict, dict]:
    """Computes shortest paths and shortest path distances."""
    with disable_print():
        p = pp.path_extraction.temporal_paths.paths_from_temporal_network_dag(
            tempnet=temporal_network,
            delta=max_delta,
        )

    paths = pp.algorithms.shortest_paths.shortest_paths(p)
    distances = pp.algorithms.shortest_paths.distance_matrix(p)

    return paths, distances


def closeness(
    graph: pp.TemporalNetwork,
    nodes: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict[int, float]:
    if nodes is None:
        nodes = graph.nodes

    tc = {node: 0 for node in nodes}

    pbar = tqdm(
        total=len(graph.ordered_times) - 1,
        disable=not verbose,
    )

    while len(graph.ordered_times) >= 2:
        t_min, t_max = graph.ordered_times[0], graph.ordered_times[-1]

        pbar.set_description(
            f"Closeness - Timestamp range [t, T] = [{t_min}, {t_max}]"
        )

        # Compute temporal shortest paths distances for interval [t_min, t_max]
        _, distances = compute_temporal_shortest_paths(
            temporal_network=graph,
            max_delta=1,
        )

        # Apply closeness for interval [t_min, t_max]
        for v in tqdm(
            iterable=nodes,
            desc="Closeness(v)",
            disable=not verbose,
            leave=False,
        ):
            value = 0
            for u, dist in distances[str(v)].items():
                if str(v) == u:
                    continue
                value += 1 / dist

            tc[v] += value

        # Remove the first timestamp `t_min`
        graph = graph.filter_edges(edge_filter=lambda v, w, ts: ts > t_min)

        pbar.update(n=1)

    pbar.close()

    return tc


def betweenness(
    graph: pp.TemporalNetwork,
    nodes: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict[int, float]:
    if nodes is None:
        nodes = graph.nodes

    tb = {node: 0 for node in nodes}

    pbar = tqdm(
        total=len(graph.ordered_times) - 1,
        disable=not verbose,
    )

    while len(graph.ordered_times) >= 2:
        t_min, t_max = graph.ordered_times[0], graph.ordered_times[-1]

        pbar.set_description(
            f"Betweenness - Timestamp range [t, T] = [{t_min}, {t_max}]"
        )

        # Compute temporal shortest paths distances for interval [t_min, t_max]
        paths, _ = compute_temporal_shortest_paths(
            temporal_network=graph,
            max_delta=1,
        )

        # Apply betweenness for interval [t_min, t_max]
        for v in tqdm(
            iterable=nodes,
            desc="Betweenness(v)",
            disable=not verbose,
            leave=False,
        ):
            value = 0

            for s in paths.keys():
                for d in paths[s].keys():
                    if s != v and s != d and v != d:
                        num_sp_sd = len(paths[str(s)][str(d)])

                        if num_sp_sd > 0:
                            num_sp_svd = len(
                                [
                                    path
                                    for path in paths[str(s)][str(d)]
                                    if str(v) in path
                                ]
                            )

                            value += num_sp_svd / num_sp_sd

            tb[v] += value

        # Remove the first timestamp `t_min`
        graph = graph.filter_edges(edge_filter=lambda v, w, ts: ts > t_min)

        pbar.update(n=1)

    pbar.close()

    return tb


def k_shell_score(
    graph: pp.TemporalNetwork,
    nodes: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict[int, float]:
    if nodes is None:
        nodes = graph.nodes

    kss = {node: 0 for node in nodes}

    for timestamp in tqdm(
        iterable=graph.ordered_times,
        desc="K-Shell score - Timestamps",
        disable=not verbose,
    ):
        g_t = to_networkx(
            graph=graph.filter_edges(
                edge_filter=lambda src, dst, ts: ts == timestamp
            ),
            create_using=nx.Graph,
        )

        # K-Shell is not defined for self-loops
        g_t.remove_edges_from(nx.selfloop_edges(g_t))

        ks_scores = _get_ks_scores(graph=g_t)
        for v in ks_scores.keys():
            if v not in nodes:
                continue
            for u in g_t.neighbors(v):
                kss[v] += min(ks_scores[v], ks_scores[u])

    return kss


def _get_ks_scores(graph: nx.Graph) -> Dict[int, int]:
    ks_scores = {}
    max_degree = max(dict(graph.degree()).values())
    for k in range(1, max_degree + 1):
        k_shell_nodes = nx.algorithms.k_shell(G=graph, k=k).nodes

        for node in k_shell_nodes:
            assert node not in ks_scores.keys()

            ks_scores[node] = k

    return ks_scores


def degree_deviation(
    graph: pp.TemporalNetwork,
    nodes: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict[int, float]:
    if nodes is None:
        nodes = graph.nodes

    # Compute degree in each slice network
    degrees = {node: [] for node in nodes}

    for timestamp in tqdm(
        iterable=graph.ordered_times,
        desc="Degree deviation - Timestamps",
        disable=not verbose,
    ):
        g_t = to_networkx(
            graph=graph.filter_edges(
                edge_filter=lambda src, dst, ts: ts == timestamp
            ),
            create_using=nx.Graph,
        )

        g_t_deg = dict(g_t.degree)

        for node in nodes:
            degrees[node].append(g_t_deg.get(node, 0))

    # Compute degree deviation
    dd = {}
    L = len(graph.ordered_times)

    for node in nodes:
        node_degrees = np.array(degrees[node])

        mean_degree = np.mean(node_degrees)
        dd[node] = np.sqrt(
            (1 / L) * np.power(node_degrees - mean_degree, 2).sum()
        )

    return dd


@contextmanager
def disable_print():
    stdout = sys.stdout
    devnull = open(os.devnull, "w")

    try:
        sys.stdout = devnull
        yield
    finally:
        sys.stdout = stdout
