from typing import List

import networkx as nx
import pathpy as pp

from rafen.activity_scoring import temporal_scores as dts

DATASET_AGGREGATION = {
    "bitcoin-alpha": {"interval": 1, "split_type": "month"},
    "bitcoin-otc": {"interval": 1, "split_type": "month"},
    "fb-forum": {"interval": 1, "split_type": "day"},
    "fb-messages": {"interval": 1, "split_type": "day"},
    "ia-radoslaw-email": {"interval": 1, "split_type": "day"},
    "ia-hypertext": {"interval": 1, "split_type": "hour"},
    "ia-enron-employees": {"interval": 1, "split_type": "month"},
}


def preprocess_graph(
    dataset_name: str, graph: nx.MultiGraph, timestamp_format_year: bool = False
) -> pp.TemporalNetwork:
    if dataset_name not in DATASET_AGGREGATION.keys():
        g = graph  # No preprocessing needed
    else:
        aggregation_kwargs = DATASET_AGGREGATION[dataset_name]
        g = dts.aggregate_timestamps(
            graph=graph,
            timestamp_format_year=timestamp_format_year,
            **aggregation_kwargs,
        )

    return dts.to_temporal_network(graph=g)


def combine_graphs(
    dataset_name: str,
    graphs: List[nx.MultiGraph],
) -> pp.TemporalNetwork:
    combined_graph = pp.TemporalNetwork()

    for graph in graphs:
        graph = preprocess_graph(dataset_name=dataset_name, graph=graph)

        for src, dst, ts in graph.tedges:
            combined_graph.add_edge(source=src, target=dst, ts=ts)

    return combined_graph
