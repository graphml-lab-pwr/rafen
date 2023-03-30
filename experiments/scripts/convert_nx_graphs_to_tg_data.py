"""Script for converting nx data to tg data."""
import pickle
from typing import List

import networkx as nx
import typer
from tqdm import tqdm

from rafen.experiments.paths import (
    FULL_GRAPHS_PATH,
    FULL_TG_GRAPHS_PATH,
    GRAPHS_PATH,
    TG_GRAPHS_PATH,
)
from rafen.utils.tg import preprocess_graph_n2v_tg


def preprocess(snapshots: List[nx.Graph], dataset: str):
    tg_snapshots = []
    node_mappings = []

    for graph in tqdm(snapshots, desc="Snapshot"):
        tg_snapshot, node_mapping = preprocess_graph_n2v_tg(graph)

        tg_snapshots.append(tg_snapshot)
        node_mappings.append(node_mapping)

    return {dataset: {"graphs": tg_snapshots, "node_mappings": node_mappings}}


app = typer.Typer()


def main(
    dataset: str = typer.Option(..., help="Dataset"),
    incremental: bool = typer.Option(False, help="Incremental"),
):
    graph_path = GRAPHS_PATH / f"{dataset}.pkl"
    output_path = TG_GRAPHS_PATH / f"{dataset}.pkl"
    if incremental:
        graph_path = FULL_GRAPHS_PATH / f"{dataset}.pkl"
        output_path = FULL_TG_GRAPHS_PATH / f"{dataset}.pkl"

    snapshots = nx.read_gpickle(graph_path)[dataset]["graphs"]
    tg_snapshots = preprocess(snapshots, dataset)

    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "wb") as f:
        pickle.dump(obj=tg_snapshots, file=f)


if __name__ == "__main__":
    typer.run(main)
