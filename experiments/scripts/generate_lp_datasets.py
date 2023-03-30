"""Script for converting nx data to tg data."""
import pickle
from typing import List, Tuple

import networkx as nx
import typer
from tqdm.auto import tqdm

from rafen.experiments.paths import GRAPHS_PATH, LINK_PREDICTION_DATASETS_PATH
from rafen.tasks.linkprediction.dataset import LinkPredictionDataset

app = typer.Typer()


def generate_link_prediction_datasets(
    graphs: List[nx.Graph],
    split_proportion: Tuple[float, float, float],
) -> List[LinkPredictionDataset]:
    datasets = []
    for snapshot_id, snapshot in tqdm(
        enumerate(graphs), desc="Snapshot", total=len(graphs)
    ):
        datasets.append(
            LinkPredictionDataset().mk_link_prediction_dataset(
                graph=snapshot,
                split_proportion=split_proportion,
                prev_nodes=list(graphs[snapshot_id - 1].nodes())
                if snapshot_id > 0
                else None,
            )
        )
    return datasets


def main(
    dataset: str = typer.Option(..., help="Dataset"),
    split_proportion: Tuple[float, float, float] = typer.Option(
        ..., help="Split proportion"
    ),
) -> None:

    graph_path = GRAPHS_PATH / f"{dataset}.pkl"

    snapshots = nx.read_gpickle(graph_path)[dataset]["graphs"]

    datasets = generate_link_prediction_datasets(
        graphs=snapshots,
        split_proportion=split_proportion,
    )

    output_path = LINK_PREDICTION_DATASETS_PATH / f"{dataset}.pkl"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "wb") as f:
        pickle.dump(obj=datasets, file=f)


if __name__ == "__main__":
    typer.run(main)
