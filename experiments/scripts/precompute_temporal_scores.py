import pickle

import pandas as pd
import typer
from tqdm import tqdm

from rafen.activity_scoring import temporal_scores as dts
from rafen.experiments.paths import (
    GRAPHS_PATH,
    TEMPORAL_SCORES_PATH,
    TEMPORAL_SCORES_PREV_PATH,
)
from rafen.utils import preprocess as dpp
from rafen.utils.io import read_yaml
from rafen.utils.misc import get_common_nodes


def main(
    dataset_name: str = typer.Option(..., help="Dataset name"),
    prev_snapshot_scoring: bool = False,
):
    cfg = read_yaml("experiments/configs/precompute_temporal_scores.yaml")

    graphs_path = GRAPHS_PATH / f"{dataset_name}.pkl"
    cached_path = (
        TEMPORAL_SCORES_PATH
        if not prev_snapshot_scoring
        else TEMPORAL_SCORES_PREV_PATH
    )

    # Get timestamp format
    timestamp_format_year = False
    if dataset_name in cfg["timestamp_format_year"]:
        timestamp_format_year = True

    # Read graphs
    dataset = pd.read_pickle(graphs_path)[dataset_name]

    metric_fns = [
        ("closeness", dts.closeness),
        ("betweenness", dts.betweenness),
        ("k_shell_score", dts.k_shell_score),
        ("degree_deviation", dts.degree_deviation),
    ]

    pbar = tqdm(iterable=metric_fns, desc="Metric")
    for metric_name, metric_fn in pbar:
        pbar.set_description(f"Metric: {metric_name}")

        values = []
        graphs = dataset["graphs"]
        preprocessed_graphs = [
            dpp.preprocess_graph(
                dataset_name=dataset_name,
                graph=graph,
                timestamp_format_year=timestamp_format_year,
            )
            for graph in graphs
        ]

        for idx, graph in tqdm(
            iterable=enumerate(graphs[1:], start=1),
            desc="Graph",
            total=len(graphs) - 1,
            leave=False,
        ):
            prev_graph_id = 0 if not prev_snapshot_scoring else idx - 1
            common_nodes = get_common_nodes(
                g_u=graphs[prev_graph_id],
                g_v=graph,
            )

            g_u = preprocessed_graphs[prev_graph_id]
            g_v = preprocessed_graphs[idx]

            s_u = metric_fn(graph=g_u, nodes=common_nodes, verbose=False)
            s_v = metric_fn(graph=g_v, nodes=common_nodes, verbose=False)

            values.append((s_u, s_v))

        metrics_ds_path = cached_path / f"{metric_name}/{dataset_name}.pkl"

        metrics_ds_path.parent.mkdir(exist_ok=True, parents=True)
        with open(metrics_ds_path, "wb") as f:
            pickle.dump(obj=values, file=f)

    pbar.close()


if __name__ == "__main__":
    typer.run(main)
