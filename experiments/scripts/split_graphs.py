"""Script for generating random temporal graphs."""
import os
import pickle
from collections import OrderedDict
from copy import deepcopy

import networkx as nx
import typer
import yaml

from rafen.data import snapshot_generator as dgg

app = typer.Typer()


def main(
    dataset: str = typer.Option(..., help="Dataset name"),
    config_path: str = typer.Option(
        "experiments/configs/split_graphs.yaml", help="Config path"
    ),
):
    with open(config_path, "r") as fin:
        cfg = yaml.safe_load(fin)
    graph_split_cfg = cfg["graphs"][dataset]

    snapshots = {}

    graph = nx.read_gpickle(
        cfg["paths"]["input"].replace("${dataset}", dataset)
    )
    timestamp_format_year = graph_split_cfg.pop("timestamp_format_year")
    dg = dgg.GraphSnapshotDateGenerator(
        graph, timestamp_format_year=timestamp_format_year
    )

    generated = {
        it["snapshot_id"]: {"graph": it["graph"]}
        for it in dg.generate(
            split_type=graph_split_cfg["split_type"],
            interval=graph_split_cfg["interval"],
        )
    }
    filtered_snapshots = deepcopy(generated)

    if "exclude" in graph_split_cfg.keys():
        for snap_id, snap in generated.items():
            if snap_id in graph_split_cfg["exclude"]:
                del filtered_snapshots[snap_id]

    if "merge" in graph_split_cfg.keys():
        for src, trg in graph_split_cfg["merge"]:
            src_graph = filtered_snapshots[src]["graph"]
            trg_graph = filtered_snapshots[trg]["graph"]
            src_graph.update(trg_graph)

            del filtered_snapshots[trg]

            filtered_snapshots[src] = {
                "snapshot_id": src,
                "graph": src_graph,
            }

    filtered_snapshots = OrderedDict(
        sorted(filtered_snapshots.items(), key=lambda x: x[0])
    )
    snapshots[dataset] = {
        "graphs": [it["graph"] for key, it in filtered_snapshots.items()],
    }

    out_path = cfg["paths"]["output"].replace("${dataset}", dataset)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(obj=snapshots, file=f)


if __name__ == "__main__":
    typer.run(main)
