import pickle

import networkx as nx
import pandas as pd
from tqdm import tqdm

from rafen.experiments.defaults import DATASETS
from rafen.experiments.paths import FULL_GRAPHS_PATH, GRAPHS_PATH


def main():
    for ds in tqdm(DATASETS, desc="Dataset"):
        incremental_graphs = []
        graph_data = pd.read_pickle(GRAPHS_PATH / f"{ds}.pkl")[ds]
        graphs = graph_data["graphs"]
        for snapshot_id in range(len(graphs)):
            incremental_graph = nx.MultiGraph()
            if graphs[0].is_directed():
                incremental_graph = nx.MultiDiGraph()

            snapshots = graphs[0 : snapshot_id + 1]

            for snapshot in snapshots:
                incremental_graph.add_edges_from(snapshot.edges(data=True))

            incremental_graphs.append(incremental_graph)

        FULL_GRAPHS_PATH.mkdir(parents=True, exist_ok=True)
        with open(FULL_GRAPHS_PATH / f"{ds}.pkl", "wb") as f:
            pickle.dump(obj={ds: {"graphs": incremental_graphs}}, file=f)


main()
