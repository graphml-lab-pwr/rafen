import pickle
from typing import Final

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rafen.experiments.defaults import DATASETS
from rafen.experiments.paths import (
    EMBEDDINGS_PATH,
    GRAPHS_PATH,
    KNOWLEDGE_TRANSFER_STUDY_PATH,
    PREV_EMBEDDINGS_PATH,
    PREV_POSTHOC_PATH,
)
from rafen.tasks.linkprediction import evaluation
from rafen.tasks.linkprediction.dataset import LinkPredictionDataset

METHODS: Final = [
    "Node2VecAligned_L2_ALL_unweighted",
    "Node2VecWeightedAligned_L2_TB",
    "Node2VecWeightedAligned_L2_EJ",
    "GAEAlignedEmbedding_L2_ALL_unweighted",
    "GAEWeightedAlignedEmbedding_L2_TB",
    "GAEWeightedAlignedEmbedding_L2_EJ",
]


def main() -> None:

    pbar = tqdm(DATASETS)
    data = []

    for ds in pbar:
        pbar.set_description(f"Dataset {ds}")

        graphs = pd.read_pickle(GRAPHS_PATH / f"{ds}.pkl")[ds]["graphs"]
        snapshots = len(graphs)

        embeddings = {
            it: pd.read_pickle(PREV_EMBEDDINGS_PATH / it / f"{ds}.pkl")
            for it in METHODS
        }
        embeddings.update(
            {
                method: pd.read_pickle(EMBEDDINGS_PATH / method / f"{ds}.pkl")
                for method in ("Node2Vec", "GAE")
            }
        )
        embeddings.update(
            {
                it.name: pd.read_pickle(it / f"{ds}.pkl")
                for it in PREV_POSTHOC_PATH.iterdir()
                if it.is_dir()
            }
        )

        for snapshot_id in tqdm(range(1, snapshots - 1), desc="Snapshot"):

            lp_ds = []
            for snap_id_diff in np.arange(-1, 2):
                lp_ds.append(
                    LinkPredictionDataset()
                    .mk_link_prediction_dataset(
                        graph=graphs[snapshot_id + snap_id_diff],
                        prev_nodes=list(graphs[snapshot_id].nodes()),
                        split_proportion=(0.6, 0.2, 0.2),
                    )
                    .merge_train_subset_with_dev()
                )

            for embedding_name, ds_embeddings in tqdm(
                embeddings.items(), desc="Embeddings", leave=False
            ):
                for run_id in tqdm(
                    range(len(ds_embeddings)), desc="Embedding Run", leave=False
                ):
                    for eval_snapshot_id, eval_snapshot_name in tqdm(
                        enumerate(["prev", "current", "next"]),
                        desc="Evaluation Snapshot",
                        leave=False,
                        total=3,
                    ):
                        score = evaluation.eval_model(
                            embeddings=ds_embeddings[run_id][snapshot_id],
                            lp_ds=lp_ds[eval_snapshot_id],
                            model="LR",
                            test_subset="test",
                        )
                        data.append(
                            {
                                "dataset": ds,
                                "snapshot_id": snapshot_id,
                                "evaluation_snapshot": eval_snapshot_name,
                                "run": run_id,
                                "auc": score["auc"],
                                "method": embedding_name,
                            }
                        )

    KNOWLEDGE_TRANSFER_STUDY_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(KNOWLEDGE_TRANSFER_STUDY_PATH, "wb") as f:
        pickle.dump(obj=data, file=f)


main()
