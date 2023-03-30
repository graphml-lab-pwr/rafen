import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import typer
from tqdm.auto import tqdm

from rafen.embeddings.aggregation import EmbeddingAverageAggregation
from rafen.embeddings.fildne import FILDNE, kFILDNE
from rafen.experiments.defaults import NODE2VEC
from rafen.experiments.paths import LINK_PREDICTION_DATASETS_PATH
from rafen.tasks.linkprediction import evaluation
from rafen.utils.io import read_yaml

app = typer.Typer()


def main(
    config: str = typer.Option(..., help="Config"),
    model: str = typer.Option(..., help="Model"),
    fildne_config: str = typer.Option(
        "experiments/configs/embeddings_aggregation/FILDNE.yaml",
        help="FILDNE path",
    ),
):
    n_jobs = int(os.getenv("NUM_WORKERS", 1))

    # Read configs
    cfg = read_yaml(config)
    fildne_cfg = read_yaml(fildne_config)

    embedding_path = Path(cfg["paths"]["embeddings"]).joinpath(model)
    output_path = Path(cfg["paths"]["output"]).joinpath(model)

    paths = {
        "graphs": Path(cfg["paths"]["graph"]),
        "output": output_path,
        "embeddings": embedding_path,
    }
    paths["output"].mkdir(parents=True, exist_ok=True)

    dataset_pbar = tqdm(cfg["datasets"])
    for dataset in dataset_pbar:
        dataset_pbar.set_description(f"Dataset ({dataset})")
        graph_data = pd.read_pickle(paths["graphs"].joinpath(f"{dataset}.pkl"))
        lp_ds = pd.read_pickle(LINK_PREDICTION_DATASETS_PATH / f"{dataset}.pkl")

        graphs = graph_data[dataset]["graphs"]
        embeddings = pd.read_pickle(
            paths["embeddings"].joinpath(f"{dataset}.pkl")
        )

        ds_results = []

        for run_id, run_embeddings in tqdm(
            enumerate(embeddings),
            desc="Run",
            leave=False,
            total=len(embeddings),
        ):
            for snapshot_id in tqdm(
                range(len(lp_ds) - 1),
                desc="Snapshot",
                leave=False,
                total=len(lp_ds) - 1,
            ):
                current_snapshot_prediction = evaluation.eval_model(
                    embeddings=embeddings[run_id][snapshot_id],
                    lp_ds=lp_ds[snapshot_id].merge_train_subset_with_dev(),
                    model="LR",
                    n_jobs=n_jobs,
                    test_subset="test",
                )
                ds_results.append(
                    {
                        "run": run_id,
                        "snapshot": snapshot_id,
                        "embeddings_aggregation": "last",
                        "prediction_snapshot": snapshot_id,
                        "auc": current_snapshot_prediction["auc"],
                        "same_snapshot_prediction": True,
                        "f1-micro": current_snapshot_prediction["f1-micro"],
                        "report": current_snapshot_prediction["report"],
                    }
                )

                next_snapshot_prediction = evaluation.eval_model(
                    embeddings=embeddings[run_id][snapshot_id],
                    lp_ds=lp_ds[snapshot_id + 1].merge_train_subset_with_dev(),
                    model="LR",
                    n_jobs=n_jobs,
                    test_subset="test",
                )
                ds_results.append(
                    {
                        "run": run_id,
                        "snapshot": snapshot_id,
                        "embeddings_aggregation": "last",
                        "prediction_snapshot": snapshot_id + 1,
                        "auc": next_snapshot_prediction["auc"],
                        "same_snapshot_prediction": False,
                        "f1-micro": next_snapshot_prediction["f1-micro"],
                        "report": next_snapshot_prediction["report"],
                    }
                )

                if model != NODE2VEC:
                    embeddings_aggregation_models: Dict[str, Any] = {
                        "mean": EmbeddingAverageAggregation(),
                        "FILDNE": FILDNE(**fildne_cfg["FILDNE"][dataset]),
                        "k-FILDNE": kFILDNE(
                            **fildne_cfg["k-FILDNE"][dataset]
                        ).fit(
                            embeddings=embeddings[run_id][: (snapshot_id + 1)],
                            last_snapshot=graphs[snapshot_id],
                        ),
                    }

                    for (
                        aggregation_model_name,
                        aggregation_model,
                    ) in embeddings_aggregation_models.items():
                        merged_snapshot_prediction = evaluation.eval_model(
                            embeddings=embeddings[run_id][: (snapshot_id + 1)],
                            lp_ds=lp_ds[
                                snapshot_id + 1
                            ].merge_train_subset_with_dev(),
                            model="LR",
                            n_jobs=n_jobs,
                            embedding_agg_model=aggregation_model,
                            test_subset="test",
                        )
                        ds_results.append(
                            {
                                "run": run_id,
                                "snapshot": snapshot_id,
                                "embeddings_aggregation": aggregation_model_name,
                                "prediction_snapshot": snapshot_id + 1,
                                "auc": merged_snapshot_prediction["auc"],
                                "f1-micro": merged_snapshot_prediction[
                                    "f1-micro"
                                ],
                                "report": merged_snapshot_prediction["report"],
                                "same_snapshot_prediction": False,
                            }
                        )

        results_df = pd.DataFrame(ds_results).astype({"snapshot": str})
        results_df["snapshot"] = results_df["snapshot"].apply(
            lambda x: f"Snapshot {x}"
        )
        results_df.sort_values(by=["snapshot", "run"]).reset_index(
            drop=True
        ).to_html(str(paths["output"].joinpath(f"{dataset}.html")))
        results_df.to_pickle(str(paths["output"].joinpath(f"{dataset}.pkl")))


if __name__ == "__main__":
    typer.run(main)
