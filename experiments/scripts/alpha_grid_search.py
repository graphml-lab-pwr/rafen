import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

from rafen.embeddings.aggregation import EmbeddingAverageAggregation
from rafen.embeddings.fildne import FILDNE, kFILDNE
from rafen.embeddings.keyedmodel import KeyedModel
from rafen.experiments.paths import (
    ALPHAS_GRID_SEARCH_GAE_RESULTS,
    ALPHAS_GRID_SEARCH_RESULTS,
    GRAPHS_PATH,
    LINK_PREDICTION_DATASETS_PATH,
    RANDOM_WALKS_PATH,
    TG_GRAPHS_PATH,
    get_cached_random_walks_path,
)
from rafen.experiments.runners import AlignmentExperimentRunner
from rafen.models.rafen_gae import RAFENGAE
from rafen.tasks.linkprediction import evaluation
from rafen.utils.io import read_yaml
from rafen.utils.misc import get_module_from_str

app = typer.Typer()


def evaluate_embeddings_lp(
    embeddings: List[List[KeyedModel]],
    lp_ds: List[Any],
    n_jobs: int,
    fildne_cfg: Dict[str, Any],
    graphs: List[Any],
    dataset: str,
) -> List[Dict[str, Any]]:
    lp_results = []
    for run_id, run_embeddings in tqdm(
        enumerate(embeddings),
        desc="LP Evaluation (Run)",
        leave=False,
        total=len(embeddings),
    ):
        for snapshot_id in tqdm(
            range(len(lp_ds) - 1),
            desc="LP Evaluation (Snapshot)",
            leave=False,
            total=len(lp_ds) - 1,
        ):
            next_snapshot_prediction = evaluation.eval_model(
                embeddings=run_embeddings[snapshot_id],
                lp_ds=lp_ds[snapshot_id + 1],
                model="LR",
                n_jobs=n_jobs,
                test_subset="dev",
            )
            lp_results.append(
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

            embeddings_aggregation_models: Dict[str, Any] = {
                "mean": EmbeddingAverageAggregation(),
                "FILDNE": FILDNE(**fildne_cfg["FILDNE"][dataset]),
                "k-FILDNE": kFILDNE(**fildne_cfg["k-FILDNE"][dataset]).fit(
                    embeddings=run_embeddings[: (snapshot_id + 1)],
                    last_snapshot=graphs[snapshot_id],
                ),
            }

            for (
                aggregation_model_name,
                aggregation_model,
            ) in embeddings_aggregation_models.items():
                merged_snapshot_prediction = evaluation.eval_model(
                    embeddings=run_embeddings[: (snapshot_id + 1)],
                    lp_ds=lp_ds[snapshot_id + 1],
                    model="LR",
                    n_jobs=n_jobs,
                    embedding_agg_model=aggregation_model,
                    test_subset="dev",
                )
                lp_results.append(
                    {
                        "run": run_id,
                        "snapshot": snapshot_id,
                        "embeddings_aggregation": aggregation_model_name,
                        "prediction_snapshot": snapshot_id + 1,
                        "auc": merged_snapshot_prediction["auc"],
                        "f1-micro": merged_snapshot_prediction["f1-micro"],
                        "report": merged_snapshot_prediction["report"],
                        "same_snapshot_prediction": False,
                    }
                )

    return lp_results


def main(
    dataset: str = typer.Option(..., help="Dataset name"),
    runs: int = typer.Option(..., help="Number of runs to perform"),
    fildne_config: str = typer.Option(
        "experiments/configs/embeddings_aggregation/FILDNE.yaml",
        help="FILDNE path",
    ),
    model_config: str = typer.Option(
        "experiments/configs/models/Node2Vec.yaml", help="Model config"
    ),
    config: str = typer.Option(
        "experiments/configs/alpha_grid_search.yaml", help="Config file"
    ),
    precomputed_rw: bool = typer.Option(
        False, help="Whether to use precomputed_rw"
    ),
):
    n_jobs = int(os.getenv("NUM_WORKERS", 1))

    precomputed_rw_path = None
    if precomputed_rw:
        cache_path = os.environ.get("RW_CACHE_PATH", "")
        if cache_path:
            precomputed_rw_path = (
                get_cached_random_walks_path(cache_path) / dataset
            )
        else:
            precomputed_rw_path = RANDOM_WALKS_PATH / dataset
        assert precomputed_rw_path.exists()

    # Read N2V config
    cfg = read_yaml(model_config)
    grid_search_cfg = read_yaml(config)

    fildne_cfg = read_yaml(Path(fildne_config))

    alpha_step = grid_search_cfg.pop("alpha_step")
    alphas = np.arange(alpha_step, 1.0, alpha_step)

    model_cls = get_module_from_str(grid_search_cfg["cls"])
    pretrained_path = Path(grid_search_cfg.pop("pretrained")) / f"{dataset}.pkl"
    pretrained = pd.read_pickle(pretrained_path)

    graph_data = pd.read_pickle(GRAPHS_PATH / f"{dataset}.pkl")[dataset]
    graphs = graph_data["graphs"]
    lp_ds = pd.read_pickle(LINK_PREDICTION_DATASETS_PATH / f"{dataset}.pkl")
    tg_graphs = pd.read_pickle(TG_GRAPHS_PATH / f"{dataset}.pkl")[dataset]

    if issubclass(model_cls, RAFENGAE):
        output_path = ALPHAS_GRID_SEARCH_GAE_RESULTS / dataset
    else:
        output_path = ALPHAS_GRID_SEARCH_RESULTS / dataset
    output_path.mkdir(exist_ok=True, parents=True)

    alphas_pbar = tqdm(alphas)
    for alpha in alphas_pbar:
        alphas_pbar.set_description(f"Alpha: {alpha}")
        alignment_cfg = deepcopy(grid_search_cfg)
        alignment_cfg["alpha"] = {dataset: alpha}

        alpha_ds_embeddings, alpha_ds_metadata = AlignmentExperimentRunner(
            tg_graphs=tg_graphs["graphs"],
            graphs=graphs,
            dataset=dataset,
            model_name="",
            model_cls=model_cls,
            model_args=deepcopy(cfg["args"][dataset]),
            nodes_mapping=tg_graphs["node_mappings"],
            logger=None,
            runs=runs,
            precomputed_rw=precomputed_rw_path,
            pretrained=pretrained,
            alignment_cfg=alignment_cfg,
            loss_scaling=True,
            prev_snapshot_alignment=True,
        ).run()

        lp_results = evaluate_embeddings_lp(
            embeddings=alpha_ds_embeddings,
            lp_ds=lp_ds,
            n_jobs=n_jobs,
            fildne_cfg=fildne_cfg,
            graphs=graphs,
            dataset=dataset,
        )

        with open(output_path / f"alpha_{str(alpha)}.pkl", "wb") as f:
            pickle.dump(
                obj={
                    "alpha": alpha,
                    "lp_results": lp_results,
                    "metadata": alpha_ds_metadata,
                },
                file=f,
            )


if __name__ == "__main__":
    typer.run(main)
