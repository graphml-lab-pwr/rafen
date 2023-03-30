import os
import pickle
from pathlib import Path

import pandas as pd
import typer
from mlflow.tracking import MlflowClient

from rafen.experiments.paths import (
    ALIGNERS_CFG_PATH,
    EMBEDDINGS_PATH,
    GRAPHS_PATH,
    N2V_CFG_PATH,
    PREV_EMBEDDINGS_PATH,
    RANDOM_WALKS_PATH,
    TG_GRAPHS_PATH,
    get_cached_random_walks_path,
)
from rafen.experiments.runners import AlignmentExperimentRunner
from rafen.utils.io import read_yaml
from rafen.utils.misc import get_module_from_str
from rafen.utils.mlflow_logger import MlFlowLogger

app = typer.Typer()


def main(
    model: str = typer.Option(..., help="Model name"),
    dataset: str = typer.Option(..., help="Dataset name"),
    runs: int = typer.Option(..., help="Number of runs to perform"),
    precomputed_rw: bool = typer.Option(
        False, help="Whether to use precomputed random walks"
    ),
    loss_scaling: bool = typer.Option(
        False, help="whether to use loss scaling"
    ),
    prev_snapshot_alignment: bool = typer.Option(
        False, help="Whether to align to previous snapshot instead of zero"
    ),
    cfg: str = typer.Option(
        "", help="Config to pass. If arg is not passed reads default n2v config"
    ),
    ignore_alpha_scaling: bool = typer.Option(
        False, help="Whether to ignore alpha scaling."
    ),
):
    if not os.getenv("MLFLOW_TRACKING_URI"):
        raise RuntimeError("ENV variable MLFLOW_TRACKING_URI is unset!")

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

    if not cfg:
        # Read N2V config if cfg not passed
        cfg = read_yaml(N2V_CFG_PATH)
    else:
        cfg = read_yaml(Path(cfg))

    if ignore_alpha_scaling:
        cfg["args"][dataset]["ignore_alpha_scaling"] = True

    alignment_cfg = read_yaml(ALIGNERS_CFG_PATH / f"{model}.yaml")
    model_cls = get_module_from_str(alignment_cfg["cls"])
    pretrained_path = Path(alignment_cfg.pop("pretrained")) / f"{dataset}.pkl"
    pretrained = pd.read_pickle(pretrained_path)

    output_path = (
        EMBEDDINGS_PATH / model
        if not prev_snapshot_alignment
        else PREV_EMBEDDINGS_PATH / model
    )
    output_path.mkdir(parents=True, exist_ok=True)

    graphs = pd.read_pickle(GRAPHS_PATH / f"{dataset}.pkl")[dataset]
    tg_graphs = pd.read_pickle(TG_GRAPHS_PATH / f"{dataset}.pkl")[dataset]

    ds_embeddings, ds_metadata = AlignmentExperimentRunner(
        tg_graphs=tg_graphs["graphs"],
        graphs=graphs["graphs"],
        dataset=dataset,
        model_name=model,
        model_cls=model_cls,
        model_args=cfg["args"][dataset],
        nodes_mapping=tg_graphs["node_mappings"],
        logger=MlFlowLogger(
            client=MlflowClient(), experiment_name=f"ags_{dataset}"
        ),
        runs=runs,
        precomputed_rw=precomputed_rw_path,
        pretrained=pretrained,
        alignment_cfg=alignment_cfg,
        loss_scaling=loss_scaling,
        prev_snapshot_alignment=prev_snapshot_alignment,
    ).run()

    with open(output_path / f"{dataset}.pkl", "wb") as f:
        pickle.dump(obj=ds_embeddings, file=f)

    with open(output_path / f"{dataset}_metadata.pkl", "wb") as f:
        pickle.dump(
            obj={
                "is_alignment_model": model_cls.is_alignment_model(),
                "metrics": ds_metadata,
                **cfg,
            },
            file=f,
        )


if __name__ == "__main__":
    typer.run(main)
