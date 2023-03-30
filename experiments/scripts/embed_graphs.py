import os
import pickle
from pathlib import Path

import pandas as pd
import typer
from mlflow.tracking import MlflowClient

from rafen.experiments.paths import (
    EMBEDDINGS_FULL_PATH,
    EMBEDDINGS_PATH,
    FULL_GRAPHS_PATH,
    FULL_TG_GRAPHS_PATH,
    GRAPHS_PATH,
    N2V_CFG_PATH,
    RANDOM_WALKS_PATH,
    TG_GRAPHS_PATH,
)
from rafen.experiments.runners import ExperimentRunner
from rafen.models.gae import BaseGNNTorchModel
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
    full_graphs: bool = typer.Option(
        False, help="Whether to train on full graphs"
    ),
    cfg: str = typer.Option(
        "", help="Config to pass. If arg is not passed reads default n2v config"
    ),
):
    if not os.getenv("MLFLOW_TRACKING_URI"):
        raise RuntimeError("ENV variable MLFLOW_TRACKING_URI is unset!")

    if not cfg:
        cfg = read_yaml(N2V_CFG_PATH)
    else:
        cfg = read_yaml(Path(cfg))

    model_cls = get_module_from_str(cfg["cls"])

    output_path = EMBEDDINGS_PATH / model
    output_path.mkdir(parents=True, exist_ok=True)

    if full_graphs:
        graphs = pd.read_pickle(FULL_GRAPHS_PATH / f"{dataset}.pkl")[dataset]
        tg_graphs = pd.read_pickle(FULL_TG_GRAPHS_PATH / f"{dataset}.pkl")[
            dataset
        ]
        output_path = EMBEDDINGS_FULL_PATH / model
        output_path.mkdir(parents=True, exist_ok=True)
        experiment_name = f"{dataset}_full"

    else:
        graphs = pd.read_pickle(GRAPHS_PATH / f"{dataset}.pkl")[dataset]
        tg_graphs = pd.read_pickle(TG_GRAPHS_PATH / f"{dataset}.pkl")[dataset]
        output_path = EMBEDDINGS_PATH / model
        output_path.mkdir(parents=True, exist_ok=True)

        if not issubclass(model_cls, BaseGNNTorchModel):
            experiment_name = dataset
        else:
            experiment_name = f"gnn_{dataset}"

    ds_embeddings, ds_metadata = ExperimentRunner(
        tg_graphs=tg_graphs["graphs"],
        graphs=graphs["graphs"],
        dataset=dataset,
        model_name=model,
        model_cls=model_cls,
        model_args=cfg["args"][dataset],
        nodes_mapping=tg_graphs["node_mappings"],
        logger=MlFlowLogger(
            client=MlflowClient(), experiment_name=experiment_name
        ),
        runs=runs,
        precomputed_rw=RANDOM_WALKS_PATH / dataset if precomputed_rw else None,
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
