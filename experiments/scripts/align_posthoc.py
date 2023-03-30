import os
import pickle
from copy import deepcopy
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import pandas as pd
import plotly.express as px
import typer
from tqdm import tqdm

from rafen.alignment.posthoc import PosthocSelectorAlignmentModel
from rafen.embeddings.keyedmodel import KeyedModel
from rafen.experiments.defaults import NODE2VEC
from rafen.experiments.metadata import convert_ds_metadata_to_df
from rafen.experiments.paths import (
    EMBEDDINGS_PATH,
    EXPERIMENTS_CFG_PATH,
    GRAPHS_PATH,
    POSTHOC_PATH,
    PREV_POSTHOC_PATH,
)
from rafen.models.regressors import OrthogonalProcrustesRegressor
from rafen.utils.io import read_yaml
from rafen.utils.mlflow_utils import get_mlflow_experiment, log_mlflow_metrics


def train_model(
    model: PosthocSelectorAlignmentModel,
) -> Tuple[KeyedModel, Dict[str, Any], Dict[str, Any]]:
    artifacts = {}

    metadata = {}
    start_time = timer()
    embedding, ref_nodes, scores = model.align()
    end_time = timer()

    calculation_time = end_time - start_time
    metadata["l2_ref_nodes_distance"] = model.l2_ref_nodes_distance()
    metadata["calculation_time"] = calculation_time
    artifacts.update(
        {
            "ref_nodes": ref_nodes,
            "scores": scores,
            "calculation_time": calculation_time,
        }
    )

    log_mlflow_metrics(metadata)

    return embedding, metadata, artifacts


def train_eval_posthoc_models(
    graphs: List[Any],
    model_cls: str,
    mlflow_run_model_name: str,
    model_args: Dict[str, Any],
    run_id: int,
    pretrained: List[Any],
    experiment_id: str,
    prev_snapshot_alignment: bool,
    selector_cache: Optional[List[Any]] = None,
):
    all_metadata = []
    pretrained = pretrained[run_id]
    embeddings = [pretrained[0]]
    graphs = graphs["graphs"]

    for idx in tqdm(
        range(1, len(graphs) - 1),
        total=len(graphs) - 1,
        leave=False,
        desc="Snapshot",
    ):
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"{mlflow_run_model_name}_run_{run_id}_snapshot_{idx}",
            nested=True,
        ):
            alignment_snapshot_id = (
                0 if not prev_snapshot_alignment else idx - 1
            )
            snapshot_selector_cache = (
                selector_cache[idx - 1] if selector_cache else None
            )
            model: PosthocSelectorAlignmentModel = (
                PosthocSelectorAlignmentModel(
                    graph=graphs[idx],
                    embedding=pretrained[idx],
                    ref_graph=graphs[alignment_snapshot_id],
                    ref_embedding=pretrained[alignment_snapshot_id],
                    selector_cache=snapshot_selector_cache,
                    node_index_mapping=None,
                    selector=model_cls,
                    selector_args=model_args,
                    regressor=OrthogonalProcrustesRegressor,
                )
            )
            embedding, metadata, artifacts = train_model(model)

            embeddings.append(embedding)
            all_metadata.append(metadata)
            # pretrained = embedding

    return embeddings, all_metadata


def main(
    model: str = typer.Option(..., help="Model name"),
    prev_snapshot_alignment: bool = typer.Option(
        False, help="Whether to align to previous snapshot instead of zero"
    ),
    use_gae_embeddings: bool = typer.Option(
        False, help="Whether to use GAE embeddings"
    ),
):
    if not os.getenv("MLFLOW_TRACKING_URI"):
        raise RuntimeError("ENV variable MLFLOW_TRACKING_URI is unset!")
    # Read config
    args = read_yaml(EXPERIMENTS_CFG_PATH / "posthoc" / f"{model}.yaml")

    if use_gae_embeddings:
        embeddings_path = EMBEDDINGS_PATH / "GAE"
        model_name = f"{model}_GAE"
    else:
        embeddings_path = EMBEDDINGS_PATH / NODE2VEC
        model_name = model

    posthoc_path = (
        PREV_POSTHOC_PATH if prev_snapshot_alignment else POSTHOC_PATH
    )
    output_path = posthoc_path / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    mlflow_run_model_name = (
        model_name if not prev_snapshot_alignment else f"{model_name}_prev"
    )
    dataset_pbar = tqdm(args["args"].keys())
    for dataset in dataset_pbar:
        experiment = get_mlflow_experiment(f"posthoc_{dataset}")
        ds_args = {
            "selection_method": args["method"],
            "selection_method_args": args["args"][dataset],
        }

        dataset_pbar.set_description(f"Dataset ({dataset})")
        graphs = pd.read_pickle(GRAPHS_PATH / f"{dataset}.pkl")

        selector_cache = None
        if "cache" in args:
            selector_cache = pd.read_pickle(
                Path(args["cache"]).joinpath(f"{dataset}.pkl")
            )

        pretrained = pd.read_pickle(embeddings_path / f"{dataset}.pkl")
        runs = len(pretrained)

        ds_embeddings = []
        ds_metadata = []

        for run_id in tqdm(range(runs), total=runs, leave=False, desc="Run"):
            with mlflow.start_run(
                experiment_id=experiment.experiment_id,
                run_name=f"{mlflow_run_model_name}_run_{run_id}_summary",
            ):
                mlflow.log_param("run", run_id)
                mlflow.set_tags(
                    {
                        "model": model_name,
                        "is_alignment_model": True,
                        "is_posthoc": True,
                        "id.run": "run_summary",
                        "prev_snapshot_alignment": prev_snapshot_alignment,
                    }
                )

                embeddings, metadata = train_eval_posthoc_models(
                    graphs=graphs[dataset],
                    model_cls=args["cls"],
                    model_args=deepcopy(ds_args),
                    pretrained=pretrained,
                    selector_cache=selector_cache,
                    run_id=run_id,
                    experiment_id=experiment.experiment_id,
                    mlflow_run_model_name=mlflow_run_model_name,
                    prev_snapshot_alignment=prev_snapshot_alignment,
                )

                ds_embeddings.append(embeddings)
                ds_metadata.append(metadata)

        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=f"{mlflow_run_model_name}_summary",
        ):
            mlflow.set_tags({"run.id": "summary"})
            df = convert_ds_metadata_to_df(ds_metadata)
            fig = px.box(
                df[df.metric_name == "calculation_time"],
                y="value",
                x="snapshot",
                points="all",
                hover_data=["run"],
                title="Calculation time",
            )
            mlflow.log_figure(figure=fig, artifact_file="time.html")

            fig = px.box(
                df[df.metric_name == "l2_ref_nodes_distance"],
                y="value",
                x="snapshot",
                points="all",
                hover_data=["run"],
                title="L2 Ref nodes distance on last epoch",
            )
            mlflow.log_figure(
                figure=fig, artifact_file="l2_ref_nodes_distance.html"
            )

        with open(output_path / f"{dataset}.pkl", "wb") as f:
            pickle.dump(obj=ds_embeddings, file=f)

        with open(output_path / f"{dataset}_metadata.pkl", "wb") as f:
            pickle.dump(
                obj={
                    "is_alignment_model": True,
                    "metrics": ds_metadata,
                },
                file=f,
            )


if __name__ == "__main__":
    typer.run(main)
