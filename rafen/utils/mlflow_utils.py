from typing import Any, Dict

import mlflow
from mlflow.entities import Experiment


def log_mlflow_metrics(metadata: Dict[str, Any]) -> None:
    for metric_id, values in metadata.items():
        if isinstance(values, list):
            if len(values) > 1:
                for epoch_id, value in enumerate(values):
                    mlflow.log_metric(metric_id, value, step=epoch_id + 1)
            else:
                mlflow.log_metric(metric_id, values[0])
        else:
            mlflow.log_metric(metric_id, values)


def get_mlflow_experiment(name: str) -> Experiment:
    experiment = mlflow.get_experiment_by_name(name=name)
    if not experiment:
        mlflow.create_experiment(name=name)
        experiment = mlflow.get_experiment_by_name(name=name)

    return experiment


def get_mlflow_run_uid_by_name(run_name: str, experiment_id: str) -> str:
    df = mlflow.search_runs(experiment_ids=[experiment_id])
    df = df[df["tags.mlflow.runName"] == run_name]
    if len(df) == 0:
        raise ValueError(
            f"Mlflow run with name: {run_name} wasn't found in the db!"
        )

    mlflow_run_id = (
        df.sort_values(by="start_time", ascending=False).iloc[0].run_id
    )
    return mlflow_run_id
