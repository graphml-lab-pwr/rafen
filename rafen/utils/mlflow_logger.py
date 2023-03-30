import dataclasses
from typing import Any, Dict, Optional

import mlflow
from mlflow.entities import Experiment, Run
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME

from rafen.utils.mlflow_utils import get_mlflow_experiment


@dataclasses.dataclass
class MlFlowRun:
    client: mlflow.tracking.MlflowClient
    experiment_id: str
    run_name: str
    parent_run_id: Optional[str] = None
    run: Run = dataclasses.field(init=False)
    run_id: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.run = self.client.create_run(
            experiment_id=self.experiment_id, tags=self._resolve_tags()
        )
        self.run_id = self.run.info.run_id

    def _resolve_tags(self) -> Dict[str, Any]:
        tags = {MLFLOW_RUN_NAME: self.run_name}
        if self.parent_run_id:
            tags[MLFLOW_PARENT_RUN_ID] = self.parent_run_id
        return tags

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        for metric_id, values in metrics.items():
            if isinstance(values, list):
                if len(values) > 1:
                    for epoch_id, value in enumerate(values):
                        self.client.log_metric(
                            run_id=self.run_id,
                            key=metric_id,
                            value=value,
                            step=epoch_id + 1,
                        )
                else:
                    self.client.log_metric(
                        run_id=self.run_id, key=metric_id, value=values[0]
                    )
            else:
                self.client.log_metric(
                    run_id=self.run_id, key=metric_id, value=values
                )

    def log_params(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            self.client.log_param(run_id=self.run_id, key=key, value=value)

    def log_figure(self, figure: Any, artifact_filename: str) -> None:
        self.client.log_figure(
            run_id=self.run_id, figure=figure, artifact_file=artifact_filename
        )

    def set_tags(self, tags: Dict[str, Any]) -> None:
        for key, value in tags.items():
            self.client.set_tag(run_id=self.run_id, key=key, value=value)

    def terminate(self) -> None:
        self.client.set_terminated(run_id=self.run_id)


@dataclasses.dataclass
class MlFlowLogger:
    client: mlflow.tracking.MlflowClient
    experiment_name: str
    experiment: Experiment = dataclasses.field(init=False)
    _runs: Dict[str, MlFlowRun] = dataclasses.field(
        init=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self.experiment = get_mlflow_experiment(self.experiment_name)

    def start_run(
        self, run_name: str, parent_run_id: Optional[str] = None
    ) -> None:
        if run_name not in self._runs.keys():
            self._runs[run_name] = MlFlowRun(
                client=self.client,
                experiment_id=self.experiment.experiment_id,
                run_name=run_name,
                parent_run_id=parent_run_id,
            )

    def get_run_id_by_name(self, run_name) -> str:
        if run_name not in self._runs.keys():
            raise ValueError(f"Run: {run_name} not found!")
        return self._runs[run_name].run_id

    def log_metrics(self, run_name: str, metrics: Dict[str, Any]) -> None:
        if run_name not in self._runs.keys():
            self.start_run(run_name=run_name)

        self._runs[run_name].log_metrics(metrics)

    def log_params(self, run_name: str, params: Dict[str, Any]) -> None:
        if run_name not in self._runs.keys():
            self.start_run(run_name=run_name)

        self._runs[run_name].log_params(params)

    def log_figure(self, run_name: str, figure: Any, artifact_filename) -> None:
        if run_name not in self._runs.keys():
            self.start_run(run_name=run_name)

        self._runs[run_name].log_figure(
            figure=figure, artifact_filename=artifact_filename
        )

    def set_tags(self, run_name: str, tags: Dict[str, Any]) -> None:
        if run_name not in self._runs.keys():
            self.start_run(run_name=run_name)

        self._runs[run_name].set_tags(tags)

    def terminate_run(self, run_name: str) -> None:
        self._runs[run_name].terminate()
