import dataclasses
import pathlib
from typing import Any, ClassVar, Dict, List, Set, Tuple, Union

import pandas as pd
from tqdm.auto import tqdm

from rafen.experiments.defaults import (
    BASE_L2_ALIGNER,
    DATASETS,
    L2_ALIGNERS,
    METRICS,
    NODE2VEC,
    POSTHOC_ALIGNERS,
)
from rafen.experiments.paths import EMBEDDINGS_PATH, POSTHOC_PATH


@dataclasses.dataclass
class MetricsParser:
    metrics: List[str] = dataclasses.field(default_factory=lambda: METRICS)
    datasets: List[str] = dataclasses.field(default_factory=lambda: DATASETS)
    aligners: List[str] = dataclasses.field(default_factory=lambda: L2_ALIGNERS)
    base_l2_aligner: str = BASE_L2_ALIGNER
    node2vec: str = NODE2VEC
    NODE2VEC_METRICS: ClassVar[Set[str]] = {"loss", "calculation_time"}
    POSTHOC_METRICS: ClassVar[Set[str]] = {
        "l2_ref_nodes_distance",
        "calculation_time",
    }

    @staticmethod
    def _read_data_file(
        path: Union[str, pathlib.Path]
    ) -> Tuple[List[Any], int, int, int]:
        data = pd.read_pickle(path)["metrics"]
        runs = len(data)
        snapshots = len(data[0])
        epochs = len(data[0][0]["loss"])
        return data, runs, snapshots, epochs

    @staticmethod
    def _parse_calculation_metric(
        data: List[Any],
        model: str,
        snapshots: int,
        runs: int,
        dataset: str,
    ) -> List[Dict[str, Any]]:

        parsed = []
        for snapshot_id in range(snapshots):
            for run_id in range(runs):
                parsed.append(
                    {
                        "metric": "calculation_time",
                        "snapshot": snapshot_id
                        if model == "Node2Vec"
                        else snapshot_id + 1,
                        "run": run_id,
                        "value": data[run_id][snapshot_id]["calculation_time"],
                        "dataset": dataset,
                        "aligner": model,
                    }
                )
        return parsed

    def _parse_metric(
        self,
        data: List[Any],
        model: str,
        metric: str,
        snapshots: int,
        runs: int,
        epochs: int,
        dataset: str,
    ) -> List[Dict[str, Any]]:
        metric_name = (
            metric
            if not (model == self.node2vec and metric == "loss")
            else "n2v_loss"
        )

        parsed = []
        for snapshot_id in range(snapshots):
            for epoch_id in range(epochs):
                for run_id in range(runs):
                    parsed.append(
                        {
                            "metric": metric_name,
                            "snapshot": snapshot_id
                            if model == "Node2Vec"
                            else snapshot_id + 1,
                            "epoch": epoch_id,
                            "run": run_id,
                            "value": data[run_id][snapshot_id][metric][
                                epoch_id
                            ],
                            "dataset": dataset,
                            "aligner": model,
                        }
                    )

        return parsed

    def _parse_ds_metrics(
        self, path: Union[str, pathlib.Path], dataset: str, model: str
    ):
        metrics = self.metrics
        if model == self.node2vec:
            metrics = self.NODE2VEC_METRICS

        parsed = []
        data, runs, snapshots, epochs = MetricsParser._read_data_file(path)
        for metric in metrics:
            if metric == "calculation_time":
                parsed += self._parse_calculation_metric(
                    data=data,
                    model=model,
                    dataset=dataset,
                    snapshots=snapshots,
                    runs=runs,
                )
            else:
                parsed += self._parse_metric(
                    data=data,
                    model=model,
                    dataset=dataset,
                    snapshots=snapshots,
                    runs=runs,
                    epochs=epochs,
                    metric=metric,
                )
        return parsed

    @staticmethod
    def _parse_posthoc_ds_metrics(
        path: Union[str, pathlib.Path], dataset: str, model: str
    ) -> List[Any]:
        parsed = []
        data = pd.read_pickle(path)["metrics"]

        runs = len(data)
        snapshots = len(data[0])

        for metric in MetricsParser.POSTHOC_METRICS:
            for run_id in range(runs):
                for snapshot_id in range(snapshots):
                    parsed.append(
                        {
                            "metric": metric,
                            "snapshot": snapshot_id + 1,
                            "run": run_id,
                            "value": data[run_id][snapshot_id][metric],
                            "dataset": dataset,
                            "aligner": model,
                        }
                    )
        return parsed

    def parse(self) -> List[Any]:
        data = []
        for ds in tqdm(self.datasets, desc="Dataset", leave=False):
            for model in tqdm(
                [*self.aligners, self.node2vec],
                desc="model",
                leave=False,
            ):
                if model in POSTHOC_ALIGNERS:
                    metric_path = POSTHOC_PATH / model / f"{ds}_metadata.pkl"
                    data += self._parse_posthoc_ds_metrics(
                        path=metric_path,
                        dataset=ds,
                        model=model,
                    )
                else:
                    metric_path = EMBEDDINGS_PATH / model / f"{ds}_metadata.pkl"
                    data += self._parse_ds_metrics(
                        path=metric_path,
                        dataset=ds,
                        model=model,
                    )
        return data
