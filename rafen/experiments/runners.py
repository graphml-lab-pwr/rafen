import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import networkx as nx
import pandas as pd
import plotly.express as px
from torch_geometric.data import Data
from tqdm.auto import tqdm

from rafen.embeddings.keyedmodel import KeyedModel
from rafen.experiments.metadata import (
    convert_ds_metadata_to_df,
    convert_loss_metadata_to_df,
)
from rafen.models.base import AlignmentModel, BaseTorchModel
from rafen.models.gae import BaseGNNTorchModel
from rafen.models.rafen_gae import RAFENGAE, RAFENGAEWeighted
from rafen.models.rafen_node2vec import RAFENNode2Vec, RAFENNode2VecWeighted
from rafen.models.random_walk_samplers import Node2VecRandomWalkSampler
from rafen.utils.mlflow_logger import MlFlowLogger
from rafen.utils.plotly_utils import plot_loss_plot


@dataclasses.dataclass
class ExperimentRunner:
    model_cls: Type[Union[BaseTorchModel, BaseGNNTorchModel]]
    model_args: Dict[str, Any]
    nodes_mapping: List[Dict[Any, Any]]
    graphs: List[nx.Graph]
    tg_graphs: List[Data]
    dataset: str
    model_name: str
    logger: Optional[MlFlowLogger]
    runs: int
    precomputed_rw: Optional[Path]
    mlflow_run_name_prefix: str = dataclasses.field(init=False)

    def __post_init__(self):
        self.mlflow_run_name_prefix = self.model_name
        if self.precomputed_rw and issubclass(
            self.model_cls, BaseGNNTorchModel
        ):
            raise ValueError(
                "Passed precomputed_rw argument to GNN based model."
            )

    @staticmethod
    def _embed_model(
        model: Union[BaseTorchModel, BaseGNNTorchModel],
    ) -> Tuple[KeyedModel, Dict[str, Any]]:
        embedding, metrics = model.embed()
        model.free_cuda_resources()
        return embedding, metrics

    def _get_run_mlflow_tags(self) -> Dict[str, Any]:
        return {
            "name": self.model_name,
            "is_alignment_model": self.model_cls.is_alignment_model(),
            "is_posthoc": (
                self.model_cls.alignment_is_posthoc()
                if issubclass(self.model_cls, AlignmentModel)
                else False
            ),
            "id.run": "run_summary",
        }

    def _get_snapshot_mlflow_tags(
        self, snapshot_id: int, run_id: int
    ) -> Dict[str, Any]:
        return {
            "model.cls": str(self.model_cls),
            "id.snapshot": str(snapshot_id),
            "id.run": str(run_id),
        }

    def _get_snapshot_pbar_loop(self) -> tqdm:
        return tqdm(
            range(len(self.tg_graphs) - 1),
            total=len(self.tg_graphs) - 1,
            desc="Snapshot",
            leave=False,
        )

    def _init_n2v_model(self, snapshot_id: int, run_id: int) -> BaseTorchModel:
        model: BaseTorchModel = self.model_cls(
            data=self.tg_graphs[snapshot_id],
            node_index_mapping=self.nodes_mapping[snapshot_id],
            random_walks_path=self.precomputed_rw
            / f"run_{run_id}"
            / f"snapshot_{snapshot_id}"
            if self.precomputed_rw
            else None,
            **self.model_args,
        )
        return model

    def _init_gnn_model(self, snapshot_id: int) -> BaseGNNTorchModel:
        model: BaseGNNTorchModel = self.model_cls(
            data=self.tg_graphs[snapshot_id],
            node_index_mapping=self.nodes_mapping[snapshot_id],
            **self.model_args,
        )
        return model

    def _train_model(
        self, snapshot_id: int, run_id: int
    ) -> Tuple[KeyedModel, Dict[str, Any]]:
        if issubclass(self.model_cls, BaseTorchModel):
            model = self._init_n2v_model(snapshot_id=snapshot_id, run_id=run_id)
        elif issubclass(self.model_cls, BaseGNNTorchModel):
            model = self._init_gnn_model(snapshot_id=snapshot_id)
        else:
            raise ValueError(
                f"Unrecognized model class: {self.model_cls}. Training failed!"
            )
        return self._embed_model(model)

    def _train_run_embeddings(
        self, run_id: int, parent_mlflow_run_id: str
    ) -> Tuple[List[KeyedModel], List[Dict[str, Any]]]:
        embeddings = []
        metrics = []

        snapshot_pbar = self._get_snapshot_pbar_loop()
        for snapshot_id in snapshot_pbar:
            run_name = f"{self.mlflow_run_name_prefix}_run_{run_id}_snapshot_{snapshot_id}"
            if self.logger:
                self.logger.start_run(
                    run_name=run_name, parent_run_id=parent_mlflow_run_id
                )
                self.logger.set_tags(
                    run_name=run_name,
                    tags=self._get_snapshot_mlflow_tags(
                        snapshot_id=snapshot_id, run_id=run_id
                    ),
                )

            snapshot_embedding, snapshot_metrics = self._train_model(
                snapshot_id=snapshot_id, run_id=run_id
            )
            if self.logger:
                self.logger.log_metrics(
                    run_name=run_name,
                    metrics={
                        k: v
                        for k, v in snapshot_metrics.items()
                        if k != "enhanced_time_log"
                    },
                )
                self.logger.terminate_run(run_name=run_name)

            embeddings.append(snapshot_embedding)
            metrics.append(snapshot_metrics)
        return embeddings, metrics

    @staticmethod
    def _convert_metrics_list_to_df(
        metrics: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        df = pd.DataFrame(metrics)
        df = df.applymap(lambda x: x[-1] if isinstance(x, list) else x)
        return df

    def _log_params(self, run_name: str):
        if self.logger:
            self.logger.log_params(run_name=run_name, params=self.model_args)

    def _execute_runs(
        self, pretrained: Optional[List[List[KeyedModel]]] = None
    ) -> Tuple[List[List[KeyedModel]], List[List[Dict[str, Any]]]]:
        ds_embeddings = []
        ds_metrics = []
        for run in tqdm(
            range(self.runs), total=self.runs, leave=False, desc="Run"
        ):
            mlflow_run_name = f"{self.mlflow_run_name_prefix}_run_{run}_summary"
            if self.logger:
                self.logger.set_tags(
                    run_name=mlflow_run_name,
                    tags=self._get_run_mlflow_tags(),
                )
                self._log_params(run_name=mlflow_run_name)

            embeddings, metrics = self._train_run_embeddings(
                run_id=run,
                parent_mlflow_run_id=(
                    self.logger.get_run_id_by_name(run_name=mlflow_run_name)
                    if self.logger
                    else ""
                ),
            )
            if pretrained:
                embeddings.insert(0, pretrained[run][0])
            metrics_df = ExperimentRunner._convert_metrics_list_to_df(metrics)

            if self.logger:
                self.logger.log_metrics(
                    run_name=mlflow_run_name,
                    metrics=metrics_df.drop(
                        "enhanced_time_log", axis=1
                    ).to_dict(orient="list"),
                )
            ds_metrics.append(metrics)
            ds_embeddings.append(embeddings)
            if self.logger:
                self.logger.terminate_run(run_name=mlflow_run_name)

        return ds_embeddings, ds_metrics

    def _get_plots(
        self,
        ds_metrics: List[List[Dict[str, Any]]],
        df: pd.DataFrame,
        summary_run_name: str,
        boxplots_metrics: List[str],
        loss_plots_metrics: List[str],
    ) -> None:
        for metric in boxplots_metrics:
            fig = px.box(
                df[df.metric_name == metric],
                y="value",
                x="snapshot",
                points="all",
                hover_data=["run"],
                title=f"{metric}",
            )
            self.logger.log_figure(
                run_name=summary_run_name,
                figure=fig,
                artifact_filename=f"{metric}.html",
            )

        for metric in loss_plots_metrics:
            fig = plot_loss_plot(
                convert_loss_metadata_to_df(ds_metrics, metric_name=metric),
                x_col="epoch",
                y_mean_col=("value", "mean"),
                y_std_col=("value", "std"),
                title=f"{metric} lineplot",
            )
            self.logger.log_figure(
                run_name=summary_run_name,
                figure=fig,
                artifact_filename=f"{metric}lineplot.html",
            )

    def prepare_mlflow_summary_run(
        self,
        ds_metrics: List[List[Dict[str, Any]]],
        boxplots_metrics: List[str],
        loss_plots_metrics: List[str],
    ):
        summary_run_name = f"{self.mlflow_run_name_prefix}_summary"
        self.logger.set_tags(
            run_name=summary_run_name, tags={"run.id": "summary"}
        )
        self._log_params(run_name=summary_run_name)
        df = convert_ds_metadata_to_df(ds_metrics)

        self._get_plots(
            ds_metrics=ds_metrics,
            df=df,
            summary_run_name=summary_run_name,
            boxplots_metrics=boxplots_metrics,
            loss_plots_metrics=loss_plots_metrics,
        )
        self.logger.terminate_run(run_name=summary_run_name)

    def run(self) -> Tuple[List[List[KeyedModel]], List[List[Dict[str, Any]]]]:
        ds_embeddings, ds_metrics = self._execute_runs()
        if self.logger:
            self.prepare_mlflow_summary_run(
                ds_metrics,
                boxplots_metrics=["loss", "calculation_time"],
                loss_plots_metrics=["loss"],
            )
        return ds_embeddings, ds_metrics


@dataclasses.dataclass
class AlignmentExperimentRunner(ExperimentRunner):
    model_cls: Type[Union[RAFENNode2Vec, RAFENGAE]]
    pretrained: List[List[KeyedModel]]
    alignment_cfg: Dict[str, Any]
    loss_scaling: bool
    prev_snapshot_alignment: bool
    selector_cache: List[Any] = dataclasses.field(init=False, default=None)
    selector_args: Dict[str, Any] = dataclasses.field(init=False, default=None)
    alignment_loss_cfg: Dict[str, Any] = dataclasses.field(
        init=False, default=None
    )

    def __post_init__(self) -> None:
        self.mlflow_run_name_prefix = (
            (
                self.model_name
                if not self.prev_snapshot_alignment
                else f"{self.model_name}_prev"
            )
            if self.logger
            else ""
        )
        # Sparse computation unavailable for aligned models
        if "sparse" in self.model_args:
            self.model_args.pop("sparse")
        self.selector_args, selector_cache_path = self._parse_selector_args(
            self.alignment_cfg["selector"], dataset=self.dataset
        )
        if not self.selector_args.get("selection_method_args", None):
            raise ValueError("Selector function not presented!")

        self.alignment_loss_cfg = {"loss_scaling_mode": self.loss_scaling}

        if "alpha" in self.alignment_cfg.keys():
            self.alignment_loss_cfg["alpha"] = self.alignment_cfg["alpha"][
                self.dataset
            ]

        if selector_cache_path:
            cache_path = Path(selector_cache_path)
            self.selector_cache = pd.read_pickle(
                cache_path / f"{self.dataset}.pkl"
            )

    @staticmethod
    def _parse_selector_args(
        cfg: Dict[str, Any], dataset: str
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        selector_args = {
            "selection_method": cfg["method"],
            "selection_method_args": cfg["args"][dataset],
        }
        return selector_args, cfg.get("cache", None)

    def _log_params(self, run_name: str):
        self.logger.log_params(run_name=run_name, params=self.model_args)
        self.logger.log_params(
            run_name=run_name,
            params={
                **self.selector_args,
                **self.alignment_loss_cfg,
                "cache": bool(self.selector_cache),
                "prev_snapshot_alignment": self.prev_snapshot_alignment,
            },
        )

    def _init_aligned_n2v_model(
        self,
        snapshot_id: int,
        alignment_snapshot_id: int,
        snapshot_selector_cache: Any,
        run_id: int,
    ):
        model: RAFENNode2Vec = self.model_cls(
            data=self.tg_graphs[snapshot_id],
            graph=self.graphs[snapshot_id],
            ref_graph=self.graphs[alignment_snapshot_id],
            ref_embedding=self.pretrained[run_id][alignment_snapshot_id],
            selector_cache=snapshot_selector_cache,
            node_index_mapping=self.nodes_mapping[snapshot_id],
            selector=self.alignment_cfg["selector"]["cls"],
            selector_args=self.selector_args,
            random_walks_path=(
                self.precomputed_rw
                / f"run_{run_id}"
                / f"snapshot_{snapshot_id}"
                if self.precomputed_rw
                else None
            ),
            **self.alignment_loss_cfg,
            **self.model_args,
        )
        return model

    def _init_aligned_gnn_model(
        self,
        snapshot_id: int,
        alignment_snapshot_id: int,
        snapshot_selector_cache: Any,
        run_id: int,
    ):
        model: RAFENGAE = self.model_cls(
            data=self.tg_graphs[snapshot_id],
            graph=self.graphs[snapshot_id],
            ref_graph=self.graphs[alignment_snapshot_id],
            ref_embedding=self.pretrained[run_id][alignment_snapshot_id],
            selector_cache=snapshot_selector_cache,
            node_index_mapping=self.nodes_mapping[snapshot_id],
            selector=self.alignment_cfg["selector"]["cls"],
            selector_args=self.selector_args,
            **self.alignment_loss_cfg,
            **self.model_args,
        )
        return model

    def _train_model(
        self, snapshot_id: int, run_id: int
    ) -> Tuple[KeyedModel, Dict[str, Any]]:
        snapshot_selector_cache = (
            self.selector_cache[snapshot_id - 1]
            if self.selector_cache
            else None
        )
        alignment_snapshot_id = (
            0 if not self.prev_snapshot_alignment else snapshot_id - 1
        )

        if issubclass(
            self.model_cls,
            (
                RAFENNode2Vec,
                RAFENNode2VecWeighted,
            ),
        ):
            model = self._init_aligned_n2v_model(
                snapshot_id=snapshot_id,
                run_id=run_id,
                alignment_snapshot_id=alignment_snapshot_id,
                snapshot_selector_cache=snapshot_selector_cache,
            )
        elif issubclass(
            self.model_cls,
            (RAFENGAE, RAFENGAEWeighted),
        ):
            model = self._init_aligned_gnn_model(
                snapshot_id=snapshot_id,
                run_id=run_id,
                alignment_snapshot_id=alignment_snapshot_id,
                snapshot_selector_cache=snapshot_selector_cache,
            )

        else:
            raise ValueError(
                f"Unrecognized model class: {self.model_cls}. Training failed!"
            )
        return self._embed_model(model)

    def _get_snapshot_pbar_loop(self) -> tqdm:
        return tqdm(
            range(1, len(self.tg_graphs)),
            total=len(self.tg_graphs) - 1,
            desc="Snapshot",
            leave=False,
        )

    def run(self):
        ds_embeddings, ds_metrics = self._execute_runs(
            pretrained=self.pretrained
        )
        if self.logger:
            self.prepare_mlflow_summary_run(
                ds_metrics,
                boxplots_metrics=[
                    "loss",
                    "calculation_time",
                    "n2v_loss"
                    if issubclass(
                        self.model_cls,
                        (
                            RAFENNode2Vec,
                            RAFENNode2VecWeighted,
                        ),
                    )
                    else "model_loss",
                    "l2_ref_nodes_distance",
                ],
                loss_plots_metrics=[
                    "loss",
                    "n2v_loss"
                    if issubclass(
                        self.model_cls,
                        (
                            RAFENNode2Vec,
                            RAFENNode2VecWeighted,
                        ),
                    )
                    else "model_loss",
                    "l2_ref_nodes_distance",
                ],
            )
        return ds_embeddings, ds_metrics


@dataclasses.dataclass
class RandomWalksSamplerRunner:
    tg_graphs: List[Data]
    dataset: str
    runs: int
    sampler_args: Dict[str, Any]
    output_dir: Path

    def sample_snapshots_rw(self, run_id: int):
        for sanpshot_id, snapshot in tqdm(
            enumerate(self.tg_graphs),
            desc="Sanpshot ",
            total=len(self.tg_graphs),
            leave=False,
        ):
            output_dir = (
                self.output_dir / f"run_{run_id}" / f"snapshot_{sanpshot_id}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            Node2VecRandomWalkSampler(
                data=snapshot,
                output_dir=output_dir,
                **self.sampler_args,
            ).sample()

    def run(self):
        for run in tqdm(range(self.runs), desc="Run ", leave=True):
            self.sample_snapshots_rw(run_id=run)
