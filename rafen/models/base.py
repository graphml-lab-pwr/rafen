import abc
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm.auto import tqdm

from rafen.embeddings.keyedmodel import KeyedModel
from rafen.utils.misc import timeit
from rafen.utils.tg import convert_tg_model_to_km


class BaseModel:
    @staticmethod
    def is_alignment_model() -> bool:
        return False

    def __repr__(self) -> str:
        return type(self).__name__


class AlignmentModel(BaseModel):
    @staticmethod
    def is_alignment_model() -> bool:
        return True

    @staticmethod
    def alignment_is_posthoc() -> bool:
        return False


class BaseTorchModel(BaseModel, abc.ABC):
    ENHANCED_TIME_MEASUREMENT_KEYS = {
        "training_step_time",
        "loss_forward_time",
        "loss_backward_time",
        "n2v_loss_forward_time",
        "alignment_loss_forward_time",
        "optimizer_step_time",
        "zero_grad_time",
    }

    def __init__(
        self,
        data: Data,
        quiet: bool = False,
        node_index_mapping: Optional[Dict[Any, Any]] = None,
        random_walks_path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.node_index_mapping = node_index_mapping

        self.model = None
        self.loader = None
        self.model_args = None
        self.learning_args = None
        self.optimizer = None

        self.log = {}
        self.epoch_log = None
        self.time_log = None
        self.epoch_time_log = None
        self._init_logs()

        self.quiet = quiet
        self.random_walks_path = random_walks_path

    def _init_logs(self) -> None:
        self.epoch_log = {
            "loss": {"agg": np.mean, "values": []},
        }
        self.time_log = {
            "epoch_time_log": {},
            **{k: [] for k in BaseTorchModel.ENHANCED_TIME_MEASUREMENT_KEYS},
        }

        self.epoch_time_log = {
            k: [] for k in BaseTorchModel.ENHANCED_TIME_MEASUREMENT_KEYS
        }

    def free_cuda_resources(self) -> None:
        del self.model
        del self.loader
        if "cuda" in self.device:
            torch.cuda.empty_cache()

    def _calculate_on_epoch_end_logs(self):
        for k, v in self.epoch_log.items():
            if k not in self.log:
                self.log[k] = []
            self.log[k].append(v["agg"](v["values"]))

    def _update_times_log_after_batch(
        self,
        zero_grad_time: float,
        training_step_time: float,
        loss_backward_time: float,
        optimizer_step_time: float,
    ) -> None:
        self.epoch_time_log["zero_grad_time"].append(zero_grad_time)
        self.epoch_time_log["training_step_time"].append(training_step_time)
        self.epoch_time_log["loss_backward_time"].append(loss_backward_time)
        self.epoch_time_log["optimizer_step_time"].append(optimizer_step_time)

    def _update_times_log_after_epoch(self, epoch_id: int) -> None:
        self.time_log["epoch_time_log"][epoch_id] = deepcopy(
            self.epoch_time_log
        )
        for k in self.ENHANCED_TIME_MEASUREMENT_KEYS:
            if self.epoch_time_log[k]:
                self.time_log[k].append(np.sum(self.epoch_time_log[k]))
            self.epoch_time_log[k] = []

    @abc.abstractmethod
    def training_step(self, batch, batch_id: int) -> Any:
        pass

    @abc.abstractmethod
    def get_optimizer(self) -> None:
        pass

    def _fit(self) -> Dict[str, Any]:
        self.model.train()
        total_epochs = self.learning_args["epochs"] + 1

        for epoch in tqdm(
            range(1, total_epochs),
            disable=self.quiet,
            desc="Epoch",
            leave=False,
        ):
            if self.random_walks_path:
                self.loader = DataLoader(
                    pd.read_pickle(
                        self.random_walks_path / f"epoch_{epoch-1}.pkl"
                    ),
                    batch_size=1,
                    shuffle=False,
                )
            for batch_id, batch in enumerate(self.loader):
                zero_grad_time = timeit(self.optimizer.zero_grad)()[1]

                if self.random_walks_path:
                    batch = (batch[0][0], batch[1][0])

                loss, training_step_time = timeit(self.training_step)(
                    batch, batch_id
                )

                loss_backward_time = timeit(loss.backward)()[1]
                optimizer_step_time = timeit(self.optimizer.step)()[1]

                self._update_times_log_after_batch(
                    zero_grad_time=zero_grad_time,
                    training_step_time=training_step_time,
                    loss_backward_time=loss_backward_time,
                    optimizer_step_time=optimizer_step_time,
                )

            self._calculate_on_epoch_end_logs()
            self._update_times_log_after_epoch(epoch_id=epoch)

        self.model.eval()
        return self.log

    def embed(self) -> Tuple[KeyedModel, Dict[str, Any]]:
        train_log, fit_time = timeit(self._fit)()

        train_log["calculation_time"] = fit_time
        train_log["enhanced_time_log"] = self.time_log
        return (
            convert_tg_model_to_km(self.model, self.node_index_mapping),
            train_log,
        )
