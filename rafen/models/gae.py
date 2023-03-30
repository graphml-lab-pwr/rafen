import abc
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch_geometric.utils
from torch.nn import Embedding
from torch_geometric.data import Data
from torch_geometric.nn import GAE, GCNConv
from tqdm.auto import tqdm

from rafen.embeddings.keyedmodel import KeyedModel
from rafen.models.base import BaseModel
from rafen.utils.misc import timeit


class BaseGNNTorchModel(BaseModel, abc.ABC):
    ENHANCED_TIME_MEASUREMENT_KEYS = {
        "training_step_time",
        "loss_forward_time",
        "loss_backward_time",
        "gnn_loss_forward_time",
        "alignment_loss_forward_time",
        "optimizer_step_time",
        "zero_grad_time",
    }

    def __init__(
        self,
        data: Data,
        quiet: bool = False,
        node_index_mapping: Optional[Dict[Any, Any]] = None,
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

    def _init_logs(self) -> None:
        self.epoch_log = {
            "loss": {"agg": np.mean, "values": []},
        }
        self.time_log = {
            "epoch_time_log": {},
            **{k: [] for k in BaseGNNTorchModel.ENHANCED_TIME_MEASUREMENT_KEYS},
        }

        self.epoch_time_log = {
            k: [] for k in BaseGNNTorchModel.ENHANCED_TIME_MEASUREMENT_KEYS
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
    def training_step(self, batch) -> Any:
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

            zero_grad_time = timeit(self.optimizer.zero_grad)()[1]

            loss, training_step_time = timeit(self.training_step)(self.data)

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
            self._convert_tg_gnn_model_to_km(),
            train_log,
        )

    def _convert_tg_gnn_model_to_km(self) -> KeyedModel:
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.encode(self.data.edge_index)

        emb_dict = {}
        for node_id, node_emb in enumerate(embeddings):
            if self.node_index_mapping:
                node_id = self.node_index_mapping[node_id]

            emb_dict[str(node_id)] = node_emb.cpu().numpy()

        return KeyedModel(
            size=embeddings.shape[1],
            node_emb_vectors=emb_dict,
            fill_unknown_nodes=False,
        )


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(GCNEncoder, self).__init__()
        self.embedding_layer = Embedding(
            num_embeddings=in_channels, embedding_dim=out_channels
        )
        self.conv1 = GCNConv(out_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.embedding_layer.weight, edge_index).relu()
        return self.conv2(x, edge_index)


class GAEEmbedding(BaseGNNTorchModel):
    def __init__(
        self,
        data: Data,
        dimensions: int,
        lr: float,
        epochs: int,
        node_index_mapping: Optional[Dict[Any, Any]] = None,
        log_steps: int = 1,
        quiet: bool = False,
        sparse: bool = True,
    ):
        super(GAEEmbedding, self).__init__(
            data=data,
            quiet=quiet,
            node_index_mapping=node_index_mapping,
        )
        if data.is_directed():
            data.edge_index = torch_geometric.utils.to_undirected(
                data.edge_index
            )
        self.model_args = dict(
            embedding_dim=dimensions,
            sparse=sparse,
        )
        self.learning_args = dict(lr=lr, epochs=epochs, log_steps=log_steps)
        self._build_model()

    def get_x_features(self) -> torch.Tensor:
        return torch.from_numpy(
            np.eye(self.data.num_nodes, dtype=np.float32)
        ).to(device=self.device)

    def _build_model(self):
        self.model = (
            GAE(
                GCNEncoder(
                    self.data.num_nodes, self.model_args["embedding_dim"]
                )
            )
            .float()
            .to(self.device)
        )
        self.data.to(self.device)
        self.get_optimizer()

    def get_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()), lr=self.learning_args["lr"]
        )

    def model_loss(
        self, z: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        return self.model.recon_loss(z=z, pos_edge_index=edge_index)

    def training_step(self, batch) -> torch.Tensor:
        self.model.train()

        z = self.model.encode(batch.edge_index)
        loss, loss_forward_time = timeit(self.model_loss)(z, batch.edge_index)

        self.epoch_log["loss"]["values"].append(loss.detach().cpu().numpy())
        self.epoch_time_log["loss_forward_time"].append(loss_forward_time)
        return loss
