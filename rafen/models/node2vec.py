from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec

from rafen.models.base import BaseTorchModel
from rafen.utils.misc import timeit


class Node2VecEmbedding(BaseTorchModel):
    """Class that wraps Node2Vec method."""

    def __init__(
        self,
        data: Data,
        dimensions: int,
        walk_length: int,
        nb_walks_per_node: int,
        p: float,
        q: float,
        batch_size: int,
        lr: float,
        w2v_epochs: int,
        context_size: int,
        node_index_mapping: Optional[Dict[Any, Any]] = None,
        log_steps: int = 1,
        quiet: bool = False,
        sparse: bool = True,
        random_walks_path: Optional[Path] = None,
    ):
        """Inits the Node2VecEmbedding class."""
        super(Node2VecEmbedding, self).__init__(
            data=data,
            quiet=quiet,
            node_index_mapping=node_index_mapping,
            random_walks_path=random_walks_path,
        )
        self.model_args = dict(
            embedding_dim=dimensions,
            walk_length=walk_length,
            walks_per_node=nb_walks_per_node,
            p=p,
            q=q,
            sparse=sparse,
            context_size=context_size,
        )
        self.learning_args = dict(
            batch_size=batch_size, lr=lr, epochs=w2v_epochs, log_steps=log_steps
        )
        self.loader = None
        self._build_model()

    def get_optimizer(self) -> None:
        if self.model_args["sparse"]:
            self.optimizer = torch.optim.SparseAdam(
                list(self.model.parameters()), lr=self.learning_args["lr"]
            )
        else:
            self.optimizer = torch.optim.Adam(
                list(self.model.parameters()), lr=self.learning_args["lr"]
            )

    def _build_model(self) -> None:
        self.model = Node2Vec(self.data.edge_index, **self.model_args).to(
            self.device
        )
        if not self.random_walks_path:
            self.loader = self.model.loader(
                batch_size=self.learning_args["batch_size"],
                shuffle=True,
                pin_memory=True,
                num_workers=0,
            )
        self.get_optimizer()

    def n2v_loss(self, pos_rw, neg_rw):
        return self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))

    def training_step(self, batch, batch_id):
        pos_rw, neg_rw = batch
        loss, loss_forward_time = timeit(self.n2v_loss)(pos_rw, neg_rw)

        self.epoch_log["loss"]["values"].append(loss.detach().cpu().numpy())
        self.epoch_time_log["loss_forward_time"].append(loss_forward_time)

        return loss
