from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from rafen.activity_scoring.selectors import get_reference_nodes
from rafen.embeddings.keyedmodel import KeyedModel
from rafen.metrics.distances import calculate_l2_distance
from rafen.models.base import AlignmentModel
from rafen.models.gae import GAEEmbedding
from rafen.utils.misc import timeit


class RAFENGAE(AlignmentModel, GAEEmbedding):
    def __init__(
        self,
        data: Data,
        dimensions: int,
        lr: float,
        epochs: int,
        selector: str,
        selector_args: Dict[Any, Any],
        graph: nx.Graph,
        ref_graph: nx.Graph,
        ref_embedding: KeyedModel,
        alpha: Optional[float] = None,
        node_index_mapping: Optional[Dict[Any, Any]] = None,
        selector_cache: Optional[Dict[Any, float]] = None,
        quiet: bool = True,
        loss_scaling_mode: bool = False,
        ignore_alpha_scaling: bool = False,
    ):
        super(RAFENGAE, self).__init__(
            data=data,
            node_index_mapping=node_index_mapping,
            dimensions=dimensions,
            lr=lr,
            epochs=epochs,
            quiet=quiet,
            sparse=False,
        )
        if not ignore_alpha_scaling and not alpha:
            raise ValueError(
                "Alpha parameter is missing. Set ignore_alpha_scaling flag or pass alpha value!"
            )
        self.node_index_mapping = node_index_mapping
        self.reference_graph = ref_graph
        self.graph = graph
        self.reverse_node_index_mapping = (
            dict(
                zip(
                    map(str, self.node_index_mapping.values()),
                    self.node_index_mapping.keys(),
                )
            )
            if self.node_index_mapping
            else None
        )

        (
            self.ref_nodes,
            self.ref_nodes_mapped,
            self.scores,
            self.scores_mapped,
        ) = get_reference_nodes(
            selector=selector,
            selector_args=selector_args,
            cache=selector_cache,
            graph=self.graph,
            reference_graph=self.reference_graph,
            reverse_node_index_mapping=self.reverse_node_index_mapping,
        )
        self.ref_nodes_mapped = torch.tensor(self.ref_nodes_mapped).to(
            self.device
        )
        self.reference_embedding = torch.tensor(
            ref_embedding.to_numpy(nodes=self.ref_nodes)
        ).to(self.device)

        if not ignore_alpha_scaling:
            self.distance_loss_weight = alpha
            self.model_loss_weight = 1 - alpha
        else:
            self.distance_loss_weight = 1
            self.model_loss_weight = 1

        self.epoch_log["alignment_loss"] = {"agg": np.mean, "values": []}
        self.epoch_log["model_loss"] = {"agg": np.mean, "values": []}
        self.epoch_log["l2_ref_nodes_distance"] = {"agg": np.mean, "values": []}
        self.alignment_loss = self._get_alignment_loss()
        self.loss_scaling_mode = loss_scaling_mode
        if self.loss_scaling_mode:
            self.model_loss_scale = None
            self.alignment_loss_scale = None

        self.embedding = None

    def l2_ref_nodes_distance(self):
        return calculate_l2_distance(
            self.reference_embedding, self.embedding[self.ref_nodes_mapped]
        )

    def _get_alignment_loss(self):
        return torch.nn.MSELoss()

    def calculate_alignment_loss(self):
        return self.alignment_loss(
            self.reference_embedding, self.embedding[self.ref_nodes_mapped]
        )

    def training_step(self, batch):
        self.model.train()

        self.embedding = self.model.encode(batch.edge_index)
        loss, loss_forward_time = timeit(self.model_loss)(
            self.embedding, batch.edge_index
        )

        self.epoch_log["loss"]["values"].append(loss.detach().cpu().numpy())
        self.epoch_time_log["loss_forward_time"].append(loss_forward_time)

        alignment_loss, alignment_loss_time = timeit(
            self.calculate_alignment_loss
        )()
        l2_ref_nodes_distance = self.l2_ref_nodes_distance()

        if not self.loss_scaling_mode:
            total_loss = (loss * self.model_loss_weight) + (
                alignment_loss * self.distance_loss_weight
            )
        else:
            if not (self.model_loss_scale or self.alignment_loss_scale):
                self.model_loss_scale = (
                    loss.detach() if self.model_loss_weight > 0 else 1
                )
                self.alignment_loss_scale = (
                    alignment_loss.detach()
                    if self.distance_loss_weight > 0
                    else 1
                )
            total_loss = (
                (loss / self.model_loss_scale) * self.model_loss_weight
            ) + (
                (alignment_loss / self.alignment_loss_scale)
                * self.distance_loss_weight
            )

        self.epoch_log["model_loss"]["values"].append(
            loss.detach().cpu().numpy()
        )
        self.epoch_log["alignment_loss"]["values"].append(
            alignment_loss.detach().cpu().numpy()
        )
        self.epoch_log["loss"]["values"].append(
            total_loss.detach().cpu().numpy()
        )
        self.epoch_log["l2_ref_nodes_distance"]["values"].append(
            l2_ref_nodes_distance.detach().cpu().numpy()
        )
        self.epoch_time_log["loss_forward_time"].append(
            loss_forward_time + alignment_loss_time
        )
        self.epoch_time_log["alignment_loss_forward_time"].append(
            alignment_loss_time
        )
        self.epoch_time_log["gnn_loss_forward_time"].append(loss_forward_time)

        return total_loss


class RAFENGAEWeighted(AlignmentModel, GAEEmbedding):
    def __init__(
        self,
        data: Data,
        dimensions: int,
        lr: float,
        epochs: int,
        selector: str,
        selector_args: Dict[Any, Any],
        graph: nx.Graph,
        ref_graph: nx.Graph,
        ref_embedding: KeyedModel,
        node_index_mapping: Optional[Dict[Any, Any]] = None,
        selector_cache: Optional[Dict[Any, float]] = None,
        quiet: bool = True,
        loss_scaling_mode: bool = False,
    ):
        super(RAFENGAEWeighted, self).__init__(
            data=data,
            node_index_mapping=node_index_mapping,
            dimensions=dimensions,
            lr=lr,
            epochs=epochs,
            quiet=quiet,
            sparse=False,
        )
        self.node_index_mapping = node_index_mapping
        self.reference_graph = ref_graph
        self.graph = graph
        self.reverse_node_index_mapping = (
            dict(
                zip(
                    map(str, self.node_index_mapping.values()),
                    self.node_index_mapping.keys(),
                )
            )
            if self.node_index_mapping
            else None
        )

        (
            self.ref_nodes,
            self.ref_nodes_mapped,
            self.scores,
            self.scores_mapped,
        ) = get_reference_nodes(
            selector=selector,
            selector_args=selector_args,
            cache=selector_cache,
            graph=self.graph,
            reference_graph=self.reference_graph,
            reverse_node_index_mapping=self.reverse_node_index_mapping,
        )
        self.ref_nodes_mapped = torch.tensor(self.ref_nodes_mapped).to(
            self.device
        )
        self.reference_embedding = torch.tensor(
            ref_embedding.to_numpy(nodes=self.ref_nodes)
        ).to(self.device)

        self.epoch_log["alignment_loss"] = {"agg": np.mean, "values": []}
        self.epoch_log["model_loss"] = {"agg": np.mean, "values": []}
        self.epoch_log["l2_ref_nodes_distance"] = {"agg": np.mean, "values": []}
        self.alignment_loss = self._get_alignment_loss()
        self.loss_scaling_mode = loss_scaling_mode
        if self.loss_scaling_mode:
            self.model_loss_scale = None
            self.alignment_loss_scale = None

        self.embedding = None

    def l2_ref_nodes_distance(self):
        return calculate_l2_distance(
            self.reference_embedding, self.embedding[self.ref_nodes_mapped]
        )

    def _get_alignment_loss(self):
        return torch.nn.MSELoss(reduction="none")

    def calculate_alignment_loss(self):
        self.scores_mapped = self.scores_mapped.to(self.device)
        self.scores_mapped = torch.nan_to_num(self.scores_mapped, nan=0.0)
        if self.scores_mapped.max() == 0.0:
            self.scores_mapped += 1.0

        return (
            self.alignment_loss(
                self.reference_embedding, self.embedding[self.ref_nodes_mapped]
            ).mean(axis=1)
            * self.scores_mapped
        ).mean()

    def training_step(self, batch):
        self.model.train()

        self.embedding = self.model.encode(batch.edge_index)
        loss, loss_forward_time = timeit(self.model_loss)(
            self.embedding, batch.edge_index
        )

        self.epoch_log["loss"]["values"].append(loss.detach().cpu().numpy())
        self.epoch_time_log["loss_forward_time"].append(loss_forward_time)

        alignment_loss, alignment_loss_time = timeit(
            self.calculate_alignment_loss
        )()
        l2_ref_nodes_distance = self.l2_ref_nodes_distance()

        if not self.loss_scaling_mode:
            total_loss = loss + alignment_loss
        else:
            if not (self.model_loss_scale or self.alignment_loss_scale):
                self.model_loss_scale = loss.detach()
                self.alignment_loss_scale = alignment_loss.detach()
            total_loss = (loss / self.model_loss_scale) + (
                (alignment_loss / self.alignment_loss_scale)
            )

        self.epoch_log["model_loss"]["values"].append(
            loss.detach().cpu().numpy()
        )
        self.epoch_log["alignment_loss"]["values"].append(
            alignment_loss.detach().cpu().numpy()
        )
        self.epoch_log["loss"]["values"].append(
            total_loss.detach().cpu().numpy()
        )
        self.epoch_log["l2_ref_nodes_distance"]["values"].append(
            l2_ref_nodes_distance.detach().cpu().numpy()
        )
        self.epoch_time_log["loss_forward_time"].append(
            loss_forward_time + alignment_loss_time
        )
        self.epoch_time_log["alignment_loss_forward_time"].append(
            alignment_loss_time
        )
        self.epoch_time_log["gnn_loss_forward_time"].append(loss_forward_time)

        return total_loss
