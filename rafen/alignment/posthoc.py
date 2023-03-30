from typing import Any, Dict, Optional, Type

import networkx as nx
import torch

from rafen.embeddings.keyedmodel import KeyedModel
from rafen.metrics.distances import calculate_l2_distance
from rafen.models.posthoc import BasePosthocAlignmentModel
from rafen.models.regressors import OrthogonalProcrustesRegressor


class PosthocSelectorAlignmentModel(BasePosthocAlignmentModel):
    def __init__(
        self,
        selector: str,
        selector_args: Dict[Any, Any],
        graph: nx.Graph,
        ref_graph: nx.Graph,
        embedding: KeyedModel,
        ref_embedding: KeyedModel,
        regressor: Type[OrthogonalProcrustesRegressor],
        selector_cache: Optional[Dict[Any, float]] = None,
        node_index_mapping: Optional[Dict[Any, Any]] = None,
    ) -> None:

        super().__init__(
            selector,
            selector_args,
            graph,
            ref_graph,
            embedding,
            ref_embedding,
            selector_cache,
            node_index_mapping,
        )
        self.calibrated_embedding = None
        self.regressor = regressor()

    def l2_ref_nodes_distance(self):
        return (
            calculate_l2_distance(
                torch.from_numpy(
                    self.reference_embedding.to_numpy(self.ref_nodes)
                ),
                torch.from_numpy(
                    self.calibrated_embedding.to_numpy(self.ref_nodes)
                ),
            )
            .cpu()
            .numpy()
            .item()
        )

    def predict(self, v):
        emb_shape = v.shape[1]
        v_nodes = v.nodes
        v = v.to_numpy(v_nodes)
        emb = self.regressor.predict(v)
        return KeyedModel(
            size=emb_shape,
            node_emb_vectors=dict(zip(v_nodes, emb)),
            fill_unknown_nodes=False,
        )

    def align(self):
        u_train = self.reference_embedding.to_numpy(self.ref_nodes)
        v_train = self.embedding.to_numpy(self.ref_nodes)
        self.regressor.fit(v_train, u_train)
        self.calibrated_embedding = self.predict(self.embedding)
        return self.calibrated_embedding, self.ref_nodes, self.scores
