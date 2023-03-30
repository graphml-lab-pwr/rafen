import abc
from typing import Any, Dict, Optional

import networkx as nx

from rafen.activity_scoring.selectors import get_reference_nodes
from rafen.embeddings.keyedmodel import KeyedModel
from rafen.models.base import AlignmentModel


class BasePosthocAlignmentModel(AlignmentModel, abc.ABC):
    def __init__(
        self,
        selector: str,
        selector_args: Dict[Any, Any],
        graph: nx.Graph,
        ref_graph: nx.Graph,
        embedding: KeyedModel,
        ref_embedding: KeyedModel,
        selector_cache: Optional[Dict[Any, float]] = None,
        node_index_mapping: Optional[Dict[Any, Any]] = None,
    ):
        super().__init__()
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
        self.reference_embedding = ref_embedding
        self.embedding = embedding

    @staticmethod
    def alignment_is_posthoc() -> bool:
        return True
