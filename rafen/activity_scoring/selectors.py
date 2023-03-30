import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

from rafen.utils.misc import get_common_nodes, get_module_from_str


def get_log_normed_scores(scores_dict: Dict[str, float]) -> Dict[str, float]:
    scores = list(scores_dict.values())
    return dict(
        zip(scores_dict.keys(), np.nan_to_num(np.log(scores), neginf=0))
    )


def get_flipped_scores(scores_dict: Dict[str, float]) -> Dict[str, float]:
    max_value = np.max(list(scores_dict.values()))
    return {
        node: 1 - (node_score / max_value)
        for node, node_score in scores_dict.items()
    }


def get_reference_nodes(
    selector: str,
    selector_args: Dict[Any, Any],
    graph: nx.Graph,
    reference_graph: nx.Graph,
    reverse_node_index_mapping: Dict[Any, Any],
    cache: Optional[Dict[Any, float]] = None,
) -> Tuple[List[Any], Optional[List[int]], Dict[Any, Any], torch.tensor]:
    log_norm_scores = False
    if "log_norm_scores" in selector_args["selection_method_args"]:
        log_norm_scores = selector_args["selection_method_args"].pop(
            "log_norm_scores"
        )

    selector: AbstractNodesSelector = get_module_from_str(selector)(
        **selector_args
    )
    ref_nodes, scores = selector.select(
        g_u=reference_graph, g_v=graph, cache=cache
    )
    ref_nodes_mapped = None
    if reverse_node_index_mapping:
        ref_nodes_mapped = [reverse_node_index_mapping[it] for it in ref_nodes]

    if log_norm_scores:
        scores = get_log_normed_scores(scores)

    scores_mapped = torch.from_numpy(
        np.array(list(get_flipped_scores(scores).values()))
    )
    return ref_nodes, ref_nodes_mapped, scores, scores_mapped


def get_percent_of_nodes_from_scores_dict(scores_dict, percent):
    if percent < 0 or percent > 1:
        raise ValueError("Wrong percent value")
    n = math.ceil(len(scores_dict) * percent)
    reference_nodes = sorted(scores_dict.keys(), key=scores_dict.get)
    reference_nodes = reference_nodes[: int(n)]
    reference_nodes = [str(node) for node in reference_nodes]
    return reference_nodes


SELECTION_METHODS = {
    "percent": get_percent_of_nodes_from_scores_dict,
}


def get_node_mapping(nodes, common_nodes):
    mapping = {}
    for node in common_nodes:
        mapping[node] = nodes.index(node)

    return mapping


def get_selection_method_from_str(selection_method: str) -> Callable:
    if selection_method in SELECTION_METHODS.keys():
        method = SELECTION_METHODS[selection_method]
    else:
        raise ValueError(
            f"Selection method {selection_method} not implemented. "
            f"Try {SELECTION_METHODS}"
        )
    return method


class AbstractNodesSelector(ABC):
    def __init__(
        self,
        selection_method: Optional[str] = None,
        selection_method_args: Optional[dict] = None,
    ):
        self.selection_method = (
            get_selection_method_from_str(selection_method)
            if selection_method
            else None
        )
        self.selection_method_args = selection_method_args

    @abstractmethod
    def get_scores_from_graph(self, g_u, g_v):
        pass

    @abstractmethod
    def get_scores_from_activity(self, a_u, a_v):
        pass

    def select(self, g_u, g_v, cache=None):
        if not self.selection_method:
            raise ValueError(
                "Node selection function undefined. "
                "Pass selection method name."
            )
        if cache:
            a_u, a_v = cache
            scores_dict = self.get_scores_from_activity(a_u=a_u, a_v=a_v)
        else:
            scores_dict = self.get_scores_from_graph(g_u=g_u, g_v=g_v)
        ref_nodes = self.selection_method(
            scores_dict=scores_dict, **self.selection_method_args
        )
        return ref_nodes, scores_dict


class EdgeJaccardNodesSelector(AbstractNodesSelector):
    def __init__(
        self,
        selection_method: Optional[str] = None,
        selection_method_args: Optional[dict] = None,
    ):
        super(EdgeJaccardNodesSelector, self).__init__(
            selection_method, selection_method_args
        )

    def get_scores_from_graph(self, g_u, g_v):
        common_nodes = get_common_nodes(g_u, g_v)

        scores_dict = {}
        for node in common_nodes:
            node_gu_neighbors = set(g_u.neighbors(int(node)))
            node_gv_neighbots = set(g_v.neighbors(int(node)))
            common_neighbors = len(
                node_gu_neighbors.intersection(node_gv_neighbots)
            )
            denominator_neighbors = len(
                node_gu_neighbors.union(node_gv_neighbots)
            )
            if denominator_neighbors != 0:
                scores_dict[node] = 1 - (
                    common_neighbors / denominator_neighbors
                )
            else:
                scores_dict[node] = 1

        return scores_dict

    def get_scores_from_activity(self, a_u, a_v):
        raise NotImplementedError("Use get_scores_from_graph fn instead!")


class FILDNEPercentSelector(AbstractNodesSelector):
    def __init__(
        self,
        selection_method: Optional[str] = None,
        selection_method_args: Optional[dict] = None,
    ):
        super().__init__(selection_method, selection_method_args)

    @staticmethod
    def scoring_fn(a_u: np.ndarray, a_v: np.ndarray) -> np.ndarray:
        return np.abs(a_u - a_v) * (
            np.pi / 2 - np.arctan(np.maximum(a_u - a_v, a_v))
        )

    @staticmethod
    def get_activity(g_u: nx.Graph, nodes: List[int]) -> np.ndarray:
        degrees = []
        for node in nodes:
            degrees.append(len(g_u.edges(node)))
        return np.array(degrees)

    def get_scores_from_graph(self, g_u, g_v):
        common_nodes: List[str] = get_common_nodes(g_u, g_v)
        common_nodes_int: List[int] = [int(it) for it in common_nodes]
        a_u = self.get_activity(g_u, common_nodes_int)
        a_v = self.get_activity(g_v, common_nodes_int)

        scores = self.scoring_fn(a_u, a_v)
        scores_dict = dict(zip(common_nodes, scores))
        return scores_dict

    def get_scores_from_activity(self, a_u, a_v):
        raise NotImplementedError("Use get_scores_from_graph fn instead!")


class TemporalCentralityMeasureSelector(AbstractNodesSelector):
    def __init__(
        self,
        selection_method: Optional[str] = None,
        selection_method_args: Optional[dict] = None,
    ) -> None:
        super().__init__(selection_method, selection_method_args)

    def get_scores_from_graph(self, g_u, g_v):
        raise NotImplementedError("Use get_scores_from_activity fn instead!")

    def get_scores_from_activity(
        self,
        a_u: Dict[int, float],
        a_v: Dict[int, float],
    ) -> Dict[int, float]:
        return {node: abs(a_u[node] - a_v[node]) for node in a_u.keys()}
