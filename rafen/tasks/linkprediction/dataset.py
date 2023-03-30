import itertools
import math
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray
from sklearn import utils as sk_utils
from torch import Tensor
from torch_geometric.utils.negative_sampling import negative_sampling


def split_edges(edges, split_ratio=0.5, data=False):
    left, right = split_list(edges, split_ratio)
    if not data:
        left = [(edge[0], edge[1]) for edge in left]
        right = [(edge[0], edge[1]) for edge in right]
    return left, right


Edge = Tuple[int, int]
EdgeList = List[Edge]
NodeSet = Set[int]
NodeList = List[int]


def split_list(
    input_list: List[Any], split_ratio: float
) -> Tuple[List[Any], List[Any]]:
    margin_item = math.ceil(len(input_list) * split_ratio)

    left = input_list[:margin_item]
    right = input_list[margin_item:]
    return left, right


@dataclass
class LinkPredictionDataset:
    x_train: Optional[NDArray] = field(init=False, default=None)
    y_train: Optional[NDArray] = field(init=False, default=None)
    x_dev: Optional[NDArray] = field(init=False, default=None)
    y_dev: Optional[NDArray] = field(init=False, default=None)
    x_test: Optional[NDArray] = field(init=False, default=None)
    y_test: Optional[NDArray] = field(init=False, default=None)
    node_mapping: Optional[Dict[int, int]] = field(init=False, default=None)

    @staticmethod
    def _filter_edge(edge: Edge, prev_nodes: List[int]) -> bool:
        return edge[0] in prev_nodes and edge[1] in prev_nodes

    @staticmethod
    def _get_edges_and_av_nodes(
        edges: List[Tuple[int, int]],
        nodes: List[int],
        prev_nodes: Optional[List[int]] = None,
    ) -> Tuple[EdgeList, NodeSet]:
        edges = list(edges)
        if prev_nodes:
            edges = [
                it
                for it in edges
                if LinkPredictionDataset._filter_edge(it, prev_nodes)
            ]
            av_nodes = set(nodes).intersection(set(prev_nodes))
        else:
            av_nodes = set(nodes)

        return edges, av_nodes

    @staticmethod
    def _convert_edgelist_to_edgeindex(
        edges: EdgeList, node_mapping: Dict[int, int]
    ) -> Tensor:
        return torch.tensor(
            [[node_mapping[edge[0]], node_mapping[edge[1]]] for edge in edges]
        ).T

    @staticmethod
    def _convert_edgeindex_to_edgelist(
        edge_index: Tensor, node_mapping: Dict[int, int]
    ) -> EdgeList:
        return [
            (node_mapping[edge[0]], node_mapping[edge[1]])
            for edge in edge_index.T.numpy()
        ]

    @property
    def reverse_node_mapping(self) -> Optional[Dict[int, int]]:
        if self.node_mapping:
            return dict(
                zip(self.node_mapping.values(), self.node_mapping.keys())
            )
        return None

    @staticmethod
    def _postprocess(
        pos_edges: EdgeList,
        neg_edges: EdgeList,
    ) -> Tuple[NDArray[Any], NDArray[Any]]:
        x = np.concatenate([pos_edges, neg_edges])
        y = np.concatenate(
            [
                np.ones(shape=(len(pos_edges), 1), dtype=int),
                np.zeros(shape=(len(neg_edges), 1), dtype=int),
            ]
        )
        x, y = sk_utils.shuffle(x, y, random_state=441)
        return x, y

    @staticmethod
    def _filter_neg_edges(
        neg_edges: EdgeList, graph: nx.Graph
    ) -> Tuple[EdgeList, int]:
        rejected_edges_num = 0
        filtered_edges = []
        for edge in neg_edges:
            if graph.has_edge(edge[0], edge[1]):
                rejected_edges_num += 1
                continue

            # For directed graphs, when using Hadamard operator, we MUST NOT
            # include an edge if the reversed one exists in the graph, because it
            # would be noise for the classification model.
            # TODO: if case we change Hadamard operator this must be fixed
            if isinstance(
                graph, (nx.DiGraph, nx.MultiDiGraph)
            ) and graph.has_edge(edge[1], edge[0]):
                rejected_edges_num += 1
                continue

            filtered_edges.append(edge)
        return filtered_edges, rejected_edges_num

    def merge_train_subset_with_dev(self) -> "LinkPredictionDataset":
        ds = deepcopy(self)
        assert ds.x_train is not None
        assert ds.x_dev is not None

        ds.x_train = np.concatenate([ds.x_train, ds.x_dev])
        ds.y_train = np.concatenate([ds.y_train, ds.y_dev])
        ds.x_dev = None
        ds.y_dev = None
        return ds

    def mk_link_prediction_dataset(
        self,
        graph: nx.Graph,
        split_proportion: Tuple[float, float, float],
        prev_nodes: Union[NodeSet, NodeList],
    ) -> "LinkPredictionDataset":
        assert sum(split_proportion) == 1.0

        pos_edges, av_nodes = LinkPredictionDataset._get_edges_and_av_nodes(
            edges=list(graph.edges()),
            nodes=list(graph.nodes()),
            prev_nodes=prev_nodes,
        )
        self.node_mapping = dict(zip(np.arange(len(av_nodes)), av_nodes))

        pos_edge_index = LinkPredictionDataset._convert_edgelist_to_edgeindex(
            edges=pos_edges, node_mapping=self.reverse_node_mapping
        )
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            force_undirected=not graph.is_directed(),
            num_nodes=len(av_nodes),
        )
        neg_edges = LinkPredictionDataset._convert_edgeindex_to_edgelist(
            edge_index=neg_edge_index, node_mapping=self.node_mapping
        )
        (
            filtered_neg_edges,
            rejected_edges_num,
        ) = LinkPredictionDataset._filter_neg_edges(neg_edges, graph=graph)

        if rejected_edges_num > 0:
            neg_edge_index_to_replace = negative_sampling(
                edge_index=pos_edge_index,
                force_undirected=not graph.is_directed(),
                num_neg_samples=rejected_edges_num * 3,
                num_nodes=len(av_nodes),
            )
            correct_neg_edges, _ = LinkPredictionDataset._filter_neg_edges(
                neg_edges=LinkPredictionDataset._convert_edgeindex_to_edgelist(
                    edge_index=neg_edge_index_to_replace,
                    node_mapping=self.node_mapping,
                ),
                graph=graph,
            )
            if len(correct_neg_edges) < rejected_edges_num:
                raise ValueError("No negative edges for replacement found!")
            filtered_neg_edges.extend(correct_neg_edges[:rejected_edges_num])

        train_dev_pos_edges, test_pos_edges = split_list(
            pos_edges, split_ratio=split_proportion[0] + split_proportion[1]
        )
        train_dev_neg_edges, test_neg_edges = split_list(
            filtered_neg_edges,
            split_ratio=split_proportion[0] + split_proportion[1],
        )
        train_pos_edges, dev_pos_edges = split_list(
            train_dev_pos_edges,
            split_ratio=split_proportion[0]
            / (split_proportion[0] + split_proportion[1]),
        )
        train_neg_edges, dev_neg_edges = split_list(
            train_dev_neg_edges,
            split_ratio=split_proportion[0]
            / (split_proportion[0] + split_proportion[1]),
        )

        self.x_train, self.y_train = LinkPredictionDataset._postprocess(
            pos_edges=train_pos_edges, neg_edges=train_neg_edges
        )
        self.x_dev, self.y_dev = LinkPredictionDataset._postprocess(
            pos_edges=dev_pos_edges, neg_edges=dev_neg_edges
        )
        self.x_test, self.y_test = LinkPredictionDataset._postprocess(
            pos_edges=test_pos_edges, neg_edges=test_neg_edges
        )
        return self


class LinkPredictionDatasetWithDev:
    """Legacy version"""

    def __init__(self):
        super(LinkPredictionDatasetWithDev, self).__init__()
        self.x_train = None
        self.y_train = None
        self.x_dev = None
        self.y_dev = None
        self.x_test = None
        self.y_test = None

    @staticmethod
    def _filter_test_edges(edges, test_edges):
        edges = set(edges)
        test_edges = set(test_edges)

        return edges.difference(test_edges)

    @staticmethod
    def _post_process(x_pos, x_neg):
        x_np = np.concatenate((np.array(x_pos), np.array(x_neg)))
        y_pos = np.ones((len(x_pos), 1), dtype=int)
        y_neg = np.zeros((len(x_neg), 1), dtype=int)
        y_np = np.concatenate((y_pos, y_neg))

        x, y = sk_utils.shuffle(x_np, y_np, random_state=0)
        return x, y

    @staticmethod
    def _filter_edge(edge, prev_nodes):
        return edge[0] in prev_nodes and edge[1] in prev_nodes

    def _sample_negative_edges(self, nodes, graph, nb_samples):
        """Samples number of non-existing edges from given graph."""
        u = deepcopy(list(nodes))
        v = deepcopy(list(nodes))
        np.random.shuffle(u)
        np.random.shuffle(v)

        neg_edges = []
        if not nb_samples:
            warnings.warn("nb_samples is equal to 0", RuntimeWarning)
            return []
        for edge in itertools.product(u, v):
            # Do not include existing edges
            if graph.has_edge(edge[0], edge[1]):
                continue

            # For directed graphs, when using Hadamard operator, we MUST NOT
            # include an edge if the reversed one exists in the graph, because it
            # would be noise for the classification model.
            # TODO: if case we change Hadamard operator this must be fixed
            if isinstance(
                graph, (nx.DiGraph, nx.MultiDiGraph)
            ) and graph.has_edge(edge[1], edge[0]):
                continue

            neg_edges.append(edge)
            if len(neg_edges) == nb_samples:
                break

        return neg_edges

    def mk_link_prediction_dataset_with_dev(
        self,
        graph,
        split_proportion,
        lp_ds: LinkPredictionDataset,
        prev_nodes=None,
    ):
        """Returns link prediction dataset from graph."""

        edges, av_nodes = LinkPredictionDataset._get_edges_and_av_nodes(
            edges=graph.edges(), nodes=graph.nodes(), prev_nodes=prev_nodes
        )
        lp_ds_test_edges = set(map(tuple, lp_ds.x_test))
        edges = LinkPredictionDatasetWithDev._filter_test_edges(
            edges=edges, test_edges=lp_ds_test_edges
        )

        train_data, dev_deta = split_edges(list(edges), split_proportion)
        neg_train_data = self._sample_negative_edges(
            nodes=av_nodes, graph=graph, nb_samples=len(train_data)
        )
        if not neg_train_data:
            warnings.warn("There is no neg_train_data", RuntimeWarning)
            return
        neg_dev_data = self._sample_negative_edges(
            nodes=av_nodes, graph=graph, nb_samples=len(dev_deta)
        )
        if not neg_dev_data:
            warnings.warn("There is no neg_test_data", RuntimeWarning)
            return

        x_train, y_train = LinkPredictionDatasetWithDev._post_process(
            train_data, neg_train_data
        )
        x_dev, y_dev = LinkPredictionDatasetWithDev._post_process(
            dev_deta, neg_dev_data
        )

        self.x_train = x_train
        self.y_train = y_train
        self.x_dev = x_dev
        self.y_dev = y_dev
        self.x_test = lp_ds.x_test
        self.y_test = lp_ds.y_test

        return self


def convert_lpds_to_dict(
    ds: LinkPredictionDataset,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    return {"train": (ds.x_train, ds.y_train), "test": (ds.x_test, ds.y_test)}
