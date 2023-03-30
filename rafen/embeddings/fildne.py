"""Refactored from FILDNE https://gitlab.com/fildne/fildne/"""
import dataclasses
from abc import abstractmethod
from collections import defaultdict
from typing import Any, List, Literal

import networkx as nx
import numpy as np
from numpy import typing as np_typing

from rafen.embeddings.keyedmodel import KeyedModel
from rafen.tasks.linkprediction.dataset import LinkPredictionDataset
from rafen.tasks.linkprediction.models import LogisticRegressionModel


def get_correct_edge_predictions(
    embeddings: List[KeyedModel], snapshot: nx.Graph
):
    # Create dataset for learning logistic regression
    common_nodes = set(embeddings[0].nodes)
    for emb in embeddings[1:]:
        common_nodes = common_nodes.intersection(set(emb.nodes))

    common_nodes = list(map(int, common_nodes))

    dataset = LinkPredictionDataset()
    dataset.mk_link_prediction_dataset(
        graph=snapshot,
        split_proportion=(0.8, 0.1, 0.1),
        prev_nodes=common_nodes,
    ).merge_train_subset_with_dev()

    x_train, y_train = dataset.x_train, dataset.y_train
    x_test, y_test = dataset.x_test, dataset.y_test

    # Extract true edges
    true_edges = []

    for x, y in zip(x_train, y_train):
        if y[0] == 1:
            true_edges.append(x)

    for x, y in zip(x_test, y_test):
        if y[0] == 1:
            true_edges.append(x)

    true_edges = np.array(true_edges)

    # Prepare validation embedding
    vms = [LogisticRegressionModel(emb) for emb in embeddings]

    for vm in vms:
        vm.fit(x_train, y_train)

    # Count correct predictions
    correct_predictions = [0] * len(embeddings)

    preds = [vm.predict(true_edges) for vm in vms]
    # Zip together outcomes from different embedding for one given edge
    preds = list(zip(*preds))

    for pred in preds:
        # More than one embedding "predicted" the edge correctly
        if sum(list(pred)) > 1:
            idx = np.random.choice([i for i, p in enumerate(pred) if p == 1])
            correct_predictions[idx] += 1
        # Only one embedding "predicted" the edge correctly
        elif sum(list(pred)) == 1:
            idx = pred.index(1)
            correct_predictions[idx] += 1

    return correct_predictions


class DirichletMultinomialModel:
    def __init__(self, alpha, N, dim):
        self._alpha = np.array(alpha)
        self._N = np.array(N)
        self._dim = dim

    def map(self):
        N = sum(self._N)
        alpha0 = sum(self._alpha)
        _map = (self._N + self._alpha - 1) / (N + alpha0 - self._dim)
        return _map

    def mle(self):
        _mle = self._N / sum(self._N)
        return _mle


def norm(s):
    if not np.any(s):  # Contains only zeros
        return s
    return s / sum(s)


@dataclasses.dataclass
class AbstractFILDNE:
    parameters: List[float] = dataclasses.field(init=False)

    @abstractmethod
    def _combine_vectors(self, vectors: np_typing.NDArray[Any]) -> List[float]:
        pass

    def predict(self, embeddings: List[KeyedModel]):
        node_vectors = defaultdict(lambda: [None] * len(embeddings))
        for idx, emb in enumerate(embeddings):
            for node in emb.nodes:
                node_vectors[node][idx] = emb.get_vector(node)

        # Construct new embedding
        emb_predicted = KeyedModel(size=embeddings[0].emb_dim)

        for node, vectors in node_vectors.items():
            emb_predicted.add_node(node, self._combine_vectors(vectors))

        return emb_predicted


@dataclasses.dataclass
class FILDNE(AbstractFILDNE):
    alpha: float

    def __post_init__(self):
        self.parameters = [self.alpha, 1 - self.alpha]

    def _combine_vectors(self, vectors: np_typing.NDArray[Any]) -> List[float]:
        vectors = [vec for vec in vectors if vec is not None]

        if len(vectors) == 1:
            return list(vectors[0])

        combined_vector = vectors[0]
        alpha = self.parameters

        for vec in vectors[1:]:
            combined_vector = alpha[0] * combined_vector + alpha[1] * vec

        return list(combined_vector)


@dataclasses.dataclass
class kFILDNE(AbstractFILDNE):
    prior_distribution: Literal["uniform", "increase"]

    def __post_init__(self):
        self.parameters = []

    def fit(self, embeddings, last_snapshot):
        corr_pred = get_correct_edge_predictions(embeddings, last_snapshot)

        if self.prior_distribution == "uniform":
            alpha = [1.0] * len(embeddings)
        elif self.prior_distribution == "increase":
            alpha = list(range(1, len(embeddings) + 1))
        else:
            raise RuntimeError(
                "Unknown prior strategy:", self.prior_distribution
            )

        # Get parameters using Dirichlet
        dirichlet = DirichletMultinomialModel(
            alpha=alpha,
            N=corr_pred,
            dim=len(embeddings),
        )

        self.parameters = dirichlet.map()
        return self

    def _combine_vectors(self, vectors):
        pv = [(p, v) for p, v in zip(self.parameters, vectors) if v is not None]

        if len(pv) == 0:
            raise RuntimeError("combine_vectors() called with no vectors!")

        parameters, vectors = zip(*pv)

        if len(vectors) == 1:
            return list(vectors[0])

        parameters = norm(np.array(parameters))
        vectors = np.array(vectors)

        weighed_vectors = (vectors.T * parameters).T
        combined_vector = np.sum(weighed_vectors, axis=0)

        return list(combined_vector)
