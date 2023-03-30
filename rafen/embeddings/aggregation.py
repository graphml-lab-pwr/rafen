from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np

from rafen.embeddings.fildne import FILDNE, kFILDNE
from rafen.embeddings.keyedmodel import KeyedModel


class EmbeddingAggregation(ABC):
    @abstractmethod
    def predict(self, embeddings: List[KeyedModel]) -> KeyedModel:
        pass


class EmbeddingAverageAggregation(EmbeddingAggregation):
    def predict(
        self,
        embeddings: Union[List[KeyedModel], List[Tuple[KeyedModel, float]]],
    ) -> KeyedModel:
        agg_embs = defaultdict(list)
        if isinstance(embeddings[0], tuple):
            embeddings = [it[0] for it in embeddings]
        for emb in embeddings:
            for n, v in emb.to_dict().items():
                agg_embs[n].append(v)

        agg_embs = {k: np.average(v, axis=0) for k, v in agg_embs.items()}

        return KeyedModel(
            size=embeddings[0].emb_dim,
            node_emb_vectors=agg_embs,
            fill_unknown_nodes=False,
        )


EMBEDDING_AGGREGATION_MODELS = {
    "Average": EmbeddingAverageAggregation,
    "FILDNE": FILDNE,
    "k-FILDNE": kFILDNE,
}
