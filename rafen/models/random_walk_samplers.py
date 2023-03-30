import abc
import dataclasses
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from tqdm.auto import tqdm


class RandomWalkSampler(abc.ABC):
    @abc.abstractmethod
    def sample(self) -> None:
        pass


@dataclasses.dataclass
class Node2VecRandomWalkSampler(RandomWalkSampler):
    data: Data
    output_dir: Path
    walk_length: int
    nb_walks_per_node: int
    p: float
    q: float
    batch_size: int
    w2v_epochs: int
    context_size: int
    num_negative_samples: int = 1

    def __post_init__(self):
        assert self.output_dir.is_dir()

        self.model = Node2Vec(
            self.data.edge_index,
            embedding_dim=1,
            walk_length=self.walk_length,
            context_size=self.context_size,
            walks_per_node=self.nb_walks_per_node,
            p=self.p,
            q=self.q,
            num_negative_samples=self.num_negative_samples,
            sparse=False,
        )
        num_workers = (
            int(os.getenv("NUM_WORKERS")) if os.getenv("NUM_WORKERS") else 0
        )
        self.loader = self.model.loader(
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )

    def _sample_epoch(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        batches = []
        for batch in tqdm(self.loader, desc="Batch", leave=False):
            batch = (batch[0].type(torch.int32), batch[1].type(torch.int32))
            batches.append(batch)
        return batches

    def sample(self) -> None:
        for epoch in tqdm(range(self.w2v_epochs), desc="Epoch ", leave=False):
            data = self._sample_epoch()
            with open(self.output_dir / f"epoch_{epoch}.pkl", "wb") as f:
                pickle.dump(obj=data, file=f, protocol=pickle.HIGHEST_PROTOCOL)
