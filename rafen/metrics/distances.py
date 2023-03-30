import numpy as np
import torch
from sklearn.metrics import pairwise_distances


def calculate_l2_distance(
    ref_embedding: torch.tensor, embedding: torch.tensor
) -> torch.tensor:
    l2_loss = torch.nn.MSELoss()
    return l2_loss(ref_embedding, embedding)
