"""Reproducibility utilities."""
import random
import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    """Set seeds for random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
