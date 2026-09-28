"""Global seeding for reproducible runs (AGENTS.md §7)."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` (all devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    """``DataLoader`` ``worker_init_fn``: derive numpy/random seeds from the torch worker seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
