import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def approximate_flops(num_tokens: int, embed_dim: int, num_layers: int):
    return int(num_layers * num_tokens * embed_dim * embed_dim * 2)
