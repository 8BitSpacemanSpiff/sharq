import torch


LEGAL_LEVELS = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12)
DECOMP = {
    0: (None, None),
    1: (1, None),
    2: (2, None),
    3: (1, 2),
    4: (4, None),
    5: (1, 4),
    6: (2, 4),
    8: (8, None),
    9: (1, 8),
    10: (2, 8),
    12: (4, 8),
}

BITWIDTH_TO_K = {4: 8, 3: 4, 2: 2}
CLIP_GRID = {
    4: torch.linspace(0.85, 1.00, 16),
    3: torch.linspace(0.75, 1.00, 16),
    2: torch.linspace(0.60, 1.00, 20),
}

