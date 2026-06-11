from dataclasses import dataclass

import torch

from sharq.candidates import deduped_representatives, filter_by_zero_policy
from sharq.constants import CLIP_GRID


@dataclass
class SelectionResult:
    codebook: tuple
    clip: float
    score: float
    best_zero_score: float
    best_no_zero_score: float
    selector: str = "histogram"
    codebook_granularity: str = "module"
    channel_codebooks: list | None = None
    channel_clips: object | None = None


def score_all(hist, bin_centers, candidates, clip_grid):
    hist = hist.to(dtype=torch.float64)
    centers = bin_centers.to(dtype=torch.float64)
    clips = clip_grid.to(dtype=torch.float64, device=centers.device)
    scores = torch.empty((len(candidates), clips.numel()), dtype=torch.float64, device=centers.device)

    for i, candidate in enumerate(candidates):
        levels = torch.tensor(candidate, dtype=torch.float64, device=centers.device)
        z_max = levels.max()
        levels_norm = clips[:, None] * levels[None, :] / z_max
        dist = (centers[None, :, None] - levels_norm[:, None, :]).pow(2).amin(dim=-1)
        scores[i] = (dist * hist[None, :]).sum(dim=-1)
    return scores


def select(hist, bin_centers, bits, zero_policy="free", clip_grid=None):
    candidates = filter_by_zero_policy(deduped_representatives(bits), zero_policy)
    if not candidates:
        raise ValueError(f"No SHARQ candidates remain for zero_policy={zero_policy}")

    if clip_grid is None:
        clip_grid = CLIP_GRID[bits]
    clip_grid = clip_grid.to(bin_centers.device)
    scores = score_all(hist, bin_centers, candidates, clip_grid)
    flat_idx = int(torch.argmin(scores).item())
    cand_idx = flat_idx // scores.shape[1]
    clip_idx = flat_idx % scores.shape[1]

    zero_scores = []
    no_zero_scores = []
    for i, candidate in enumerate(candidates):
        best = float(scores[i].min().item())
        if 0 in candidate:
            zero_scores.append(best)
        else:
            no_zero_scores.append(best)

    return SelectionResult(
        codebook=tuple(int(z) for z in candidates[cand_idx]),
        clip=float(clip_grid[clip_idx].item()),
        score=float(scores[cand_idx, clip_idx].item()),
        best_zero_score=min(zero_scores) if zero_scores else float("inf"),
        best_no_zero_score=min(no_zero_scores) if no_zero_scores else float("inf"),
        selector="histogram",
    )
