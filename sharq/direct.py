import torch

from sharq.candidates import deduped_representatives, filter_by_zero_policy
from sharq.constants import CLIP_GRID
from sharq.scoring import SelectionResult


def _expanded_group_scales(W, codebook, clip, group_size):
    if group_size == -1:
        group_size = W.shape[-1]
    if W.shape[-1] % group_size != 0:
        raise ValueError("SHARQ group_size must divide the input dimension.")
    n_groups = W.shape[-1] // group_size
    grouped = W.abs().view(*W.shape[:-1], n_groups, group_size)
    gmax = grouped.amax(dim=-1)
    scale = gmax.mul(float(clip) / float(max(codebook)))
    return scale.repeat_interleave(group_size, dim=-1)


def _quantize_direct(W, codebook, clip, group_size):
    scale = _expanded_group_scales(W, codebook, clip, group_size)
    safe_scale = torch.clamp(scale, min=torch.finfo(W.dtype).tiny)
    t = W / safe_scale
    levels = torch.tensor(
        sorted(set([-int(z) for z in codebook] + [int(z) for z in codebook])),
        dtype=W.dtype,
        device=W.device,
    )
    idx_right = torch.searchsorted(levels.contiguous(), t.contiguous()).clamp(max=levels.numel() - 1)
    idx_left = (idx_right - 1).clamp(min=0)
    left = levels[idx_left]
    right = levels[idx_right]
    idx = torch.where((t - left).abs() <= (right - t).abs(), idx_left, idx_right)
    q = safe_scale * levels[idx]
    return torch.where(scale == 0, torch.zeros_like(q), q)


def _hessian_score(err, H_col):
    if len(H_col.shape) == 2:
        H_col = H_col.unsqueeze(0)
    return torch.sum((err @ H_col) * err).double()


def score_candidate_direct(W, H_col, codebook, clip, group_size, objective="hessian"):
    Q = _quantize_direct(W, codebook, clip, group_size)
    err = Q - W
    if objective == "hessian":
        return _hessian_score(err, H_col)
    if objective == "mse":
        return torch.sum(err.pow(2)).double()
    raise ValueError(f"Unsupported direct SHARQ objective: {objective}")


def select_direct(W, H_col, bits, group_size=-1, zero_policy="free", clip_grid=None, objective="hessian"):
    candidates = filter_by_zero_policy(deduped_representatives(bits), zero_policy)
    if not candidates:
        raise ValueError(f"No SHARQ candidates remain for zero_policy={zero_policy}")
    if clip_grid is None:
        clip_grid = CLIP_GRID[bits]
    clip_grid = clip_grid.to(device=W.device, dtype=W.dtype)

    best = None
    best_zero_score = float("inf")
    best_no_zero_score = float("inf")
    for candidate in candidates:
        candidate_best = float("inf")
        candidate_best_clip = None
        for clip in clip_grid:
            score = float(score_candidate_direct(W, H_col, candidate, clip, group_size, objective).item())
            if score < candidate_best:
                candidate_best = score
                candidate_best_clip = float(clip.item())
        if 0 in candidate:
            best_zero_score = min(best_zero_score, candidate_best)
        else:
            best_no_zero_score = min(best_no_zero_score, candidate_best)
        if best is None or candidate_best < best.score:
            best = SelectionResult(
                codebook=tuple(int(z) for z in candidate),
                clip=candidate_best_clip,
                score=candidate_best,
                best_zero_score=best_zero_score,
                best_no_zero_score=best_no_zero_score,
                selector="direct",
            )

    best.best_zero_score = best_zero_score
    best.best_no_zero_score = best_no_zero_score
    return best
