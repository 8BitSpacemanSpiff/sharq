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


def _score_row(err, H):
    return torch.sum((err @ H) * err).double()


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


def select_direct_channelwise(W, H_col, bits, group_size=-1, zero_policy="free", clip_grid=None):
    candidates = filter_by_zero_policy(deduped_representatives(bits), zero_policy)
    if not candidates:
        raise ValueError(f"No SHARQ candidates remain for zero_policy={zero_policy}")
    if clip_grid is None:
        clip_grid = CLIP_GRID[bits]
    clip_grid = clip_grid.to(device=W.device, dtype=W.dtype)
    if len(H_col.shape) == 2:
        H_col = H_col.unsqueeze(0)

    n_heads, n_rows, _ = W.shape
    channel_codebooks = []
    channel_clips = torch.empty((n_heads, n_rows), dtype=torch.float32, device=W.device)
    total_score = 0.0
    best_zero_score = 0.0
    best_no_zero_score = 0.0

    for h in range(n_heads):
        head_codebooks = []
        H = H_col[h]
        for r in range(n_rows):
            row = W[h:h + 1, r:r + 1, :]
            row_best = None
            row_best_zero = float("inf")
            row_best_no_zero = float("inf")
            for candidate in candidates:
                candidate_best = float("inf")
                candidate_best_clip = None
                for clip in clip_grid:
                    q = _quantize_direct(row, candidate, clip, group_size)
                    score = float(_score_row(q.reshape(1, -1) - row.reshape(1, -1), H).item())
                    if score < candidate_best:
                        candidate_best = score
                        candidate_best_clip = float(clip.item())
                if 0 in candidate:
                    row_best_zero = min(row_best_zero, candidate_best)
                else:
                    row_best_no_zero = min(row_best_no_zero, candidate_best)
                if row_best is None or candidate_best < row_best.score:
                    row_best = SelectionResult(
                        codebook=tuple(int(z) for z in candidate),
                        clip=candidate_best_clip,
                        score=candidate_best,
                        best_zero_score=row_best_zero,
                        best_no_zero_score=row_best_no_zero,
                        selector="direct",
                        codebook_granularity="channel",
                    )
            head_codebooks.append(row_best.codebook)
            channel_clips[h, r] = row_best.clip
            total_score += row_best.score
            best_zero_score += row_best_zero
            best_no_zero_score += row_best_no_zero
        channel_codebooks.append(head_codebooks)

    return SelectionResult(
        codebook=tuple(),
        clip=float(channel_clips.float().mean().item()),
        score=float(total_score),
        best_zero_score=float(best_zero_score),
        best_no_zero_score=float(best_no_zero_score),
        selector="direct",
        codebook_granularity="channel",
        channel_codebooks=channel_codebooks,
        channel_clips=channel_clips.cpu(),
    )
