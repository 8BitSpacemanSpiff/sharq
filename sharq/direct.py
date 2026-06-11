import torch

from sharq.candidates import deduped_representatives, filter_by_zero_policy
from sharq.constants import CLIP_GRID
from sharq.scoring import SelectionResult
from utils.quant_utils import fake_quantize


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


def _quantize_direct_many_clips(W, codebook, clip_grid, group_size):
    if group_size == -1:
        group_size = W.shape[-1]
    if W.shape[-1] % group_size != 0:
        raise ValueError("SHARQ group_size must divide the input dimension.")
    n_groups = W.shape[-1] // group_size
    grouped = W.abs().view(*W.shape[:-1], n_groups, group_size)
    gmax = grouped.amax(dim=-1)
    base_scale = gmax / float(max(codebook))
    scale = clip_grid[:, None, None, None] * base_scale[None, ...]
    scale = scale.repeat_interleave(group_size, dim=-1)
    safe_scale = torch.clamp(scale, min=torch.finfo(W.dtype).tiny)
    t = W.unsqueeze(0) / safe_scale
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


def _score_rows_many_clips(err, H_col):
    return torch.einsum("chri,hij,chrj->chr", err, H_col, err).double()


def _quantize_with_scale(W, codebook, scale):
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


def select_uniform_scale_channelwise(W, H_col, scale, zero, maxq, bits, zero_policy="free", topk=None):
    candidates = filter_by_zero_policy(deduped_representatives(bits), zero_policy)
    if not candidates:
        raise ValueError(f"No SHARQ candidates remain for zero_policy={zero_policy}")
    if len(H_col.shape) == 2:
        H_col = H_col.unsqueeze(0)

    n_heads, n_rows, n_cols = W.shape
    scale_full = scale.expand(*W.shape[:-1], 1).expand_as(W)
    uniform_q = fake_quantize(W, scale, zero, maxq)
    uniform_scores = _score_rows_many_clips((uniform_q - W).unsqueeze(0), H_col).squeeze(0)

    best_scores = uniform_scores.clone()
    best_candidate_idx = torch.full((n_heads, n_rows), -1, dtype=torch.long, device=W.device)
    best_zero_scores = torch.full_like(best_scores, float("inf"))
    best_no_zero_scores = torch.full_like(best_scores, float("inf"))

    candidate_totals = []
    for candidate_idx, candidate in enumerate(candidates):
        q = _quantize_with_scale(W, candidate, scale_full)
        scores = _score_rows_many_clips((q - W).unsqueeze(0), H_col).squeeze(0)
        candidate_totals.append((float(scores.sum().item()), candidate))
        if 0 in candidate:
            best_zero_scores = torch.minimum(best_zero_scores, scores)
        else:
            best_no_zero_scores = torch.minimum(best_no_zero_scores, scores)
        update = scores < best_scores
        best_scores = torch.where(update, scores, best_scores)
        best_candidate_idx = torch.where(
            update,
            torch.full_like(best_candidate_idx, candidate_idx),
            best_candidate_idx,
        )
        del q, scores

    candidate_totals.sort(key=lambda item: item[0])
    if topk is not None and topk > 0:
        candidates = [candidate for _, candidate in candidate_totals[:topk]]

    best_candidate_idx_cpu = best_candidate_idx.cpu()
    channel_codebooks = []
    improved = int((best_candidate_idx >= 0).sum().item())
    for h in range(n_heads):
        head_codebooks = []
        for r in range(n_rows):
            idx = int(best_candidate_idx_cpu[h, r].item())
            if idx >= 0:
                head_codebooks.append(tuple(int(z) for z in candidates[idx]))
            else:
                head_codebooks.append(tuple())
        channel_codebooks.append(head_codebooks)

    print(
        ">>> SHARQ/uniform_scale: "
        f"improved_channels={improved}/{n_heads * n_rows}, "
        f"uniform_score={float(uniform_scores.sum().item()):.6e}, "
        f"best_score={float(best_scores.sum().item()):.6e}"
    )

    return SelectionResult(
        codebook=tuple(),
        clip=1.0,
        score=float(best_scores.sum().item()),
        best_zero_score=float(best_zero_scores.sum().item()),
        best_no_zero_score=float(best_no_zero_scores.sum().item()),
        selector="uniform_scale",
        codebook_granularity="channel",
        channel_codebooks=channel_codebooks,
        channel_clips=torch.ones((n_heads, n_rows), dtype=torch.float32),
        channel_uniform_score=uniform_scores.cpu(),
        shortlisted_codebooks=[tuple(int(z) for z in candidate) for candidate in candidates],
    )


def make_online_uniform_scale_selection(bits, shortlisted_codebooks=None):
    return SelectionResult(
        codebook=tuple(),
        clip=1.0,
        score=0.0,
        best_zero_score=0.0,
        best_no_zero_score=0.0,
        selector="uniform_scale_online",
        codebook_granularity="channel",
        online_uniform_scale=True,
        shortlisted_codebooks=shortlisted_codebooks,
    )


def select_direct(W, H_col, bits, group_size=-1, zero_policy="free", clip_grid=None, objective="hessian"):
    candidates = filter_by_zero_policy(deduped_representatives(bits), zero_policy)
    if not candidates:
        raise ValueError(f"No SHARQ candidates remain for zero_policy={zero_policy}")
    if clip_grid is None:
        clip_grid = CLIP_GRID[bits]
    clip_grid = clip_grid.to(device=W.device, dtype=W.dtype)

    if objective == "hessian":
        return _select_direct_gpu(W, H_col, candidates, group_size, clip_grid)

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


def _select_direct_gpu(W, H_col, candidates, group_size, clip_grid):
    if len(H_col.shape) == 2:
        H_col = H_col.unsqueeze(0)
    best = None
    best_zero_score = float("inf")
    best_no_zero_score = float("inf")

    for candidate in candidates:
        q = _quantize_direct_many_clips(W, candidate, clip_grid, group_size)
        err = q - W.unsqueeze(0)
        scores = _score_rows_many_clips(err, H_col).sum(dim=(1, 2))
        candidate_best_score, clip_idx = torch.min(scores, dim=0)
        candidate_best = float(candidate_best_score.item())
        candidate_best_clip = float(clip_grid[int(clip_idx.item())].item())
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
                codebook_granularity="module",
            )
        del q, err, scores

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
    best_scores = torch.full((n_heads, n_rows), float("inf"), dtype=torch.float64, device=W.device)
    best_clips = torch.empty((n_heads, n_rows), dtype=torch.float32, device=W.device)
    best_candidate_idx = torch.full((n_heads, n_rows), -1, dtype=torch.long, device=W.device)
    best_zero_scores = torch.full_like(best_scores, float("inf"))
    best_no_zero_scores = torch.full_like(best_scores, float("inf"))

    for candidate_idx, candidate in enumerate(candidates):
        q = _quantize_direct_many_clips(W, candidate, clip_grid, group_size)
        err = q - W.unsqueeze(0)
        scores = _score_rows_many_clips(err, H_col)
        candidate_best_scores, clip_idx = torch.min(scores, dim=0)
        if 0 in candidate:
            best_zero_scores = torch.minimum(best_zero_scores, candidate_best_scores)
        else:
            best_no_zero_scores = torch.minimum(best_no_zero_scores, candidate_best_scores)
        update = candidate_best_scores < best_scores
        best_scores = torch.where(update, candidate_best_scores, best_scores)
        best_clips = torch.where(update, clip_grid[clip_idx].float(), best_clips)
        best_candidate_idx = torch.where(
            update,
            torch.full_like(best_candidate_idx, candidate_idx),
            best_candidate_idx,
        )
        del q, err, scores

    best_candidate_idx_cpu = best_candidate_idx.cpu()
    channel_codebooks = []
    for h in range(n_heads):
        head_codebooks = []
        for r in range(n_rows):
            head_codebooks.append(tuple(int(z) for z in candidates[int(best_candidate_idx_cpu[h, r].item())]))
        channel_codebooks.append(head_codebooks)

    return SelectionResult(
        codebook=tuple(),
        clip=float(best_clips.float().mean().item()),
        score=float(best_scores.sum().item()),
        best_zero_score=float(best_zero_scores.sum().item()),
        best_no_zero_score=float(best_no_zero_scores.sum().item()),
        selector="direct",
        codebook_granularity="channel",
        channel_codebooks=channel_codebooks,
        channel_clips=best_clips.cpu(),
    )
