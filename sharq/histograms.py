import torch


def build_histogram(W, h, group_size=128, bins=1024):
    if W.shape[-1] != h.numel():
        raise ValueError("h must have one importance per input feature")
    if group_size == -1:
        group_size = W.shape[-1]
    if W.shape[-1] % group_size != 0:
        raise ValueError("input dimension must be divisible by group_size")

    device = W.device
    hist = torch.zeros(bins, dtype=torch.float64, device=device)
    centers = (torch.arange(bins, dtype=torch.float64, device=device) + 0.5) / bins

    abs_w = W.detach().abs().float().reshape(-1, W.shape[-1])
    h = h.detach().float().to(device)
    n_groups = W.shape[-1] // group_size
    grouped_w = abs_w.view(abs_w.shape[0], n_groups, group_size)
    grouped_h = h.view(n_groups, group_size)

    skipped = 0
    for g in range(n_groups):
        vals = grouped_w[:, g, :]
        gmax = vals.max(dim=-1, keepdim=True).values
        keep = gmax.squeeze(-1) > 0
        skipped += int((~keep).sum().item())
        if not torch.any(keep):
            continue
        norm = (vals[keep] / gmax[keep]).clamp(0, 1)
        idx = torch.clamp((norm * bins).long(), max=bins - 1)
        weights = grouped_h[g].expand_as(norm).double()
        hist.scatter_add_(0, idx.reshape(-1), weights.reshape(-1))

    return hist.cpu(), centers.cpu(), skipped

