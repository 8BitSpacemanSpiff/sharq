import torch


def build_signed_levels(codebook, *, device=None, dtype=torch.float32):
    levels = sorted(set([-int(z) for z in codebook] + [int(z) for z in codebook]))
    return torch.tensor(levels, device=device, dtype=dtype)


def quantize_to_codebook(w, scale, signed_levels):
    levels = signed_levels.to(device=w.device, dtype=w.dtype)
    scale = scale.to(device=w.device, dtype=w.dtype)
    zero_scale = scale == 0
    safe_scale = torch.clamp(scale, min=torch.finfo(w.dtype).tiny)
    t = w / safe_scale

    idx_right = torch.searchsorted(levels.contiguous(), t.contiguous()).clamp(max=levels.numel() - 1)
    idx_left = (idx_right - 1).clamp(min=0)
    left = levels[idx_left]
    right = levels[idx_right]
    choose_left = (t - left).abs() <= (right - t).abs()
    idx = torch.where(choose_left, idx_left, idx_right)
    q = safe_scale * levels[idx]
    q = torch.where(zero_scale.expand_as(q), torch.zeros_like(q), q)
    return q, idx


def quantize_to_magnitude_code(w, scale, codebook):
    signed_levels = build_signed_levels(codebook, device=w.device, dtype=w.dtype)
    q, _ = quantize_to_codebook(w, scale, signed_levels)
    z = torch.round((q / torch.clamp(scale, min=torch.finfo(w.dtype).tiny)).abs()).to(torch.long)
    cb = torch.tensor(codebook, device=w.device, dtype=torch.long)
    mag_idx = (z.unsqueeze(-1) == cb).to(torch.long).argmax(dim=-1)
    sign = (q < 0).to(torch.long)
    bits = int(torch.ceil(torch.log2(torch.tensor(len(codebook), dtype=torch.float32))).item()) + 1
    return (sign << (bits - 1)) | mag_idx
