import json
from pathlib import Path

import torch

from sharq.export import hardware_record
from sharq.histograms import build_histogram
from sharq.scoring import select


def hessian_importance(H_col):
    if len(H_col.shape) == 2:
        return torch.diagonal(H_col, dim1=-2, dim2=-1).float()
    return torch.diagonal(H_col, dim1=-2, dim2=-1).float().mean(dim=0)


def select_for_module(weight, H_col, bits, group_size, hist_bins, zero_policy="free", clip_grid=None):
    h = hessian_importance(H_col).to(weight.device)
    hist, centers, skipped = build_histogram(weight.float(), h, group_size=group_size, bins=hist_bins)
    result = select(hist, centers, bits, zero_policy=zero_policy, clip_grid=clip_grid)
    return result, skipped


def write_selection_artifacts(out_dir, records):
    if out_dir is None:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selection = {}
    hw_meta = []
    for name, result, bits in records:
        selection[name] = {
            "bits": int(bits),
            "selector": result.selector,
            "codebook_granularity": result.codebook_granularity,
            "codebook": [int(z) for z in result.codebook],
            "clip": float(result.clip),
            "score": float(result.score),
            "best_zero_score": float(result.best_zero_score),
            "best_no_zero_score": float(result.best_no_zero_score),
        }
        if result.codebook_granularity == "channel":
            if result.channel_codebooks is not None:
                selection[name]["channel_codebooks"] = [
                    [[int(z) for z in codebook] for codebook in head]
                    for head in result.channel_codebooks
                ]
            if result.channel_clips is not None:
                selection[name]["channel_clips"] = result.channel_clips.tolist()
            if result.channel_uniform_score is not None:
                selection[name]["channel_uniform_score"] = result.channel_uniform_score.tolist()
            if result.shortlisted_codebooks is not None:
                selection[name]["shortlisted_codebooks"] = [
                    [int(z) for z in codebook] for codebook in result.shortlisted_codebooks
                ]
        else:
            hw_meta.append(hardware_record(name, bits, result.codebook, result.clip))
    (out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    (out / "hw_meta.json").write_text(json.dumps(hw_meta, indent=2), encoding="utf-8")
