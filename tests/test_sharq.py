import torch

from sharq.candidates import dedupe, enumerate_candidates
from sharq.constants import DECOMP, LEGAL_LEVELS
from sharq.direct import select_direct, select_direct_channelwise, select_uniform_scale_channelwise
from sharq.histograms import build_histogram
from sharq.quantizer import build_signed_levels, quantize_to_codebook
from sharq.scoring import score_all


def test_candidate_counts():
    assert len(enumerate_candidates(4)) == 165
    assert len(enumerate_candidates(3)) == 330
    assert len(enumerate_candidates(2)) == 55
    assert len(dedupe(enumerate_candidates(2))) == 25


def test_decomp_completeness():
    for z in LEGAL_LEVELS:
        assert z in DECOMP
        terms = [term for term in DECOMP[z] if term is not None]
        assert len(terms) <= 2
        assert all(term in (1, 2, 4, 8) for term in terms)
        assert sum(terms) == z


def test_quantize_to_codebook_matches_bruteforce():
    weights = torch.tensor([-20.0, -7.1, -0.3, 0.0, 0.2, 2.9, 9.1, 20.0])
    scale = torch.full_like(weights, 0.5)
    for codebook in [(0, 1, 2, 4), (1, 3, 6, 12), (1, 2, 3, 4, 5, 6, 8, 12)]:
        levels = build_signed_levels(codebook, dtype=weights.dtype)
        q, idx = quantize_to_codebook(weights, scale, levels)
        brute_idx = torch.argmin((weights[:, None] / scale[:, None] - levels[None, :]).abs(), dim=-1)
        brute = scale * levels[brute_idx]
        assert torch.equal(idx, brute_idx)
        assert torch.allclose(q, brute)


def test_histogram_vs_direct_synthetic():
    torch.manual_seed(0)
    W = torch.randn(16, 128)
    h = torch.rand(128) + 0.1
    hist, centers, skipped = build_histogram(W, h, group_size=128, bins=4096)
    assert skipped == 0

    candidates = [(0, 1, 2, 4, 5, 6, 8, 12), (1, 2, 3, 4, 5, 6, 8, 10)]
    clips = torch.tensor([0.85, 0.95, 1.0])
    hist_scores = score_all(hist, centers, candidates, clips)

    abs_w = W.abs()
    norm = abs_w / abs_w.max(dim=-1, keepdim=True).values
    weights = h.expand_as(norm)

    direct = torch.empty_like(hist_scores)
    for i, candidate in enumerate(candidates):
        levels = torch.tensor(candidate, dtype=norm.dtype)
        zmax = levels.max()
        for j, clip in enumerate(clips):
            levels_norm = clip * levels / zmax
            err = (norm[..., None] - levels_norm).pow(2).amin(dim=-1)
            direct[i, j] = (err * weights).sum()

    rel = (hist_scores - direct).abs() / direct.clamp_min(1e-12)
    assert torch.all(rel < 0.01)


def test_direct_selector_returns_valid_result():
    W = torch.tensor([[[0.0, 0.1, -0.5, 1.0], [0.0, -0.2, 0.4, -0.8]]])
    H = torch.eye(4).unsqueeze(0)
    result = select_direct(W, H, bits=3, group_size=-1, zero_policy="force_zero")
    assert result.selector == "direct"
    assert 0 in result.codebook
    assert len(result.codebook) == 4
    assert 0.75 <= result.clip <= 1.0


def test_direct_channelwise_selector_returns_valid_result():
    W = torch.tensor([[[0.0, 0.1, -0.5, 1.0], [0.0, -0.2, 0.4, -0.8]]])
    H = torch.eye(4).unsqueeze(0)
    result = select_direct_channelwise(W, H, bits=3, group_size=-1, zero_policy="force_zero")
    assert result.selector == "direct"
    assert result.codebook_granularity == "channel"
    assert len(result.channel_codebooks) == 1
    assert len(result.channel_codebooks[0]) == 2
    assert all(0 in codebook for codebook in result.channel_codebooks[0])
    assert result.channel_clips.shape == (1, 2)


def test_uniform_scale_selector_reports_channel_choices():
    W = torch.tensor([[[0.0, 0.1, -0.5, 1.0], [0.0, -0.2, 0.4, -0.8]]])
    H = torch.eye(4).unsqueeze(0)
    scale = torch.ones(1, 2, 1) / 7
    zero = torch.full_like(scale, 4)
    result = select_uniform_scale_channelwise(W, H, scale, zero, torch.tensor(7), bits=3, zero_policy="free")
    assert result.selector == "uniform_scale"
    assert result.codebook_granularity == "channel"
    assert len(result.channel_codebooks[0]) == 2
    assert result.channel_uniform_score.shape == (1, 2)
