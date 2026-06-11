from fractions import Fraction
from itertools import combinations

from sharq.constants import BITWIDTH_TO_K, LEGAL_LEVELS


def enumerate_candidates(bits):
    if bits not in BITWIDTH_TO_K:
        raise ValueError(f"Unsupported SHARQ bitwidth: {bits}")
    return [tuple(c) for c in combinations(LEGAL_LEVELS, BITWIDTH_TO_K[bits])]


def canonicalize(subset):
    nonzero = [z for z in subset if z != 0]
    if not nonzero:
        return tuple(Fraction(0, 1) for _ in subset)
    base = min(nonzero)
    return tuple(Fraction(z, base) for z in subset)


def dedupe(candidates):
    groups = {}
    for subset in candidates:
        key = canonicalize(subset)
        groups.setdefault(key, []).append(tuple(subset))
    return groups


def deduped_representatives(bits):
    reps = []
    for group in dedupe(enumerate_candidates(bits)).values():
        reps.append(min(group, key=lambda c: (max(c), c)))
    return sorted(reps)


def filter_by_zero_policy(candidates, zero_policy):
    if zero_policy == "force_zero":
        return [candidate for candidate in candidates if 0 in candidate]
    if zero_policy == "force_no_zero":
        return [candidate for candidate in candidates if 0 not in candidate]
    if zero_policy == "free":
        return list(candidates)
    raise ValueError(f"Unsupported zero policy: {zero_policy}")
