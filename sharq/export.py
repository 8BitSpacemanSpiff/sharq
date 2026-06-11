from sharq.constants import DECOMP


def enable_table(codebook):
    return [list(DECOMP[int(z)]) for z in codebook]


def hardware_record(module, bits, codebook, clip):
    return {
        "module": module,
        "bits": int(bits),
        "codebook_z": [int(z) for z in codebook],
        "clip": float(clip),
        "enable_table": enable_table(codebook),
    }

