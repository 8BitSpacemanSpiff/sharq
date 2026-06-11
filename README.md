# SHARQ

This repository maintains SHARQ (SHift-Add Routed Quantization) on top of the
SamsungLabs BoA post-training quantization codebase.

SHARQ adds a hardware-legal non-uniform codebook pre-pass and changes BoA's
rounding target from a uniform integer grid to the selected shift-add codebook.
The BoA compensation logic, Hessian handling, and block traversal remain the
upstream implementation.

## SHARQ usage

Uniform BoA remains the default:

```bash
python main.py --llm_path <model> --w_bits 4 --codebook uniform
```

Run SHARQ codebook selection and BoA quantization:

```bash
python main.py \
  --llm_path <model> \
  --w_bits 4 \
  --codebook sharq \
  --sharq_selector direct \
  --sharq_group_size 128 \
  --sharq_hist_bins 1024 \
  --sharq_out outputs/sharq-w4
```

`--sharq_out` writes:

- `selection.json`: selected codebook, clip, score, and zero/no-zero best scores
  per module.
- `hw_meta.json`: hardware routing metadata, including the enable table derived
  from the legal shift-add decomposition.

Current implementation notes:

- Supported SHARQ bitwidths are W4, W3, and W2.
- Legal magnitude levels are `{0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12}`.
- Zero-inclusive and no-zero codebooks compete in the same exhaustive search.
- SHARQ currently disables `--act_order_col` because column reordering needs an
  additional permutation-aware mapping for group-wise input scales.
- The core SHARQ modules live under `sharq/`; BoA is patched only at the
  quantization target boundary in `quantizers/boa.py`.
- `--sharq_selector direct` is the accuracy-first selector. It scores each
  candidate codebook on the actual BoA-preprocessed weights with BoA's Hessian
  objective. `--sharq_selector histogram` keeps the earlier fast proxy for
  ablations and optimization work.

## Validation

The repository includes focused tests under `tests/` for candidate enumeration,
hardware decomposition, nearest-codebook quantization, and histogram scoring.

```bash
python -m pytest tests
```

Baseline parity should be checked before reporting SHARQ numbers:

```bash
python main.py --llm_path <tiny-or-test-model> --w_bits 4 --codebook uniform
```

Then compare against:

```bash
python main.py --llm_path <same-model> --w_bits 4 --codebook sharq --sharq_out outputs/check
```

The original BoA README follows for upstream context.

# BoA
This repository contains the code for the ICML 2025 paper [**BoA: Attention-Aware Post-Training Quantization without Backpropagation**](https://arxiv.org/abs/2406.13474). 

The current release includes the following features:
  - Implementation of the proposed BoA: `boa.py`
  - Quantization of OPT, Llama, Llama2, Llama3, Qwen2.5, Qwen3 models: `main.py`
  - Evaluating the perplexity and 0-shot accuracy (8 tasks) of quantized models

## Dependencies
 - see `requirements.txt`

## BoA options
 - `block_v`: whether to apply block-wise objective for the value projection layer. In memory-limited cases, we can significantly reduce memory by de-activating this option, but at the expense of a slight performance degradation.
 - `act_order_col`: whether to re-order columns before the quantization based on the column-wise Hessian $\mathbf{H}_{col}$ (GPTQ heuristic)
 - `act_order_row`: whether to re-order rows before the quantization based on the row-wise Hessian $\mathbf{H}_{row}$
 - `qparam_comput`: how to select quantization grids. Grids can be determined with a naive MinMax or to minimize the weight perturbation (MMSE) or the layer-wise reconstruction error (Hessian)

For more details on other arguments, please refer to [process_args.py](utils/process_args.py).

## Experimental Results
 - Setup
    - NVIDIA H100 GPU has been used.
    - `block_v` option has been activated.
    - `qparam_comput` option has been set to `Hessian`.
    - Test all cases for `act_order_row` and `act_order_col` and report the best results with respect to Wiki2 PPL.

### Results on Qwen2.5 Models
 - INT2 weight-only quantization
   
    | Size | `act_order_row` | `act_order_col` | Wiki2 ($\downarrow$) | C4-new ($\downarrow$) | 0-shot ($\uparrow$) |
    | - | - | - | - | - | - |
    | 0.5B | O | O | 144.7 | 455.8 | 32.56 |
    | 1.5B | O | O | 58.09 | 235.7 | 36.93 |
    | 3B | X | O | 26.55 | 90.77 | 43.28 |
    | 7B | X | O | 23.14 | 103.4 | 43.79 |
    | 14B | O | O | 12.05 | 37.64 | 57.4 |

 - INT3 weight-only quantization

    | Size | `act_order_row` | `act_order_col` | Wiki2 ($\downarrow$) | C4-new ($\downarrow$) | 0-shot ($\uparrow$) |
    | - | - | - | - | - | - |
    | 0.5B | O | O | 20.12 | 40.14 | 45.68 |
    | 1.5B | X | O | 12.03 | 23.52 | 56.90 |
    | 3B | O | O | 9.541 | 17.32 | 60.29 |
    | 7B | O | O | 9.054 | 18.19 | 66.12 |
    | 14B | O | O | 6.492 | 12.20 | 71.80 |

 - Quantization processing time

    | Size | Time (min) |
    | - | - |
    | 0.5B | 6.145 |
    | 1.5B | 22.81|
    | 3B | 39.59 |
    | 7B | 63.14 |
    | 14B | 150.9 |

### Results on Qwen3 Models
 - INT2 weight-only quantization

    | Size | `act_order_row` | `act_order_col` | Wiki2 ($\downarrow$) | C4-new ($\downarrow$) | 0-shot ($\uparrow$) |
    | - | - | - | - | - | - |
    | 4B | X | O | 78.57 | 302.8 | 35.13 |
    | 8B | X | O | 32.53 | 96.43 | 38.47 |
    | 14B | X | O | 24.29 | 75.76 | 42.41 |

 - INT3 weight-only quantization

    | Size | `act_order_row` | `act_order_col` | Wiki2 ($\downarrow$) | C4-new ($\downarrow$) | 0-shot ($\uparrow$) |
    | - | - | - | - | - | - |
    | 4B | X | O | 28.19 | 48.24 | 45.25 |
    | 8B | X | O | 15.62 | 29.60 | 53.69 |
    | 14B | X | O | 10.62 | 18.61 | 66.69 |

 - Quantization processing time

    | Size | Time (min) |
    | - | - |
    | 4B | 43.10 |
    | 8B | 83.77 |
    | 14B | 132.6 |

## License
This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/) (CC BY-NC).

## Citation
If you find this work is useful for your research, please cite our paper:
```bash
@inproceedings{kimboa,
  title={BoA: Attention-aware Post-training Quantization without Backpropagation},
  author={Kim, Junhan and Kim, Ho-young and Cho, Eulrang and Lee, Chungman and Kim, Joonyoung and Jeon, Yongkweon},
  booktitle={Forty-second International Conference on Machine Learning}
}
```
