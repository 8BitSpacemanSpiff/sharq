# SHARQ

This repository maintains SHARQ (SHift-Add Routed Quantization) on top of the
SamsungLabs BoA post-training quantization codebase.

SHARQ explores hardware-legal non-uniform alternatives to BoA's uniform weight
bins. The current accuracy-first path keeps BoA's original scale/zero grid
search, then compares uniform bins against shortlisted SHARQ codebooks inside
BoA's column-wise quantize-and-compensate loop. This matters because BoA mutates
the remaining weights after every compensated column, so codebook decisions made
before the loop can become stale.

The hardware-legal magnitude family is:

```text
{0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12}
```

Each level is expressible as the sum of at most two shift terms from
`{1, 2, 4, 8}`. The original hardware story is still preserved as a target, but
the present research priority is accuracy first, then hardware constraints, then
runtime optimization.

## SHARQ usage

Uniform BoA remains the default:

```bash
python main.py --llm_path <model> --w_bits 4 --codebook uniform
```

Run SHARQ with the current accuracy-first online selector:

```bash
python main.py \
  --llm_path <model> \
  --w_bits 3 \
  --codebook sharq \
  --sharq_selector uniform_scale \
  --sharq_zero_policy free \
  --sharq_topk_candidates 1 \
  --sharq_out outputs/sharq-w3-online-top1
```

`--sharq_out` writes:

- `selection.json`: selected codebook, clip, score, and zero/no-zero best scores
  per module.
- `hw_meta.json`: hardware routing metadata, including the enable table derived
  from the legal shift-add decomposition.

## Current Algorithm Flow

For `--sharq_selector uniform_scale`, SHARQ runs as follows:

1. BoA computes Hessian/statistics as usual.
2. BoA runs its original uniform grid search to choose per-channel scale and
   zero-point.
3. SHARQ ranks legal codebooks using that uniform-scale view and keeps the top
   `--sharq_topk_candidates` codebooks as a shortlist.
4. Inside BoA's actual column-wise compensation loop, for the current compensated
   weight column:
   - compute the original uniform quantized value;
   - compute candidate SHARQ values using the same scale;
   - choose the closest value elementwise;
   - feed that quantized value into BoA's normal error compensation update.
5. Evaluation runs exactly as in upstream BoA.

This online choice is slower than uniform quantization, but it avoids the stale
pre-pass problem where a codebook is chosen before BoA has updated the remaining
weights.

Other selectors are still available for ablations:

- `--sharq_selector histogram`: older fast histogram proxy.
- `--sharq_selector direct`: scores full codebooks before the BoA loop.
- `--sharq_selector uniform_scale`: current online accuracy-first path.

Current implementation notes:

- Supported SHARQ bitwidths are W4, W3, and W2.
- Zero-inclusive and no-zero codebooks compete in the same exhaustive search.
- SHARQ currently disables `--act_order_col` because column reordering needs an
  additional permutation-aware mapping for group-wise input scales.
- The core SHARQ modules live under `sharq/`; BoA is patched only at the
  quantization target boundary in `quantizers/boa.py`.

## OPT-125M Smoke Result

These are not paper-quality numbers, but they are useful for checking the method
before moving to Llama 3.2 1B.

Setup: OPT-125M, W3, WikiText-2 calibration, `nsamples=16`, `seqlen=512`, no
`block_v`, no activation reordering.

| Method | WikiText-2 PPL | C4-new PPL | Time |
| - | -: | -: | -: |
| Uniform BoA W3 | 46.298 | 39.968 | 291.153s |
| SHARQ online top-1 | 44.124 | 36.621 | 703.326s |
| SHARQ online top-3 | 44.129 | 35.553 | 1321.036s |

Top-1 is the practical setting. Top-3 gives better C4-new PPL, but is much
slower.

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

## Llama 3.2 1B Accuracy Track

Use this track for meaningful SHARQ comparisons. OPT-125M with tiny calibration
is only a smoke test.

If the Meta checkpoint is gated, authenticate first:

```bash
huggingface-cli login
```

or:

```bash
export HF_TOKEN=<your_token>
```

Full precision reference:

```bash
python main.py \
  --llm_path meta-llama/Llama-3.2-1B \
  --eval_fp \
  --calib_data c4 \
  --nsamples 128 \
  --seqlen 2048
```

Uniform BoA W4 baseline:

```bash
python main.py \
  --llm_path meta-llama/Llama-3.2-1B \
  --w_bits 4 \
  --codebook uniform \
  --calib_data c4 \
  --nsamples 128 \
  --seqlen 2048 \
  --qparam_comput Hessian \
  --block_v
```

Uniform BoA W3 baseline:

```bash
python main.py \
  --llm_path meta-llama/Llama-3.2-1B \
  --w_bits 3 \
  --codebook uniform \
  --calib_data c4 \
  --nsamples 128 \
  --seqlen 2048 \
  --qparam_comput Hessian \
  --block_v
```

SHARQ W3 online top-1:

```bash
python main.py \
  --llm_path meta-llama/Llama-3.2-1B \
  --w_bits 3 \
  --codebook sharq \
  --sharq_selector uniform_scale \
  --sharq_zero_policy free \
  --sharq_topk_candidates 1 \
  --sharq_out outputs/llama32-1b-sharq-w3-online-top1 \
  --calib_data c4 \
  --nsamples 128 \
  --seqlen 2048 \
  --qparam_comput Hessian \
  --block_v
```

SHARQ W3 online top-3:

```bash
python main.py \
  --llm_path meta-llama/Llama-3.2-1B \
  --w_bits 3 \
  --codebook sharq \
  --sharq_selector uniform_scale \
  --sharq_zero_policy free \
  --sharq_topk_candidates 3 \
  --sharq_out outputs/llama32-1b-sharq-w3-online-top3 \
  --calib_data c4 \
  --nsamples 128 \
  --seqlen 2048 \
  --qparam_comput Hessian \
  --block_v
```

If the full setting is too slow for the first Llama run, use an intermediate
calibration size:

```bash
python main.py \
  --llm_path meta-llama/Llama-3.2-1B \
  --w_bits 3 \
  --codebook sharq \
  --sharq_selector uniform_scale \
  --sharq_zero_policy free \
  --sharq_topk_candidates 1 \
  --sharq_out outputs/llama32-1b-sharq-w3-online-top1-smoke \
  --calib_data c4 \
  --nsamples 32 \
  --seqlen 1024 \
  --qparam_comput Hessian \
  --block_v
```

If free zero/no-zero selection is unstable, rerun the same command with:

```bash
--sharq_zero_policy force_zero
```

## Qwen2.5-7B Paper-Scale Track

Use this track for the next W3 experiments. The goal is to first reproduce a
strong uniform BoA baseline on Qwen2.5-7B with paper-scale calibration, then run
the online SHARQ selector against the fair matching uniform configuration.

### Fresh VM Setup

```bash
git clone https://github.com/8BitSpacemanSpiff/sharq.git
cd sharq

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install pytest
```

If Hugging Face rate limits or gated model access become an issue:

```bash
huggingface-cli login
```

or:

```bash
export HF_TOKEN=<your_token>
```

Verify the environment sees the GPU:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

Watch the run from another terminal:

```bash
watch -n 1 nvidia-smi
```

Run the unit tests once:

```bash
python -m pytest tests
```

Create output folders:

```bash
mkdir -p logs outputs
```

### Quick Qwen Sanity Run

Use this before launching the full 128-sample run. It should reach the first
transformer block quickly and confirms the Qwen custom modeling path works on
the instance.

```bash
python main.py \
  --llm_path Qwen/Qwen2.5-7B \
  --w_bits 3 \
  --codebook uniform \
  --calib_data c4 \
  --nsamples 4 \
  --seqlen 512 \
  --qparam_comput Hessian \
  --block_v \
  --act_order_row \
  --act_order_col \
  2>&1 | tee logs/qwen25-7b-uniform-w3-smoke.log
```

### Paper-Style Uniform Baseline

This is the stronger upstream-style BoA W3 baseline. It uses both column and row
activation ordering.

```bash
python main.py \
  --llm_path Qwen/Qwen2.5-7B \
  --w_bits 3 \
  --codebook uniform \
  --calib_data c4 \
  --nsamples 128 \
  --seqlen 2048 \
  --qparam_comput Hessian \
  --block_v \
  --act_order_row \
  --act_order_col \
  2>&1 | tee logs/qwen25-7b-uniform-w3-paper.log
```

### Fair Uniform Baseline For SHARQ

SHARQ currently does not support `--act_order_col`, so this run is the fair
apples-to-apples uniform baseline for SHARQ comparisons.

```bash
python main.py \
  --llm_path Qwen/Qwen2.5-7B \
  --w_bits 3 \
  --codebook uniform \
  --calib_data c4 \
  --nsamples 128 \
  --seqlen 2048 \
  --qparam_comput Hessian \
  --block_v \
  --act_order_row \
  2>&1 | tee logs/qwen25-7b-uniform-w3-fair.log
```

### SHARQ W3 Online Top-1

This is the primary SHARQ run. It keeps BoA's uniform scale search, shortlists
the best legal SHARQ codebook, and chooses between uniform and SHARQ levels
inside the BoA compensation loop.

```bash
python main.py \
  --llm_path Qwen/Qwen2.5-7B \
  --w_bits 3 \
  --codebook sharq \
  --sharq_selector uniform_scale \
  --sharq_zero_policy free \
  --sharq_topk_candidates 1 \
  --sharq_out outputs/qwen25-7b-sharq-w3-online-top1 \
  --calib_data c4 \
  --nsamples 128 \
  --seqlen 2048 \
  --qparam_comput Hessian \
  --block_v \
  --act_order_row \
  2>&1 | tee logs/qwen25-7b-sharq-w3-online-top1.log
```

### Optional SHARQ W3 Online Top-3

Run this only after top-1 completes. It is slower, but can improve PPL by giving
the online selector more legal codebooks to compare.

```bash
python main.py \
  --llm_path Qwen/Qwen2.5-7B \
  --w_bits 3 \
  --codebook sharq \
  --sharq_selector uniform_scale \
  --sharq_zero_policy free \
  --sharq_topk_candidates 3 \
  --sharq_out outputs/qwen25-7b-sharq-w3-online-top3 \
  --calib_data c4 \
  --nsamples 128 \
  --seqlen 2048 \
  --qparam_comput Hessian \
  --block_v \
  --act_order_row \
  2>&1 | tee logs/qwen25-7b-sharq-w3-online-top3.log
```

### Runtime Notes

After `Start quantization`, the code first captures the inputs to the first
transformer block over all calibration samples. For `nsamples=128` and
`seqlen=2048`, this can be quiet for a while before the first layer line:

```text
>>>> Quantizing 1-th Transformer Block.... (1/28)
```

During that stage, `nvidia-smi` should still show a Python process using GPU
memory and nonzero utilization. The implementation uses BoA's simultaneous-head
path once block quantization begins: weights are reshaped to
`[num_heads, head_dim, hidden_size]`, so heads are processed in a batched tensor
rather than one head at a time.

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
