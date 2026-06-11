import argparse
from pathlib import Path

def get_boa_arguments(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument("--cache_dir", type=str, default='cache')
    parser.add_argument("--print_memory_usage", action='store_true')
    
    ## Model
    parser.add_argument("--llm_path", type=str, default='facebook/opt-125m')
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--eval_fp", action='store_true', help='Whether to evaluate the original fp model performance')
    
    ## Calib. Data
    parser.add_argument('--calib_data', type=str, default="wikitext2", choices=["c4", "wikitext2"])
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--seqlen', type=int, default=2048, help='Length of input sequences')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')

    ## Quant. Configs.
    parser.add_argument('--w_bits', type=int, default=2)
    parser.add_argument('--w_sym', action="store_true")
    parser.add_argument('--codebook', type=str, default='uniform', choices=['uniform', 'sharq'])
    parser.add_argument('--sharq_selector', type=str, default='direct', choices=['direct', 'histogram'])
    parser.add_argument('--sharq_group_size', type=int, default=128)
    parser.add_argument('--sharq_hist_bins', type=int, default=1024)
    parser.add_argument('--sharq_out', type=str, default=None)
    parser.add_argument('--sharq_zero_policy', type=str, default='free', choices=['free', 'force_zero', 'force_no_zero'])
    parser.add_argument('--sharq_clip_min', type=float, default=None)
    parser.add_argument('--sharq_clip_max', type=float, default=None)
    parser.add_argument('--sharq_clip_steps', type=int, default=None)
    
    ## BoA Options
    parser.add_argument('--qparam_comput', type=str, default='Hessian', choices=['MinMax', 'MMSE', 'Hessian'], help="How to determine Quant. Params")
    parser.add_argument('--block_v', action="store_true", help="Whether to apply block-wise objective for the value projection. In memory-limited cases, we can significantly reduce memory by de-activating this option, but at the expense of a slight performance degradation.")
    parser.add_argument('--act_order_col', action='store_true', help='Whether to reorder columns based on column-wise Hessian diagonals')
    parser.add_argument('--act_order_row', action='store_true', help='Whether to reorder rows based on row-wise Hessian diagonals')

    parser.add_argument('--replace', type=float, default=1, help='Value to be replaced for the Hessian diagonal elements corresponding to dead neurons')
    
    # LM Eval Arguments
    parser.add_argument("--lm_eval", action="store_true", help="Evaluate the model on LM Eval tasks.")
    parser.add_argument('--tasks', nargs='+', default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada_openai", "lambada_standard", "openbookqa", "boolq"])
    parser.add_argument('--lm_eval_batch_size', type=int, default=16, help='Batch size for evaluating with lm eval harness.')
    
    args = parser.parse_args()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    if args.tokenizer_path is None:
        args.tokenizer_path = args.llm_path
    args.llm_name = args.tokenizer_path.split('/')[-1]
    args.llm_type = args.llm_name.split('-')[0]

    args.replace = 1 / args.seqlen

    return args


def get_boa_weight_quant_infos(args):
    qconfigs = {
        "w_bits": args.w_bits,
        "w_sym": args.w_sym,
        "codebook": args.codebook,
        "sharq_selector": args.sharq_selector,
        "sharq_group_size": args.sharq_group_size,
        "sharq_hist_bins": args.sharq_hist_bins,
        "sharq_zero_policy": args.sharq_zero_policy,
        "sharq_clip_min": args.sharq_clip_min,
        "sharq_clip_max": args.sharq_clip_max,
        "sharq_clip_steps": args.sharq_clip_steps,
    }
    boa_opts = {
        "qparam_comput": args.qparam_comput,
        "block_v": args.block_v,
        'act_order_col': args.act_order_col, 
        'act_order_row': args.act_order_row, 
    }
    hyperparams = {"replace": args.replace}
    
    return qconfigs, boa_opts, hyperparams
