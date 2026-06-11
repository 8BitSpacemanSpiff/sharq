## Custom Modification
 - When computing `H_row` for query/key projection layers, we need to compute the covariance of query/key **after** applying RoPE.
 - To cache outputs after applying RoPE, we add `nn.Identity` layers (`layers.*.self_attn.rot_out_Q (rot_out_K)`) after the `apply_rotary_pos_emb` module.
 - Please refer to this when quantizing LLMs of other classes (except Llama and Qwen).
