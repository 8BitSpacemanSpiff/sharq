import torch
from torch import nn
from .utils import cleanup_memory

def skip(*args, **kwargs):
    pass
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip


def get_opt(model_path):
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    return model


def get_llama(model_path):
    from custom_modelings.modeling_llama import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model_path, attn_implementation="eager", torch_dtype='auto')
    return model


def get_qwen2(model_path):
    from custom_modelings.modeling_qwen2 import Qwen2ForCausalLM
    model = Qwen2ForCausalLM.from_pretrained(model_path, attn_implementation="eager", torch_dtype='auto')
    return model

def get_qwen3(model_path):
    from custom_modelings.modeling_qwen3 import Qwen3ForCausalLM
    model = Qwen3ForCausalLM.from_pretrained(model_path, attn_implementation="eager", torch_dtype='auto')
    return model

    
def get_model(model_path):
    if 'llama' in model_path:
        return get_llama(model_path)
    elif 'opt' in model_path:
        return get_opt(model_path)
    elif "qwen2" in model_path.lower():
        return get_qwen2(model_path)
    elif "qwen3" in model_path.lower():
        return get_qwen3(model_path)
    else:
        raise NotImplemented(f"Not support {model_path.split('/')[-1]} model.")
    

def get_transformer_blocks(llm):
    llm_type = type(llm).__name__
    if llm_type == "OPTForCausalLM":
        transformer_blocks = llm.model.decoder.layers
    elif llm_type == "LlamaForCausalLM":
        transformer_blocks = llm.model.layers
    elif llm_type in ["Qwen2ForCausalLM", "Qwen3ForCausalLM"]:
        transformer_blocks = llm.model.layers
    else:
        raise NotImplementedError(f"{llm_type} models are NOT supported.")

    return transformer_blocks


def get_head_info(llm):
    llm_type = type(llm).__name__
    if llm_type == "OPTForCausalLM":
        n_heads = llm.config.num_attention_heads
        n_kv_heads = n_heads
        head_dim = llm.config.hidden_size // n_heads
    elif llm_type == "LlamaForCausalLM":
        n_heads = llm.config.num_attention_heads
        n_kv_heads = llm.config.num_key_value_heads
        head_dim = getattr(llm.config, "head_dim", llm.config.hidden_size // n_heads)
    elif llm_type in ["Qwen2ForCausalLM", "Qwen3ForCausalLM"]:
        n_heads = llm.config.num_attention_heads
        n_kv_heads = llm.config.num_key_value_heads
        head_dim = getattr(llm.config, "head_dim", llm.config.hidden_size // n_heads)
    else:
        raise NotImplementedError(f"{llm_type} models are NOT supported.")

    return n_heads, n_kv_heads, head_dim


def get_rotary_emb(llm):
    llm_type = type(llm).__name__
    if llm_type == "OPTForCausalLM":
        rotary_emb = None
    elif llm_type == "LlamaForCausalLM":
        rotary_emb = llm.model.rotary_emb
    elif llm_type in ["Qwen2ForCausalLM", "Qwen3ForCausalLM"]:
        rotary_emb = llm.model.rotary_emb
    else:
        raise NotImplementedError(f"{llm_type} models are NOT supported.")

    return rotary_emb


@torch.no_grad()
def cache_first_transformer_input(llm, data):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = next(iter(llm.parameters())).dtype
    inps = torch.zeros((len(data), llm.seqlen, llm.config.hidden_size), dtype=dtype, device=dev)
    block_kwargs = {}

    transformer_blocks = get_transformer_blocks(llm)
    set_device_before_first_transformer_block(llm, dev)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.n_data = 0

            # Sliding Attention specific
            self.attention_type = getattr(module, "attention_type", None)

        def forward(self, inp, **kwargs):
            inps[self.n_data] = inp[0]
            if self.n_data == 0:
                for key, value in kwargs.items():
                    block_kwargs[key] = value
            self.n_data += 1
            raise ValueError
    
    transformer_blocks[0] = transformer_blocks[0].to(dev)
    transformer_blocks[0] = Catcher(transformer_blocks[0])
    for batch in data:
        try:
            llm(batch[0].to(dev))
        except ValueError:
            pass
    transformer_blocks[0] = transformer_blocks[0].module

    set_device_before_first_transformer_block(llm, dev="cpu")

    cleanup_memory(verbose=False)

    return inps, block_kwargs


def set_device_before_first_transformer_block(llm, dev):
    llm_type = type(llm).__name__
    if llm_type == "LlamaForCausalLM":
        llm.model.embed_tokens = llm.model.embed_tokens.to(dev)
    elif llm_type in ["Qwen2ForCausalLM", "Qwen3ForCausalLM"]:
        llm.model.embed_tokens = llm.model.embed_tokens.to(dev)
    elif llm_type == "OPTForCausalLM":
        llm.model.decoder.embed_tokens = llm.model.decoder.embed_tokens.to(dev) 
        llm.model.decoder.embed_positions = llm.model.decoder.embed_positions.to(dev)
        if llm.model.decoder.project_in is not None:
            llm.model.decoder.project_in = llm.model.decoder.project_in.to(dev)
    else:
        raise NotImplementedError(f"{llm_type} models are not supported yet")
    

@torch.no_grad()
def get_logits_from_last_hidden_states(llm, hidden_states):
    llm_type = type(llm).__name__
    if llm_type == "LlamaForCausalLM":
        if llm.model.norm is not None:
            hidden_states = llm.model.norm(hidden_states)
        lm_logits = llm.lm_head(hidden_states)
    elif llm_type in ["Qwen2ForCausalLM", "Qwen3ForCausalLM"]:
        if llm.model.norm is not None:
            hidden_states = llm.model.norm(hidden_states)
        lm_logits = llm.lm_head(hidden_states)
    elif llm_type == "OPTForCausalLM":
        if llm.model.decoder.final_layer_norm is not None:
            hidden_states = llm.model.decoder.final_layer_norm(hidden_states)
        if llm.model.decoder.project_out is not None:
            hidden_states = llm.model.decoder.project_out(hidden_states)
        lm_logits = llm.lm_head(hidden_states)
    else:
        raise NotImplementedError(f"{llm_type} models are not supported yet")
    
    return lm_logits


def set_device_after_last_transformer_block(llm, dev):
    llm_type = type(llm).__name__
    if llm_type == "LlamaForCausalLM":
        if llm.model.norm is not None:
            llm.model.norm = llm.model.norm.to(dev)
        llm.lm_head = llm.lm_head.to(dev)
    elif llm_type in ["Qwen2ForCausalLM", "Qwen3ForCausalLM"]:
        if llm.model.norm is not None:
            llm.model.norm = llm.model.norm.to(dev)
        llm.lm_head = llm.lm_head.to(dev)
    elif llm_type == "OPTForCausalLM":
        if llm.model.decoder.final_layer_norm is not None:
            llm.model.decoder.final_layer_norm = llm.model.decoder.final_layer_norm.to(dev)
        if llm.model.decoder.project_out is not None:
            llm.model.decoder.project_out = llm.model.decoder.project_out.to(dev)
        llm.lm_head = llm.lm_head.to(dev)
    else:
        raise NotImplementedError(f"{llm_type} models are not supported yet")