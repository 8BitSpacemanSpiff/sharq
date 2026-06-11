import torch

from quantizers.utils import get_cholesky_of_inverse, reorder_col, reverse_reorder_col, reorder_row, reverse_reorder_row
from utils.quant_utils import fake_quantize, filter_dead_neuron, damping
from utils.utils import cleanup_memory
from sharq.candidates import deduped_representatives, filter_by_zero_policy
from sharq.direct import make_online_uniform_scale_selection, select_direct, select_direct_channelwise, select_uniform_scale_channelwise
from sharq.quantizer import build_signed_levels, quantize_to_codebook

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class BoA:
    def __init__(self, layer, opts, hyperparams):
        self.layer = layer
        W = self.layer.weight.data
        self.org_shape, self.org_dtype = W.shape, W.dtype

        self.quantizer = None
        self.H_col = None
        self.H_row = None
        self.sharq_selection = None
        self.sharq_group_size = -1
        self.sharq_scale_override = None
        self.sharq_zero_override = None
        self.sharq_bits = None
        self.sharq_zero_policy = "free"
        self.sharq_online_candidates = None

        self.qparam_comput = opts['qparam_comput']
        self.act_order_col = opts['act_order_col']
        self.act_order_row = opts['act_order_row']
        self.hyperparams = hyperparams


    def quant(self, print_memory_usage=False):
        assert self.quantizer is not None, "Quantizer should be defined first."
        assert self.H_col is not None, "Hessian should be computed first."

        W, H_col, H_row = self.preprocess()
        
        # compute quantization grid
        if self.sharq_selection is None or self.sharq_selection.online_uniform_scale:
            if self.qparam_comput == "MinMax":
                scale, zero = self.quantizer.find_params_H(W, None, search=False)
            elif self.qparam_comput == "MMSE":
                scale, zero = self.quantizer.find_params_H(W, None, search=True)
            elif self.qparam_comput == "Hessian":
                scale, zero = self.quantizer.find_params_H(W, H_col, search=True)
            else:
                raise NotImplementedError()
            if self.sharq_selection is None:
                self.quantizer.scale = scale.reshape(self.quantizer.scale.shape)
                self.quantizer.zero = zero.reshape(self.quantizer.zero.shape)
        else:
            if self.act_order_col:
                raise NotImplementedError("SHARQ group-wise scales do not yet support act_order_col.")
            if self.sharq_scale_override is not None:
                scale, zero = self.sharq_scale_override.to(W.device), self.sharq_zero_override.to(W.device)
            else:
                scale, zero = self._find_sharq_params(W)

        # Hessian-based re-ordering for columns
        if self.act_order_col:
            W, H_col, invperm_col = reorder_col(W, H_col)
        
        if H_row is None:
            Q = self.gptq(W, H_col, scale, zero)
        else: 
            # Hessian-based re-ordering for rows
            if self.act_order_row:
                W, H_row, scale, zero, invperm_row  = reorder_row(W, H_row, scale, zero)
            
            Q = self.boa(W, H_col, H_row, scale, zero)
            
            # reverse re-ordering for rows
            if self.act_order_row:
                Q = reverse_reorder_row(Q, invperm_row)
        
        # reverse re-ordering for columns
        if self.act_order_col:
            Q = reverse_reorder_col(Q, invperm_col)
        
        if print_memory_usage:
            print(f'\t |GPU memory: {torch.cuda.max_memory_allocated("cuda") / 1024**3:.3f}|')
        
        # assign quantized (fake-quant) weights
        self.layer.weight.data = Q.reshape(self.org_shape).to(self.org_dtype)


    def gptq(self, W, H_col, scale, zero, return_err=False, row_offset=0):
        U_col = get_cholesky_of_inverse(H_col)
        Q = torch.zeros_like(W)
        Err = torch.zeros_like(W)
        for idx_col in range(W.shape[-1]):
            # quantization
            w = W[..., idx_col].unsqueeze(-1)
            scale_col = self._scale_at_col(scale, idx_col)
            zero_col = self._scale_at_col(zero, idx_col)
            q = self._fake_quantize(w, scale_col, zero_col, row_offset=row_offset)
            Q[..., idx_col] = q.squeeze(-1)

            # error compensation
            err = (w - q) / U_col[..., idx_col, idx_col][:, None, None]
            Err[..., idx_col] = err.squeeze(-1)
            W[..., idx_col:] -= err @ U_col[..., idx_col, idx_col:].unsqueeze(-2)

        if return_err:
            return Q, Err
        else:
            return Q
    

    def boa(self, W, H_col, H_row, scale, zero):
        U_col = get_cholesky_of_inverse(H_col)
        U_row = get_cholesky_of_inverse(H_row)
        Q = torch.zeros_like(W)
        for idx_row in range(W.shape[1]):
            # quantization
            W_sub = W[:, idx_row, :].unsqueeze(-2)
            Q_sub, Err = self.gptq(
                W_sub,
                H_col,
                scale[:, idx_row, :].unsqueeze(-2),
                zero[:, idx_row, :].unsqueeze(-2),
                return_err=True,
                row_offset=idx_row,
            )
            Q[:, idx_row, :] = Q_sub.squeeze(-2)

            # error compensation
            W[:, idx_row:, :] -= (U_row.transpose(-1, -2)[:, idx_row:, idx_row].unsqueeze(-1) @ Err @ U_col) / U_row[:, idx_row, idx_row][:, None, None]

        return Q


    def set_sharq(self, selection, group_size, bits=None, zero_policy="free"):
        self.sharq_selection = selection
        self.sharq_group_size = group_size
        self.sharq_bits = bits
        self.sharq_zero_policy = zero_policy


    def set_sharq_with_uniform_scale(self, selection, scale, zero):
        self.sharq_selection = selection
        self.sharq_group_size = -1
        self.sharq_scale_override = scale.detach().cpu()
        self.sharq_zero_override = zero.detach().cpu()


    def set_sharq_uniform_scale_online(self, bits, zero_policy, topk=None):
        self.sharq_group_size = -1
        self.sharq_bits = bits
        self.sharq_zero_policy = zero_policy
        if topk is not None and topk > 0:
            selection, _, _ = self.select_sharq_uniform_scale(bits, zero_policy, topk=topk)
            candidates = selection.shortlisted_codebooks
        else:
            candidates = filter_by_zero_policy(deduped_representatives(bits), zero_policy)
        self.sharq_selection = make_online_uniform_scale_selection(bits, candidates)
        self.sharq_online_candidates = candidates


    def select_sharq_direct(self, bits, group_size, zero_policy, clip_grid, granularity="module"):
        W, H_col, _ = self._prepare_tensors(clear=False)
        if granularity == "channel":
            return select_direct_channelwise(W, H_col, bits, group_size, zero_policy, clip_grid)
        return select_direct(W, H_col, bits, group_size, zero_policy, clip_grid, objective="hessian")


    def select_sharq_uniform_scale(self, bits, zero_policy, topk=None):
        W, H_col, _ = self._prepare_tensors(clear=False)
        if self.qparam_comput == "MinMax":
            scale, zero = self.quantizer.find_params_H(W, None, search=False)
        elif self.qparam_comput == "MMSE":
            scale, zero = self.quantizer.find_params_H(W, None, search=True)
        elif self.qparam_comput == "Hessian":
            scale, zero = self.quantizer.find_params_H(W, H_col, search=True)
        else:
            raise NotImplementedError()
        selection = select_uniform_scale_channelwise(
            W,
            H_col,
            scale,
            zero,
            self.quantizer.maxq.to(W.device),
            bits,
            zero_policy,
            topk=topk,
        )
        return selection, scale, zero


    def _fake_quantize(self, w, scale, zero, row_offset=0):
        if self.sharq_selection is None:
            return fake_quantize(w, scale, zero, self.quantizer.maxq)
        if self.sharq_selection.online_uniform_scale:
            return self._fake_quantize_uniform_scale_online(w, scale, zero)
        if self.sharq_selection.codebook_granularity == "channel":
            return self._fake_quantize_channelwise(w, scale, zero, row_offset)
        levels = build_signed_levels(self.sharq_selection.codebook, device=w.device, dtype=w.dtype)
        q, _ = quantize_to_codebook(w, scale, levels)
        return q


    def _fake_quantize_uniform_scale_online(self, w, scale, zero):
        uniform_q = fake_quantize(w, scale, zero, self.quantizer.maxq)
        best_q = uniform_q.clone()
        best_err = (uniform_q - w).abs()
        safe_scale = torch.clamp(scale, min=torch.finfo(w.dtype).tiny)
        t = w / safe_scale
        for codebook in self.sharq_online_candidates:
            levels = build_signed_levels(codebook, device=w.device, dtype=w.dtype)
            idx_right = torch.searchsorted(levels.contiguous(), t.contiguous()).clamp(max=levels.numel() - 1)
            idx_left = (idx_right - 1).clamp(min=0)
            left = levels[idx_left]
            right = levels[idx_right]
            idx = torch.where((t - left).abs() <= (right - t).abs(), idx_left, idx_right)
            q = safe_scale * levels[idx]
            err = (q - w).abs()
            update = err < best_err
            best_err = torch.where(update, err, best_err)
            best_q = torch.where(update, q, best_q)
        return best_q


    def _fake_quantize_channelwise(self, w, scale, zero, row_offset):
        q = torch.empty_like(w)
        for h, head_codebooks in enumerate(self.sharq_selection.channel_codebooks):
            for r in range(w.shape[1]):
                codebook = head_codebooks[row_offset + r]
                if len(codebook) == 0:
                    q[h, r, :] = fake_quantize(w[h, r, :], scale[h, r, :], zero[h, r, :], self.quantizer.maxq)
                    continue
                levels = build_signed_levels(codebook, device=w.device, dtype=w.dtype)
                q[h, r, :], _ = quantize_to_codebook(w[h, r, :], scale[h, r, :], levels)
        return q


    def _scale_at_col(self, tensor, idx_col):
        if tensor.shape[-1] == 1:
            return tensor
        group_size = self.sharq_group_size if self.sharq_group_size != -1 else tensor.shape[-1]
        group_idx = min(idx_col // group_size, tensor.shape[-1] - 1)
        return tensor[..., group_idx].unsqueeze(-1)


    def _find_sharq_params(self, W):
        group_size = self.sharq_group_size if self.sharq_group_size != -1 else W.shape[-1]
        if W.shape[-1] % group_size != 0:
            raise ValueError("SHARQ group_size must divide the input dimension.")
        n_groups = W.shape[-1] // group_size
        grouped = W.abs().view(*W.shape[:-1], n_groups, group_size)
        gmax = grouped.amax(dim=-1)
        if self.sharq_selection.codebook_granularity == "channel":
            clips = self.sharq_selection.channel_clips.to(device=W.device, dtype=W.dtype)
            zmax = torch.tensor(
                [[max(codebook) for codebook in head] for head in self.sharq_selection.channel_codebooks],
                device=W.device,
                dtype=W.dtype,
            )
            scale = gmax * clips.unsqueeze(-1) / zmax.unsqueeze(-1)
            zero = torch.zeros_like(scale)
            return scale, zero
        zmax = max(self.sharq_selection.codebook)
        scale = gmax.mul(float(self.sharq_selection.clip) / float(zmax))
        zero = torch.zeros_like(scale)
        return scale, zero
    

    def preprocess(self):
        return self._prepare_tensors(clear=True)


    def _prepare_tensors(self, clear):
        W = self.layer.weight.data.clone()
        W = W.float()

        H_col, H_row = self.H_col.clone(), self.H_row.clone() if self.H_row is not None else None
        W, H_col = filter_dead_neuron(W, H_col, replace=self.hyperparams['replace'], apply_damping=True)
        if H_row is not None:
            H_row = damping(H_row)
        if len(H_col.shape) == 2:  # common Hessian for all heads
            H_col = H_col.unsqueeze(0)
        
        n_heads = H_row.shape[0] if H_row is not None else H_col.shape[0]
        hidden_size = W.shape[-1]
        head_dim = W.shape[0] // n_heads
        W = W.view(n_heads, head_dim, hidden_size)

        if clear:
            self.H_col = None
            self.H_row = None

        return W, H_col, H_row


    def free(self):
        self.H_col = None
        self.H_row = None

        cleanup_memory(verbose=False)
