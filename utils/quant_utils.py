import torch


def quantize(x, scale, zero, maxq):
    x_int = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return x_int


def fake_quantize(x, scale, zero, maxq):
    return scale * (quantize(x, scale, zero, maxq) - zero)


def grid_search(w, w_max, w_min, maxq, sym, power, n_grids, H_col, search=True):
    scale = (w_max - w_min) / maxq
    if sym:
        zero = torch.full_like(scale, (maxq + 1) / 2)
    else:
        zero = torch.round(-w_min / scale)

    if not search:
        return scale, zero
    
    else:
        best_score = torch.full_like(w_min, 1e10)
        best_scale = scale
        best_zero = zero

        for i in range(n_grids):
            p = 1 - i / n_grids
            new_w_min = p * w_min
            new_w_max = p * w_max

            for round in ("floor", "ceil"):
                new_scale = (new_w_max - new_w_min) / maxq
                if sym:
                    new_zero = zero
                else:
                    new_zero = torch.floor(-new_w_min / new_scale) if round=="floor" else torch.ceil(-new_w_min / new_scale)
                q = fake_quantize(w, new_scale, new_zero, maxq)

                q -= w
                if H_col is not None:
                    score = torch.sum((q @ H_col) * q, dim=-1, keepdim=True)
                else:
                    q.abs_()            
                    q.pow_(power)
                    score = torch.sum(q, dim=-1, keepdim=True) 
                tmp = score < best_score
                if torch.any(tmp):
                    best_score[tmp] = score[tmp]
                    best_scale[tmp] = new_scale[tmp]
                    best_zero[tmp] = new_zero[tmp]

        return best_scale, best_zero


def damping(H, percdamp=.01):  
    # Calculate the mean of diagonals across all heads  
    mean_diags = torch.mean(torch.diagonal(H, dim1=-2, dim2=-1), dim=-1)  

    # Add the damping values back into the original tensor along the diagonals  
    H.diagonal(dim1=-2, dim2=-1).add_(mean_diags.view(-1, *[1]*(len(H.shape)-2)), alpha=percdamp)  

    return H


def filter_dead_neuron(W, H_col, replace=1/2048, percdamp=.01, apply_damping=True):
    if len(H_col.shape) == 2:  
        H_col = H_col.unsqueeze(0)  
    num_heads, in_features = H_col.shape[0], H_col.shape[-1]  
    W = W.view(num_heads, -1, in_features)  

    # Extract the diagonals of H and find indices where they are equal to 0  
    diagonals = torch.diagonal(H_col, dim1=-2, dim2=-1)  
    idx_dead = (diagonals == 0)  

    # Set the corresponding columns of W to 0 and replace the dead neurons in H with the given value  
    mask = ~idx_dead.unsqueeze(-2)
    W *= mask  
    H_col.diagonal(dim1=-2, dim2=-1)[idx_dead] = replace  

    if apply_damping:
        H_col = damping(H_col, percdamp)

    W = W.view(-1, in_features)  
    H_col = H_col.squeeze()

    return W, H_col