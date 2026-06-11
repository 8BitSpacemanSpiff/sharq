import torch
from utils.quant_utils import damping


def get_cholesky_of_inverse(H):
    U = torch.zeros_like(H)
    for i in range(len(H)):
        compute_cholesky = False
        while not compute_cholesky:
            try:
                U[i] = torch.linalg.cholesky(
                    torch.cholesky_inverse(torch.linalg.cholesky(H[i])), upper=True
                )
                compute_cholesky = True
            except:
                H[i] = damping(H[i])
    
    return U


def reorder_col(W, H_col):
    org_shape = W.shape
    hidden_size = org_shape[-1]
    if H_col.shape[0] == 1:  # Common Hessian for all heads
        W = W.view(1, -1, hidden_size)

    perm = torch.argsort(torch.diagonal(H_col, dim1=-2, dim2=-1), dim=-1, descending=True)
    W = torch.gather(W, dim=-1, index=perm.unsqueeze(-2).expand(-1, W.shape[-2], -1))
    H_col = torch.gather(
        torch.gather(H_col, dim=-2, index=perm.unsqueeze(-1).expand(-1, -1, H_col.shape[-1])),
        dim=-1, index=perm.unsqueeze(-2).expand(-1, H_col.shape[-2], -1)
    )
    invperm = torch.argsort(perm, dim=-1)
  
    W = W.view(org_shape)

    return W, H_col, invperm


def reverse_reorder_col(W, invperm):
    org_shape = W.shape
    hidden_size = W.shape[-1]

    W = W.reshape(invperm.shape[0], -1, hidden_size)
    W = torch.gather(W, dim=-1, index=invperm.unsqueeze(-2).expand(-1, W.shape[-2], -1))    
    W = W.reshape(org_shape)
    
    return W


def reorder_row(W, H_row, scale, zero):
    perm = torch.argsort(torch.diagonal(H_row, dim1=-2, dim2=-1), dim=-1, descending=True)
    W = torch.gather(W, dim=-2, index=perm.unsqueeze(-1).expand(-1, -1, W.shape[-1]))
    H_row = torch.gather(
        torch.gather(H_row, dim=-2, index=perm.unsqueeze(-1).expand(-1, -1, H_row.shape[-1])),
        dim=-1, index=perm.unsqueeze(-2).expand(-1, H_row.shape[-2], -1)
    )
    scale = torch.gather(scale, dim=-2, index=perm.unsqueeze(-1).expand(-1, -1, scale.shape[-1]))
    zero = torch.gather(zero, dim=-2, index=perm.unsqueeze(-1).expand(-1, -1, zero.shape[-1]))
    invperm = torch.argsort(perm, dim=-1)


    return W, H_row, scale, zero, invperm 
    

def reverse_reorder_row(W, invperm_row):
    W = torch.gather(W, dim=-2, index=invperm_row.unsqueeze(-1).expand(-1, -1, W.shape[-1]))    
        
    return W