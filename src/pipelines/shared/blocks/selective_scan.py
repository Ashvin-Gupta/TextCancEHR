# Copyright (c) 2023, Tri Dao, Albert Gu.
# Adapted for Nightingale project

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    Reference implementation of selective scan operation in pure PyTorch.

    This is a fallback implementation that provides the same functionality as the CUDA-optimized
    version but uses standard PyTorch operations. It's slower but more compatible and useful
    for debugging and development.

    Args:
        u (torch.Tensor): Input tensor of shape (batch, d_inner, seqlen)
        delta (torch.Tensor): Delta tensor of shape (batch, d_inner, seqlen)
        A (torch.Tensor): A matrix of shape (d_inner, d_state)
        B (torch.Tensor): B matrix of shape (batch, d_state, seqlen) or (batch, d_inner, d_state, seqlen)
        C (torch.Tensor): C matrix of shape (batch, d_state, seqlen) or (batch, d_inner, d_state, seqlen)
        D (torch.Tensor, optional): D matrix of shape (d_inner,)
        z (torch.Tensor, optional): Gate tensor of shape (batch, d_inner, seqlen)
        delta_bias (torch.Tensor, optional): Delta bias of shape (d_inner,)
        delta_softplus (bool): Whether to apply softplus to delta
        return_last_state (bool): Whether to return the final state

    Returns:
        torch.Tensor: Output tensor of shape (batch, d_inner, seqlen)
        torch.Tensor (optional): Last state if return_last_state=True
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()

    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    if A.is_complex():
        # Complex implementation
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()

    x = A.new_zeros((batch, dim, dstate), dtype=A.dtype)
    ys = []

    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)

    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])

    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (batch dim L)

    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)

    if return_last_state:
        return out, last_state
    else:
        return out


def mamba_inner_ref(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                   out_proj_weight, out_proj_bias,
                   A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                   C_proj_bias=None, delta_softplus=True, checkpoint_lvl=0):
    """
    Reference implementation of the inner Mamba computation in pure PyTorch.

    This combines the convolution, projection, and selective scan operations that form
    the core of a Mamba block.

    Args:
        xz (torch.Tensor): Input tensor of shape (batch, seqlen, 2 * d_inner)
        conv1d_weight (torch.Tensor): 1D convolution weight of shape (d_inner, d_conv) or (d_inner, 1, d_conv)
        conv1d_bias (torch.Tensor): 1D convolution bias
        x_proj_weight (torch.Tensor): Projection weight for x
        delta_proj_weight (torch.Tensor): Projection weight for delta
        out_proj_weight (torch.Tensor): Output projection weight
        out_proj_bias (torch.Tensor): Output projection bias
        A (torch.Tensor): A matrix for selective scan
        B (torch.Tensor, optional): B matrix for selective scan
        C (torch.Tensor, optional): C matrix for selective scan
        D (torch.Tensor, optional): D matrix for selective scan
        delta_bias (torch.Tensor, optional): Delta bias
        B_proj_bias (torch.Tensor, optional): B projection bias
        C_proj_bias (torch.Tensor, optional): C projection bias
        delta_softplus (bool): Whether to apply softplus to delta
        checkpoint_lvl (int): Checkpointing level (unused in reference)

    Returns:
        torch.Tensor: Output tensor
    """
    L = xz.shape[-2]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    x, z = xz.chunk(2, dim=-1)

    # 1D Convolution
    # Note: Reference implementation accepts both (d_inner, d_conv) and (d_inner, 1, d_conv)
    # F.conv1d expects (out_channels, in_channels/groups, kernel_size)
    # For grouped convolution with groups=d_inner, we need the singleton dimension
    if conv1d_weight.dim() == 2:
        conv1d_weight = conv1d_weight.unsqueeze(1)  # (d_inner, 1, d_conv)

    if conv1d_bias is not None:
        x = F.conv1d(x.transpose(1, 2), conv1d_weight, conv1d_bias, padding=conv1d_weight.shape[-1] - 1, groups=x.shape[-1])
    else:
        x = F.conv1d(x.transpose(1, 2), conv1d_weight, padding=conv1d_weight.shape[-1] - 1, groups=x.shape[-1])
    x = x[..., :L].transpose(1, 2)

    # Projections
    x_dbl = F.linear(rearrange(x, 'b l d -> (b l) d'), x_proj_weight)  # (bl d)
    delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), 'd (b l) -> b d l', b=x.shape[0], l=L)
    if delta_bias is not None:
        delta = delta + delta_bias[..., None]

    if B is None:  # variable B
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        B = rearrange(B, "(b l) dstate -> b dstate l", b=x.shape[0], l=L).contiguous()
    else:
        B = repeat(B, "dstate -> b dstate l", b=x.shape[0], l=L)

    if C is None:  # variable C
        C = x_dbl[:, -d_state:]  # (bl dstate)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        C = rearrange(C, "(b l) dstate -> b dstate l", b=x.shape[0], l=L).contiguous()
    else:
        C = repeat(C, "dstate -> b dstate l", b=x.shape[0], l=L)

    # Note: B and C already have shape (b, dstate, l) which is what selective_scan_ref expects
    y = selective_scan_ref(x.transpose(1, 2), delta, A, B, C, D, z=z.transpose(1, 2), delta_softplus=delta_softplus)
    y_out = y.transpose(1, 2).contiguous().view(-1, y.shape[1])
    return F.linear(rearrange(y.transpose(1, 2), 'b l d -> (b l) d'), out_proj_weight, out_proj_bias)


# Try to import optimized implementations, fall back to reference if not available
HAS_SELECTIVE_SCAN_CUDA = False
selective_scan_fn = selective_scan_ref
mamba_inner_fn = mamba_inner_ref

try:
    # Try to import the CUDA functions one by one
    import mamba_ssm.ops.selective_scan_interface as ssm_ops
    selective_scan_fn = ssm_ops.selective_scan_fn
    mamba_inner_fn = ssm_ops.mamba_inner_fn
    HAS_SELECTIVE_SCAN_CUDA = True
except (ImportError, AttributeError) as e:
    # Fall back to reference implementations
    pass


def selective_scan_wrapper(*args, **kwargs):
    """
    Wrapper that automatically chooses between CUDA-optimized and reference implementation.

    Falls back to pure PyTorch implementation if CUDA ops are not available or if tensors are on CPU.
    """
    # Check if CUDA is available AND if the input tensors are on CUDA
    if HAS_SELECTIVE_SCAN_CUDA and len(args) > 0 and hasattr(args[0], 'is_cuda') and args[0].is_cuda:
        return selective_scan_fn(*args, **kwargs)
    else:
        return selective_scan_ref(*args, **kwargs)


def mamba_inner_wrapper(*args, **kwargs):
    """
    Wrapper that automatically chooses between CUDA-optimized and reference implementation.

    Falls back to pure PyTorch implementation if CUDA ops are not available.

    Note: The CUDA kernel (mamba_inner_fn from mamba_ssm) expects conv1d_weight
    to have shape (d_inner, 1, d_conv) and squeezes it internally.
    The reference implementation (mamba_inner_ref) handles both (d_inner, d_conv)
    and (d_inner, 1, d_conv) internally.
    """
    if HAS_SELECTIVE_SCAN_CUDA:
        # Ensure conv1d_weight has proper memory layout for CUDA kernel
        # The issue is that einops rearrange can produce tensors with unexpected strides
        if len(args) > 1:
            args = list(args)
            conv1d_weight = args[1]
            # Make sure it's contiguous to avoid stride issues after rearrange
            if not conv1d_weight.is_contiguous():
                args[1] = conv1d_weight.contiguous()
        # CUDA kernel expects conv1d_weight shape (d_inner, 1, d_conv) and handles squeezing internally
        return mamba_inner_fn(*tuple(args), **kwargs)
    else:
        # Reference implementation handles shape normalization internally
        return mamba_inner_ref(*args, **kwargs)