# Copyright (c) 2023, Tri Dao, Albert Gu.
# Adapted for Nightingale project

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from .selective_scan import selective_scan_wrapper, mamba_inner_wrapper, HAS_SELECTIVE_SCAN_CUDA

# Try to import optional optimizations
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    HAS_CAUSAL_CONV1D = True
    print("Successfully imported causal_conv1d optimizations")
except Exception as e:
    print(f"Failed to import causal_conv1d. Error details: {e}")
    # Check if it's a version mismatch
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    causal_conv1d_fn, causal_conv1d_update = None, None
    HAS_CAUSAL_CONV1D = False

class RMSNorm(nn.Module):
    """Root-mean-square normalization matching the reference Mamba implementation."""

    def __init__(self, dim: int, eps: float = 1e-6, *, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x_float = x.float()
        rms = x_float.square().mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(rms + self.eps)
        return (self.weight * x_norm.to(original_dtype))


class MambaBlock(nn.Module):
    """
    Implementation of a single Mamba block adapted for the Nightingale architecture.

    This is the core building block of the Mamba architecture, implementing a selective
    state space model with causal convolution and gated linear units. The implementation
    automatically falls back to pure PyTorch operations when CUDA optimizations are not available.

    Args:
        d_model (int): Model dimension
        d_state (int): SSM state expansion factor, typically 16
        d_conv (int): Local convolution width, typically 4
        expand (int): Block expansion factor, typically 2
        dt_rank (str or int): Rank of Δ (delta) projection. 'auto' sets it to ceil(d_model/16)
        dt_min (float): Minimum value for delta initialization
        dt_max (float): Maximum value for delta initialization
        dt_init (str): Initialization method for delta ('random' or 'constant')
        dt_scale (float): Scale factor for delta initialization
        dt_init_floor (float): Floor value for delta initialization
        conv_bias (bool): Whether to use bias in convolution
        bias (bool): Whether to use bias in linear projections
        use_fast_path (bool): Whether to use optimized CUDA kernels when available
        layer_idx (int, optional): Layer index for caching
        device (torch.device, optional): Device to place the module
        dtype (torch.dtype, optional): Data type for the module
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        layer_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path and HAS_SELECTIVE_SCAN_CUDA and HAS_CAUSAL_CONV1D
        self.layer_idx = layer_idx

        # Input projection: d_model -> 2 * d_inner (for x and z branches)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # 1D Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        # Activation function
        self.activation = "silu"  # SiLU activation
        self.act = nn.SiLU()

        # SSM projections
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize dt (delta) projection
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias to be between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization for A matrix
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, inference_params=None):
        """
        Forward pass of the Mamba block.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_len).
                1 for real tokens, 0 for padding tokens. This is used to mask padding in the output.
            inference_params: Parameters for inference caching (unused in this implementation)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch, seqlen, dim = hidden_states.shape

        # Input projection and split into x and z branches
        xz = self.in_proj(hidden_states)  # (batch, seqlen, d_inner * 2)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Use fast path if available, otherwise fall back to reference implementation
        if self.use_fast_path and inference_params is None:
            # Fast path using optimized kernels
            # Note: mamba_inner_fn expects xz in (batch, d_inner*2, seqlen) format
            # Note: conv1d.weight has shape (d_inner, 1, d_conv); CUDA kernel squeezes internally
            xz_transposed = rearrange(xz, "b l d -> b d l").contiguous()  # (batch, d_inner*2, seqlen)
            out = mamba_inner_wrapper(
                xz_transposed, self.conv1d.weight, self.conv1d.bias, self.x_proj.weight, self.dt_proj.weight,
                self.out_proj.weight, self.out_proj.bias,
                A, None, None, self.D.float(), delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            # mamba_inner_fn already returns (batch, seqlen, d_model) - no reshape needed!
        else:
            # Reference implementation
            x, z = xz.chunk(2, dim=-1)  # (batch, seqlen, d_inner)

            # 1D Convolution
            x = rearrange(x, "b l d -> b d l")
            if HAS_CAUSAL_CONV1D and x.is_cuda:
                # Use optimized causal conv if available and on GPU
                # Note: conv1d.weight has shape (d_inner, 1, d_conv) but CUDA kernel expects (d_inner, d_conv)
                x = causal_conv1d_fn(
                    x,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    activation=self.activation,
                )
            else:
                # Fallback to standard conv1d
                x = self.conv1d(x)[..., :seqlen]  # (batch, d_inner, seqlen)
                x = self.act(x)
                
            x = rearrange(x, "b d l -> b l d")

            # SSM computation
            x_dbl = self.x_proj(rearrange(x, "b l d -> (b l) d"))  # (batch * seqlen, dt_rank + 2 * d_state)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()  # (d_inner, batch * seqlen)
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)

            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

            # Apply delta bias and softplus
            dt = dt + self.dt_proj.bias[..., None]
            dt = F.softplus(dt)

            # Selective scan
            y = selective_scan_wrapper(
                rearrange(x, "b l d -> b d l"),
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=rearrange(z, "b l d -> b d l"),
                delta_softplus=False,  # Already applied above
            )
            out = rearrange(y, "b d l -> b l d")

            # Output projection
            out = self.out_proj(out)

        # Apply attention mask to zero out padding positions if provided
        if attention_mask is not None:
            # Expand attention mask to match output dimensions
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(out)
            out = out * attention_mask_expanded.float()

        return out

    def step(self, hidden_states: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor):
        """
        Single step inference for autoregressive generation.

        This method is used during inference to process one token at a time while maintaining
        the internal state of the convolution and SSM components.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch, d_model)
            conv_state (torch.Tensor): Convolution state tensor
            ssm_state (torch.Tensor): SSM state tensor

        Returns:
            torch.Tensor: Output tensor of shape (batch, d_model)
        """
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == self.d_model
        batch_size = hidden_states.shape[0]

        # Input projection
        xz = self.in_proj(hidden_states)  # (batch, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # (batch, d_inner)

        # Conv step
        if HAS_CAUSAL_CONV1D and causal_conv1d_update is not None and x.is_cuda:
            # Note: conv1d.weight has shape (d_inner, 1, d_conv) but CUDA kernel expects (d_inner, d_conv)
            x = causal_conv1d_update(
                x,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
            )
        else:
            # Fallback implementation
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            # self.conv1d.weight has shape (d_inner, 1, d_conv), squeeze out singleton dimension
            conv_weight = self.conv1d.weight.squeeze(1)  # (d_inner, d_conv)
            x = torch.sum(conv_state * conv_weight.unsqueeze(0), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x)

        # SSM step
        x_db = self.x_proj(x)  # (batch, dt_rank + 2 * d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)  # (batch, d_inner)
        A = -torch.exp(self.A_log.float())

        # SSM computation
        dt = dt + self.dt_proj.bias
        dt = F.softplus(dt)
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
        dB = torch.einsum("bd,bn->bdn", dt, B)
        ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C.to(dtype))
        y = y + self.D.to(dtype) * x
        y = y * F.silu(z)  # (batch, d_inner)

        # Output projection
        out = self.out_proj(y)
        return out.unsqueeze(1)

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None, **kwargs):
        """
        Allocate cache for inference.

        This method pre-allocates the necessary state tensors for efficient autoregressive
        generation during inference.

        Args:
            batch_size (int): Batch size for inference
            max_seqlen (int): Maximum sequence length for inference
            dtype: Data type for the cache tensors
            **kwargs: Additional keyword arguments

        Returns:
            dict: Dictionary containing the allocated cache tensors
        """
        device = self.out_proj.weight.device
        dtype = self.out_proj.weight.dtype if dtype is None else dtype

        # Convolution state: (batch, d_inner, d_conv)
        conv_state = torch.zeros(
            batch_size, self.d_inner, self.d_conv, device=device, dtype=dtype
        )

        # SSM state: (batch, d_inner, d_state)
        ssm_state = torch.zeros(
            batch_size, self.d_inner, self.d_state, device=device, dtype=dtype
        )

        return {"conv_state": conv_state, "ssm_state": ssm_state}


class MambaResidualBlock(nn.Module):
    """
    Proper Mamba block with residual connections and feed-forward layer.

    Architecture:
    - x = x + dropout(mamba_mixer(norm(x)))
    - x = x + dropout(ffn(norm(x)))

    This matches the canonical Mamba architecture from the paper.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        layer_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.dropout = dropout

        # RMSNorm for pre-normalization in residual blocks
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

        # Mamba SSM mixer
        self.mamba_mixer = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            conv_bias=conv_bias,
            bias=bias,
            use_fast_path=use_fast_path,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )
        self._use_fast_path = self.mamba_mixer.use_fast_path

        # Feed-forward network
        d_ff = d_model * 4  # Standard expansion factor
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.SiLU(),
            nn.Linear(d_ff, d_model, bias=bias),
        )

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass with proper residual connections.

        Args:
            hidden_states: Input tensor of shape (batch, seqlen, d_model)
            attention_mask: Optional attention mask of shape (batch, seqlen)

        Returns:
            Output tensor of same shape as input
        """
        # First sublayer: norm → mamba mixer → dropout → residual
        normed = self.norm1(hidden_states)
        mamba_out = self.mamba_mixer(normed, attention_mask=attention_mask)
        hidden_states = hidden_states + self.dropout1(mamba_out)

        # Second sublayer: norm → FFN → dropout → residual
        normed = self.norm2(hidden_states)
        ffn_out = self.ffn(normed)
        hidden_states = hidden_states + self.dropout2(ffn_out)

        return hidden_states

    def step(self, hidden_states: torch.Tensor, cache: dict) -> tuple[torch.Tensor, dict]:
        """Single-token inference. Mirrors the residual structure used in forward()."""

        # First sublayer: norm → mamba mixer step → dropout → residual
        normed = self.norm1(hidden_states)
        conv_state = cache["conv_state"]
        ssm_state = cache["ssm_state"]

        mamba_out = self.mamba_mixer.step(normed, conv_state, ssm_state)
        mamba_out = mamba_out.squeeze(1)  # step() returns (batch, 1, d_model)
        hidden_states = hidden_states + self.dropout1(mamba_out)

        # Second sublayer: norm → FFN → dropout → residual
        normed = self.norm2(hidden_states)
        ffn_out = self.dropout2(self.ffn(normed))
        hidden_states = hidden_states + ffn_out

        return hidden_states, cache

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int = 1):
        """
        Allocate cache for inference (forward the call to the mamba mixer).
        """
        return self.mamba_mixer.allocate_inference_cache(batch_size, max_seqlen)

    @property
    def use_fast_path(self) -> bool:
        """Expose fast-path flag for parity with bare MambaBlock."""
        return self.mamba_mixer.use_fast_path