from __future__ import annotations

import torch
import torch.nn as nn

from .local_scan import gather_visible_tokens, scatter_visible_tokens


class ConvFFN(nn.Module):
    """Feed-forward branch with depthwise spatial mixing for [B, N, C] tokens."""

    def __init__(
        self,
        dim: int,
        expand_ratio: float = 4.0,
        dropout: float = 0.0,
        dw_kernel: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        if dw_kernel <= 0 or dw_kernel % 2 == 0:
            raise ValueError(f"dw_kernel must be a positive odd integer, got {dw_kernel}")

        hidden = int(dim * expand_ratio)
        self.dim = dim
        self.hidden = hidden
        self.expand_ratio = expand_ratio
        self.dw_kernel = int(dw_kernel)

        self.fc1 = nn.Linear(dim, hidden, bias=bias)
        self.dw_conv = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=dw_kernel,
            padding=dw_kernel // 2,
            groups=hidden,
            bias=bias,
        )
        self.norm = nn.LayerNorm(hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        ids_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, C]. N can be H*W or sparse visible-token count when ids_keep is given.
            H, W: full spatial dimensions.
            ids_keep: optional MAE visible-token indices [B, K].
        Returns:
            [B, N, C]
        """
        B, N, C = x.shape
        if C != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got C={C}")

        sparse = ids_keep is not None and N != H * W
        if sparse:
            x = scatter_visible_tokens(x, ids_keep, H, W)
            N = H * W
        if N != H * W:
            raise ValueError(f"N={N} must equal H*W={H * W} when ids_keep is not provided")

        x = self.fc1(x)
        x = x.transpose(1, 2).contiguous().view(B, self.hidden, H, W)
        x = self.dw_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        if sparse:
            x = gather_visible_tokens(x, ids_keep)
        return x
