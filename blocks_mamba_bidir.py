import warnings

import torch
import torch.nn as nn

from conv_ffn import ConvFFN
from local_scan import LocalScanMamba

try:
    from mamba_ssm import Mamba2 as Mamba
except ImportError:
    Mamba = None
    warnings.warn(
        "mamba_ssm not found. Install with: pip install causal-conv1d mamba-ssm",
        ImportWarning,
        stacklevel=2,
    )


class _ReverseSequenceMamba(nn.Module):
    def __init__(self, mamba_layer: nn.Module):
        super().__init__()
        self.mamba = mamba_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flip(x, dims=[1])
        x = self.mamba(x)
        return torch.flip(x, dims=[1])


class BiMambaBlock(nn.Module):
    """Bidirectional Mamba block for [B, N, C] tokens."""

    supports_s1_s2 = True

    def __init__(
        self,
        dim: int,
        use_local_scan: bool = False,
        local_scan_window_size: int = 4,
        scan_direction: str = "horizontal",
        use_convffn: bool = False,
        convffn_expand_ratio: float = 4.0,
        convffn_dw_kernel: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert Mamba is not None, "mamba_ssm is not installed!"

        self.dim = dim
        self.use_local_scan = use_local_scan
        self.use_convffn = use_convffn
        self.norm = nn.LayerNorm(dim)

        self.mamba_fwd = Mamba(d_model=dim, d_state=64, d_conv=4, expand=2)
        self.mamba_bwd = Mamba(d_model=dim, d_state=64, d_conv=4, expand=2)
        # S1 Local Scan: when enabled, Mamba processes tokens within each
        # non-overlapping spatial window independently.  Bidirectionality is
        # achieved *within each window* (intra-window forward + intra-window
        # backward) rather than reversing the full global sequence.  This
        # preserves the locality principle of windowed attention while still
        # capturing both scan directions inside each local region.
        self.local_scan_fwd = (
            LocalScanMamba(
                dim=dim,
                mamba_layer=self.mamba_fwd,
                window_size=local_scan_window_size,
                scan_direction=scan_direction,
            )
            if use_local_scan
            else None
        )
        self.local_scan_bwd = (
            LocalScanMamba(
                dim=dim,
                mamba_layer=_ReverseSequenceMamba(self.mamba_bwd),
                window_size=local_scan_window_size,
                scan_direction=scan_direction,
            )
            if use_local_scan
            else None
        )
        self.norm2 = nn.LayerNorm(dim) if use_convffn else None
        self.ffn = (
            ConvFFN(
                dim=dim,
                expand_ratio=convffn_expand_ratio,
                dropout=dropout,
                dw_kernel=convffn_dw_kernel,
            )
            if use_convffn
            else None
        )

    @staticmethod
    def _resolve_hw(
        x: torch.Tensor,
        H: int | None,
        W: int | None,
        ids_keep: torch.Tensor | None,
    ) -> tuple[int, int]:
        if H is not None and W is not None:
            return H, W
        if ids_keep is not None:
            raise ValueError("H and W are required when ids_keep is provided")
        side = int(x.shape[1] ** 0.5)
        if side * side != x.shape[1]:
            raise ValueError(f"Cannot infer square H/W from N={x.shape[1]}")
        return side, side

    def forward(
        self,
        x: torch.Tensor,
        H: int | None = None,
        W: int | None = None,
        ids_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        if self.use_local_scan or self.use_convffn:
            H, W = self._resolve_hw(x, H, W, ids_keep)

        if torch.onnx.is_in_onnx_export():
            # Mamba SSM is not ONNX-exportable; approximate as identity.
            # ConvFFN IS exportable, so we keep it for a closer approximation.
            x = x + residual
            if self.use_convffn:
                x = x + self.ffn(self.norm2(x), H, W, ids_keep=ids_keep)
            return x

        if self.use_local_scan:
            y_fwd = self.local_scan_fwd(x, H, W, ids_keep=ids_keep)
            y_bwd = self.local_scan_bwd(x, H, W, ids_keep=ids_keep)
        else:
            y_fwd = self.mamba_fwd(x)
            x_rev = torch.flip(x, dims=[1])
            y_bwd = self.mamba_bwd(x_rev)
            y_bwd = torch.flip(y_bwd, dims=[1])

        x = y_fwd + y_bwd + residual
        if self.use_convffn:
            x = x + self.ffn(self.norm2(x), H, W, ids_keep=ids_keep)
        return x
