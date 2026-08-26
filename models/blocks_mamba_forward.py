import warnings

import torch
import torch.nn as nn

from .conv_ffn import ConvFFN
from .local_scan import LocalScanMamba

try:
    from mamba_ssm import Mamba2 as Mamba
except ImportError:
    Mamba = None
    warnings.warn(
        "mamba_ssm not found. Install with: pip install causal-conv1d mamba-ssm",
        ImportWarning,
        stacklevel=2,
    )


class MambaBlock(nn.Module):
    supports_s1_s2 = True

    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
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
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.local_scan = (
            LocalScanMamba(
                dim=dim,
                mamba_layer=self.mamba,
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
        if torch.onnx.is_in_onnx_export():
            # Mamba SSM is not ONNX-exportable; approximate as identity.
            # ConvFFN IS exportable, so we keep it for a closer approximation.
            x = x + self.norm(x)
            if self.use_convffn:
                H, W = self._resolve_hw(x, H, W, ids_keep)
                x = x + self.ffn(self.norm2(x), H, W, ids_keep=ids_keep)
            return x

        if self.use_local_scan or self.use_convffn:
            H, W = self._resolve_hw(x, H, W, ids_keep)

        if self.use_local_scan:
            x = x + self.local_scan(self.norm(x), H, W, ids_keep=ids_keep)
        else:
            x = x + self.mamba(self.norm(x))

        if self.use_convffn:
            x = x + self.ffn(self.norm2(x), H, W, ids_keep=ids_keep)
        return x
