from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_visible_tokens(x: torch.Tensor, ids_keep: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Scatter sparse visible tokens into a zero-filled full spatial grid."""
    B, _, C = x.shape
    full = x.new_zeros(B, H * W, C)
    index = ids_keep.to(device=x.device, dtype=torch.long).unsqueeze(-1).expand(-1, -1, C)
    return full.scatter(1, index, x)


def gather_visible_tokens(x: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
    """Gather visible tokens from a full spatial grid using MAE ids_keep."""
    B, _, C = x.shape
    index = ids_keep.to(device=x.device, dtype=torch.long).unsqueeze(-1).expand(B, -1, C)
    return torch.gather(x, dim=1, index=index)


class WindowPartition(nn.Module):
    """Non-overlapping window partition/reverse for flattened 2D tokens."""

    def __init__(self, window_size: int):
        super().__init__()
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self.ws = int(window_size)

    def partition(self, x: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, int]:
        """
        Args:
            x: [B, H*W, C]
            H, W: spatial dimensions
        Returns:
            windows: [B*num_windows, ws*ws, C]
            num_windows: windows per sample
        """
        B, N, C = x.shape
        ws = self.ws
        if N != H * W:
            raise ValueError(f"N={N} must equal H*W={H * W}")
        x = x.view(B, H, W, C)
        Hp = ((H + ws - 1) // ws) * ws
        Wp = ((W + ws - 1) // ws) * ws
        if Hp != H or Wp != W:
            x = F.pad(x, (0, 0, 0, Wp - W, 0, Hp - H))
            H, W = Hp, Wp
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        num_windows = (H // ws) * (W // ws)
        windows = x.view(B * num_windows, ws * ws, C)
        return windows, num_windows

    def reverse(self, windows: torch.Tensor, B: int, H: int, W: int) -> torch.Tensor:
        """Reverse windows back to [B, H*W, C]."""
        ws = self.ws
        Hp = ((H + ws - 1) // ws) * ws
        Wp = ((W + ws - 1) // ws) * ws
        C = windows.shape[-1]
        expected = B * (Hp // ws) * (Wp // ws)
        if windows.shape[0] != expected:
            raise ValueError(f"windows batch={windows.shape[0]} must equal {expected}")

        x = windows.view(B, Hp // ws, Wp // ws, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, Hp, Wp, C)
        x = x[:, :H, :W, :].contiguous()
        return x.view(B, H * W, C)


class LocalScanMamba(nn.Module):
    """Run an existing Mamba layer inside local non-overlapping spatial windows."""

    def __init__(
        self,
        dim: int,
        mamba_layer: nn.Module,
        window_size: int = 4,
        scan_direction: str = "horizontal",
    ):
        super().__init__()
        if scan_direction not in {"horizontal", "vertical"}:
            raise ValueError(f"scan_direction must be 'horizontal' or 'vertical', got {scan_direction}")
        self.dim = dim
        self.window_size = int(window_size)
        self.scan_direction = scan_direction
        self.partitioner = WindowPartition(window_size)
        self.mamba = mamba_layer

    def _prepare_sequence(self, x: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, int, int]:
        if self.scan_direction == "vertical":
            B = x.shape[0]
            x = x.view(B, H, W, self.dim).permute(0, 2, 1, 3).contiguous()
            return x.view(B, H * W, self.dim), W, H
        return x, H, W

    def _restore_direction(self, x: torch.Tensor, B: int, H_eff: int, W_eff: int) -> torch.Tensor:
        if self.scan_direction == "vertical":
            x = x.view(B, H_eff, W_eff, self.dim).permute(0, 2, 1, 3).contiguous()
            return x.view(B, H_eff * W_eff, self.dim)
        return x

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
            H, W: spatial dimensions of the full feature grid.
            ids_keep: optional MAE visible-token indices [B, K].
        Returns:
            [B, N, C] with the same visible-token shape as input.
        """
        B, N, C = x.shape
        if C != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got C={C}")

        sparse = ids_keep is not None and N != H * W
        if sparse:
            x = scatter_visible_tokens(x, ids_keep, H, W)

        x, H_eff, W_eff = self._prepare_sequence(x, H, W)
        windows, _ = self.partitioner.partition(x, H_eff, W_eff)
        out_windows = self.mamba(windows)
        out = self.partitioner.reverse(out_windows, B, H_eff, W_eff)
        out = self._restore_direction(out, B, H_eff, W_eff)

        if sparse:
            out = gather_visible_tokens(out, ids_keep)
        return out
