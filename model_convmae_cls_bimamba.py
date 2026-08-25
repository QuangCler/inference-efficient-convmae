import torch
import torch.nn as nn
from model_convmae_bimamba import convmae_bimamba


class ConvMAE_BiMamba_Cls(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        use_local_scan: bool = False,
        local_scan_window_size: int = 4,
        scan_direction: str = "horizontal",
        use_convffn: bool = False,
        convffn_expand_ratio: float = 4.0,
        convffn_dw_kernel: int = 3,
    ):
        super().__init__()

        mae = convmae_bimamba(
            use_local_scan=use_local_scan,
            local_scan_window_size=local_scan_window_size,
            scan_direction=scan_direction,
            use_convffn=use_convffn,
            convffn_expand_ratio=convffn_expand_ratio,
            convffn_dw_kernel=convffn_dw_kernel,
        )

        self.patch_embed1 = mae.patch_embed1
        self.patch_embed2 = mae.patch_embed2
        self.patch_embed3 = mae.patch_embed3
        self.patch_embed4 = mae.patch_embed4

        self.stage1_output_decode = mae.stage1_output_decode
        self.stage2_output_decode = mae.stage2_output_decode

        self.pos_embed = mae.pos_embed

        self.blocks1 = mae.blocks1
        self.blocks2 = mae.blocks2
        self.blocks3 = mae.blocks3

        self.norm = mae.norm

        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x, None)   # Ghost block

        stage1_embed = self.stage1_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x, None)

        stage2_embed = self.stage2_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed3(x)
        H3, W3 = x.shape[-2], x.shape[-1]
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)
        x = x + self.pos_embed

        for blk in self.blocks3:
            if getattr(blk, "supports_s1_s2", False):
                x = blk(x, H3, W3)
            else:
                x = blk(x)

        x = x + stage1_embed + stage2_embed

        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


def convmae_bimamba_cls(**kwargs):
    return ConvMAE_BiMamba_Cls(**kwargs)
