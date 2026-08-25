import torch
import torch.nn as nn
from functools import partial

from model_convmae_baseline import convmae_baseline


class ConvMAE_Baseline_Cls(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        mae = convmae_baseline()

        # ===== Encoder =====
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

        # ===== Classifier =====
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x)

        stage1_embed = self.stage1_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)

        stage2_embed = self.stage2_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed3(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)
        x = x + self.pos_embed

        for blk in self.blocks3:
            x = blk(x)

        x = x + stage1_embed + stage2_embed

        x = self.norm(x)

        x = x.mean(dim=1)  # global pooling
        return self.head(x)


def convmae_baseline_cls(**kwargs):
    return ConvMAE_Baseline_Cls(**kwargs)