"""Optimizer construction with layer-wise LR decay — architecture-agnostic.

The layer-id assignment (`get_layer_id_for_convvit`) buckets patch-embeds + stages 1/2 into
layer 0, each stage-3 Transformer block into its own layer, and head/norm into the last layer.
This is identical for the baseline and Ghost encoders (both name their stacks blocks1/2/3), so
layer-wise decay is applied the same way to both — a fairness requirement.

Mirrors util/lr_decay.py but lives here so the finetune pipeline is self-contained and does not
inherit the repo's timm-0.3.2 assumptions.
"""
import torch


def get_layer_id_for_convvit(name: str, num_layers: int) -> int:
    if name in ("cls_token", "pos_embed"):
        return 0
    if name.startswith("patch_embed"):      # patch_embed1..4
        return 0
    if name.startswith("blocks1") or name.startswith("blocks2"):
        return 0                              # conv/ghost stages = first layer group
    if name.startswith("blocks3"):
        return int(name.split(".")[1]) + 1    # each transformer block is its own layer
    return num_layers                         # norm / fc_norm / head


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=(), layer_decay=0.65):
    """BEiT/ELECTRA-style layer-wise LR decay parameter groups."""
    no_weight_decay_list = set(no_weight_decay_list)
    num_layers = len(model.blocks3) + 1
    layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n in no_weight_decay_list:   # biases, norms, pos_embed → no decay
            g_decay, this_decay = "no_decay", 0.0
        else:
            g_decay, this_decay = "decay", weight_decay
        layer_id = get_layer_id_for_convvit(n, num_layers)
        key = f"layer_{layer_id}_{g_decay}"
        if key not in groups:
            groups[key] = {"lr_scale": layer_scales[layer_id], "weight_decay": this_decay, "params": []}
        groups[key]["params"].append(p)
    return list(groups.values())


def build_optimizer(model, cfg, world_size: int):
    """AdamW with layer-wise LR decay; lr from the fair eff-batch rule."""
    no_wd = model.no_weight_decay() if hasattr(model, "no_weight_decay") else ()
    groups = param_groups_lrd(model, weight_decay=cfg.weight_decay,
                              no_weight_decay_list=no_wd, layer_decay=cfg.layer_decay)
    lr = cfg.resolved_lr(world_size)
    optimizer = torch.optim.AdamW(groups, lr=lr)
    return optimizer, lr
