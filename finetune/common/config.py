"""Shared fair-comparison recipe.

`BASE` is the single source of truth for every knob that must be identical between the baseline
and Ghost runs. Per-dataset overrides (`DATASETS`) set only task-shaped things (task mode,
#classes, epochs, input size) and are also shared across the two models. A concrete run is
`resolve(dataset, model, seed, **cli)`; the ONLY per-model differences allowed are `model` and
`finetune` (checkpoint path), plus `batch_size`/`accum_iter` as long as `eff_bs` stays constant.

Keeping this in one importable module (not copied YAML) is what guarantees the two models see
the same recipe — see EXPERIMENT_DESIGN.md §2/§7.
"""
from dataclasses import dataclass, field, replace
from typing import Optional


# --------------------------------------------------------------------------------------
# BASE recipe — held identical across both models and (except task-shape) across datasets.
# Values follow the original ConvMAE ImageNet finetune recipe (FINETUNE.md).
# --------------------------------------------------------------------------------------
@dataclass
class FairConfig:
    # --- identity of the run ---
    model: str = "convvit_base_patch16"     # or "convvit_ghost_base_patch16"
    finetune: str = ""                       # pretrain checkpoint path
    dataset: str = ""
    task: str = "identity"                   # identity | attribute | verification | liveness
    seed: int = 0

    # --- data ---
    data_path: str = ""
    input_size: int = 224
    nb_classes: int = 1000                   # #ids / #attrs / 2 (liveness), set per dataset

    # --- fair batch/optim (eff_bs is the invariant, NOT batch_size) ---
    eff_batch_size: int = 1024               # batch_size * accum_iter * world_size — CONSTANT
    batch_size: int = 128                    # per-GPU; adjusted to fit VRAM
    accum_iter: int = 1
    epochs: int = 100
    warmup_epochs: int = 5
    blr: float = 5e-4                        # lr = blr * eff_bs / 256
    lr: Optional[float] = None
    min_lr: float = 1e-6
    layer_decay: float = 0.65
    weight_decay: float = 0.05
    clip_grad: Optional[float] = None

    # --- regularisation / augmentation (identical for both models) ---
    drop_path: float = 0.1
    smoothing: float = 0.1
    aa: str = "rand-m9-mstd0.5-inc1"
    reprob: float = 0.25
    remode: str = "pixel"
    recount: int = 1
    mixup: float = 0.8
    cutmix: float = 1.0
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    mixup_mode: str = "batch"
    color_jitter: Optional[float] = None

    # --- model head ---
    global_pool: bool = True

    # --- runtime ---
    amp_dtype: str = "bf16"                  # bf16 on 5090; smoke path forces fp32
    num_workers: int = 4
    use_dali: bool = False
    dali_device: str = "mixed"               # "mixed" (GPU) | "cpu" fallback (sm_120)
    output_dir: str = ""
    smoke: bool = False

    def resolved_lr(self, world_size: int) -> float:
        if self.lr is not None:
            return self.lr
        return self.blr * (self.batch_size * self.accum_iter * world_size) / 256


# --------------------------------------------------------------------------------------
# Per-dataset overrides — task shape only; shared by baseline & ghost.
# (epochs scaled to dataset size; large ID sets get metric-friendly heads.)
# --------------------------------------------------------------------------------------
DATASETS = {
    "casia":    dict(task="identity",     nb_classes=10575, epochs=40,  input_size=224),
    "arface":   dict(task="identity",     nb_classes=126,   epochs=100, input_size=224),
    "scface":   dict(task="identity",     nb_classes=130,   epochs=100, input_size=224),
    "multipie": dict(task="identity",     nb_classes=337,   epochs=60,  input_size=224),
    "celeba":   dict(task="attribute",    nb_classes=40,    epochs=30,  input_size=224,
                     mixup=0.0, cutmix=0.0, smoothing=0.0),   # multi-label: no mixup/smoothing
    "lfw":      dict(task="verification", nb_classes=5749,  epochs=50,  input_size=224,
                     mixup=0.0, cutmix=0.0),
    "3dmad":    dict(task="liveness",     nb_classes=2,     epochs=30,  input_size=224,
                     mixup=0.0, cutmix=0.0, smoothing=0.0),
}

# The model arms and their pretrain checkpoints (relative to repo root). baseline/ghost run on
# CPU/any GPU; the two mamba arms are CUDA-only (mamba_ssm + causal_conv1d) and use the S1-S2 head.
MODELS = {
    "baseline":     dict(model="convvit_base_patch16",             finetune="models/convmae_base_pretrain_epoch300.pt"),
    "ghost":        dict(model="convvit_ghost_base_patch16",       finetune="models/allghost_epoch_300.pt"),
    "bimamba":      dict(model="convmae_bimamba_base_patch16",     finetune="models/ghost_bimamba_pretrain_epoch300.pt"),
    "forwardmamba": dict(model="convmae_forwardmamba_base_patch16", finetune="models/ghost_forwardmamba_pretrain_epoch300.pt"),
}

SEEDS = [0, 1, 2]


def resolve(dataset: str, model: str, seed: int = 0, **overrides) -> FairConfig:
    """Build a concrete FairConfig for (dataset, model, seed), applying dataset + model presets.

    `model` is the fair-comparison arm name ("baseline" | "ghost"). CLI `**overrides` win last
    (e.g. batch_size for VRAM, output_dir, smoke). eff_batch_size must not be overridden per-model.
    """
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset '{dataset}', have {list(DATASETS)}")
    if model not in MODELS:
        raise KeyError(f"unknown model arm '{model}', have {list(MODELS)}")
    cfg = FairConfig()
    cfg = replace(cfg, dataset=dataset, seed=seed, **DATASETS[dataset])
    cfg = replace(cfg, **MODELS[model])
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg
