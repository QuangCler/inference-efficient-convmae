"""Task-keyed criterion factory (loss only; the head is the model's built-in `model.head`).

Mirrors main_finetune.py's criterion selection for identity, and adds the attribute/
verification/liveness losses. There are NO parallel heads: build_model already sizes
`model.head = Linear(768, nb_classes)` per dataset (40 for attribute, #ids for identity/
verification-proxy, 2 for liveness). This factory only picks the loss and declares the
eval mode; the actual eval math lives in engine.evaluate / metrics.py.

Fairness: keyed only on cfg.task — no model-name branch.
"""
import torch.nn as nn

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


# eval mode per task: how engine.evaluate should score the model.
EVAL_MODE = {
    "identity": "logits_topk",
    "liveness": "liveness",
    "attribute": "multilabel",
    "verification": "embedding",
}


def build_criterion(cfg, mixup_active: bool):
    """Return the training criterion for cfg.task (mirrors main_finetune.py for identity)."""
    task = cfg.task
    if task in ("identity", "verification"):
        # verification trains an identity-softmax proxy -> same CE family as identity.
        if mixup_active:
            return SoftTargetCrossEntropy()
        if cfg.smoothing > 0:
            return LabelSmoothingCrossEntropy(smoothing=cfg.smoothing)
        return nn.CrossEntropyLoss()
    if task == "liveness":
        # 2-class CE; smoothing off by default for this task (see config override).
        if cfg.smoothing > 0:
            return LabelSmoothingCrossEntropy(smoothing=cfg.smoothing)
        return nn.CrossEntropyLoss()
    if task == "attribute":
        return nn.BCEWithLogitsLoss()
    raise KeyError(f"unknown task '{task}'")


def eval_mode(cfg):
    return EVAL_MODE[cfg.task]


def mixup_allowed(cfg):
    """Mixup/cutmix only make sense for the identity single-label task."""
    return cfg.task == "identity"
