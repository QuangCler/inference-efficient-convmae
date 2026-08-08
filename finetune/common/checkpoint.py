"""Load a ConvMAE *pretrain* checkpoint into a finetune encoder.

Generalises the checkpoint-loading logic from the original `main_finetune.py` so it works
for BOTH the baseline and the Ghost encoders and is robust to the classification head being
resized per dataset (identity / attribute / liveness all have different #outputs).

The pretrain checkpoints (../models/*.pt) hold the full MAE (encoder + decoder + multi-scale
fusion) under key 'model'. At finetune we keep only the encoder weights; decoder/mask_token/
stage*_output_decode/patch_embed(alias) are expected-and-ignored. The only weights that must
be freshly initialised are the new head and, under global-pool, fc_norm.
"""
import torch
from timm.models.layers import trunc_normal_


# Keys we deliberately do NOT expect to find in a pretrain checkpoint (they are new at finetune).
_EXPECTED_MISSING_GLOBAL_POOL = {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"}
_EXPECTED_MISSING_CLS_TOKEN = {"head.weight", "head.bias"}


def load_pretrained_encoder(model, ckpt_path, global_pool=True, strict_check=True, verbose=True):
    """Load pretrained ConvMAE encoder weights into `model` (in place).

    Args:
        model: a finetune encoder (baseline or ghost) *before* DDP wrapping.
        ckpt_path: path to a ../models/*.pt pretrain checkpoint (top-level key 'model').
        global_pool: must match how the encoder was built (adds fc_norm to the expected-missing set).
        strict_check: if True, assert the set of missing keys is exactly the head (+fc_norm) —
            this catches silent architecture/key mismatches (e.g. wrong encoder for a checkpoint).
        verbose: print the load message.

    Returns:
        the torch load_state_dict result (with .missing_keys / .unexpected_keys).
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint_model = checkpoint["model"] if "model" in checkpoint else checkpoint

    # Drop a head whose shape does not match the (per-dataset) target head.
    state_dict = model.state_dict()
    for k in ("head.weight", "head.bias"):
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            if verbose:
                print(f"[checkpoint] removing mismatched key {k} from pretrain checkpoint")
            del checkpoint_model[k]

    msg = model.load_state_dict(checkpoint_model, strict=False)
    if verbose:
        print(f"[checkpoint] loaded {ckpt_path}")
        print(f"[checkpoint] missing={sorted(msg.missing_keys)}  #unexpected={len(msg.unexpected_keys)}")

    if strict_check:
        expected = _EXPECTED_MISSING_GLOBAL_POOL if global_pool else _EXPECTED_MISSING_CLS_TOKEN
        missing = set(msg.missing_keys)
        # The head may legitimately be absent from `missing` if we kept a same-shape head, so
        # only require that nothing OUTSIDE the expected set is missing.
        unexpected_missing = missing - expected
        assert not unexpected_missing, (
            f"Unexpected missing keys when loading {ckpt_path}: {sorted(unexpected_missing)}. "
            "This usually means the encoder architecture does not match the checkpoint."
        )

    # Freshly initialise the classification head (matches original main_finetune.py).
    if hasattr(model, "head") and isinstance(model.head, torch.nn.Linear):
        trunc_normal_(model.head.weight, std=2e-5)
        if model.head.bias is not None:
            torch.nn.init.zeros_(model.head.bias)

    return msg
