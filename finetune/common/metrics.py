"""Task metrics. Model-agnostic: identical code for both arms.

sklearn is used when importable (guarded), else numpy fallbacks are provided for ROC-AUC and
average precision so the CPU smoke test runs with no sklearn installed. All functions take
torch tensors or numpy arrays and return plain python floats / dicts.
"""
import numpy as np
import torch

try:  # sklearn is optional
    from sklearn.metrics import roc_auc_score as _sk_auc, average_precision_score as _sk_ap
    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float64)


# ------------------------------------------------------------------ ROC-AUC / AP fallbacks
def roc_auc(scores, labels):
    """Binary ROC-AUC. labels in {0,1}. numpy rank-statistic (Mann-Whitney U) fallback."""
    scores, labels = _to_np(scores), _to_np(labels).astype(int)
    if _HAS_SK and 0 < labels.sum() < len(labels):
        return float(_sk_auc(labels, scores))
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros(len(counts))
    np.add.at(sum_ranks, inv, ranks)
    ranks = (sum_ranks / counts)[inv]
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision(scores, labels):
    """Binary average precision (area under precision-recall, step interpolation)."""
    scores, labels = _to_np(scores), _to_np(labels).astype(int)
    if labels.sum() == 0:
        return float("nan")
    if _HAS_SK:
        return float(_sk_ap(labels, scores))
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    tp = np.cumsum(labels)
    precision = tp / np.arange(1, len(labels) + 1)
    recall_hits = labels == 1
    return float(precision[recall_hits].sum() / labels.sum())


# ------------------------------------------------------------------ identity / liveness
def accuracy_topk(output, target, ks=(1, 5)):
    """Top-k accuracy (%). Returns list aligned with `ks`."""
    maxk = min(max(ks), output.shape[1])
    batch = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = []
    for k in ks:
        kk = min(k, maxk)
        res.append(float(correct[:kk].reshape(-1).float().sum().mul_(100.0 / batch).item()))
    return res


# ------------------------------------------------------------------ attribute (multi-label)
def multilabel_metrics(logits, targets):
    """logits/targets: [N,40]. Returns dict(acc=mean per-attr acc, mAP=macro AP)."""
    logits, targets = _to_np(logits), _to_np(targets).astype(int)
    preds = (logits > 0).astype(int)  # sigmoid(x)>0.5 <=> x>0
    per_attr_acc = (preds == targets).mean(axis=0)
    aps = []
    for j in range(logits.shape[1]):
        ap = average_precision(logits[:, j], targets[:, j])
        if not np.isnan(ap):
            aps.append(ap)
    return {"acc": float(per_attr_acc.mean()),
            "mAP": float(np.mean(aps)) if aps else float("nan")}


# ------------------------------------------------------------------ verification
def verification_metrics(embA, embB, labels, far_target=1e-3, n_folds=1):
    """Cosine-sim verification metrics.

    embA/embB: [N,D] embeddings; labels: [N] in {0,1} (1 == same). Returns dict with
    auc, tar_at_far (TAR at FAR=far_target), acc (best-threshold accuracy). n_folds>1
    runs the k-fold LFW protocol (threshold chosen on train folds); default single split.
    """
    a, b = _to_np(embA), _to_np(embB)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sims = (a * b).sum(axis=1)
    labels = _to_np(labels).astype(int)

    auc = roc_auc(sims, labels)
    tar = _tar_at_far(sims, labels, far_target)

    if n_folds <= 1:
        acc = _best_thresh_acc(sims, labels)
    else:
        acc = _kfold_acc(sims, labels, n_folds)
    return {"auc": auc, "tar_at_far": tar, "acc": acc}


def _best_thresh_acc(sims, labels):
    thr = np.unique(sims)
    best = 0.0
    for t in thr:
        acc = ((sims >= t).astype(int) == labels).mean()
        best = max(best, acc)
    return float(best)


def _kfold_acc(sims, labels, n_folds):
    """LFW-style: pick threshold maximising acc on the other folds, eval on held-out fold."""
    idx = np.arange(len(sims))
    folds = np.array_split(idx, n_folds)
    accs = []
    thr_candidates = np.unique(sims)
    for i in range(n_folds):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        best_t, best_a = thr_candidates[0], -1.0
        for t in thr_candidates:
            a = ((sims[train] >= t).astype(int) == labels[train]).mean()
            if a > best_a:
                best_a, best_t = a, t
        accs.append(((sims[test] >= best_t).astype(int) == labels[test]).mean())
    return float(np.mean(accs))


def _tar_at_far(sims, labels, far_target):
    """TAR at the largest threshold whose FAR <= far_target."""
    pos, neg = sims[labels == 1], sims[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = np.sort(np.unique(sims))[::-1]
    best_tar = 0.0
    for t in thr:
        far = (neg >= t).mean()
        if far <= far_target:
            best_tar = max(best_tar, (pos >= t).mean())
    return float(best_tar)


# ------------------------------------------------------------------ liveness (PAD)
def liveness_metrics(scores, labels, threshold=0.5):
    """PAD metrics; attack = positive (label 1). scores = P(attack).

    Convention: predict attack if score >= threshold (default 0.5). Also reports the
    EER-based operating point. Returns APCER, BPCER, ACER, HTER, plus eer/auc.
      APCER = frac of attacks classified as bonafide (missed attacks)
      BPCER = frac of bonafide classified as attack (false alarms)
      ACER  = (APCER + BPCER) / 2
      HTER  = HTER at the EER threshold (== EER for a single split)
    """
    scores = _to_np(scores)
    labels = _to_np(labels).astype(int)
    attack, bona = labels == 1, labels == 0
    n_att, n_bon = int(attack.sum()), int(bona.sum())

    pred_attack = scores >= threshold
    apcer = float((~pred_attack[attack]).mean()) if n_att else float("nan")
    bpcer = float((pred_attack[bona]).mean()) if n_bon else float("nan")
    acer = float(np.nanmean([apcer, bpcer]))

    # EER threshold sweep
    eer, hter = _eer_hter(scores, attack, bona)
    auc = roc_auc(scores, labels)
    return {"apcer": apcer, "bpcer": bpcer, "acer": acer, "hter": hter,
            "eer": eer, "auc": auc}


def _eer_hter(scores, attack, bona):
    if attack.sum() == 0 or bona.sum() == 0:
        return float("nan"), float("nan")
    thr = np.sort(np.unique(scores))
    best_gap, eer, hter = 1e9, float("nan"), float("nan")
    for t in thr:
        pred = scores >= t
        far = pred[bona].mean()            # bonafide accepted as attack
        frr = (~pred[attack]).mean()       # attack rejected as bonafide
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap, eer, hter = gap, (far + frr) / 2.0, (far + frr) / 2.0
    return float(eer), float(hter)
