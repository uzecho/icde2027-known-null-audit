#!/usr/bin/env python3
"""Shared probing machinery for the viseme-ambiguity experiments.

Six scripts grew near-identical copies of this (04/11, 05/03, 06/01, 08/01, 09/01, 09/03).
Two of those copies carried control bugs that produced confident wrong answers. This module
is the single version, and the controls assert their own preconditions rather than relying
on whoever writes the next script remembering them.

Typical use:

    from probelib import (load_frames, speaker_folds, pair_list, paired_sample,
                          gather, fit_folds, boot, assembly_check, permute_labels)

    F = load_frames()
    folds = speaker_folds(F, n=3, seed=0)
    for pa, pb in pair_list(F, n_other=40):
        sel, y = paired_sample(F, pa, pb, per_class=4000, seed=0)
        X = gather(F, "CTXONLY.npy", sel)
        acc = fit_folds(X, y, folds[sel])
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_

AMAP = str(Path(__file__).resolve().parent / "lrs2")
if AMAP not in sys.path:
    sys.path.insert(0, AMAP)
from common import same_viseme                                    # noqa: E402

_LRS2_HINT = (
    "The features behind this path are derived from the licence-restricted LRS2 "
    "corpus and are not redistributed with this artifact. See "
    "LRS2_REGENERATION.md to rebuild them from your own licensed copy, then set "
    "$LRS2_DERIVED (and $LRS2_ROOT for the corpus itself).")


def _lrs2(sub, _var="LRS2_DERIVED"):
    """Path under the LRS2-derived feature tree named by $LRS2_DERIVED."""
    import os
    import sys as _s
    from pathlib import Path as _P
    root = os.environ.get(_var)
    if not root:
        if not getattr(_lrs2, "_warned_" + _var, False):
            print(f"  note: ${_var} is not set. {_LRS2_HINT}", file=_s.stderr)
            setattr(_lrs2, "_warned_" + _var, True)
        root = f"${_var}-NOT-SET"
    return _P(root) / sub


def _lrs2_corpus(sub):
    """Path under the LRS2 corpus itself, named by $LRS2_ROOT."""
    return _lrs2(sub, _var="LRS2_ROOT")

FRAMES = _lrs2("ambiguity_map/frames")
OFFSET_IDX = 1
# Paper 3's published same-viseme means. Any new assembly of these arms must reproduce
# them; if it does not, the arrays are not what they claim to be.
PAPER3 = {"X.npy": 0.834, "CTXONLY.npy": 0.771, "LOCAL0.npy": 0.644,
          "LOCAL2.npy": 0.702, "LOCAL5.npy": 0.749, "SWAP.npy": 0.627,
          "MASKGEOM.npy": 0.602}
PAPER3_TOL = 0.035
FLOOR_PAIRED = 0.4978          # measured permutation floor, paired sampling


# ------------------------------------------------------------------ loading
class Frames:
    """meta.json / y / utt / kept, loaded once."""

    def __init__(self, root: Path = FRAMES):
        self.root = root
        meta = json.load(open(root / "meta.json"))
        self.utts, self.phones = meta["utts"], meta["phones"]
        self.offset = meta["offsets"][len(meta["offsets"]) // 2]
        self.y = np.load(root / "y.npy")[:, OFFSET_IDX]
        self.utt = np.load(root / "utt.npy")
        self.kept = np.load(root / "kept.npy")
        self.speaker = np.array([u["speaker"] for u in self.utts])
        self.bounds = np.searchsorted(self.utt, np.arange(len(self.utts) + 1))

    def __len__(self):
        return len(self.y)


def load_frames(root: Path = FRAMES) -> Frames:
    return Frames(root)


def speaker_folds(F: Frames, n: int = 3, seed: int = 0) -> np.ndarray:
    """Per-ROW fold id, assigned by speaker so no talker spans the split."""
    rng = np.random.default_rng(seed)
    uq = np.unique(F.speaker)
    m = {s: i for s, i in zip(uq, rng.integers(0, n, len(uq)))}
    return np.array([m[s] for s in F.speaker])[F.utt]


def pair_list(F: Frames, n_other: int = 40, seed: int = 0):
    """The 10 same-viseme pairs plus a random control sample, as (pa, pb, group)."""
    rng = np.random.default_rng(seed)
    PH = F.phones
    sv = [(x, y) for i, x in enumerate(PH) for y in PH[i + 1:] if same_viseme(x, y)]
    ov = [(x, y) for i, x in enumerate(PH) for y in PH[i + 1:] if not same_viseme(x, y)]
    ov = [ov[i] for i in rng.choice(len(ov), min(n_other, len(ov)), replace=False)]
    return ([(a, b, "same") for a, b in sv] + [(a, b, "other") for a, b in ov])


# ------------------------------------------------------------------ sampling
def paired_sample(F: Frames, pa: str, pb: str, per_class: int = 4000,
                  seed: int = 0, mask: np.ndarray | None = None):
    """Draw both phones from the SAME utterances in equal numbers.

    Without this, clip identity predicts the label and the permutation floor sits at 0.60
    rather than 0.50. Returns (sel, y) with sel sorted, or (None, None) if too few rows.
    """
    rng = np.random.default_rng(seed)
    ia, ib = F.phones.index(pa), F.phones.index(pb)
    keep = (F.y == ia) | (F.y == ib)
    if mask is not None:
        keep &= mask
    idx = np.nonzero(keep)[0]
    if len(idx) < 400:
        return None, None
    # utt is non-decreasing and idx is sorted, so utterance blocks are contiguous.
    # Masking idx once per utterance is O(n_utts * len(idx)) and does not finish.
    uu = F.utt[idx]
    starts = np.flatnonzero(np.r_[True, uu[1:] != uu[:-1]])
    ends = np.r_[starts[1:], len(uu)]
    yi = F.y[idx]
    sel = []
    for s, e in zip(starts, ends):
        blk, yb = idx[s:e], yi[s:e]
        ya, yb2 = blk[yb == ia], blk[yb == ib]
        c = min(len(ya), len(yb2))
        if c:
            sel.append(ya[rng.permutation(len(ya))[:c]])
            sel.append(yb2[rng.permutation(len(yb2))[:c]])
    if not sel:
        return None, None
    sel = np.sort(np.concatenate(sel))
    if len(sel) < 400:
        return None, None
    if len(sel) > 2 * per_class:
        sel = np.sort(rng.choice(sel, 2 * per_class, replace=False))
    return sel, (F.y[sel] == ib).astype(np.int64)


def gather(F: Frames, arm: str, sel: np.ndarray, chunk: int = 100_000) -> np.ndarray:
    """Read `sel` rows of an arm by ONE sequential pass, not scattered reads.

    Fancy-indexing a memmapped arm costs ~17 s per pair on this NAS against ~22 s for a
    whole sequential pass. Call once per arm with the union of all pairs' rows when probing
    many pairs; see `gather_union`.
    """
    src = np.load(F.root / arm, mmap_mode="r")
    out = np.empty((len(sel), src.shape[1]), np.float16)
    order = np.argsort(sel)
    s_sorted = sel[order]
    for s0 in range(0, len(src), chunk):
        e0 = min(s0 + chunk, len(src))
        i0, i1 = np.searchsorted(s_sorted, s0), np.searchsorted(s_sorted, e0)
        if i1 > i0:
            out[order[i0:i1]] = np.asarray(src[s0:e0])[s_sorted[i0:i1] - s0]
    return out


def gather_union(F: Frames, arm: str, union: np.ndarray, chunk: int = 100_000):
    """One sequential pass covering every pair's rows at once. `union` must be sorted."""
    return gather(F, arm, union, chunk)


def phone_positions(F: Frames, align: Path | None = None):
    """Map every frame row to its position in its utterance's phone sequence.

    Returns (seqs, row_pos, row_seq): `seqs[u]` is the phone-id sequence for utterance u,
    `row_pos[r]` the index of the phone owning row r (-1 if none), `row_seq[r]` its utterance.

    Needed by anything that reasons about neighbouring PHONES rather than neighbouring
    frames -- the phonotactic controls, and any position-in-utterance restriction.
    """
    from common import FPS, WORK as AWORK, read_jsonl
    align = align or (AWORK / "align_mfa.jsonl")
    ali = {r["uid"]: r["phones"] for r in read_jsonl(align)}
    pid = {p: i for i, p in enumerate(F.phones)}
    seqs = []
    row_pos = np.full(len(F), -1, np.int32)
    row_seq = np.full(len(F), -1, np.int32)
    for ui, u in enumerate(F.utts):
        ph = ali.get(u["uid"], [])
        seqs.append(np.array([pid.get(p, -1) for p, _, _ in ph], np.int32))
        lo, hi = F.bounds[ui], F.bounds[ui + 1]
        if hi <= lo or not ph:
            continue
        t = (F.kept[lo:hi] + F.offset) / FPS
        st = np.array([x[1] for x in ph])
        en = np.array([x[2] for x in ph])
        g = np.searchsorted(st, t, side="right") - 1          # boundaries are sorted
        g2 = np.clip(g, 0, len(ph) - 1)
        ok = (g >= 0) & (g < len(ph)) & (t >= st[g2]) & (t < en[g2])
        row_pos[lo:hi] = np.where(ok, g2, -1)
        row_seq[lo:hi] = ui
    return seqs, row_pos, row_seq


# -------------------------------------------------------------------- probe
def fit(Xtr, ytr, Xte, yte, l2: float = 1.0, iters: int = 200) -> float:
    """Balanced accuracy of an L2 logistic probe. The project's standard readout."""
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = torch.zeros(Xtr.shape[1], 1, device=Xtr.device, requires_grad=True)
    b = torch.zeros(1, device=Xtr.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, history_size=10,
                            tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")
    yt = ytr.float().unsqueeze(1)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = F_.binary_cross_entropy_with_logits(Xtr @ w + b, yt) \
            + (0.5 / l2 / len(Xtr)) * (w * w).sum()
        loss.backward()
        return loss

    opt.step(step)
    with torch.no_grad():
        pred = ((Xte @ w + b).squeeze(1) > 0).long()
        return float(np.mean([(pred[yte == c] == c).float().mean().item()
                              if (yte == c).any() else 0.5 for c in (0, 1)]))


def fit_folds(X, y, fold, n_folds: int = 3, min_fold: int = 120, device="cpu"):
    """Per-fold balanced accuracy; NaN for cells too small or single-class."""
    dev = torch.device(device)
    Xt = torch.as_tensor(np.asarray(X, np.float32)).to(dev)
    yt = torch.as_tensor(np.asarray(y, np.int64)).to(dev)
    out = []
    for f in range(n_folds):
        te = torch.as_tensor(fold == f).to(dev)
        tr = torch.as_tensor(fold != f).to(dev)
        yb = yt[te]
        if int(te.sum()) < min_fold or int(tr.sum()) < min_fold \
                or yb.sum() == 0 or yb.sum() == len(yb):
            out.append(float("nan"))
            continue
        out.append(fit(Xt[tr], yt[tr], Xt[te], yt[te]))
    return out


# ----------------------------------------------------------------- controls
def shuffle_rows(X, seed: int = 0):
    """Shuffle a feature block's OWN rows. The correct null for a candidate block.

    Identical covariance, marginals and effective rank; zero frame-to-label information.
    Do NOT substitute Gaussian noise -- it has ~100x the effective rank of a real block,
    so beating it only means diluting the real features less. That produced three false
    positives in 05_frontend_ceiling before it was caught.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    return X[rng.permutation(len(X))]


def effective_rank(X, eps: float = 1e-12) -> float:
    """Participation ratio of the singular values. Use it to check a candidate block and
    its control are comparable before trusting any difference between them."""
    X = np.asarray(X, np.float64)
    X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False) ** 2
    s = s[s > eps]
    return float(s.sum() ** 2 / (s * s).sum()) if len(s) else 0.0


def permute_labels(y, groups=None, seed: int = 0, min_per_group: float = 2.0):
    """Permutation floor for labels.

    Within-group shuffling is correct only when groups are densely populated. Below about
    two items per group the permutation is mostly the identity and the "floor" reports the
    UNSHUFFLED signal -- that produced a fake 0.623 floor in 09 where the real floor was
    0.497. This refuses to do it silently.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).copy()
    if groups is None:
        return y[rng.permutation(len(y))]
    groups = np.asarray(groups)
    density = len(y) / max(len(np.unique(groups)), 1)
    if density < min_per_group:
        warnings.warn(
            f"within-group shuffle is degenerate at {density:.2f} items/group "
            f"(< {min_per_group}); falling back to a GLOBAL shuffle. "
            f"Report the floor you actually measured.", stacklevel=2)
        return y[rng.permutation(len(y))]
    for g in np.unique(groups):
        m = groups == g
        y[m] = y[m][rng.permutation(int(m.sum()))]
    return y


# --------------------------------------------------------------- statistics
def boot(d, n: int = 20000, seed: int = 0):
    """Bootstrap mean and 95% CI over the unit of analysis (usually pairs)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if not len(d):
        return float("nan"), float("nan"), float("nan")
    m = rng.choice(d, (n, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def assembly_check(means: dict, tol: float = PAPER3_TOL, strict: bool = True):
    """Verify a fresh assembly reproduces paper 3 before any new number is read.

    `means` maps arm filename -> measured same-viseme mean. Returns the list of mismatches;
    raises when strict. Paper 3's confusability matrix was once silently mis-ordered against
    the BRAVEn CTC head -- this check costs nothing and would have caught it.
    """
    bad = []
    for arm, got in means.items():
        want = PAPER3.get(arm)
        if want is None:
            continue
        flag = abs(got - want) > tol
        print(f"  {arm:<16s} ours {got:.3f}   paper3 {want:.3f}"
              + ("   <-- MISMATCH" if flag else ""))
        if flag:
            bad.append((arm, got, want))
    if bad and strict:
        raise SystemExit(
            f"REFUSING: {len(bad)} arm(s) disagree with paper 3 by more than {tol}. "
            f"The arrays are not what they claim to be.")
    return bad
