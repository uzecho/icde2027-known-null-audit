"""Shared paths, identifiers and the frame<->time mapping for the ambiguity map.

The goal of this pipeline: for every pair of phonemes, measure how separable the
pair is from the VIDEO encoder features, and how separable it is from AUDIO of
the same clips. The difference between the two is the information the visual
pathway is missing but that exists in the signal.

Frame mapping, measured (not assumed) on 8 clips, exact on all of them:

    T_feat == floor(duration_seconds * 25) - 1

so the encoder drops one frame to convolution padding. A feature frame therefore
sits slightly later than its naive index, and since a phoneme lasts only 2-5
frames at 25 fps a one-frame error is not negligible. FRAME_OFFSET makes that
explicit and 05_probe.py sweeps it -- if the map moves under a +/-1 frame shift,
the map is not trustworthy.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

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

FPS = 25.0
# feature frame t covers [t + FRAME_OFFSET, t + FRAME_OFFSET + 1) video frames.
# 0.5 = centre of the frame; the +1 accounts for the dropped leading frame.
FRAME_OFFSET = 1.5

LRS2_ROOT = _lrs2_corpus("lrs2/mvlrs_v1/main")
FEAT_ROOT = _lrs2("feats")          # (T, 768) float16
WORK      = _lrs2("ambiguity_map")  # everything we produce
SPLITS    = ("train", "valid", "test")

# 39 ARPAbet phonemes, stress stripped. Matches g2p_en and MFA's English dictionary.
ARPABET = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P",
    "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]
PHONE_ID = {p: i for i, p in enumerate(ARPABET)}
SIL_ID = -1  # silence / out of any phone

# Same grouping used by scripts/confusion_matrix.py -- keep them identical so the
# ambiguity map and the substitution analysis can be read against each other.
VISEME = {
    "bilabial /p,b,m/":   {"P", "B", "M"},
    "labiodental /f,v/":  {"F", "V"},
    "alveolar /t,d,n/":   {"T", "D", "N"},
    "sibilant /s,z/":     {"S", "Z"},
    "velar /k,g/":        {"K", "G"},
    "dental /th,dh/":     {"TH", "DH"},
}


def same_viseme(a: str, b: str) -> str | None:
    """Return the viseme group name if a and b share one, else None."""
    for name, g in VISEME.items():
        if a in g and b in g:
            return name
    return None


def uid_to_paths(uid: str) -> tuple[Path, Path]:
    """'6330311066473698535_00011' -> (mp4, txt) under the LRS2 tree."""
    spk, utt = uid.rsplit("_", 1)
    d = LRS2_ROOT / spk
    return d / f"{utt}.mp4", d / f"{utt}.txt"


def read_transcript(txt: Path) -> str | None:
    """LRS2 .txt begins 'Text:  AND FOR ME ...'. Returns the words, or None."""
    try:
        for line in txt.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Text:"):
                t = line[5:].strip()
                return t or None
    except OSError:
        pass
    return None


def frame_times(n_frames: int, offset: float = FRAME_OFFSET):
    """Centre time in seconds of each encoder feature frame."""
    return [(t + offset) / FPS for t in range(n_frames)]


def strip_stress(p: str) -> str:
    """'AA1' -> 'AA'. MFA and g2p_en both emit trailing stress digits."""
    return p.rstrip("012").upper()


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
            n += 1
    return n


# The Auto-AVSR frontend output (3D conv + ResNet, 512-d) on the SAME frame
# grid as the encoder output. Probing both separates local visual evidence from
# what the Conformer's self-attention adds using utterance context.
FRONT_ROOT = _lrs2("front512")


def feat_path(split: str, uid: str, root: Path | None = None) -> Path:
    return (root or FEAT_ROOT) / split / f"{uid}.npy"


def iter_feats(split: str):
    """Yield (uid, size_bytes) for non-empty cached feature files.

    Size is checked because a killed writer leaves zero-byte files behind, and an
    existence-only check silently skips them forever.
    """
    d = FEAT_ROOT / split
    if not d.is_dir():
        return
    with os.scandir(d) as it:
        for e in it:
            if e.name.endswith(".npy"):
                st = e.stat()
                if st.st_size > 128:
                    yield e.name[:-4], st.st_size
