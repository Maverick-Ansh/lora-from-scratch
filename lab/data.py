"""Corpus construction from Python source already present on the machine.

Why source code, and why on-disk:

* The experiment machine has **no internet**, so there is no pre-trained
  checkpoint and no dataset to download.  Everything -- pre-training included
  -- has to come from bytes that are already there.  A Linux box with a
  scientific Python stack has ~190 MB of real, highly structured text sitting
  in ``dist-packages``.
* It gives clean **domain boundaries**.  ``sympy`` (symbolic algebra),
  ``torch`` (tensor/ML), and ``matplotlib`` (plotting) are written in
  recognisably different dialects of the same language.  Pre-train on a general
  mixture, hold those three out entirely, and adapting to one of them is a
  genuine distribution shift of exactly the kind LoRA claims to handle -- with
  none of the licensing or leakage ambiguity of a scraped corpus.

Splits are **by file**, never by offset, so a validation snippet is never a
continuation of a training snippet.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SITE = "/usr/local/lib/python3.12/dist-packages"
STDLIB = "/usr/lib/python3.12"

# Pre-training mixture: general-purpose Python, deliberately excluding every
# domain we later adapt to.  Sized so 15k steps is roughly two passes rather
# than four -- enough repetition to learn the language, not enough to make the
# base model's held-out loss a memorisation artifact.
#
# Note what is *absent*: no symbolic-algebra library, no deep-learning
# framework, no plotting library.  Those are the three adaptation targets, and
# keeping their idioms out of pre-training is what makes the adaptation a real
# distribution shift.
PRETRAIN_ROOTS = [
    STDLIB,
    f"{SITE}/scipy",
    f"{SITE}/pandas",
    f"{SITE}/sklearn",
    f"{SITE}/IPython",
    f"{SITE}/numpy",
    f"{SITE}/pyspark",
]

# Held-out adaptation domains: never seen during pre-training.
DOMAIN_ROOTS = {
    "sympy": [f"{SITE}/sympy"],
    "torch": [f"{SITE}/torch"],
    "matplotlib": [f"{SITE}/matplotlib"],
}


@dataclass
class Corpus:
    train: np.ndarray  # uint8
    val: np.ndarray

    def __repr__(self) -> str:
        return f"Corpus(train={self.train.nbytes/1e6:.1f}MB, val={self.val.nbytes/1e6:.1f}MB)"


def _iter_py_files(roots: list[str]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # test trees are noisy and full of near-duplicates
            dirnames[:] = [d for d in dirnames if d not in {"tests", "test", "__pycache__"}]
            for fn in filenames:
                if fn.endswith(".py"):
                    files.append(Path(dirpath) / fn)
    return files


def _stable_hash(p: Path) -> int:
    return int(hashlib.md5(str(p).encode()).hexdigest()[:8], 16)


def build_corpus(roots: list[str], val_frac: float = 0.02, max_bytes: int | None = None) -> Corpus:
    """Concatenate every ``.py`` file under ``roots`` into uint8 arrays.

    File-level split via a stable hash of the path: the same file always lands
    on the same side, so re-running with different ``max_bytes`` does not
    reshuffle the validation set.
    """
    files = _iter_py_files(roots)
    files.sort(key=_stable_hash)  # deterministic, content-independent shuffle

    n_val = max(1, int(len(files) * val_frac))
    val_files, train_files = files[:n_val], files[n_val:]

    def read_all(fs: list[Path], cap: int | None) -> np.ndarray:
        chunks, total = [], 0
        for f in fs:
            try:
                b = f.read_bytes()
            except OSError:
                continue
            # Non-ASCII is rare in source and would waste vocab on partial
            # UTF-8 continuation bytes; drop those files.
            if not b or max(b) > 127:
                b = bytes(c for c in b if c < 128)
            chunks.append(np.frombuffer(b, dtype=np.uint8))
            total += len(b)
            if cap is not None and total >= cap:
                break
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint8)

    val_cap = None if max_bytes is None else max(1_000_000, int(max_bytes * val_frac))
    return Corpus(train=read_all(train_files, max_bytes), val=read_all(val_files, val_cap))


def cached_corpus(name: str, roots: list[str], cache_dir: str = "/kaggle/working/corpus",
                  **kw) -> Corpus:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    tr, va = d / f"{name}_train.npy", d / f"{name}_val.npy"
    if tr.exists() and va.exists():
        return Corpus(train=np.load(tr), val=np.load(va))
    c = build_corpus(roots, **kw)
    np.save(tr, c.train)
    np.save(va, c.val)
    return c
