"""Pack repo source into verifiable base64 chunks for an air-gapped box.

The experiment machine (Kaggle backend behind Colab) has no network at all --
no pip, no git, no DNS.  The only channel is the notebook itself, so source has
to be retyped into cells.  A single 24 KB base64 blob turned out to be
unreliable over that channel (length preserved, contents silently altered), so
this splits the payload into checksummed chunks: corruption is localised to one
chunk and only that chunk has to be resent.

    python tools/pack.py                # writes tools/_chunks/*.txt + manifest

lzma rather than gzip: ~25% smaller on source text, which is 25% fewer
characters to move by hand.
"""

import base64
import hashlib
import io
import lzma
import shutil
import sys
import tarfile
from pathlib import Path

SKIP = {"__pycache__", ".pytest_cache", ".git", ".ipynb_checkpoints", "_chunks"}
CHUNK = 3000
ROOT = Path(__file__).resolve().parent.parent


def build_tar(paths: list[str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for p in paths:
            for f in sorted((ROOT / p).rglob("*")):
                if f.is_file() and not any(s in f.parts for s in SKIP):
                    tar.add(f, arcname=str(f.relative_to(ROOT)).replace("\\", "/"))
    return lzma.compress(buf.getvalue(), preset=9)


def main(paths: list[str]) -> None:
    raw = build_tar(paths)
    b64 = base64.b64encode(raw).decode()
    out = ROOT / "tools" / "_chunks"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    chunks = [b64[i:i + CHUNK] for i in range(0, len(b64), CHUNK)]
    for i, c in enumerate(chunks):
        (out / f"c{i:02d}.txt").write_text(c)

    print(f"payload      {len(raw):,} bytes lzma -> {len(b64):,} b64 chars")
    print(f"chunks       {len(chunks)} x {CHUNK}")
    print(f"md5(b64)     {hashlib.md5(b64.encode()).hexdigest()}")
    print(f"md5(payload) {hashlib.md5(raw).hexdigest()}")
    for i, c in enumerate(chunks):
        print(f"  c{i:02d} len={len(c):5d} md5={hashlib.md5(c.encode()).hexdigest()[:8]}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["lora", "lab", "experiments"])
