"""Print per-file size/checksum, and flag anything that breaks raw-string transfer.

Source is shipped to the air-gapped experiment box by pasting it into notebook
cells wrapped in r'''...''' literals.  Two things would silently break that:
a literal triple-single-quote in the file, or a trailing backslash.  This
checks both, and prints the md5 the box must reproduce.
"""

import hashlib
import pathlib
import sys

FILES = [
    "lab/__init__.py", "lab/model.py", "lab/data.py", "lab/train.py", "lab/methods.py",
    "experiments/01_pretrain.py",
    "lora/__init__.py", "lora/layers.py", "lora/inject.py", "lora/utils.py",
    "lora/variants.py",
]

TRIPLE = "'" * 3

for f in sys.argv[1:] or FILES:
    b = pathlib.Path(f).read_bytes()
    t = b.decode()
    flags = []
    if b"\r\n" in b:
        flags.append("CRLF")
    if TRIPLE in t:
        flags.append("TRIPLE_SQ")
    if t.rstrip("\n").endswith("\\"):
        flags.append("TRAILING_BACKSLASH")
    print(f"{f:28s} {len(b):6d}B  md5={hashlib.md5(b).hexdigest()}  {' '.join(flags) or 'ok'}")
