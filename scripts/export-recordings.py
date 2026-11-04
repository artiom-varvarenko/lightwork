#!/usr/bin/env python3
"""Copy real recorded table sessions into recordings/<name>/ for the public repo: the requests, the replies, timing, cost,
the pictures sent and the pictures and models that came back. Mic audio is left out unless --audio is given.

Usage: scripts/export-recordings.py <name> <session stamp> [<session stamp> ...] [--audio]
A stamp is the start of a folder name under .lightwork/play-calls/ (20261104T154554Z takes -brain, -sam, -image, -voice too).
Refuses a file with a key in it. Never touches what it copies from.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / ".lightwork" / "play-calls"
KEY = re.compile(rb"LLM_[0-9]{6,}_[A-Za-z0-9]")
SKIP = {".wav"}


def main(argv: list[str]) -> int:
    audio = "--audio" in argv
    args = [a for a in argv if a != "--audio"]
    if len(args) < 2:
        print(__doc__)
        return 2
    name, stamps = args[0], args[1:]
    out = ROOT / "recordings" / name
    if out.exists():
        print(f"refusing: {out} exists")
        return 1
    files = calls = size = 0
    for stamp in stamps:
        folders = sorted(p for p in SOURCE.glob(f"{stamp}*") if p.is_dir())
        if not folders:
            print(f"refusing: no session {stamp}")
            return 1
        for folder in folders:
            for src in sorted(p for p in folder.rglob("*") if p.is_file()):
                if src.suffix in SKIP and not audio:
                    continue
                body = src.read_bytes()
                if KEY.search(body):
                    print(f"refusing: a key in {src}")
                    return 1
                dst = out / folder.name / src.relative_to(folder)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                files += 1
                size += len(body)
                calls += src.name == "meta.json"
    print(f"{out}: {calls} model calls, {files} files, {size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
