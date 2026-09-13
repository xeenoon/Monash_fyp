#!/usr/bin/env python3
"""Offline integrity checks for the checked-in dungeon material set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assert manifest["license"] == "CC0-1.0"
    assert manifest["resolution"] == "2k"
    roles = {material["role"] for material in manifest["materials"]}
    assert roles == {"floor", "wall", "exit", "lock", "cloth"}
    for material in manifest["materials"]:
        assert material["width_m"] > 0
        assert material["page"].startswith("https://polyhaven.com/a/")
        for channel, selection in material["files"].items():
            path = args.runtime / f"{material['role']}_{channel}.{selection['extension']}"
            assert path.is_file(), path
            assert path.stat().st_size > 1024, path
            assert md5(path) == selection["md5"], path
    for material in manifest.get("local_materials", []):
        path = args.runtime / material["file"]
        assert path.is_file(), path
        assert md5(path) == material["md5"], path
    print("dungeon asset tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

