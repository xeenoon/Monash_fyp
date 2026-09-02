#!/usr/bin/env python3
"""Reproduce the checked-in CC0 dungeon runtime material set."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "textures" / "dungeon" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "textures" / "dungeon" / "runtime"
USER_AGENT = "gameport-dungeon-material-fetcher/1.0"


def digest(path: Path) -> str:
    value = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def download(url: str, destination: Path, expected_md5: str, force: bool) -> None:
    if destination.exists() and not force and digest(destination) == expected_md5:
        print(f"{destination.name}: verified")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    actual = digest(temporary)
    if actual != expected_md5:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {destination.name}: {actual} != {expected_md5}")
    temporary.replace(destination)
    print(f"{destination.name}: downloaded")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    resolution = manifest["resolution"]
    for material in manifest["materials"]:
        files = fetch_json(f"https://api.polyhaven.com/files/{material['id']}")
        for local_channel, selection in material["files"].items():
            entry = files[selection["channel"]][resolution][selection["extension"]]
            if entry["md5"] != selection["md5"]:
                raise ValueError(
                    f"upstream checksum changed for {material['id']} {local_channel}: "
                    f"{entry['md5']} != {selection['md5']}"
                )
            filename = f"{material['role']}_{local_channel}.{selection['extension']}"
            download(entry["url"], args.output / filename, selection["md5"], args.force)
    print(f"dungeon materials ready under {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
