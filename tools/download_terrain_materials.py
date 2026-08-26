#!/usr/bin/env python3
"""Download and extract the CC0 terrain micro-material source library."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "textures" / "manifest.json"
CHANNELS = {
    "Diffuse": ("jpg", "albedo"),
    "nor_gl": ("png", "normal_gl"),
    "Rough": ("png", "roughness"),
    "Displacement": ("png", "height"),
    "AO": ("jpg", "ao"),
}


def _download(url: str, destination: Path, expected_md5: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (expected_md5 is None or
                                 hashlib.md5(destination.read_bytes()).hexdigest() == expected_md5):
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "terrain-gen-material-fetcher/1.0"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if expected_md5 is not None:
        digest = hashlib.md5(temporary.read_bytes()).hexdigest()
        if digest != expected_md5:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"checksum mismatch for {url}: {digest} != {expected_md5}")
    temporary.replace(destination)


def _polyhaven(material: dict, destination: Path) -> None:
    request = urllib.request.Request(
        f"https://api.polyhaven.com/files/{material['id']}",
        headers={"User-Agent": "terrain-gen-material-fetcher/1.0"},
    )
    with urllib.request.urlopen(request) as response:
        files = json.load(response)
    resolution = material["resolution"]
    for api_channel, (extension, local_channel) in CHANNELS.items():
        entry = files.get(api_channel, {}).get(resolution, {}).get(extension)
        if entry is None:
            raise ValueError(f"{material['id']} lacks {api_channel} {resolution} {extension}")
        target = destination / f"{material['id']}_{local_channel}_{resolution}.{extension}"
        print(f"{material['id']}: {local_channel} -> {target.name}", flush=True)
        _download(entry["url"], target, entry["md5"])


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"unsafe archive member: {member.filename}")
        package.extractall(destination)


def _archive(material: dict, destination: Path, cache: Path) -> None:
    suffix = ".zip" if material["archive"].lower().endswith(".zip") else ".rar"
    archive = cache / f"{material['id']}{suffix}"
    print(f"{material['id']}: package -> {archive.name}", flush=True)
    _download(material["archive"], archive)
    if suffix == ".zip":
        _safe_extract_zip(archive, destination)
    else:
        seven_zip = shutil.which("7z")
        if not seven_zip:
            raise RuntimeError("extracting PolyScan .rar packages requires 7z")
        subprocess.run([seven_zip, "x", "-y", f"-o{destination}", str(archive)], check=True,
                       stdout=subprocess.DEVNULL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source", type=Path, default=ROOT / "textures" / "source")
    parser.add_argument("--cache", type=Path, default=ROOT / "textures" / ".downloads")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for material in manifest["materials"]:
        destination = args.source / material["id"]
        marker = destination / ".complete"
        if marker.exists() and not args.force:
            print(f"{material['id']}: already complete", flush=True)
            continue
        destination.mkdir(parents=True, exist_ok=True)
        if material["provider"] == "polyhaven":
            _polyhaven(material, destination)
        else:
            _archive(material, destination, args.cache)
        marker.write_text(json.dumps(material, indent=2) + "\n", encoding="utf-8")
    print(f"downloaded {len(manifest['materials'])} CC0 materials under {args.source}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
