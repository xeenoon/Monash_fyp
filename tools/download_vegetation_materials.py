#!/usr/bin/env python3
"""Fetch the vegetation pack's original CC0 scanned bark maps (requires curl).

The generated source manifest pins every URL and checksum for offline audits.
Re-running verifies existing files; --refresh re-queries the upstream API.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/vegetation"
MANIFEST = OUT / "material_sources.json"
SOURCES = {"broadleaf_bark": "bark_brown_02", "conifer_bark": "pine_bark"}
CHANNELS = {"albedo": ("Diffuse", "4k", "jpg"),
            "normal": ("nor_gl", "4k", "png"),
            "orm": ("arm", "4k", "jpg"),
            "height": ("Displacement", "2k", "png")}


def curl(url, output=None):
    command = ["curl", "--fail", "--location", "--silent", "--show-error",
               "--retry", "3", "--max-time", "180", "-A", "gameport-material-fetcher/1.0", url]
    if output:
        command += ["--output", str(output)]
    return subprocess.check_output(command)


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    (OUT / "textures").mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists() and not args.refresh:
        manifest = json.loads(MANIFEST.read_text())
    else:
        manifest = {"license": "CC0-1.0", "materials": []}
        for role, source in SOURCES.items():
            data = json.loads(curl(f"https://api.polyhaven.com/files/{source}"))
            material = {"role": role, "source": source, "page": f"https://polyhaven.com/a/{source}",
                        "width_m": 1.0, "files": {}}
            for channel, (key, resolution, extension) in CHANNELS.items():
                entry = data[key][resolution][extension]
                material["files"][channel] = {**entry, "resolution": resolution,
                    "path": f"textures/{role}_{channel}.{extension}"}
            manifest["materials"].append(material)
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    def fetch(entry):
        path = OUT / entry["path"]
        if not path.exists() or md5(path) != entry["md5"]:
            temporary = path.with_suffix(path.suffix + ".part")
            curl(entry["url"], temporary)
            if md5(temporary) != entry["md5"]:
                raise ValueError(f"Checksum mismatch: {path}")
            temporary.replace(path)
        print(f"Verified {path.name}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, [f for m in manifest["materials"] for f in m["files"].values()]))


if __name__ == "__main__":
    main()
