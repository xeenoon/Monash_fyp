#!/usr/bin/env python3
"""Acquire the ignored Poly Haven benchmark atomically from its manifest."""
import argparse, hashlib, json, pathlib, sys, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]

def digest(path, algorithm):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def get(asset, root):
    output = root / asset["path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and digest(output, asset["algorithm"]) == asset["checksum"]:
        print("verified", output)
        return
    part = output.with_suffix(output.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(asset["url"])
    if start: request.add_header("Range", "bytes=%d-" % start)
    try:
        with urllib.request.urlopen(request) as response, part.open("ab" if start else "wb") as stream:
            while True:
                block = response.read(1024 * 1024)
                if not block: break
                stream.write(block)
    except Exception as exc:
        raise RuntimeError("%s (%s remains for resume)" % (exc, part))
    if digest(part, asset["algorithm"]) != asset["checksum"]:
        raise RuntimeError("verification failed for %s; removed partial file" % output)
    part.replace(output)
    print("verified", output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "benchmark-assets")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "benchmark-assets.manifest.json").read_text())
    try:
        for asset in manifest["files"]: get(asset, args.output)
    except Exception as exc:
        print("download failed:", exc, file=sys.stderr)
        print("retry: python3 tools/download_benchmark_assets.py --output %s" % args.output, file=sys.stderr)
        return 1
    return 0
if __name__ == "__main__": raise SystemExit(main())
