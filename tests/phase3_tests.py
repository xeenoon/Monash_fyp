#!/usr/bin/env python3
"""End-to-end test for the Phase 3 builder, validator, atlas, and C loader."""

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def digest_tree(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True)
    parser.add_argument("--loader", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="terrain-phase3-") as temporary:
        root = Path(temporary)
        dem = Image.new("F", (17, 17))
        elevations = [100.0 + x * 2.0 + y * 3.0 + ((x * y) % 5) * 0.25
                      for y in range(17) for x in range(17)]
        elevations[8 * 17 + 8] = float("nan")
        dem.putdata(elevations)
        dem_path = root / "dem.tiff"
        dem.save(dem_path)
        imagery = Image.new("RGBA", (17, 17))
        imagery.putdata([(x * 15, y * 15, (x * 11 + y * 7) & 255,
                          round(255 * x / 16))
                         for y in range(17) for x in range(17)])
        imagery_path = root / "imagery.png"
        imagery.save(imagery_path)

        outputs = [root / "first", root / "second"]
        for output in outputs:
            run("python3", args.tool, "build", "--dem", str(dem_path),
                "--imagery", str(imagery_path), "--output", str(output),
                "--extent", "2647000", "1160000", "2648000", "1161000",
                "--profile", "EPSG:2056", "--levels", "3", "--samples", "5",
                "--imagery-size", "8", "--gutter", "1", "--source", "synthetic",
                "--source-version", "test-v1")
            run("python3", args.tool, "validate", str(output))
        if digest_tree(outputs[0]) != digest_tree(outputs[1]):
            raise SystemExit("tile output is not deterministic")
        root_imagery = Image.open(outputs[0] / "imagery" / "0" / "0" / "0.png")
        if root_imagery.mode != "RGBA" or root_imagery.getchannel("A").getextrema() == (255, 255):
            raise SystemExit("material blend alpha was not preserved")

        atlas = root / "atlas.png"
        run("python3", args.tool, "inspect", str(outputs[0]),
            "--level", "2", "--card-size", "64", "--output", str(atlas))
        if Image.open(atlas).size != (512, 256):
            raise SystemExit("inspection atlas has unexpected dimensions")

        root_tile = outputs[0] / "tiles" / "0" / "0" / "0.trn"
        run(args.loader, str(root_tile))
        corrupt = root / "corrupt.trn"
        damaged = bytearray(root_tile.read_bytes())
        damaged[-1] ^= 0x80
        corrupt.write_bytes(damaged)
        run(args.loader, str(corrupt), "checksum")

        r16 = root / "r16"
        run("python3", args.tool, "build", "--dem", str(dem_path),
            "--imagery", str(imagery_path), "--output", str(r16),
            "--extent", "2647000", "1160000", "2648000", "1161000",
            "--profile", "EPSG:2056", "--levels", "2", "--samples", "5",
            "--imagery-size", "8", "--gutter", "1", "--height-encoding", "r16")
        run("python3", args.tool, "validate", str(r16))
        run(args.loader, str(r16 / "tiles" / "0" / "0" / "0.trn"),
            "allow-all-valid", str(r16))

    print("phase 3 offline terrain tile tests passed")


if __name__ == "__main__":
    main()
