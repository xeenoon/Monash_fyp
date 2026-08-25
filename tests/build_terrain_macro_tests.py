#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.png"
        first = root / "first.png"
        second = root / "second.png"
        # Deliberately high-frequency colour input with a broad left/right tint.
        image = Image.new("RGB", (128, 128))
        pixels = image.load()
        for y in range(128):
            for x in range(128):
                checker = 75 if (x + y) % 2 else -75
                pixels[x, y] = (max(0, min(255, 80 + x + checker)),
                                max(0, min(255, 150 - checker)),
                                max(0, min(255, 90 + checker)))
        image.save(source)

        command = [sys.executable, str(args.tool), "--source", str(source),
                   "--size", "16", "--blur-radius", "1.25"]
        subprocess.run(command + ["--output", str(first)], check=True)
        subprocess.run(command + ["--output", str(second)], check=True)

        with Image.open(first) as macro:
            assert macro.mode == "RGB"
            assert macro.size == (16, 16)
            # The broad source gradient survives, while the checker is gone.
            assert ImageStat.Stat(macro.crop((0, 0, 8, 16))).mean[0] < \
                   ImageStat.Stat(macro.crop((8, 0, 16, 16))).mean[0]
        with Image.open(first) as a, Image.open(second) as b:
            assert ImageChops.difference(a, b).getbbox() is None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
