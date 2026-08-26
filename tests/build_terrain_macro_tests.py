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
        labels = root / "labels.png"
        # Deliberately high-frequency colour input with a broad left/right tint.
        image = Image.new("RGB", (128, 128))
        pixels = image.load()
        for y in range(128):
            for x in range(128):
                checker = 12 if (x + y) % 2 else -12
                pixels[x, y] = (60 + x // 2 + checker,
                                110 + x // 4 - checker,
                                55 + checker)
        image.save(source)

        # A turquoise lake occupies the middle. It must be labelled rejected
        # and must not leak into the resulting material colour field.
        pixels = image.load()
        for y in range(36, 92):
            for x in range(36, 92):
                pixels[x, y] = (20, 145, 190)
        # Explicit neutral rock on the right gives the alpha blend two known
        # endpoints while the lake remains a rejected hole between them.
        for y in range(128):
            for x in range(104, 128):
                pixels[x, y] = (92, 88, 82)
        image.save(source)

        command = [sys.executable, str(args.tool), "--source", str(source),
                   "--size", "16", "--blur-radius", "1.25",
                   "--labels-output", str(labels)]
        subprocess.run(command + ["--output", str(first)], check=True)
        subprocess.run(command + ["--output", str(second)], check=True)

        with Image.open(first) as macro:
            assert macro.mode == "RGBA"
            assert macro.size == (16, 16)
            # The broad source gradient survives, while the checker is gone.
            assert ImageStat.Stat(macro.crop((0, 0, 8, 16))).mean[0] < \
                   ImageStat.Stat(macro.crop((8, 0, 16, 16))).mean[0]
            centre = macro.getpixel((8, 8))
            assert centre[2] < 165, centre
            assert centre[1] - centre[0] < 90, centre
            assert macro.getpixel((1, 8))[3] > 220
            assert macro.getpixel((15, 8))[3] < 30
        with Image.open(labels) as material_labels:
            rock, grass, rejected = material_labels.getpixel((8, 8))
            assert rejected > rock and rejected > grass
        with Image.open(first) as a, Image.open(second) as b:
            assert ImageChops.difference(a, b).getbbox() is None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
