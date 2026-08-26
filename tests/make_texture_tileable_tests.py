#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def boundary_error(image: np.ndarray) -> float:
    horizontal = np.abs(image[:, 0].astype(float) - image[:, -1]).mean()
    vertical = np.abs(image[0].astype(float) - image[-1]).mean()
    return float(horizontal + vertical)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.png"
        output = root / "tileable.png"
        y, x = np.mgrid[:128, :128]
        image = np.stack((x * 2, y * 2, (x + y) % 64 * 4), axis=-1).astype(np.uint8)
        Image.fromarray(image, "RGB").save(source)
        subprocess.run([sys.executable, str(args.tool), str(source), str(output)], check=True)
        result = np.asarray(Image.open(output).convert("RGB"))
        assert result.shape == image.shape
        assert boundary_error(result) < boundary_error(image) * 0.12
        centre_jump = np.abs(result[:, 63].astype(float) - result[:, 64]).mean()
        assert centre_jump < 35.0, centre_jump
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
