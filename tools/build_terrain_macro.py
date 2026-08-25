#!/usr/bin/env python3
"""Build a deliberately low-frequency terrain colour map.

This is the small replacement for the exemplar/corpus synthesizer in the
texture-mapping worktree. It keeps only broad Alpine colour placement; authored
stone and grass textures are expected to provide all close-range detail.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "assets" / "terrain_albedo.png"
DEFAULT_OUTPUT = ROOT / "assets" / "terrain_macro.png"


def build_macro(source: Path, output: Path, size: int, blur_radius: float) -> None:
    if size < 2:
        raise ValueError("size must be at least 2 pixels")
    if blur_radius < 0.0:
        raise ValueError("blur radius cannot be negative")

    with Image.open(source) as image:
        rgb = image.convert("RGB")
        # BOX makes the low-resolution texels true area samples rather than a
        # sharpened miniature. The small Gaussian then removes the last traces
        # of roads, scree lines, and source-photo texture.
        macro = rgb.resize((size, size), Image.Resampling.BOX)
    if blur_radius > 0.0:
        macro = macro.filter(ImageFilter.GaussianBlur(blur_radius))

    output.parent.mkdir(parents=True, exist_ok=True)
    macro.save(output, "PNG", optimize=False, compress_level=9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Alpine source image (default: assets/terrain_albedo.png)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="output PNG (default: assets/terrain_macro.png)")
    parser.add_argument("--size", type=int, default=64,
                        help="macro samples per side (default: 64)")
    parser.add_argument("--blur-radius", type=float, default=1.5,
                        help="Gaussian radius in macro texels (default: 1.5)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_macro(args.source, args.output, args.size, args.blur_radius)
    print(f"wrote {args.output} ({args.size}x{args.size}, blur={args.blur_radius:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
