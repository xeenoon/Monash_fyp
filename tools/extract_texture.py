#!/usr/bin/env python3
"""Extract the terrain albedo texture from the SWISSIMAGE data.

Reads the SWISSIMAGE tile that covers the same 1 km x 1 km region as the
heightmap (centre tile 2647-1160), resizes it to a power-of-two square, and
writes ``assets/terrain_albedo.png`` for the renderer to sample.

Pillow only; the tile is a plain RGB TIFF.
"""

import os

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Same tile as the heightmap so texture and geometry line up 1:1.
TILE = os.path.join(ROOT, "alps-data", "swissimage_2024_2647-1160_rgb.tif")
OUT_SIZE = 2048  # power-of-two square, plenty for a 1 km field
OUT = os.path.join(ROOT, "assets", "terrain_albedo.png")


def main():
    im = Image.open(TILE).convert("RGB")
    print(f"loaded {TILE} ({im.size[0]}x{im.size[1]}, {im.mode})")
    im = im.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    im.save(OUT, "PNG")
    print(f"wrote {OUT} ({OUT_SIZE}x{OUT_SIZE})")


if __name__ == "__main__":
    main()
