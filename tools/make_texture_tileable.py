#!/usr/bin/env python3
"""Turn one colour, scalar, or tangent-normal map into a periodic texture.

The source boundary is moved to the image centre, where a multiband cross blend
repairs it. The new repeat boundary comes from adjacent pixels in the original
interior, so sampling across wrap no longer exposes the old edge. Apply the
same options to every channel in a PBR set to keep their features registered.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _heal_centre(image: np.ndarray, axis: int, half_width: int) -> np.ndarray:
    values = np.moveaxis(image, axis, 1).copy()
    length = values.shape[1]
    centre = length // 2
    half_width = max(2, min(half_width, centre - 2, length - centre - 2))
    lo, hi = centre - half_width, centre + half_width
    low = ndimage.gaussian_filter(values, sigma=(0.0, max(1.5, half_width * 0.16), 0.0),
                                  mode="wrap")
    detail = values - low
    count = hi - lo + 1
    blend = _smoothstep(np.linspace(0.0, 1.0, count, dtype=np.float32))
    blend = blend[None, :, None]

    # Palette/lighting changes broadly. Detail comes from reflected pixels on
    # either side of the old edge, which avoids a conspicuous blurred stripe.
    low_bridge = low[:, lo:lo + 1] * (1.0 - blend) + low[:, hi:hi + 1] * blend
    offsets = np.arange(count)
    left_indices = np.maximum(0, lo - np.minimum(offsets, half_width - 1))
    right_indices = np.minimum(length - 1, hi + np.minimum(count - 1 - offsets,
                                                           half_width - 1))
    detail_bridge = (detail[:, left_indices] * (1.0 - blend)
                     + detail[:, right_indices] * blend)
    values[:, lo:hi + 1] = low_bridge + detail_bridge
    return np.moveaxis(values, 1, axis)


def make_tileable(pixels: np.ndarray, overlap: float, normal_map: bool = False) -> np.ndarray:
    if pixels.ndim == 2:
        pixels = pixels[..., None]
    if pixels.ndim != 3:
        raise ValueError("expected a 2-D image")
    if not 0.02 <= overlap <= 0.45:
        raise ValueError("overlap must be between 0.02 and 0.45")
    result = np.roll(pixels.astype(np.float32),
                     shift=(pixels.shape[0] // 2, pixels.shape[1] // 2), axis=(0, 1))
    result = _heal_centre(result, 1, int(round(pixels.shape[1] * overlap * 0.5)))
    result = _heal_centre(result, 0, int(round(pixels.shape[0] * overlap * 0.5)))
    if normal_map and result.shape[2] >= 3:
        normal = result[..., :3] * 2.0 - 1.0
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1.0e-6)
        result[..., :3] = normal * 0.5 + 0.5
    return np.clip(result, 0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overlap", type=float, default=0.24,
                        help="fraction of each axis repaired around the old boundary")
    parser.add_argument("--normal-map", action="store_true",
                        help="renormalize RGB as an OpenGL/DX tangent normal")
    args = parser.parse_args()
    with Image.open(args.input) as source:
        mode = "RGBA" if source.mode == "RGBA" else ("L" if source.mode in ("L", "I;16") else "RGB")
        array = np.asarray(source.convert(mode), dtype=np.float32) / 255.0
    result = make_tileable(array, args.overlap, args.normal_map)
    if result.shape[-1] == 1:
        result = result[..., 0]
    encoded = np.clip(np.rint(result * 255.0), 0, 255).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded, mode=mode).save(args.output)
    print(f"wrote periodic texture {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
