#!/usr/bin/env python3
"""Build sparse cliff-detail inserts from the shadowed Alpine imagery.

The existing tile imagery remains the high-level albedo.  This tool derives a
small bank of steep-slope patches, removes their low-frequency illumination,
and stores only where and how much detail should be inserted.  It never
generates a replacement terrain texture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    value = rgb.astype(np.float32) / 255.0
    return np.where(value <= 0.04045, value / 12.92,
                    ((value + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    value = np.maximum(linear, 0.0)
    encoded = np.where(value <= 0.0031308, value * 12.92,
                       1.055 * value ** (1.0 / 2.4) - 0.055)
    return np.clip(np.rint(encoded * 255.0), 0, 255).astype(np.uint8)


def _wrap_patch(image: np.ndarray, y: int, x: int, size: int) -> np.ndarray:
    y %= image.shape[0]
    x %= image.shape[1]
    rows = (np.arange(y, y + size) % image.shape[0])[:, None]
    cols = np.arange(x, x + size) % image.shape[1]
    return image[rows, cols]


def collect_patches(images: list[np.ndarray], patch_size: int,
                    count: int, seed: int) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    candidates: list[tuple[float, int, int, int]] = []
    for image_index, image in enumerate(images):
        linear = _srgb_to_linear(image)
        luminance = linear @ np.array((0.2126, 0.7152, 0.0722), np.float32)
        gradient_y, gradient_x = np.gradient(luminance)
        gradient = np.hypot(gradient_x, gradient_y)
        stride = max(patch_size // 2, 1)
        limit_y = max(image.shape[0] - patch_size, 1)
        limit_x = max(image.shape[1] - patch_size, 1)
        for y in range(rng.integers(0, stride), limit_y, stride):
            for x in range(rng.integers(0, stride), limit_x, stride):
                score = float(np.mean(gradient[y:y + patch_size,
                                             x:x + patch_size]))
                candidates.append((score, image_index, int(y), int(x)))

    if not candidates:
        raise ValueError("no imagery patches available")
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:max(count * 6, count)]
    patches = np.stack([
        _wrap_patch(images[item[1]], item[2], item[3], patch_size)
        for item in selected
    ])

    # High-pass in linear light suppresses baked sun/shadow while retaining
    # rock grain.  The wrap filter matches the runtime metric repeat.
    linear = _srgb_to_linear(patches)
    luminance = linear @ np.array((0.2126, 0.7152, 0.0722), np.float32)
    blurred = np.empty_like(linear)
    radius = max(1, patch_size // 8)
    for index, values in enumerate(luminance):
        padded = np.pad(values, radius, mode="wrap")
        integral = padded.cumsum(0).cumsum(1)
        integral = np.pad(integral, ((1, 0), (1, 0)))
        low = (integral[2 * radius + 1:, 2 * radius + 1:] -
               integral[:-2 * radius - 1, 2 * radius + 1:] -
               integral[2 * radius + 1:, :-2 * radius - 1] +
               integral[:-2 * radius - 1, :-2 * radius - 1])
        low /= float((2 * radius + 1) ** 2)
        ratio = np.clip(luminance[index] / np.maximum(low, 1e-5), 0.55, 1.45)
        blurred[index] = ratio[..., None] * np.mean(luminance[index])
    return _linear_to_srgb(blurred), [f"patch_{index}" for index in range(len(patches))]


def build_inserts(imagery_root: Path, detail_output: Path, mask_output: Path,
                  stats_output: Path, atlas_size: int, cell_size: int,
                  source_count: int, seed: int) -> None:
    if cell_size < 16 or cell_size > atlas_size or atlas_size % cell_size:
        raise ValueError("cell size must divide the atlas size")
    cells_per_side = atlas_size // cell_size
    cell_count = cells_per_side ** 2
    files = sorted(imagery_root.rglob("*.png"))
    if not files:
        raise ValueError(f"no imagery under {imagery_root}")

    images = []
    for path in files:
        with Image.open(path) as image:
            images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    patches, names = collect_patches(images, cell_size, source_count, seed)
    if len(patches) < cell_count:
        repeats = (cell_count + len(patches) - 1) // len(patches)
        patches = np.concatenate([patches] * repeats)[:cell_count]

    # Deterministically assign matched steep exemplars to atlas cells.
    order = np.arange(len(patches))
    blocks = np.array_split(order, cell_count)
    atlas = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)
    for index, block in enumerate(blocks):
        y = (index // cells_per_side) * cell_size
        x = (index % cells_per_side) * cell_size
        atlas[y:y + cell_size, x:x + cell_size] = patches[block[-1]]

    # The insertion mask is intentionally sparse: zero means use existing
    # macro imagery unchanged; one permits only high-frequency residual detail.
    mask = np.zeros((atlas_size, atlas_size), dtype=np.uint8)
    edge = max(cell_size // 8, 1)
    for index in range(cell_count):
        y = (index // cells_per_side) * cell_size
        x = (index % cells_per_side) * cell_size
        mask[y + edge:y + cell_size - edge,
             x + edge:x + cell_size - edge] = 255

    detail_output.parent.mkdir(parents=True, exist_ok=True)
    mask_output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(atlas, "RGB").save(detail_output, "PNG", optimize=True)
    Image.fromarray(mask, "L").save(mask_output, "PNG", optimize=True)
    stats = {
        "atlas": str(detail_output),
        "mask": str(mask_output),
        "atlas_size": atlas_size,
        "cell_size": cell_size,
        "source_images": len(images),
        "source_patches": min(source_count, len(order)),
        "seed": seed,
        "operation": "add_high_frequency_detail_only",
    }
    stats_output.write_text(json.dumps(stats, indent=2) + "\n")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagery", type=Path,
                        default=root / "alps-data/trn-alps-16km-shadowed/imagery/5")
    parser.add_argument("--detail-output", type=Path,
                        default=root / "textures/runtime/shadowed_cliff_detail.png")
    parser.add_argument("--mask-output", type=Path,
                        default=root / "textures/runtime/shadowed_cliff_mask.png")
    parser.add_argument("--stats-output", type=Path,
                        default=root / "textures/runtime/shadowed_cliff_stats.json")
    parser.add_argument("--atlas-size", type=int, default=1024)
    parser.add_argument("--cell-size", type=int, default=128)
    parser.add_argument("--source-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=91746)
    args = parser.parse_args()

    build_inserts(args.imagery, args.detail_output, args.mask_output,
                  args.stats_output, args.atlas_size, args.cell_size,
                  args.source_count, args.seed)
    print(f"wrote {args.detail_output} and insertion mask {args.mask_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
