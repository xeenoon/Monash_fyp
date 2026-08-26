#!/usr/bin/env python3
"""Build compact rock + grass runtime atlases from the CC0 PBR library."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*")
                  if path.suffix.lower() in IMAGE_SUFFIXES)


def _find_channel(directory: Path, channel: str) -> tuple[Path | None, bool]:
    candidates = _files(directory)
    rules = {
        "albedo": ("albedo", "basecolor", "base_color", "diffuse", "_color"),
        "normal_gl": ("normalgl", "normal_gl", "nor_gl"),
        "normal_dx": ("normaldx", "normal_dx", "nor_dx"),
        "normal": ("normal",),
        "roughness": ("roughness", "_rough"),
        "height": ("displacement", "_disp", "height"),
        "ao": ("ambientocclusion", "_ao", "ao"),
    }
    for path in candidates:
        name = path.stem.lower().replace(" ", "").replace("-", "_")
        if any(token in name for token in rules[channel]):
            return path, channel == "normal_dx"
    return None, False


def _rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(resized, dtype=np.float32) / 255.0


def _scalar(path: Path | None, size: int, default: float) -> np.ndarray:
    if path is None:
        return np.full((size, size), default, dtype=np.float32)
    with Image.open(path) as image:
        resized = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(resized, dtype=np.float32) / 255.0


def _highpass_luminance(albedo: np.ndarray) -> np.ndarray:
    srgb = albedo
    linear = np.where(srgb <= 0.04045, srgb / 12.92,
                      ((srgb + 0.055) / 1.055) ** 2.4)
    luminance = linear @ np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)
    low = ndimage.gaussian_filter(luminance, sigma=max(3.0, albedo.shape[0] / 24.0),
                                  mode="wrap")
    ratio = np.clip(luminance / np.maximum(low, 1.0e-4), 0.55, 1.45)
    # Stored in a linear UNORM texture: 0.5 is the exact multiplicative neutral.
    encoded = ratio * 0.5
    return np.repeat(encoded[..., None], 3, axis=-1)


def _normal(path: Path | None, directx: bool, size: int) -> np.ndarray:
    if path is None:
        result = np.zeros((size, size, 3), dtype=np.float32)
        result[..., 2] = 1.0
    else:
        result = _rgb(path, size) * 2.0 - 1.0
        if directx:
            result[..., 1] *= -1.0
        result /= np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1.0e-5)
    return result * 0.5 + 0.5


def _normalized_height(path: Path | None, size: int) -> np.ndarray:
    height = _scalar(path, size, 0.5)
    low, high = np.percentile(height, (1.0, 99.0))
    if high - low < 1.0e-5:
        return np.full_like(height, 0.5)
    return np.clip((height - low) / (high - low), 0.0, 1.0)


def _gutter(tile: np.ndarray, width: int) -> np.ndarray:
    padding = ((width, width), (width, width), (0, 0))
    return np.pad(tile, padding, mode="wrap")


def _save(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.clip(np.rint(pixels * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(encoded, "RGB" if encoded.shape[2] == 3 else "RGBA").save(
        path, "PNG", optimize=True
    )


def _load_bank(manifest_path: Path, source_root: Path, bank: str,
               expected: int) -> list[tuple[str, dict, Path]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    materials = manifest["materials"]
    if len(materials) != expected:
        raise ValueError(f"the runtime shader expects {expected} {bank} materials, "
                         f"found {len(materials)}")
    return [(bank, material, source_root / material["id"])
            for material in materials]


def build(rock_manifest: Path, rock_source: Path,
          grass_manifest: Path, grass_source: Path, output: Path,
          cell_size: int, gutter: int) -> None:
    entries = (
        _load_bank(rock_manifest, rock_source, "rock", 9)
        + _load_bank(grass_manifest, grass_source, "grass", 6)
    )
    if cell_size < 16 or gutter < 1 or gutter * 2 >= cell_size:
        raise ValueError("invalid cell size/gutter")
    grid = math.ceil(math.sqrt(len(entries)))
    padded = cell_size + gutter * 2
    albedo_atlas = np.zeros((padded * grid, padded * grid, 3), dtype=np.float32)
    normal_atlas = np.zeros_like(albedo_atlas)
    ormh_atlas = np.zeros((padded * grid, padded * grid, 4), dtype=np.float32)
    layout = []

    for index, (bank, material, directory) in enumerate(entries):
        albedo_path, _ = _find_channel(directory, "albedo")
        normal_path, directx = _find_channel(directory, "normal_gl")
        if normal_path is None:
            normal_path, directx = _find_channel(directory, "normal_dx")
        if normal_path is None:
            normal_path, directx = _find_channel(directory, "normal")
        roughness_path, _ = _find_channel(directory, "roughness")
        height_path, _ = _find_channel(directory, "height")
        ao_path, _ = _find_channel(directory, "ao")
        if albedo_path is None:
            raise FileNotFoundError(f"no albedo found for {material['id']} under {directory}")
        print(f"{index}: {bank}/{material['id']} ({albedo_path.name})", flush=True)
        albedo = _gutter(_highpass_luminance(_rgb(albedo_path, cell_size)), gutter)
        normal = _gutter(_normal(normal_path, directx, cell_size), gutter)
        ao = _scalar(ao_path, cell_size, 1.0)
        roughness = _scalar(roughness_path, cell_size, 0.82)
        height = _normalized_height(height_path, cell_size)
        ormh = _gutter(np.stack((ao, roughness, np.zeros_like(ao), height), axis=-1), gutter)
        row, column = divmod(index, grid)
        ys, xs = slice(row * padded, (row + 1) * padded), slice(column * padded, (column + 1) * padded)
        albedo_atlas[ys, xs] = albedo
        normal_atlas[ys, xs] = normal
        ormh_atlas[ys, xs] = ormh
        layout.append({"index": index, "bank": bank, "id": material["id"],
                       "row": row, "column": column,
                       "width_m": material.get("width_m", material.get("runtime_width_m", 2.0))})

    _save(output / "terrain_micro_albedo.png", albedo_atlas)
    _save(output / "terrain_micro_normal.png", normal_atlas)
    _save(output / "terrain_micro_ormh.png", ormh_atlas)
    metadata = {
        "grid": grid,
        "cell_size": cell_size,
        "gutter": gutter,
        "banks": {"rock": {"first": 0, "count": 9},
                  "grass": {"first": 9, "count": 6}},
        "materials": layout,
    }
    (output / "atlas.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rock-manifest", type=Path,
                        default=ROOT / "textures" / "manifest.json")
    parser.add_argument("--rock-source", type=Path,
                        default=ROOT / "textures" / "source")
    parser.add_argument("--grass-manifest", type=Path,
                        default=ROOT / "textures" / "grass_manifest.json")
    parser.add_argument("--grass-source", type=Path,
                        default=ROOT / "textures" / "grass-source")
    parser.add_argument("--output", type=Path, default=ROOT / "textures" / "runtime")
    parser.add_argument("--cell-size", type=int, default=512)
    parser.add_argument("--gutter", type=int, default=8)
    args = parser.parse_args()
    build(args.rock_manifest, args.rock_source,
          args.grass_manifest, args.grass_source,
          args.output, args.cell_size, args.gutter)
    print(f"wrote 4 x 4 rock + grass runtime atlases under {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
