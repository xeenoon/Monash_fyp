#!/usr/bin/env python3
"""Tile-agnostic procedural re-colouring of the terrain macro colour map.

Scope (deliberately narrow): this owns *colour only*.  The material/biome
*shape* — where rock/grass/snow go — is taken as given from the DEM heightmap
recorded in the dataset and is out of scope here.

Everything is a pure function of **world position** (metres), so adjacent tiles
evaluate the same field at their shared edge and agree automatically: seamless
with zero neighbour anchoring, extendable to infinity, and portable to a GPU
shader later (offline bake now == on-stream evaluation later).

Colour model (v1: learned-palette + noise):
  * Each material has an *extensible* set of palettes (default 5) in
    ``terrain_palettes.json``, seeded from real imagery but freely editable.
  * A low-frequency, domain-warped fractal field per material selects a smooth
    blend among that material's palettes — this is the "areas of grey vs orange
    vs green rock" macro selector, drifting continuously over the world.
  * Fine world-space colour noise breaks up the flat palette wash.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from terrain_tiles import read_tile  # noqa: E402

SIZE = 256
PALETTE_PATH = ROOT / "tools" / "terrain_palettes.json"

# --- macro selector tuning (world metres) ---
BIOME_SCALE_M = 6000.0      # size of a palette region (grey vs orange zones)
WARP_STRENGTH = 0.6         # domain-warp of the selector, in lattice units
SELECTOR_TEMP = 0.14        # softmax temperature; smaller = sharper regions
FINE_SCALE_M = 45.0         # within-region colour texture scale
FINE_AMP = 0.045            # within-region colour texture amplitude
SEED = {"rock": 1201, "grass": 5507, "snow": 9109}

# --- material (biome) from DEM: GIVEN input, placeholder thresholds only ---
SNOW_LO_M, SNOW_HI_M = 2400.0, 2850.0
ROCK_LO_SLOPE, ROCK_HI_SLOPE = 0.45, 1.05


# ----------------------------------------------------------------------------
# World-space value-noise / fBm — deterministic pure function of world coords.
# ----------------------------------------------------------------------------
def _hash2(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    h = (ix.astype(np.int64) * np.int64(374761393) +
         iy.astype(np.int64) * np.int64(668265263) +
         np.int64(seed) * np.int64(2246822519)) & np.int64(0x7FFFFFFFFFFFFFFF)
    h = ((h ^ (h >> 13)) * np.int64(1274126177)) & np.int64(0x7FFFFFFFFFFFFFFF)
    return (h % np.int64(1_000_003)).astype(np.float64) / 1_000_003.0


def value_noise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = x - x0, y - y0
    ix, iy = x0.astype(np.int64), y0.astype(np.int64)
    ux, uy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    v00 = _hash2(ix, iy, seed);       v10 = _hash2(ix + 1, iy, seed)
    v01 = _hash2(ix, iy + 1, seed);   v11 = _hash2(ix + 1, iy + 1, seed)
    top = v00 * (1 - ux) + v10 * ux
    bottom = v01 * (1 - ux) + v11 * ux
    return top * (1 - uy) + bottom * uy


def fbm(x: np.ndarray, y: np.ndarray, seed: int, octaves: int = 4,
        lacunarity: float = 2.0, gain: float = 0.5) -> np.ndarray:
    total = np.zeros_like(x, dtype=np.float64)
    amplitude, frequency, norm = 0.5, 1.0, 0.0
    for octave in range(octaves):
        total += amplitude * value_noise(x * frequency, y * frequency, seed + octave * 1013)
        norm += amplitude
        amplitude *= gain
        frequency *= lacunarity
    return total / norm


def domain_warp(x: np.ndarray, y: np.ndarray, seed: int,
                strength: float) -> tuple[np.ndarray, np.ndarray]:
    warp_x = fbm(x, y, seed + 71) * 2.0 - 1.0
    warp_y = fbm(x, y, seed + 131) * 2.0 - 1.0
    return x + strength * warp_x, y + strength * warp_y


def smoothstep(low: float, high: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - low) / max(high - low, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ----------------------------------------------------------------------------
# Palette selector + per-material colour.
# ----------------------------------------------------------------------------
def palette_weights(world_x: np.ndarray, world_y: np.ndarray, count: int,
                    seed: int) -> np.ndarray:
    """Smooth per-palette blend weights that drift across the world (…, count)."""
    xw, yw = domain_warp(world_x / BIOME_SCALE_M, world_y / BIOME_SCALE_M,
                         seed, WARP_STRENGTH)
    fields = np.stack([fbm(xw, yw, seed + p * 9973) for p in range(count)], axis=-1)
    fields = fields / SELECTOR_TEMP
    fields -= fields.max(axis=-1, keepdims=True)
    weights = np.exp(fields)
    return weights / weights.sum(axis=-1, keepdims=True)


def material_colour(world_x: np.ndarray, world_y: np.ndarray,
                    palettes: list[dict], seed: int) -> np.ndarray:
    base = np.asarray([p["rgb"] for p in palettes], dtype=np.float32) / 255.0
    weights = palette_weights(world_x, world_y, len(palettes), seed)
    colour = weights @ base
    fine = (fbm(world_x / FINE_SCALE_M, world_y / FINE_SCALE_M, seed + 555)[..., None]
            - 0.5) * FINE_AMP
    return np.clip(colour + fine, 0.0, 1.0)


# ----------------------------------------------------------------------------
# DEM (given biome shape) → soft material weights.
# ----------------------------------------------------------------------------
def load_height_slope(dataset: Path, x: int, y: int) -> tuple[np.ndarray, np.ndarray,
                                                              tuple[float, float, float, float]]:
    tile = read_tile(dataset / "tiles" / "5" / str(x) / f"{y}.trn")
    heights = np.asarray(tile.decoded_heights(), dtype=np.float32).reshape(tile.height, tile.width)
    g = tile.gutter
    heights = heights[g:-g, g:-g]
    height = np.asarray(Image.fromarray(heights, "F").resize((SIZE, SIZE), Image.Resampling.BILINEAR))
    metres_per_sample = (tile.extent[2] - tile.extent[0]) / max(heights.shape[1] - 1, 1)
    dz_dy, dz_dx = np.gradient(height, metres_per_sample)
    slope = np.hypot(dz_dx, dz_dy)
    return height, slope, tile.extent


def material_weights(height: np.ndarray, slope: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    snow = smoothstep(SNOW_LO_M, SNOW_HI_M, height)
    rock = smoothstep(ROCK_LO_SLOPE, ROCK_HI_SLOPE, slope) * (1.0 - snow)
    grass = np.clip(1.0 - snow - rock, 0.0, 1.0)
    return grass, rock, snow


def world_coordinates(extent: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    west, south, east, north = extent
    py, px = np.indices((SIZE, SIZE))
    world_x = west + (px + 0.5) / SIZE * (east - west)
    world_y = north - (py + 0.5) / SIZE * (north - south)
    return world_x, world_y


def bake_tile(dataset: Path, x: int, y: int, palettes: dict) -> np.ndarray:
    """Return the recoloured macro map (RGBA uint8) for one tile."""
    height, slope, extent = load_height_slope(dataset, x, y)
    world_x, world_y = world_coordinates(extent)
    grass, rock, snow = material_weights(height, slope)
    colour = (grass[..., None] * material_colour(world_x, world_y, palettes["grass"], SEED["grass"]) +
              rock[..., None] * material_colour(world_x, world_y, palettes["rock"], SEED["rock"]) +
              snow[..., None] * material_colour(world_x, world_y, palettes["snow"], SEED["snow"]))
    rgb = np.rint(np.clip(colour, 0.0, 1.0) * 255.0).astype(np.uint8)
    alpha = np.rint(np.clip(grass, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.dstack((rgb, alpha))


# ----------------------------------------------------------------------------
# Palette learning from real imagery (bootstrap the editable config).
# ----------------------------------------------------------------------------
def learn_palettes(dataset: Path, tiles: list[tuple[int, int]], count: int = 5) -> dict:
    from scipy.cluster.vq import kmeans2
    sys.path.insert(0, str(ROOT / "tools"))
    from build_terrain_macro import classify_materials  # noqa: E402

    buckets: dict[str, list[np.ndarray]] = {"rock": [], "grass": [], "snow": []}
    for x, y in tiles:
        path = dataset / "imagery" / "5" / str(x) / f"{y}.png"
        if not path.exists():
            continue
        rgb = np.asarray(Image.open(path).convert("RGB"))[1:-1, 1:-1]
        grass_mask, rock_mask, _ = classify_materials(rgb)
        unit = rgb.astype(np.float32) / 255.0
        high, low = unit.max(2), unit.min(2)
        saturation = (high - low) / np.maximum(high, 1e-5)
        luminance = unit @ np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)
        snow_mask = (luminance > 0.64) & (saturation < 0.28)
        buckets["snow"].append(rgb[snow_mask])
        buckets["grass"].append(rgb[grass_mask & ~snow_mask])
        buckets["rock"].append(rgb[rock_mask & ~snow_mask])

    rng = np.random.default_rng(0)
    palettes: dict[str, list[dict]] = {}
    for material, chunks in buckets.items():
        pixels = np.concatenate([c for c in chunks if len(c)], axis=0).astype(np.float64)
        if len(pixels) > 40000:
            pixels = pixels[rng.choice(len(pixels), 40000, replace=False)]
        centroids, _ = kmeans2(pixels, count, seed=0, minit="++")
        order = np.argsort(centroids @ np.array((0.2126, 0.7152, 0.0722)))
        palettes[material] = [
            {"name": f"{material}_{i}", "rgb": [int(round(v)) for v in centroids[c]]}
            for i, c in enumerate(order)
        ]
    return palettes


def load_palettes(path: Path = PALETTE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["materials"]


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "alps-data" / "trn-alps-16km")
    sub = parser.add_subparsers(dest="command", required=True)

    learn = sub.add_parser("learn", help="cluster real imagery into an editable palette config")
    learn.add_argument("--count", type=int, default=5)
    learn.add_argument("--tiles", type=int, default=6, help="grid radius of tiles to sample")

    bake = sub.add_parser("bake-block", help="bake an NxN tile block and write a mosaic")
    bake.add_argument("--origin", type=int, nargs=2, default=(19, 9), metavar=("X", "Y"))
    bake.add_argument("--size", type=int, default=3)
    bake.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "learn":
        tiles = [(x, y) for x in range(17, 17 + args.tiles) for y in range(8, 8 + args.tiles)]
        palettes = learn_palettes(args.dataset, tiles, args.count)
        PALETTE_PATH.write_text(json.dumps({"materials": palettes}, indent=2), encoding="utf-8")
        print(f"wrote {PALETTE_PATH}")
        for material, entries in palettes.items():
            print(f"  {material}: " + ", ".join(str(e["rgb"]) for e in entries))
        return

    if args.command == "bake-block":
        palettes = load_palettes()
        ox, oy = args.origin
        rows = []
        for y in range(oy, oy + args.size):
            row = [bake_tile(args.dataset, x, y, palettes)[:, :, :3]
                   for x in range(ox, ox + args.size)]
            rows.append(np.concatenate(row, axis=1))
        mosaic = np.concatenate(rows, axis=0)
        args.output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mosaic, "RGB").save(args.output / "recolor_block.png")
        # seam-continuity check: max colour step across every internal tile edge
        steps = []
        for k in range(1, args.size):
            col = k * SIZE
            steps.append(float(np.abs(mosaic[:, col].astype(int) - mosaic[:, col - 1].astype(int)).mean()))
            row = k * SIZE
            steps.append(float(np.abs(mosaic[row].astype(int) - mosaic[row - 1].astype(int)).mean()))
        print(f"wrote {args.output/'recolor_block.png'}  ({args.size}x{args.size} tiles)")
        print(f"mean colour step across internal tile seams: {np.mean(steps):.3f} / 255")
        return


if __name__ == "__main__":
    main()
