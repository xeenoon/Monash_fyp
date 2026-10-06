#!/usr/bin/env python3
"""Remove roads, buildings, water and beaches from orthophoto tiles, at any scale.

    python3 tools/clean_orthophotos.py INPUT [INPUT ...] --out DIR [--jobs 8]
        [--work-res 0.5] [--margin-m 40] [--preview 1024] [--force]

INPUT is any mix of GeoTIFF files and directories (searched recursively for
*.tif / *.tiff). Each file must carry its GeoTIFF tie point and pixel size
(SWISSIMAGE does), which is how tiles are placed and how neighbours are found;
nothing is inferred from file names. When several files cover the same tile
(e.g. the 2 m and the 10 cm product), the finest one is used.

Stages, each parallel and each resumable (finished outputs are skipped unless
--force):

  1. cache     every tile box-resampled once to --work-res (default 0.5 m:
               fine enough to see a farm track, small enough to be quick)
  2. clean     per tile, with a --margin-m border borrowed from whichever
               neighbours exist so features crossing tile edges are caught
               whole and fills continue across seams:
                 detect  tools/infrastructure_colour_mask.py (colour + form,
                         no vector data)
                 fill    tools/infrastructure_fill.py (exemplar mosaic from
                         untouched terrain within ~40 m)

Outputs under --out:
  cache/<E>-<N>.npy          working-resolution RGB
  clean/<E>-<N>.png + .pgw   cleaned tile, georeferenced by world file
  mask/<E>-<N>.png           what was removed (white)
  review/<E>-<N>.jpg         raw | mask overlay | cleaned, for checking by eye
  manifest.json              per tile: source, extent, masked fractions
  preview.png                (--preview N) stitched NxN overview of all tiles
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infrastructure_colour_mask import infrastructure_mask, overlay  # noqa: E402
from infrastructure_fill import exemplar_fill  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


# --- Discovery ----------------------------------------------------------------

def geo_info(path: Path) -> dict | None:
    """Tie point (top-left, map units) and pixel size from GeoTIFF tags."""
    try:
        with Image.open(path) as im:
            tags = im.tag_v2
            tie, scale = tags.get(33922), tags.get(33550)
            if not tie or not scale:
                return None
            return {"path": str(path), "x0": float(tie[3]), "y0": float(tie[4]),
                    "pixel": float(scale[0]), "width": im.size[0], "height": im.size[1]}
    except Exception:
        return None


def discover(inputs: list[str]) -> dict[tuple[int, int], dict]:
    """Tiles keyed by (west, north) of their top-left corner in metres,
    choosing the finest source for each footprint."""
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += [f for f in p.rglob("*") if f.suffix.lower() in (".tif", ".tiff")]
        elif p.is_file():
            files.append(p)
    tiles: dict[tuple[int, int], dict] = {}
    for f in sorted(files):
        info = geo_info(f)
        if info is None:
            print(f"skip (no georeference): {f}", file=sys.stderr)
            continue
        info["size_m"] = info["pixel"] * info["width"]
        key = (int(round(info["x0"])), int(round(info["y0"])))
        if key not in tiles or info["pixel"] < tiles[key]["pixel"]:
            tiles[key] = info
    return tiles


def tile_name(key: tuple[int, int], size_m: float) -> str:
    """Swisstopo-style km name of the tile's south-west corner, e.g. 2640-1160."""
    return f"{key[0] // 1000}-{int(round(key[1] - size_m)) // 1000}"


# --- Stage 1: cache ------------------------------------------------------------

def cache_tile(info: dict, cache_dir: str, work_res: float, force: bool) -> str:
    name = tile_name((int(round(info["x0"])), int(round(info["y0"]))), info["size_m"])
    dest = Path(cache_dir) / f"{name}.npy"
    if dest.exists() and not force:
        return name
    size = int(round(info["size_m"] / work_res))
    with Image.open(info["path"]) as im:
        im.draft("RGB", (size, size))  # JPEG-in-TIFF decodes at reduced scale
        rgb = im.convert("RGB")
        if rgb.size != (size, size):
            rgb = rgb.resize((size, size), Image.BOX if rgb.size[0] > size else Image.BICUBIC)
    np.save(dest, np.asarray(rgb))
    return name


# --- Stage 2: clean --------------------------------------------------------------

def padded(name: str, cache_dir: Path, margin: int) -> tuple[np.ndarray, np.ndarray]:
    """The tile with `margin` pixels of its 8 neighbours around it, and a mask
    of which padding pixels are real (missing neighbours are mirror-filled and
    excluded from detection)."""
    e, n = (int(v) for v in name.split("-"))
    centre = np.load(cache_dir / f"{name}.npy")
    s = centre.shape[0]
    canvas = np.zeros((s + 2 * margin, s + 2 * margin, 3), np.uint8)
    real = np.zeros(canvas.shape[:2], bool)
    for dy in (-1, 0, 1):          # rows: north (-1) to south (+1)
        for dx in (-1, 0, 1):
            path = cache_dir / f"{e + dx}-{n - dy}.npy"
            if not path.exists():
                continue
            tile = centre if (dx, dy) == (0, 0) else np.load(path, mmap_mode="r")
            if tile.shape[0] != s:
                continue
            # The part of this tile that lands in the canvas.
            ys = slice(max(0, margin + dy * s), min(canvas.shape[0], margin + dy * s + s))
            xs = slice(max(0, margin + dx * s), min(canvas.shape[1], margin + dx * s + s))
            ty = slice(ys.start - (margin + dy * s), ys.stop - (margin + dy * s))
            tx = slice(xs.start - (margin + dx * s), xs.stop - (margin + dx * s))
            canvas[ys, xs] = tile[ty, tx]
            real[ys, xs] = True
    if not real.all():
        # Mirror the centre into absent neighbours so filters see plausible
        # surroundings; those pixels never donate or get detected.
        mirrored = np.pad(centre, ((margin, margin), (margin, margin), (0, 0)), mode="reflect")
        canvas[~real] = mirrored[~real]
    return canvas, real


def clean_tile(name: str, out: str, work_res: float, margin_m: float, force: bool) -> dict:
    out_dir = Path(out)
    dest = out_dir / "clean" / f"{name}.png"
    stats_path = out_dir / "stats" / f"{name}.json"
    if dest.exists() and stats_path.exists() and not force:
        return json.loads(stats_path.read_text())
    margin = int(round(margin_m / work_res))
    canvas, real = padded(name, out_dir / "cache", margin)
    result = infrastructure_mask(canvas, metres_per_pixel=work_res)
    mask = result["mask"] & real
    filled, _ = exemplar_fill(canvas, mask, metres_per_pixel=work_res,
                              donor_exclude=result["donor_exclude"])
    core = (slice(margin, canvas.shape[0] - margin), slice(margin, canvas.shape[1] - margin))
    raw, clean, cmask = canvas[core], filled[core], mask[core]
    Image.fromarray(clean).save(dest)
    Image.fromarray((cmask * 255).astype(np.uint8)).save(out_dir / "mask" / f"{name}.png")
    view = {k: v[core] for k, v in result.items() if k != "donor_exclude"}
    review = np.concatenate([raw, overlay(raw, view), clean], axis=1)
    Image.fromarray(review).resize((review.shape[1] // 2, review.shape[0] // 2),
                                   Image.LANCZOS).save(out_dir / "review" / f"{name}.jpg", quality=88)
    e, n = (int(v) for v in name.split("-"))
    size_m = raw.shape[0] * work_res
    # World file: pixel size, rotation terms, then the centre of the top-left pixel.
    (out_dir / "clean" / f"{name}.pgw").write_text(
        f"{work_res}\n0\n0\n{-work_res}\n{e * 1000 + work_res / 2}\n{n * 1000 + size_m - work_res / 2}\n")
    stats = {"tile": name, "masked": float(cmask.mean()),
             "road": float(view["road"].mean()), "building": float(view["building"].mean()),
             "water": float((view["water"] | view["beach"]).mean())}
    stats_path.write_text(json.dumps(stats))
    return stats


# --- Preview -----------------------------------------------------------------------

def preview(out_dir: Path, names: list[str], size: int) -> Path:
    keys = [tuple(int(v) for v in n.split("-")) for n in names]
    es, ns = sorted({k[0] for k in keys}), sorted({k[1] for k in keys}, reverse=True)
    cell = max(1, size // max(len(es), len(ns)))
    canvas = Image.new("RGB", (cell * len(es), cell * len(ns)))
    for name, (e, n) in zip(names, keys):
        tile = Image.open(out_dir / "clean" / f"{name}.png").resize((cell, cell), Image.LANCZOS)
        canvas.paste(tile, (es.index(e) * cell, ns.index(n) * cell))
    dest = out_dir / "preview.png"
    canvas.resize((size, size) if len(es) == len(ns) else canvas.size, Image.LANCZOS).save(dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="GeoTIFF files and/or directories")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--work-res", type=float, default=0.5, help="metres per pixel")
    parser.add_argument("--margin-m", type=float, default=40.0,
                        help="neighbour context around each tile")
    parser.add_argument("--preview", type=int, default=0, metavar="N",
                        help="also write an NxN stitched preview of every tile")
    parser.add_argument("--force", action="store_true", help="redo finished outputs")
    args = parser.parse_args()

    for sub in ("cache", "clean", "mask", "review", "stats"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    tiles = discover(args.inputs)
    if not tiles:
        sys.exit("no georeferenced tiles found")
    print(f"{len(tiles)} tiles; caching at {args.work_res} m with {args.jobs} jobs", flush=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        names = list(pool.map(cache_tile, tiles.values(), [str(args.out / "cache")] * len(tiles),
                              [args.work_res] * len(tiles), [args.force] * len(tiles)))
        print("cleaning", flush=True)
        stats = []
        for done, s in enumerate(pool.map(clean_tile, names, [str(args.out)] * len(names),
                                          [args.work_res] * len(names),
                                          [args.margin_m] * len(names),
                                          [args.force] * len(names)), start=1):
            stats.append(s)
            print(f"  [{done}/{len(names)}] {s['tile']}: removed {s['masked'] * 100:.2f}% "
                  f"(roads {s['road'] * 100:.2f}%, buildings {s['building'] * 100:.2f}%, "
                  f"water+shore {s['water'] * 100:.2f}%)", flush=True)
    manifest = {"work_res_m": args.work_res, "margin_m": args.margin_m,
                "tiles": {n: {**tiles[k], **s} for k, n, s in
                          zip(tiles.keys(), names, sorted(stats, key=lambda s: names.index(s["tile"])))}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if args.preview:
        print("preview:", preview(args.out, names, args.preview))


if __name__ == "__main__":
    main()
