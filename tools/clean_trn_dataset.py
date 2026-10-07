#!/usr/bin/env python3
"""Remove roads, buildings and water from a built terrain tile dataset.

    python3 tools/clean_trn_dataset.py SOURCE_DATASET OUTPUT_DATASET \
        [--jobs 8] [--block 8] [--margin-m 150] [--work DIR]

Works on the finest imagery level of a `terrain_tiles.py` dataset (the
256 px + gutter PNG pyramid), so it runs on anything that has been built --
no GeoTIFFs needed. The finest level is stitched into one disk-backed mosaic,
cleaned in parallel blocks with the same detector and exemplar fill as
tools/clean_orthophotos.py, and every coarser level is regenerated from it the
way the builder does (linear-light Lanczos), gutters included.

Stages, each resumable:
  1. mosaic   finest-level tile cores -> work/raw.u8 (memmap)
  2. detect   per block, with a --margin-m border read straight from the
              mosaic; each block's canvas-sized mask is saved
  3. merge    all block masks OR-ed into one mosaic mask, so a structure
              straddling a block edge is judged the same from both sides
  4. fill     per block, written into work/clean.u8
  5. pyramid  every level re-encoded to OUTPUT/imagery with gutters

OUTPUT shares SOURCE's height tiles by symlink (imagery URIs are relative).
Also writes OUTPUT/cleaning-overview.png: before | after at 2048 px.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infrastructure_colour_mask import infrastructure_mask  # noqa: E402
from infrastructure_fill import exemplar_fill  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


class Layout:
    def __init__(self, source: Path, work: Path, block: int):
        manifest = json.loads((source / "manifest.json").read_text())
        self.source, self.work, self.manifest = source, work, manifest
        self.level = manifest["levels"] - 1
        self.tiles = 1 << self.level
        self.tile = manifest["imagery_size"]
        self.gutter = manifest.get("gutter", 1)
        self.size = self.tiles * self.tile
        extent = manifest["extent"]
        self.metres_per_pixel = (extent[2] - extent[0]) / self.size
        self.block = block
        self.blocks = (self.tiles + block - 1) // block
        self.block_px = block * self.tile

    def memmap(self, name: str, dtype=np.uint8, channels: int = 3, mode: str = "r+"):
        shape = (self.size, self.size, channels) if channels > 1 else (self.size, self.size)
        return np.memmap(self.work / name, dtype=dtype, mode=mode, shape=shape)


# --- 1. mosaic -------------------------------------------------------------------

def mosaic_column(args) -> int:
    source, work, block, x = args
    lay = Layout(Path(source), Path(work), block)
    raw = lay.memmap("raw.u8")
    g, t = lay.gutter, lay.tile
    for y in range(lay.tiles):
        im = np.asarray(Image.open(lay.source / "imagery" / str(lay.level) / str(x) / f"{y}.png").convert("RGB"))
        raw[y * t:(y + 1) * t, x * t:(x + 1) * t] = im[g:g + t, g:g + t]
    raw.flush()
    return x


# --- 2. detect / 4. fill ---------------------------------------------------------

def canvas_box(lay: Layout, bx: int, by: int, margin: int):
    y0, x0 = by * lay.block_px - margin, bx * lay.block_px - margin
    y1 = min(lay.size, (by + 1) * lay.block_px) + margin
    x1 = min(lay.size, (bx + 1) * lay.block_px) + margin
    return y0, y1, x0, x1


def read_canvas(img: np.memmap, y0, y1, x0, x1) -> tuple[np.ndarray, np.ndarray]:
    """Slice with out-of-mosaic area mirror-padded and flagged unreal."""
    size = img.shape[0]
    cy0, cy1, cx0, cx1 = max(0, y0), min(size, y1), max(0, x0), min(size, x1)
    inner = np.asarray(img[cy0:cy1, cx0:cx1])
    pad = ((cy0 - y0, y1 - cy1), (cx0 - x0, x1 - cx1)) + (((0, 0),) if inner.ndim == 3 else ())
    real = np.zeros((y1 - y0, x1 - x0), bool)
    real[cy0 - y0:cy0 - y0 + inner.shape[0], cx0 - x0:cx0 - x0 + inner.shape[1]] = True
    if any(p for pair in pad[:2] for p in pair):
        inner = np.pad(inner, pad, mode="reflect" if inner.ndim == 3 else "constant")
    return inner, real


def detect_block(args) -> str:
    source, work, block, bx, by, margin = args
    lay = Layout(Path(source), Path(work), block)
    dest = lay.work / "detect" / f"{bx}_{by}.npz"
    if dest.exists():
        return dest.name
    raw = lay.memmap("raw.u8", mode="r")
    box = canvas_box(lay, bx, by, margin)
    canvas, real = read_canvas(raw, *box)
    result = infrastructure_mask(canvas, metres_per_pixel=lay.metres_per_pixel)
    np.savez_compressed(dest, mask=result["mask"] & real, donor_exclude=result["donor_exclude"],
                        box=np.array(box))
    return dest.name


def fill_block(args) -> dict:
    source, work, block, bx, by, margin = args
    lay = Layout(Path(source), Path(work), block)
    done = lay.work / "filled" / f"{bx}_{by}"
    if done.exists():
        return json.loads(done.read_text())
    raw = lay.memmap("raw.u8", mode="r")
    mask_all = lay.memmap("mask.u8", channels=1, mode="r")
    exclude_all = lay.memmap("exclude.u8", channels=1, mode="r")
    clean = lay.memmap("clean.u8")
    box = canvas_box(lay, bx, by, margin)
    canvas, real = read_canvas(raw, *box)
    mask, _ = read_canvas(mask_all, *box)
    exclude, _ = read_canvas(exclude_all, *box)
    mask = (mask > 0) & real
    filled, _ = exemplar_fill(canvas, mask, metres_per_pixel=lay.metres_per_pixel,
                              donor_exclude=exclude > 0)
    y0, x0 = box[0], box[2]
    cy0, cx0 = by * lay.block_px, bx * lay.block_px
    cy1, cx1 = min(lay.size, cy0 + lay.block_px), min(lay.size, cx0 + lay.block_px)
    clean[cy0:cy1, cx0:cx1] = filled[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    clean.flush()
    stats = {"block": [bx, by], "masked": float(mask[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0].mean())}
    done.write_text(json.dumps(stats))
    return stats


# --- 5. pyramid ------------------------------------------------------------------

def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    s = np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)
    return np.round(s * 255.0).astype(np.uint8)


def downsample_level(work: Path, src_name: str, dst_name: str, src_size: int, rows: int = 2048) -> None:
    """Halve a mosaic in linear light (2x2 box: what Lanczos converges to
    for an exact factor-of-two step, without its ringing at seams)."""
    src = np.memmap(work / src_name, dtype=np.uint8, mode="r", shape=(src_size, src_size, 3))
    dst_size = src_size // 2
    dst = np.memmap(work / dst_name, dtype=np.uint8, mode="w+", shape=(dst_size, dst_size, 3))
    for y in range(0, src_size, rows):
        lin = srgb_to_linear(np.asarray(src[y:y + rows]))
        h = lin.shape[0] // 2
        avg = lin[:2 * h].reshape(h, 2, dst_size, 2, 3).mean(axis=(1, 3))
        dst[y // 2:y // 2 + h] = linear_to_srgb(avg)
    dst.flush()


def write_level_column(args) -> int:
    work, name, level, size, tile, gutter, out, x = args
    img = np.memmap(Path(work) / name, dtype=np.uint8, mode="r", shape=(size, size, 3))
    count = size // tile
    col = Path(out) / "imagery" / str(level) / str(x)
    col.mkdir(parents=True, exist_ok=True)
    xs = np.clip(np.arange(x * tile - gutter, (x + 1) * tile + gutter), 0, size - 1)
    for y in range(count):
        ys = np.clip(np.arange(y * tile - gutter, (y + 1) * tile + gutter), 0, size - 1)
        Image.fromarray(np.asarray(img[np.ix_(ys, xs)])).save(col / f"{y}.png", compress_level=6)
    return x


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--block", type=int, default=8, help="tiles per block side")
    p.add_argument("--margin-m", type=float, default=150.0)
    p.add_argument("--work", type=Path, help="scratch dir (default OUTPUT.work)")
    a = p.parse_args()
    work = a.work or a.output.with_name(a.output.name + ".work")
    for sub in ("detect", "filled"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    lay = Layout(a.source, work, a.block)
    margin = int(round(a.margin_m / lay.metres_per_pixel))
    print(f"{a.source.name}: level {lay.level}, {lay.tiles}^2 tiles, {lay.size}^2 px at "
          f"{lay.metres_per_pixel:.2f} m/px; {lay.blocks}^2 blocks, margin {margin} px", flush=True)
    common = (str(a.source), str(work), a.block)
    blocks = [(bx, by) for by in range(lay.blocks) for bx in range(lay.blocks)]

    with ProcessPoolExecutor(a.jobs) as pool:
        if not (work / "raw.done").exists():
            lay.memmap("raw.u8", mode="w+").flush()
            list(pool.map(mosaic_column, [(*common, x) for x in range(lay.tiles)]))
            (work / "raw.done").touch()
        print("mosaic ready; detecting", flush=True)
        for i, _ in enumerate(pool.map(detect_block, [(*common, bx, by, margin) for bx, by in blocks]), 1):
            if i % max(1, len(blocks) // 20) == 0:
                print(f"  detect {i}/{len(blocks)}", flush=True)

        if not (work / "mask.done").exists():
            mask = lay.memmap("mask.u8", channels=1, mode="w+")
            excl = lay.memmap("exclude.u8", channels=1, mode="w+")
            for bx, by in blocks:
                d = np.load(work / "detect" / f"{bx}_{by}.npz")
                y0, y1, x0, x1 = (int(v) for v in d["box"])
                cy0, cy1, cx0, cx1 = max(0, y0), min(lay.size, y1), max(0, x0), min(lay.size, x1)
                sub = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
                mask[cy0:cy1, cx0:cx1] |= d["mask"][sub].astype(np.uint8)
                excl[cy0:cy1, cx0:cx1] |= d["donor_exclude"][sub].astype(np.uint8)
            mask.flush(); excl.flush()
            (work / "mask.done").touch()
            if not (work / "clean.u8").exists():
                shutil.copyfile(work / "raw.u8", work / "clean.u8")
        print("masks merged; filling", flush=True)
        total = 0.0
        for i, s in enumerate(pool.map(fill_block, [(*common, bx, by, margin) for bx, by in blocks]), 1):
            total += s["masked"]
            if i % max(1, len(blocks) // 20) == 0:
                print(f"  fill {i}/{len(blocks)}", flush=True)
        print(f"removed {total / len(blocks) * 100:.2f}% of the imagery", flush=True)

        # Pyramid.
        a.output.mkdir(parents=True, exist_ok=True)
        name, size = "clean.u8", lay.size
        for level in range(lay.level, -1, -1):
            if level != lay.level:
                nxt = f"level{level}.u8"
                downsample_level(work, name, nxt, size)
                name, size = nxt, size // 2
            list(pool.map(write_level_column, [(str(work), name, level, size, lay.tile, lay.gutter,
                                                str(a.output), x) for x in range(size // lay.tile)]))
            print(f"  level {level} written", flush=True)

    tiles_link = a.output / "tiles"
    if not tiles_link.exists():
        tiles_link.symlink_to(a.source.resolve() / "tiles")
    manifest = dict(lay.manifest)
    manifest["source"] = dict(manifest.get("source", {}),
                              cleaned="roads, buildings and water removed by tools/clean_trn_dataset.py")
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # Overview: before | after.
    raw = lay.memmap("raw.u8", mode="r")
    step = max(1, lay.size // 2048)
    before = Image.fromarray(np.asarray(raw[::step, ::step]))
    after = Image.fromarray(np.asarray(lay.memmap("clean.u8", mode="r")[::step, ::step]))
    ov = Image.new("RGB", (before.width * 2 + 16, before.height), "white")
    ov.paste(before, (0, 0)); ov.paste(after, (before.width + 16, 0))
    ov.save(a.output / "cleaning-overview.png")
    print("done:", a.output)


if __name__ == "__main__":
    main()
