#!/usr/bin/env python3
"""Flag AI-corrected terrain masters that diverge from the real satellite
photo they were derived from -- colour hallucination or, worse, an entirely
different scene.

Problem this solves: masters like alps-data/trn-alps-16km/redrepair-masters/
were produced by running each 2km tile through an external AI tool
independently (same lineage as tools/grade_shadowfree_masters.py's shadow-
removal masters). Most tiles come out fine, but every so often one comes back
wrong -- and manually eyeballing 64+ tiles against a reference doesn't scale.
This compares each corrected master against the real SWISSIMAGE orthophoto
tiles it was made from (alps-data/alps-hires-20g/imagery/, the earliest,
un-edited source available -- verified to reproduce
alps-data/trn-alps-16km-shadowed/shadowed-masters/ to within ~3/255 mean
channel error) and reports which ones diverge.

Two independent signals were validated by hand against ~15 spot-checked
tiles before being wired up here (see docs/graded_masters_pipeline.md for the
adjacent grading pipeline; this audit is upstream of it -- catch a bad master
before spending a grading pass on it):

  1. purple_fraction_delta -- share of pixels reading as purple/magenta/violet
     (hue band, saturation- and value-gated so dim near-grey noise doesn't
     count) in the corrected master minus the same share in the real photo.
     Catches vivid purple hallucination directly. High precision on its own.

  2. structure_corr, gated to snow-dominated references -- every confirmed
     failure on this dataset is the same shape: a real photo that's mostly
     glacier/snow (bright, low-saturation) comes back as a completely
     different low-elevation scene (forest, meadow), sometimes with a purple
     cast and sometimes just a lavender/mauve tint that's too desaturated to
     trip signal 1. Plain colour statistics can't tell "shadow correctly
     brightened" from "wrong scene entirely" -- but coarse spatial structure
     can: shrink both images to a small grayscale thumbnail and correlate
     them. A real photo and its legitimately shadow-corrected version still
     share their large-scale relief silhouette (correlation ~0.6-0.9 in
     spot checks); a wrong-scene replacement does not (correlation ~-0.2 to
     0.2). This signal is *unreliable* on non-snow references -- dense,
     uniformly-textured forest has no large-scale structure for a coarse
     correlation to lock onto, so it reads noisy-low even when the tile is
     fine (confirmed on a plain-forest false positive during validation).
     Gating on `snow_fraction_reference >= SNOW_GATE` keeps it to the regime
     it was actually validated in.

Anything that trips either signal gets a side-by-side reference|corrected
preview written out, since both signals have some false-positive rate and
the whole point is a short list a human can glance at, not a verdict.

Usage:
    python3 tools/audit_corrected_texture_colors.py \\
        --corrected-dir alps-data/trn-alps-16km/redrepair-masters \\
        --output /tmp/redrepair-color-audit

Then look at <output>/report.csv (worst-first) and
<output>/flagged/*_compare.png (reference | corrected, flagged tiles only).
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # source orthophoto cells are large, legitimate TIFFs

ROOT = Path(__file__).resolve().parents[1]
COORD = re.compile(r"^E(\d+)-(\d+)_N(\d+)-(\d+)\.png$")
CELL_METERS = 1000
CELL_PX = 512  # reference resolution per 1km cell before assembly

PURPLE_HUE_MIN = 0.68
PURPLE_HUE_MAX = 0.92
PURPLE_MIN_SATURATION = 0.15
PURPLE_MIN_VALUE = 0.12
PURPLE_FLAG_DELTA = 0.03  # 3 percentage points of newly-purple pixels

SNOW_MIN_VALUE = 0.75
SNOW_MAX_SATURATION = 0.15
SNOW_GATE = 0.15          # only trust structure_corr when the real photo is at least this snowy

STRUCTURE_SIZE = 48       # coarse grayscale thumbnail side, for the structural correlation
STRUCTURE_FLAG = 0.5      # below this, a snow-gated tile is a probable wrong-scene replacement


def parse_extent(path: Path) -> tuple[int, int, int, int]:
    match = COORD.match(path.name)
    if not match:
        raise ValueError(f"not a coordinate-named master: {path.name}")
    return tuple(int(value) for value in match.groups())  # east_min, east_max, north_min, north_max


def rgb_to_hsv(image01: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = image01[..., 0], image01[..., 1], image01[..., 2]
    maxc = np.max(image01, axis=-1)
    minc = np.min(image01, axis=-1)
    value = maxc
    delta = maxc - minc
    saturation = np.divide(delta, maxc, out=np.zeros_like(maxc), where=maxc > 1e-6)

    hue = np.zeros_like(maxc)
    safe_delta = np.where(delta > 1e-9, delta, 1.0)
    is_r = (maxc == r) & (delta > 1e-9)
    is_g = (maxc == g) & (delta > 1e-9) & ~is_r
    is_b = (maxc == b) & (delta > 1e-9) & ~is_r & ~is_g
    hue[is_r] = (((g - b) / safe_delta) % 6)[is_r]
    hue[is_g] = (((b - r) / safe_delta) + 2)[is_g]
    hue[is_b] = (((r - g) / safe_delta) + 4)[is_b]
    hue = hue / 6.0
    return hue, saturation, value


def purple_fraction(hue: np.ndarray, saturation: np.ndarray, value: np.ndarray) -> float:
    mask = ((hue >= PURPLE_HUE_MIN) & (hue <= PURPLE_HUE_MAX) &
            (saturation >= PURPLE_MIN_SATURATION) & (value >= PURPLE_MIN_VALUE))
    return float(mask.mean())


def snow_fraction(saturation: np.ndarray, value: np.ndarray) -> float:
    return float(((value >= SNOW_MIN_VALUE) & (saturation <= SNOW_MAX_SATURATION)).mean())


def structure_correlation(reference01: np.ndarray, corrected01: np.ndarray, size: int = STRUCTURE_SIZE) -> float:
    """Normalized cross-correlation of coarse grayscale thumbnails. High for the
    same scene under different lighting/colour; near zero or negative for a
    wholesale scene swap. Unreliable when there's no large-scale structure to
    begin with (see module docstring) -- caller must gate on that."""
    def coarse_gray(image01: np.ndarray) -> np.ndarray:
        image_u8 = (np.clip(image01, 0, 1) * 255).astype(np.uint8)
        small = Image.fromarray(image_u8).convert("L").resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(small, dtype=np.float64).ravel()

    a, b = coarse_gray(reference01), coarse_gray(corrected01)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 1e-9 else 0.0


def find_cell(imagery_dir: Path, east_km: int, north_km: int) -> Path | None:
    matches = glob.glob(str(imagery_dir / f"image_swissimage*_{east_km}-{north_km}.tif"))
    return Path(matches[0]) if matches else None


def build_reference(imagery_dir: Path, extent: tuple[int, int, int, int],
                    output_size: tuple[int, int]) -> np.ndarray | None:
    east_min, east_max, north_min, north_max = extent
    cells_x = (east_max - east_min) // CELL_METERS
    cells_y = (north_max - north_min) // CELL_METERS
    canvas = Image.new("RGB", (cells_x * CELL_PX, cells_y * CELL_PX))
    for row in range(cells_y):
        north_cell = north_max // CELL_METERS - 1 - row  # row 0 = northernmost = top
        for col in range(cells_x):
            east_cell = east_min // CELL_METERS + col
            cell_path = find_cell(imagery_dir, east_cell, north_cell)
            if cell_path is None:
                return None
            with Image.open(cell_path) as cell_image:
                resized = cell_image.convert("RGB").resize((CELL_PX, CELL_PX), Image.Resampling.LANCZOS)
            canvas.paste(resized, (col * CELL_PX, row * CELL_PX))
    if canvas.size != output_size:
        canvas = canvas.resize(output_size, Image.Resampling.LANCZOS)
    return np.asarray(canvas).astype(np.float32) / 255.0


def save_compare(path: Path, reference01: np.ndarray, corrected01: np.ndarray) -> None:
    reference_u8 = (np.clip(reference01, 0, 1) * 255).astype(np.uint8)
    corrected_u8 = (np.clip(corrected01, 0, 1) * 255).astype(np.uint8)
    gap = np.full((reference_u8.shape[0], 8, 3), 255, dtype=np.uint8)
    combined = np.concatenate([reference_u8, gap, corrected_u8], axis=1)
    Image.fromarray(combined, "RGB").save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corrected-dir", type=Path,
                        default=ROOT / "alps-data" / "trn-alps-16km" / "redrepair-masters",
                        help="directory of coordinate-named AI-corrected masters to audit")
    parser.add_argument("--reference-imagery-dir", type=Path,
                        default=ROOT / "alps-data" / "alps-hires-20g" / "imagery",
                        help="directory of raw image_swissimage*_<e>-<n>.tif 1km source cells")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--purple-flag-delta", type=float, default=PURPLE_FLAG_DELTA)
    parser.add_argument("--snow-gate", type=float, default=SNOW_GATE)
    parser.add_argument("--structure-flag", type=float, default=STRUCTURE_FLAG)
    args = parser.parse_args()

    masters = sorted(args.corrected_dir.glob("E*_N*.png"))
    if not masters:
        raise SystemExit(f"no coordinate-named masters in {args.corrected_dir}")

    rows: list[dict[str, object]] = []
    skipped: list[str] = []
    for index, path in enumerate(masters, start=1):
        extent = parse_extent(path)
        with Image.open(path) as image:
            corrected01 = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
        reference01 = build_reference(args.reference_imagery_dir, extent,
                                      (corrected01.shape[1], corrected01.shape[0]))
        if reference01 is None:
            skipped.append(path.name)
            print(f"[{index}/{len(masters)}] {path.name}: SKIPPED (missing source imagery cell)")
            continue

        hue_ref, sat_ref, val_ref = rgb_to_hsv(reference01)
        hue_corr, sat_corr, val_corr = rgb_to_hsv(corrected01)
        purple_ref = purple_fraction(hue_ref, sat_ref, val_ref)
        purple_corr = purple_fraction(hue_corr, sat_corr, val_corr)
        purple_delta = purple_corr - purple_ref
        snow_ref = snow_fraction(sat_ref, val_ref)
        structure_corr = structure_correlation(reference01, corrected01)

        purple_hit = purple_delta > args.purple_flag_delta
        scene_hit = snow_ref >= args.snow_gate and structure_corr < args.structure_flag
        reason = "purple" if purple_hit else ("scene_mismatch" if scene_hit else "")
        if purple_hit and scene_hit:
            reason = "purple+scene_mismatch"

        rows.append({
            "name": path.name,
            "flagged": bool(purple_hit or scene_hit),
            "reason": reason,
            "purple_fraction_reference": purple_ref,
            "purple_fraction_corrected": purple_corr,
            "purple_fraction_delta": purple_delta,
            "snow_fraction_reference": snow_ref,
            "structure_corr": structure_corr,
        })
        print(f"[{index}/{len(masters)}] {path.name}: purple_delta={purple_delta:+.3f}  "
              f"snow_ref={snow_ref:.3f}  structure_corr={structure_corr:+.3f}"
              + (f"  <-- {reason}" if reason else ""))

    if not rows:
        raise SystemExit("no tiles had matching reference imagery -- nothing to report")

    def severity(row: dict[str, object]) -> float:
        if not row["flagged"]:
            return -1.0
        score = 0.0
        if "purple" in str(row["reason"]):
            score = max(score, 1.0 + float(row["purple_fraction_delta"]))
        if "scene_mismatch" in str(row["reason"]):
            score = max(score, 0.5 + (args.structure_flag - float(row["structure_corr"])))
        return score

    rows.sort(key=severity, reverse=True)

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    flagged = [row for row in rows if row["flagged"]]
    if flagged:
        flagged_dir = args.output / "flagged"
        flagged_dir.mkdir(exist_ok=True)
        for row in flagged:
            path = args.corrected_dir / str(row["name"])
            extent = parse_extent(path)
            with Image.open(path) as image:
                corrected01 = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
            reference01 = build_reference(args.reference_imagery_dir, extent,
                                          (corrected01.shape[1], corrected01.shape[0]))
            save_compare(flagged_dir / f"{path.stem}_compare.png", reference01, corrected01)

    print(f"\n{len(rows)} tiles analysed, {len(skipped)} skipped (no source imagery)")
    print(f"\n{len(flagged)} tile(s) flagged for regeneration:")
    for row in flagged:
        print(f"  {row['name']}  [{row['reason']}]  purple_delta={row['purple_fraction_delta']:+.3f}  "
              f"snow_ref={row['snow_fraction_reference']:.3f}  structure_corr={row['structure_corr']:+.3f}")
    if skipped:
        print(f"\nskipped (no matching source cell): {', '.join(skipped)}")
    print(f"\nwrote {csv_path}")
    if flagged:
        print(f"side-by-side previews: {args.output / 'flagged'}/*_compare.png")


if __name__ == "__main__":
    main()
