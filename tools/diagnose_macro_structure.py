#!/usr/bin/env python3
"""Macro-structure diagnostic for baked material tiles.

"Looks fine at 256, mush at 32" is the baker's core failure: it reproduces
fine texture energy but not the kilometre-scale landform structure. This tool
makes that measurable. For a generated tile (and an optional real reference) it:

  * writes a downsample pyramid  <name>_128/64/32/16.png  (the eyeball test),
  * computes radial-PSD band energy in a macro band (128..32 px) and a fine
    band (16..4 px), reported as a ratio to the real reference,
  * FAILS if the macro/fine energy ratio is far from the reference -- i.e. the
    tile spends its energy on grit instead of structure.

Example:
    python3 tools/diagnose_macro_structure.py build-bake/full_rock.png \
        --real alps-data/trn-alps-16km/imagery/5/20/12.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label

Image.MAX_IMAGE_PIXELS = None


def load(p: Path, size: int = 256) -> Image.Image:
    return Image.open(p).convert("RGB").resize((size, size))


def radial_psd(im: Image.Image) -> np.ndarray:
    g = np.asarray(im.convert("L"), float)
    g = g - g.mean()
    g *= np.hanning(g.shape[0])[:, None] * np.hanning(g.shape[1])[None, :]
    f = np.fft.fftshift(np.abs(np.fft.fft2(g)) ** 2)
    cy, cx = np.array(f.shape) // 2
    y, x = np.indices(f.shape)
    r = np.hypot(x - cx, y - cy).astype(int)
    return np.bincount(r.ravel(), f.ravel()) / np.maximum(np.bincount(r.ravel()), 1)


def band(psd: np.ndarray, lo: int, hi: int) -> float:
    """Energy for feature scales in [lo,hi] px (freq = 256/px cycles/image)."""
    f_lo, f_hi = 256 // hi, 256 // lo   # px->cycles
    return float(np.sum(psd[max(f_lo, 1):f_hi + 1]))


def write_pyramid(im: Image.Image, out_base: Path):
    for n in (128, 64, 32, 16):
        p = out_base.with_name(f"{out_base.stem}_{n}{out_base.suffix}")
        im.resize((n, n), Image.BOX).resize((256, 256), Image.NEAREST).save(p)


CLASS_COLORS = np.array(((125, 125, 125), (46, 150, 62), (245, 245, 245)), np.int16)


def circular_max_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Maximum normalized correlation over every circular translation."""
    first = np.asarray(first, np.float64)
    second = np.asarray(second, np.float64)
    first = (first - first.mean()) / max(float(first.std()), 1.0e-8)
    second = (second - second.mean()) / max(float(second.std()), 1.0e-8)
    correlation = np.fft.ifft2(np.fft.fft2(first) * np.conj(np.fft.fft2(second))).real
    return float(np.max(correlation) / first.size)


def _label_image(path: Path, size: int) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB").resize(
        (size, size), Image.Resampling.NEAREST), np.int16)
    difference = rgb[:, :, None, :] - CLASS_COLORS[None, None, :, :]
    return np.sum(difference * difference, axis=3).argmin(2).astype(np.uint8)


def label_max_correlation(first: np.ndarray, second: np.ndarray) -> float:
    values = []
    for material in range(3):
        a, b = first == material, second == material
        if a.any() and (~a).any() and b.any() and (~b).any():
            values.append(circular_max_correlation(a, b))
    return float(np.mean(values)) if values else 0.0


def orientation_histogram(image: Image.Image, bins: int = 18) -> np.ndarray:
    grey = np.asarray(image.convert("L").resize((256, 256)), np.float32)
    gy, gx = np.gradient(grey)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    magnitude = np.hypot(gx, gy)
    histogram, _ = np.histogram(angle, np.linspace(0, np.pi, bins + 1), weights=magnitude)
    return histogram / max(float(histogram.sum()), 1.0e-8)


def component_statistics(labels: np.ndarray) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for material, name in enumerate(("rock", "grass", "snow")):
        components, count = label(labels == material, structure=np.ones((3, 3), np.uint8))
        areas = np.bincount(components.ravel())[1:] if count else np.empty(0)
        result[name] = {
            "count": int(count),
            "median_area": float(np.median(areas)) if len(areas) else 0.0,
            "p90_area": float(np.percentile(areas, 90)) if len(areas) else 0.0,
            "coverage": float(np.mean(labels == material)),
        }
    return result


def compare_images(generated_path: Path, reference_path: Path,
                   generated_labels_path: Path | None = None,
                   reference_labels_path: Path | None = None) -> dict[str, object]:
    """Measure high structural similarity and low positional similarity separately."""
    generated, reference = load(generated_path), load(reference_path)
    generated_full = np.asarray(Image.open(generated_path).convert("L"), np.float32)
    reference_full = np.asarray(Image.open(reference_path).convert("L"), np.float32)
    generated_gradient = float(np.mean(np.hypot(*np.gradient(generated_full))))
    reference_gradient = float(np.mean(np.hypot(*np.gradient(reference_full))))
    generated_luma_std = float(generated_full.std())
    reference_luma_std = float(reference_full.std())
    gp, rp = radial_psd(generated), radial_psd(reference)
    generated_ratio = band(gp, 32, 128) / max(band(gp, 4, 16), 1.0e-9)
    reference_ratio = band(rp, 32, 128) / max(band(rp, 4, 16), 1.0e-9)
    structure_retention = generated_ratio / max(reference_ratio, 1.0e-9)
    rgb_correlations: dict[str, float] = {}
    for size in (64, 32, 16):
        first = np.asarray(generated.convert("L").resize((size, size), Image.Resampling.BOX))
        second = np.asarray(reference.convert("L").resize((size, size), Image.Resampling.BOX))
        rgb_correlations[str(size)] = circular_max_correlation(first, second)
    generated_orientation = orientation_histogram(generated)
    reference_orientation = orientation_histogram(reference)
    orientation_similarity = 1.0 - 0.5 * float(np.abs(
        generated_orientation - reference_orientation).sum())
    payload: dict[str, object] = {
        "macro_fine_ratio": {
            "generated": generated_ratio,
            "reference": reference_ratio,
            "relative": structure_retention,
        },
        "max_shifted_rgb_correlation": rgb_correlations,
        "orientation_histogram_similarity": orientation_similarity,
        "full_resolution": {
            "generated_luma_std": generated_luma_std,
            "reference_luma_std": reference_luma_std,
            "luma_std_ratio": generated_luma_std / max(reference_luma_std, 1.0e-8),
            "generated_gradient_energy": generated_gradient,
            "reference_gradient_energy": reference_gradient,
            "gradient_energy_ratio": generated_gradient / max(reference_gradient, 1.0e-8),
        },
    }
    maximum_label_correlation = None
    if generated_labels_path and reference_labels_path:
        label_correlations: dict[str, float] = {}
        for size in (64, 32, 16):
            generated_labels = _label_image(generated_labels_path, size)
            reference_labels = _label_image(reference_labels_path, size)
            label_correlations[str(size)] = label_max_correlation(
                generated_labels, reference_labels)
        maximum_label_correlation = max(label_correlations.values())
        payload["max_shifted_label_correlation"] = label_correlations
        payload["components_generated_64"] = component_statistics(
            _label_image(generated_labels_path, 64))
        payload["components_reference_64"] = component_statistics(
            _label_image(reference_labels_path, 64))
    maximum_rgb_correlation = max(rgb_correlations.values())
    payload["summary"] = {
        "structure_retention": structure_retention,
        "max_rgb_correlation": maximum_rgb_correlation,
        "max_label_correlation": maximum_label_correlation,
        "orientation_similarity": orientation_similarity,
        "luma_std_ratio": generated_luma_std / max(reference_luma_std, 1.0e-8),
        "gradient_energy_ratio": generated_gradient / max(reference_gradient, 1.0e-8),
        "structure_pass": bool(0.55 <= structure_retention <= 1.85),
        "novelty_pass": bool(maximum_rgb_correlation < 0.65 and
                             (maximum_label_correlation is None or maximum_label_correlation < 0.65)),
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tile", type=Path, help="generated tile PNG")
    ap.add_argument("--real", type=Path, default=None, help="real reference tile PNG")
    ap.add_argument("--pyramid-dir", type=Path, default=None, help="write downsample pyramid here")
    ap.add_argument("--macro-band", type=int, nargs=2, default=(32, 128), metavar=("LO", "HI"))
    ap.add_argument("--fine-band", type=int, nargs=2, default=(4, 16), metavar=("LO", "HI"))
    ap.add_argument("--min-ratio", type=float, default=0.7,
                    help="fail if (tile macro/fine) / (real macro/fine) is below this")
    ap.add_argument("--tile-labels", type=Path, default=None,
                    help="optional generated categorical preview")
    ap.add_argument("--real-labels", type=Path, default=None,
                    help="optional reference categorical preview")
    ap.add_argument("--json", type=Path, default=None,
                    help="write full quality/novelty metrics as JSON")
    ap.add_argument("--max-correlation", type=float, default=0.65,
                    help="fail when shifted spatial correlation exceeds this")
    args = ap.parse_args()

    tile = load(args.tile)
    if args.pyramid_dir:
        args.pyramid_dir.mkdir(parents=True, exist_ok=True)
        write_pyramid(tile, args.pyramid_dir / args.tile.name)
        print(f"wrote pyramid to {args.pyramid_dir}")

    tp = radial_psd(tile)
    t_macro, t_fine = band(tp, *args.macro_band), band(tp, *args.fine_band)
    t_ratio = t_macro / max(t_fine, 1e-9)
    print(f"{args.tile.name}: macro/fine energy ratio = {t_ratio:.3f}")

    if not args.real:
        print("(no --real reference; ratio is informational only)")
        return
    rp = radial_psd(load(args.real))
    r_macro, r_fine = band(rp, *args.macro_band), band(rp, *args.fine_band)
    r_ratio = r_macro / max(r_fine, 1e-9)
    rel = t_ratio / max(r_ratio, 1e-9)
    print(f"{args.real.name}: macro/fine energy ratio = {r_ratio:.3f}")
    print(f"macro-band energy vs real = {t_macro / max(r_macro,1e-9):.2f}x   "
          f"fine-band energy vs real = {t_fine / max(r_fine,1e-9):.2f}x")
    metrics = compare_images(args.tile, args.real, args.tile_labels, args.real_labels)
    maximum_rgb = metrics["summary"]["max_rgb_correlation"]
    maximum_label = metrics["summary"]["max_label_correlation"]
    structure_ok = rel >= args.min_ratio
    novelty_ok = maximum_rgb < args.max_correlation and (
        maximum_label is None or maximum_label < args.max_correlation)
    verdict = "PASS" if structure_ok and novelty_ok else "FAIL"
    print(f"structure retention = {rel:.2f} of real "
          f"(minimum {args.min_ratio})")
    print(f"maximum shifted RGB correlation = {maximum_rgb:.3f}; "
          f"label correlation = {maximum_label if maximum_label is not None else 'n/a'}")
    print(f"quality + novelty -> {verdict} (correlation ceiling {args.max_correlation})")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
