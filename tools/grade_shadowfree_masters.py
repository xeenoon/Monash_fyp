#!/usr/bin/env python3
"""Colour-grade the AI shadow-removal masters and make every shared edge agree.

Problem this solves: the masters in alps-data/trn-alps-16km/shadowfree-masters/
came out of an external AI shadow-removal tool that processed each 2km tile
independently. That leaves two separate defects, discovered and fixed in this
order (see docs/graded_masters_pipeline.md for the full narrative and the
diagnostic swatches that motivated each step):

  1. Every tile came out too bright/pale relative to shaders/terrain.frag's own
     neutral-rock reference (material_macro_reference_luminance(7u) = 0.192
     linear) -- that's what read as "washed out / chalky" in full sun once
     exposure was fixed to stop masking it.
  2. Independently-graded neighbours disagree with each other: not just in
     brightness, but in how much fine surface detail they carry (a smooth tile
     next to a densely-cracked one), and the underlying AI-synthesized detail
     doesn't even geometrically continue across a shared border.

Pipeline (each stage's parameters were tuned and validated on one
worst-case 2x2 cluster before running on the full 19-tile set -- see the doc):

  A. Shared gamma, not per-tile.  One gamma computed by pooling sub-highlight
     ("probably rock, not snow") pixels across ALL masters and matching their
     median to the neutral-rock target. A per-tile gamma is the #1 way to
     create a NEW brightness seam that wasn't there before -- this can't,
     because it's the same function everywhere. Local contrast / saturation /
     midtone contrast / highlight protection are per-pixel and applied the
     same way (see grade() below).
  B. Whole-tile gain compensation, solved once across every adjacent pair
     simultaneously (a small least-squares "graph Laplacian" problem, the same
     idea orthomosaic/panorama tools use for radiometric normalization). Two
     independent solves: an additive RGB correction, and a multiplicative gain
     on each tile's detail layer (image minus a small blur) for texture
     energy. This is a WHOLE-TILE constant, not a border patch -- it can't
     introduce a new seam of its own, only remove the disagreement that's
     already there. This deliberately replaces a naive "start top-left, sweep
     right" pairwise pass: sequential correction accumulates error along the
     sweep path and depends on visit order; solving every constraint at once
     doesn't, and the 19-tile adjacency graph here is a single connected
     component, so one global solve covers the whole dataset in one shot.
  C. Residual feather right at each border, closing the small gap a whole-tile
     constant can't reach (it matches averages, not the literal boundary
     pixel). Small margin -- this is polish on top of B, not the primary fix.
  D. Multi-band (band-pass) blend across a wide, PROPERLY TAPERING margin per
     edge -- low frequencies blend over a wide band, high frequencies over a
     narrow one, and (this mattered) every band's transition is explicitly
     capped at the blend window's own half-size, so it's mathematically
     guaranteed to reach exactly the unblended tile by the window edge. An
     earlier version let a coarse band's blur radius exceed the window and
     got truncated, which produced a hard line partway through the "blend" --
     not at the seam, at the edge of the blend region itself.

Usage:
    python3 tools/grade_shadowfree_masters.py \\
        --source-dir alps-data/trn-alps-16km/shadowfree-masters \\
        --output-dir alps-data/trn-alps-16km/graded-masters-v2

Then install into a runtime dataset the same way as the original masters:
    python3 tools/build_graded_alps_dataset.py --masters-dir <output-dir> \\
        --output alps-data/trn-alps-16km-graded-v2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

COORD = re.compile(r"^E(\d+)-(\d+)_N(\d+)-(\d+)\.png$")

TARGET_LUMINANCE = 0.192 ** (1.0 / 2.4) * 1.055 - 0.055  # ~0.47 sRGB; shaders/terrain.frag rock ref
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
HIGHLIGHT_LOW = 0.75
HIGHLIGHT_HIGH = 0.92
GAMMA_RANGE = (0.5, 3.0)
LOCAL_CONTRAST_RADIUS = 24
LOCAL_CONTRAST_STRENGTH = 0.35
SATURATION = 1.12
MIDTONE_CONTRAST = 0.08

BORDER_BAND = 12          # px -- measuring the brightness offset at a shared edge
TEXTURE_RADIUS = 6        # px -- local-std texture-energy neighbourhood
TEXTURE_BORDER_BAND = 24  # px -- texture energy is noisier, average a wider strip
RGB_FEATHER_MARGIN = 40
DETAIL_FEATHER_MARGIN = 64
BLEND_TAPER = 200         # px -- half-height of the multi-band blend window
BLEND_RADII = [2, 6, 18, 54, 162]


# ---------------------------------------------------------------- utilities

def srgb_luma(image: np.ndarray) -> np.ndarray:
    return np.tensordot(image, LUMA_WEIGHTS, axes=([2], [0]))


def smoothstep(low: float, high: float, x: np.ndarray) -> np.ndarray:
    if high <= low:
        return (x >= high).astype(np.float32)
    t = np.clip((x - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def gaussian_blur01(image: np.ndarray, radius: int) -> np.ndarray:
    pil_image = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
    return np.asarray(pil_image.filter(ImageFilter.GaussianBlur(radius))).astype(np.float32) / 255.0


def box_blur255(image: np.ndarray, radius: int) -> np.ndarray:
    pil_image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    return np.asarray(pil_image.filter(ImageFilter.BoxBlur(radius))).astype(np.float32)


def box_blur01(field: np.ndarray, radius: int) -> np.ndarray:
    pil_field = Image.fromarray((np.clip(field, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
    return np.asarray(pil_field.filter(ImageFilter.BoxBlur(radius))).astype(np.float32) / 255.0


def texture_energy(luma_255: np.ndarray, radius: int) -> np.ndarray:
    luma_01 = np.clip(luma_255, 0.0, 255.0) / 255.0
    mean = box_blur01(luma_01, radius)
    mean_sq = box_blur01(luma_01 * luma_01, radius)
    return np.sqrt(np.clip(mean_sq - mean * mean, 0.0, None)) * 255.0


def edge_strip(array: np.ndarray, side: str, band: int) -> np.ndarray:
    if side == "right": return array[:, -band:]
    if side == "left": return array[:, :band]
    if side == "top": return array[:band, :]
    if side == "bottom": return array[-band:, :]
    raise ValueError(side)


# ---------------------------------------------------------- stage A: gamma

def parse_coord(path: Path) -> tuple[int, int, int, int]:
    match = COORD.match(path.name)
    if not match:
        raise ValueError(f"unexpected master name: {path.name}")
    return tuple(int(x) for x in match.groups())  # east_min, east_max, north_min, north_max


def find_adjacency(names: list[str], coords: dict[str, tuple[int, int, int, int]]
                   ) -> list[tuple[str, str, str]]:
    """('a','b','horizontal') means a is directly west of b; ('a','b','vertical')
    means a is directly north of b. Each undirected edge appears exactly once."""
    edges = []
    for a in names:
        ae0, ae1, an0, an1 = coords[a]
        for b in names:
            if a == b:
                continue
            be0, be1, bn0, bn1 = coords[b]
            if ae1 == be0 and an0 == bn0 and an1 == bn1:
                edges.append((a, b, "horizontal"))
            if an0 == bn1 and ae0 == be0 and ae1 == be1:
                edges.append((a, b, "vertical"))
    return edges


def pooled_global_gamma(images: dict[str, np.ndarray]) -> float:
    all_below = []
    for array in images.values():
        luma = srgb_luma(array)
        below = luma[luma < HIGHLIGHT_LOW]
        if below.size:
            all_below.append(below)
    reference = float(np.median(np.concatenate(all_below)))
    gamma = float(np.clip(np.log(TARGET_LUMINANCE) / np.log(max(reference, 1e-4)), *GAMMA_RANGE))
    print(f"pooled sub-highlight median (all {len(images)} masters) = {reference:.3f} "
          f"-> shared gamma = {gamma:.3f}")
    return gamma


def grade_base(image01: np.ndarray, gamma: float) -> np.ndarray:
    """Per-pixel grading only -- same function applied identically to every tile.
    Returns 0-255 float."""
    image = np.clip(image01, 0.0, 1.0)
    luma = srgb_luma(image)
    corrected = np.power(image, gamma)
    blurred = gaussian_blur01(corrected, LOCAL_CONTRAST_RADIUS)
    corrected = corrected + (corrected - blurred) * LOCAL_CONTRAST_STRENGTH
    gray = srgb_luma(corrected)[..., None]
    corrected = gray + (corrected - gray) * SATURATION
    corrected = corrected + (corrected - 0.5) * MIDTONE_CONTRAST * (1.0 - np.abs(corrected - 0.5) * 2.0)
    corrected = np.clip(corrected, 0.0, 1.0)
    protect = smoothstep(HIGHLIGHT_LOW, HIGHLIGHT_HIGH, luma)[..., None]
    return np.clip(corrected * (1.0 - protect) + image * protect, 0.0, 1.0) * 255.0


# --------------------------------------------------- stage B: gain compensation

def solve_gain_compensation(keys: list[str], edges: list[tuple[str, str, str]],
                            offsets: dict[tuple[str, str], np.ndarray | float]
                            ) -> dict[str, np.ndarray | float]:
    """Least-squares: correction[a] - correction[b] ~= -offset[(a,b)] for every
    edge, solved across ALL edges at once (not sequentially). Under-determined by
    exactly one free constant per connected component (shifting every tile the
    same way changes no pairwise agreement) -- lstsq's minimum-norm solution
    handles that by centering each component's corrections around zero, which is
    what you want here (no arbitrary "anchor tile")."""
    index = {k: i for i, k in enumerate(keys)}
    n, m = len(keys), len(edges)
    A = np.zeros((m, n))
    sample = next(iter(offsets.values()))
    channels = 1 if np.isscalar(sample) or np.asarray(sample).ndim == 0 else np.asarray(sample).shape[0]
    B = np.zeros((m, channels))
    for row, (a, b, _axis) in enumerate(edges):
        A[row, index[a]] = 1.0
        A[row, index[b]] = -1.0
        B[row, :] = -np.asarray(offsets[(a, b)]).reshape(channels)
    solution, *_ = np.linalg.lstsq(A, B, rcond=None)
    if channels == 1:
        return {k: float(solution[index[k]].item()) for k in keys}
    return {k: solution[index[k]] for k in keys}


def whole_tile_rgb_correction(images: dict[str, np.ndarray],
                              edges: list[tuple[str, str, str]]) -> dict[str, np.ndarray]:
    offsets = {}
    for a, b, axis in edges:
        if axis == "horizontal":
            oa, ob = edge_strip(images[a], "right", BORDER_BAND), edge_strip(images[b], "left", BORDER_BAND)
        else:
            oa, ob = edge_strip(images[a], "bottom", BORDER_BAND), edge_strip(images[b], "top", BORDER_BAND)
        offsets[(a, b)] = oa.reshape(-1, 3).mean(0) - ob.reshape(-1, 3).mean(0)
    correction = solve_gain_compensation(list(images), edges, offsets)
    print("whole-tile RGB correction range: "
          f"{min(float(np.abs(c).max()) for c in correction.values()):.2f} to "
          f"{max(float(np.abs(c).max()) for c in correction.values()):.2f} (0-255 scale)")
    return {k: np.clip(images[k] + correction[k][None, None, :], 0, 255) for k in images}


def whole_tile_detail_gain(images: dict[str, np.ndarray],
                           edges: list[tuple[str, str, str]]) -> dict[str, np.ndarray]:
    luma = {k: srgb_luma(v / 255.0) * 255.0 for k, v in images.items()}
    energy = {k: texture_energy(v, TEXTURE_RADIUS) for k, v in luma.items()}
    offsets = {}
    for a, b, axis in edges:
        if axis == "horizontal":
            ea = edge_strip(energy[a], "right", TEXTURE_BORDER_BAND).mean()
            eb = edge_strip(energy[b], "left", TEXTURE_BORDER_BAND).mean()
        else:
            ea = edge_strip(energy[a], "bottom", TEXTURE_BORDER_BAND).mean()
            eb = edge_strip(energy[b], "top", TEXTURE_BORDER_BAND).mean()
        offsets[(a, b)] = np.log(max(ea, 1e-3)) - np.log(max(eb, 1e-3))
    log_gain = solve_gain_compensation(list(images), edges, offsets)
    gain = {k: float(np.exp(v)) for k, v in log_gain.items()}
    print(f"whole-tile detail gain range: {min(gain.values()):.3f} to {max(gain.values()):.3f}")
    out = {}
    for k in images:
        blurred = box_blur255(images[k], TEXTURE_RADIUS)
        detail = images[k] - blurred
        out[k] = np.clip(blurred + detail * gain[k], 0, 255)
    return out


# --------------------------------------------------- stage C: residual feather

def feather_weight(length: int, margin: int) -> np.ndarray:
    x = np.arange(length, dtype=np.float32)
    return np.clip(1.0 - x / margin, 0.0, 1.0)


def feather_rgb_pair(a: np.ndarray, b: np.ndarray, axis: str, margin: int) -> None:
    band = margin
    if axis == "horizontal":
        offset = a[:, -band:, :].mean(axis=(0, 1)) - b[:, :band, :].mean(axis=(0, 1))
        wa, wb = feather_weight(band, margin)[::-1], feather_weight(band, margin)
        a[:, -band:, :] -= offset * 0.5 * wa[None, :, None]
        b[:, :band, :] += offset * 0.5 * wb[None, :, None]
    else:
        offset = a[-band:, :, :].mean(axis=(0, 1)) - b[:band, :, :].mean(axis=(0, 1))
        wa, wb = feather_weight(band, margin)[::-1], feather_weight(band, margin)
        a[-band:, :, :] -= offset * 0.5 * wa[:, None, None]
        b[:band, :, :] += offset * 0.5 * wb[:, None, None]


def feather_detail_pair(a: np.ndarray, b: np.ndarray, axis: str, margin: int, radius: int) -> None:
    blur_a, blur_b = box_blur255(a, radius), box_blur255(b, radius)
    detail_a, detail_b = a - blur_a, b - blur_b
    energy_a = texture_energy(srgb_luma(a / 255.0) * 255.0, radius)
    energy_b = texture_energy(srgb_luma(b / 255.0) * 255.0, radius)

    band = margin
    if axis == "horizontal":
        ea, eb = edge_strip(energy_a, "right", band).mean(), edge_strip(energy_b, "left", band).mean()
    else:
        ea, eb = edge_strip(energy_a, "bottom", band).mean(), edge_strip(energy_b, "top", band).mean()
    log_offset = np.log(max(ea, 1e-3)) - np.log(max(eb, 1e-3))

    w = feather_weight(band, margin)
    gain_a_1d = np.exp(-0.5 * log_offset * w[::-1])
    gain_b_1d = np.exp(0.5 * log_offset * w)
    if axis == "horizontal":
        detail_a[:, -band:, :] *= gain_a_1d[None, :, None]
        detail_b[:, :band, :] *= gain_b_1d[None, :, None]
    else:
        detail_a[-band:, :, :] *= gain_a_1d[:, None, None]
        detail_b[:band, :, :] *= gain_b_1d[:, None, None]

    a[...] = np.clip(blur_a + detail_a, 0, 255)
    b[...] = np.clip(blur_b + detail_b, 0, 255)


# --------------------------------------------------- stage D: multi-band blend

def mirror_extend_down(image: np.ndarray, amount: int) -> np.ndarray:
    strip = image[-amount:, :, :][::-1, :, :]
    return np.concatenate([image[-amount:, :, :], strip], axis=0)


def mirror_extend_up(image: np.ndarray, amount: int) -> np.ndarray:
    strip = image[:amount, :, :][::-1, :, :]
    return np.concatenate([strip, image[:amount, :, :]], axis=0)


def bandpass_levels(image: np.ndarray, radii: list[int]) -> list[np.ndarray]:
    out, prev_blur = [], image
    for r in radii:
        blur = box_blur255(prev_blur, r)
        out.append(prev_blur - blur)
        prev_blur = blur
    out.append(prev_blur)  # coarsest residual (DC term)
    return out


def blend_vertical(north: np.ndarray, south: np.ndarray, taper: int,
                   radii: list[int]) -> tuple[np.ndarray, np.ndarray]:
    north_window = mirror_extend_down(north, taper)
    south_window = mirror_extend_up(south, taper)
    window_h = 2 * taper
    y = np.arange(window_h, dtype=np.float32)

    north_bands = bandpass_levels(north_window, radii)
    south_bands = bandpass_levels(south_window, radii)

    result = np.zeros((window_h, north.shape[1], 3), dtype=np.float32)
    for i, r in enumerate(radii):
        # Every band's transition is capped at `taper`, so it is GUARANTEED to
        # reach 0/1 exactly at this window's own edges, whatever its blur
        # radius is -- this is the fix for the truncated-band hard-line bug.
        width = min(r * 1.5, taper)
        mask = smoothstep(taper - width, taper + width, y)[:, None, None]
        result += north_bands[i] * (1.0 - mask) + south_bands[i] * mask
    mask_residual = smoothstep(0, window_h, y)[:, None, None]
    result += north_bands[-1] * (1.0 - mask_residual) + south_bands[-1] * mask_residual
    result = np.clip(result, 0, 255)

    north_out, south_out = north.copy(), south.copy()
    north_out[-taper:, :, :] = result[:taper, :, :]
    south_out[:taper, :, :] = result[taper:, :, :]
    return north_out, south_out


def blend_edge(a: np.ndarray, b: np.ndarray, axis: str, taper: int,
               radii: list[int]) -> tuple[np.ndarray, np.ndarray]:
    if axis == "vertical":
        return blend_vertical(a, b, taper, radii)
    a_t, b_t = a.transpose(1, 0, 2), b.transpose(1, 0, 2)
    a_out, b_out = blend_vertical(a_t, b_t, taper, radii)
    return a_out.transpose(1, 0, 2), b_out.transpose(1, 0, 2)


# --------------------------------------------------------------- verification

def seam_score_rgb(a: np.ndarray, b: np.ndarray, axis: str, band: int = BORDER_BAND) -> float:
    if axis == "horizontal":
        sa, sb = edge_strip(a, "right", band), edge_strip(b, "left", band)
    else:
        sa, sb = edge_strip(a, "bottom", band), edge_strip(b, "top", band)
    return float(np.abs(sa.reshape(-1, 3).mean(0) - sb.reshape(-1, 3).mean(0)).mean())


def seam_score_texture(a: np.ndarray, b: np.ndarray, axis: str,
                       band: int = TEXTURE_BORDER_BAND) -> float:
    ea = texture_energy(srgb_luma(a / 255.0) * 255.0, TEXTURE_RADIUS)
    eb = texture_energy(srgb_luma(b / 255.0) * 255.0, TEXTURE_RADIUS)
    if axis == "horizontal":
        sa, sb = edge_strip(ea, "right", band), edge_strip(eb, "left", band)
    else:
        sa, sb = edge_strip(ea, "bottom", band), edge_strip(eb, "top", band)
    return abs(float(sa.mean()) - float(sb.mean()))


# --------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path,
                        default=Path("alps-data/trn-alps-16km/shadowfree-masters"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.source_dir.glob("E*_N*.png"))
    if not paths:
        raise SystemExit(f"no coordinate-named masters in {args.source_dir}")
    names = [p.name for p in paths]
    coords = {p.name: parse_coord(p) for p in paths}
    edges = find_adjacency(names, coords)
    print(f"{len(names)} masters, {len(edges)} shared edges")

    raw = {p.name: np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0 for p in paths}

    # --- A: shared-gamma grading ---
    gamma = pooled_global_gamma(raw)
    graded = {name: grade_base(array, gamma) for name, array in raw.items()}

    # --- B: whole-tile gain compensation (global solve, all edges at once) ---
    graded = whole_tile_rgb_correction(graded, edges)
    graded = whole_tile_detail_gain(graded, edges)

    # --- C: residual feather per edge ---
    for a, b, axis in edges:
        feather_rgb_pair(graded[a], graded[b], axis, RGB_FEATHER_MARGIN)
    for a, b, axis in edges:
        feather_detail_pair(graded[a], graded[b], axis, DETAIL_FEATHER_MARGIN, TEXTURE_RADIUS)

    # --- D: multi-band blend per edge ---
    for a, b, axis in edges:
        graded[a], graded[b] = blend_edge(graded[a], graded[b], axis, BLEND_TAPER, BLEND_RADII)

    # --- verification: every edge, before vs after ---
    print("\nseam scores after the full pipeline (brightness / texture-energy):")
    worst_rgb, worst_tex = 0.0, 0.0
    for a, b, axis in edges:
        rgb_score = seam_score_rgb(graded[a], graded[b], axis)
        tex_score = seam_score_texture(graded[a], graded[b], axis)
        worst_rgb, worst_tex = max(worst_rgb, rgb_score), max(worst_tex, tex_score)
        print(f"  {a} <-> {b} ({axis}): brightness {rgb_score:5.2f}  texture-energy {tex_score:5.2f}")
    print(f"\nworst brightness seam: {worst_rgb:.2f}   worst texture-energy seam: {worst_tex:.2f}")
    print("(reference: raw pre-AI satellite scored ~4-6 on brightness with this same metric)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in graded.items():
        Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(args.output_dir / name)
    print(f"\nsaved {len(graded)} graded masters to {args.output_dir}")


if __name__ == "__main__":
    main()
