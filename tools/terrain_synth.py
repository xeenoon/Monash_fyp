#!/usr/bin/env python3
"""Full-map, transition-aware terrain texture synthesis demo.

The source imagery is treated as one continuous donor manifold.  Dense patches
carry perceptual colour, continuous material mixture, structure, frequency and
terrain-geometry descriptors.  An explicit graph connects real geographic
neighbours first and visually compatible distant neighbours second.  During
synthesis candidate retrieval walks that graph toward a slowly varying target
style; the minimum-cost seam cutter remains the final pixel-level operation.

This is deliberately a reference/demo implementation: descriptors and graph
diagnostics are kept inspectable instead of hidden in a learned embedding.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
import json
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, shift as image_shift
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from terrain_tiles import read_tile  # noqa: E402
from procedural_gap_demo import rgb_patch_error, rgb_quilt_take_mask  # noqa: E402

SIZE = 256
# Patch size varies by row and advance is size-overlap: pass13's non-periodic
# placement is retained.  The two banks represent geological and local scales.
LEVELS = (((144, 160, 176), 64), ((84, 96, 108), 40))
LOWFREQ_SIGMA = 40.0
MACRO_SMOOTH_SIGMA = 110.0

W_TERRAIN = 0.75
W_STYLE = 2.4
W_GRAPH = 1.2
W_LOCALITY = 0.95
W_OVERLAP = 3.0
W_GRADIENT = 1.5
W_CONTEXT = 0.6
W_CONFIDENCE = 2.0
W_REUSE_FAR = 1.6
REUSE_FAR_R = 150.0
SRC_CELL = 32
COH_LENGTH_PX = 150.0
MAX_POOL = 128
MIN_PATCH_CONFIDENCE = 0.97  # at most about 6% ordinary repaired support
DEFAULT_BAD_TILES = {(14, 5), (15, 5)}

# style vector layout
LAB = slice(0, 3)
LAB_STD = slice(3, 6)
MATERIAL = slice(6, 9)       # grass, rock, snow; probabilities sum to one
ORIENTATION = slice(9, 11)   # axial cos(2 theta), sin(2 theta), anisotropy-weighted
ANISOTROPY = 11
GRAD_HIST = slice(12, 16)
EDGE_DENSITY = 16
LOW_ROUGHNESS = 17
MID_ROUGHNESS = 18
STYLE_DIM = 19

STYLE_SCALE = np.array([
    0.22, 0.35, 0.35, 0.22, 0.28, 0.28,
    0.55, 0.55, 0.45, 0.55, 0.55, 0.55,
    0.45, 0.45, 0.45, 0.45, 0.35, 0.30, 0.30,
], np.float32)


def _channels(rgb: np.ndarray) -> tuple[np.ndarray, ...]:
    u = rgb.astype(np.float32) / 255.0
    r, g, b = u[..., 0], u[..., 1], u[..., 2]
    lum = u @ np.array((0.2126, 0.7152, 0.0722), np.float32)
    mx, mn = u.max(-1), u.min(-1)
    return r, g, b, lum, (mx - mn) / np.maximum(mx, 1.0e-5)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Vectorized D65 CIE Lab, scaled to roughly unit descriptor ranges."""
    u = rgb.astype(np.float32) / 255.0
    linear = np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)
    xyz = linear @ np.array(((0.4124564, 0.3575761, 0.1804375),
                             (0.2126729, 0.7151522, 0.0721750),
                             (0.0193339, 0.1191920, 0.9503041)), np.float32).T
    xyz /= np.array((0.95047, 1.0, 1.08883), np.float32)
    delta = 6.0 / 29.0
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4.0 / 29.0)
    return np.stack((116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])), -1)


def material_probability_map(rgb: np.ndarray) -> np.ndarray:
    """Continuous grass/rock/snow guide; never used as hard retrieval bins."""
    r, g, b, lum, sat = _channels(rgb)
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    snow = sigmoid((lum - 0.64) * 15.0) * sigmoid((0.18 - sat) * 14.0)
    green = g - 0.5 * (r + b)
    grass = sigmoid((green - 0.005) * 22.0) * sigmoid((0.74 - lum) * 9.0) * (1 - 0.72 * snow)
    rock = np.maximum(0.04, 1.0 - grass - snow)
    probs = np.stack((grass, rock, snow), -1).astype(np.float32)
    return probs / np.maximum(probs.sum(-1, keepdims=True), 1.0e-6)


def detect_nonterrain(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (repair mask, hard anomaly mask) with vegetation-aware detection.

    Dark rock is intentionally protected.  Roads/trees are only detected in a
    green neighbourhood; water and magenta contamination are independent.
    """
    r, g, b, lum, sat = _channels(rgb)
    green_ctx = gaussian_filter(g - r, 10.0) > 0.015
    tree = (g >= b - 0.02) & (g > r - 0.07) & (lum < 0.31) & (g > 0.05) & green_ctx
    road = (lum > 0.60) & (sat < 0.13) & green_ctx
    water = (b > r + 0.035) & (b > g + 0.015) & (lum < 0.52)
    magenta = (r > g + 0.11) & (b > g + 0.07) & (sat > 0.22)
    hard = binary_dilation(road | water | magenta, iterations=5)
    repair = binary_dilation(tree | hard, iterations=4)
    return repair, hard


def clean_source(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove detected content and retain an honest per-pixel confidence map."""
    repair, hard = detect_nonterrain(rgb)
    if not repair.any():
        return rgb.copy(), np.full(rgb.shape[:2], 255, np.uint8), hard
    nearest = distance_transform_edt(repair, return_distances=False, return_indices=True)
    filled = rgb[tuple(nearest)].astype(np.float32)
    softened = gaussian_filter(filled, (2.0, 2.0, 0))
    cleaned = rgb.copy()
    cleaned[repair] = np.clip(softened[repair], 0, 255).astype(np.uint8)
    confidence = np.full(repair.shape, 255, np.uint8)
    confidence[repair] = 128
    confidence[hard] = 32
    return cleaned, confidence, hard


def discover_level5(dataset: Path) -> list[tuple[int, int]]:
    imagery = dataset / "imagery" / "5"
    terrain = dataset / "tiles" / "5"
    found: list[tuple[int, int]] = []
    for png in imagery.glob("*/*.png"):
        try:
            x, y = int(png.parent.name), int(png.stem)
        except ValueError:
            continue
        if (terrain / str(x) / f"{y}.trn").exists():
            found.append((x, y))
    return sorted(found, key=lambda p: (p[1], p[0]))


def tile_rgb(dataset: Path, x: int, y: int) -> np.ndarray | None:
    path = dataset / "imagery" / "5" / str(x) / f"{y}.png"
    return np.asarray(Image.open(path).convert("RGB"))[1:-1, 1:-1].copy() if path.exists() else None


def tile_dem(dataset: Path, x: int, y: int) -> tuple[np.ndarray, np.ndarray] | None:
    path = dataset / "tiles" / "5" / str(x) / f"{y}.trn"
    if not path.exists():
        return None
    tile = read_tile(path)
    height = np.asarray(tile.decoded_heights(), np.float32).reshape(tile.height, tile.width)[1:-1, 1:-1]
    height = np.asarray(Image.fromarray(height, "F").resize((SIZE, SIZE), Image.Resampling.BILINEAR))
    metres_per_pixel = (tile.extent[2] - tile.extent[0]) / SIZE
    dy, dx = np.gradient(height, metres_per_pixel)
    return height, np.hypot(dx, dy)


class Atlas:
    """Clean, complete source map plus coordinate identity for every donor tile."""

    def __init__(self, dataset: Path, coordinates: Iterable[tuple[int, int]],
                 exclude: set[tuple[int, int]] = frozenset()):
        self.rgb: list[np.ndarray] = []
        self.height: list[np.ndarray] = []
        self.slope: list[np.ndarray] = []
        self.confidence: list[np.ndarray] = []
        self.hard: list[np.ndarray] = []
        self.coordinates: list[tuple[int, int]] = []
        for x, y in coordinates:
            if (x, y) in exclude:
                continue
            rgb, dem = tile_rgb(dataset, x, y), tile_dem(dataset, x, y)
            if rgb is None or dem is None:
                continue
            clean, confidence, hard = clean_source(rgb)
            self.rgb.append(clean)
            # Float16 keeps the full-map prototype below a gigabyte; descriptor
            # precision is still well above the source DEM sampling precision.
            self.height.append(dem[0].astype(np.float16))
            self.slope.append(dem[1].astype(np.float16))
            self.confidence.append(confidence)
            self.hard.append(hard)
            self.coordinates.append((x, y))
        if not self.rgb:
            raise RuntimeError("donor atlas is empty")
        self.min_x = min(p[0] for p in self.coordinates)
        self.min_y = min(p[1] for p in self.coordinates)
        print(f"atlas: {len(self.rgb)} cleaned tiles", flush=True)

    def global_xy(self, tile: np.ndarray, local_x: np.ndarray,
                  local_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coords = np.asarray(self.coordinates, np.int32)[tile]
        return ((coords[:, 0] - self.min_x) * SIZE + local_x,
                (coords[:, 1] - self.min_y) * SIZE + local_y)


def _integral(values: np.ndarray) -> np.ndarray:
    if values.ndim == 2:
        values = values[..., None]
    result = np.zeros((values.shape[0] + 1, values.shape[1] + 1, values.shape[2]), np.float64)
    result[1:, 1:] = values.cumsum(0, dtype=np.float64).cumsum(1, dtype=np.float64)
    return result


def _rect_mean(ii: np.ndarray, ys: np.ndarray, xs: np.ndarray,
               height: int, width: int) -> np.ndarray:
    total = (ii[ys + height, xs + width] - ii[ys, xs + width] -
             ii[ys + height, xs] + ii[ys, xs])
    return (total / float(height * width)).astype(np.float32)


def _origins(patch: int, stride: int) -> np.ndarray:
    result = list(range(0, SIZE - patch + 1, stride))
    if result[-1] != SIZE - patch:
        result.append(SIZE - patch)
    return np.asarray(result, np.int32)


def _style_features(rgb: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    lab = srgb_to_lab(rgb)
    materials = material_probability_map(rgb)
    gray = _channels(rgb)[3]
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    jxx, jyy, jxy = gx * gx, gy * gy, gx * gy
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    bins = np.floor(angle * (4.0 / np.pi)).astype(np.int8).clip(0, 3)
    hist = [(magnitude * (bins == k)).astype(np.float32) for k in range(4)]
    low = gaussian_filter(gray, 12.0)
    mid = np.abs(gaussian_filter(gray, 2.0) - gaussian_filter(gray, 10.0))
    channels = [lab, lab * lab, materials, jxx, jyy, jxy, magnitude,
                np.stack(hist, -1), low, low * low, mid]
    return gray, channels


def describe_rgb(rgb: np.ndarray) -> np.ndarray:
    """Descriptor for an arbitrary known-image region (used at gap borders)."""
    _, fields = _style_features(rgb)
    lab, _, materials, jxx, jyy, jxy, magnitude, hist, low, _, mid = fields
    mlab, lab_std = lab.mean((0, 1)), lab.std((0, 1))
    material = materials.mean((0, 1))
    axx, ayy, axy = float(jxx.mean()), float(jyy.mean()), float(jxy.mean())
    delta = np.sqrt((axx - ayy) ** 2 + 4 * axy ** 2)
    anis = delta / max(axx + ayy, 1.0e-7)
    orientation = np.array(((axx - ayy) / max(delta, 1.0e-7),
                            2 * axy / max(delta, 1.0e-7)), np.float32) * anis
    ghist = np.array([channel.mean() for channel in np.moveaxis(hist, -1, 0)], np.float32)
    ghist /= max(float(ghist.sum()), 1.0e-6)
    return np.concatenate((mlab / np.array((100, 45, 45), np.float32),
                           lab_std / np.array((35, 30, 30), np.float32), material,
                           orientation, np.array((anis,), np.float32), ghist,
                           np.array((magnitude.mean() / 0.18, low.std() / 0.18,
                                     mid.mean() / 0.12), np.float32))).astype(np.float32)


@dataclass
class BoundaryStyleField:
    points: np.ndarray
    styles: np.ndarray

    def sample(self, x: float, y: float) -> tuple[np.ndarray, float]:
        distances = np.hypot(self.points[:, 0] - x, self.points[:, 1] - y)
        count = min(6, len(distances)); nearest = np.argpartition(distances, count - 1)[:count]
        weights = 1.0 / np.maximum(distances[nearest], 32.0) ** 2
        return (np.average(self.styles[nearest], axis=0, weights=weights).astype(np.float32),
                float(distances[nearest].min()))


def build_boundary_style_field(dataset: Path, ox: int, oy: int,
                               size: int) -> BoundaryStyleField | None:
    """Sample known imagery around a held-out block to anchor its latent style."""
    observations: list[tuple[tuple[float, float], np.ndarray]] = []
    for i in range(size):
        for tx, ty, point, side in (
            (ox + i, oy - 1, ((i + 0.5) * SIZE, -0.2 * SIZE), "north"),
            (ox + i, oy + size, ((i + 0.5) * SIZE, (size + 0.2) * SIZE), "south"),
        ):
            rgb = tile_rgb(dataset, tx, ty)
            if rgb is not None:
                clean = clean_source(rgb)[0]
                band = clean[SIZE // 2:] if side == "north" else clean[:SIZE // 2]
                observations.append((point, describe_rgb(band)))
    for j in range(size):
        for tx, ty, point, side in (
            (ox - 1, oy + j, (-0.2 * SIZE, (j + 0.5) * SIZE), "west"),
            (ox + size, oy + j, ((size + 0.2) * SIZE, (j + 0.5) * SIZE), "east"),
        ):
            rgb = tile_rgb(dataset, tx, ty)
            if rgb is not None:
                clean = clean_source(rgb)[0]
                band = clean[:, SIZE // 2:] if side == "west" else clean[:, :SIZE // 2]
                observations.append((point, describe_rgb(band)))
    if not observations:
        return None
    return BoundaryStyleField(np.asarray([item[0] for item in observations], np.float32),
                              np.asarray([item[1] for item in observations], np.float32))


@dataclass
class PatchBank:
    patch: int
    tile: np.ndarray
    y: np.ndarray
    x: np.ndarray
    global_xy: np.ndarray
    geometry: np.ndarray
    style: np.ndarray
    confidence: np.ndarray
    transition_strength: np.ndarray
    transition_orientation: np.ndarray
    geographic: np.ndarray
    visual: np.ndarray
    geometry_tree: cKDTree
    style_tree: cKDTree
    source_tree: cKDTree
    transition_tree: cKDTree | None
    transition_nodes: np.ndarray

    def neighbours(self, node: int) -> np.ndarray:
        return np.unique(np.concatenate((self.geographic[node], self.visual[node])))


def _normalized_style(style: np.ndarray) -> np.ndarray:
    return style / STYLE_SCALE[None]


def build_patch_bank(atlas: Atlas, patch: int) -> PatchBank:
    """Extract a dense descriptor bank and its geographic/visual graph."""
    stride = max(patch // 3, 24)
    axis = _origins(patch, stride)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    ys, xs = yy.ravel(), xx.ravel()
    count_per_tile = len(ys)
    records: dict[str, list[np.ndarray]] = {key: [] for key in
        ("tile", "y", "x", "geometry", "style", "confidence", "strength", "orientation")}

    for tile_index, (rgb, height16, slope16, confidence) in enumerate(zip(
            atlas.rgb, atlas.height, atlas.slope, atlas.confidence)):
        height, slope = height16.astype(np.float32), slope16.astype(np.float32)
        _, fields = _style_features(rgb)
        lab, lab2, materials, jxx, jyy, jxy, magnitude, hist, low, low2, mid = fields
        dy, dx = np.gradient(height)
        aspect_norm = np.maximum(np.hypot(dx, dy), 1.0e-5)
        curvature = np.abs(np.gradient(dx, axis=1) + np.gradient(dy, axis=0))
        arrays = [lab, lab2, materials, jxx, jyy, jxy, magnitude, hist, low, low2, mid,
                  height, height * height, slope, slope * slope,
                  dx / aspect_norm * slope, dy / aspect_norm * slope, curvature,
                  confidence.astype(np.float32) / 255.0]
        means = [_rect_mean(_integral(value), ys, xs, patch, patch) for value in arrays]
        (mlab, mlab2, material, mjxx, mjyy, mjxy, edge, ghist, mlow, mlow2, mmid,
         mh, mh2, ms, ms2, maspx, maspy, mcurv, mconfidence) = means
        lab_std = np.sqrt(np.maximum(mlab2 - mlab * mlab, 0.0))
        delta = np.sqrt((mjxx - mjyy) ** 2 + 4 * mjxy ** 2)[:, 0]
        energy = (mjxx + mjyy)[:, 0]
        anis = delta / np.maximum(energy, 1.0e-7)
        ori = np.stack(((mjxx[:, 0] - mjyy[:, 0]) / np.maximum(delta, 1.0e-7),
                        2 * mjxy[:, 0] / np.maximum(delta, 1.0e-7)), -1) * anis[:, None]
        ghist = ghist / np.maximum(ghist.sum(1, keepdims=True), 1.0e-6)
        low_std = np.sqrt(np.maximum(mlow2[:, 0] - mlow[:, 0] ** 2, 0.0))
        style = np.column_stack((mlab / np.array((100, 45, 45), np.float32),
                                 lab_std / np.array((35, 30, 30), np.float32),
                                 material, ori, anis, ghist,
                                 edge[:, 0] / 0.18, low_std / 0.18, mmid[:, 0] / 0.12))
        hstd = np.sqrt(np.maximum(mh2[:, 0] - mh[:, 0] ** 2, 0.0))
        sstd = np.sqrt(np.maximum(ms2[:, 0] - ms[:, 0] ** 2, 0.0))
        geometry = np.column_stack((mh[:, 0] / 1200.0, hstd / 350.0,
                                    ms[:, 0] / 0.7, sstd / 0.5,
                                    maspx[:, 0] / 0.7, maspy[:, 0] / 0.7,
                                    mcurv[:, 0] / 35.0)).astype(np.float32)

        half = patch // 2
        mat_ii = _integral(materials)
        left = _rect_mean(mat_ii, ys, xs, patch, half)
        right = _rect_mean(mat_ii, ys, xs + patch - half, patch, half)
        top = _rect_mean(mat_ii, ys, xs, half, patch)
        bottom = _rect_mean(mat_ii, ys + patch - half, xs, half, patch)
        horizontal = np.linalg.norm(left - right, axis=1)
        vertical = np.linalg.norm(top - bottom, axis=1)
        strength = np.maximum(horizontal, vertical)
        orientation = np.where(horizontal >= vertical, 0.0, np.pi / 2).astype(np.float32)

        records["tile"].append(np.full(count_per_tile, tile_index, np.int32))
        records["y"].append(ys.copy()); records["x"].append(xs.copy())
        records["geometry"].append(geometry); records["style"].append(style.astype(np.float32))
        records["confidence"].append(mconfidence[:, 0])
        records["strength"].append(strength.astype(np.float32))
        records["orientation"].append(orientation)

    tile = np.concatenate(records["tile"]); y = np.concatenate(records["y"]); x = np.concatenate(records["x"])
    geometry = np.concatenate(records["geometry"]); style = np.concatenate(records["style"])
    confidence = np.concatenate(records["confidence"])
    strength = np.concatenate(records["strength"]); orientation = np.concatenate(records["orientation"])
    # Inpainting preserves atlas coverage but never becomes a cheap source of
    # large texture regions.  This support-level gate is the practical analogue
    # of dilating an object exclusion mask by the patch radius: any patch whose
    # support contains substantial uncertain reconstruction is removed before
    # the graph is built.
    eligible = confidence >= MIN_PATCH_CONFIDENCE
    if not eligible.any():
        raise RuntimeError(f"no clean donor patches remain at size {patch}")
    tile, y, x = tile[eligible], y[eligible], x[eligible]
    geometry, style = geometry[eligible], style[eligible]
    confidence = confidence[eligible]
    strength, orientation = strength[eligible], orientation[eligible]
    gx, gy = atlas.global_xy(tile, x + patch // 2, y + patch // 2)
    source_xy = np.column_stack((gx, gy)).astype(np.float32)
    normalized = _normalized_style(style)
    geometry_tree, style_tree, source_tree = cKDTree(geometry), cKDTree(normalized), cKDTree(source_xy)
    k = min(9, len(tile))
    geo_d, geographic = source_tree.query(source_xy, k=k)
    visual_d, visual = style_tree.query(normalized, k=k)
    geographic, visual = np.atleast_2d(geographic)[:, 1:], np.atleast_2d(visual)[:, 1:]
    # Geographic edges are real only while support remains nearby.  Invalid far
    # slots fall back to self and are harmless during pool deduplication.
    too_far = np.atleast_2d(geo_d)[:, 1:] > patch * 1.75
    geographic = np.where(too_far, np.arange(len(tile), dtype=np.int32)[:, None], geographic)
    transition_nodes = np.flatnonzero((strength > 0.18) & (confidence > 0.82)).astype(np.int32)
    transition_tree = cKDTree(normalized[transition_nodes]) if len(transition_nodes) else None
    return PatchBank(patch, tile, y, x, source_xy, geometry, style, confidence,
                     strength, orientation, geographic.astype(np.int32), visual.astype(np.int32),
                     geometry_tree, style_tree, source_tree, transition_tree, transition_nodes)


def target_geometry(height: np.ndarray, slope: np.ndarray, y: int, x: int,
                    h: int, w: int) -> np.ndarray:
    patch_h, patch_s = height[y:y + h, x:x + w], slope[y:y + h, x:x + w]
    dy, dx = np.gradient(patch_h)
    norm = np.maximum(np.hypot(dx, dy), 1.0e-5)
    curvature = np.abs(np.gradient(dx, axis=1) + np.gradient(dy, axis=0))
    return np.array((patch_h.mean() / 1200.0, patch_h.std() / 350.0,
                     patch_s.mean() / 0.7, patch_s.std() / 0.5,
                     np.mean(dx / norm * patch_s) / 0.7,
                     np.mean(dy / norm * patch_s) / 0.7,
                     curvature.mean() / 35.0), np.float32)


def predict_target_style(bank: PatchBank, geometry: np.ndarray, k: int = 64) -> np.ndarray:
    """KNN regression from terrain geometry to a continuous donor style state."""
    k = min(k, len(bank.tile))
    distance, nodes = bank.geometry_tree.query(geometry, k=k)
    distance, nodes = np.atleast_1d(distance), np.atleast_1d(nodes)
    weights = bank.confidence[nodes] / np.maximum(distance, 0.035) ** 2
    # Trim the far half: keeps local geometry modes instead of averaging the
    # entire mountain into one grey-green texture.
    keep = distance <= np.quantile(distance, 0.55)
    weights *= keep
    result = np.average(bank.style[nodes], axis=0, weights=np.maximum(weights, 1.0e-6)).astype(np.float32)
    # Means are appropriate for colour/material coordinates, but averaging
    # variances and band energy makes every predicted texture unnaturally bland.
    # Project those statistics back onto the upper-middle of *real* nearby
    # donors.  Candidate retrieval remains continuous; it simply stops asking
    # for an off-manifold "average roughness" that no source patch possesses.
    support = bank.style[nodes[keep]] if keep.any() else bank.style[nodes]
    energetic = np.array((3, 4, 5, EDGE_DENSITY, LOW_ROUGHNESS, MID_ROUGHNESS))
    result[energetic] = np.quantile(support[:, energetic], 0.65, axis=0)
    return result


def bounded_style_target(raw: np.ndarray, current: np.ndarray | None,
                         amount: float = 0.36) -> np.ndarray:
    """Slow latent z(x): make one bounded move from current toward raw target."""
    if current is None:
        return raw.copy()
    delta = raw - current
    norm = float(np.linalg.norm(delta / STYLE_SCALE))
    if norm > amount:
        delta *= amount / norm
    result = current + delta
    result[MATERIAL] = np.clip(result[MATERIAL], 0, None)
    result[MATERIAL] /= max(float(result[MATERIAL].sum()), 1.0e-6)
    return result.astype(np.float32)


def graph_path(bank: PatchBank, start: int, goal: int,
               max_expansions: int = 900) -> list[int]:
    """Bounded A* path through the explicit donor compatibility graph."""
    if start == goal:
        return [start]
    target = _normalized_style(bank.style[goal:goal + 1])[0]
    norm_style = _normalized_style(bank.style)
    queue: list[tuple[float, float, int]] = [(float(np.linalg.norm(norm_style[start] - target)), 0.0, start)]
    previous: dict[int, int] = {}
    best_cost = {start: 0.0}
    best_node, best_h = start, queue[0][0]
    for _ in range(max_expansions):
        if not queue:
            break
        _, cost, node = heapq.heappop(queue)
        if cost != best_cost.get(node):
            continue
        heuristic = float(np.linalg.norm(norm_style[node] - target))
        if heuristic < best_h:
            best_node, best_h = node, heuristic
        if node == goal or heuristic < 0.12:
            best_node = node
            break
        geo_set = set(int(v) for v in bank.geographic[node])
        for neighbour in bank.neighbours(node):
            neighbour = int(neighbour)
            if neighbour == node:
                continue
            style_edge = float(np.linalg.norm(norm_style[node] - norm_style[neighbour]))
            # Actual source adjacency is the gold-standard cheap edge.
            edge = (0.28 if neighbour in geo_set else 1.0) * style_edge + (0.0 if neighbour in geo_set else 0.08)
            trial = cost + edge + 0.25 * (1.0 - bank.confidence[neighbour])
            if trial >= best_cost.get(neighbour, np.inf):
                continue
            best_cost[neighbour] = trial; previous[neighbour] = node
            h = float(np.linalg.norm(norm_style[neighbour] - target))
            heapq.heappush(queue, (trial + 0.72 * h, trial, neighbour))
    path = [best_node]
    while path[-1] != start and path[-1] in previous:
        path.append(previous[path[-1]])
    path.reverse()
    return path


def candidate_pool(bank: PatchBank, target_style: np.ndarray, anchor: int | None,
                   rng: np.random.Generator) -> tuple[np.ndarray, int, dict[str, int]]:
    """60% real-source, 30% graph, 10% global/transition candidate generation."""
    normalized_target = target_style / STYLE_SCALE
    _, global_nodes = bank.style_tree.query(normalized_target, k=min(12, len(bank.tile)))
    global_nodes = np.atleast_1d(global_nodes).astype(np.int32)
    pools: list[np.ndarray] = [global_nodes]
    provenance = {"geographic": 0, "visual": 0, "global": len(global_nodes), "transition": 0}
    waypoint = int(global_nodes[0])
    if anchor is not None:
        goal = waypoint
        path = graph_path(bank, anchor, goal)
        waypoint = path[1] if len(path) > 1 else path[0]
        first_geo = np.unique(np.concatenate((bank.geographic[anchor], bank.geographic[waypoint])))
        geographic = np.unique(np.concatenate((first_geo, bank.geographic[first_geo].ravel())))
        visual = np.unique(np.concatenate((bank.visual[anchor], bank.visual[waypoint],
                                           bank.visual[bank.visual[waypoint]].ravel())))
        pools.extend((geographic[:64], visual[:32], np.asarray(path[:5], np.int32)))
        provenance["geographic"] = min(64, len(geographic)); provenance["visual"] = min(32, len(visual))
        material_change = float(np.linalg.norm(bank.style[anchor, MATERIAL] - target_style[MATERIAL]))
        if material_change > 0.10 and bank.transition_tree is not None:
            _, local = bank.transition_tree.query(normalized_target, k=min(12, len(bank.transition_nodes)))
            transitions = bank.transition_nodes[np.atleast_1d(local)]
            pools.append(transitions); provenance["transition"] = len(transitions)
    combined = np.unique(np.concatenate(pools)).astype(np.int32)
    if len(combined) > MAX_POOL:
        distance = np.linalg.norm(_normalized_style(bank.style[combined]) - normalized_target, axis=1)
        keep = np.argpartition(distance + rng.random(len(distance)) * 0.025, MAX_POOL - 1)[:MAX_POOL]
        combined = combined[keep]
    return combined, waypoint, provenance


def frequency_decoupled_patch(base: np.ndarray, detail: np.ndarray,
                              dy: float = 0.0, dx: float = 0.0,
                              strength: float = 0.42) -> np.ndarray:
    """Keep base macro/mid structure but import only compatible fine frequency.

    A tiny subpixel phase shift prevents a detail donor's sampling grid from
    becoming locked to the placement grid.  No low-frequency colour or complete
    recognizable feature is transferred from the second donor.
    """
    base_f, detail_f = base.astype(np.float32), detail.astype(np.float32)
    detail_high = detail_f - gaussian_filter(detail_f, (2.2, 2.2, 0))
    if dx or dy:
        detail_high = image_shift(detail_high, (dy, dx, 0), order=1, mode="reflect", prefilter=False)
    base_high = base_f - gaussian_filter(base_f, (2.2, 2.2, 0))
    scale = np.clip(float(base_high.std()) / max(float(detail_high.std()), 1.0e-4), 0.55, 1.8)
    # Add, rather than cross-fade, the independently sourced fine band.  A
    # cross-fade reduced RMS detail when the phases were uncorrelated.
    return np.clip(base_f + strength * scale * detail_high, 0, 255).astype(np.uint8)


def stitch_block(dataset: Path, ox: int, oy: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    strips = []
    for j, y in enumerate(range(oy, oy + n)):
        row = []
        for i, x in enumerate(range(ox, ox + n)):
            tile = read_tile(dataset / "tiles" / "5" / str(x) / f"{y}.trn")
            hh = np.asarray(tile.decoded_heights(), np.float32).reshape(tile.height, tile.width)[1:-1, 1:-1]
            row.append(hh if i == 0 else hh[:, 1:])
        strip = np.concatenate(row, 1); strips.append(strip if j == 0 else strip[1:])
        metres_per_pixel = (tile.extent[2] - tile.extent[0]) / SIZE
    native = np.concatenate(strips, 0); extent = SIZE * n
    height = np.asarray(Image.fromarray(native, "F").resize((extent, extent), Image.Resampling.BILINEAR))
    dy, dx = np.gradient(height, metres_per_pixel)
    return height, np.hypot(dx, dy)


@dataclass
class SynthesisResult:
    rgb: np.ndarray
    donor: np.ndarray
    source_x: np.ndarray
    source_y: np.ndarray
    age: np.ndarray
    reuse: np.ndarray
    target_style: np.ndarray
    selected_material: np.ndarray
    transition: np.ndarray
    confidence: np.ndarray
    graph_kind: np.ndarray
    detail_donor: np.ndarray
    metrics: dict


def _source_anchor(bank: PatchBank, atlas: Atlas, donor: np.ndarray,
                   sx: np.ndarray, sy: np.ndarray, y: int, x: int,
                   h: int, w: int) -> int | None:
    points = []
    if x > 0:
        points.append((y + h // 2, x - 1))
    if y > 0:
        points.append((y - 1, x + w // 2))
    for py, px in points:
        tile = int(donor[py, px])
        if tile < 0:
            continue
        gx, gy = atlas.global_xy(np.array([tile]), np.array([sx[py, px]]), np.array([sy[py, px]]))
        _, node = bank.source_tree.query((float(gx[0]), float(gy[0])))
        return int(node)
    return None


def synth_block(target_h: np.ndarray, target_s: np.ndarray, atlas: Atlas,
                seed: int = 7, boundary_style: BoundaryStyleField | None = None) -> SynthesisResult:
    extent = target_h.shape[0]
    rgb = np.zeros((extent, extent, 3), np.uint8); known = np.zeros((extent, extent), bool)
    donor = np.full((extent, extent), -1, np.int32)
    source_x = np.full((extent, extent), -1, np.int32); source_y = source_x.copy()
    age = np.zeros((extent, extent), np.float32); reuse = np.zeros_like(age)
    style_map = np.zeros((extent, extent, STYLE_DIM), np.float32)
    selected_material = np.zeros((extent, extent, 3), np.float32)
    transition_map = np.zeros((extent, extent), np.float32)
    confidence_map = np.zeros((extent, extent), np.float32)
    graph_kind = np.zeros((extent, extent), np.uint8)
    detail_donor = np.full((extent, extent), -1, np.int32)
    source_use: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    rng = np.random.default_rng(seed)
    metrics: dict[str, object] = {"banks": {}, "placements": 0, "transition_placements": 0,
                                  "frequency_decoupled_placements": 0,
                                  "pool_totals": {"geographic": 0, "visual": 0,
                                                  "global": 0, "transition": 0}}

    for sizes, overlap in LEVELS:
        banks = {size: build_patch_bank(atlas, size) for size in sizes}
        for size, bank in banks.items():
            metrics["banks"][str(size)] = {"patches": int(len(bank.tile)),
                                            "transition_corridors": int(len(bank.transition_nodes)),
                                            "graph_edges": int(bank.geographic.size + bank.visual.size)}
        print(f"  level sizes={sizes}: {sum(len(b.tile) for b in banks.values())} graph nodes", flush=True)
        y = 0
        while y < extent:
            size = int(rng.choice(sizes)); step = size - overlap; bank = banks[size]
            h = min(size, extent - y); x = 0
            while x < extent:
                w = min(size, extent - x)
                geometry = target_geometry(target_h, target_s, y, x, h, w)
                raw_style = predict_target_style(bank, geometry)
                if boundary_style is not None:
                    border_style, border_distance = boundary_style.sample(x + 0.5 * w, y + 0.5 * h)
                    border_weight = 0.18 + 0.56 * np.exp(-border_distance / (1.6 * SIZE))
                    raw_style = ((1.0 - border_weight) * raw_style + border_weight * border_style).astype(np.float32)
                    raw_style[MATERIAL] /= max(float(raw_style[MATERIAL].sum()), 1.0e-6)
                anchor = _source_anchor(bank, atlas, donor, source_x, source_y, y, x, h, w)
                current_style = bank.style[anchor] if anchor is not None else None
                desired = bounded_style_target(raw_style, current_style)
                pool, waypoint, provenance = candidate_pool(bank, desired, anchor, rng)
                for key, value in provenance.items():
                    metrics["pool_totals"][key] += int(value)
                existing = rgb[y:y + h, x:x + w]; existing_known = known[y:y + h, x:x + w]
                overlap_mask = np.zeros((h, w), bool)
                if x > 0:
                    overlap_mask[:, :min(overlap, w)] = True
                if y > 0:
                    overlap_mask[:min(overlap, h), :] = True
                overlap_mask &= existing_known
                target_norm = desired / STYLE_SCALE
                style_cost = np.mean((_normalized_style(bank.style[pool]) - target_norm) ** 2, axis=1)
                terrain_cost = np.mean((bank.geometry[pool] - geometry[None]) ** 2, axis=1)
                waypoint_style = bank.style[waypoint] / STYLE_SCALE
                graph_cost = np.mean((_normalized_style(bank.style[pool]) - waypoint_style) ** 2, axis=1)
                confidence_cost = (1.0 - bank.confidence[pool]) ** 2
                context_cost = np.zeros(len(pool), np.float32)
                if overlap_mask.any():
                    context_lab = srgb_to_lab(existing[overlap_mask]).mean(0) / np.array((100, 45, 45))
                    context_cost = np.linalg.norm(bank.style[pool, LAB] - context_lab, axis=1)
                reuse_cost = np.zeros(len(pool), np.float32)
                for position, node in enumerate(pool):
                    node = int(node)
                    key = (int(bank.tile[node]), int(bank.x[node]) // SRC_CELL, int(bank.y[node]) // SRC_CELL)
                    uses = source_use.get(key, ())
                    reuse_cost[position] = sum(np.hypot(py - y, px - x) > REUSE_FAR_R for py, px in uses)
                prior_age = float(np.mean(age[y:y + h, x:x + w][overlap_mask])) if overlap_mask.any() else 0.0
                locality_cost = np.ones(len(pool), np.float32)
                if anchor is not None:
                    geo_nodes = bank.geographic[anchor]
                    visual_nodes = bank.visual[anchor]
                    locality_cost[np.isin(pool, visual_nodes)] = 0.38
                    locality_cost[np.isin(pool, geo_nodes)] = 0.0
                    locality_cost[pool == waypoint] = 0.12
                    # Strong at the start of a run; after ~150 px the graph may
                    # naturally leave the source region for a compatible one.
                    locality_cost *= 0.22 + 0.78 * np.exp(-prior_age / COH_LENGTH_PX)
                pre = (W_TERRAIN * terrain_cost + W_STYLE * style_cost + W_GRAPH * graph_cost +
                       W_CONTEXT * context_cost + W_CONFIDENCE * confidence_cost + W_REUSE_FAR * reuse_cost)
                pre += W_LOCALITY * locality_cost
                shortlist_count = min(32, len(pool))
                shortlist = pool[np.argpartition(pre, shortlist_count - 1)[:shortlist_count]]
                pre_by_node = {int(node): float(cost) for node, cost in zip(pool, pre)}
                best, best_cost = int(shortlist[0]), np.inf
                for node in shortlist:
                    node = int(node); tile = int(bank.tile[node]); sy, sx = int(bank.y[node]), int(bank.x[node])
                    candidate = atlas.rgb[tile][sy:sy + h, sx:sx + w]
                    colour, gradient, _ = rgb_patch_error(existing, candidate, overlap_mask)
                    cost = pre_by_node[node] + W_OVERLAP * colour + W_GRADIENT * gradient + 0.02 * rng.random()
                    if cost < best_cost:
                        best, best_cost = node, cost
                tile, sy, sx = int(bank.tile[best]), int(bank.y[best]), int(bank.x[best])
                candidate = atlas.rgb[tile][sy:sy + h, sx:sx + w]
                detail_tile = tile
                if size <= 108 and len(shortlist) > 1:
                    alternatives = shortlist[bank.tile[shortlist] != tile]
                    if not len(alternatives):
                        alternatives = shortlist[shortlist != best]
                    if len(alternatives):
                        material_error = np.linalg.norm(
                            bank.style[alternatives, MATERIAL] - bank.style[best, MATERIAL], axis=1)
                        orientation_error = np.linalg.norm(
                            bank.style[alternatives, ORIENTATION] - bank.style[best, ORIENTATION], axis=1)
                        energy_error = np.abs(
                            bank.style[alternatives, MID_ROUGHNESS] - desired[MID_ROUGHNESS])
                        detail_node = int(alternatives[np.argmin(
                            5.0 * material_error + orientation_error + 1.5 * energy_error)])
                        detail_tile = int(bank.tile[detail_node])
                        dsy, dsx = int(bank.y[detail_node]), int(bank.x[detail_node])
                        detail_patch = atlas.rgb[detail_tile][dsy:dsy + h, dsx:dsx + w]
                        candidate = frequency_decoupled_patch(
                            candidate, detail_patch, float(rng.uniform(-1.25, 1.25)),
                            float(rng.uniform(-1.25, 1.25)))
                        metrics["frequency_decoupled_placements"] += 1
                take = rgb_quilt_take_mask(existing, candidate, existing_known,
                                            min(overlap, h, w), x > 0, y > 0)
                yy, xx = np.indices((h, w)); region = (slice(y, y + h), slice(x, x + w))
                rgb[region][take] = candidate[take]; known[region][take] = True
                donor[region][take] = tile
                source_x[region][take] = (sx + xx)[take]; source_y[region][take] = (sy + yy)[take]
                is_geo = anchor is not None and best in set(int(v) for v in bank.geographic[anchor])
                age[region][take] = prior_age + step if is_geo else 0.0
                key = (tile, sx // SRC_CELL, sy // SRC_CELL)
                source_use.setdefault(key, []).append((y, x)); reuse[region][take] = len(source_use[key])
                style_map[region][take] = desired
                selected_material[region][take] = bank.style[best, MATERIAL]
                transition_map[region][take] = bank.transition_strength[best]
                confidence_map[region][take] = bank.confidence[best]
                is_visual = anchor is not None and best in set(int(v) for v in bank.visual[anchor])
                graph_kind[region][take] = 1 if is_geo else (2 if is_visual else 3)
                detail_donor[region][take] = detail_tile
                metrics["placements"] += 1
                metrics["transition_placements"] += int(bank.transition_strength[best] > 0.18)
                x += step
            y += step
        del banks

    if not known.all():
        nearest = distance_transform_edt(~known, return_distances=False, return_indices=True)
        for array in (rgb, donor, source_x, source_y, age, reuse, style_map,
                      selected_material, transition_map, confidence_map, graph_kind, detail_donor):
            array[...] = array[tuple(nearest)]
    metrics["mean_confidence"] = float(confidence_map.mean())
    metrics["mean_transition_strength"] = float(transition_map.mean())
    return SynthesisResult(rgb, donor, source_x, source_y, age, reuse, style_map,
                           selected_material, transition_map, confidence_map, graph_kind,
                           detail_donor, metrics)


def normalize_lowfreq(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32)
    low = gaussian_filter(f, (LOWFREQ_SIGMA, LOWFREQ_SIGMA, 0))
    smooth = gaussian_filter(low, (MACRO_SMOOTH_SIGMA, MACRO_SMOOTH_SIGMA, 0))
    return np.clip(f - low + smooth, 0, 255).astype(np.uint8)


def source_discontinuities(donor: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    jump = np.zeros(donor.shape, bool)
    jump[:, 1:] |= ((donor[:, 1:] != donor[:, :-1]) | (sx[:, 1:] != sx[:, :-1] + 1) |
                    (sy[:, 1:] != sy[:, :-1]))
    jump[1:] |= ((donor[1:] != donor[:-1]) | (sx[1:] != sx[:-1]) |
                 (sy[1:] != sy[:-1] + 1))
    return jump


def _style_rgb(style: np.ndarray) -> np.ndarray:
    # Material diagnostic, deliberately not a synthesized colour result.
    palette = np.array(((52, 132, 55), (132, 126, 116), (235, 241, 246)), np.float32)
    return np.clip(style[..., MATERIAL] @ palette, 0, 255).astype(np.uint8)


def write_outputs(output: Path, atlas: Atlas, result: SynthesisResult,
                  normalized: np.ndarray, elapsed: float) -> None:
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normalized, "RGB").save(output / "synth_block.png")
    Image.fromarray(result.rgb, "RGB").save(output / "synth_block_raw.png")
    Image.fromarray(normalized, "RGB").resize((normalized.shape[1] * 2, normalized.shape[0] * 2),
                                               Image.Resampling.NEAREST).save(output / "synth_block_2x.png")
    jump = source_discontinuities(result.donor, result.source_x, result.source_y)
    coord = np.zeros((*result.donor.shape, 3), np.uint8)
    coord[..., 0] = np.clip(result.source_x, 0, 255)
    coord[..., 1] = np.clip(result.source_y, 0, 255)
    coord[..., 2] = np.clip(result.donor / max(int(result.donor.max()), 1) * 230 + 20, 0, 255)
    jump_rgb = np.zeros_like(coord); jump_rgb[jump] = (255, 40, 40)
    Image.fromarray(coord, "RGB").save(output / "source_coord_map.png")
    Image.fromarray(jump_rgb, "RGB").save(output / "source_jumps.png")
    Image.fromarray(np.clip(result.age / max(float(result.age.max()), 1) * 255, 0, 255).astype(np.uint8), "L").save(output / "source_run_age.png")
    Image.fromarray(np.clip((result.reuse - 1) / 3 * 255, 0, 255).astype(np.uint8), "L").save(output / "source_reuse_heatmap.png")
    Image.fromarray(_style_rgb(result.target_style), "RGB").save(output / "target_style_material.png")
    selected = np.zeros_like(result.target_style); selected[..., MATERIAL] = result.selected_material
    Image.fromarray(_style_rgb(selected), "RGB").save(output / "selected_material.png")
    Image.fromarray(np.clip(result.transition * 420, 0, 255).astype(np.uint8), "L").save(output / "transition_corridors.png")
    Image.fromarray(np.clip(result.confidence * 255, 0, 255).astype(np.uint8), "L").save(output / "donor_confidence.png")
    graph_palette = np.array(((0, 0, 0), (48, 190, 92), (80, 120, 235), (238, 156, 52)), np.uint8)
    Image.fromarray(graph_palette[np.clip(result.graph_kind, 0, 3)], "RGB").save(output / "graph_candidate_kind.png")
    detail_viz = np.clip(result.detail_donor / max(int(result.detail_donor.max()), 1) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(detail_viz, "L").save(output / "detail_donor_map.png")
    result.metrics.update({"atlas_tiles": len(atlas.rgb), "elapsed_seconds": elapsed,
                           "source_coherent_fraction": 1.0 - float(jump.mean())})
    (output / "metrics.json").write_text(json.dumps(result.metrics, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "alps-data" / "trn-alps-16km")
    parser.add_argument("--origin", type=int, nargs=2, default=(19, 9), metavar=("X", "Y"))
    parser.add_argument("--size", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true",
                        help="Use a 14x11 regional atlas for rapid iteration; default uses the full map")
    args = parser.parse_args()
    started = time.monotonic()
    ox, oy = args.origin; block = {(ox + i, oy + j) for i in range(args.size) for j in range(args.size)}
    coordinates = discover_level5(args.dataset)
    if args.quick:
        coordinates = [(x, y) for x, y in coordinates if ox - 5 <= x <= ox + 8 and oy - 4 <= y <= oy + 6]
    atlas = Atlas(args.dataset, coordinates, exclude=block | DEFAULT_BAD_TILES)
    print("stitching target terrain…", flush=True)
    height, slope = stitch_block(args.dataset, ox, oy, args.size)
    print("building graph and synthesizing…", flush=True)
    boundary_style = build_boundary_style_field(args.dataset, ox, oy, args.size)
    result = synth_block(height, slope, atlas, args.seed, boundary_style)
    normalized = normalize_lowfreq(result.rgb)
    elapsed = time.monotonic() - started
    write_outputs(args.output, atlas, result, normalized, elapsed)
    print(f"wrote {args.output / 'synth_block.png'} in {elapsed:.1f}s "
          f"({result.metrics['source_coherent_fraction']:.1%} source-coherent)")


if __name__ == "__main__":
    main()
