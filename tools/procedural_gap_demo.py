#!/usr/bin/env python3
"""Create a deterministic first-pass terrain-imagery gap reconstruction.

The demo deliberately does *not* read the target tile's RGB when creating its
replacement.  It learns a grass/rock/snow mapping from two adjacent, valid
tiles using their imagery plus DEM-derived height/slope, then uses that map to
guide a coarse-to-fine, coherent RGB source-coordinate field over larger,
target-excluding donor mosaics.

It is an offline experiment, not yet part of the terrain tile builder.  The
debug images make its semantic and texture decisions reviewable before that
integration.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import (distance_transform_edt, gaussian_filter, label, map_coordinates,
                           shift as image_shift)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from terrain_tiles import read_tile  # noqa: E402
from build_terrain_macro import classify_materials  # noqa: E402

SIZE = 256
GUTTER = 1
CLASS_ROCK, CLASS_GRASS, CLASS_SNOW = range(3)
CLASS_COLORS = np.array(((112, 112, 112), (48, 142, 57), (245, 245, 245)), dtype=np.uint8)
CLASS_NAMES = ("rock", "grass", "snow")
SEAM_GUIDE_COLOR = np.array((255, 0, 0), dtype=np.uint8)
# Legacy descriptor helpers remain below for comparing experimental variants;
# the default fast quilt no longer evaluates them for every candidate patch.
MORPH_RADII = (1, 2, 4, 8, 16, 32, 48)
MORPH_SIGMAS = (0.75, 1.5, 3.0)
ORIENT_BINS = 8
SYNTHESIS_PASSES = 1
MORPH_SCORE_CANDIDATES = 24
MORPH_SOFT_TEMPERATURE = 1.5
# Explicitly continue material contours that are cut by a known tile edge.
# The guide is learned from the donor SDF approaching that edge, not from a
# material-specific rule.
EDGE_FIT_DEPTH = 32
EDGE_GUIDE_DEPTH = 48
EDGE_SDF_CLIP = 12.0
EDGE_LABEL_WEIGHT = 3.0
EDGE_SDF_WEIGHT = 1.25
EDGE_DECAY_PIXELS = 18.0
EDGE_OCCUPANCY_WEIGHT = 4.0

# Final hierarchical coherence pass.  Patch quilting is good at borrowing real
# donor detail, but repeated patch replacement tends to fracture long source
# regions.  This pass measures the spatial correlation of every material in the
# donors and reconnects the synthesized field to the same correlation length,
# without any material-specific rules.
COHERENCE_RADII = (4, 8, 16)
COHERENCE_MAX_DIFFUSION_STEPS = 24
COHERENCE_TARGET_RADIUS = 8
COHERENCE_MICRO_WEIGHT = 0.10
COHERENCE_GUIDE_WEIGHT = 3.0
COHERENCE_GUIDE_DECAY = 20.0
COHERENCE_HARD_EDGE_DEPTH = 6
COHERENCE_DETAIL_BAND = 1.0

# Local topology model.  Autocorrelation alone cannot distinguish one long
# coherent swatch from many similarly-spaced islands.  Learn how many connected
# shapes occur in terrain-matched donor regions and how their area is distributed,
# then choose the local coherence scale that reproduces those statistics.
SHAPE_REGION = 96
SHAPE_REGION_STRIDE = 32
SHAPE_REFERENCE_STRIDE = 16
SHAPE_MIN_AREA = 2
SHAPE_NEIGHBOURS = 12
SHAPE_MAX_DIFFUSION_STEPS = 16
SHAPE_DENSITY_MATCH_WEIGHT = 1.5
SHAPE_REFERENCE_DENSITY_WEIGHT = 0.70
SHAPE_LOCAL_DENSITY_SIGMA = 18.0
SHAPE_LOCAL_DENSITY_ITERATIONS = 28
SHAPE_LOCAL_DENSITY_RATE = 0.10
SHAPE_AREA_THRESHOLDS = (2, 4, 8, 16, 32, 64)

# Boundary interpenetration model.  The topology pass deliberately produces a
# coherent macro segmentation, so a separate final stage is responsible for
# restoring the small donor-like incursions where materials bleed into one
# another.  The model is pair-generic: no material gets special treatment.
# For every unordered material pair it measures (a) excess raw transition
# length over a 2 px macro boundary, (b) the fraction of the boundary band
# occupied by the opposite macro material at several depths, (c) the 90th
# percentile penetration depth, and (d) directional cross-boundary mass.
BLEED_SMOOTH_SIGMA = 2.0
BLEED_DEPTHS = (1, 2, 4, 6)
BLEED_DEPTH_NORMALIZER = 16.0
BLEED_MAX_RESTORE_DEPTH = 14.0
BLEED_REGION = 96
BLEED_REGION_STRIDE = 32
BLEED_REFERENCE_STRIDE = 16
BLEED_NEIGHBOURS = 12
BLEED_DENSITY_MATCH_WEIGHT = 0.80
BLEED_SOURCE_SUPPORT_SIGMA = 1.0
# Keep restored boundary detail subordinate to the coherent segmentation.  A
# weak anchor reintroduced more donor speckle than the target terrain state
# supports (and pushed global boundary length well beyond the reference).
BLEED_MACRO_ANCHOR = 1.0
BLEED_MIN_PAIR_TRANSITIONS = 8
BLEED_PROFILE_FLOOR = np.asarray((0.15, 0.03, 0.03, 0.025, 0.02,
                                  0.20, 0.012, 0.012), dtype=np.float32)

# Whole-image RGB reconstruction.  Material labels guide the search but never
# split the imagery into independent material pools.  Each accepted patch
# writes both RGB and a source-coordinate field, and later patches strongly
# prefer the source-space continuation implied by already-written neighbours.
TEXTURE_LEVELS = ((96, 32, 8), (48, 16, 8), (24, 8, 8))
TEXTURE_DESCRIPTOR_SIZE = 8
TEXTURE_SHORTLIST = 48
TEXTURE_BOUNDARY_BAND = 16
TEXTURE_LABEL_WEIGHT = 30.0
TEXTURE_OVERLAP_WEIGHT = 3.0
TEXTURE_GRADIENT_WEIGHT = 1.5
TEXTURE_COHERENCE_WEIGHT = 2.5
TEXTURE_SCALE_WEIGHT = 0.35
TEXTURE_BOUNDARY_WEIGHT = 32.0
TEXTURE_REUSE_WEIGHT = 2.0
TEXTURE_LOCAL_SHORTLIST = 32
# Maximum depth (px) the neighbour-snapping min-cut may reach inward before it
# must return to real donor content.  The boundary line itself is always exact;
# this bounds how far the donor may interlock into the last target columns/rows
# along a jagged seam where a tone/content clash remains, so the transition is
# irregular rather than a straight discontinuity.  Cross-seam grading keeps the
# clash small, so the cut mostly stays shallow and only deepens where needed.
TEXTURE_EDGE_SNAP_DEPTH = 8

# Per-swatch colour grading.  Quilted swatches come from differently-exposed
# source regions, leaving hard light/dark block seams even when the material is
# right.  Before a swatch is cut in, its exposure/colour is regraded to match
# the pixels it joins instead of searching for a better-lit swatch.  The grade
# is a per-channel Reinhard transfer in linear light, regularised toward the
# original by STRENGTH so texture contrast survives and exposure cannot drift,
# clamped so a mixed-material overlap cannot force a wild recolour.
TEXTURE_GRADE_STRENGTH = 0.8
TEXTURE_GRADE_FALLBACK_STRENGTH = 0.5
TEXTURE_GRADE_MIN_OVERLAP = 24
TEXTURE_GRADE_STD_CLAMP = (0.7, 1.4)
TEXTURE_GRADE_MAX_SHIFT = 0.20


@dataclass
class TerrainSample:
    x: int
    y: int
    rgb: np.ndarray
    height: np.ndarray
    slope: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    colour_class: np.ndarray | None = None


def image_path(dataset: Path, x: int, y: int) -> Path:
    return dataset / "imagery" / "5" / str(x) / f"{y}.png"


def tile_path(dataset: Path, x: int, y: int) -> Path:
    return dataset / "tiles" / "5" / str(x) / f"{y}.trn"


def load_sample(dataset: Path, x: int, y: int) -> TerrainSample:
    with Image.open(image_path(dataset, x, y)) as source:
        rgb = np.asarray(source.convert("RGB"))[GUTTER:-GUTTER, GUTTER:-GUTTER].copy()
    tile = read_tile(tile_path(dataset, x, y))
    heights = np.asarray(tile.decoded_heights(), dtype=np.float32).reshape(tile.height, tile.width)
    g = tile.gutter
    heights = heights[g:-g, g:-g]
    height = np.asarray(Image.fromarray(heights, "F").resize((SIZE, SIZE), Image.Resampling.BILINEAR))
    metres_per_height_sample = (tile.extent[2] - tile.extent[0]) / max(heights.shape[1] - 1, 1)
    dz_dy, dz_dx = np.gradient(height, metres_per_height_sample * 0.5)
    slope = np.hypot(dz_dx, dz_dy)
    return TerrainSample(x, y, rgb, height, slope, dz_dx, dz_dy)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    unit = rgb.astype(np.float32) / 255.0
    return np.where(unit <= 0.04045, unit / 12.92, ((unit + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    unit = np.clip(rgb, 0.0, 1.0)
    return np.where(unit <= 0.0031308, unit * 12.92, 1.055 * unit ** (1.0 / 2.4) - 0.055)


def colour_labels(rgb: np.ndarray) -> np.ndarray:
    """Use the repository macro classifier, plus its documented snow rule."""
    grass, rock, _rejected = classify_materials(rgb)
    unit = rgb.astype(np.float32) / 255.0
    high, low = unit.max(axis=2), unit.min(axis=2)
    saturation = (high - low) / np.maximum(high, 1.0e-5)
    luminance = unit @ np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)
    snow = (luminance > 0.64) & (saturation < 0.28)
    result = np.full(rgb.shape[:2], CLASS_ROCK, dtype=np.uint8)
    result[grass] = CLASS_GRASS
    result[snow] = CLASS_SNOW
    return result


def material_probabilities(target: TerrainSample, donors: list[TerrainSample]) -> np.ndarray:
    """Learn P(material | shared physical elevation, shared slope) from donors.

    This intentionally retains uncertainty.  A heightmap cannot prove whether
    a particular ledge is grass, bare rock, or snow, so the output is a smooth
    categorical probability field rather than a hard altitude threshold.  The
    bin axes are shared metres and slope units: each tile's percentile rank
    would incorrectly make the same physical altitude a different state.
    """
    height_bins, slope_bins = 32, 16
    low_height = min(float(d.height.min()) for d in donors)
    high_height = max(float(d.height.max()) for d in donors)
    height_span = max(high_height - low_height, 25.0)
    low_slope = min(float(d.slope.min()) for d in donors)
    high_slope = max(float(d.slope.max()) for d in donors)
    slope_span = max(high_slope - low_slope, 0.01)
    evidence = np.full((height_bins, slope_bins, 3), 0.25, dtype=np.float32)
    for donor in donors:
        h = np.clip(((donor.height - low_height) / height_span * height_bins).astype(np.int32), 0, height_bins - 1)
        s = np.clip(((donor.slope - low_slope) / slope_span * slope_bins).astype(np.int32), 0, slope_bins - 1)
        for kind in range(3):
            np.add.at(evidence[:, :, kind], (h.ravel(), s.ravel()),
                      (donor.colour_class == kind).ravel().astype(np.float32))
    # Nearby terrain states should influence each other, while the modest
    # pseudo-count keeps unobserved combinations uncertain rather than black.
    evidence = gaussian_filter(evidence, sigma=(1.35, 1.0, 0.0), mode="nearest")
    evidence /= evidence.sum(axis=2, keepdims=True)
    h = np.clip((target.height - low_height) / height_span, 0.0, 1.0) * (height_bins - 1)
    s = np.clip((target.slope - low_slope) / slope_span, 0.0, 1.0) * (slope_bins - 1)
    coordinates = np.stack((h, s))
    probabilities = np.dstack([
        map_coordinates(evidence[:, :, kind], coordinates, order=1, mode="nearest")
        for kind in range(3)
    ])
    return probabilities / probabilities.sum(axis=2, keepdims=True)


def smooth_noise(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Independent 40 px / 15 px / 5 px region noise, normalized per octave."""
    result = np.zeros(shape, dtype=np.float32)
    for weight, radius in ((0.55, 20.0), (0.30, 7.5), (0.15, 2.5)):
        octave = gaussian_filter(rng.standard_normal(shape), radius, mode="wrap")
        octave = (octave - octave.mean()) / max(float(octave.std()), 1.0e-5)
        result += weight * octave
    return result


def remove_tiny_components(labels: np.ndarray, minimum_area: int = 18) -> np.ndarray:
    """Replace tiny categorical islands with the dominant eight-neighbour class."""
    result = labels.copy()
    structure = np.ones((3, 3), dtype=np.uint8)
    for kind in range(3):
        components, count = label(result == kind, structure=structure)
        for component in range(1, count + 1):
            component_mask = components == component
            if int(component_mask.sum()) < minimum_area:
                result[component_mask] = 255
    while np.any(result == 255):
        changed = False
        for y, x in np.argwhere(result == 255):
            y0, y1 = max(y - 1, 0), min(y + 2, result.shape[0])
            x0, x1 = max(x - 1, 0), min(x + 2, result.shape[1])
            neighbours = result[y0:y1, x0:x1]
            neighbours = neighbours[neighbours != 255]
            if neighbours.size:
                result[y, x] = np.bincount(neighbours, minlength=3).argmax()
                changed = True
        if not changed:
            result[result == 255] = CLASS_ROCK
    return result


def stochastic_labels(probabilities: np.ndarray, seed: int = 2308) -> np.ndarray:
    """Turn probabilities into coherent terrain regions rather than pixel flips."""
    scores = np.log(np.maximum(probabilities, 1.0e-5))
    for kind in range(3):
        scores[:, :, kind] += 0.85 * smooth_noise(probabilities.shape[:2],
                                                   np.random.default_rng(seed + kind * 7919))
    return remove_tiny_components(scores.argmax(axis=2).astype(np.uint8))


def minimum_vertical_seam(cost: np.ndarray) -> np.ndarray:
    """Return one minimum-cost top-to-bottom seam column per row."""
    height, width = cost.shape
    energy = np.empty_like(cost, dtype=np.float32)
    parents = np.zeros((height, width), dtype=np.int16)
    energy[0, :] = cost[0, :]
    for y in range(1, height):
        for x in range(width):
            x0, x1 = max(x - 1, 0), min(x + 2, width)
            predecessor = x0 + int(np.argmin(energy[y - 1, x0:x1]))
            energy[y, x] = cost[y, x] + energy[y - 1, predecessor]
            parents[y, x] = predecessor
    seam = np.empty(height, dtype=np.int16)
    seam[-1] = int(np.argmin(energy[-1, :]))
    for y in range(height - 1, 0, -1):
        seam[y - 1] = parents[y, seam[y]]
    return seam


def patch_seam_cost(existing: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Cheap seams prefer agreement and existing categorical boundaries."""
    boundary = ((existing != np.roll(existing, 1, 0)) |
                (existing != np.roll(existing, -1, 0)) |
                (existing != np.roll(existing, 1, 1)) |
                (existing != np.roll(existing, -1, 1)))
    return (existing != candidate).astype(np.float32) - 0.40 * boundary.astype(np.float32)


def quilt_take_mask(existing: np.ndarray, candidate: np.ndarray, known: np.ndarray,
                    overlap: int, has_left: bool, has_top: bool) -> np.ndarray:
    """Cut a coherent minimum-cost seam through each already-filled overlap."""
    ownership = np.ones(existing.shape, dtype=bool)
    # Equal-cost seams are common in categorical images.  A very small centre
    # bias keeps those ties inside the overlap rather than choosing column zero
    # and degenerating back into a full rectangular overwrite.
    coordinate = np.arange(overlap, dtype=np.float32)
    centre_bias = 0.02 * ((coordinate - 0.5 * (overlap - 1)) / max(overlap, 1)) ** 2
    if has_left:
        cost = patch_seam_cost(existing[:, :overlap], candidate[:, :overlap])
        seam = minimum_vertical_seam(cost + centre_bias[None, :])
        ownership[:, :overlap] &= np.arange(overlap)[None, :] >= seam[:, None]
    if has_top:
        cost = patch_seam_cost(existing[:overlap, :], candidate[:overlap, :]).T
        seam = minimum_vertical_seam(cost + centre_bias[None, :])
        ownership[:overlap, :] &= np.arange(overlap)[:, None] >= seam[None, :]
    # A seam only decides ownership where two patches overlap.  Unwritten
    # pixels always come from the new patch.
    return ~known | ownership


def boundary_length(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask[:, 1:] != mask[:, :-1]) +
                 np.count_nonzero(mask[1:, :] != mask[:-1, :]))


def axial_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Unsigned orientation difference: 0 and pi describe the same line."""
    return 0.5 * np.abs(np.angle(np.exp(2j * (first - second))))


def morphology_descriptor(labels: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Material scale, edge roughness, and terrain-relative shape descriptors."""
    feature_count = 1 + len(MORPH_RADII) + len(MORPH_SIGMAS) + 2 * ORIENT_BINS + 4
    hy, hx = np.gradient(height.astype(np.float32))
    gradient_angle = np.arctan2(hy, hx)
    contour_angle = gradient_angle + np.pi / 2.0
    terrain_strength = np.hypot(hx, hy)
    output = []
    for kind in range(3):
        mask = labels == kind
        fill = float(mask.mean())
        if fill < 0.005 or fill > 0.995:
            output.append(np.full(feature_count, np.nan, dtype=np.float32)); continue
        m = mask.astype(np.float32); denom = max(fill * (1.0 - fill), 1e-5)
        corr = []
        for radius in MORPH_RADII:
            values = [np.mean(m[:, :-radius] * m[:, radius:]),
                      np.mean(m[:-radius, :] * m[radius:, :]),
                      np.mean(m[:-radius, :-radius] * m[radius:, radius:]),
                      np.mean(m[radius:, :-radius] * m[:-radius, radius:])]
            corr.append((float(np.mean(values)) - fill * fill) / denom)
        perimeter = max(boundary_length(mask), 1.0)
        rough = [np.log(perimeter / max(boundary_length(
            gaussian_filter(m, sigma, mode="nearest") >= 0.5), 1.0))
                 for sigma in MORPH_SIGMAS]
        sdf = gaussian_filter(signed_distance(labels, kind).astype(np.float32), 1.0)
        by, bx = np.gradient(sdf)
        tangent_angle = np.arctan2(by, bx) + np.pi / 2.0
        boundary = (np.abs(sdf) < 1.5) & (terrain_strength > 1e-4)

        def orientation_hist(reference: np.ndarray) -> np.ndarray:
            delta = axial_difference(tangent_angle[boundary], reference[boundary])
            if delta.size == 0:
                return np.full(ORIENT_BINS, np.nan, dtype=np.float32)
            histogram, _ = np.histogram(delta, bins=ORIENT_BINS, range=(0.0, np.pi / 2.0))
            return (histogram / max(histogram.sum(), 1)).astype(np.float32)

        components, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        elongation, alignment, areas = [], [], []
        for component_id in range(1, count + 1):
            coords = np.argwhere(components == component_id)
            if len(coords) < 12:
                continue
            covariance = np.cov(coords.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            major = eigenvectors[:, np.argmax(eigenvalues)]
            major_angle = np.arctan2(major[0], major[1])
            centre_y, centre_x = np.round(coords.mean(axis=0)).astype(int)
            elongation.append(np.sqrt(eigenvalues.max() / max(eigenvalues.min(), 1e-5)))
            alignment.append(1.0 - 2.0 * axial_difference(major_angle,
                                                            contour_angle[centre_y, centre_x]) / np.pi)
            areas.append(len(coords))
        if elongation:
            component_shape = [np.log(np.median(areas)), np.median(elongation),
                               np.percentile(elongation, 90), np.mean(alignment)]
        else:
            component_shape = [np.nan] * 4
        output.append(np.asarray([fill, *corr, *rough, *orientation_hist(contour_angle),
                                  *orientation_hist(gradient_angle), *component_shape], dtype=np.float32))
    return np.asarray(output)


def morphology_cost(actual: np.ndarray, desired: np.ndarray, scale: np.ndarray) -> float:
    valid = np.isfinite(actual) & np.isfinite(desired) & np.isfinite(scale)
    if not valid.any(): return 0.0
    error = (actual[valid] - desired[valid]) / np.maximum(scale[valid], 0.05)
    feature_weights = np.ones(actual.shape[-1], dtype=np.float32)
    orientation_start = 1 + len(MORPH_RADII) + len(MORPH_SIGMAS)
    feature_weights[orientation_start:orientation_start + 2 * ORIENT_BINS] = 1.75
    feature_weights[orientation_start + 2 * ORIENT_BINS:] = 1.50
    weights = np.broadcast_to(feature_weights, actual.shape)[valid]
    return float(np.average(error * error, weights=weights))


def signed_distance(labels: np.ndarray, kind: int) -> np.ndarray:
    mask = labels == kind
    return distance_transform_edt(mask) - distance_transform_edt(~mask)


def extrapolate_edge_sdf(labels: np.ndarray, side: str, fit_depth: int = EDGE_FIT_DEPTH,
                         guide_depth: int = EDGE_GUIDE_DEPTH) -> np.ndarray:
    """Predict per-class signed distance fields through a cut tile edge.

    ``side`` names the edge of the *known donor* that touches the target:
    ``south`` for the donor above the target, and ``west`` for the donor to the
    target's right.  A least-squares derivative is fitted through the final
    donor band for every coordinate along the seam, smoothed along the seam,
    then extrapolated into the unknown tile.  The zero crossings therefore
    continue the position, width and tangent of shapes that were cut off.
    """
    fit_depth = int(np.clip(fit_depth, 4, SIZE))
    guide_depth = int(np.clip(guide_depth, 1, SIZE))
    t = np.arange(fit_depth, dtype=np.float32)
    centred = t - t.mean()
    denominator = max(float(np.sum(centred * centred)), 1.0e-6)
    output = []

    for kind in range(3):
        field = gaussian_filter(signed_distance(labels, kind).astype(np.float32),
                                sigma=0.8, mode="nearest")
        if side == "south":
            band = field[-fit_depth:, :]
            mean = band.mean(axis=0)
            derivative = np.sum(centred[:, None] * (band - mean[None, :]), axis=0) / denominator
            derivative = gaussian_filter(derivative, 1.5, mode="nearest")
            derivative = np.clip(derivative, -1.5, 1.5)
            seam_value = mean + derivative * (fit_depth - 1 - t.mean())
            predicted = np.stack([seam_value + derivative * (distance + 1)
                                  for distance in range(guide_depth)], axis=0)
        elif side == "west":
            band = field[:, :fit_depth]
            mean = band.mean(axis=1)
            derivative = np.sum((band - mean[:, None]) * centred[None, :], axis=1) / denominator
            derivative = gaussian_filter(derivative, 1.5, mode="nearest")
            derivative = np.clip(derivative, -1.5, 1.5)
            seam_value = mean + derivative * (0.0 - t.mean())
            # Moving into the target means moving in the negative donor-x direction.
            predicted = np.stack([seam_value - derivative * (distance + 1)
                                  for distance in range(guide_depth)], axis=1)
        else:
            raise ValueError(f"unsupported donor edge: {side}")
        output.append(predicted.astype(np.float32))

    return np.asarray(output)


def extrapolate_edge_occupancy(labels: np.ndarray, side: str,
                               fit_depth: int = EDGE_FIT_DEPTH,
                               guide_depth: int = EDGE_GUIDE_DEPTH) -> np.ndarray:
    """Continue local class porosity through an edge independently of class.

    Signed distance describes the macro shape, but two equally shaped regions
    can be solid or fragmented.  This extrapolates smoothed one-hot occupancy
    so loose fragments remain loose and dense material remains dense near the
    seam.  No class receives a different rule or weight.
    """
    fit_depth = int(np.clip(fit_depth, 4, SIZE))
    guide_depth = int(np.clip(guide_depth, 1, SIZE))
    t = np.arange(fit_depth, dtype=np.float32)
    centred = t - t.mean()
    denominator = max(float(np.sum(centred * centred)), 1.0e-6)
    output = []
    for kind in range(3):
        field = gaussian_filter((labels == kind).astype(np.float32),
                                sigma=1.25, mode="nearest")
        if side == "south":
            band = field[-fit_depth:, :]
            mean = band.mean(axis=0)
            derivative = np.sum(centred[:, None] * (band - mean[None, :]), axis=0) / denominator
            derivative = np.clip(gaussian_filter(derivative, 1.5, mode="nearest"), -0.12, 0.12)
            seam_value = mean + derivative * (fit_depth - 1 - t.mean())
            predicted = np.stack([seam_value + derivative * (distance + 1)
                                  for distance in range(guide_depth)], axis=0)
        elif side == "west":
            band = field[:, :fit_depth]
            mean = band.mean(axis=1)
            derivative = np.sum((band - mean[:, None]) * centred[None, :], axis=1) / denominator
            derivative = np.clip(gaussian_filter(derivative, 1.5, mode="nearest"), -0.12, 0.12)
            seam_value = mean + derivative * (0.0 - t.mean())
            predicted = np.stack([seam_value - derivative * (distance + 1)
                                  for distance in range(guide_depth)], axis=1)
        else:
            raise ValueError(f"unsupported donor edge: {side}")
        output.append(np.clip(predicted, 0.0, 1.0).astype(np.float32))
    return np.asarray(output)


def edge_hard_constraints(north: TerrainSample, east: TerrainSample,
                          north_guide_labels: np.ndarray,
                          east_guide_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a generic shallow continuation band with joint corner handling."""
    depth = min(COHERENCE_HARD_EDGE_DEPTH, north_guide_labels.shape[0],
                east_guide_labels.shape[1], SIZE)
    fixed = np.zeros((SIZE, SIZE), dtype=bool)
    fixed_labels = np.zeros((SIZE, SIZE), dtype=np.uint8)

    north_mask = np.zeros_like(fixed)
    north_mask[:depth, :] = True
    north_values = np.zeros_like(fixed_labels)
    north_values[:depth, :] = north_guide_labels[:depth, :]

    east_mask = np.zeros_like(fixed)
    east_mask[:, SIZE - depth:] = True
    east_values = np.zeros_like(fixed_labels)
    for edge_depth in range(depth):
        east_values[:, SIZE - 1 - edge_depth] = east_guide_labels[:, edge_depth]

    only_north = north_mask & ~east_mask
    only_east = east_mask & ~north_mask
    agreement = north_mask & east_mask & (north_values == east_values)
    fixed[only_north | only_east | agreement] = True
    fixed_labels[only_north] = north_values[only_north]
    fixed_labels[only_east] = east_values[only_east]
    fixed_labels[agreement] = north_values[agreement]

    # Literal donor edges remain exact. Where the two one-pixel constraints
    # disagree at the north-east corner, the east assignment wins only at that
    # one unavoidable pixel; the surrounding overlap is resolved jointly.
    fixed[0, :] = True
    fixed_labels[0, :] = north.colour_class[-1, :]
    fixed[:, -1] = True
    fixed_labels[:, -1] = east.colour_class[:, 0]
    return fixed, fixed_labels


def edge_guide_cost(candidate_sdf: np.ndarray, candidate_labels: np.ndarray,
                    guide_sdf: np.ndarray, guide_labels: np.ndarray,
                    distance_from_edge: np.ndarray) -> np.ndarray:
    """Score candidate payloads against an extrapolated cut-shape guide."""
    decay = np.exp(-distance_from_edge.astype(np.float32) / EDGE_DECAY_PIXELS)
    decay /= max(float(decay.mean()), 1.0e-6)
    label_error = (candidate_labels != guide_labels[None]).astype(np.float32)
    label_cost = (label_error * decay[None]).mean(axis=(1, 2))
    sdf_error = np.abs(np.clip(candidate_sdf, -EDGE_SDF_CLIP, EDGE_SDF_CLIP) -
                       np.clip(guide_sdf[None], -EDGE_SDF_CLIP, EDGE_SDF_CLIP))
    sdf_cost = (sdf_error * decay[None, None]).mean(axis=(1, 2, 3)) / EDGE_SDF_CLIP
    return EDGE_LABEL_WEIGHT * label_cost + EDGE_SDF_WEIGHT * sdf_cost


def class_autocorrelation(mask: np.ndarray, radius: int) -> float:
    """Normalized two-point correlation, averaged over axes and diagonals."""
    m = mask.astype(np.float32)
    fill = float(m.mean())
    denominator = max(fill * (1.0 - fill), 1.0e-6)
    values = []
    if radius < m.shape[1]:
        values.append(float(np.mean(m[:, :-radius] * m[:, radius:])))
    if radius < m.shape[0]:
        values.append(float(np.mean(m[:-radius, :] * m[radius:, :])))
    if radius < m.shape[0] and radius < m.shape[1]:
        values.append(float(np.mean(m[:-radius, :-radius] * m[radius:, radius:])))
        values.append(float(np.mean(m[radius:, :-radius] * m[:-radius, radius:])))
    if not values:
        return 0.0
    return (float(np.mean(values)) - fill * fill) / denominator


def calibrate_class_biases(scores: np.ndarray, desired_density: np.ndarray,
                            fixed: np.ndarray, fixed_labels: np.ndarray,
                            iterations: int = 80) -> np.ndarray:
    """Choose global class biases so argmax occupancy matches learned density.

    The density target is learned from P(material | terrain), not hard-coded.
    Fixed seam pixels are removed from the free-pixel target before fitting.
    """
    biases = np.zeros(3, dtype=np.float32)
    free = ~fixed
    total = scores.shape[0] * scores.shape[1]
    target_counts = desired_density.astype(np.float64) * total
    fixed_counts = np.bincount(fixed_labels[fixed].ravel(), minlength=3).astype(np.float64)
    remaining = np.maximum(target_counts - fixed_counts, 1.0)
    target_free = remaining / remaining.sum()
    for _ in range(iterations):
        labels = (scores + biases[None, None, :]).argmax(axis=2)
        counts = np.bincount(labels[free].ravel(), minlength=3).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        actual = counts / counts.sum()
        biases += (0.28 * np.log(target_free / actual)).astype(np.float32)
    labels = (scores + biases[None, None, :]).argmax(axis=2).astype(np.uint8)
    labels[fixed] = fixed_labels[fixed]
    return labels


def restore_boundary_microdetail(macro: np.ndarray, original: np.ndarray,
                                 width: float = COHERENCE_DETAIL_BAND) -> np.ndarray:
    """Restore donor-like edge jitter without fragmenting macro regions.

    Only pixels very close to a macro boundary may reuse the pre-refinement
    label, and only when that label already exists in the local macro
    neighbourhood.  Thus tiny donor details decorate a coherent region rather
    than punching unrelated islands through its interior.
    """
    boundary = np.zeros(macro.shape, dtype=bool)
    boundary[:, 1:] |= macro[:, 1:] != macro[:, :-1]
    boundary[:, :-1] |= macro[:, :-1] != macro[:, 1:]
    boundary[1:, :] |= macro[1:, :] != macro[:-1, :]
    boundary[:-1, :] |= macro[:-1, :] != macro[1:, :]
    band = distance_transform_edt(~boundary) <= width
    allowed = np.zeros((*macro.shape, 3), dtype=bool)
    for kind in range(3):
        allowed[:, :, kind] = gaussian_filter((macro == kind).astype(np.float32),
                                               1.5, mode="nearest") > 0.03
    yy, xx = np.indices(macro.shape)
    use = band & allowed[yy, xx, original]
    result = macro.copy()
    result[use] = original[use]
    return result


def learn_direction_model(donors: tuple[TerrainSample, TerrainSample], kind: int) -> tuple[bool, float]:
    """Learn whether a class is elongated along gradient or contour directions.

    The choice and anisotropy strength come entirely from donor geometry.  No
    material name participates in this decision.
    """
    gradient_alignment = []
    contour_alignment = []
    elongations = []
    for donor in donors:
        mask = donor.colour_class == kind
        sdf = gaussian_filter(signed_distance(donor.colour_class, kind).astype(np.float32),
                              1.0, mode="nearest")
        by, bx = np.gradient(sdf)
        tangent = np.arctan2(by, bx) + np.pi / 2.0
        gradient = np.arctan2(donor.gradient_y, donor.gradient_x)
        contour = gradient + np.pi / 2.0
        boundary = np.abs(sdf) < 1.5
        if np.any(boundary):
            dg = axial_difference(tangent[boundary], gradient[boundary])
            dc = axial_difference(tangent[boundary], contour[boundary])
            gradient_alignment.append(float(np.mean(np.cos(2.0 * dg))))
            contour_alignment.append(float(np.mean(np.cos(2.0 * dc))))

        components, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        for component_id in range(1, count + 1):
            coords = np.argwhere(components == component_id)
            if len(coords) < 12:
                continue
            covariance = np.cov(coords.T)
            eigenvalues = np.linalg.eigvalsh(covariance)
            elongations.append(float(np.sqrt(eigenvalues.max() / max(eigenvalues.min(), 1.0e-5))))

    along_gradient = (np.mean(gradient_alignment) if gradient_alignment else 0.0) >= \
                     (np.mean(contour_alignment) if contour_alignment else 0.0)
    ratio = float(np.clip(np.median(elongations) if elongations else 1.0, 1.0, 3.0))
    return bool(along_gradient), ratio


def shifted_edge(values: np.ndarray, dy: int, dx: int) -> np.ndarray:
    padded = np.pad(values, ((1, 1), (1, 1)), mode="edge")
    height, width = values.shape
    return padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]


def directional_diffusion_step(values: np.ndarray, gradient_x: np.ndarray,
                               gradient_y: np.ndarray, along_gradient: bool,
                               anisotropy: float) -> np.ndarray:
    """One spatially varying diffusion step following measured terrain direction."""
    magnitude = np.hypot(gradient_x, gradient_y) + 1.0e-6
    ux = gradient_x / magnitude
    uy = gradient_y / magnitude
    if not along_gradient:
        ux, uy = -uy, ux

    accumulator = 1.5 * values
    weight_sum = np.full(values.shape, 1.5, dtype=np.float32)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)):
        norm = float(np.hypot(dx, dy))
        vx, vy = dx / norm, dy / norm
        alignment = (ux * vx + uy * vy) ** 2
        # There is always some cross-direction diffusion, but donor elongation
        # controls how much more strongly support travels along its preferred
        # terrain direction.
        weight = 0.35 + (anisotropy - 0.35) * alignment
        accumulator += weight * shifted_edge(values, dy, dx)
        weight_sum += weight
    return accumulator / np.maximum(weight_sum, 1.0e-6)


def component_profile(mask: np.ndarray) -> np.ndarray:
    """Multi-scale count + size spectrum for one categorical channel.

    For every component-area threshold this records both how many shapes remain
    and how much material mass those shapes own.  This is much more diagnostic
    than one median area: e.g. two fields may contain the same number of large
    grass ribbons while one also contains hundreds of detached 2-7 px islands.
    """
    components, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(components.ravel())[1:].astype(np.float32)
    total = max(float(areas.sum()), 1.0)
    features: list[float] = []
    for threshold in SHAPE_AREA_THRESHOLDS:
        selected = areas >= threshold
        features.append(float(np.count_nonzero(selected)) * 4096.0 / float(mask.size))
        features.append(float(areas[selected].sum()) / total if np.any(selected) else 0.0)
    if areas.size:
        pixel_weighted_area = float(np.sum(areas * areas) / total)
        largest_fraction = float(areas.max()) / total
    else:
        pixel_weighted_area = 0.0
        largest_fraction = 0.0
    features.extend((np.log1p(pixel_weighted_area), largest_fraction))
    return np.asarray(features, dtype=np.float32)

def all_component_profiles(labels: np.ndarray) -> np.ndarray:
    return np.asarray([component_profile(labels == kind) for kind in range(3)],
                      dtype=np.float32)


def component_profile_cost(actual: np.ndarray, desired: np.ndarray,
                           scale: np.ndarray) -> float:
    """Error in the full connected-component size spectrum.

    Counts at 8/16/32/64 px get more weight than the 2 px raw count.  Tiny donor
    fragments are useful as boundary detail, but they must not determine the
    macro topology.  Mass-at-threshold terms ensure that surviving shapes grow
    to donor-like area instead of merely reducing the number of islands.
    """
    weights = []
    floors = []
    for threshold in SHAPE_AREA_THRESHOLDS:
        if threshold <= 4:
            count_weight = 0.55
        elif threshold <= 16:
            count_weight = 1.5
        else:
            count_weight = 2.1
        weights.extend((count_weight, 1.35))
        floors.extend((2.0, 0.025))
    weights.extend((2.25, 1.75))
    floors.extend((0.30, 0.035))
    weights = np.asarray(weights, dtype=np.float32)
    floors = np.asarray(floors, dtype=np.float32)
    denominator = np.maximum(scale, floors)
    error = (actual - desired) / denominator
    return float(np.average(error * error, weights=weights))

def terrain_region_descriptor(sample: TerrainSample, x: int, y: int,
                              size: int) -> np.ndarray:
    h = sample.height[y:y + size, x:x + size]
    s = sample.slope[y:y + size, x:x + size]
    gx = sample.gradient_x[y:y + size, x:x + size]
    gy = sample.gradient_y[y:y + size, x:x + size]
    strength = np.hypot(gx, gy)
    # The doubled-angle mean is orientation-safe: a line at theta and theta+pi
    # describes the same terrain direction.
    angle = np.arctan2(gy, gx)
    orientation_x = float(np.mean(np.cos(2.0 * angle) * strength))
    orientation_y = float(np.mean(np.sin(2.0 * angle) * strength))
    return np.asarray((float(h.mean()), float(h.std()), float(s.mean()),
                       float(s.std()), float(strength.mean()),
                       orientation_x, orientation_y), dtype=np.float32)


def shape_region_origins(size: int, stride: int) -> list[int]:
    if size >= SIZE:
        return [0]
    result = list(range(0, SIZE - size + 1, stride))
    if result[-1] != SIZE - size:
        result.append(SIZE - size)
    return result


def build_shape_reference_library(
        donors: tuple[TerrainSample, TerrainSample]) -> tuple[np.ndarray, np.ndarray,
                                                               np.ndarray, np.ndarray,
                                                               np.ndarray]:
    """Measure topology in sliding donor regions.

    No class-specific rule is learned here.  Every donor window contributes its
    terrain state, class density, connected-shape count and area distribution.
    Target regions later retrieve donor windows with similar terrain.
    """
    descriptors, profiles, densities = [], [], []
    origins = shape_region_origins(SHAPE_REGION, SHAPE_REFERENCE_STRIDE)
    for donor in donors:
        for y in origins:
            for x in origins:
                labels_region = donor.colour_class[y:y + SHAPE_REGION,
                                                   x:x + SHAPE_REGION]
                descriptors.append(terrain_region_descriptor(donor, x, y, SHAPE_REGION))
                profiles.append(all_component_profiles(labels_region))
                densities.append(np.asarray([(labels_region == kind).mean()
                                             for kind in range(3)], dtype=np.float32))
    descriptors = np.asarray(descriptors, dtype=np.float32)
    profiles = np.asarray(profiles, dtype=np.float32)
    densities = np.asarray(densities, dtype=np.float32)
    descriptor_scale = np.maximum(np.std(descriptors, axis=0), 1.0e-4)
    profile_floor = []
    for _threshold in SHAPE_AREA_THRESHOLDS:
        profile_floor.extend((2.0, 0.025))
    profile_floor.extend((0.30, 0.035))
    profile_scale = np.maximum(np.nanstd(profiles, axis=0),
                               np.asarray(profile_floor, dtype=np.float32)[None, :])
    density_scale = np.maximum(np.std(densities, axis=0), 0.05)
    return descriptors, profiles, densities, descriptor_scale, profile_scale, density_scale


def predict_region_shape(target: TerrainSample, probabilities: np.ndarray,
                         x: int, y: int,
                         reference: tuple[np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray, np.ndarray]
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    descriptors, profiles, densities, descriptor_scale, profile_scale, density_scale = reference
    descriptor = terrain_region_descriptor(target, x, y, SHAPE_REGION)
    probability_density = probabilities[y:y + SHAPE_REGION,
                                        x:x + SHAPE_REGION].mean(axis=(0, 1))
    terrain_error = np.mean(((descriptors - descriptor[None]) /
                             descriptor_scale[None]) ** 2, axis=1)
    density_error = np.mean(((densities - probability_density[None]) /
                             density_scale[None]) ** 2, axis=1)
    lookup_cost = terrain_error + SHAPE_DENSITY_MATCH_WEIGHT * density_error
    count = min(SHAPE_NEIGHBOURS, len(lookup_cost))
    nearest = np.argpartition(lookup_cost, count - 1)[:count]
    weights = np.exp(-(lookup_cost[nearest] - lookup_cost[nearest].min()) * 1.5)
    weights /= max(float(weights.sum()), 1.0e-6)
    desired_profile = np.sum(profiles[nearest] * weights[:, None, None], axis=0)
    donor_density = np.sum(densities[nearest] * weights[:, None], axis=0)
    desired_density = (SHAPE_REFERENCE_DENSITY_WEIGHT * donor_density +
                       (1.0 - SHAPE_REFERENCE_DENSITY_WEIGHT) * probability_density)
    desired_density /= max(float(desired_density.sum()), 1.0e-6)
    return desired_profile.astype(np.float32), desired_density.astype(np.float32), profile_scale


def raised_cosine_window(size: int) -> np.ndarray:
    one = np.hanning(size).astype(np.float32)
    # A small non-zero skirt makes the first/last target pixels participate too.
    one = np.maximum(one, 0.06)
    return one[:, None] * one[None, :]


def calibrate_local_class_biases(scores: np.ndarray, desired_density: np.ndarray,
                                 fixed: np.ndarray, fixed_labels: np.ndarray) -> np.ndarray:
    """Match class occupancy locally instead of only over the whole tile.

    The previous global density calibration could satisfy the total grass count
    by placing too much grass in the edge band.  This compares blurred local
    occupancy to the terrain-matched donor density field and therefore cannot
    pay for a deficit in one region with excess pixels somewhere else.
    """
    biases = np.zeros_like(scores, dtype=np.float32)
    for _ in range(SHAPE_LOCAL_DENSITY_ITERATIONS):
        current = (scores + biases).argmax(axis=2).astype(np.uint8)
        current[fixed] = fixed_labels[fixed]
        actual = np.dstack([
            gaussian_filter((current == kind).astype(np.float32),
                            SHAPE_LOCAL_DENSITY_SIGMA, mode="nearest")
            for kind in range(3)
        ])
        delta = desired_density - actual
        biases += SHAPE_LOCAL_DENSITY_RATE * delta
        biases -= biases.mean(axis=2, keepdims=True)
        np.clip(biases, -2.5, 2.5, out=biases)
    result = (scores + biases).argmax(axis=2).astype(np.uint8)
    result[fixed] = fixed_labels[fixed]
    return result


def coherence_refine(labels: np.ndarray, target: TerrainSample,
                     donors: tuple[TerrainSample, TerrainSample],
                     probabilities: np.ndarray, north_guide_labels: np.ndarray,
                     east_guide_labels: np.ndarray) -> np.ndarray:
    """Match donor shape count and size locally, then preserve seam continuity.

    This replaces the old single global C(8) target.  A correlation value can be
    matched by either one large ribbon or dozens of islands, which is why the
    former output still had roughly twice as many grass/rock components as the
    reference.  Here each terrain region learns the donor distribution of:
      * number of connected shapes,
      * median and p90 shape area,
      * pixel-weighted shape area,
      * fraction owned by the largest local shape.
    The local directional diffusion depth is selected to match those metrics.
    """
    reference = build_shape_reference_library(donors)
    one_hot = np.stack([(labels == kind).astype(np.float32) for kind in range(3)], axis=2)
    score_sum = np.zeros_like(one_hot, dtype=np.float32)
    weight_sum = np.zeros((SIZE, SIZE), dtype=np.float32)
    density_sum = np.zeros_like(one_hot, dtype=np.float32)
    window = raised_cosine_window(SHAPE_REGION)
    directions = [learn_direction_model(donors, kind) for kind in range(3)]
    origins = shape_region_origins(SHAPE_REGION, SHAPE_REGION_STRIDE)

    for y in origins:
        for x in origins:
            desired_profile, desired_density, profile_scale = predict_region_shape(
                target, probabilities, x, y, reference)
            density_sum[y:y + SHAPE_REGION, x:x + SHAPE_REGION] += \
                window[:, :, None] * desired_density[None, None, :]
            for kind in range(3):
                support = one_hot[y:y + SHAPE_REGION, x:x + SHAPE_REGION, kind].copy()
                gx = target.gradient_x[y:y + SHAPE_REGION, x:x + SHAPE_REGION]
                gy = target.gradient_y[y:y + SHAPE_REGION, x:x + SHAPE_REGION]
                along_gradient, anisotropy = directions[kind]
                fill = float(np.clip(desired_density[kind], 0.002, 0.998))
                best_error = np.inf
                best_support = support.copy()
                best_threshold = float(np.quantile(support, 1.0 - fill))

                for step in range(SHAPE_MAX_DIFFUSION_STEPS + 1):
                    threshold = float(np.quantile(support, 1.0 - fill))
                    trial = support >= threshold
                    error = component_profile_cost(component_profile(trial),
                                                   desired_profile[kind],
                                                   profile_scale[kind])
                    # A tiny preference for shallower diffusion breaks ties in
                    # favour of retaining real donor boundary detail.
                    error += 0.002 * step
                    if error < best_error:
                        best_error = error
                        best_support = support.copy()
                        best_threshold = threshold
                    support = directional_diffusion_step(support, gx, gy,
                                                         along_gradient, anisotropy)

                spread = max(float(best_support.std()), 1.0e-3)
                local_score = (best_support - best_threshold) / spread
                score_sum[y:y + SHAPE_REGION, x:x + SHAPE_REGION, kind] += \
                    local_score * window
            weight_sum[y:y + SHAPE_REGION, x:x + SHAPE_REGION] += window

    score_sum /= np.maximum(weight_sum[:, :, None], 1.0e-6)
    desired_density_field = density_sum / np.maximum(weight_sum[:, :, None], 1.0e-6)
    desired_density_field /= np.maximum(desired_density_field.sum(axis=2, keepdims=True), 1.0e-6)

    # Retain a weak source-detail and per-pixel terrain prior.  Neither term is
    # strong enough to recreate the former island/confetti problem.
    scores = score_sum + COHERENCE_MICRO_WEIGHT * (one_hot - 0.5)
    scores += 0.10 * np.log(np.maximum(probabilities, 1.0e-4))

    # The extrapolated cut-shape guide remains soft away from the seam.  In the
    # previous version 24 complete rows/columns were hard-fixed, which is why a
    # grass-heavy seam could dominate the whole edge region.  Only the literal
    # seam is immutable now; local shape-count/density metrics control its depth.
    north_depth = min(north_guide_labels.shape[0], SIZE)
    for row in range(1, north_depth):
        weight = COHERENCE_GUIDE_WEIGHT * np.exp(-float(row) / COHERENCE_GUIDE_DECAY)
        scores[row, np.arange(SIZE), north_guide_labels[row]] += weight
    east_depth = min(east_guide_labels.shape[1], SIZE)
    for depth in range(1, east_depth):
        x = SIZE - 1 - depth
        weight = COHERENCE_GUIDE_WEIGHT * np.exp(-float(depth) / COHERENCE_GUIDE_DECAY)
        scores[np.arange(SIZE), x, east_guide_labels[:, depth]] += weight

    fixed, fixed_labels = edge_hard_constraints(
        donors[0], donors[1], north_guide_labels, east_guide_labels)

    macro = calibrate_local_class_biases(scores, desired_density_field,
                                         fixed, fixed_labels)
    macro[fixed] = fixed_labels[fixed]
    return macro

def material_pairs() -> tuple[tuple[int, int], ...]:
    return tuple((first, second) for first in range(3) for second in range(first + 1, 3))


def smooth_categorical(labels: np.ndarray, sigma: float = BLEED_SMOOTH_SIGMA) -> np.ndarray:
    """Low-pass a categorical map without inventing intermediate labels."""
    support = np.dstack([
        gaussian_filter((labels == kind).astype(np.float32), sigma, mode="nearest")
        for kind in range(3)
    ])
    return support.argmax(axis=2).astype(np.uint8)


def pair_transition_count(labels: np.ndarray, first: int, second: int) -> int:
    horizontal = (((labels[:, :-1] == first) & (labels[:, 1:] == second)) |
                  ((labels[:, :-1] == second) & (labels[:, 1:] == first)))
    vertical = (((labels[:-1, :] == first) & (labels[1:, :] == second)) |
                ((labels[:-1, :] == second) & (labels[1:, :] == first)))
    return int(np.count_nonzero(horizontal) + np.count_nonzero(vertical))


def pair_boundary_pixels(labels: np.ndarray, first: int, second: int) -> np.ndarray:
    """Pixels incident to a first/second boundary in a categorical map."""
    result = np.zeros(labels.shape, dtype=bool)
    horizontal = (((labels[:, :-1] == first) & (labels[:, 1:] == second)) |
                  ((labels[:, :-1] == second) & (labels[:, 1:] == first)))
    result[:, :-1] |= horizontal
    result[:, 1:] |= horizontal
    vertical = (((labels[:-1, :] == first) & (labels[1:, :] == second)) |
                ((labels[:-1, :] == second) & (labels[1:, :] == first)))
    result[:-1, :] |= vertical
    result[1:, :] |= vertical
    return result


def pair_bleed_profile(labels: np.ndarray, first: int, second: int) -> np.ndarray:
    """Measure how strongly two materials interpenetrate at their macro boundary.

    The descriptor deliberately normalizes away the amount of macro boundary.
    A long but perfectly clean boundary therefore scores as low bleed, while a
    short boundary containing many fingers, pinholes and incursions scores high.

    Layout:
      0                  log(raw pair transitions / macro pair transitions)
      1..len(DEPTHS)     opposite-material fraction in macro boundary bands
      next               90th percentile penetration depth / normalizer
      final two          macro first->raw second and second->raw first pixel mass
    """
    macro = smooth_categorical(labels)
    macro_transition_count = pair_transition_count(macro, first, second)
    pair_domain = (macro == first) | (macro == second)
    if macro_transition_count < BLEED_MIN_PAIR_TRANSITIONS or np.count_nonzero(pair_domain) < 32:
        return np.full(1 + len(BLEED_DEPTHS) + 3, np.nan, dtype=np.float32)

    raw_transition_count = pair_transition_count(labels, first, second)
    excess = (raw_transition_count + 1.0) / (macro_transition_count + 1.0)
    boundary = pair_boundary_pixels(macro, first, second)
    distance = distance_transform_edt(~boundary)
    first_to_second = (macro == first) & (labels == second)
    second_to_first = (macro == second) & (labels == first)
    swapped = first_to_second | second_to_first

    band_fractions: list[float] = []
    for depth in BLEED_DEPTHS:
        band = (distance <= depth) & pair_domain
        band_fractions.append(float(swapped[band].mean()) if np.any(band) else 0.0)

    penetrations = distance[swapped]
    q90 = float(np.quantile(penetrations, 0.90)) if penetrations.size else 0.0
    return np.asarray((np.log(max(excess, 1.0e-5)), *band_fractions,
                       q90 / BLEED_DEPTH_NORMALIZER,
                       float(first_to_second.mean()), float(second_to_first.mean())),
                      dtype=np.float32)


def all_bleed_profiles(labels: np.ndarray) -> np.ndarray:
    return np.asarray([pair_bleed_profile(labels, *pair) for pair in material_pairs()],
                      dtype=np.float32)


def bleed_profile_cost(actual: np.ndarray, desired: np.ndarray,
                       scale: np.ndarray) -> float:
    valid = np.isfinite(actual) & np.isfinite(desired) & np.isfinite(scale)
    if not np.any(valid):
        return 0.0
    weights = np.asarray((2.2, 1.0, 1.0, 1.15, 1.20, 1.7, 0.85, 0.85),
                         dtype=np.float32)
    denominator = np.maximum(scale, BLEED_PROFILE_FLOOR)
    error = (actual - desired) / denominator
    return float(np.average((error[valid] * error[valid]), weights=weights[valid]))


def build_bleed_reference_library(
        donors: tuple[TerrainSample, TerrainSample]) -> tuple[np.ndarray, np.ndarray,
                                                               np.ndarray, np.ndarray,
                                                               np.ndarray, np.ndarray]:
    """Learn pairwise bleed statistics in terrain-matched donor windows."""
    descriptors, profiles, densities = [], [], []
    origins = shape_region_origins(BLEED_REGION, BLEED_REFERENCE_STRIDE)
    for donor in donors:
        for y in origins:
            for x in origins:
                region = donor.colour_class[y:y + BLEED_REGION, x:x + BLEED_REGION]
                descriptors.append(terrain_region_descriptor(donor, x, y, BLEED_REGION))
                profiles.append(all_bleed_profiles(region))
                densities.append(np.asarray([(region == kind).mean() for kind in range(3)],
                                            dtype=np.float32))
    descriptors = np.asarray(descriptors, dtype=np.float32)
    profiles = np.asarray(profiles, dtype=np.float32)
    densities = np.asarray(densities, dtype=np.float32)
    descriptor_scale = np.maximum(np.std(descriptors, axis=0), 1.0e-4)
    profile_scale = np.maximum(np.nanstd(profiles, axis=0), BLEED_PROFILE_FLOOR[None, :])
    density_scale = np.maximum(np.std(densities, axis=0), 0.05)
    return descriptors, profiles, densities, descriptor_scale, profile_scale, density_scale


def predict_region_bleed(target: TerrainSample, probabilities: np.ndarray,
                         x: int, y: int,
                         reference: tuple[np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray, np.ndarray]
                         ) -> tuple[np.ndarray, np.ndarray]:
    descriptors, profiles, densities, descriptor_scale, profile_scale, density_scale = reference
    descriptor = terrain_region_descriptor(target, x, y, BLEED_REGION)
    expected_density = probabilities[y:y + BLEED_REGION, x:x + BLEED_REGION].mean(axis=(0, 1))
    terrain_error = np.mean(((descriptors - descriptor[None]) /
                             descriptor_scale[None]) ** 2, axis=1)
    density_error = np.mean(((densities - expected_density[None]) /
                             density_scale[None]) ** 2, axis=1)
    lookup_cost = terrain_error + BLEED_DENSITY_MATCH_WEIGHT * density_error
    count = min(BLEED_NEIGHBOURS, len(lookup_cost))
    nearest = np.argpartition(lookup_cost, count - 1)[:count]
    weights = np.exp(-(lookup_cost[nearest] - lookup_cost[nearest].min()) * 1.5)
    weights /= max(float(weights.sum()), 1.0e-6)

    values = profiles[nearest]
    weighted = weights[:, None, None]
    valid = np.isfinite(values)
    numerator = np.nansum(values * weighted, axis=0)
    denominator = np.sum(weighted * valid, axis=0)
    desired = np.divide(numerator, denominator,
                        out=np.full_like(numerator, np.nan),
                        where=denominator > 1.0e-8)
    return desired.astype(np.float32), profile_scale


def proposed_edge_gain(labels: np.ndarray, proposed_kind: int) -> np.ndarray:
    """Change in 4-neighbour transition count if each pixel became proposed_kind."""
    old = np.zeros(labels.shape, dtype=np.float32)
    new = np.zeros(labels.shape, dtype=np.float32)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbour = shifted_edge(labels, dy, dx)
        old += neighbour != labels
        new += neighbour != proposed_kind
    return new - old


def _restore_direction(current: np.ndarray, source: np.ndarray, stable_macro: np.ndarray,
                       distance: np.ndarray, from_kind: int, to_kind: int,
                       desired_mass: float, desired_depth: float,
                       desired_log_excess: float) -> np.ndarray:
    """Move one directional bleed mass toward its donor-measured target."""
    result = current.copy()
    target_count = int(round(max(desired_mass, 0.0) * result.size))
    existing = (stable_macro == from_kind) & (result == to_kind)
    existing_count = int(np.count_nonzero(existing))

    if existing_count > target_count:
        # Clean the least source-supported incursions first.  This handles a
        # pair that already bleeds more than its terrain-matched donor examples.
        source_support = gaussian_filter((source == to_kind).astype(np.float32),
                                         BLEED_SOURCE_SUPPORT_SIGMA, mode="nearest")
        candidates = np.argwhere(existing)
        if candidates.size:
            score = source_support[existing] * np.exp(-distance[existing] /
                                                      max(desired_depth, 1.0))
            remove_count = min(existing_count - target_count, len(score))
            if remove_count > 0:
                order = np.argpartition(score, remove_count - 1)[:remove_count]
                yy, xx = candidates[order].T
                result[yy, xx] = from_kind
        return result

    need = target_count - existing_count
    if need <= 0:
        return result

    eligible = ((stable_macro == from_kind) & (source == to_kind) &
                (result != to_kind) & (distance <= desired_depth))
    if not np.any(eligible):
        return result

    source_support = gaussian_filter((source == to_kind).astype(np.float32),
                                     BLEED_SOURCE_SUPPORT_SIGMA, mode="nearest")
    edge_gain = proposed_edge_gain(result, to_kind)
    target_excess = float(np.exp(desired_log_excess))
    jagged_weight = float(np.clip((target_excess - 1.0) / 4.0, 0.0, 1.0))
    depth_support = np.exp(-distance / max(desired_depth * 0.55, 1.0))
    score = (0.55 * source_support + 0.30 * depth_support +
             0.15 * jagged_weight * np.clip((edge_gain + 4.0) / 8.0, 0.0, 1.0))
    positions = np.argwhere(eligible)
    values = score[eligible]
    take_count = min(need, len(values))
    if take_count > 0:
        order = np.argpartition(values, len(values) - take_count)[-take_count:]
        yy, xx = positions[order].T
        result[yy, xx] = to_kind
    return result


def refine_region_bleed(macro: np.ndarray, source_detail: np.ndarray,
                        desired: np.ndarray, profile_scale: np.ndarray) -> np.ndarray:
    """Restore donor micro-boundary structure while preserving macro topology."""
    current = macro.copy()
    stable_macro = smooth_categorical(macro)
    pairs = material_pairs()
    for pair_index, (first, second) in enumerate(pairs):
        target = desired[pair_index]
        if not np.all(np.isfinite(target)):
            continue
        boundary = pair_boundary_pixels(stable_macro, first, second)
        if np.count_nonzero(boundary) < BLEED_MIN_PAIR_TRANSITIONS:
            continue
        distance = distance_transform_edt(~boundary)
        desired_depth = float(np.clip(target[1 + len(BLEED_DEPTHS)] *
                                      BLEED_DEPTH_NORMALIZER,
                                      1.0, BLEED_MAX_RESTORE_DEPTH))
        # The final two descriptor entries are directional macro->raw masses.
        current = _restore_direction(current, source_detail, stable_macro, distance,
                                     first, second, float(target[-2]), desired_depth,
                                     float(target[0]))
        current = _restore_direction(current, source_detail, stable_macro, distance,
                                     second, first, float(target[-1]), desired_depth,
                                     float(target[0]))

        # Directional mass controls how much material crosses the macro edge.
        # If the resulting edge is still too geometrically clean, add only
        # source-supported pixels that increase pair transitions until the
        # donor-normalized transition excess is approached.
        actual = pair_bleed_profile(current, first, second)
        if np.all(np.isfinite(actual)) and actual[0] < target[0]:
            macro_transitions = pair_transition_count(smooth_categorical(current), first, second)
            target_raw = int(round(np.exp(float(target[0])) * (macro_transitions + 1) - 1))
            current_raw = pair_transition_count(current, first, second)
            transition_deficit = max(0, target_raw - current_raw)
            if transition_deficit:
                pair_source = (((stable_macro == first) & (source_detail == second)) |
                               ((stable_macro == second) & (source_detail == first)))
                eligible = pair_source & (current != source_detail) & (distance <= desired_depth)
                if np.any(eligible):
                    gain_first = proposed_edge_gain(current, first)
                    gain_second = proposed_edge_gain(current, second)
                    gain = np.where(source_detail == first, gain_first, gain_second)
                    support = np.zeros(current.shape, dtype=np.float32)
                    for kind in (first, second):
                        local = gaussian_filter((source_detail == kind).astype(np.float32),
                                                BLEED_SOURCE_SUPPORT_SIGMA, mode="nearest")
                        support[source_detail == kind] = local[source_detail == kind]
                    score = np.maximum(gain, 0.0) + 0.35 * support
                    positions = np.argwhere(eligible & (gain > 0.0))
                    if positions.size:
                        values = score[eligible & (gain > 0.0)]
                        order = np.argsort(values)[::-1]
                        accumulated = 0.0
                        for index in order:
                            yy, xx = positions[index]
                            current[yy, xx] = source_detail[yy, xx]
                            accumulated += max(float(gain[yy, xx]), 0.0)
                            if accumulated >= transition_deficit:
                                break
    return current


def bleed_refine(macro: np.ndarray, source_detail: np.ndarray, target: TerrainSample,
                 donors: tuple[TerrainSample, TerrainSample], probabilities: np.ndarray,
                 fixed: np.ndarray, fixed_labels: np.ndarray) -> np.ndarray:
    """Match locally learned texture bleed after macro shape synthesis.

    The source_detail map is the pre-coherence donor-patch result.  We never add
    arbitrary noise.  Instead, donor-like micro labels are selectively restored
    until each terrain region reaches the amount, depth and pairwise complexity
    measured from terrain-matched donor regions.
    """
    reference = build_bleed_reference_library(donors)
    vote_sum = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    weight_sum = np.zeros((SIZE, SIZE), dtype=np.float32)
    window = raised_cosine_window(BLEED_REGION)
    origins = shape_region_origins(BLEED_REGION, BLEED_REGION_STRIDE)

    for y in origins:
        for x in origins:
            desired, profile_scale = predict_region_bleed(target, probabilities, x, y, reference)
            region = refine_region_bleed(
                macro[y:y + BLEED_REGION, x:x + BLEED_REGION],
                source_detail[y:y + BLEED_REGION, x:x + BLEED_REGION],
                desired, profile_scale)
            for kind in range(3):
                vote_sum[y:y + BLEED_REGION, x:x + BLEED_REGION, kind] += \
                    window * (region == kind)
            weight_sum[y:y + BLEED_REGION, x:x + BLEED_REGION] += window

    vote_sum /= np.maximum(weight_sum[:, :, None], 1.0e-6)
    vote_sum += BLEED_MACRO_ANCHOR * np.dstack([(macro == kind).astype(np.float32)
                                                for kind in range(3)])
    result = vote_sum.argmax(axis=2).astype(np.uint8)
    result[fixed] = fixed_labels[fixed]
    return result


def print_bleed_metrics(prefix: str, labels: np.ndarray) -> None:
    """Human-readable global diagnostics for regression runs."""
    print(prefix, flush=True)
    for pair_index, (first, second) in enumerate(material_pairs()):
        profile = pair_bleed_profile(labels, first, second)
        if not np.all(np.isfinite(profile)):
            continue
        excess = float(np.exp(profile[0]))
        depth = float(profile[1 + len(BLEED_DEPTHS)] * BLEED_DEPTH_NORMALIZER)
        forward = float(profile[-2])
        backward = float(profile[-1])
        print(f"  {CLASS_NAMES[first]} / {CLASS_NAMES[second]}: "
              f"transition_excess={excess:.3f}, q90_depth={depth:.2f}px, "
              f"cross_mass={forward + backward:.4f}", flush=True)


def target_context(result: np.ndarray, north: np.ndarray, east: np.ndarray,
                   x: int, y: int, size: int) -> np.ndarray:
    """Return target-space context, using real neighbour labels beyond its edges.

    The north/east donors are immutable context rather than synthesized data.
    The unavailable north-east corner is edge-clamped; it never contains the
    payload being evaluated and only supplies a neutral boundary for metrics.
    """
    yy, xx = np.indices((size, size))
    source_y, source_x = yy + y, xx + x
    output = result[np.clip(source_y, 0, SIZE - 1), np.clip(source_x, 0, SIZE - 1)].copy()
    from_north = source_y < 0
    north_x = np.clip(source_x, 0, SIZE - 1)
    output[from_north] = north[np.clip(source_y[from_north] + SIZE, 0, SIZE - 1),
                                north_x[from_north]]
    from_east = (source_x >= SIZE) & (source_y >= 0)
    east_y = np.clip(source_y, 0, SIZE - 1)
    output[from_east] = east[east_y[from_east],
                             np.clip(source_x[from_east] - SIZE, 0, SIZE - 1)]
    return output


def save_three_way_class_preview(output: Path | None, north_labels: np.ndarray,
                                 east_labels: np.ndarray, target_labels: np.ndarray,
                                 filename: str, known: np.ndarray | None = None) -> None:
    """Save the current categorical target beside both immutable donor tiles.

    Layout matches the normal atlas debug view: north donor in the upper-left,
    target in the lower-left, east donor in the lower-right, and the unavailable
    upper-right tile left black.  During an incomplete synthesis pass, target
    pixels not written yet are black rather than being misrepresented as rock.
    """
    if output is None:
        return
    pass_dir = output / "passes"
    pass_dir.mkdir(parents=True, exist_ok=True)

    north_rgb = CLASS_COLORS[north_labels]
    east_rgb = CLASS_COLORS[east_labels]
    target_rgb = CLASS_COLORS[target_labels].copy()
    if known is not None:
        target_rgb[~known] = 0

    canvas = np.zeros((SIZE * 2, SIZE * 2, 3), dtype=np.uint8)
    canvas[:SIZE, :SIZE] = north_rgb
    canvas[SIZE:, :SIZE] = target_rgb
    canvas[SIZE:, SIZE:] = east_rgb
    canvas = mark_tile_seams(canvas)
    Image.fromarray(canvas, "RGB").save(pass_dir / filename)
    print(f"    wrote {pass_dir / filename}", flush=True)


def save_shape_polygon_preview(output: Path | None, labels: np.ndarray,
                               stage: str) -> None:
    """Export the synthesized categorical shapes without texture or donors.

    The combined image shows the mutually exclusive polygon field.  Individual
    RGBA images isolate each material on transparency, which makes thin bridges,
    islands, and the exact region footprint easy to inspect.
    """
    if output is None:
        return
    polygon_dir = output / "shape_polygons"
    polygon_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(CLASS_COLORS[labels], "RGB").save(
        polygon_dir / f"{stage}_combined.png")
    for kind, name in enumerate(CLASS_NAMES):
        rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
        mask = labels == kind
        rgba[mask, :3] = CLASS_COLORS[kind]
        rgba[mask, 3] = 255
        Image.fromarray(rgba, "RGBA").save(polygon_dir / f"{stage}_{name}.png")


def write_macro_micro_diagnostics(output: Path, result, north: TerrainSample,
                                  east: TerrainSample) -> None:
    """Write inspectable pass15 shape stages without consulting target RGB."""
    output.mkdir(parents=True, exist_ok=True)
    save_shape_polygon_preview(output, result.initial_labels, "01_quilted_shapes")
    save_shape_polygon_preview(output, result.macro_labels, "02_coherent_macro_shapes")
    save_shape_polygon_preview(output, result.labels, "03_final_shapes")
    save_three_way_class_preview(output, north.colour_class, east.colour_class,
                                 result.initial_labels, "pass_01_probability_seed.png")
    save_three_way_class_preview(output, north.colour_class, east.colour_class,
                                 result.macro_labels, "pass_02_macro_sdf.png")
    save_three_way_class_preview(output, north.colour_class, east.colour_class,
                                 result.labels, "pass_03_micro_sdf.png")
    north_guide = np.repeat(north.colour_class[-1:, :], EDGE_GUIDE_DEPTH, axis=0)
    east_guide = np.repeat(east.colour_class[:, :1], EDGE_GUIDE_DEPTH, axis=1)
    save_edge_guide_preview(output, north_guide, east_guide)
    Image.fromarray(result.macro_low_rgb, "RGB").save(output / "macro_low_frequency.png")
    Image.fromarray(np.clip(result.transition_density /
                            max(float(result.transition_density.max()), 1.0e-8) * 255,
                            0, 255).astype(np.uint8), "L").save(
        output / "micro_transition_density.png")
    for name, values in (("macro_shape_source_map", result.macro_source_map),
                         ("micro_shape_source_map", result.micro_source_map)):
        maximum = max(int(values.max()), 1)
        image = np.clip(values / maximum * 255, 0, 255).astype(np.uint8)
        Image.fromarray(image, "L").save(output / f"{name}.png")
    payload = dict(result.metrics)
    payload["placements"] = [
        {"material": int(item.material), "source": [int(item.source_x), int(item.source_y)],
         "centre": [float(item.centre_x), float(item.centre_y)],
         "scale": float(item.scale), "angle_degrees": float(item.angle),
         "aspect": float(item.aspect), "elastic_pixels": float(item.elastic)}
        for item in result.placements
    ]
    (output / "macro_micro_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compose_macro_low_background(transformed_low: np.ndarray, shape_mask: np.ndarray,
                                 target_labels: np.ndarray,
                                 donors: list[TerrainSample], seed: int = 1515) -> np.ndarray:
    """Fill non-shape background with real low bands from selected style families."""
    from terrain_shape_synth import soft_choice
    rng = np.random.default_rng(seed)
    patch, stride = 128, 80
    origins = patch_axis_origins(patch, stride)
    window_1d = np.maximum(np.hanning(patch).astype(np.float32), 0.08)
    window = window_1d[:, None] * window_1d[None, :]
    accumulated = np.zeros((SIZE, SIZE, 3), np.float32)
    weights = np.zeros((SIZE, SIZE), np.float32)
    for y in origins:
        for x in origins:
            desired = np.bincount(target_labels[y:y + patch, x:x + patch].ravel(),
                                  minlength=3) / float(patch * patch)
            candidates, scores = [], []
            for donor_index, donor in enumerate(donors):
                for sy in (0, 64, SIZE - patch):
                    for sx in (0, 64, SIZE - patch):
                        labels = donor.colour_class[sy:sy + patch, sx:sx + patch]
                        density = np.bincount(labels.ravel(), minlength=3) / float(patch * patch)
                        candidates.append((donor_index, sy, sx))
                        scores.append(float(np.mean((density - desired) ** 2)))
            selected = candidates[soft_choice(np.asarray(scores), rng, temperature=0.08)]
            donor_index, sy, sx = selected
            low = gaussian_filter(donors[donor_index].rgb[sy:sy + patch, sx:sx + patch]
                                  .astype(np.float32), (16.0, 16.0, 0))
            accumulated[y:y + patch, x:x + patch] += low * window[..., None]
            weights[y:y + patch, x:x + patch] += window
    background = accumulated / np.maximum(weights[..., None], 1.0e-6)
    alpha = gaussian_filter(shape_mask.astype(np.float32), 5.0)[..., None]
    transformed = transformed_low.astype(np.float32)
    if shape_mask.any():
        grade = background[shape_mask].mean(0) - transformed[shape_mask].mean(0)
        transformed = np.clip(transformed + np.clip(grade, -42.0, 42.0), 0, 255)
    # Macro imagery is allowed to carry structure, but its colour must remain in
    # the active style family.  A partial alpha avoids a transformed low patch
    # reading as an opaque photographic plate.
    alpha *= 0.25
    return np.clip(background * (1.0 - alpha) + transformed * alpha,
                   0, 255).astype(np.uint8)


def save_edge_guide_preview(output: Path | None, north_labels: np.ndarray,
                            east_labels: np.ndarray) -> None:
    """Show the class-generic soft continuation field used at both edges."""
    if output is None:
        return
    polygon_dir = output / "shape_polygons"
    polygon_dir.mkdir(parents=True, exist_ok=True)
    votes = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    north_depth = min(north_labels.shape[0], SIZE)
    for row in range(north_depth):
        weight = np.exp(-float(row) / COHERENCE_GUIDE_DECAY)
        votes[row, np.arange(SIZE), north_labels[row]] += weight
    east_depth = min(east_labels.shape[1], SIZE)
    for depth in range(east_depth):
        x = SIZE - 1 - depth
        weight = np.exp(-float(depth) / COHERENCE_GUIDE_DECAY)
        votes[np.arange(SIZE), x, east_labels[:, depth]] += weight
    known = votes.sum(axis=2) > 0.0
    preview = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    preview[known] = CLASS_COLORS[votes.argmax(axis=2)[known]]
    Image.fromarray(preview, "RGB").save(
        polygon_dir / "00_edge_continuation_guides.png")
    strength = votes.max(axis=2)
    strength /= max(float(strength.max()), 1.0e-6)
    Image.fromarray(np.rint(strength * 255.0).astype(np.uint8), "L").save(
        polygon_dir / "00_edge_constraint_strength.png")


def constrained_patch_labels(target: TerrainSample, north: TerrainSample, east: TerrainSample,
                             probabilities: np.ndarray, preview_output: Path | None = None,
                             refine: bool = True) -> np.ndarray:
    """Quilt categorical donor patches into one target, constrained by a 2-D halo.

    `north` and `east` are immutable known material.  The target is never read
    for RGB labels: its DEM provides relative terrain shape plus absolute
    elevation and slope descriptors.  Overlapping donor patches are joined on
    minimum-cost categorical seams instead of overwriting rectangular regions.
    """
    patch, stride, halo = 64, 32, 32
    context_size = patch + 2 * halo
    donors = (north, east)
    candidates = []
    print("Preparing donor patch candidates…", flush=True)
    for donor in donors:
        donor_sdf = np.asarray([signed_distance(donor.colour_class, kind) for kind in range(3)])
        for y in range(0, SIZE - context_size + 1, 8):
            for x in range(0, SIZE - context_size + 1, 8):
                # Centre the 64 px payload in a 128 px source context so both
                # its north and east 32 px halos are available for edge tests.
                payload_y = y + halo
                payload_x = x + halo
                height = donor.height[payload_y:payload_y + patch,
                                      payload_x:payload_x + patch]
                slope = donor.slope[payload_y:payload_y + patch,
                                    payload_x:payload_x + patch]
                payload = donor.colour_class[payload_y:payload_y + patch,
                                             payload_x:payload_x + patch]
                candidates.append((payload,
                                   donor_sdf[:, payload_y:payload_y + patch,
                                             payload_x:payload_x + patch],
                                   donor.colour_class[y:y + halo,
                                                      payload_x:payload_x + patch],
                                   donor.colour_class[payload_y:payload_y + patch,
                                                      payload_x + patch:payload_x + patch + halo],
                                   donor_sdf[:, y:y + halo, payload_x:payload_x + patch],
                                   donor_sdf[:, payload_y:payload_y + patch,
                                             payload_x + patch:payload_x + patch + halo],
                                   (height - height.mean()) / max(float(height.std()), 1.0),
                                   (slope - slope.mean()) / max(float(slope.std()), 0.02),
                                   float(height.mean()), float(height.std()), float(slope.mean()),
                                   np.bincount(payload.ravel(), minlength=3).astype(np.float32) / payload.size))
                if len(candidates) % 100 == 0:
                    print(f"  described {len(candidates)} donor contexts…", flush=True)
    candidate_labels = np.asarray([item[0] for item in candidates])
    candidate_payload_sdf = np.asarray([item[1] for item in candidates])
    candidate_north = np.asarray([item[2] for item in candidates])
    candidate_east = np.asarray([item[3] for item in candidates])
    candidate_north_sdf = np.asarray([item[4] for item in candidates])
    candidate_east_sdf = np.asarray([item[5] for item in candidates])
    candidate_height = np.asarray([item[6] for item in candidates])
    candidate_slope = np.asarray([item[7] for item in candidates])
    candidate_mean_height = np.asarray([item[8] for item in candidates], dtype=np.float32)
    candidate_height_std = np.asarray([item[9] for item in candidates], dtype=np.float32)
    candidate_mean_slope = np.asarray([item[10] for item in candidates], dtype=np.float32)
    candidate_density = np.asarray([item[11] for item in candidates], dtype=np.float32)
    print(f"Prepared {len(candidates)} candidates; synthesizing {SYNTHESIS_PASSES} passes…", flush=True)
    result = np.zeros((SIZE, SIZE), dtype=np.uint8)
    known = np.zeros((SIZE, SIZE), dtype=bool)
    north_sdf = np.asarray([signed_distance(north.colour_class, kind) for kind in range(3)])
    east_sdf = np.asarray([signed_distance(east.colour_class, kind) for kind in range(3)])
    north_guide_sdf = extrapolate_edge_sdf(north.colour_class, "south")
    east_guide_sdf = extrapolate_edge_sdf(east.colour_class, "west")
    north_guide_occupancy = extrapolate_edge_occupancy(north.colour_class, "south")
    east_guide_occupancy = extrapolate_edge_occupancy(east.colour_class, "west")
    north_guide_scores = north_guide_sdf + EDGE_OCCUPANCY_WEIGHT * (north_guide_occupancy - 0.5)
    east_guide_scores = east_guide_sdf + EDGE_OCCUPANCY_WEIGHT * (east_guide_occupancy - 0.5)
    north_guide_labels = north_guide_scores.argmax(axis=0).astype(np.uint8)
    east_guide_labels = east_guide_scores.argmax(axis=0).astype(np.uint8)
    save_edge_guide_preview(preview_output, north_guide_labels, east_guide_labels)
    rng = np.random.default_rng(23808)
    for pass_index in range(SYNTHESIS_PASSES):
        print(f"  Synthesis pass {pass_index + 1}/{SYNTHESIS_PASSES}…", flush=True)
        axis_origins = list(range(0, SIZE - patch + 1, stride))
        if axis_origins[-1] != SIZE - patch:
            axis_origins.append(SIZE - patch)
        # Raster order gives every patch a conventional top and/or left
        # overlap.  This makes a true quilting seam possible and removes the
        # 64 px rectangular replacements produced by the old random order.
        ordered_origins = [(x, y) for y in axis_origins for x in axis_origins]
        for origin_index, (x, y) in enumerate(ordered_origins, start=1):
                if origin_index == 1 or origin_index % 50 == 0 or origin_index == len(ordered_origins):
                    print(f"    patch {origin_index}/{len(ordered_origins)}", flush=True)
                h_abs = target.height[y:y + patch, x:x + patch]
                s_abs = target.slope[y:y + patch, x:x + patch]
                h = (h_abs - h_abs.mean()) / max(float(h_abs.std()), 1.0)
                s = (s_abs - s_abs.mean()) / max(float(s_abs.std()), 0.02)
                terrain_cost = ((candidate_height - h) ** 2).mean(axis=(1, 2))
                terrain_cost += 0.55 * ((candidate_slope - s) ** 2).mean(axis=(1, 2))
                # Shape is deliberately relative, but morphology also needs
                # the absolute terrain state and expected class balance.
                height_cost = ((candidate_mean_height - float(h_abs.mean())) / 100.0) ** 2
                height_cost += 0.20 * ((candidate_height_std - float(h_abs.std())) / 50.0) ** 2
                height_cost += 0.20 * ((candidate_mean_slope - float(s_abs.mean())) / 0.20) ** 2
                target_density = probabilities[y:y + patch, x:x + patch].mean(axis=(0, 1))
                density_cost = ((candidate_density - target_density) ** 2).mean(axis=1)
                payload_known = known[y:y + patch, x:x + patch]
                context = ((candidate_labels != result[y:y + patch, x:x + patch]) * payload_known).sum(axis=(1, 2)) / max(int(payload_known.sum()), 1)
                local_probability = np.broadcast_to(probabilities[y:y + patch, x:x + patch],
                                                    candidate_labels.shape + (3,))
                prior = -np.log(np.take_along_axis(local_probability,
                                                    candidate_labels[..., None], axis=3)[..., 0]).mean(axis=(1, 2))
                # These terms must influence the patch itself, not merely the
                # desired descriptor calculated after the shortlist is made.
                cost = terrain_cost + 0.35 * height_cost + 0.25 * density_cost + 0.15 * context + 0.18 * prior
                # Every payload whose *context* reaches a neighbour receives
                # a constraint, preventing later interior patches from
                # eroding a continuation established at the exact seam.
                north_depth = max(0, halo - y)
                if north_depth:
                    north_context = north.colour_class[SIZE - north_depth:SIZE, x:x + patch]
                    candidate_context = candidate_north[:, halo - north_depth:halo, :]
                    cost += 2.5 * (candidate_context != north_context[None]).mean(axis=(1, 2))
                    north_halo_sdf = north_sdf[:, SIZE - north_depth:SIZE, x:x + patch]
                    candidate_sdf_context = candidate_north_sdf[:, :, halo - north_depth:halo, :]
                    cost += 0.35 * np.abs(candidate_sdf_context - north_halo_sdf[None]).mean(axis=(1, 2, 3)) / halo
                east_width = max(0, x + patch - (SIZE - halo))
                if east_width:
                    east_context = east.colour_class[y:y + patch, :east_width]
                    candidate_context = candidate_east[:, :, :east_width]
                    cost += 2.5 * (candidate_context != east_context[None]).mean(axis=(1, 2))
                    east_halo_sdf = east_sdf[:, y:y + patch, :east_width]
                    candidate_sdf_context = candidate_east_sdf[:, :, :, :east_width]
                    cost += 0.35 * np.abs(candidate_sdf_context - east_halo_sdf[None]).mean(axis=(1, 2, 3)) / halo

                # The halo terms above only say that the source situation before
                # the cut looks similar.  These terms explicitly score what the
                # candidate does *after* the cut against an extrapolated SDF, so
                # an incoming ribbon cannot immediately flare, pinch or shift.
                if y < EDGE_GUIDE_DEPTH:
                    guide_rows = min(patch, EDGE_GUIDE_DEPTH - y)
                    if guide_rows > 0:
                        guide_sdf = north_guide_sdf[:, y:y + guide_rows, x:x + patch]
                        guide_labels = north_guide_labels[y:y + guide_rows, x:x + patch]
                        candidate_sdf = candidate_payload_sdf[:, :, :guide_rows, :]
                        candidate_label_strip = candidate_labels[:, :guide_rows, :]
                        distance = np.broadcast_to(np.arange(y, y + guide_rows, dtype=np.float32)[:, None],
                                                   (guide_rows, patch))
                        cost += edge_guide_cost(candidate_sdf, candidate_label_strip,
                                                guide_sdf, guide_labels, distance)

                if x + patch > SIZE - EDGE_GUIDE_DEPTH:
                    guide_cols = min(patch, x + patch - (SIZE - EDGE_GUIDE_DEPTH))
                    if guide_cols > 0:
                        target_x0 = patch - guide_cols
                        global_x0 = x + target_x0
                        # east guide column 0 predicts target x=SIZE-1, so reverse
                        # the guide slice into ordinary left-to-right target order.
                        depth_start = SIZE - 1 - (global_x0 + guide_cols - 1)
                        depth_end = SIZE - global_x0
                        guide_sdf = east_guide_sdf[:, y:y + patch, depth_start:depth_end][:, :, ::-1]
                        guide_labels = east_guide_labels[y:y + patch, depth_start:depth_end][:, ::-1]
                        candidate_sdf = candidate_payload_sdf[:, :, :, target_x0:]
                        candidate_label_strip = candidate_labels[:, :, target_x0:]
                        global_x = np.arange(global_x0, global_x0 + guide_cols, dtype=np.float32)
                        distance_1d = SIZE - 1 - global_x
                        distance = np.broadcast_to(distance_1d[None, :], (patch, guide_cols))
                        cost += edge_guide_cost(candidate_sdf, candidate_label_strip,
                                                guide_sdf, guide_labels, distance)

                choice_count = min(MORPH_SCORE_CANDIDATES, len(cost))
                best = np.argpartition(cost, choice_count - 1)[:choice_count]
                # The chosen stochastic candidate is the candidate actually
                # written.  Keep this as one stage: recomputing a hard argmin
                # afterwards silently kills entropy and repeats the same shapes.
                weights = np.exp(-(cost[best] - cost[best].min()) /
                                 MORPH_SOFT_TEMPERATURE)
                selected = best[rng.choice(len(best), p=weights / weights.sum())]
                existing = result[y:y + patch, x:x + patch]
                chosen = candidate_labels[selected]
                take = quilt_take_mask(existing, chosen, payload_known,
                                       patch - stride, x > 0, y > 0)
                result[y:y + patch, x:x + patch][take] = chosen[take]
                known[y:y + patch, x:x + patch][take] = True

        save_three_way_class_preview(
            preview_output, north.colour_class, east.colour_class, result,
            f"pass_{pass_index + 1:02d}_patch_synthesis.png", known)

    result[~known] = stochastic_labels(probabilities)[~known]
    # The two physical tile seams are known exactly.  Candidate costs shape the
    # continuation inward, while these one-pixel anchors guarantee no visible
    # categorical discontinuity at the atlas boundary itself.
    result[0, :] = north.colour_class[-1, :]
    result[:, -1] = east.colour_class[:, 0]
    save_three_way_class_preview(
        preview_output, north.colour_class, east.colour_class, result,
        f"pass_{SYNTHESIS_PASSES + 1:02d}_seam_anchored.png")
    save_shape_polygon_preview(preview_output, result, "01_quilted_shapes")

    if not refine:
        print("Quick mode: skipping coherence and bleed refinement.", flush=True)
        return result

    # Patch synthesis deliberately retains source detail, but repeated patch
    # ownership decisions still shorten long connected swatches.  Rebuild only
    # the macro coherence from donor-measured correlation, then put the source
    # micro-boundary detail back.
    source_detail = result.copy()
    result = coherence_refine(result, target, (north, east), probabilities,
                              north_guide_labels, east_guide_labels)
    result = remove_tiny_components(result, minimum_area=2)
    fixed, fixed_labels = edge_hard_constraints(
        north, east, north_guide_labels, east_guide_labels)
    result[fixed] = fixed_labels[fixed]
    save_three_way_class_preview(
        preview_output, north.colour_class, east.colour_class, result,
        f"pass_{SYNTHESIS_PASSES + 2:02d}_coherence.png")
    save_shape_polygon_preview(preview_output, result, "02_coherent_macro_shapes")

    print_bleed_metrics("Bleed metrics before adaptive restore:", result)
    donor_profiles = np.asarray([all_bleed_profiles(north.colour_class),
                                 all_bleed_profiles(east.colour_class)], dtype=np.float32)
    donor_mean = np.nanmean(donor_profiles, axis=0)
    print("Terrain-independent donor bleed sanity check:", flush=True)
    for pair_index, (first, second) in enumerate(material_pairs()):
        profile = donor_mean[pair_index]
        if not np.all(np.isfinite(profile)):
            continue
        print(f"  {CLASS_NAMES[first]} / {CLASS_NAMES[second]}: "
              f"transition_excess={np.exp(profile[0]):.3f}, "
              f"q90_depth={profile[1 + len(BLEED_DEPTHS)] * BLEED_DEPTH_NORMALIZER:.2f}px, "
              f"cross_mass={profile[-2] + profile[-1]:.4f}", flush=True)

    result = bleed_refine(result, source_detail, target, (north, east), probabilities,
                          fixed, fixed_labels)
    save_three_way_class_preview(
        preview_output, north.colour_class, east.colour_class, result,
        f"pass_{SYNTHESIS_PASSES + 3:02d}_bleed_refine.png")
    save_shape_polygon_preview(preview_output, result, "03_final_shapes")
    print_bleed_metrics("Bleed metrics after adaptive restore:", result)
    print("Synthesis complete.", flush=True)
    return result


def flat_classes(labels: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[labels]


def tensor_orientation(gx: np.ndarray, gy: np.ndarray) -> tuple[float, float]:
    """Return axial structure orientation and normalized anisotropy."""
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    trace = jxx + jyy
    delta = float(np.hypot(jxx - jyy, 2.0 * jxy))
    return 0.5 * float(np.arctan2(2.0 * jxy, jxx - jyy)), delta / max(trace, 1.0e-6)


def semantic_layout_error(candidates: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Class-balanced full-layout mismatch for one or many candidate patches."""
    density = np.bincount(target.ravel(), minlength=3).astype(np.float32) / target.size
    # Square-root balancing prevents broad rock from hiding rare transitions
    # without letting a tiny class dominate the complete neighbourhood score.
    class_weight = 1.0 / np.sqrt(np.maximum(density, 0.05))
    pixel_weight = class_weight[target]
    mismatch = candidates != target if candidates.ndim == target.ndim else \
        candidates != target[None]
    axes = tuple(range(mismatch.ndim - target.ndim, mismatch.ndim))
    return np.sum(mismatch * pixel_weight, axis=axes) / max(float(pixel_weight.sum()), 1.0e-6)


def patch_axis_origins(patch: int, stride: int, extent: int = SIZE) -> list[int]:
    origins = list(range(0, extent - patch + 1, stride))
    if origins[-1] != extent - patch:
        origins.append(extent - patch)
    return origins


def load_texture_mosaic(dataset: Path, origin_x: int, origin_y: int,
                        width_tiles: int = 2, height_tiles: int = 2) -> TerrainSample:
    """Load a contiguous target-excluding source neighbourhood as one domain."""
    rows: list[list[TerrainSample]] = []
    for tile_y in range(origin_y, origin_y + height_tiles):
        row = []
        for tile_x in range(origin_x, origin_x + width_tiles):
            sample = load_sample(dataset, tile_x, tile_y)
            sample.colour_class = colour_labels(sample.rgb)
            row.append(sample)
        rows.append(row)

    def mosaic(attribute: str) -> np.ndarray:
        return np.concatenate([
            np.concatenate([getattr(sample, attribute) for sample in row], axis=1)
            for row in rows
        ], axis=0)

    return TerrainSample(origin_x, origin_y, mosaic("rgb"), mosaic("height"),
                         mosaic("slope"), mosaic("gradient_x"), mosaic("gradient_y"),
                         mosaic("colour_class"))


def load_texture_sources(dataset: Path, target_x: int, target_y: int,
                         preferred: list[tuple[int, int]] | None = None
                         ) -> list[TerrainSample]:
    """Load clean style-family sources, or the legacy four adjacent mosaics.

    Pass15 supplies a 70/20/10 whole-map selection made by the macro solver.
    Individual native-resolution tiles are deliberate: no RGB frequency band
    is scaled with a shape, and every selected source retains its real pixels.
    """
    if preferred:
        from terrain_synth import clean_source, discover_level5
        clean_pool: list[TerrainSample] = []
        available = set(discover_level5(dataset))
        candidates: list[tuple[int, int]] = []
        # Preserve style locality, but search outward when a chosen tile would
        # require so much repair that its Voronoi-like inpaint becomes texture.
        for radius in range(4):
            for px, py in preferred:
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if radius and max(abs(dx), abs(dy)) != radius:
                            continue
                        candidate = (int(px + dx), int(py + dy))
                        if candidate in available and candidate not in candidates:
                            candidates.append(candidate)
        # Deterministic global reserve ensures every material can still find a
        # clean donor if an entire local style family is contaminated.
        reserve = sorted(available, key=lambda p: min(
            np.hypot(p[0] - q[0], p[1] - q[1]) for q in preferred))
        candidates.extend(candidate for candidate in reserve if candidate not in candidates)
        rejected = 0
        for x, y in candidates:
            if (x, y) == (target_x, target_y):
                continue
            try:
                sample = load_sample(dataset, int(x), int(y))
            except (FileNotFoundError, OSError):
                continue
            cleaned, confidence, hard = clean_source(sample.rgb)
            repair_fraction = float(np.mean(confidence < 255))
            hard_fraction = float(np.mean(hard))
            if repair_fraction > 0.12 or hard_fraction > 0.06:
                rejected += 1
                continue
            sample.rgb = cleaned
            sample.colour_class = colour_labels(sample.rgb)
            sample.cleanup_confidence = float(confidence.mean() / 255.0)
            clean_pool.append(sample)
            if len(clean_pool) >= 80:
                break
        if clean_pool:
            # A locality-ordered first-N list can still contain fourteen grass
            # tiles.  Reserve four donors for every material, then fill the
            # remaining slots from the closest clean style families.
            sources: list[TerrainSample] = []
            source_keys: set[tuple[int, int]] = set()
            for kind in range(len(CLASS_NAMES)):
                ranked = sorted(
                    enumerate(clean_pool),
                    key=lambda item: (float(np.mean(item[1].colour_class == kind)) +
                                      0.08 * np.exp(-item[0] / 24.0)),
                    reverse=True)
                added = 0
                for _rank, sample in ranked:
                    key = (sample.x, sample.y)
                    if key not in source_keys:
                        sources.append(sample)
                        source_keys.add(key)
                        added += 1
                    if added >= 4:
                        break
            for sample in clean_pool:
                key = (sample.x, sample.y)
                if key not in source_keys:
                    sources.append(sample)
                    source_keys.add(key)
                if len(sources) >= 14:
                    break
            sources = sources[:14]
            print(f"  texture donors: {len(sources)} material-balanced clean tiles "
                  f"({rejected} low-confidence candidates rejected)", flush=True)
            return sources
        raise ValueError("whole-map style selection produced no usable texture sources")

    # Compatibility path used by --legacy-shapes.
    grid_size = 32
    origins = [
        (int(np.clip(target_x, 0, grid_size - 2)),
         int(np.clip(target_y - 2, 0, grid_size - 2))),
        (int(np.clip(target_x + 1, 0, grid_size - 2)),
         int(np.clip(target_y - 1, 0, grid_size - 2))),
        (int(np.clip(target_x - 2, 0, grid_size - 2)),
         int(np.clip(target_y - 1, 0, grid_size - 2))),
        (int(np.clip(target_x, 0, grid_size - 2)),
         int(np.clip(target_y + 1, 0, grid_size - 2))),
    ]
    unique_origins = []
    for origin in origins:
        covered = {(x, y) for y in range(origin[1], origin[1] + 2)
                   for x in range(origin[0], origin[0] + 2)}
        if (target_x, target_y) not in covered and origin not in unique_origins:
            unique_origins.append(origin)
    if not unique_origins:
        raise ValueError("could not construct a target-excluding texture source mosaic")
    return [load_texture_mosaic(dataset, *origin) for origin in unique_origins]


def load_material_exemplar_sources(dataset: Path, exclude: frozenset = frozenset(),
                                   per_material: int = 6, min_purity: float = 0.75
                                   ) -> dict[int, list[TerrainSample]]:
    """Whole-map search for the cleanest *and best-textured* tiles per material.

    The local style pool is often mixed — a rocky target has no pure grass
    neighbour, so the 14 texture donors can be <0.5 grass everywhere.  This
    scans every level-5 tile once to score per-material purity and 2-12 px
    texture energy, keeps the tiles above ``min_purity``, and ranks *those* by
    energy.  Ranking by purity alone selects flat meadow/snow tiles (a pure
    grass field can be nearly featureless); gating on purity then maximising
    texture energy yields real, richly-textured material to quilt from.
    """
    from terrain_synth import discover_level5, clean_source, tile_rgb
    scored: dict[int, list[tuple[float, float, int, int]]] = {
        k: [] for k in range(len(CLASS_NAMES))}
    # Rank on a downsampled copy: classification and a texture proxy at 96 px are
    # ~7x cheaper than at full resolution and preserve per-tile purity/energy
    # ordering.  Full-resolution tiles are only loaded for the chosen handful.
    scan = 96
    for (x, y) in discover_level5(dataset):
        if (x, y) in exclude:
            continue
        rgb = tile_rgb(dataset, x, y)
        if rgb is None:
            continue
        small = np.asarray(Image.fromarray(rgb, "RGB").resize((scan, scan),
                                                              Image.Resampling.BILINEAR))
        classes = colour_labels(small)
        smallf = small.astype(np.float32)
        mid = (gaussian_filter(smallf, (0.75, 0.75, 0), mode="nearest") -
               gaussian_filter(smallf, (4.5, 4.5, 0), mode="nearest"))
        energy = float(np.sqrt(np.mean(mid * mid)))
        for kind in range(len(CLASS_NAMES)):
            scored[kind].append((float(np.mean(classes == kind)), energy, int(x), int(y)))
    sources: dict[int, list[TerrainSample]] = {}
    for kind in range(len(CLASS_NAMES)):
        pure = [entry for entry in scored[kind] if entry[0] >= min_purity]
        if len(pure) < per_material:  # relax when a material is genuinely rare
            pure = sorted(scored[kind], key=lambda e: e[0], reverse=True)[:per_material * 4]
        pure.sort(key=lambda entry: entry[1], reverse=True)  # richest texture first
        chosen: list[TerrainSample] = []
        for purity, energy, x, y in pure:
            if len(chosen) >= per_material:
                break
            try:
                sample = load_sample(dataset, x, y)
            except (FileNotFoundError, OSError):
                continue
            cleaned, confidence, hard = clean_source(sample.rgb)
            if float(np.mean(confidence < 255)) > 0.12 or float(np.mean(hard)) > 0.06:
                continue
            sample.rgb = cleaned
            sample.colour_class = colour_labels(sample.rgb)
            chosen.append(sample)
        picked = [(e[0], e[1]) for e in pure[:len(chosen)]]
        note = (f"purity>={min_purity:.2f}, energy {picked[0][1]:.1f}..{picked[-1][1]:.1f}"
                if picked else "none")
        print(f"  {CLASS_NAMES[kind]} exemplar sources: {len(chosen)} ({note})", flush=True)
        sources[kind] = chosen
    return sources


def build_texture_patch_database(donors: list[TerrainSample], patch: int,
                                 source_stride: int) -> dict[str, np.ndarray]:
    """Describe intact RGB+semantic donor patches without material splitting."""
    from terrain_synth import srgb_to_lab
    thumb_axis = np.rint(np.linspace(0, patch - 1, TEXTURE_DESCRIPTOR_SIZE)).astype(int)
    indexes, extents, thumbnails, densities, terrain, terrain_angles = [], [], [], [], [], []
    mean_labs, source_positions = [], []
    north_contexts, east_contexts, north_valid, east_valid = [], [], [], []
    top_edges, top_gradients, right_edges, right_gradients = [], [], [], []
    band = min(TEXTURE_BOUNDARY_BAND, patch // 3)
    context_axis = np.rint(np.linspace(0, band - 1, 4)).astype(int)
    for donor_index, donor in enumerate(donors):
        y_origins = patch_axis_origins(patch, source_stride, donor.rgb.shape[0])
        x_origins = patch_axis_origins(patch, source_stride, donor.rgb.shape[1])
        for sy in y_origins:
            for sx in x_origins:
                labels = donor.colour_class[sy:sy + patch, sx:sx + patch]
                patch_rgb = donor.rgb[sy:sy + patch, sx:sx + patch]
                indexes.append((donor_index, sy, sx))
                extents.append((donor.rgb.shape[0], donor.rgb.shape[1]))
                thumbnails.append(labels[np.ix_(thumb_axis, thumb_axis)])
                densities.append(np.bincount(labels.ravel(), minlength=3) / labels.size)
                height = donor.height[sy:sy + patch, sx:sx + patch]
                slope = donor.slope[sy:sy + patch, sx:sx + patch]
                terrain.append((float(height.mean()), float(height.std()), float(slope.mean())))
                angle, anisotropy = tensor_orientation(
                    donor.gradient_x[sy:sy + patch, sx:sx + patch],
                    donor.gradient_y[sy:sy + patch, sx:sx + patch])
                terrain_angles.append((angle, anisotropy))
                mean_labs.append(srgb_to_lab(patch_rgb).mean(axis=(0, 1)))
                source_positions.append(((donor.x * SIZE + sx + patch * 0.5) / SIZE,
                                         (donor.y * SIZE + sy + patch * 0.5) / SIZE))
                top_edges.append(patch_rgb[0, thumb_axis])
                top_gradients.append(patch_rgb[1, thumb_axis].astype(np.int16) -
                                     patch_rgb[0, thumb_axis].astype(np.int16))
                right_edges.append(patch_rgb[thumb_axis, -1])
                right_gradients.append(patch_rgb[thumb_axis, -1].astype(np.int16) -
                                       patch_rgb[thumb_axis, -2].astype(np.int16))
                if sy >= band:
                    context = donor.rgb[sy - band:sy, sx:sx + patch]
                    north_contexts.append(context[context_axis][:, thumb_axis])
                    north_valid.append(True)
                else:
                    north_contexts.append(np.zeros((4, TEXTURE_DESCRIPTOR_SIZE, 3),
                                                   dtype=np.uint8))
                    north_valid.append(False)
                if sx + patch + band <= donor.rgb.shape[1]:
                    context = donor.rgb[sy:sy + patch, sx + patch:sx + patch + band]
                    east_contexts.append(context[thumb_axis][:, context_axis])
                    east_valid.append(True)
                else:
                    east_contexts.append(np.zeros((TEXTURE_DESCRIPTOR_SIZE, 4, 3),
                                                  dtype=np.uint8))
                    east_valid.append(False)
    return {
        "index": np.asarray(indexes, dtype=np.int16),
        "extent": np.asarray(extents, dtype=np.int16),
        "thumbnail": np.asarray(thumbnails, dtype=np.uint8),
        "density": np.asarray(densities, dtype=np.float32),
        "terrain": np.asarray(terrain, dtype=np.float32),
        "terrain_angle": np.asarray(terrain_angles, dtype=np.float32),
        "mean_lab": np.asarray(mean_labs, dtype=np.float32),
        "source_position": np.asarray(source_positions, dtype=np.float32),
        "north_context": np.asarray(north_contexts, dtype=np.uint8),
        "east_context": np.asarray(east_contexts, dtype=np.uint8),
        "north_valid": np.asarray(north_valid, dtype=bool),
        "east_valid": np.asarray(east_valid, dtype=bool),
        "top_edge": np.asarray(top_edges, dtype=np.uint8),
        "top_gradient": np.asarray(top_gradients, dtype=np.int16),
        "right_edge": np.asarray(right_edges, dtype=np.uint8),
        "right_gradient": np.asarray(right_gradients, dtype=np.int16),
    }


def source_room_cost(database: dict[str, np.ndarray], target_x: int,
                     target_y: int, patch: int) -> np.ndarray:
    """Penalize source seeds that cannot propagate across the complete target."""
    source_y = database["index"][:, 1].astype(np.float32)
    source_x = database["index"][:, 2].astype(np.float32)
    height = database["extent"][:, 0].astype(np.float32)
    width = database["extent"][:, 1].astype(np.float32)
    base_x = source_x - target_x
    base_y = source_y - target_y
    overflow_x = np.maximum(-base_x, 0.0) + np.maximum(base_x + SIZE - width, 0.0)
    overflow_y = np.maximum(-base_y, 0.0) + np.maximum(base_y + SIZE - height, 0.0)
    return (overflow_x + overflow_y) / max(float(patch), 1.0)


def boundary_descriptor_cost(database: dict[str, np.ndarray], target_x: int,
                             target_y: int, patch: int,
                             north: TerrainSample, east: TerrainSample) -> np.ndarray:
    """Cheap global known-RGB context score used before candidate shortlisting."""
    count = len(database["index"])
    cost = np.zeros(count, dtype=np.float32)
    terms = 0
    band = min(TEXTURE_BOUNDARY_BAND, patch // 3)
    patch_axis = np.rint(np.linspace(0, patch - 1, TEXTURE_DESCRIPTOR_SIZE)).astype(int)
    context_axis = np.rint(np.linspace(0, band - 1, 4)).astype(int)
    if target_y == 0:
        known = north.rgb[-band:, target_x:target_x + patch]
        thumb = known[context_axis][:, patch_axis].astype(np.float32) / 255.0
        difference = database["north_context"].astype(np.float32) / 255.0 - thumb[None]
        term = np.mean(difference * difference, axis=(1, 2, 3)) / 0.04
        edge = north.rgb[-1, target_x:target_x + patch][patch_axis].astype(np.float32)
        gradient = (north.rgb[-1, target_x:target_x + patch][patch_axis].astype(np.float32) -
                    north.rgb[-2, target_x:target_x + patch][patch_axis].astype(np.float32))
        term += 2.0 * np.mean(((database["top_edge"].astype(np.float32) - edge[None]) /
                              255.0) ** 2, axis=(1, 2)) / 0.04
        term += np.mean(((database["top_gradient"].astype(np.float32) - gradient[None]) /
                         255.0) ** 2, axis=(1, 2)) / 0.02
        term[~database["north_valid"]] = 3.0
        cost += term
        terms += 1
    if target_x + patch == SIZE:
        known = east.rgb[target_y:target_y + patch, :band]
        thumb = known[patch_axis][:, context_axis].astype(np.float32) / 255.0
        difference = database["east_context"].astype(np.float32) / 255.0 - thumb[None]
        term = np.mean(difference * difference, axis=(1, 2, 3)) / 0.04
        edge = east.rgb[target_y:target_y + patch, 0][patch_axis].astype(np.float32)
        gradient = (east.rgb[target_y:target_y + patch, 1][patch_axis].astype(np.float32) -
                    east.rgb[target_y:target_y + patch, 0][patch_axis].astype(np.float32))
        term += 2.0 * np.mean(((database["right_edge"].astype(np.float32) - edge[None]) /
                              255.0) ** 2, axis=(1, 2)) / 0.04
        term += np.mean(((database["right_gradient"].astype(np.float32) - gradient[None]) /
                         255.0) ** 2, axis=(1, 2)) / 0.02
        term[~database["east_valid"]] = 3.0
        cost += term
        terms += 1
    return cost / max(terms, 1)


def texture_coherence_cost(database: dict[str, np.ndarray], target_x: int, target_y: int,
                           patch: int, donor_map: np.ndarray, source_x: np.ndarray,
                           source_y: np.ndarray, known: np.ndarray,
                           refinement_level: int) -> np.ndarray:
    """Prefer source origins propagated from already synthesized neighbours."""
    anchors: list[tuple[int, float, float]] = []
    samples = np.linspace(0, patch - 1, 5).round().astype(int)
    if target_x > 0:
        for local_y in samples:
            gy, gx = target_y + local_y, target_x - 1
            if known[gy, gx]:
                anchors.append((int(donor_map[gy, gx]),
                                float(source_x[gy, gx] + 1),
                                float(source_y[gy, gx] - local_y)))
    if target_y > 0:
        for local_x in samples:
            gy, gx = target_y - 1, target_x + local_x
            if known[gy, gx]:
                anchors.append((int(donor_map[gy, gx]),
                                float(source_x[gy, gx] - local_x),
                                float(source_y[gy, gx] + 1)))
    # At finer levels the previous level supplies a coarse initialization even
    # for the first raster patch, analogous to PatchMatch pyramid upsampling.
    if refinement_level and known[target_y + patch // 2, target_x + patch // 2]:
        gy, gx = target_y + patch // 2, target_x + patch // 2
        anchors.append((int(donor_map[gy, gx]),
                        float(source_x[gy, gx] - patch // 2),
                        float(source_y[gy, gx] - patch // 2)))
    if not anchors:
        return np.zeros(len(database["index"]), dtype=np.float32)

    donor_index = database["index"][:, 0]
    source_origin_y = database["index"][:, 1].astype(np.float32)
    source_origin_x = database["index"][:, 2].astype(np.float32)
    errors = []
    for anchor_donor, expected_x, expected_y in anchors:
        spatial = np.hypot(source_origin_x - expected_x,
                           source_origin_y - expected_y) / max(float(patch), 1.0)
        errors.append(np.minimum(spatial, 3.0) + 1.5 * (donor_index != anchor_donor))
    values = np.asarray(errors, dtype=np.float32)
    # A corner may legitimately join two source regions.  Reward agreement
    # with the best-supported half rather than averaging incompatible anchors.
    keep = max(1, (values.shape[0] + 1) // 2)
    return np.partition(values, keep - 1, axis=0)[:keep].mean(axis=0)


def rgb_patch_error(existing: np.ndarray, candidate: np.ndarray,
                    mask: np.ndarray) -> tuple[float, float, float]:
    """Compare colour, gradient phase and frequency over an existing overlap."""
    if not np.any(mask):
        return 0.0, 0.0, 0.0
    first = existing.astype(np.float32) / 255.0
    second = candidate.astype(np.float32) / 255.0
    colour_error = float(np.mean((first[mask] - second[mask]) ** 2) / 0.04)
    first_luma = first @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    second_luma = second @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    first_gy, first_gx = np.gradient(first_luma)
    second_gy, second_gx = np.gradient(second_luma)
    gradient_error = float(np.mean(((first_gx[mask] - second_gx[mask]) ** 2 +
                                    (first_gy[mask] - second_gy[mask]) ** 2)) / 0.02)
    first_frequency = float(np.sqrt(np.mean(first_gx[mask] ** 2 + first_gy[mask] ** 2)))
    second_frequency = float(np.sqrt(np.mean(second_gx[mask] ** 2 + second_gy[mask] ** 2)))
    scale_error = float(np.log((second_frequency + 1.0e-4) /
                               (first_frequency + 1.0e-4)) ** 2)
    return colour_error, gradient_error, scale_error


def boundary_condition_error(donor: TerrainSample, sx: int, sy: int, patch: int,
                             target_x: int, target_y: int,
                             north: TerrainSample, east: TerrainSample) -> float:
    """Compare real RGB context on both sides of each known target boundary."""
    band = min(TEXTURE_BOUNDARY_BAND, patch // 3)
    errors = []
    if target_y == 0:
        known_edge = north.rgb[-1, target_x:target_x + patch].astype(np.float32) / 255.0
        candidate_edge = donor.rgb[sy, sx:sx + patch].astype(np.float32) / 255.0
        known_gradient = (north.rgb[-1, target_x:target_x + patch].astype(np.float32) -
                          north.rgb[-2, target_x:target_x + patch].astype(np.float32)) / 255.0
        candidate_gradient = (donor.rgb[sy + 1, sx:sx + patch].astype(np.float32) -
                              donor.rgb[sy, sx:sx + patch].astype(np.float32)) / 255.0
        errors.append(2.0 * float(np.mean((known_edge - candidate_edge) ** 2) / 0.04))
        errors.append(float(np.mean((known_gradient - candidate_gradient) ** 2) / 0.02))
        if sy < band:
            errors.append(3.0)
        else:
            known = north.rgb[-band:, target_x:target_x + patch]
            context = donor.rgb[sy - band:sy, sx:sx + patch]
            errors.extend(rgb_patch_error(known, context,
                                          np.ones(known.shape[:2], dtype=bool))[:2])
    if target_x + patch == SIZE:
        known_edge = east.rgb[target_y:target_y + patch, 0].astype(np.float32) / 255.0
        candidate_edge = donor.rgb[sy:sy + patch, sx + patch - 1].astype(np.float32) / 255.0
        known_gradient = (east.rgb[target_y:target_y + patch, 1].astype(np.float32) -
                          east.rgb[target_y:target_y + patch, 0].astype(np.float32)) / 255.0
        candidate_gradient = (donor.rgb[sy:sy + patch, sx + patch - 1].astype(np.float32) -
                              donor.rgb[sy:sy + patch, sx + patch - 2].astype(np.float32)) / 255.0
        errors.append(2.0 * float(np.mean((known_edge - candidate_edge) ** 2) / 0.04))
        errors.append(float(np.mean((known_gradient - candidate_gradient) ** 2) / 0.02))
        if sx + patch + band > donor.rgb.shape[1]:
            errors.append(3.0)
        else:
            known = east.rgb[target_y:target_y + patch, :band]
            context = donor.rgb[sy:sy + patch, sx + patch:sx + patch + band]
            errors.extend(rgb_patch_error(known, context,
                                          np.ones(known.shape[:2], dtype=bool))[:2])
    return float(np.mean(errors)) if errors else 0.0


def rgb_quilt_take_mask(existing: np.ndarray, candidate: np.ndarray, known: np.ndarray,
                        overlap: int, has_left: bool, has_top: bool) -> np.ndarray:
    """Minimum-error cuts preserve donor high frequencies without feathering."""
    first = existing.astype(np.float32) / 255.0
    second = candidate.astype(np.float32) / 255.0
    colour_cost = np.mean((first - second) ** 2, axis=2)
    first_luma = first @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    second_luma = second @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    first_gy, first_gx = np.gradient(first_luma)
    second_gy, second_gx = np.gradient(second_luma)
    cost = colour_cost + 0.5 * ((first_gx - second_gx) ** 2 +
                               (first_gy - second_gy) ** 2)
    ownership = np.ones(known.shape, dtype=bool)
    coordinate = np.arange(overlap, dtype=np.float32)
    centre_bias = 1.0e-5 * ((coordinate - 0.5 * (overlap - 1)) /
                            max(overlap, 1)) ** 2
    if has_left:
        seam = minimum_vertical_seam(cost[:, :overlap] + centre_bias[None, :])
        ownership[:, :overlap] &= np.arange(overlap)[None, :] >= seam[:, None]
    if has_top:
        seam = minimum_vertical_seam(cost[:overlap, :].T + centre_bias[None, :])
        ownership[:overlap, :] &= np.arange(overlap)[:, None] >= seam[None, :]
    return ~known | ownership


def residual_quilt_take_mask(existing: np.ndarray, candidate: np.ndarray,
                             known: np.ndarray, overlap: int,
                             has_left: bool, has_top: bool) -> np.ndarray:
    """Minimum-cut ownership for a signed native-frequency residual.

    Residual patches must not be feathered together: independently phased
    rock/grass detail cancels under averaging and exposes the blurry macro
    colour field.  Cutting the signed bands themselves preserves native donor
    contrast while still hiding the join along the cheapest residual seam.
    """
    first = existing.astype(np.float32) / 255.0
    second = candidate.astype(np.float32) / 255.0
    colour_cost = np.mean((first - second) ** 2, axis=2)
    first_luma = first @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    second_luma = second @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    first_gy, first_gx = np.gradient(first_luma)
    second_gy, second_gx = np.gradient(second_luma)
    cost = colour_cost + 0.5 * ((first_gx - second_gx) ** 2 +
                               (first_gy - second_gy) ** 2)
    ownership = np.ones(known.shape, dtype=bool)
    coordinate = np.arange(overlap, dtype=np.float32)
    centre_bias = 1.0e-5 * ((coordinate - 0.5 * (overlap - 1)) /
                            max(overlap, 1)) ** 2
    if has_left:
        seam = minimum_vertical_seam(cost[:, :overlap] + centre_bias[None, :])
        ownership[:, :overlap] &= np.arange(overlap)[None, :] >= seam[:, None]
    if has_top:
        seam = minimum_vertical_seam(cost[:overlap, :].T + centre_bias[None, :])
        ownership[:overlap, :] &= np.arange(overlap)[:, None] >= seam[None, :]
    return ~known | ownership


def source_coordinate_visualization(donor_map: np.ndarray, source_x: np.ndarray,
                                    source_y: np.ndarray) -> np.ndarray:
    """Encode donor identity and source XY so discontinuities are conspicuous."""
    maximum_x = max(int(source_x.max()), 1)
    maximum_y = max(int(source_y.max()), 1)
    red = np.rint(np.clip(source_x, 0, maximum_x) / maximum_x * 255).astype(np.uint8)
    green = np.rint(np.clip(source_y, 0, maximum_y) / maximum_y * 255).astype(np.uint8)
    maximum_donor = max(int(donor_map.max()), 1)
    blue = np.rint(np.clip(donor_map, 0, maximum_donor) /
                   maximum_donor * 224.0 + 24.0).astype(np.uint8)
    return np.dstack((red, green, blue))


def sample_native_source_field(base: np.ndarray, donors: list[TerrainSample],
                               donor_map: np.ndarray, source_x: np.ndarray,
                               source_y: np.ndarray) -> np.ndarray:
    """Resolve a minimum-cut source-coordinate field to exact donor pixels."""
    native = base.astype(np.float32).copy()
    for donor_index, donor in enumerate(donors):
        mask = donor_map == donor_index
        if not np.any(mask):
            continue
        sy = np.clip(source_y[mask], 0, donor.rgb.shape[0] - 1)
        sx = np.clip(source_x[mask], 0, donor.rgb.shape[1] - 1)
        native[mask] = donor.rgb[sy, sx]
    return np.clip(native, 0, 255).astype(np.uint8)


def relayer_native_microtexture(
        base: np.ndarray, donors: list[TerrainSample],
        donor_map: np.ndarray, source_x: np.ndarray, source_y: np.ndarray,
        fine_field: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        macro_sigma: float = 12.0, detail_sigma: float = 2.0,
        mid_gain: float = 1.05, fine_gain: float = 0.95,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relayer exact native detail while keeping ownership scale per band.

    ``donor_map`` is the coarse 96 px ownership field.  It contributes only
    the middle band, so its large coherent regions cannot impose their colour.
    ``fine_field`` is the 24 px ownership field and contributes only detail
    above two pixels, so its many cuts cannot read as little photographic
    plates.  Broad colour remains entirely controlled by the macro synthesis.
    """
    mid_native = sample_native_source_field(base, donors, donor_map, source_x, source_y)
    if fine_field is None:
        fine_native = mid_native
    else:
        fine_native = sample_native_source_field(base, donors, *fine_field)
    blur = lambda image, sigma: gaussian_filter(
        image.astype(np.float32), (sigma, sigma, 0), mode="nearest")
    base_low = blur(base, macro_sigma)
    mid = blur(mid_native, detail_sigma) - blur(mid_native, macro_sigma)
    fine = fine_native.astype(np.float32) - blur(fine_native, detail_sigma)
    relayered = np.clip(base_low + mid_gain * mid + fine_gain * fine,
                        0, 255).astype(np.uint8)
    return relayered, mid_native, fine_native


def _nearest_isolate(donors: list[TerrainSample], kind: int) -> np.ndarray:
    """Legacy fallback: nearest-fill the densest donor when no pure source exists.

    Only reached when the whole-map exemplar search returns nothing for a
    material, so a small Voronoi-fill is preferable to crashing.
    """
    densest = int(np.argmax([np.mean(d.colour_class == kind) for d in donors]))
    donor = donors[densest]
    valid = donor.colour_class == kind
    if np.count_nonzero(valid) < 256:
        return donor.rgb.astype(np.float32)
    nearest = distance_transform_edt(~valid, return_distances=False,
                                     return_indices=True)
    return donor.rgb[tuple(nearest)].astype(np.float32)


def build_material_exemplar(sources: list[TerrainSample], kind: int,
                            rng: np.random.Generator, patch: int = 80,
                            overlap: int = 20) -> np.ndarray | None:
    """Min-cut quilt a full-tile pure-``kind`` image from high-purity source tiles.

    Every output pixel is a real sample from one intact source (Efros-Freeman
    minimum-error boundary cut, no feathering or Gaussian blend), so the native
    2-32 px and sub-2 px detail survives — the previous nearest-fill produced
    Voronoi plateaus that blurred into grey blobs, and a hann-blended quilt
    smeared away the very detail it was meant to inject.  Returns ``None`` when
    no source is pure enough, so the caller can fall back.
    """
    usable = [s for s in sources if float(np.mean(s.colour_class == kind)) >= 0.5]
    if not usable:
        return None
    step = patch - overlap
    origins = list(range(0, SIZE - patch + 1, step))
    if not origins or origins[-1] != SIZE - patch:
        origins.append(SIZE - patch)
    result = np.zeros((SIZE, SIZE, 3), np.float32)
    filled = np.zeros((SIZE, SIZE), dtype=bool)
    for y in origins:
        for x in origins:
            # Sample a few windows and keep the *best-textured* one that is still
            # genuinely this material.  Maximising purity alone would pick the
            # flattest windows (uniform grass classifies purer than textured
            # grass) and throw the detail away, so among windows that clear a
            # purity floor we take the one with the most 2-12 px energy.
            best_crop, best_energy, fallback, fallback_purity = None, -1.0, None, -1.0
            for _ in range(5):
                src = usable[int(rng.integers(0, len(usable)))]
                sy = int(rng.integers(0, SIZE - patch + 1))
                sx = int(rng.integers(0, SIZE - patch + 1))
                crop = src.rgb[sy:sy + patch, sx:sx + patch].astype(np.float32)
                purity = float(np.mean(src.colour_class[sy:sy + patch, sx:sx + patch] == kind))
                if purity > fallback_purity:
                    fallback_purity, fallback = purity, crop
                if purity >= 0.7:
                    band = (gaussian_filter(crop, (2.0, 2.0, 0), mode="nearest") -
                            gaussian_filter(crop, (12.0, 12.0, 0), mode="nearest"))
                    energy = float(np.sqrt(np.mean(band * band)))
                    if energy > best_energy:
                        best_energy, best_crop = energy, crop
            if best_crop is None:
                best_crop = fallback
            region = result[y:y + patch, x:x + patch]
            known = filled[y:y + patch, x:x + patch]
            take = rgb_quilt_take_mask(region, best_crop, known, overlap,
                                       has_left=x > 0, has_top=y > 0)
            region[take] = best_crop[take]
            known[:] = True
    return result


def relayer_material_microtexture(base: np.ndarray, labels: np.ndarray,
                                  donors: list[TerrainSample], seed: int = 1515,
                                  material_exemplars: dict[int, list[TerrainSample]] | None = None
                                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                             list[tuple[int, int, int]]]:
    """Render grid-free native bands from clean per-material style exemplars.

    Each material's mid (2-12 px), mesoscale (12-32 px) and fine (<2 px) bands
    are extracted from a high-purity exemplar min-cut quilted out of real
    whole-map ``kind`` tiles (``material_exemplars``), so no nearest-fill mush
    and no blend-blur enter the residual.  Soft material masks merge the
    residuals at the macro/micro SDF boundary; broad donor colour never enters
    this operation.
    """
    rng = np.random.default_rng(seed)
    material_exemplars = material_exemplars or {}
    # Clean-donor band-energy targets across the pool (used only to size the
    # additive top-up, so quiet tiles are not forced to over-sharpen).
    mid_energy, meso_energy, fine_energy = [], [], []
    for donor in donors:
        rgb = donor.rgb.astype(np.float32)
        smooth_2 = gaussian_filter(rgb, (2.0, 2.0, 0), mode="nearest")
        smooth_12 = gaussian_filter(rgb, (12.0, 12.0, 0), mode="nearest")
        smooth_32 = gaussian_filter(rgb, (32.0, 32.0, 0), mode="nearest")
        mid_energy.append(float(np.sqrt(np.mean((smooth_2 - smooth_12) ** 2))))
        meso_energy.append(float(np.sqrt(np.mean((smooth_12 - smooth_32) ** 2))))
        fine_energy.append(float(np.sqrt(np.mean((rgb - smooth_2) ** 2))))
    mid_energy = np.asarray(mid_energy, np.float32)
    meso_energy = np.asarray(meso_energy, np.float32)
    fine_energy = np.asarray(fine_energy, np.float32)
    mid = np.zeros_like(base, dtype=np.float32)
    meso = np.zeros_like(base, dtype=np.float32)
    fine = np.zeros_like(base, dtype=np.float32)
    selections: list[tuple[int, int, int]] = []
    material_colours: list[np.ndarray] = []
    for kind in range(len(CLASS_NAMES)):
        exemplar = build_material_exemplar(material_exemplars.get(kind, []), kind, rng)
        if exemplar is None:
            exemplar = _nearest_isolate(donors, kind)
        smooth_2 = gaussian_filter(exemplar, (2.0, 2.0, 0), mode="nearest")
        smooth_12 = gaussian_filter(exemplar, (12.0, 12.0, 0), mode="nearest")
        smooth_32 = gaussian_filter(exemplar, (32.0, 32.0, 0), mode="nearest")
        mid_band = smooth_2 - smooth_12
        meso_band = smooth_12 - smooth_32       # NEW: 12-32 px mesoscale
        fine_band = exemplar - smooth_2
        # Texture *structure* is luminance; hue belongs to the re-anchored macro
        # colour.  A textured snow window is often snow+grass, so its residual
        # carries green that reads as mould once added to white snow.  Keep each
        # material's band chroma only in proportion to that material's own colour
        # saturation: snow -> near-grey luminance detail, grass -> full colour.
        mean_colour = exemplar.reshape(-1, 3).mean(axis=0)
        saturation = float((mean_colour.max() - mean_colour.min()) /
                           max(float(mean_colour.max()), 1.0e-5))
        chroma_keep = float(np.clip((saturation - 0.05) * 3.5, 0.10, 1.0))

        def desaturate(band: np.ndarray) -> np.ndarray:
            luma = band.mean(axis=2, keepdims=True)
            return luma + chroma_keep * (band - luma)

        mid_band = desaturate(mid_band)
        meso_band = desaturate(meso_band)
        fine_band = desaturate(fine_band)
        weight = gaussian_filter((labels == kind).astype(np.float32), 2.0,
                                 mode="nearest")[..., None]
        mid += weight * mid_band
        meso += weight * meso_band
        fine += weight * fine_band
        n_sources = len(material_exemplars.get(kind, []))
        selections.append((kind, n_sources, n_sources))
        donor_means = [donor.rgb[donor.colour_class == kind].mean(axis=0)
                       for donor in donors
                       if np.count_nonzero(donor.colour_class == kind) >= 256]
        if not donor_means:
            donor_means = [donor.rgb.reshape(-1, 3).mean(axis=0) for donor in donors]
        material_colours.append(np.median(np.asarray(donor_means), axis=0))
    # Keep the multiscale quilt result intact.  The previous version reblurred
    # it to sigma 12 and rebuilt mid/fine from scratch, which threw away the
    # coarse/mid structure the coarse->fine passes had already earned (measured
    # ~37% mid-band loss).  Instead we re-anchor broad per-material colour and
    # top up only the residual energy overlap-blending eroded — additively, so
    # every band the quilt produced is written exactly once and preserved.
    base_f = base.astype(np.float32)
    broad = gaussian_filter(base_f, (12.0, 12.0, 0), mode="nearest")
    # Re-anchor only the broad colour of each SDF material.  Residual exemplars
    # are zero mean, so without this a grass-heavy local style family can tint
    # snow/rock green even though their geometry is correct.
    colour_grade = np.zeros_like(base_f)
    for kind, desired_colour in enumerate(material_colours):
        material = labels == kind
        current_colour = broad[material].mean(axis=0) if np.any(material) else desired_colour
        shift = np.clip(desired_colour - current_colour, -48.0, 48.0)
        weight = gaussian_filter(material.astype(np.float32), 3.0,
                                 mode="nearest")[..., None]
        colour_grade += weight * shift
    # Measure how far the assembled tile's mid/meso/fine energy sits below the
    # clean-donor population and add exactly that shortfall once from the crisp
    # per-material residual bands.  The 12-32 px mesoscale band is the character
    # the sigma-16 macro composition smooths away, so it is topped up here too.
    def band_topup(low_sigma: float, high_sigma: float | None,
                   target: np.ndarray) -> float:
        low = gaussian_filter(base_f, (low_sigma, low_sigma, 0), mode="nearest")
        band = base_f - low if high_sigma is None else \
            low - gaussian_filter(base_f, (high_sigma, high_sigma, 0), mode="nearest")
        rms = float(np.sqrt(np.mean(band * band)))
        return float(np.clip(np.median(target) / max(rms, 1.0e-5) - 1.0, 0.0, 1.5))

    mid_topup = band_topup(2.0, 12.0, mid_energy)
    meso_topup = band_topup(12.0, 32.0, meso_energy)
    fine_topup = band_topup(2.0, None, fine_energy)
    result = np.clip(base_f + 0.80 * colour_grade + mid_topup * mid +
                     meso_topup * meso + fine_topup * fine, 0, 255).astype(np.uint8)
    return result, np.clip(mid + 128, 0, 255).astype(np.uint8), \
        np.clip(fine + 128, 0, 255).astype(np.uint8), selections


def source_discontinuities(donor_map: np.ndarray, source_x: np.ndarray,
                           source_y: np.ndarray) -> np.ndarray:
    horizontal = ((donor_map[:, 1:] != donor_map[:, :-1]) |
                  (source_x[:, 1:] != source_x[:, :-1] + 1) |
                  (source_y[:, 1:] != source_y[:, :-1]))
    vertical = ((donor_map[1:, :] != donor_map[:-1, :]) |
                (source_x[1:, :] != source_x[:-1, :]) |
                (source_y[1:, :] != source_y[:-1, :] + 1))
    result = np.zeros((SIZE, SIZE), dtype=bool)
    result[:, 1:] |= horizontal
    result[1:, :] |= vertical
    return result


def source_polygon_labels(donor_map: np.ndarray, source_x: np.ndarray,
                          source_y: np.ndarray) -> tuple[np.ndarray, int]:
    """Label each maximal contiguous-source region (one min-cut polygon).

    Two adjacent target pixels belong to the same polygon when they were copied
    from the same donor under the same translation, i.e. the source coordinate
    steps by exactly +1 across the shared edge.  Connected components of that
    "no-cut" adjacency graph are the polygons the quilting cuts carved out.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    same_left = ((donor_map[:, 1:] == donor_map[:, :-1]) &
                 (source_x[:, 1:] == source_x[:, :-1] + 1) &
                 (source_y[:, 1:] == source_y[:, :-1]))
    same_up = ((donor_map[1:, :] == donor_map[:-1, :]) &
               (source_x[1:, :] == source_x[:-1, :]) &
               (source_y[1:, :] == source_y[:-1, :] + 1))
    index = np.arange(SIZE * SIZE).reshape(SIZE, SIZE)
    rows = np.concatenate((index[:, 1:][same_left], index[1:, :][same_up]))
    cols = np.concatenate((index[:, :-1][same_left], index[:-1, :][same_up]))
    graph = coo_matrix((np.ones(rows.size, dtype=bool), (rows, cols)),
                       shape=(SIZE * SIZE, SIZE * SIZE))
    count, labels = connected_components(graph, directed=False)
    return labels.reshape(SIZE, SIZE), count


def polygon_outline_mask(labels: np.ndarray) -> np.ndarray:
    """Pixels on a boundary between two different source polygons.

    Only the higher-index side of each boundary is marked, giving a 1 px line
    rather than the 2 px double edge marking both neighbours would produce.
    """
    outline = np.zeros(labels.shape, dtype=bool)
    outline[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    outline[1:, :] |= labels[1:, :] != labels[:-1, :]
    return outline


def source_polygon_visualization(rebuilt: np.ndarray, donor_map: np.ndarray,
                                 source_x: np.ndarray, source_y: np.ndarray
                                 ) -> tuple[np.ndarray, np.ndarray, int]:
    """Draw polygon outlines over the reconstruction and as flat-filled regions."""
    labels, count = source_polygon_labels(donor_map, source_x, source_y)
    outline = polygon_outline_mask(labels)
    overlaid = rebuilt.copy()
    overlaid[outline] = (255, 0, 255)
    colours = np.random.default_rng(count).integers(45, 235, (count, 3), dtype=np.uint8)
    filled = colours[labels]
    filled[outline] = 0
    return overlaid, filled, count


def _zoom(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return image
    return np.asarray(Image.fromarray(image, "RGB").resize(
        (image.shape[1] * factor, image.shape[0] * factor), Image.Resampling.NEAREST))


def seam_merge_workbench(rebuilt: np.ndarray, north: np.ndarray, east: np.ndarray,
                         donor_map: np.ndarray, source_x: np.ndarray,
                         source_y: np.ndarray, band: int = 32, zoom: int = 4,
                         alpha: float = 0.5) -> dict[str, np.ndarray]:
    """Magnified seam canvases for hand-tuning the target/donor merge.

    For each seam it returns two views, both drawn with the target's source
    polygon outlines so cut regions are visible next to the donor:

    * ``*_split``   – target band and donor band placed side by side across a
      marked seam, so the mismatch just inside the boundary is obvious at zoom.
    * ``*_overlay`` – the donor band as an opaque base with the target band and
      its polygon outlines composited on top at ``alpha``; a scratch layer for
      overlaying/annotating merges directly onto the donor imagery.
    """
    band = int(np.clip(band, 4, SIZE))
    outline = polygon_outline_mask(source_polygon_labels(donor_map, source_x, source_y)[0])
    seam_bar_v = np.full((SIZE, 2, 3), (255, 0, 0), dtype=np.uint8)
    seam_bar_h = np.full((2, SIZE, 3), (255, 0, 0), dtype=np.uint8)

    def overlay(donor_band: np.ndarray, target_band: np.ndarray,
                outline_band: np.ndarray) -> np.ndarray:
        merged = (alpha * target_band.astype(np.float32) +
                  (1.0 - alpha) * donor_band.astype(np.float32))
        merged = np.rint(merged).astype(np.uint8)
        merged[outline_band] = (255, 0, 255)
        return merged

    # East seam: target's right `band` columns meet the donor's left `band`.
    target_east = rebuilt[:, SIZE - band:].copy()
    target_east[outline[:, SIZE - band:]] = (255, 0, 255)
    east_split = np.concatenate((target_east, seam_bar_v, east[:, :band]), axis=1)
    east_overlay = overlay(east[:, :band], rebuilt[:, SIZE - band:],
                           outline[:, SIZE - band:])

    # North seam: the donor's bottom `band` rows sit above the target's top rows.
    target_north = rebuilt[:band].copy()
    target_north[outline[:band]] = (255, 0, 255)
    north_split = np.concatenate((north[SIZE - band:], seam_bar_h, target_north), axis=0)
    north_overlay = overlay(north[SIZE - band:], rebuilt[:band], outline[:band])

    return {
        "seam_workbench_east": _zoom(east_split, zoom),
        "seam_workbench_east_overlay": _zoom(east_overlay, zoom),
        "seam_workbench_north": _zoom(north_split, zoom),
        "seam_workbench_north_overlay": _zoom(north_overlay, zoom),
    }


def _edge_seam_cost(donor_band: np.ndarray, neighbour_band: np.ndarray) -> np.ndarray:
    """Per-pixel colour+gradient disagreement between donor and neighbour bands.

    Both inputs are shaped (depth, length, 3); the returned (depth, length) cost
    is what an Efros-Freeman min-cut minimises when deciding, per length index,
    how deep the neighbour must reach before switching back to donor content.
    """
    luma = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    first = donor_band.astype(np.float32) / 255.0
    second = neighbour_band.astype(np.float32) / 255.0
    colour = np.mean((first - second) ** 2, axis=2)
    first_gy, first_gx = np.gradient(first @ luma)
    second_gy, second_gx = np.gradient(second @ luma)
    return colour + 0.5 * ((first_gx - second_gx) ** 2 + (first_gy - second_gy) ** 2)


def snap_generated_edges(target_rgb: np.ndarray, north_rgb: np.ndarray,
                         east_rgb: np.ndarray,
                         band: int = TEXTURE_EDGE_SNAP_DEPTH
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Snap the synthesized target onto its neighbours with a hard min-cut seam.

    No averaging or feathering: every output pixel is a hard donor *or* hard
    neighbour sample, so donor high frequencies survive intact.  The boundary
    row/column is forced to the neighbour verbatim (seam RMSE 0), and an
    Efros-Freeman minimum-cost seam decides, per column (north) / per row
    (east), how few interior pixels must also follow the neighbour before
    switching back to real donor content.  The neighbour is extended inward as
    its own rows/columns mirrored past the seam (real terrain, not a blur), and
    the switch is placed where donor and neighbour already agree — where patch
    selection matched the edge well, that is depth 1 and nothing else changes.
    The shared corner can equal only one neighbour; east owns it here.
    """
    band = int(np.clip(band, 1, SIZE))
    result = target_rgb.copy()
    snapped = np.zeros((SIZE, SIZE), dtype=np.float32)

    def seam_depths(cost: np.ndarray) -> np.ndarray:
        # cost is (band, length); the transposed min-cut yields one switch depth
        # per length index, kept >=1 so the exact boundary line always snaps.
        seam = minimum_vertical_seam(np.ascontiguousarray(cost.T.astype(np.float32)))
        return np.clip(seam.astype(np.int32), 1, band)

    # North: neighbour's bottom rows mirrored downward past the seam.
    north_ext = np.stack([north_rgb[SIZE - 1 - i] for i in range(band)])
    depth_north = seam_depths(_edge_seam_cost(target_rgb[:band], north_ext))
    for column in range(SIZE):
        depth = int(depth_north[column])
        result[:depth, column] = north_ext[:depth, column]
        snapped[:depth, column] = 1.0

    # East: neighbour's left columns mirrored rightward past the seam.  Index k
    # is distance inward from the right edge, so target column SIZE-1-k pairs
    # with neighbour column k.
    east_ext = np.stack([east_rgb[:, k] for k in range(band)])
    donor_east = np.stack([target_rgb[:, SIZE - 1 - k] for k in range(band)])
    depth_east = seam_depths(_edge_seam_cost(donor_east, east_ext))
    for row in range(SIZE):
        depth = int(depth_east[row])
        for k in range(depth):
            result[row, SIZE - 1 - k] = east_ext[k, row]
            snapped[row, SIZE - 1 - k] = 1.0

    # The two mirrored bands overlap near the shared corner, each overwriting the
    # other's exact boundary line.  Re-lock both lines so the seams stay exact
    # regardless of interior snap depth; the single shared corner can equal only
    # one neighbour, so east keeps it and north wears the one-pixel residual.
    result[0, :] = north_rgb[SIZE - 1]
    result[:, SIZE - 1] = east_rgb[:, 0]
    snapped[0, :] = 1.0
    snapped[:, SIZE - 1] = 1.0
    return result, snapped


def distant_patch_repetition(rgb: np.ndarray, patch: int = 16,
                             stride: int = 8) -> tuple[float, float, float]:
    """Measure repeated high-frequency motifs away from their local neighbours."""
    unit = rgb.astype(np.float32) / 255.0
    luma = unit @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(luma, (patch, patch))[::stride, ::stride]
    height, width = windows.shape[:2]
    features = windows.reshape(-1, patch * patch)
    features -= features.mean(axis=1, keepdims=True)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1.0e-6)
    correlation = features @ features.T
    yy, xx = np.indices((height, width))
    positions = np.column_stack((yy.ravel(), xx.ravel()))
    local = np.max(np.abs(positions[:, None] - positions[None]), axis=2) < 3
    correlation[local] = -2.0
    best = correlation.max(axis=1)
    return float(best.mean()), float(np.quantile(best, 0.95)), float(np.mean(best > 0.95))


def grade_patch_to_context(candidate: np.ndarray, existing: np.ndarray,
                           overlap: np.ndarray, fallback_mean: np.ndarray | None,
                           strength: float = TEXTURE_GRADE_STRENGTH) -> np.ndarray:
    """Regrade a donor swatch so its exposure/colour matches what it joins.

    Reduces the hard light/dark block seams left by quilting swatches from
    differently-exposed source regions, without picking a different swatch.  A
    per-channel Reinhard transfer (linear light) maps the swatch's overlap
    statistics onto the already-placed neighbour statistics, regularised toward
    the original swatch by ``strength`` and clamped so texture contrast survives
    and a mixed-material overlap cannot force a wild recolour.  With too little
    overlap it falls back to nudging the mean toward the surrounding real tiles.
    """
    candidate_linear = srgb_to_linear(candidate)
    if int(np.count_nonzero(overlap)) >= TEXTURE_GRADE_MIN_OVERLAP:
        source = candidate_linear[overlap]
        destination = srgb_to_linear(existing)[overlap]
        source_mean = source.mean(axis=0)
        source_std = source.std(axis=0) + 1.0e-4
        scale = np.clip(destination.std(axis=0) / source_std, *TEXTURE_GRADE_STD_CLAMP)
        graded = (candidate_linear - source_mean) * scale + destination.mean(axis=0)
    elif fallback_mean is not None:
        source_mean = candidate_linear.reshape(-1, 3).mean(axis=0)
        graded = candidate_linear + (fallback_mean - source_mean)
        strength = min(strength, TEXTURE_GRADE_FALLBACK_STRENGTH)
    else:
        return candidate
    shift = np.clip(strength * (graded - candidate_linear),
                    -TEXTURE_GRADE_MAX_SHIFT, TEXTURE_GRADE_MAX_SHIFT)
    graded = np.clip(candidate_linear + shift, 0.0, 1.0)
    return np.rint(linear_to_srgb(graded) * 255.0).astype(np.uint8)


def native_frequency_band(source: np.ndarray, level: int,
                          candidate_index: int, phase_seed: int = 0) -> np.ndarray:
    """Extract one independently phased, zero-mean native-resolution band.

    Macro low colour is already present in ``existing``.  Each level gets an
    independent deterministic phase derived from its source identity, and no
    transform is shared between bands.  Only zero-mean residual energy moves.
    """
    phase_x = ((candidate_index * 17 + level * 11 + phase_seed * 5) % 13) - 6
    phase_y = ((candidate_index * 29 + level * 7 + phase_seed * 3) % 13) - 6
    shifted = image_shift(source.astype(np.float32), (phase_y, phase_x, 0),
                          order=1, mode="reflect", prefilter=False)
    if level == 0:
        band = gaussian_filter(shifted, (2.0, 2.0, 0)) - \
               gaussian_filter(shifted, (16.0, 16.0, 0))
        weight = 0.55
    elif level == 1:
        band = gaussian_filter(shifted, (1.2, 1.2, 0)) - \
               gaussian_filter(shifted, (6.0, 6.0, 0))
        weight = 0.45
    else:
        band = shifted - gaussian_filter(shifted, (2.0, 2.0, 0))
        weight = 0.95
    band -= band.mean(axis=(0, 1), keepdims=True)
    # Prevent one unusually contrasty donor from dominating an entire region.
    rms = float(np.sqrt(np.mean(band * band)))
    band *= min(1.0, 28.0 / max(rms, 1.0e-4))
    return weight * band


def native_frequency_candidate(existing: np.ndarray, source: np.ndarray,
                               level: int, candidate_index: int,
                               phase_seed: int = 0) -> np.ndarray:
    return np.clip(existing.astype(np.float32) +
                   native_frequency_band(source, level, candidate_index, phase_seed),
                   0, 255).astype(np.uint8)


def irregular_axis_origins(patch: int, stride: int,
                           rng: np.random.Generator) -> list[int]:
    """Cover one axis without locking placements to a fixed payload period."""
    last = SIZE - patch
    result = [0]
    while result[-1] < last:
        advance = int(rng.integers(max(1, round(stride * 0.72)),
                                   max(2, round(stride * 1.28)) + 1))
        following = min(last, result[-1] + advance)
        if following == result[-1]:
            break
        result.append(following)
    return result


def semantic_texture_quilt(target_labels: np.ndarray, target: TerrainSample,
                           donors: list[TerrainSample], north: TerrainSample,
                           east: TerrainSample, output: Path | None = None,
                           macro_low: np.ndarray | None = None,
                           material_exemplars: dict[int, list[TerrainSample]] | None = None,
                           uniform_material: int | None = None
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coarse-to-fine semantic synthesis or pass15 native multiband rendering.

    ``uniform_material`` marks a pure single-material tile: the coarse-to-fine
    *semantic* patch search is pointless when every patch is the same class, so
    it is skipped and the native detail comes straight from the material relayer
    on the macro-low base — that search is the dominant per-preset cost (~49s)
    for a uniform result it cannot improve.
    """
    from terrain_shape_synth import soft_choice
    from terrain_synth import srgb_to_lab
    generated = (macro_low.copy() if macro_low is not None else
                 np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
    known = (np.ones((SIZE, SIZE), dtype=bool) if macro_low is not None else
             np.zeros((SIZE, SIZE), dtype=bool))
    donor_map = np.full((SIZE, SIZE), -1, dtype=np.int16)
    source_x = np.full((SIZE, SIZE), -1, dtype=np.int16)
    source_y = np.full((SIZE, SIZE), -1, dtype=np.int16)
    source_usage = [np.zeros(source.rgb.shape[:2], dtype=np.int16) for source in donors]
    rng = np.random.default_rng(151500)
    # Exposure anchor for the first swatch of a region, which has no overlap yet:
    # keep the whole tile in the surrounding real tiles' colour ballpark.
    context_mean = srgb_to_linear(
        np.concatenate([north.rgb, east.rgb], axis=0)).reshape(-1, 3).mean(axis=0)
    texture_dir = output / "texture_synthesis" if output is not None else None
    if texture_dir is not None:
        texture_dir.mkdir(parents=True, exist_ok=True)
    frequency_source_fields: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    levels = [] if uniform_material is not None else list(enumerate(TEXTURE_LEVELS))
    if uniform_material is not None:
        print("  uniform tile: exemplar-quilt base (semantic search skipped)", flush=True)
        # Seed the base with a coherent min-cut quilt of the pure material rather
        # than the smooth macro-low: the relayer alone on macro-low leaves blotchy
        # blobs, whereas the exemplar carries real mid/meso structure the relayer
        # then only needs to colour-anchor.
        exemplar = build_material_exemplar(
            (material_exemplars or {}).get(uniform_material, []), uniform_material, rng)
        if exemplar is not None:
            generated = np.clip(exemplar, 0, 255).astype(np.uint8)
    for level, (patch, stride, source_stride) in levels:
        print(f"  texture level {level + 1}/{len(TEXTURE_LEVELS)}: "
              f"patch={patch}, stride={stride}", flush=True)
        database = build_texture_patch_database(donors, patch, source_stride)
        thumb_axis = np.rint(np.linspace(0, patch - 1, TEXTURE_DESCRIPTOR_SIZE)).astype(int)
        origins = (irregular_axis_origins(patch, stride, rng)
                   if macro_low is not None else patch_axis_origins(patch, stride))
        if macro_low is not None:
            # One signed native-resolution residual lives at each output pixel.
            # Do not overlap-average independently phased crops: that destroys
            # their actual texture and leaves only macro-scale blur.
            level_band = np.zeros((SIZE, SIZE, 3), np.float32)
            level_known = np.zeros((SIZE, SIZE), bool)
            selected_band_rms: list[float] = []
        for target_y in origins:
            for target_x in origins:
                target_patch_labels = target_labels[target_y:target_y + patch,
                                                    target_x:target_x + patch]
                target_thumb = target_patch_labels[np.ix_(thumb_axis, thumb_axis)]
                semantic_thumb = semantic_layout_error(database["thumbnail"], target_thumb)
                target_density = (np.bincount(target_patch_labels.ravel(), minlength=3) /
                                  target_patch_labels.size)
                density_error = np.mean((database["density"] - target_density[None]) ** 2,
                                        axis=1)
                target_height = target.height[target_y:target_y + patch,
                                              target_x:target_x + patch]
                target_slope = target.slope[target_y:target_y + patch,
                                            target_x:target_x + patch]
                terrain_descriptor = np.asarray((target_height.mean(), target_height.std(),
                                                 target_slope.mean()), dtype=np.float32)
                terrain_scale = np.asarray((100.0, 50.0, 0.20), dtype=np.float32)
                terrain_error = np.mean(((database["terrain"] - terrain_descriptor[None]) /
                                         terrain_scale[None]) ** 2, axis=1)
                target_angle, target_anisotropy = tensor_orientation(
                    target.gradient_x[target_y:target_y + patch, target_x:target_x + patch],
                    target.gradient_y[target_y:target_y + patch, target_x:target_x + patch])
                orientation_error = (2.0 * axial_difference(database["terrain_angle"][:, 0],
                                                            target_angle) / np.pi) ** 2
                orientation_error *= np.sqrt(np.maximum(
                    database["terrain_angle"][:, 1] * target_anisotropy, 0.0))
                coherence = texture_coherence_cost(
                    database, target_x, target_y, patch, donor_map, source_x, source_y,
                    known, level)
                room_error = source_room_cost(database, target_x, target_y, patch)
                boundary_descriptor = boundary_descriptor_cost(
                    database, target_x, target_y, patch, north, east)
                descriptor_cost = (8.0 * semantic_thumb + 1.5 * density_error +
                                   0.20 * terrain_error + 0.35 * orientation_error +
                                   1.0 * coherence + 1.0 * room_error +
                                   12.0 * boundary_descriptor)
                shortlist_count = min(TEXTURE_SHORTLIST, len(descriptor_cost))
                if level == 0:
                    shortlist = np.argpartition(
                        descriptor_cost, shortlist_count - 1)[:shortlist_count]
                else:
                    local_count = min(TEXTURE_LOCAL_SHORTLIST, len(descriptor_cost))
                    global_count = min(TEXTURE_SHORTLIST - local_count, len(descriptor_cost))
                    local_cost = coherence + 0.5 * room_error
                    local = np.argpartition(local_cost, local_count - 1)[:local_count]
                    global_candidates = np.argpartition(
                        descriptor_cost, global_count - 1)[:global_count]
                    shortlist = np.unique(np.concatenate((local, global_candidates)))

                existing = generated[target_y:target_y + patch, target_x:target_x + patch]
                existing_known = known[target_y:target_y + patch, target_x:target_x + patch]
                overlap = patch - stride
                overlap_mask = np.zeros((patch, patch), dtype=bool)
                if target_x > 0:
                    overlap_mask[:, :overlap] = True
                if target_y > 0:
                    overlap_mask[:overlap, :] = True
                overlap_mask &= existing_known
                if macro_low is not None:
                    existing_level_band = level_band[target_y:target_y + patch,
                                                     target_x:target_x + patch]
                    existing_level_known = level_known[target_y:target_y + patch,
                                                       target_x:target_x + patch]
                    residual_overlap_mask = overlap_mask & existing_level_known

                yy, xx = np.indices((patch, patch), dtype=np.int16)
                existing_donor = donor_map[target_y:target_y + patch,
                                           target_x:target_x + patch]
                existing_source_x = source_x[target_y:target_y + patch,
                                             target_x:target_x + patch]
                existing_source_y = source_y[target_y:target_y + patch,
                                             target_x:target_x + patch]
                level_coherence_weight = TEXTURE_COHERENCE_WEIGHT
                active_position = None
                active_lab = None
                if overlap_mask.any():
                    existing_ids = existing_donor[overlap_mask]
                    existing_ids = existing_ids[existing_ids >= 0]
                    if len(existing_ids):
                        active_id = int(np.bincount(existing_ids.astype(int)).argmax())
                        same = overlap_mask & (existing_donor == active_id)
                        active_position = np.array((
                            donors[active_id].x + float(existing_source_x[same].mean()) / SIZE,
                            donors[active_id].y + float(existing_source_y[same].mean()) / SIZE),
                            np.float32)
                    active_lab = srgb_to_lab(existing[overlap_mask]).mean(0)
                evaluated, evaluated_scores = [], []
                phase_seed = target_y * 509 + target_x
                for candidate_index in shortlist:
                    donor_index, sy, sx = database["index"][candidate_index]
                    donor = donors[int(donor_index)]
                    candidate_labels = donor.colour_class[sy:sy + patch, sx:sx + patch]
                    raw_candidate_rgb = donor.rgb[sy:sy + patch, sx:sx + patch]
                    if macro_low is not None:
                        candidate_band = native_frequency_band(
                            raw_candidate_rgb, level, int(candidate_index), phase_seed)
                        overlap_error, gradient_error, scale_error = rgb_patch_error(
                            existing_level_band, candidate_band, residual_overlap_mask)
                        candidate_rgb = None
                    else:
                        candidate_rgb = raw_candidate_rgb
                        overlap_error, gradient_error, scale_error = rgb_patch_error(
                            existing, candidate_rgb, overlap_mask)
                    label_error = float(semantic_layout_error(candidate_labels,
                                                              target_patch_labels))
                    boundary_error = boundary_condition_error(
                        donor, int(sx), int(sy), patch, target_x, target_y,
                        north, east)
                    usage = source_usage[int(donor_index)][sy:sy + patch,
                                                          sx:sx + patch]
                    same_mapping = (existing_known & (existing_donor == donor_index) &
                                    (existing_source_x == int(sx) + xx) &
                                    (existing_source_y == int(sy) + yy))
                    reuse_error = float(np.mean(np.minimum(
                        np.maximum(usage - same_mapping.astype(np.int16), 0), 2)))
                    if active_position is None:
                        source_style_error = 0.0
                    else:
                        source_distance = np.linalg.norm(
                            database["source_position"][candidate_index] - active_position)
                        saturated_source = 1.0 - np.exp(-source_distance / 2.0)
                        colour_distance = (np.linalg.norm(
                            database["mean_lab"][candidate_index] - active_lab) / 35.0
                            if active_lab is not None else 0.0)
                        material_distance = float(np.linalg.norm(
                            database["density"][candidate_index] - target_density))
                        source_style_error = (0.60 * saturated_source +
                                              0.25 * colour_distance +
                                              0.15 * material_distance)
                    total = (TEXTURE_LABEL_WEIGHT * label_error +
                             TEXTURE_OVERLAP_WEIGHT * overlap_error +
                             TEXTURE_GRADIENT_WEIGHT * gradient_error +
                             level_coherence_weight * float(coherence[candidate_index]) +
                             4.0 * float(room_error[candidate_index]) +
                             TEXTURE_SCALE_WEIGHT * scale_error +
                             TEXTURE_BOUNDARY_WEIGHT * boundary_error +
                             TEXTURE_REUSE_WEIGHT * reuse_error +
                             2.0 * source_style_error)
                    evaluated.append(int(candidate_index)); evaluated_scores.append(float(total))

                chosen_position = soft_choice(np.asarray(evaluated_scores), rng,
                                              temperature=0.55, finalists=16)
                selected = evaluated[chosen_position]

                donor_index, sy, sx = database["index"][selected]
                donor = donors[int(donor_index)]
                # Regrade the chosen swatch onto the exposure/colour of what it
                # joins before cutting, so the seam is a boundary between matched
                # tones rather than a hard light/dark block edge.  For a swatch
                # touching a tile edge the reference is augmented with the donor
                # across that seam (the one neighbour quilting never looks at),
                # so the boundary content is toned to the real donor as well.
                grade_reference = existing.copy()
                grade_overlap = overlap_mask.copy()
                edge = min(TEXTURE_BOUNDARY_BAND, patch)
                if target_x + patch == SIZE:
                    grade_reference[:, patch - edge:] = \
                        east.rgb[target_y:target_y + patch, :edge]
                    grade_overlap[:, patch - edge:] = True
                if target_y == 0:
                    grade_reference[:edge, :] = \
                        north.rgb[SIZE - edge:, target_x:target_x + patch]
                    grade_overlap[:edge, :] = True
                if macro_low is not None:
                    native_band = native_frequency_band(
                        donor.rgb[sy:sy + patch, sx:sx + patch], level,
                        selected, phase_seed)
                    take = residual_quilt_take_mask(
                        existing_level_band, native_band, existing_level_known,
                        overlap, target_x > 0, target_y > 0)
                else:
                    candidate_rgb = grade_patch_to_context(
                        donor.rgb[sy:sy + patch, sx:sx + patch], grade_reference,
                        grade_overlap, context_mean)
                    take = rgb_quilt_take_mask(existing, candidate_rgb, existing_known,
                                               overlap, target_x > 0, target_y > 0)
                # Keep the usage map synchronized with seam ownership.  Exact
                # propagation into the same target pixels is free; copying a
                # donor motif into a second target location is not.
                replaced = take & existing_known
                for old_donor in range(len(donors)):
                    old = replaced & (existing_donor == old_donor)
                    if np.any(old):
                        np.add.at(source_usage[old_donor],
                                  (existing_source_y[old], existing_source_x[old]), -1)
                np.add.at(source_usage[int(donor_index)],
                          (int(sy) + yy[take], int(sx) + xx[take]), 1)
                if macro_low is not None:
                    selected_band_rms.append(float(np.sqrt(np.mean(native_band * native_band))))
                    existing_level_band[take] = native_band[take]
                    existing_level_known[take] = True
                else:
                    generated[target_y:target_y + patch, target_x:target_x + patch][take] = \
                        candidate_rgb[take]
                known[target_y:target_y + patch, target_x:target_x + patch][take] = True
                donor_region = donor_map[target_y:target_y + patch, target_x:target_x + patch]
                source_x_region = source_x[target_y:target_y + patch, target_x:target_x + patch]
                source_y_region = source_y[target_y:target_y + patch, target_x:target_x + patch]
                donor_region[take] = int(donor_index)
                source_x_region[take] = int(sx) + xx[take]
                source_y_region[take] = int(sy) + yy[take]

        if macro_low is not None:
            current_rms = float(np.sqrt(np.mean(level_band * level_band)))
            desired_rms = float(np.median(selected_band_rms)) if selected_band_rms else current_rms
            native_gain = (1.15, 1.10, 1.00)[level]
            level_band *= native_gain * min(1.6, desired_rms / max(current_rms, 1.0e-5))
            generated = np.clip(generated.astype(np.float32) + level_band,
                                0, 255).astype(np.uint8)
            frequency_source_fields.append((donor_map.copy(), source_x.copy(), source_y.copy()))
        if texture_dir is not None:
            Image.fromarray(mark_target_seams(generated), "RGB").save(
                texture_dir / f"level_{level + 1}_{patch}px_rgb.png")
            Image.fromarray(mark_target_seams(source_coordinate_visualization(
                donor_map, source_x, source_y)), "RGB").save(
                texture_dir / f"level_{level + 1}_{patch}px_source_uv.png")

    if macro_low is not None:
        native_mid = native_fine = None
        if frequency_source_fields:  # the uniform fast path has none
            coarse_field = frequency_source_fields[0]
            fine_field = frequency_source_fields[-1]
            _, native_mid, native_fine = relayer_native_microtexture(
                generated, donors, *coarse_field, fine_field=fine_field)
        generated, material_mid, material_fine, material_sources = \
            relayer_material_microtexture(generated, target_labels, donors,
                                          material_exemplars=material_exemplars)
        if texture_dir is not None:
            if native_mid is not None:
                Image.fromarray(native_mid, "RGB").save(
                    texture_dir / "native_midfrequency_mosaic.png")
                Image.fromarray(native_fine, "RGB").save(
                    texture_dir / "native_highfrequency_mosaic.png")
            Image.fromarray(material_mid, "RGB").save(
                texture_dir / "material_midfrequency_residual.png")
            Image.fromarray(material_fine, "RGB").save(
                texture_dir / "material_highfrequency_residual.png")
            (texture_dir / "material_microtexture_sources.txt").write_text(
                "\n".join(f"material={kind} mid_donor={mid} fine_donor={fine}"
                          for kind, mid, fine in material_sources) + "\n",
                encoding="utf-8")
            Image.fromarray(mark_target_seams(generated), "RGB").save(
                texture_dir / "level_4_native_microtexture_relayer.png")

    return generated, donor_map, source_x, source_y


def material_luminance(labels: np.ndarray, atlas_path: Path,
                       seed: int = 1515) -> np.ndarray:
    """Legacy preview using a per-material *bank*, never one fixed atlas cell.

    The pass15 renderer no longer calls this function: native RGB bands come
    from cleaned whole-map style families.  Keeping the preview safe avoids the
    former fixed (7,10,16) cells and shared modular phase reappearing elsewhere.
    """
    atlas = np.asarray(Image.open(atlas_path).convert("RGB"))
    grid, cell, gutter = 5, 512, 8
    banks = (np.arange(0, 10), np.arange(5, 20), np.arange(15, 25))
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((SIZE, SIZE), dtype=np.int32)
    result = np.ones((SIZE, SIZE), dtype=np.float32)
    # Irregular 53 px regions break alignment with both 64 px shape support and
    # every RGB frequency level.  Each region gets an independent cell/phase.
    for y0 in range(0, SIZE, 53):
        for x0 in range(0, SIZE, 53):
            y1, x1 = min(SIZE, y0 + 61), min(SIZE, x0 + 61)
            for kind, bank in enumerate(banks):
                index = int(rng.choice(bank)); row, column = divmod(index, grid)
                phase_x, phase_y = rng.integers(0, cell, size=2)
                local_y, local_x = yy[y0:y1, x0:x1], xx[y0:y1, x0:x1]
                sx = gutter + ((local_x + phase_x) % cell)
                sy = gutter + ((local_y + phase_y) % cell)
                sample = srgb_to_linear(atlas[row * (cell + 2 * gutter) + sy,
                                              column * (cell + 2 * gutter) + sx])
                lum = sample @ np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)
                factor = np.clip(lum / max(float(lum.mean()), 1.0e-3), 0.72, 1.28)
                material = labels[y0:y1, x0:x1] == kind
                result[y0:y1, x0:x1][material] = factor[material]
    return result


def unmarked_layout(north: np.ndarray, west: np.ndarray, target: np.ndarray) -> np.ndarray:
    canvas = np.zeros((SIZE * 2, SIZE * 2, 3), dtype=np.uint8)
    canvas[:SIZE, :SIZE] = north
    canvas[SIZE:, :SIZE] = target
    canvas[SIZE:, SIZE:] = west
    return canvas


def layout(north: np.ndarray, west: np.ndarray, target: np.ndarray) -> np.ndarray:
    return mark_tile_seams(unmarked_layout(north, west, target))


def mark_tile_seams(canvas: np.ndarray) -> np.ndarray:
    """Draw the level-5 tile boundaries on a multi-tile debug image."""
    result = canvas.copy()
    if result.shape[0] > SIZE:
        result[SIZE, :, :3] = SEAM_GUIDE_COLOR
    if result.shape[1] > SIZE:
        result[:, SIZE, :3] = SEAM_GUIDE_COLOR
    return result


def mark_target_seams(target_image: np.ndarray) -> np.ndarray:
    """Mark the target's known north and east boundaries in a cropped view."""
    result = target_image.copy()
    result[0, :, :3] = SEAM_GUIDE_COLOR
    result[:, -1, :3] = SEAM_GUIDE_COLOR
    return result


def save_titled_strip(path: Path, panels: list[tuple[str, np.ndarray]],
                      show_seams: bool = True) -> None:
    """Save comparable target-space panels with identical seam overlays."""
    title_height = 30
    canvas = Image.new("RGB", (SIZE * len(panels), SIZE + title_height), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        x = index * SIZE
        draw.text((x + 5, 8), title, fill="white")
        displayed = mark_target_seams(panel) if show_seams else panel
        canvas.paste(Image.fromarray(displayed, "RGB"), (x, title_height))
    canvas.save(path)


def write_audit_outputs(output: Path, target: TerrainSample, north: TerrainSample,
                        east: TerrainSample, refined: bool) -> None:
    """Write held-out reference and inspectable synthesis-stage comparisons.

    Target RGB is classified only after synthesis has completed.  It is an
    evaluation image and can never influence the generated labels.
    """
    reference = colour_labels(target.rgb)
    Image.fromarray(target.rgb, "RGB").save(
        output / f"raw_target_{target.x}_{target.y}.png")
    Image.fromarray(CLASS_COLORS[reference], "RGB").save(
        output / f"reference_target_classes_{target.x}_{target.y}.png")

    polygon_dir = output / "shape_polygons"
    quilted = np.asarray(Image.open(polygon_dir / "01_quilted_shapes_combined.png").convert("RGB"))
    if refined:
        macro = np.asarray(Image.open(
            polygon_dir / "02_coherent_macro_shapes_combined.png").convert("RGB"))
        final = np.asarray(Image.open(
            polygon_dir / "03_final_shapes_combined.png").convert("RGB"))
        guide = np.asarray(Image.open(
            polygon_dir / "00_edge_continuation_guides.png").convert("RGB"))
        save_titled_strip(polygon_dir / "thinking_stages.png", [
            ("edge continuation field", guide),
            ("donor quilt", quilted),
            ("coherent shapes", macro),
            ("final boundary detail", final),
        ])
        save_titled_strip(output / "comparison_overview.png", [
            (f"raw target {target.x}/{target.y}", target.rgb),
            ("held-out reference classes", CLASS_COLORS[reference]),
            ("generated macro shapes", macro),
            ("generated final classes", final),
        ])

    reconstruction_path = output / "texture_reconstruction_target.png"
    if reconstruction_path.exists():
        rebuilt = np.asarray(Image.open(reconstruction_path).convert("RGB"))
        rebuilt_classes = colour_labels(rebuilt)
        texture_panels = [
            (f"held-out raw target {target.x}/{target.y}", target.rgb),
            ("semantic target", CLASS_COLORS[reference]),
            ("coherent RGB reconstruction", rebuilt),
            ("reclassified reconstruction", CLASS_COLORS[rebuilt_classes]),
        ]
        save_titled_strip(output / "texture_comparison.png", texture_panels)
        save_titled_strip(output / "texture_comparison_with_seams.png", texture_panels)
        save_titled_strip(output / "texture_comparison_without_seams.png", texture_panels,
                          show_seams=False)
        unblended_path = output / "texture_reconstruction_target_unblended.png"
        if unblended_path.exists():
            unblended = np.asarray(Image.open(unblended_path).convert("RGB"))
            blend_mask = np.asarray(Image.open(output / "texture_edge_blend_mask.png").convert("RGB"))
            blend_panels = [
                ("held-out raw target", target.rgb),
                ("selected patches before blend", unblended),
                ("target-side edge blend", rebuilt),
                ("blend strength", blend_mask),
            ]
            save_titled_strip(output / "texture_edge_blend_comparison_with_seams.png",
                              blend_panels)
            save_titled_strip(output / "texture_edge_blend_comparison_without_seams.png",
                              blend_panels, show_seams=False)

        def seam_metrics(rgb: np.ndarray) -> tuple[float, float, float, float]:
            unit = rgb.astype(np.float32) / 255.0
            north_unit = north.rgb.astype(np.float32) / 255.0
            east_unit = east.rgb.astype(np.float32) / 255.0
            north_cross = unit[0] - north_unit[-1]
            east_cross = east_unit[:, 0] - unit[:, -1]
            north_gradient = north_unit[-1] - north_unit[-2]
            east_gradient = east_unit[:, 1] - east_unit[:, 0]
            return (float(np.sqrt(np.mean(north_cross * north_cross))),
                    float(np.sqrt(np.mean(east_cross * east_cross))),
                    float(np.sqrt(np.mean((north_cross - north_gradient) ** 2))),
                    float(np.sqrt(np.mean((east_cross - east_gradient) ** 2))))

        actual = seam_metrics(target.rgb)
        generated = seam_metrics(rebuilt)
        with (output / "texture_synthesis_metrics.txt").open("a", encoding="utf-8") as stream:
            for prefix, values in (("heldout", actual), ("generated", generated)):
                stream.write(f"{prefix}_north_seam_rgb_rmse={values[0]:.6f}\n")
                stream.write(f"{prefix}_east_seam_rgb_rmse={values[1]:.6f}\n")
                stream.write(f"{prefix}_north_gradient_rmse={values[2]:.6f}\n")
                stream.write(f"{prefix}_east_gradient_rmse={values[3]:.6f}\n")

    (polygon_dir / "README.md").write_text(
        "# Shape decision audit\n\n"
        "All panels show categorical polygons only; they contain no generated surface texture. "
        "Red lines mark the target's north and east seams.\n\n"
        "- `00_edge_continuation_guides.png`: the generic shape/occupancy continuation predicted from both donors; black is outside the guide's decaying 48 px reach.\n"
        "- `00_edge_constraint_strength.png`: greyscale strength of that soft guide.\n"
        "- `01_quilted_shapes_*`: intact donor fragments selected by terrain and edge context.\n"
        "- `02_coherent_macro_shapes_*`: connected macro regions after local donor-shape matching.\n"
        "- `03_final_shapes_*`: macro regions after donor-conditioned boundary detail is restored.\n"
        "- `thinking_stages.png`: the four decision stages side by side.\n",
        encoding="utf-8")


def heightmap_with_scale(target: TerrainSample, east: TerrainSample, north: TerrainSample) -> Image.Image:
    """Raw DEMs in one shared metres scale; blank quadrant has no terrain tile."""
    minimum = min(float(sample.height.min()) for sample in (target, east, north))
    maximum = max(float(sample.height.max()) for sample in (target, east, north))
    span = max(maximum - minimum, 1.0)

    def greyscale(sample: TerrainSample) -> np.ndarray:
        value = np.clip((sample.height - minimum) / span * 255.0, 0, 255).astype(np.uint8)
        return np.dstack((value, value, value))

    terrain = Image.fromarray(layout(greyscale(north), greyscale(east), greyscale(target)), "RGB")
    scale_width = 112
    output = Image.new("RGB", (terrain.width + scale_width, terrain.height), "black")
    output.paste(terrain, (0, 0))
    draw = ImageDraw.Draw(output)
    bar_x0, bar_x1 = terrain.width + 20, terrain.width + 44
    for row in range(terrain.height):
        value = 255 - round(255 * row / max(terrain.height - 1, 1))
        draw.line((bar_x0, row, bar_x1, row), fill=(value, value, value))
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = round((1.0 - fraction) * (terrain.height - 1))
        metres = minimum + fraction * span
        draw.line((bar_x1, y, bar_x1 + 5, y), fill="white")
        draw.text((bar_x1 + 9, max(y - 5, 0)), f"{metres:.0f} m", fill="white")
    draw.text((terrain.width + 12, 8), "elevation", fill="white")
    return output


def write_texture_outputs(output: Path, target: TerrainSample, east: TerrainSample,
                          north: TerrainSample, target_labels: np.ndarray,
                          texture_sources: list[TerrainSample],
                          macro_low: np.ndarray | None = None,
                          snap_edges: bool = True,
                          material_exemplars: dict[int, list[TerrainSample]] | None = None,
                          uniform_material: int | None = None) -> None:
    """Run and export only the semantic-guided RGB/source-UV synthesis.

    ``snap_edges`` copies the real north/east neighbour imagery a few pixels
    into the top/right of the tile so a held-out gap joins its true neighbours
    seamlessly.  That is correct for gap-filling but wrong for pure-material
    showcase tiles, whose neighbours carry an unrelated material: pass
    ``snap_edges=False`` to ship the untouched synthesis instead.

    ``material_exemplars`` supplies whole-map high-purity source tiles per
    material for the native-band relayer (see ``load_material_exemplar_sources``).
    """
    print("Synthesizing coherent RGB source-coordinate field…", flush=True)
    unblended, donor_map, source_x, source_y = semantic_texture_quilt(
        target_labels, target, texture_sources, north, east, output, macro_low,
        material_exemplars=material_exemplars, uniform_material=uniform_material)
    if snap_edges:
        rebuilt, blend_mask = snap_generated_edges(unblended, north.rgb, east.rgb)
    else:
        rebuilt = unblended.copy()
        blend_mask = np.zeros((SIZE, SIZE), dtype=np.float32)
    Image.fromarray(unblended, "RGB").save(
        output / "texture_reconstruction_target_unblended.png")
    Image.fromarray(rebuilt, "RGB").save(output / "texture_reconstruction_target.png")
    Image.fromarray(rebuilt, "RGB").save(
        output / "texture_reconstruction_target_without_seams.png")
    Image.fromarray(mark_target_seams(rebuilt), "RGB").save(
        output / "texture_reconstruction_target_with_seams.png")
    unblended_full = unmarked_layout(north.rgb, east.rgb, unblended)
    Image.fromarray(unblended_full, "RGB").save(
        output / "procedural_reconstruction_unblended_without_seams.png")
    Image.fromarray(mark_tile_seams(unblended_full), "RGB").save(
        output / "procedural_reconstruction_unblended_with_seams.png")
    blend_mask_image = np.rint(blend_mask * 255.0).astype(np.uint8)
    Image.fromarray(blend_mask_image, "L").save(output / "texture_edge_blend_mask.png")
    full_clean = unmarked_layout(north.rgb, east.rgb, rebuilt)
    full_marked = mark_tile_seams(full_clean)
    Image.fromarray(full_marked, "RGB").save(output / "procedural_reconstruction.png")
    Image.fromarray(full_clean, "RGB").save(
        output / "procedural_reconstruction_without_seams.png")
    Image.fromarray(full_marked, "RGB").save(
        output / "procedural_reconstruction_with_seams.png")
    uv_visualization = source_coordinate_visualization(donor_map, source_x, source_y)
    Image.fromarray(mark_target_seams(uv_visualization), "RGB").save(
        output / "texture_source_coordinates.png")
    Image.fromarray(uv_visualization, "RGB").save(
        output / "texture_source_coordinates_without_seams.png")
    Image.fromarray(mark_target_seams(uv_visualization), "RGB").save(
        output / "texture_source_coordinates_with_seams.png")
    discontinuities = source_discontinuities(donor_map, source_x, source_y)
    discontinuity_rgb = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    discontinuity_rgb[discontinuities] = (255, 40, 40)
    Image.fromarray(mark_target_seams(discontinuity_rgb), "RGB").save(
        output / "texture_source_discontinuities.png")
    Image.fromarray(discontinuity_rgb, "RGB").save(
        output / "texture_source_discontinuities_without_seams.png")
    Image.fromarray(mark_target_seams(discontinuity_rgb), "RGB").save(
        output / "texture_source_discontinuities_with_seams.png")
    polygon_overlay, polygon_filled, polygon_count = source_polygon_visualization(
        rebuilt, donor_map, source_x, source_y)
    Image.fromarray(mark_target_seams(polygon_overlay), "RGB").save(
        output / "texture_source_polygons.png")
    Image.fromarray(polygon_overlay, "RGB").save(
        output / "texture_source_polygons_without_seams.png")
    Image.fromarray(polygon_filled, "RGB").save(
        output / "texture_source_polygons_filled.png")
    polygon_mosaic = unmarked_layout(north.rgb, east.rgb, polygon_overlay)
    Image.fromarray(mark_tile_seams(polygon_mosaic), "RGB").save(
        output / "procedural_reconstruction_polygons.png")
    Image.fromarray(polygon_mosaic, "RGB").save(
        output / "procedural_reconstruction_polygons_without_seams.png")
    print(f"  source polygons (cut regions): {polygon_count}", flush=True)
    workbench = seam_merge_workbench(rebuilt, north.rgb, east.rgb,
                                     donor_map, source_x, source_y)
    for name, canvas in workbench.items():
        Image.fromarray(canvas, "RGB").save(output / f"{name}.png")
    np.savez_compressed(output / "texture_source_field.npz", donor=donor_map,
                        source_x=source_x, source_y=source_y)
    reconstructed_labels = colour_labels(rebuilt)
    Image.fromarray(mark_target_seams(CLASS_COLORS[reconstructed_labels]), "RGB").save(
        output / "texture_reconstruction_classes.png")
    coherent_fraction = 1.0 - float(np.count_nonzero(discontinuities)) / discontinuities.size
    semantic_agreement = float(np.mean(reconstructed_labels == target_labels))
    coordinates = np.stack((donor_map, source_y, source_x), axis=2).reshape(-1, 3)
    _unique, coordinate_counts = np.unique(coordinates, axis=0, return_counts=True)
    unique_coordinate_fraction = float(len(coordinate_counts) / len(coordinates))
    duplicated_target_fraction = float(
        coordinate_counts[coordinate_counts > 1].sum() / len(coordinates))
    repetition_mean, repetition_p95, repetition_fraction = distant_patch_repetition(unblended)
    final_repetition = distant_patch_repetition(rebuilt)
    unblended_labels = colour_labels(unblended)
    unblended_semantic_agreement = float(np.mean(unblended_labels == target_labels))
    class_ious = []
    for kind in range(3):
        intersection = np.count_nonzero((reconstructed_labels == kind) & (target_labels == kind))
        union = np.count_nonzero((reconstructed_labels == kind) | (target_labels == kind))
        class_ious.append(float(intersection / max(union, 1)))
    (output / "texture_synthesis_metrics.txt").write_text(
        f"source_coordinate_coherent_fraction={coherent_fraction:.6f}\n"
        f"unique_source_coordinate_fraction={unique_coordinate_fraction:.6f}\n"
        f"target_pixels_using_duplicated_source={duplicated_target_fraction:.6f}\n"
        f"distant_patch_max_correlation_mean={repetition_mean:.6f}\n"
        f"distant_patch_max_correlation_p95={repetition_p95:.6f}\n"
        f"distant_patch_fraction_above_0.95={repetition_fraction:.6f}\n"
        f"blended_distant_patch_max_correlation_mean={final_repetition[0]:.6f}\n"
        f"blended_distant_patch_max_correlation_p95={final_repetition[1]:.6f}\n"
        f"blended_distant_patch_fraction_above_0.95={final_repetition[2]:.6f}\n"
        f"unblended_semantic_agreement={unblended_semantic_agreement:.6f}\n"
        f"reconstructed_semantic_agreement={semantic_agreement:.6f}\n"
        + "".join(f"{name}_iou={value:.6f}\n"
                  for name, value in zip(CLASS_NAMES, class_ious)),
        encoding="utf-8")
    print(f"  coherent source-coordinate pixels: {coherent_fraction:.1%}; "
          f"duplicated-source pixels: {duplicated_target_fraction:.1%}; "
          f"reclassified semantic agreement: {semantic_agreement:.1%}", flush=True)


def write_outputs(output: Path, target: TerrainSample, west: TerrainSample, north: TerrainSample,
                  target_labels: np.ndarray, probabilities: np.ndarray, atlas_path: Path,
                  texture_sources: list[TerrainSample],
                  macro_low: np.ndarray | None = None,
                  snap_edges: bool = True,
                  material_exemplars: dict[int, list[TerrainSample]] | None = None,
                  uniform_material: int | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    print(f"Writing output images to {output}…", flush=True)
    # Raw source overview intentionally includes the red-marked target for audit.
    Image.fromarray(layout(north.rgb, west.rgb, target.rgb), "RGB").save(output / "raw_three_tiles.png")
    heightmap_with_scale(target, west, north).save(output / "raw_heightmaps_metres.png")
    Image.fromarray(layout(flat_classes(north.colour_class), flat_classes(west.colour_class),
                           flat_classes(target_labels)), "RGB").save(output / "height_material_classes.png")
    Image.fromarray(flat_classes(target_labels), "RGB").save(output / "constrained_patch_classes.png")
    probability_colours = (probabilities @ CLASS_COLORS.astype(np.float32)).round().astype(np.uint8)
    Image.fromarray(probability_colours, "RGB").save(output / "height_material_probabilities.png")

    # The atlas argument remains accepted for CLI compatibility but is
    # intentionally unused: independently stamping per-material detail destroys
    # texture phase.
    _ = atlas_path
    write_texture_outputs(output, target, west, north, target_labels, texture_sources,
                          macro_low, snap_edges=snap_edges,
                          material_exemplars=material_exemplars,
                          uniform_material=uniform_material)

    summary = output / "README.md"
    summary.write_text(
        "# Procedural gap reconstruction — pass15\n\n"
        "Layout: north donor at upper-left, target at lower-left, east donor at lower-right. "
        "The empty upper-right quadrant is intentionally absent. Red lines mark tile seams in every multi-tile debug image.\n\n"
        "- `raw_three_tiles.png`: source imagery; target visibly retains its red outline for audit.\n"
        "- `raw_heightmaps_metres.png`: raw per-tile DEMs on one shared greyscale elevation range, with metres legend.\n"
        "- `height_material_classes.png`: donor RGB labels and final constrained target labels. "
        "White=snow, grey=rock, green=grass.\n"
        "- `constrained_patch_classes.png`: final macro/micro SDF material field.\n"
        "- `shape_polygons/01_*`, `02_*`, `03_*`: probability seed, transformed 96-224 px macro shapes, and 12-64 px transition-weighted micro breakup.\n"
        "- `passes/pass_XX_*.png`: three-way donor/target categorical previews for the same three shape stages.\n"
        "- `height_material_probabilities.png`: target expectation from the shared-metre elevation/slope density model.\n"
        "- `procedural_reconstruction.png`: target RGB generated without reading target RGB. Transformed donor imagery "
        "contributes only broad macro colour. Independent material-balanced clean donors contribute native 2-12 px and "
        "sub-2 px residual bands; broad material colour is re-anchored separately.\n"
        "- `texture_source_coordinates.png`: source UV visualization; smooth colour ramps are coherent copied regions and "
        "abrupt colour jumps are donor switches. Blue distinguishes source mosaics. Paired seam/no-seam variants are included.\n"
        "- `texture_source_discontinuities.png`: every non-unit source-coordinate transition in red.\n"
        "- `texture_source_field.npz`: exact donor index and integer source X/Y fields for quantitative auditing.\n"
        "- `texture_synthesis/level_*`: retrieval states plus the final native microtexture relayer and isolated residual diagnostics.\n"
        "- `texture_synthesis_metrics.txt`: coordinate coherence and semantic agreement diagnostics.\n",
        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "alps-data" / "trn-alps-16km")
    parser.add_argument("--target", type=int, nargs=2, default=(23, 8), metavar=("X", "Y"))
    parser.add_argument("--west-donor", type=int, nargs=2, default=(24, 8), metavar=("X", "Y"),
                        help="first clean donor (named for compatibility; may be any neighbour)")
    parser.add_argument("--north-donor", type=int, nargs=2, default=(23, 7), metavar=("X", "Y"),
                        help="second clean donor (named for compatibility; may be any neighbour)")
    parser.add_argument("--atlas", type=Path, default=ROOT / "textures/runtime/terrain_micro_albedo.png")
    parser.add_argument("--output", type=Path, default=Path("/tmp/je5Z7y"))
    parser.add_argument("--quick", action="store_true",
                        help="use 160 full-map tiles for faster macro/micro iteration")
    parser.add_argument("--texture-only", action="store_true",
                        help="reuse output/constrained_patch_classes.png and rerun only RGB synthesis")
    parser.add_argument("--legacy-shapes", action="store_true",
                        help="use the pre-pass15 two-donor categorical patch pipeline")
    parser.add_argument("--shape-tile-limit", type=int,
                        help="development-only cap on streamed whole-map shape tiles")
    args = parser.parse_args()
    print("Loading target and donor terrain…", flush=True)
    target = load_sample(args.dataset, *args.target)
    west = load_sample(args.dataset, *args.west_donor)
    north = load_sample(args.dataset, *args.north_donor)
    for donor in (west, north):
        donor.colour_class = colour_labels(donor.rgb)
    if args.texture_only:
        class_path = args.output / "constrained_patch_classes.png"
        if not class_path.exists():
            parser.error(f"--texture-only requires {class_path}")
        class_rgb = np.asarray(Image.open(class_path).convert("RGB"))
        difference = class_rgb[:, :, None, :].astype(np.int32) - \
            CLASS_COLORS[None, None, :, :].astype(np.int32)
        labels = np.sum(difference * difference, axis=3).argmin(axis=2).astype(np.uint8)
        print("Loading target-excluding texture-source mosaics…", flush=True)
        preferred = None
        shape_metrics = args.output / "macro_micro_metrics.json"
        macro_low_path = args.output / "macro_low_frequency_filled.png"
        if shape_metrics.exists():
            payload = json.loads(shape_metrics.read_text(encoding="utf-8"))
            preferred = []
            for values in payload.get("texture_source_coordinates", []):
                source = tuple(int(value) for value in values)
                if source != (target.x, target.y) and source not in preferred:
                    preferred.append(source)
                if len(preferred) >= 14:
                    break
            for values in payload.get("selected_style_sources", []):
                source = tuple(int(value) for value in values)
                if source != (target.x, target.y) and source not in preferred:
                    preferred.append(source)
                if len(preferred) >= 14:
                    break
            for placement in payload.get("placements", []):
                source = tuple(int(value) for value in placement["source"])
                if source != (target.x, target.y) and source not in preferred:
                    preferred.append(source)
                if len(preferred) >= 14:
                    break
            preferred = preferred[:14]
        texture_sources = load_texture_sources(args.dataset, target.x, target.y, preferred)
        macro_low = (np.asarray(Image.open(macro_low_path).convert("RGB"))
                     if macro_low_path.exists() else None)
        write_texture_outputs(args.output, target, west, north, labels, texture_sources,
                              macro_low)
        refined = (args.output / "shape_polygons" /
                   "03_final_shapes_combined.png").exists()
        write_audit_outputs(args.output, target, north, west, refined=refined)
        print(f"wrote {args.output}")
        return
    shape_library = None
    shape_result = None
    if args.legacy_shapes:
        print("Learning shared elevation/slope material probabilities…", flush=True)
        probabilities = material_probabilities(target, [west, north])
        labels = constrained_patch_labels(target, north, west, probabilities, args.output,
                                          refine=not args.quick)
    else:
        from terrain_shape_synth import WholeMapShapeLibrary, synthesize_macro_micro
        block = {tuple(args.target)}
        limit = args.shape_tile_limit if args.shape_tile_limit is not None else (160 if args.quick else None)
        shape_library = WholeMapShapeLibrary(args.dataset, colour_labels,
                                             exclude=block, max_tiles=limit)
        print("Learning whole-map material mass field…", flush=True)
        whole_map_probabilities = shape_library.predict_probabilities(target.height, target.slope)
        boundary_probabilities = material_probabilities(target, [west, north])
        probabilities = 0.72 * whole_map_probabilities + 0.28 * boundary_probabilities
        probabilities /= np.maximum(probabilities.sum(axis=2, keepdims=True), 1.0e-6)
        print("Synthesizing transformed macro and micro SDF shapes…", flush=True)
        shape_result = synthesize_macro_micro(
            shape_library, target.height, target.slope, probabilities,
            north.colour_class, west.colour_class)
        labels = shape_result.labels
        write_macro_micro_diagnostics(args.output, shape_result, north, west)
    print("Loading target-excluding texture-source mosaics…", flush=True)
    preferred = (shape_library.texture_source_coordinates((target.x, target.y))
                 if shape_library is not None else None)
    if shape_result is not None and preferred is not None:
        metrics_path = args.output / "macro_micro_metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["texture_source_coordinates"] = [
            [int(value) for value in source] for source in preferred]
        metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    texture_sources = load_texture_sources(args.dataset, target.x, target.y, preferred)
    macro_low = None
    if shape_result is not None:
        macro_low = compose_macro_low_background(
            shape_result.macro_low_rgb, shape_result.macro_source_map >= 0,
            labels, texture_sources)
        Image.fromarray(macro_low, "RGB").save(args.output / "macro_low_frequency_filled.png")
    write_outputs(args.output, target, west, north, labels, probabilities, args.atlas,
                  texture_sources, macro_low)
    write_audit_outputs(args.output, target, north, west,
                        refined=(shape_result is not None or not args.quick))
    counts = np.bincount(labels.ravel(), minlength=3)
    print(f"target {target.x}/{target.y}; donor east {west.x}/{west.y}; donor north {north.x}/{north.y}")
    print("target classes: " + ", ".join(f"{name}={count}" for name, count in zip(CLASS_NAMES, counts)))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
