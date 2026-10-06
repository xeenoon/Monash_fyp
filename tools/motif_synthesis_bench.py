#!/usr/bin/env python3
"""Isolated hierarchical motif and transition synthesis benchmarks.

This tool deliberately does not run the whole-map baker.  It tests the pieces
that must work first: persistent motif ancestry, cross-scale phase preservation,
frequency-aware ownership, continuous style retrieval, and signed-distance
transition motifs.  Results and quantitative audits are written beneath
``/tmp/fullregen/bench`` by default.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import erf
from scipy.ndimage import (distance_transform_edt, gaussian_filter, label,
                           map_coordinates)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from diagnose_macro_structure import circular_max_correlation  # noqa: E402
from procedural_gap_demo import (  # noqa: E402
    CLASS_COLORS, CLASS_GRASS, CLASS_ROCK, CLASS_SNOW, colour_labels,
)
from terrain_synth import clean_source, discover_level5, tile_rgb  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
CLASS_COUNT = 3
CLASS_NAMES = ("rock", "grass", "snow")
PATCH = 192
# A wide overlap is not free. Where one donor has a sparse feature and the other
# is flat, both sides of that feature cost the seam the same, so the min-cut
# keeps it or drops it on a coin flip - and never invents one. Two thirds of the
# canvas sat inside an overlap at STEP 128, which cost pure snow a third of its
# 1-16 px energy; a 32 px overlap still hides the cut and keeps the structure.
STEP = 160
OVERLAP = PATCH - STEP


@dataclass
class MotifExemplar:
    rgb: np.ndarray
    labels: np.ndarray
    source: tuple[int, int]
    origin: tuple[int, int]
    material_mix: np.ndarray
    descriptor: np.ndarray
    orientation: float
    coherence: float
    transition_pairs: tuple[tuple[int, int], ...]


@dataclass
class MotifPlan:
    motif_id: int
    material: int
    parent_index: int
    source: tuple[int, int]
    angle: float
    scale: float
    style: np.ndarray


@dataclass
class RenderResult:
    low: np.ndarray
    mid: np.ndarray
    high: np.ndarray
    rgb: np.ndarray
    donor_id: np.ndarray
    source_x: np.ndarray
    source_y: np.ndarray
    motif_id: np.ndarray
    band_owner: np.ndarray


def _grey(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) @ np.array((.299, .587, .114), np.float32)


def _tensor_descriptor(rgb: np.ndarray) -> tuple[float, float]:
    luminance = _grey(rgb)
    gy, gx = np.gradient(gaussian_filter(luminance, 1.0))
    jxx, jyy = float(np.mean(gx * gx)), float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    orientation = float(np.mod(.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2,
                               np.pi))
    coherence = float(np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) /
                      max(jxx + jyy, 1.0e-6))
    return orientation, coherence


def patch_descriptor(rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float32)
    luminance = _grey(rgb)
    low = gaussian_filter(luminance, 16.0)
    medium = gaussian_filter(luminance, 3.0) - low
    high = luminance - gaussian_filter(luminance, 3.0)
    _orientation, coherence = _tensor_descriptor(rgb)
    mix = np.bincount(labels.ravel(), minlength=CLASS_COUNT) / labels.size
    return np.array((
        *(values.mean((0, 1)) / 255.0),
        luminance.std() / 64.0,
        np.hypot(*np.gradient(luminance)).mean() / 32.0,
        coherence,
        medium.std() / 32.0,
        high.std() / 24.0,
        *mix,
    ), np.float32)


def transition_pairs(labels: np.ndarray) -> tuple[tuple[int, int], ...]:
    found: set[tuple[int, int]] = set()
    for first in range(CLASS_COUNT):
        for second in range(first + 1, CLASS_COUNT):
            horizontal = (((labels[:, :-1] == first) & (labels[:, 1:] == second)) |
                          ((labels[:, :-1] == second) & (labels[:, 1:] == first)))
            vertical = (((labels[:-1] == first) & (labels[1:] == second)) |
                        ((labels[:-1] == second) & (labels[1:] == first)))
            if int(horizontal.sum() + vertical.sum()) >= 20:
                found.add((first, second))
    return tuple(sorted(found))


def build_motif_bank(dataset: Path, excluded: set[tuple[int, int]],
                     max_tiles: int = 768) -> list[MotifExemplar]:
    coordinates = [value for value in discover_level5(dataset) if value not in excluded]
    if len(coordinates) > max_tiles:
        indexes = np.linspace(0, len(coordinates) - 1, max_tiles).round().astype(int)
        coordinates = [coordinates[int(index)] for index in indexes]
    bank: list[MotifExemplar] = []
    origins = (0, 32, 64)
    for tile_index, source in enumerate(coordinates, 1):
        rgb = tile_rgb(dataset, *source)
        if rgb is None:
            continue
        clean, _confidence, hard = clean_source(rgb)
        labels = colour_labels(clean)
        for oy in origins:
            for ox in origins:
                hard_crop = hard[oy:oy + PATCH, ox:ox + PATCH]
                if float(hard_crop.mean()) > .18:
                    continue
                # Keep the original photographic RGB. Dark vegetation and the
                # fine contrast it contributes are motif content, not defects.
                # The cleaned copy is used only to derive stable material labels;
                # hard road/water/magenta contamination still rejects the crop.
                crop = rgb[oy:oy + PATCH, ox:ox + PATCH].copy()
                cleaned_crop = clean[oy:oy + PATCH, ox:ox + PATCH]
                crop[hard_crop] = cleaned_crop[hard_crop]
                classes = labels[oy:oy + PATCH, ox:ox + PATCH].copy()
                mix = np.bincount(classes.ravel(), minlength=CLASS_COUNT) / classes.size
                orientation, coherence = _tensor_descriptor(crop)
                bank.append(MotifExemplar(
                    crop, classes, source, (ox, oy), mix.astype(np.float32),
                    patch_descriptor(crop, classes), orientation, coherence,
                    transition_pairs(classes)))
        if tile_index % 40 == 0 or tile_index == len(coordinates):
            print(f"  bank {tile_index}/{len(coordinates)} tiles; motifs={len(bank)}",
                  flush=True)
    if not bank:
        raise RuntimeError("motif bank is empty")
    return bank


def split_bands(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = rgb.astype(np.float32)
    low = gaussian_filter(values, (16.0, 16.0, 0), mode="reflect")
    smooth = gaussian_filter(values, (3.0, 3.0, 0), mode="reflect")
    return low, smooth - low, values - smooth


def transform_coordinates(size: int, angle: float, scale: float,
                          bend: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices((size, size), dtype=np.float32)
    centre = (size - 1) / 2
    dx, dy = (xx - centre) / scale, (yy - centre) / scale
    cosine, sine = np.cos(angle), np.sin(angle)
    source_x = cosine * dx + sine * dy + centre
    source_y = -sine * dx + cosine * dy + centre
    if bend:
        source_x += bend * np.sin((source_y - centre) / max(size / 7, 1))
    return source_y, source_x


def transform_exemplar(exemplar: MotifExemplar, angle: float, scale: float,
                       bend: float = 0.0
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray]:
    """Warp every band and provenance through one shared source-coordinate map."""
    low, mid, high = split_bands(exemplar.rgb)
    source_y, source_x = transform_coordinates(PATCH, angle, scale, bend)
    warped = []
    for band in (low, mid, high):
        channels = [map_coordinates(band[..., channel], (source_y, source_x),
                                    order=3, mode="reflect", prefilter=True)
                    for channel in range(3)]
        warped.append(np.stack(channels, -1).astype(np.float32))
    # Cubic rotation still changes residual RMS slightly. Restore the source
    # energy per channel instead of manufacturing unrelated sharpening noise.
    for output, source in zip(warped[1:], (mid, high)):
        source_rms = np.sqrt(np.mean(source * source, axis=(0, 1)))
        output_rms = np.sqrt(np.mean(output * output, axis=(0, 1)))
        output *= (source_rms / np.maximum(output_rms, 1.0e-5))[None, None]
    warped_labels = map_coordinates(exemplar.labels.astype(np.float32),
                                    (source_y, source_x), order=0,
                                    mode="reflect").astype(np.uint8)
    return (*warped, warped_labels, source_x, source_y)


def _vseam(error: np.ndarray) -> np.ndarray:
    height, width = error.shape
    cost = error.astype(np.float32).copy()
    parent = np.zeros((height, width), np.int16)
    for y in range(1, height):
        for x in range(width):
            lo, hi = max(0, x - 1), min(width, x + 2)
            previous = lo + int(np.argmin(cost[y - 1, lo:hi]))
            parent[y, x] = previous
            cost[y, x] += cost[y - 1, previous]
    take = np.zeros((height, width), bool)
    x = int(np.argmin(cost[-1]))
    for y in range(height - 1, -1, -1):
        take[y, x + 1:] = True
        x = int(parent[y, x])
    return take


def _origins(extent: int) -> list[int]:
    values = list(range(0, extent - PATCH + 1, STEP))
    values.append(extent - PATCH)
    return sorted(set(values))


def _plane_fit(band: np.ndarray) -> np.ndarray:
    """Least-squares plane of each channel, evaluated over the patch."""
    yy, xx = np.indices(band.shape[:2], dtype=np.float32)
    basis = np.stack((np.ones_like(xx), xx / PATCH, yy / PATCH), -1).reshape(-1, 3)
    coefficients = np.linalg.lstsq(basis, band.reshape(-1, band.shape[-1]), rcond=None)[0]
    return (basis @ coefficients).reshape(band.shape).astype(np.float32)


def _overlap_window() -> np.ndarray:
    """Partition of unity that is flat inside a patch and ramps over the overlap.

    A full-width Hanning window averages up to four donors at every pixel, which
    cancels their low-frequency shading and flattens macro contrast.  Ramping
    only across ``OVERLAP`` keeps each patch interior at unit weight, so the LOW
    band retains real donor variation while still crossing over smoothly.
    """
    ramp = (.5 - .5 * np.cos(np.pi * (np.arange(OVERLAP) + .5) / OVERLAP)).astype(np.float32)
    profile = np.ones(PATCH, np.float32)
    profile[:OVERLAP] = ramp
    profile[PATCH - OVERLAP:] = ramp[::-1]
    return profile[:, None] * profile[None]


def continuous_style_field(size: int, seed: int,
                           descriptor_rows: np.ndarray) -> np.ndarray:
    """Slowly varying descriptor targets, never a discrete visible state.

    The targets travel along the principal directions of the real descriptor
    cloud rather than varying every channel on its own.  Independent channels
    ask for combinations no donor has - pink snow, bright low-contrast rock -
    and the nearest available answer to an impossible request is a bland patch.
    """
    rng = np.random.default_rng(seed)
    centre = descriptor_rows.mean(0)
    centred = descriptor_rows - centre
    components = np.linalg.svd(centred, full_matrices=False)[2][:3]
    scores = centred @ components.T
    field = np.repeat(np.repeat(centre.astype(np.float32)[None, None], size, 0), size, 1)
    for index, component in enumerate(components):
        noise = gaussian_filter(rng.standard_normal((size, size)),
                                max(size / (5.5 + index), 28), mode="reflect")
        noise = (noise - noise.mean()) / max(float(noise.std()), 1.0e-6)
        # Quantile mapping, so the canvas spends its area on each style in the
        # same proportion the donors do. A fixed range spread the area evenly
        # instead, which on a bimodal material hands half the map to the blank
        # mode and leaves the median window emptier than the median donor.
        uniform = .5 * (1 + erf(noise / np.sqrt(2)))
        ordered = np.sort(scores[:, index])
        coefficient = ordered[np.clip((uniform * (len(ordered) - 1)).round(), 0,
                                      len(ordered) - 1).astype(np.int32)]
        field += coefficient[..., None].astype(np.float32) * component.astype(np.float32)
    return field


def orientation_field(size: int, first: float, second: float | None,
                      seed: int) -> np.ndarray:
    y, x = np.indices((size, size), dtype=np.float32)
    if second is None:
        base = first + .42 * (x / max(size - 1, 1) - .5)
    else:
        blend = np.clip((x - size * .38) / (size * .24), 0, 1)
        base = first * (1 - blend) + second * blend
    noise = gaussian_filter(np.random.default_rng(seed).standard_normal((size, size)),
                            size / 12, mode="reflect")
    noise = noise / max(float(noise.std()), 1.0e-6)
    return np.mod(base + .10 * noise, np.pi).astype(np.float32)


def candidate_pool(bank: list[MotifExemplar], material: int,
                   purity: float = .72) -> np.ndarray:
    indexes = [index for index, exemplar in enumerate(bank)
               if exemplar.material_mix[material] >= purity]
    if len(indexes) < 8:
        indexes = list(np.argsort([exemplar.material_mix[material]
                                  for exemplar in bank])[-max(8, len(bank) // 12):])
    return np.asarray(indexes, np.int32)


def choose_parent(bank: list[MotifExemplar], pool: np.ndarray,
                  target_style: np.ndarray, target_angle: float,
                  rng: np.random.Generator) -> int:
    count = min(48, len(pool))
    candidates = rng.choice(pool, count, replace=False)
    descriptors = np.asarray([bank[int(index)].descriptor for index in candidates])
    scale = np.maximum(np.std(descriptors, axis=0), .08)
    cost = np.mean(((descriptors - target_style) / scale) ** 2, axis=1)
    delta = np.asarray([bank[int(index)].orientation for index in candidates]) - target_angle
    cost += .30 * (1 - np.cos(2 * delta))
    finalists = np.argsort(cost)[:min(6, len(cost))]
    weights = np.exp(-(cost[finalists] - cost[finalists[0]]) /
                     max(float(np.std(cost[finalists])), .15))
    weights /= weights.sum()
    return int(candidates[int(rng.choice(finalists, p=weights))])


def child_candidates(bank: list[MotifExemplar], pool: np.ndarray,
                     parent: MotifExemplar, target_style: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    # Ancestry means the same piece of terrain, not the same 192 px window. The
    # crops inside one 256 px tile overlap by more than 80%, so a child confined
    # to the parent's own tile can only restamp it; the eight neighbouring tiles
    # continue the same formation with pixels the parent has not already used.
    local = [index for index in pool
             if abs(bank[int(index)].source[0] - parent.source[0]) <= 1 and
             abs(bank[int(index)].source[1] - parent.source[1]) <= 1]
    descriptors = np.asarray([bank[int(index)].descriptor for index in pool])
    scale = np.maximum(descriptors.std(0), .08)
    costs = np.mean(((descriptors - target_style) / scale) ** 2, axis=1)
    compatible = pool[np.argsort(costs)[:min(24, len(pool))]].tolist()
    # Shared ancestry dominates, with descriptor-compatible substitutions and
    # a small amount of global exploration. The motif dies after its 2x2 group.
    selected = list(dict.fromkeys(local + compatible))
    if len(selected) < 12:
        selected.extend(int(value) for value in rng.choice(
            pool, min(12 - len(selected), len(pool)), replace=False))
    return np.asarray(selected, np.int32)


def render_material(bank: list[MotifExemplar], material: int, size: int,
                    seed: int, first_angle: float,
                    second_angle: float | None = None) -> RenderResult:
    rng = np.random.default_rng(seed)
    pool = candidate_pool(bank, material, .90 if material == CLASS_SNOW else .72)
    descriptor_rows = np.asarray([bank[int(index)].descriptor for index in pool])
    # Seam cost is normalized by the contrast actually present. The floor has to
    # scale with the material: a fixed one lets a blank candidate always beat a
    # structured one on low-contrast snow, and the blandness then feeds back as
    # the canvas it is compared against.
    seam_floor = max(float(np.median(descriptor_rows[:, 6])) * 32 * .75, .5)
    style_field = continuous_style_field(size, seed + 11, descriptor_rows)
    angle_field = orientation_field(size, first_angle, second_angle, seed + 19)
    low_sum = np.zeros((size, size, 3), np.float32)
    low_weight = np.zeros((size, size), np.float32)
    mid_canvas = np.zeros((size, size, 3), np.float32)
    high_canvas = np.zeros_like(mid_canvas)
    known = np.zeros((size, size), bool)
    donor_id = np.full((size, size), -1, np.int16)
    source_x_map = np.full((size, size), -1, np.int16)
    source_y_map = np.full((size, size), -1, np.int16)
    motif_map = np.full((size, size), -1, np.int16)
    band_owner = np.zeros((size, size), np.uint8)
    axis = _origins(size)
    motif_plans: dict[tuple[int, int], MotifPlan] = {}
    motif_id = 0
    low_window = _overlap_window()
    for row_index, oy in enumerate(axis):
        for column_index, ox in enumerate(axis):
            cy, cx = oy + PATCH // 2, ox + PATCH // 2
            group = (row_index // 2, column_index // 2)
            target_style = style_field[min(cy, size - 1), min(cx, size - 1)]
            target_angle = float(angle_field[min(cy, size - 1), min(cx, size - 1)])
            if group not in motif_plans:
                parent_index = choose_parent(bank, pool, target_style, target_angle, rng)
                parent = bank[parent_index]
                motif_plans[group] = MotifPlan(
                    motif_id, material, parent_index, parent.source,
                    target_angle - parent.orientation,
                    float(rng.uniform(.88, 1.14)), target_style.copy())
                motif_id += 1
            plan = motif_plans[group]
            parent = bank[plan.parent_index]
            candidates = child_candidates(bank, pool, parent, target_style, rng)
            # Candidate selection remains inside the motif lineage. Seam cost is
            # normalized, so a bland patch cannot win merely by having no edges.
            costs, transformed_rows = [], []
            for index in rng.choice(candidates, min(14, len(candidates)), replace=False):
                exemplar = bank[int(index)]
                rotation = target_angle - exemplar.orientation
                transformed = transform_exemplar(
                    exemplar, rotation, plan.scale, bend=float(rng.uniform(-2.0, 2.0)))
                # Residual bands are deliberately never rescaled towards the
                # style field. Rescaling looks like a cheap way to even out
                # contrast between neighbours, but it drains real donor energy -
                # measurably about a fifth of the 1-16 px bands. The field's
                # authority is exercised in selection and in LOW tone instead.
                mid = transformed[1]
                descriptor_scale = np.maximum(descriptor_rows.std(0), .08)
                descriptor_cost = float(np.mean(((exemplar.descriptor - target_style) /
                                                 descriptor_scale) ** 2))
                seam_cost = 0.0
                if ox > 0:
                    existing = mid_canvas[oy:oy + PATCH, ox:ox + OVERLAP]
                    denominator = max(float(np.std(existing) + np.std(mid[:, :OVERLAP])),
                                      seam_floor)
                    seam_cost += float(np.mean((existing - mid[:, :OVERLAP]) ** 2) /
                                       denominator ** 2)
                if oy > 0:
                    existing = mid_canvas[oy:oy + OVERLAP, ox:ox + PATCH]
                    denominator = max(float(np.std(existing) + np.std(mid[:OVERLAP])),
                                      seam_floor)
                    seam_cost += float(np.mean((existing - mid[:OVERLAP]) ** 2) /
                                       denominator ** 2)
                lineage_cost = 0.0 if exemplar.source == plan.source else .45
                costs.append(descriptor_cost + .30 * seam_cost + lineage_cost)
                transformed_rows.append((int(index), transformed))
            finalists = np.argsort(costs)[:min(4, len(costs))]
            shifted = np.asarray(costs)[finalists] - float(np.min(costs))
            weights = np.exp(-shifted / max(float(np.std(shifted)), .18)); weights /= weights.sum()
            selected = int(rng.choice(finalists, p=weights))
            exemplar_index, transformed = transformed_rows[selected]
            low, mid, high, _labels, source_x, source_y = transformed
            ys, xs = slice(oy, oy + PATCH), slice(ox, ox + PATCH)
            # The generated style field, not the donor, owns broad tone. The
            # donor's own mean and gradient are replaced by the field sampled
            # across this patch, which is continuous between neighbours, so the
            # LOW band keeps every donor's curvature and texture without the
            # rectangles that appear where two donors disagree about brightness.
            low = low - _plane_fit(low) + style_field[ys, xs, :3] * 255
            low_sum[ys, xs] += low * low_window[..., None]
            low_weight[ys, xs] += low_window
            take = np.ones((PATCH, PATCH), bool)
            # Cut on the MID band alone. Two donors never agree in HIGH, so an
            # error surface that includes it is dominated by "how textured is
            # this pixel" and routes every seam through the blandest available
            # path, quietly discarding the structure the donors were chosen for.
            reconstructed = mid
            existing_reconstructed = mid_canvas[ys, xs]
            if ox > 0:
                error = np.mean((existing_reconstructed[:, :OVERLAP] -
                                 reconstructed[:, :OVERLAP]) ** 2, axis=2)
                take[:, :OVERLAP] = _vseam(error)
            if oy > 0:
                error = np.mean((existing_reconstructed[:OVERLAP] -
                                 reconstructed[:OVERLAP]) ** 2, axis=2)
                take[:OVERLAP] &= _vseam(error.T).T
            take |= ~known[ys, xs]
            mid_canvas[ys, xs][take] = mid[take]
            high_canvas[ys, xs][take] = high[take]
            known[ys, xs] |= take
            donor_id[ys, xs][take] = exemplar_index
            source_x_map[ys, xs][take] = np.clip(
                source_x[take] + bank[exemplar_index].origin[0], 0, 255).astype(np.int16)
            source_y_map[ys, xs][take] = np.clip(
                source_y[take] + bank[exemplar_index].origin[1], 0, 255).astype(np.int16)
            motif_map[ys, xs][take] = plan.motif_id
            band_owner[ys, xs][take] = 3
    low = low_sum / np.maximum(low_weight[..., None], 1.0e-6)
    rgb = np.clip(low + mid_canvas + high_canvas, 0, 255).astype(np.uint8)
    return RenderResult(low, mid_canvas, high_canvas, rgb, donor_id,
                        source_x_map, source_y_map, motif_map, band_owner)


def signed_distance(mask: np.ndarray) -> np.ndarray:
    return (distance_transform_edt(mask) - distance_transform_edt(~mask)).astype(np.float32)


def transition_pool(bank: list[MotifExemplar], first: int, second: int) -> np.ndarray:
    pair = tuple(sorted((first, second)))
    adjacent = [index for index, exemplar in enumerate(bank)
                if pair in exemplar.transition_pairs]
    indexes = [index for index in adjacent
               if bank[index].material_mix[first] > .08 and
               bank[index].material_mix[second] > .08 and
               bank[index].material_mix[first] + bank[index].material_mix[second] > .70]
    if not indexes:
        indexes = [index for index in adjacent
                   if bank[index].material_mix[first] > .01 and
                   bank[index].material_mix[second] > .01]
    # A thin snow finger or vegetation pocket can occupy much less than ten
    # percent of a 192 px crop while still containing the useful perpendicular
    # transition phrase.  Adjacency, rather than balanced area, is the hard
    # requirement here; the target SDF owns the new boundary geometry.
    return np.asarray(indexes, np.int32)


def _transition_boundary(labels: np.ndarray, first: int, second: int) -> np.ndarray:
    boundary = np.zeros(labels.shape, bool)
    horizontal = (((labels[:, :-1] == first) & (labels[:, 1:] == second)) |
                  ((labels[:, :-1] == second) & (labels[:, 1:] == first)))
    vertical = (((labels[:-1] == first) & (labels[1:] == second)) |
                ((labels[:-1] == second) & (labels[1:] == first)))
    boundary[:, :-1] |= horizontal
    boundary[:, 1:] |= horizontal
    boundary[:-1] |= vertical
    boundary[1:] |= vertical
    return boundary


def transform_transition(exemplar: MotifExemplar, first: int, second: int,
                         target_sdf: np.ndarray, target_centre: tuple[int, int],
                         output_origin: tuple[int, int], scale: float
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray]:
    """Warp a donor phrase by distance along/across a novel target boundary.

    The target SDF owns all boundary geometry. The donor contributes only the
    photographic transition sequence perpendicular to its own local edge.
    Every frequency band uses the exact same source-coordinate map.
    """
    source_boundary = _transition_boundary(exemplar.labels, first, second)
    sy_values, sx_values = np.nonzero(source_boundary)
    if not len(sy_values):
        raise ValueError("transition exemplar has no boundary pixels")
    centre = np.array((PATCH / 2, PATCH / 2))
    nearest = int(np.argmin((sy_values - centre[0]) ** 2 +
                            (sx_values - centre[1]) ** 2))
    source_cy, source_cx = int(sy_values[nearest]), int(sx_values[nearest])
    source_sdf = signed_distance(exemplar.labels == second)
    source_gy, source_gx = np.gradient(gaussian_filter(source_sdf, 3.0))
    source_normal = np.array((source_gx[source_cy, source_cx],
                              source_gy[source_cy, source_cx]), np.float32)
    source_normal /= max(float(np.linalg.norm(source_normal)), 1.0e-6)
    source_tangent = np.array((-source_normal[1], source_normal[0]), np.float32)

    target_cy, target_cx = target_centre
    origin_y, origin_x = output_origin
    target_gy, target_gx = np.gradient(gaussian_filter(target_sdf, 2.0))
    target_normal = np.array((target_gx[target_cy, target_cx],
                              target_gy[target_cy, target_cx]), np.float32)
    target_normal /= max(float(np.linalg.norm(target_normal)), 1.0e-6)
    target_tangent = np.array((-target_normal[1], target_normal[0]), np.float32)
    yy, xx = np.indices((PATCH, PATCH), dtype=np.float32)
    world_x, world_y = xx + origin_x, yy + origin_y
    along = ((world_x - target_cx) * target_tangent[0] +
             (world_y - target_cy) * target_tangent[1]) / scale
    across = target_sdf[origin_y:origin_y + PATCH,
                         origin_x:origin_x + PATCH] / scale
    source_x = source_cx + along * source_tangent[0] + across * source_normal[0]
    source_y = source_cy + along * source_tangent[1] + across * source_normal[1]

    warped: list[np.ndarray] = []
    for band in split_bands(exemplar.rgb):
        channels = [map_coordinates(band[..., channel], (source_y, source_x),
                                    order=3, mode="reflect", prefilter=True)
                    for channel in range(3)]
        warped.append(np.stack(channels, -1).astype(np.float32))
    # Preserve residual energy after cubic interpolation, as in the pure motif
    # transform. This is a scalar band correction, never invented sharpening.
    for output, source in zip(warped[1:], split_bands(exemplar.rgb)[1:]):
        source_rms = np.sqrt(np.mean(source * source, axis=(0, 1)))
        output_rms = np.sqrt(np.mean(output * output, axis=(0, 1)))
        output *= (source_rms / np.maximum(output_rms, 1.0e-5))[None, None]
    return (*warped, source_x, source_y)


def render_transition_texture(bank: list[MotifExemplar], first: int, second: int,
                              size: int, sdf: np.ndarray, seed: int) -> RenderResult:
    """Render mixed-material motifs along a novel target boundary."""
    rng = np.random.default_rng(seed)
    pool = transition_pool(bank, first, second)
    if not len(pool):
        raise RuntimeError(f"no transition motifs for {first}/{second}")
    # Reuse the hierarchical compositor's ownership machinery by constructing
    # boundary-centred placements directly.
    low_sum = np.zeros((size, size, 3), np.float32)
    low_weight = np.zeros((size, size), np.float32)
    mid_sum = np.zeros((size, size, 3), np.float32)
    mid_weight = np.zeros((size, size), np.float32)
    high = np.zeros_like(mid_sum)
    known = np.zeros((size, size), bool)
    donor = np.full((size, size), -1, np.int16)
    sx_map = np.full((size, size), -1, np.int16)
    sy_map = np.full((size, size), -1, np.int16)
    motif_map = np.full((size, size), -1, np.int16)
    owner = np.zeros((size, size), np.uint8)
    owner_cost = np.full((size, size), np.inf, np.float32)
    # These benchmark boundaries cross the image predominantly top-to-bottom.
    # Take one SDF minimum per 96 px of arc progression, instead of sampling
    # every rasterized edge pixel and changing ancestry every few dozen pixels.
    point_rows = list(range(48, size, 96))
    if not point_rows or point_rows[-1] < size - 64:
        point_rows.append(size - 48)
    points = [(row, int(np.argmin(np.abs(sdf[row])))) for row in point_rows]
    window_1d = np.maximum(np.hanning(PATCH).astype(np.float32), .04)
    window = window_1d[:, None] * window_1d[None]
    previous_index: int | None = None
    for motif_id, (cy, cx) in enumerate(points):
        oy, ox = int(np.clip(cy - PATCH // 2, 0, size - PATCH)), int(np.clip(cx - PATCH // 2, 0, size - PATCH))
        candidates = rng.choice(pool, min(20, len(pool)), replace=False)
        # Prefer a balanced transition and a coherent source boundary phrase.
        descriptor_scale = np.maximum(
            np.std([bank[int(index)].descriptor for index in pool], axis=0), .08)
        costs = []
        for index in candidates:
            current = bank[int(index)]
            cost = abs(float(current.material_mix[first] -
                             current.material_mix[second])) - .15 * current.coherence
            cost += 3.0 * max(0.0, 1.0 - float(current.material_mix[first] +
                                               current.material_mix[second]))
            if previous_index is not None:
                previous = bank[previous_index]
                descriptor_cost = float(np.mean(((current.descriptor - previous.descriptor) /
                                                 descriptor_scale) ** 2))
                cost += .30 * descriptor_cost
                if current.source == previous.source:
                    cost -= .45
            costs.append(cost)
        exemplar_index = int(candidates[int(np.argmin(costs))])
        previous_index = exemplar_index
        exemplar = bank[exemplar_index]
        low_patch, mid_patch, high_patch, source_x, source_y = transform_transition(
            exemplar, first, second, sdf, (cy, cx), (oy, ox),
            float(rng.uniform(.9, 1.12)))
        ys, xs = slice(oy, oy + PATCH), slice(ox, ox + PATCH)
        local_sdf = sdf[ys, xs]
        support = np.abs(local_sdf) < 72
        low_alpha = window * np.clip((72 - np.abs(local_sdf)) / 40, 0, 1)
        low_sum[ys, xs] += low_patch * low_alpha[..., None]
        low_weight[ys, xs] += low_alpha
        mid_alpha = window * support
        mid_sum[ys, xs] += mid_patch * mid_alpha[..., None]
        mid_weight[ys, xs] += mid_alpha
        local_y, local_x = np.indices((PATCH, PATCH), dtype=np.float32)
        placement_cost = (local_y + oy - cy) ** 2 + (local_x + ox - cx) ** 2
        take = support & (placement_cost < owner_cost[ys, xs])
        # High-frequency ownership is hard and narrower than the mid strip.
        high_take = take & (np.abs(local_sdf) < 48)
        high[ys, xs][high_take] = high_patch[high_take]
        known[ys, xs] |= support
        owner_cost[ys, xs][take] = placement_cost[take]
        donor[ys, xs][take] = exemplar_index
        sx_map[ys, xs][take] = np.clip(source_x[take] + exemplar.origin[0], 0, 255).astype(np.int16)
        sy_map[ys, xs][take] = np.clip(source_y[take] + exemplar.origin[1], 0, 255).astype(np.int16)
        motif_map[ys, xs][take] = motif_id
        owner[ys, xs][take] = 2; owner[ys, xs][high_take] = 3
    low = low_sum / np.maximum(low_weight[..., None], 1.0e-6)
    mid = mid_sum / np.maximum(mid_weight[..., None], 1.0e-6)
    rgb = np.clip(low + mid + high, 0, 255).astype(np.uint8)
    return RenderResult(low, mid, high, rgb, donor, sx_map, sy_map, motif_map, owner)


def compose_transition(first: RenderResult, second: RenderResult,
                       transition: RenderResult, sdf: np.ndarray) -> RenderResult:
    """Wide LOW blend, narrow MID replacement, hard HIGH ownership."""
    size = sdf.shape[0]
    low_weight = np.clip(.5 + sdf / 56, 0, 1)[..., None]
    low = first.low * (1 - low_weight) + second.low * low_weight
    transition_low_weight = np.clip((72 - np.abs(sdf)) / 40, 0, 1)[..., None]
    valid_transition = (transition.donor_id >= 0)[..., None]
    low = np.where(valid_transition, low * (1 - transition_low_weight) +
                   transition.low * transition_low_weight, low)
    hard_second = sdf >= 0
    mid = np.where(hard_second[..., None], second.mid, first.mid)
    high = np.where(hard_second[..., None], second.high, first.high)
    mid_take = (np.abs(sdf) < 56) & (transition.donor_id >= 0)
    high_take = (np.abs(sdf) < 36) & (transition.band_owner == 3)
    mid[mid_take] = transition.mid[mid_take]
    high[high_take] = transition.high[high_take]
    rgb = np.clip(low + mid + high, 0, 255).astype(np.uint8)
    donor = np.where(mid_take, transition.donor_id,
                     np.where(hard_second, second.donor_id, first.donor_id))
    source_x = np.where(mid_take, transition.source_x,
                        np.where(hard_second, second.source_x, first.source_x))
    source_y = np.where(mid_take, transition.source_y,
                        np.where(hard_second, second.source_y, first.source_y))
    motif = np.where(mid_take, transition.motif_id,
                     np.where(hard_second, second.motif_id, first.motif_id))
    owner = np.where(high_take, 3, np.where(mid_take, 2, 1)).astype(np.uint8)
    return RenderResult(low, mid, high, rgb, donor.astype(np.int16),
                        source_x.astype(np.int16), source_y.astype(np.int16),
                        motif.astype(np.int16), owner)


def dog_energy(rgb: np.ndarray) -> dict[str, float]:
    grey = _grey(rgb)
    sigmas = (.5, 1, 2, 4, 8, 16, 32, 64)
    blurred = [grey] + [gaussian_filter(grey, sigma, mode="reflect") for sigma in sigmas]
    names = ("1_2", "2_4", "4_8", "8_16", "16_32", "32_64", "64_128", "128_256")
    return {name: float(np.std(blurred[index] - blurred[index + 1]))
            for index, name in enumerate(names)}


def structure_coherence(rgb: np.ndarray, windows=(8, 16, 32, 64, 128)) -> dict[str, float]:
    grey = _grey(rgb)
    gy, gx = np.gradient(grey)
    result = {}
    for window in windows:
        sigma = max(window / 3, 1)
        jxx = gaussian_filter(gx * gx, sigma)
        jyy = gaussian_filter(gy * gy, sigma)
        jxy = gaussian_filter(gx * gy, sigma)
        coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum(jxx + jyy, 1.0e-6)
        weight = np.sqrt(jxx + jyy)
        result[str(window)] = float(np.sum(coherence * weight) / max(float(weight.sum()), 1.0e-6))
    return result


def windowed_metrics(rgb: np.ndarray, window: int = PATCH) -> dict[str, object]:
    """Median statistics over donor-sized windows of a render.

    Donor references are single ``PATCH`` crops, so a whole-image statistic is
    not comparable to them: at a 128 px coherence window a 192 px crop can only
    report its own dominant orientation, while a 1024 px render reports genuine
    local variation.  Cutting the render into donor-sized windows first makes
    both sides measure the same thing.
    """
    crops = [rgb[oy:oy + window, ox:ox + window]
             for oy in range(0, rgb.shape[0] - window + 1, window)
             for ox in range(0, rgb.shape[1] - window + 1, window)]
    energy_rows = [dog_energy(crop) for crop in crops]
    coherence_rows = [structure_coherence(crop) for crop in crops]
    return {
        "dog_energy": {key: float(np.median([row[key] for row in energy_rows]))
                       for key in energy_rows[0]},
        "coherence": {key: float(np.median([row[key] for row in coherence_rows]))
                      for key in coherence_rows[0]},
        "windows": len(crops),
    }


def boundary_energy_profile(rgb: np.ndarray, sdf: np.ndarray) -> dict[str, float]:
    gradient = np.hypot(*np.gradient(_grey(rgb)))
    result = {}
    for lo, hi in ((0, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64)):
        selected = (np.abs(sdf) >= lo) & (np.abs(sdf) < hi)
        result[f"{lo}_{hi}"] = float(gradient[selected].mean()) if selected.any() else 0.0
    return result


def patch_stride_peak(rgb: np.ndarray, stride: int = STEP) -> float:
    grey = _grey(rgb)
    local = gaussian_filter(grey, 6)[::8, ::8]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(local - local.mean())))
    centre = np.array(spectrum.shape) // 2
    radius = max(1, round(rgb.shape[0] / stride))
    yy, xx = np.indices(spectrum.shape)
    rr = np.hypot(xx - centre[1], yy - centre[0])
    ring = spectrum[(rr >= radius - 1) & (rr <= radius + 1)]
    background = spectrum[(rr >= max(2, radius - 5)) & (rr <= radius + 5)]
    return float(np.max(ring) / max(float(np.median(background)), 1.0e-6))


def provenance_metrics(result: RenderResult) -> dict[str, object]:
    valid = result.motif_id >= 0
    ids, counts = np.unique(result.motif_id[valid], return_counts=True)
    donors = np.unique(result.donor_id[result.donor_id >= 0])
    return {
        "motifs": int(len(ids)),
        "donors": int(len(donors)),
        "median_motif_pixels": float(np.median(counts)) if len(counts) else 0.0,
        "p90_motif_pixels": float(np.percentile(counts, 90)) if len(counts) else 0.0,
    }


def novelty_score(rgb: np.ndarray, bank: list[MotifExemplar]) -> float:
    output = np.asarray(Image.fromarray(rgb).convert("L").resize((64, 64), Image.Resampling.BOX))
    values = []
    for exemplar in bank[::max(1, len(bank) // 96)]:
        donor = np.asarray(Image.fromarray(exemplar.rgb).convert("L").resize((64, 64), Image.Resampling.BOX))
        values.append(circular_max_correlation(output, donor))
    return float(max(values, default=0.0))


def write_render(output: Path, name: str, result: RenderResult,
                 bank: list[MotifExemplar], sdf: np.ndarray | None = None) -> dict[str, object]:
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(result.low, 0, 255).astype(np.uint8), "RGB").save(directory / "01_low.png")
    mid_view = np.clip(result.low + result.mid, 0, 255).astype(np.uint8)
    Image.fromarray(mid_view, "RGB").save(directory / "02_low_mid.png")
    Image.fromarray(result.rgb, "RGB").save(directory / "03_final.png")
    motif_preview = ((result.motif_id.astype(np.int32) * 67) % 255).astype(np.uint8)
    motif_preview[result.motif_id < 0] = 0
    Image.fromarray(motif_preview, "L").save(directory / "motif_id.png")
    donor_preview = ((result.donor_id.astype(np.int32) * 43) % 255).astype(np.uint8)
    donor_preview[result.donor_id < 0] = 0
    Image.fromarray(donor_preview, "L").save(directory / "donor_id.png")
    Image.fromarray(result.band_owner * 80, "L").save(directory / "band_owner.png")
    np.savez_compressed(directory / "provenance.npz", donor_id=result.donor_id,
                        source_x=result.source_x, source_y=result.source_y,
                        motif_id=result.motif_id, band_owner=result.band_owner)
    metrics: dict[str, object] = {
        "dog_energy": dog_energy(result.rgb),
        "coherence": structure_coherence(result.rgb),
        "windowed": windowed_metrics(result.rgb),
        "patch_stride_peak": patch_stride_peak(result.rgb),
        "provenance": provenance_metrics(result),
        "max_donor_correlation_64": novelty_score(result.rgb, bank),
        "stage_energy": {
            "low": dog_energy(np.clip(result.low, 0, 255).astype(np.uint8)),
            "low_mid": dog_energy(mid_view),
            "final": dog_energy(result.rgb),
        },
    }
    if sdf is not None:
        metrics["boundary_energy"] = boundary_energy_profile(result.rgb, sdf)
        Image.fromarray(np.clip(sdf + 128, 0, 255).astype(np.uint8), "L").save(directory / "target_sdf.png")
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
    return metrics


def reference_material_metrics(bank: list[MotifExemplar], material: int) -> dict[str, object]:
    pool = candidate_pool(bank, material, .90 if material == CLASS_SNOW else .72)
    if len(pool) > 48:
        pool = pool[np.linspace(0, len(pool) - 1, 48).round().astype(int)]
    selected = [bank[int(index)] for index in pool]
    energy_rows = [dog_energy(exemplar.rgb) for exemplar in selected]
    coherence_rows = [structure_coherence(exemplar.rgb) for exemplar in selected]
    return {
        "dog_energy": {key: float(np.median([row[key] for row in energy_rows]))
                       for key in energy_rows[0]},
        "coherence": {key: float(np.median([row[key] for row in coherence_rows]))
                      for key in coherence_rows[0]},
        "samples": len(selected),
    }


def curved_sdf(size: int, amplitude: float = 120, period: float = 620) -> np.ndarray:
    y, x = np.indices((size, size), dtype=np.float32)
    boundary = size / 2 + amplitude * np.sin(y * 2 * np.pi / period)
    return (x - boundary).astype(np.float32)


def terrain_snow_sdf(size: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices((size, size), dtype=np.float32)
    height = (1800 + .9 * y + 85 * np.sin(x / 72) +
              55 * np.sin((x + y) / 43))
    dy, dx = np.gradient(height)
    slope = np.hypot(dx, dy)
    curvature = gaussian_filter(height, 3) - gaussian_filter(height, 18)
    base = x - size * .52
    retention = -1.5 * curvature + 22 * np.clip(slope - np.median(slope), -2, 2)
    sdf = base + retention
    return sdf.astype(np.float32), height.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "alps-data" / "trn-alps-16km")
    parser.add_argument("--output", type=Path, default=Path("/tmp/fullregen/bench"))
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--training-tiles", type=int, default=768)
    parser.add_argument("--seed", type=int, default=8181)
    parser.add_argument("--bank-cache", type=Path, default=None,
                        help="pickle the motif bank here and reuse it; building "
                             "it from imagery takes about ninety seconds")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    excluded = {(x, y) for y in range(16) for x in range(16, 32)}
    if args.bank_cache is not None and args.bank_cache.exists():
        print(f"Loading cached motif bank from {args.bank_cache}…", flush=True)
        bank = pickle.loads(args.bank_cache.read_bytes())
    else:
        print("Building target-excluding mixed motif bank…", flush=True)
        bank = build_motif_bank(args.dataset, excluded, args.training_tiles)
        if args.bank_cache is not None:
            args.bank_cache.parent.mkdir(parents=True, exist_ok=True)
            args.bank_cache.write_bytes(pickle.dumps(bank, protocol=4))
    references = {CLASS_NAMES[material]: reference_material_metrics(bank, material)
                  for material in range(CLASS_COUNT)}
    (args.output / "reference_distributions.json").write_text(
        json.dumps(references, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Rendering pure-material coherence cases…", flush=True)
    rock = render_material(bank, CLASS_ROCK, args.size, args.seed + 1, np.deg2rad(20))
    rock_two = render_material(bank, CLASS_ROCK, args.size, args.seed + 2,
                               np.deg2rad(20), np.deg2rad(70))
    grass = render_material(bank, CLASS_GRASS, args.size, args.seed + 3, np.deg2rad(35))
    snow = render_material(bank, CLASS_SNOW, args.size, args.seed + 4, np.deg2rad(80))
    metrics: dict[str, object] = {
        "pure_rock_rotating": write_render(args.output, "pure_rock_rotating", rock, bank),
        "pure_rock_two_motifs": write_render(args.output, "pure_rock_two_motifs", rock_two, bank),
        "pure_grass": write_render(args.output, "pure_grass", grass, bank),
        "pure_snow": write_render(args.output, "pure_snow", snow, bank),
    }

    print("Rendering transition-relative cases…", flush=True)
    straight = np.indices((args.size, args.size), dtype=np.float32)[1] - args.size / 2
    curved = curved_sdf(args.size)
    terrain_sdf, terrain_height = terrain_snow_sdf(args.size)
    for name, first_material, second_material, sdf, seed_offset in (
            ("straight_grass_rock", CLASS_GRASS, CLASS_ROCK, straight, 11),
            ("curved_grass_rock", CLASS_GRASS, CLASS_ROCK, curved, 12),
            ("terrain_rock_snow", CLASS_ROCK, CLASS_SNOW, terrain_sdf, 13)):
        first = grass if first_material == CLASS_GRASS else rock
        second = rock if second_material == CLASS_ROCK else snow
        transition = render_transition_texture(bank, first_material, second_material,
                                               args.size, sdf, args.seed + seed_offset)
        combined = compose_transition(first, second, transition, sdf)
        metrics[name] = write_render(args.output, name, combined, bank, sdf)
        if name == "terrain_rock_snow":
            Image.fromarray(np.clip((terrain_height - terrain_height.min()) /
                                    (terrain_height.max() - terrain_height.min()) * 255,
                                    0, 255).astype(np.uint8), "L").save(
                                        args.output / name / "synthetic_height.png")

    # Aggregate acceptance ratios against real material motifs. One seed is not
    # a measurement on a bimodal donor pool - pure snow moves by a third between
    # seeds - so each pure case is repeated and accepted on the median.
    print("Repeating pure-material cases for a stable acceptance…", flush=True)
    repeats: dict[str, list[dict[str, float]]] = {}
    for case, material_id, angle in (("pure_rock_rotating", CLASS_ROCK, 20),
                                     ("pure_grass", CLASS_GRASS, 35),
                                     ("pure_snow", CLASS_SNOW, 80)):
        rows = [metrics[case]["windowed"]]
        for offset in (137, 421):
            repeat = render_material(bank, material_id, args.size,
                                     args.seed + offset, np.deg2rad(angle))
            rows.append(windowed_metrics(repeat.rgb))
        repeats[case] = rows

    acceptance: dict[str, object] = {}
    for case, material in (("pure_rock_rotating", "rock"),
                           ("pure_grass", "grass"),
                           ("pure_snow", "snow")):
        rows = repeats[case]
        generated = {family: {key: float(np.median([row[family][key] for row in rows]))
                              for key in rows[0][family]}
                     for family in ("dog_energy", "coherence")}
        reference = references[material]
        acceptance[case] = {
            "dog_ratio": {key: generated["dog_energy"][key] /
                          max(reference["dog_energy"][key], 1.0e-6)
                          for key in generated["dog_energy"]},
            "coherence_ratio": {key: generated["coherence"][key] /
                                max(reference["coherence"][key], 1.0e-6)
                                for key in generated["coherence"]},
            "fine_frequency_pass": all(
                generated["dog_energy"][key] /
                max(reference["dog_energy"][key], 1.0e-6) >= .8
                for key in ("1_2", "2_4", "4_8", "8_16")),
            "coherence_64_pass": generated["coherence"]["64"] /
                                 max(reference["coherence"]["64"], 1.0e-6) >= .8,
            "seeds": len(rows),
            "per_seed_dog_1_2": [row["dog_energy"]["1_2"] /
                                 max(reference["dog_energy"]["1_2"], 1.0e-6) for row in rows],
            "whole_image_coherence": metrics[case]["coherence"],
        }
    metrics["acceptance"] = acceptance
    (args.output / "summary.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "README.md").write_text(
        "# Hierarchical motif synthesis benchmarks\n\n"
        "Each case contains LOW, LOW+MID, final, motif/donor/band ownership, exact provenance, "
        "and metrics. The full mountain baker was intentionally not run.\n\n"
        "Acceptance requires at least 0.8× donor energy in the 1–16 px bands and at least "
        "0.8× donor structure coherence at 64 px, while 64×64 donor correlation remains low.\n"
        "Both sides of those ratios are measured over donor-sized (192 px) windows; the "
        "whole-image coherence is reported alongside for reference. Each pure-material "
        "case is rendered with three seeds and accepted on the median, because a bimodal "
        "donor pool moves a single seed by a third.\n",
        encoding="utf-8")
    print(f"wrote benchmark suite to {args.output}", flush=True)


if __name__ == "__main__":
    main()
