#!/usr/bin/env python3
"""Generate a novel, DEM-conditioned Alpine material layout.

This module is the positional firewall between training imagery and generation.
Training imagery may be used to learn P(material | terrain) and short material
boundary phrases.  Generation receives only destination DEM fields, the learned
library, and a seed; it never reads destination RGB.

The layout is intentionally generated at a coarse resolution.  A separate RGB
renderer can then add macro colour, mesostructure, and native microtexture without
giving any reference image authority over destination coordinates.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import (distance_transform_edt, gaussian_filter,
                           gaussian_laplace, label, map_coordinates,
                           uniform_filter)
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from procedural_gap_demo import (  # noqa: E402
    CLASS_COLORS, CLASS_GRASS, CLASS_ROCK, CLASS_SNOW, colour_labels,
)
from terrain_synth import (  # noqa: E402
    clean_source, discover_level5, tile_dem, tile_rgb,
)

Image.MAX_IMAGE_PIXELS = None
CLASS_COUNT = 3


@dataclass
class ShapePhrase:
    material: int
    mask: np.ndarray
    source: tuple[int, int]
    orientation: float
    area: int


@dataclass
class StructureLibrary:
    coordinates: list[tuple[int, int]]
    feature_rows: np.ndarray
    material_rows: np.ndarray
    phrases: list[ShapePhrase]
    feature_tree: cKDTree


@dataclass
class LayoutResult:
    height: np.ndarray
    slope: np.ndarray
    curvature: np.ndarray
    roughness: np.ndarray
    orientation: np.ndarray
    probabilities: np.ndarray
    base_labels: np.ndarray
    shape_labels: np.ndarray
    labels: np.ndarray
    shape_source_map: np.ndarray
    streak_kind: np.ndarray
    flow_accumulation: np.ndarray
    structure_orientation: np.ndarray
    structure_coherence: np.ndarray
    feature_scale: np.ndarray
    local_contrast: np.ndarray
    lithology: np.ndarray
    vegetation_density: np.ndarray
    metrics: dict[str, object]


def _resize_float(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(values.astype(np.float32), "F").resize(
        size, Image.Resampling.BILINEAR), np.float32)


def _resize_labels(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(values.astype(np.uint8), "L").resize(
        size, Image.Resampling.NEAREST), np.uint8)


def assemble_dem(dataset: Path, bounds: tuple[int, int, int, int],
                 tile_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Assemble only DEM-derived fields; this function never opens imagery."""
    x0, x1, y0, y1 = bounds
    height = np.zeros(((y1 - y0) * tile_size, (x1 - x0) * tile_size), np.float32)
    slope = np.zeros_like(height)
    for j, y in enumerate(range(y0, y1)):
        for i, x in enumerate(range(x0, x1)):
            dem = tile_dem(dataset, x, y)
            if dem is None:
                raise FileNotFoundError(f"missing destination DEM tile {x}/{y}")
            ys, xs = slice(j * tile_size, (j + 1) * tile_size), slice(i * tile_size, (i + 1) * tile_size)
            height[ys, xs] = _resize_float(dem[0], (tile_size, tile_size))
            slope[ys, xs] = _resize_float(dem[1], (tile_size, tile_size))
    return height, slope


def terrain_features(height: np.ndarray, slope: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return curvature, roughness, downhill orientation, and search features."""
    broad = gaussian_filter(height, 3.0, mode="nearest")
    curvature = gaussian_laplace(broad, 2.0, mode="nearest")
    mean = uniform_filter(height, 7, mode="nearest")
    roughness = np.sqrt(np.maximum(uniform_filter(height * height, 7, mode="nearest") - mean * mean, 0))
    dy, dx = np.gradient(broad)
    orientation = np.arctan2(-dy, -dx).astype(np.float32)
    features = np.stack((
        height / 1000.0,
        np.clip(slope, 0, 4) / 2.0,
        np.clip(curvature, -20, 20) / 10.0,
        np.clip(roughness, 0, 160) / 80.0,
        np.cos(orientation),
        np.sin(orientation),
    ), axis=-1).astype(np.float32)
    return curvature.astype(np.float32), roughness.astype(np.float32), orientation, features


def flow_accumulation(height: np.ndarray) -> np.ndarray:
    """Small D8 drainage model, normalized logarithmically to [0,1]."""
    h, w = height.shape
    padded = np.pad(height, 1, mode="edge")
    neighbours = []
    offsets = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbours.append(padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w])
            offsets.append((dy, dx))
    stack = np.stack(neighbours)
    choice = stack.argmin(0)
    yy, xx = np.indices((h, w))
    receiver = np.full(h * w, -1, np.int64)
    for index, (dy, dx) in enumerate(offsets):
        selected = (choice == index) & (stack[index] < height)
        ry, rx = yy[selected] + dy, xx[selected] + dx
        receiver[np.flatnonzero(selected)] = ry * w + rx
    accumulation = np.ones(h * w, np.float32)
    for source in np.argsort(height.ravel())[::-1]:
        destination = receiver[int(source)]
        if destination >= 0:
            accumulation[destination] += accumulation[int(source)]
    result = np.log1p(accumulation.reshape(h, w))
    low, high = np.percentile(result, (35, 99.5))
    return np.clip((result - low) / max(float(high - low), 1.0e-6), 0, 1).astype(np.float32)


def generated_structure_fields(height: np.ndarray, slope: np.ndarray,
                               terrain_orientation: np.ndarray,
                               flow: np.ndarray, seed: int
                               ) -> tuple[np.ndarray, ...]:
    """Generate phase-new formation controls conditioned on destination terrain."""
    shape = height.shape
    selector = gaussian_filter(correlated_noise(shape, seed + 1), max(min(shape) / 28, 4))
    # Some formations run downhill; others follow contour.  The selector varies
    # only at formation scale, so neighbouring motifs agree with one another.
    base = terrain_orientation + (selector > 0.05) * (np.pi / 2)
    deviation = correlated_noise(
        shape, seed + 2,
        (max(min(shape) / 8, 12), max(min(shape) / 20, 5), max(min(shape) / 50, 2)))
    structure_orientation = np.mod(base + 0.52 * deviation, np.pi).astype(np.float32)
    coherence_noise = correlated_noise(
        shape, seed + 3,
        (max(min(shape) / 10, 10), max(min(shape) / 24, 4), max(min(shape) / 60, 2)))
    coherence = 1 / (1 + np.exp(-(0.7 + 0.85 * coherence_noise +
                                  0.35 * np.clip(slope, 0, 2))))
    scale_noise = correlated_noise(
        shape, seed + 4,
        (max(min(shape) / 7, 12), max(min(shape) / 18, 5), max(min(shape) / 45, 2)))
    feature_scale = 64 + 128 / (1 + np.exp(-scale_noise))
    contrast = 0.72 + 0.58 / (1 + np.exp(-correlated_noise(shape, seed + 5)))
    lithology = correlated_noise(
        shape, seed + 6,
        (max(min(shape) / 6, 14), max(min(shape) / 15, 6), max(min(shape) / 38, 2.5)))
    height_unit = (height - np.percentile(height, 5)) / max(
        float(np.percentile(height, 95) - np.percentile(height, 5)), 1)
    vegetation_noise = correlated_noise(
        shape, seed + 7,
        (max(min(shape) / 9, 10), max(min(shape) / 25, 4), max(min(shape) / 60, 2)))
    # Flow accumulation is a plausibility proposal, not a rendered answer.
    # Smooth displacement changes the paths, while a broad stochastic gate
    # chooses which drainage families develop dense vegetation at all.
    displacement_y = correlated_noise(
        shape, seed + 8,
        (max(min(shape) / 12, 8), max(min(shape) / 30, 3), 1.5))
    displacement_x = correlated_noise(
        shape, seed + 9,
        (max(min(shape) / 11, 8), max(min(shape) / 28, 3), 1.5))
    yy, xx = np.indices(shape, dtype=np.float32)
    displacement = max(min(shape) / 70, 3)
    displaced_flow = map_coordinates(
        gaussian_filter(flow, 0.9),
        (yy + displacement_y * displacement,
         xx + displacement_x * displacement),
        order=1, mode="reflect")
    gate = 1 / (1 + np.exp(-(vegetation_noise - 0.12) * 3.2))
    selected_flow = displaced_flow * gate
    vegetation_score = (1.75 * selected_flow + 0.72 * vegetation_noise -
                        0.8 * np.clip(slope - 0.55, 0, 2) -
                        0.55 * np.clip(height_unit, 0, 1))
    vegetation = 1 / (1 + np.exp(-(vegetation_score - 0.8) * 2.1))
    return (structure_orientation, coherence.astype(np.float32),
            feature_scale.astype(np.float32), contrast.astype(np.float32),
            lithology.astype(np.float32), vegetation.astype(np.float32))


def _axial_orientation(mask: np.ndarray) -> float:
    yy, xx = np.nonzero(mask)
    if len(xx) < 3:
        return 0.0
    points = np.column_stack((xx - xx.mean(), yy - yy.mean()))
    _, vectors = np.linalg.eigh(points.T @ points / len(points))
    vector = vectors[:, -1]
    return float(np.mod(np.arctan2(vector[1], vector[0]), np.pi))


def _even_sample(values: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    if len(values) <= count:
        return values
    indexes = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[int(i)] for i in indexes]


def build_structure_library(dataset: Path, excluded: set[tuple[int, int]],
                            max_tiles: int = 192, seed: int = 4401) -> StructureLibrary:
    """Learn terrain/material statistics and short boundary phrases.

    No assembled source map or source-space adjacency is stored.  Each phrase is
    an independently transformed crop, and its source identity is retained only
    so generation can limit reuse.
    """
    coordinates = [p for p in discover_level5(dataset) if p not in excluded]
    coordinates = _even_sample(coordinates, max_tiles)
    rng = np.random.default_rng(seed)
    feature_rows: list[np.ndarray] = []
    material_rows: list[np.ndarray] = []
    phrases: list[ShapePhrase] = []
    for index, (x, y) in enumerate(coordinates, 1):
        rgb, dem = tile_rgb(dataset, x, y), tile_dem(dataset, x, y)
        if rgb is None or dem is None:
            continue
        clean, confidence, _ = clean_source(rgb)
        labels = colour_labels(clean)
        curvature, roughness, orientation, features = terrain_features(*dem)
        valid = confidence >= 247
        flat = np.flatnonzero(valid)
        if len(flat):
            chosen = rng.choice(flat, min(1024, len(flat)), replace=False)
            feature_rows.append(features.reshape(-1, features.shape[-1])[chosen])
            material_rows.append(labels.ravel()[chosen])

        # Random crops retain local geological phrases but destroy the source
        # tile's larger sentence and all between-crop adjacency.
        for _ in range(8):
            extent = int(rng.integers(64, 161))
            oy = int(rng.integers(0, labels.shape[0] - extent + 1))
            ox = int(rng.integers(0, labels.shape[1] - extent + 1))
            crop = labels[oy:oy + extent, ox:ox + extent]
            for material in range(CLASS_COUNT):
                mask = crop == material
                occupancy = float(mask.mean())
                if not 0.12 <= occupancy <= 0.88:
                    continue
                # Canonical phrase scale is 8..24 layout pixels.  Later placement
                # adds another independent scale/aspect/elastic transform.
                target = int(np.clip(round(extent / 5), 12, 36))
                resized = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255, "L").resize(
                    (target, target), Image.Resampling.BILINEAR), np.float32) / 255.0
                binary = resized >= 0.5
                if binary.sum() >= 12:
                    phrases.append(ShapePhrase(material, resized, (x, y),
                                               _axial_orientation(binary), int(binary.sum())))
        if index % 32 == 0 or index == len(coordinates):
            print(f"  library {index}/{len(coordinates)} tiles; phrases={len(phrases)}", flush=True)
    if not feature_rows or not phrases:
        raise RuntimeError("structure library is empty")
    feature_array = np.concatenate(feature_rows).astype(np.float32)
    material_array = np.concatenate(material_rows).astype(np.uint8)
    return StructureLibrary(coordinates, feature_array, material_array, phrases,
                            cKDTree(feature_array))


def predict_probabilities(library: StructureLibrary, features: np.ndarray,
                          neighbours: int = 24) -> np.ndarray:
    flat = features.reshape(-1, features.shape[-1])
    distances, indexes = library.feature_tree.query(flat, k=min(neighbours, len(library.feature_rows)), workers=-1)
    distances = np.atleast_2d(distances)
    indexes = np.atleast_2d(indexes)
    if distances.shape[0] != len(flat):
        distances, indexes = distances.T, indexes.T
    weights = 1.0 / np.maximum(distances, 0.035) ** 2
    result = np.zeros((len(flat), CLASS_COUNT), np.float32)
    labels = library.material_rows[indexes]
    for material in range(CLASS_COUNT):
        result[:, material] = np.sum(weights * (labels == material), axis=1)
    result += 0.08 * weights.sum(axis=1, keepdims=True)
    result /= np.maximum(result.sum(axis=1, keepdims=True), 1.0e-8)
    result = result.reshape((*features.shape[:2], CLASS_COUNT))
    result = gaussian_filter(result, (3.5, 3.5, 0), mode="nearest")
    return result / np.maximum(result.sum(2, keepdims=True), 1.0e-8)


def correlated_noise(shape: tuple[int, int], seed: int, radii: tuple[float, ...] | None = None) -> np.ndarray:
    radii = radii or (max(min(shape) / 7, 12), max(min(shape) / 15, 6), max(min(shape) / 32, 3))
    weights = (0.55, 0.30, 0.15)
    rng = np.random.default_rng(seed)
    result = np.zeros(shape, np.float32)
    for weight, radius in zip(weights, radii):
        field = gaussian_filter(rng.standard_normal(shape), radius, mode="reflect")
        field = (field - field.mean()) / max(float(field.std()), 1.0e-6)
        result += weight * field
    return result


def oriented_macro_variation(orientation: np.ndarray, seed: int) -> np.ndarray:
    """Novel broad contrast elongated along the destination flow field.

    The phase comes entirely from the seed.  Destination geometry contributes
    direction but cannot prescribe where a light/dark geological phrase starts.
    """
    shape = orientation.shape
    broad = correlated_noise(shape, seed,
                             (max(min(shape) / 7, 12),
                              max(min(shape) / 16, 6),
                              max(min(shape) / 38, 3)))
    detail = correlated_noise(shape, seed + 4099,
                              (max(min(shape) / 20, 5),
                               max(min(shape) / 45, 2.5),
                               max(min(shape) / 90, 1.2)))
    yy, xx = np.indices(shape, dtype=np.float32)
    elongated = np.zeros(shape, np.float32)
    length = max(min(shape) / 24, 5)
    for offset in np.linspace(-length, length, 7):
        elongated += map_coordinates(
            detail,
            (yy + np.sin(orientation) * offset,
             xx + np.cos(orientation) * offset),
            order=1, mode="reflect")
    elongated /= 7
    elongated = (elongated - elongated.mean()) / max(float(elongated.std()), 1.0e-6)
    result = 0.68 * broad + 0.32 * elongated
    return (result - result.mean()) / max(float(result.std()), 1.0e-6)


def remove_small_components(labels_in: np.ndarray, minimum: int) -> np.ndarray:
    result = labels_in.copy()
    for material in range(CLASS_COUNT):
        components, count = label(result == material, structure=np.ones((3, 3), np.uint8))
        for component in range(1, count + 1):
            mask = components == component
            if int(mask.sum()) >= minimum:
                continue
            ring = gaussian_filter(mask.astype(np.float32), 1.5) > 0.01
            neighbours = result[ring & ~mask]
            if len(neighbours):
                result[mask] = int(np.bincount(neighbours, minlength=CLASS_COUNT).argmax())
    return result


def _paste_phrase(field: np.ndarray, phrase: ShapePhrase, centre: tuple[int, int],
                  target_angle: float, scale: float, aspect: float,
                  rng: np.random.Generator) -> np.ndarray:
    image = Image.fromarray(np.clip(phrase.mask * 255, 0, 255).astype(np.uint8), "L")
    width = max(5, int(round(image.width * scale * aspect)))
    height = max(5, int(round(image.height * scale / max(aspect, 1.0e-3))))
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    angle = np.degrees(target_angle - phrase.orientation + rng.normal(0, 0.32))
    image = image.rotate(float(angle), Image.Resampling.BILINEAR, expand=True)
    local = np.asarray(image, np.float32) / 255.0
    # A second smooth displacement prevents recognizable transformed cut-outs.
    if min(local.shape) > 5:
        yy, xx = np.indices(local.shape, dtype=np.float32)
        radius = max(min(local.shape) / 5, 2)
        dy = gaussian_filter(rng.normal(size=local.shape), radius)
        dx = gaussian_filter(rng.normal(size=local.shape), radius)
        dy *= 2.2 / max(float(dy.std()), 1.0e-5)
        dx *= 2.2 / max(float(dx.std()), 1.0e-5)
        local = map_coordinates(local, (yy + dy, xx + dx), order=1, mode="constant")
    cy, cx = centre
    y0, x0 = cy - local.shape[0] // 2, cx - local.shape[1] // 2
    y1, x1 = y0 + local.shape[0], x0 + local.shape[1]
    sy0, sx0 = max(0, -y0), max(0, -x0)
    sy1, sx1 = local.shape[0] - max(0, y1 - field.shape[0]), local.shape[1] - max(0, x1 - field.shape[1])
    dy0, dx0 = max(0, y0), max(0, x0)
    dy1, dx1 = dy0 + max(sy1 - sy0, 0), dx0 + max(sx1 - sx0, 0)
    field.fill(0)
    if dy1 > dy0 and dx1 > dx0:
        field[dy0:dy1, dx0:dx1] = local[sy0:sy1, sx0:sx1]
    return field


def place_shape_phrases(scores: np.ndarray, probabilities: np.ndarray,
                        orientation: np.ndarray, library: StructureLibrary,
                        seed: int, placements: int = 180) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    work = scores.copy()
    source_map = np.full(scores.shape[:2], -1, np.int32)
    scratch = np.zeros(scores.shape[:2], np.float32)
    phrase_by_material = [[p for p in library.phrases if p.material == material]
                          for material in range(CLASS_COUNT)]
    source_ids = {source: i for i, source in enumerate(library.coordinates)}
    recent: list[tuple[int, int]] = []
    audit: list[dict[str, object]] = []
    desired = probabilities.sum((0, 1))
    for _ in range(placements):
        current = work.argmax(2)
        actual = np.bincount(current.ravel(), minlength=CLASS_COUNT)
        deficit = (desired - actual) / np.maximum(desired, 64)
        # Mostly repair the largest deficit, with exploration to avoid a rigid
        # material-placement order.
        material = int(np.argmax(deficit + rng.normal(0, 0.035, CLASS_COUNT)))
        location = probabilities[..., material] * (0.25 + np.maximum(
            probabilities[..., material] - gaussian_filter((current == material).astype(np.float32), 14), 0))
        margin = np.ones(location.shape, np.float32)
        edge = min(20, max(2, min(location.shape) // 8))
        ramp = np.linspace(0.1, 1, edge)
        margin[:edge] *= ramp[:, None]; margin[-edge:] *= ramp[::-1, None]
        margin[:, :edge] *= ramp[None]; margin[:, -edge:] *= ramp[None, ::-1]
        location *= margin
        if float(location.sum()) <= 0:
            continue
        location /= location.sum()
        cy, cx = divmod(int(rng.choice(location.size, p=location.ravel())), location.shape[1])
        candidates = phrase_by_material[material]
        allowed = [p for p in candidates if p.source not in recent[-3:]] or candidates
        phrase = allowed[int(rng.integers(len(allowed)))]
        # Phrases follow either downhill or contour direction, sampled per
        # placement.  This preserves geological plausibility without source phase.
        target_angle = float(orientation[cy, cx] + (np.pi / 2 if rng.random() < 0.46 else 0))
        scale = float(rng.uniform(0.75, 2.25))
        aspect = float(rng.uniform(0.68, 1.48))
        mask = _paste_phrase(scratch, phrase, (cy, cx), target_angle, scale, aspect, rng)
        support = mask > 0.08
        if support.sum() < 10 or float(probabilities[..., material][support].mean()) < 0.12:
            continue
        strength = float(rng.uniform(1.1, 2.1))
        work[..., material] += strength * mask
        for other in range(CLASS_COUNT):
            if other != material:
                work[..., other] -= 0.30 * strength * mask
        source_map[mask > 0.52] = source_ids.get(phrase.source, -1)
        recent.append(phrase.source)
        audit.append({"material": material, "source": list(phrase.source),
                      "centre": [cx, cy], "scale": scale, "aspect": aspect,
                      "angle": target_angle})
    return work, source_map, audit


def trace_streaks(scores: np.ndarray, height: np.ndarray, slope: np.ndarray,
                  curvature: np.ndarray, probabilities: np.ndarray,
                  seed: int, count: int = 36) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Trace new downhill snow tongues, drainage marks, and scree gullies."""
    rng = np.random.default_rng(seed)
    work = scores.copy()
    h, w = height.shape
    smooth = gaussian_filter(height, 2.2, mode="nearest")
    dy, dx = np.gradient(smooth)
    magnitude = np.hypot(dx, dy)
    vx, vy = -dx / np.maximum(magnitude, 1.0e-5), -dy / np.maximum(magnitude, 1.0e-5)
    streak_kind = np.full((h, w), 255, np.uint8)
    audit: list[dict[str, object]] = []
    height_rank = (height - np.percentile(height, 5)) / max(float(np.percentile(height, 95) - np.percentile(height, 5)), 1)
    for index in range(count):
        snow = rng.random() < 0.42
        material = CLASS_SNOW if snow else CLASS_ROCK
        if snow:
            origin_weight = np.clip(height_rank, 0, 1) ** 2 * (0.25 + probabilities[..., CLASS_SNOW])
            origin_weight *= np.exp(-np.clip(slope, 0, 5) / 3.5)
        else:
            origin_weight = (0.12 + np.clip(slope, 0, 3)) * (0.2 + probabilities[..., CLASS_ROCK])
            origin_weight *= 0.5 + np.clip(-curvature / 12, 0, 1)
        origin_weight[:3] = 0; origin_weight[-3:] = 0; origin_weight[:, :3] = 0; origin_weight[:, -3:] = 0
        origin_weight /= max(float(origin_weight.sum()), 1.0e-8)
        y, x = divmod(int(rng.choice(h * w, p=origin_weight.ravel())), w)
        origin = (x, y)
        points = [(float(x), float(y))]
        momentum = np.array((vx[y, x], vy[y, x]), np.float32)
        length = int(rng.integers(max(12, min(h, w) // 18), max(20, min(h, w) // 5)))
        for _ in range(length):
            iy, ix = int(np.clip(round(y), 0, h - 1)), int(np.clip(round(x), 0, w - 1))
            downhill = np.array((vx[iy, ix], vy[iy, ix]), np.float32)
            wander = rng.normal(0, 0.17, 2).astype(np.float32)
            momentum = 0.76 * momentum + 0.24 * downhill + wander
            momentum /= max(float(np.linalg.norm(momentum)), 1.0e-5)
            x += float(momentum[0]); y += float(momentum[1])
            if x < 1 or y < 1 or x >= w - 1 or y >= h - 1:
                break
            points.append((x, y))
        if len(points) < 8:
            continue
        width = int(rng.integers(2, 7 if snow else 5))
        image = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(image)
        draw.line([(round(px), round(py)) for px, py in points], fill=255, width=width,
                  joint="curve")
        mask = gaussian_filter(np.asarray(image, np.float32) / 255.0, max(width * 0.45, 0.8))
        strength = float(rng.uniform(1.25, 2.25))
        work[..., material] += strength * mask
        for other in range(CLASS_COUNT):
            if other != material:
                work[..., other] -= 0.22 * strength * mask
        streak_kind[mask > 0.30] = material
        audit.append({"material": material, "origin": list(origin),
                      "end": [points[-1][0], points[-1][1]],
                      "length": len(points), "width": width})
    return work, streak_kind, audit


def generate_layout(height: np.ndarray, slope: np.ndarray, library: StructureLibrary,
                    seed: int = 1515, placements: int = 180,
                    streaks: int = 36) -> LayoutResult:
    """Generate a layout from DEM fields, a learned library, and a seed only."""
    curvature, roughness, orientation, features = terrain_features(height, slope)
    probabilities = predict_probabilities(library, features)
    scores = np.log(np.maximum(probabilities, 1.0e-5))
    for material in range(CLASS_COUNT):
        scores[..., material] += 0.82 * correlated_noise(height.shape, seed + material * 7919)
    base_labels = remove_small_components(scores.argmax(2).astype(np.uint8), max(12, height.size // 12000))
    scores, source_map, placement_audit = place_shape_phrases(
        scores, probabilities, orientation, library, seed + 101, placements)
    shape_labels = remove_small_components(scores.argmax(2).astype(np.uint8), max(10, height.size // 16000))
    scores, streak_kind, streak_audit = trace_streaks(
        scores, height, slope, curvature, probabilities, seed + 202, streaks)
    # Break snow edges according to terrain and fresh stochastic phase.  This
    # exposes convex/rough protrusions while retaining snow in local hollows,
    # avoiding a uniformly feathered segmentation boundary.
    preliminary = scores.argmax(2)
    snow = preliminary == CLASS_SNOW
    if snow.any() and (~snow).any():
        snow_sdf = distance_transform_edt(snow) - distance_transform_edt(~snow)
        edge_weight = np.exp(-np.abs(snow_sdf) / 6.0)
        snow_detail = correlated_noise(
            height.shape, seed + 303,
            (max(min(height.shape) / 34, 3),
             max(min(height.shape) / 70, 1.5), 0.8))
        retention = (0.50 * np.clip(-curvature / 10, -1, 1) -
                     0.32 * np.clip(slope - 0.65, -1, 2) -
                     0.22 * np.clip(roughness / 80, 0, 2) +
                     0.62 * snow_detail)
        scores[..., CLASS_SNOW] += edge_weight * retention
    labels = remove_small_components(scores.argmax(2).astype(np.uint8), max(8, height.size // 20000))
    flow = flow_accumulation(height)
    (structure_orientation, coherence, feature_scale, local_contrast,
     lithology, vegetation) = generated_structure_fields(
         height, slope, orientation, flow, seed + 404)
    metrics = {
        "seed": seed,
        "training_tiles": len(library.coordinates),
        "training_samples": int(len(library.feature_rows)),
        "shape_phrases": len(library.phrases),
        "accepted_placements": len(placement_audit),
        "streaks": len(streak_audit),
        "material_fraction": (np.bincount(labels.ravel(), minlength=CLASS_COUNT) / labels.size).tolist(),
        "placements": placement_audit,
        "streak_audit": streak_audit,
    }
    return LayoutResult(height, slope, curvature, roughness, orientation,
                        probabilities, base_labels, shape_labels, labels,
                        source_map, streak_kind, flow, structure_orientation,
                        coherence, feature_scale, local_contrast, lithology,
                        vegetation, metrics)


def _normalise(values: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    lo, hi = np.percentile(values, (low, high))
    return np.clip((values - lo) / max(float(hi - lo), 1.0e-6) * 255, 0, 255).astype(np.uint8)


def write_layout_checkpoints(output: Path, result: LayoutResult) -> None:
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_normalise(result.height), "L").save(output / "00_destination_height.png")
    Image.fromarray(_normalise(result.slope), "L").save(output / "00_destination_slope.png")
    Image.fromarray(_normalise(result.curvature), "L").save(output / "00_destination_curvature.png")
    flow = np.stack((np.cos(result.orientation) * .5 + .5,
                     np.sin(result.orientation) * .5 + .5,
                     np.full(result.orientation.shape, .5)), -1)
    Image.fromarray(np.clip(flow * 255, 0, 255).astype(np.uint8), "RGB").save(output / "00_destination_orientation.png")
    Image.fromarray(np.clip(result.probabilities @ CLASS_COLORS, 0, 255).astype(np.uint8), "RGB").save(output / "01_dem_material_probabilities.png")
    Image.fromarray(CLASS_COLORS[result.base_labels], "RGB").save(output / "02_seeded_base_layout.png")
    Image.fromarray(CLASS_COLORS[result.shape_labels], "RGB").save(output / "03_shape_phrase_layout.png")
    source = result.shape_source_map.astype(np.float32)
    source_rgb = np.zeros((*source.shape, 3), np.uint8)
    valid = source >= 0
    source_rgb[..., 0] = ((source * 73) % 255).astype(np.uint8)
    source_rgb[..., 1] = ((source * 151) % 255).astype(np.uint8)
    source_rgb[..., 2] = ((source * 211) % 255).astype(np.uint8)
    source_rgb[~valid] = 0
    Image.fromarray(source_rgb, "RGB").save(output / "03_shape_source_map.png")
    Image.fromarray(CLASS_COLORS[result.labels], "RGB").save(output / "04_streak_layout.png")
    streak_rgb = np.zeros((*result.streak_kind.shape, 3), np.uint8)
    valid = result.streak_kind != 255
    streak_rgb[valid] = CLASS_COLORS[result.streak_kind[valid]]
    Image.fromarray(streak_rgb, "RGB").save(output / "04_streak_paths.png")
    structure_rgb = np.stack((np.cos(result.structure_orientation) * .5 + .5,
                              np.sin(result.structure_orientation) * .5 + .5,
                              result.structure_coherence), -1)
    Image.fromarray(np.clip(structure_rgb * 255, 0, 255).astype(np.uint8), "RGB").save(
        output / "04_structure_orientation_coherence.png")
    Image.fromarray(np.clip(result.structure_coherence * 255, 0, 255).astype(np.uint8), "L").save(
        output / "04_structure_coherence.png")
    Image.fromarray(np.clip((result.feature_scale - 64) / 128 * 255, 0, 255).astype(np.uint8), "L").save(
        output / "04_structure_feature_scale.png")
    Image.fromarray(np.clip(result.local_contrast / 1.3 * 255, 0, 255).astype(np.uint8), "L").save(
        output / "04_structure_local_contrast.png")
    Image.fromarray(np.clip((result.lithology * .22 + .5) * 255, 0, 255).astype(np.uint8), "L").save(
        output / "04_structure_lithology.png")
    Image.fromarray(np.clip(result.flow_accumulation * 255, 0, 255).astype(np.uint8), "L").save(
        output / "04_flow_accumulation.png")
    Image.fromarray(np.clip(result.vegetation_density * 255, 0, 255).astype(np.uint8), "L").save(
        output / "04_vegetation_density.png")
    np.savez_compressed(output / "layout_fields.npz",
                        height=result.height, slope=result.slope,
                        curvature=result.curvature, roughness=result.roughness,
                        orientation=result.orientation,
                        probabilities=result.probabilities,
                        base_labels=result.base_labels,
                        shape_labels=result.shape_labels, labels=result.labels,
                        shape_source_map=result.shape_source_map,
                        streak_kind=result.streak_kind,
                        flow_accumulation=result.flow_accumulation,
                        structure_orientation=result.structure_orientation,
                        structure_coherence=result.structure_coherence,
                        feature_scale=result.feature_scale,
                        local_contrast=result.local_contrast,
                        lithology=result.lithology,
                        vegetation_density=result.vegetation_density)
    (output / "layout_metrics.json").write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "alps-data" / "trn-alps-16km")
    parser.add_argument("--tiles", type=int, nargs=4, default=(16, 32, 0, 16),
                        metavar=("X0", "X1", "Y0", "Y1"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/fullregen"))
    parser.add_argument("--tile-size", type=int, default=32,
                        help="layout pixels per 256px source tile")
    parser.add_argument("--training-tiles", type=int, default=192)
    parser.add_argument("--placements", type=int, default=180)
    parser.add_argument("--streaks", type=int, default=36)
    parser.add_argument("--seed", type=int, default=1515)
    args = parser.parse_args()
    x0, x1, y0, y1 = args.tiles
    excluded = {(x, y) for y in range(y0, y1) for x in range(x0, x1)}
    print("Building target-excluding structure library…", flush=True)
    library = build_structure_library(args.dataset, excluded, args.training_tiles, args.seed + 17)
    print("Assembling destination DEM only…", flush=True)
    height, slope = assemble_dem(args.dataset, tuple(args.tiles), args.tile_size)
    print("Generating novel semantic layout…", flush=True)
    result = generate_layout(height, slope, library, args.seed, args.placements, args.streaks)
    write_layout_checkpoints(args.output, result)
    print(f"wrote layout checkpoints to {args.output}", flush=True)


if __name__ == "__main__":
    main()
