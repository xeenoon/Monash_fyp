#!/usr/bin/env python3
"""Whole-map macro/micro SDF shape synthesis for procedural_gap_demo.

This module is intentionally independent of the demo's patch quilter.  It
streams the cleaned level-5 map into geometry-conditioned macro and micro shape
banks, then composes transformed signed-distance fields under local material
mass constraints.  Only macro low-frequency RGB is transformed with the shape;
native-resolution mid/high bands are added later by procedural_gap_demo.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Bump whenever _shape/_build or the classifier's semantics change, so stale
# on-disk shape banks are ignored instead of silently reused.
SHAPE_LIBRARY_CACHE_VERSION = 1

import numpy as np
from PIL import Image
from scipy.ndimage import (binary_dilation, distance_transform_edt, find_objects,
                           gaussian_filter, label, map_coordinates)
from scipy.spatial import cKDTree

from terrain_synth import clean_source, discover_level5, srgb_to_lab, tile_dem, tile_rgb

SIZE = 256
CLASS_ROCK, CLASS_GRASS, CLASS_SNOW = range(3)
CLASS_COUNT = 3
MACRO_MIN_AREA = 550
MACRO_MAX_AREA = 52000
MICRO_MIN_AREA = 18
MICRO_MAX_AREA = 2600
MACRO_ITERATIONS = 36
MICRO_ITERATIONS = 72
# A micro placement may only introduce a class that forms the local macro
# boundary, or one the target probability field genuinely calls for at that
# pixel.  Below this probability a third class is treated as absent, so a
# snow/rock corridor never sprouts grass just because it isn't the background.
THIRD_MATERIAL_THRESHOLD = 0.10
MACRO_SCORE_CANDIDATES = 24
MICRO_SCORE_CANDIDATES = 12
SOFT_FINALISTS = 16
MASS_REGION = 128
MASS_STRIDE = 64


@dataclass
class ShapeExemplar:
    material: int
    mask: np.ndarray
    source_x: int
    source_y: int
    bounds: tuple[int, int, int, int]
    area: float
    aspect_ratio: float
    orientation: float
    geometry: np.ndarray
    context: np.ndarray
    mean_lab: np.ndarray
    mean_rgb: np.ndarray
    confidence: float


@dataclass
class Placement:
    material: int
    source_x: int
    source_y: int
    centre_x: float
    centre_y: float
    scale: float
    angle: float
    aspect: float
    elastic: float
    mask: np.ndarray


@dataclass
class ShapeSynthesisResult:
    initial_labels: np.ndarray
    macro_labels: np.ndarray
    labels: np.ndarray
    macro_low_rgb: np.ndarray
    macro_source_map: np.ndarray
    micro_source_map: np.ndarray
    transition_density: np.ndarray
    desired_density: np.ndarray
    placements: list[Placement]
    metrics: dict[str, object]


def signed_distance(mask: np.ndarray) -> np.ndarray:
    return (distance_transform_edt(mask) - distance_transform_edt(~mask)).astype(np.float32)


def axial_orientation(mask: np.ndarray) -> float:
    yy, xx = np.nonzero(mask)
    if len(xx) < 3:
        return 0.0
    points = np.column_stack((xx - xx.mean(), yy - yy.mean()))
    covariance = points.T @ points / max(len(points), 1)
    values, vectors = np.linalg.eigh(covariance)
    vector = vectors[:, int(np.argmax(values))]
    return float(np.mod(np.arctan2(vector[1], vector[0]), np.pi))


def soft_choice(scores: np.ndarray, rng: np.random.Generator,
                temperature: float = 1.5, finalists: int = SOFT_FINALISTS) -> int:
    """Select softly from the best candidates; every supplied score matters."""
    scores = np.asarray(scores, np.float64)
    count = min(finalists, len(scores))
    best = np.argpartition(scores, count - 1)[:count]
    shifted = scores[best] - float(scores[best].min())
    weights = np.exp(-shifted / max(temperature, 1.0e-6))
    weights /= max(float(weights.sum()), 1.0e-12)
    return int(best[rng.choice(len(best), p=weights)])


def mass_error(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Global plus overlapping 128 px material-mass quota error."""
    error = 0.0
    origins = list(range(0, SIZE - MASS_REGION + 1, MASS_STRIDE))
    if origins[-1] != SIZE - MASS_REGION:
        origins.append(SIZE - MASS_REGION)
    regions = [(0, 0, SIZE)] + [(y, x, MASS_REGION) for y in origins for x in origins]
    for y, x, extent in regions:
        desired = probabilities[y:y + extent, x:x + extent].sum((0, 1))
        actual = np.bincount(labels[y:y + extent, x:x + extent].ravel(),
                             minlength=CLASS_COUNT).astype(np.float64)
        error += float(np.mean(((actual - desired) / np.maximum(desired, 96.0)) ** 2))
    return error / len(regions)


class WholeMapShapeLibrary:
    def __init__(self, dataset: Path, classifier: Callable[[np.ndarray], np.ndarray],
                 exclude: set[tuple[int, int]] = frozenset(),
                 max_tiles: int | None = None, use_cache: bool = True):
        self.dataset = dataset
        self.classifier = classifier
        coordinates = [p for p in discover_level5(dataset) if p not in exclude]
        if max_tiles is not None:
            coordinates = coordinates[:max_tiles]
        self.coordinates = coordinates
        self.macro: list[ShapeExemplar] = []
        self.micro: list[ShapeExemplar] = []
        self._geometry_rows: list[np.ndarray] = []
        self._density_rows: list[np.ndarray] = []
        self._sample_sources: list[tuple[int, int]] = []
        self._clean_cache: dict[tuple[int, int], np.ndarray] = {}
        self.selected_sources: list[tuple[int, int]] = []
        cache_path = self._cache_path(exclude, max_tiles) if use_cache else None
        if cache_path is not None and cache_path.exists():
            print(f"Loading cached shape bank from {cache_path.name}…", flush=True)
            self._load_cache(cache_path)
        else:
            self._build()
            if cache_path is not None:
                self._save_cache(cache_path)
        if not self.macro or not self.micro:
            raise RuntimeError("whole-map shape bank is empty")
        self.geometry = np.asarray(self._geometry_rows, np.float32)
        self.densities = np.asarray(self._density_rows, np.float32)
        self.geometry_tree = cKDTree(self.geometry)
        self._macro_by_material = [np.asarray([i for i, shape in enumerate(self.macro)
                                               if shape.material == kind], np.int32)
                                   for kind in range(CLASS_COUNT)]
        self._micro_by_material = [np.asarray([i for i, shape in enumerate(self.micro)
                                               if shape.material == kind], np.int32)
                                   for kind in range(CLASS_COUNT)]

    def _cache_path(self, exclude: set[tuple[int, int]],
                    max_tiles: int | None) -> Path:
        digest = hashlib.sha1(repr((
            str(self.dataset.resolve()), tuple(sorted(exclude)), max_tiles,
            len(self.coordinates), SHAPE_LIBRARY_CACHE_VERSION)).encode()).hexdigest()[:16]
        return self.dataset / ".shape_cache" / f"shape_bank_{digest}.pkl"

    _CACHE_ATTRS = ("coordinates", "macro", "micro", "_geometry_rows",
                    "_density_rows", "_sample_sources")

    def _save_cache(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {attr: getattr(self, attr) for attr in self._CACHE_ATTRS}
        payload["version"] = SHAPE_LIBRARY_CACHE_VERSION
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)  # atomic: never leave a half-written cache
        print(f"  cached shape bank to {path.name}", flush=True)

    def _load_cache(self, path: Path) -> None:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("version") != SHAPE_LIBRARY_CACHE_VERSION:
            self._build()
            return
        for attr in self._CACHE_ATTRS:
            setattr(self, attr, payload[attr])

    def clean_tile(self, x: int, y: int) -> np.ndarray:
        key = (x, y)
        if key not in self._clean_cache:
            rgb = tile_rgb(self.dataset, x, y)
            if rgb is None:
                raise FileNotFoundError(key)
            self._clean_cache[key] = clean_source(rgb)[0]
            # Hold enough distinct sources that the macro candidate loop (which
            # samples many tiles per iteration) stops thrashing and re-cleaning.
            if len(self._clean_cache) > 512:
                first = next(iter(self._clean_cache))
                if first != key:
                    self._clean_cache.pop(first)
        return self._clean_cache[key]

    def _shape(self, material: int, component: np.ndarray, source_x: int, source_y: int,
               bounds: tuple[int, int, int, int], height: np.ndarray, slope: np.ndarray,
               labels: np.ndarray, lab: np.ndarray, rgb: np.ndarray,
               confidence: np.ndarray) -> ShapeExemplar:
        y0, y1, x0, x1 = bounds
        support = component[y0:y1, x0:x1]
        area = float(support.sum())
        ys, xs = np.nonzero(component)
        selected_h, selected_s = height[component], slope[component]
        dy, dx = np.gradient(height)
        geometry = np.array((selected_h.mean() / 1200.0,
                             selected_h.std() / 350.0,
                             selected_s.mean() / 0.7,
                             selected_s.std() / 0.5,
                             np.mean(dx[component]) / 40.0,
                             np.mean(dy[component]) / 40.0), np.float32)
        context_mask = binary_dilation(component, iterations=8)
        context = np.bincount(labels[context_mask].ravel(), minlength=CLASS_COUNT).astype(np.float32)
        context /= max(float(context.sum()), 1.0)
        return ShapeExemplar(
            material, support.astype(bool), source_x, source_y, bounds, area,
            float((x1 - x0) / max(y1 - y0, 1)), axial_orientation(component), geometry,
            context, lab[component].mean(0).astype(np.float32),
            rgb[component].mean(0).astype(np.float32), float(confidence[component].mean() / 255.0))

    def _build(self) -> None:
        print(f"Building whole-map shape bank from {len(self.coordinates)} tiles…", flush=True)
        for tile_index, (source_x, source_y) in enumerate(self.coordinates, start=1):
            rgb = tile_rgb(self.dataset, source_x, source_y)
            dem = tile_dem(self.dataset, source_x, source_y)
            if rgb is None or dem is None:
                continue
            clean, confidence, _ = clean_source(rgb)
            labels = self.classifier(clean)
            height, slope = dem
            lab = srgb_to_lab(clean)
            # Geometry/material samples form the whole-map probability model.
            for y0 in range(0, SIZE, 64):
                for x0 in range(0, SIZE, 64):
                    h = height[y0:y0 + 64, x0:x0 + 64]
                    s = slope[y0:y0 + 64, x0:x0 + 64]
                    block_labels = labels[y0:y0 + 64, x0:x0 + 64]
                    self._geometry_rows.append(np.array((h.mean() / 1200.0, h.std() / 350.0,
                                                         s.mean() / 0.7, s.std() / 0.5), np.float32))
                    density = np.bincount(block_labels.ravel(), minlength=CLASS_COUNT).astype(np.float32)
                    self._density_rows.append(density / density.sum())
                    self._sample_sources.append((source_x, source_y))
            for material in range(CLASS_COUNT):
                components, count = label(labels == material)
                slices = find_objects(components)
                macro_candidates: list[tuple[float, ShapeExemplar]] = []
                micro_candidates: list[tuple[float, ShapeExemplar]] = []
                for component_id in range(1, count + 1):
                    sl = slices[component_id - 1]
                    if sl is None:
                        continue
                    component = components == component_id
                    area = int(component.sum())
                    y0, y1 = sl[0].start, sl[0].stop
                    x0, x1 = sl[1].start, sl[1].stop
                    height_px, width_px = y1 - y0, x1 - x0
                    if confidence[component].mean() < 247:
                        continue
                    bounds = (y0, y1, x0, x1)
                    if (MACRO_MIN_AREA <= area <= MACRO_MAX_AREA and
                            max(height_px, width_px) >= 30 and min(height_px, width_px) >= 8):
                        shape = self._shape(material, component, source_x, source_y, bounds,
                                            height, slope, labels, lab, clean, confidence)
                        # Prefer complete components over tile-clipped rectangles.
                        edge_hits = sum((y0 == 0, y1 == SIZE, x0 == 0, x1 == SIZE))
                        macro_candidates.append((area / (1 + 1.5 * edge_hits), shape))
                    if (MICRO_MIN_AREA <= area <= MICRO_MAX_AREA and
                            max(height_px, width_px) <= 72 and min(height_px, width_px) >= 2):
                        shape = self._shape(material, component, source_x, source_y, bounds,
                                            height, slope, labels, lab, clean, confidence)
                        entropy = -float(np.sum(shape.context * np.log(shape.context + 1.0e-6)))
                        micro_candidates.append((entropy + 0.001 * area, shape))
                macro_candidates.sort(key=lambda item: item[0], reverse=True)
                micro_candidates.sort(key=lambda item: item[0], reverse=True)
                self.macro.extend(shape for _, shape in macro_candidates[:3])
                self.micro.extend(shape for _, shape in micro_candidates[:8])
            if tile_index % 128 == 0 or tile_index == len(self.coordinates):
                print(f"  indexed {tile_index}/{len(self.coordinates)} tiles: "
                      f"macro={len(self.macro)}, micro={len(self.micro)}", flush=True)

    def predict_probabilities(self, height: np.ndarray, slope: np.ndarray) -> np.ndarray:
        grid = 16
        field = np.zeros((grid, grid, CLASS_COUNT), np.float32)
        block = SIZE // grid
        for gy in range(grid):
            for gx in range(grid):
                y0, x0 = gy * block, gx * block
                h, s = height[y0:y0 + block, x0:x0 + block], slope[y0:y0 + block, x0:x0 + block]
                descriptor = np.array((h.mean() / 1200.0, h.std() / 350.0,
                                       s.mean() / 0.7, s.std() / 0.5), np.float32)
                distance, nodes = self.geometry_tree.query(descriptor, k=min(64, len(self.geometry)))
                distance, nodes = np.atleast_1d(distance), np.atleast_1d(nodes)
                weights = 1.0 / np.maximum(distance, 0.025) ** 2
                field[gy, gx] = np.average(self.densities[nodes], axis=0, weights=weights)
        channels = [np.asarray(Image.fromarray(field[..., kind], "F").resize(
                    (SIZE, SIZE), Image.Resampling.BILINEAR)) for kind in range(CLASS_COUNT)]
        result = gaussian_filter(np.stack(channels, -1), (12.0, 12.0, 0), mode="nearest")
        return result / np.maximum(result.sum(2, keepdims=True), 1.0e-6)

    def _candidate_indexes(self, bank: str, material: int, geometry: np.ndarray,
                           active_source: tuple[int, int] | None,
                           rng: np.random.Generator, count: int) -> np.ndarray:
        shapes = self.macro if bank == "macro" else self.micro
        indexes = self._macro_by_material[material] if bank == "macro" else self._micro_by_material[material]
        if not len(indexes):
            return indexes
        geom = np.asarray([shapes[int(i)].geometry for i in indexes])
        descriptor_cost = np.mean((geom - geometry[None]) ** 2, axis=1)
        descriptor_count = min(max(1, round(count * 0.20)), len(indexes))
        descriptor = indexes[np.argpartition(descriptor_cost, descriptor_count - 1)[:descriptor_count]]
        local: list[int] = []
        if active_source is not None:
            for index in indexes:
                shape = shapes[int(index)]
                if np.hypot(shape.source_x - active_source[0], shape.source_y - active_source[1]) <= 2.25:
                    local.append(int(index))
        local_count = min(max(1, round(count * 0.70)), len(local))
        local_selected = (rng.choice(local, local_count, replace=False).astype(np.int32)
                          if local_count else np.empty(0, np.int32))
        global_count = min(max(1, count - len(local_selected) - len(descriptor)), len(indexes))
        global_selected = rng.choice(indexes, global_count, replace=False).astype(np.int32)
        return np.unique(np.concatenate((local_selected, descriptor, global_selected))).astype(np.int32)

    def texture_source_coordinates(self, target: tuple[int, int], count: int = 14) -> list[tuple[int, int]]:
        rng = np.random.default_rng(1515)
        selected: list[tuple[int, int]] = []
        # Roughly 70%: the most recently accepted micro/macro families and
        # their actual neighbours.  Micro families are appended after macro
        # placement, so walking backwards prevents an early run of one missing
        # macro material (often snow) from monopolising every RGB donor.
        families = list(dict.fromkeys(reversed(self.selected_sources)))
        for source in families:
            for dy, dx in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)):
                candidate = (source[0] + dx, source[1] + dy)
                if candidate in self.coordinates and candidate != target and candidate not in selected:
                    selected.append(candidate)
                if len(selected) >= round(count * 0.70):
                    break
            if len(selected) >= round(count * 0.70):
                break
        # 20% descriptor-near is represented by other shape placements.
        for source in families:
            if source != target and source not in selected:
                selected.append(source)
            if len(selected) >= round(count * 0.90):
                break
        # 10% true global exploration.
        candidates = [p for p in self.coordinates if p != target and p not in selected]
        if candidates:
            selected.extend(tuple(p) for p in rng.choice(candidates, min(count - len(selected), len(candidates)),
                                                         replace=False))
        return selected[:count]


def _geometry_at(height: np.ndarray, slope: np.ndarray, y: int, x: int,
                 radius: int = 48) -> np.ndarray:
    y0, y1 = max(0, y - radius), min(SIZE, y + radius)
    x0, x1 = max(0, x - radius), min(SIZE, x + radius)
    h, s = height[y0:y1, x0:x1], slope[y0:y1, x0:x1]
    dy, dx = np.gradient(h)
    return np.array((h.mean() / 1200.0, h.std() / 350.0,
                     s.mean() / 0.7, s.std() / 0.5,
                     dx.mean() / 40.0, dy.mean() / 40.0), np.float32)


def _transform(shape: ShapeExemplar, centre: tuple[int, int], target_size: float,
               angle: float, aspect: float, elastic: float,
               rng: np.random.Generator,
               source_rgb: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Aggressively transform geometry; RGB, when present, carries low band only."""
    mask_image = Image.fromarray(shape.mask.astype(np.uint8) * 255, "L")
    source_height, source_width = shape.mask.shape
    base = np.sqrt(max(source_height * source_width, 1))
    scale = target_size / base
    width = max(4, int(round(source_width * scale * aspect)))
    height = max(4, int(round(source_height * scale / max(aspect, 1.0e-3))))
    transformed_mask = mask_image.resize((width, height), Image.Resampling.BILINEAR).rotate(
        angle, resample=Image.Resampling.BILINEAR, expand=True)
    transformed_rgb = None
    if source_rgb is not None:
        y0, y1, x0, x1 = shape.bounds
        crop = source_rgb[y0:y1, x0:x1].astype(np.float32)
        # Normalized masked blur: only pixels belonging to this material shape
        # contribute to its broad colour.  A plain rectangular blur mixes the
        # surrounding classes in (e.g. grass around a snow shape), which reads as
        # snow sitting over a green underlayer once the low band is transformed.
        shape_mask = shape.mask.astype(np.float32)
        numerator = gaussian_filter(crop * shape_mask[..., None], (28.0, 28.0, 0))
        denominator = gaussian_filter(shape_mask, 28.0)[..., None]
        low = numerator / np.maximum(denominator, 1.0e-4)
        transformed_rgb = np.asarray(Image.fromarray(np.clip(low, 0, 255).astype(np.uint8), "RGB")
                                     .resize((width, height), Image.Resampling.BILINEAR)
                                     .rotate(angle, resample=Image.Resampling.BILINEAR, expand=True))
    local_mask = np.asarray(transformed_mask, np.float32) / 255.0
    if elastic > 0 and min(local_mask.shape) > 4:
        noise_y = gaussian_filter(rng.normal(size=local_mask.shape), max(min(local_mask.shape) / 7, 2))
        noise_x = gaussian_filter(rng.normal(size=local_mask.shape), max(min(local_mask.shape) / 7, 2))
        noise_y *= elastic / max(float(noise_y.std()), 1.0e-5)
        noise_x *= elastic / max(float(noise_x.std()), 1.0e-5)
        yy, xx = np.indices(local_mask.shape, dtype=np.float32)
        coords = (np.clip(yy + noise_y, 0, local_mask.shape[0] - 1),
                  np.clip(xx + noise_x, 0, local_mask.shape[1] - 1))
        local_mask = map_coordinates(local_mask, coords, order=1, mode="nearest")
        if transformed_rgb is not None:
            transformed_rgb = np.dstack([map_coordinates(transformed_rgb[..., channel], coords,
                                                          order=1, mode="nearest")
                                         for channel in range(3)]).astype(np.uint8)
    local_mask = local_mask >= 0.5
    canvas = np.zeros((SIZE, SIZE), bool)
    rgb_canvas = np.zeros((SIZE, SIZE, 3), np.uint8) if transformed_rgb is not None else None
    cy, cx = centre
    y0, x0 = int(round(cy - local_mask.shape[0] / 2)), int(round(cx - local_mask.shape[1] / 2))
    y1, x1 = y0 + local_mask.shape[0], x0 + local_mask.shape[1]
    sy0, sx0 = max(0, -y0), max(0, -x0)
    sy1, sx1 = local_mask.shape[0] - max(0, y1 - SIZE), local_mask.shape[1] - max(0, x1 - SIZE)
    dy0, dx0 = max(0, y0), max(0, x0)
    dy1, dx1 = dy0 + max(0, sy1 - sy0), dx0 + max(0, sx1 - sx0)
    if dy1 > dy0 and dx1 > dx0:
        canvas[dy0:dy1, dx0:dx1] = local_mask[sy0:sy1, sx0:sx1]
        if rgb_canvas is not None:
            rgb_canvas[dy0:dy1, dx0:dx1] = transformed_rgb[sy0:sy1, sx0:sx1]
    return canvas, rgb_canvas


def _calibrate_fields(fields: np.ndarray, probabilities: np.ndarray,
                      north: np.ndarray, east: np.ndarray, iterations: int = 16,
                      hard_edges: bool = True) -> np.ndarray:
    fields = fields.copy()
    desired_global = probabilities.mean((0, 1))
    for _ in range(iterations):
        labels = fields.argmax(0)
        actual = np.stack([gaussian_filter((labels == kind).astype(np.float32), 38.0,
                                           mode="nearest") for kind in range(CLASS_COUNT)])
        actual_global = np.bincount(labels.ravel(), minlength=CLASS_COUNT) / labels.size
        # Fields are measured in pixels of signed distance, so probability-unit
        # corrections below one were effectively inert.  Apply both local and
        # global quota errors in the same distance scale.
        fields += 1.25 * (np.moveaxis(probabilities, -1, 0) - actual)
        fields += (5.0 * (desired_global - actual_global))[:, None, None]
        fields -= fields.mean(0, keepdims=True)
        if hard_edges:
            # Exact known categorical boundary anchors.
            fields[:, 0, :] -= 2.0; fields[north[-1], 0, np.arange(SIZE)] += 6.0
            fields[:, :, -1] -= 2.0; fields[east[:, 0], np.arange(SIZE), -1] += 6.0
    return fields


def synthesize_macro_micro(library: WholeMapShapeLibrary, height: np.ndarray,
                           slope: np.ndarray, probabilities: np.ndarray,
                           north_labels: np.ndarray, east_labels: np.ndarray,
                           seed: int = 1515,
                           hard_edges: bool = True) -> ShapeSynthesisResult:
    rng = np.random.default_rng(seed)
    smooth = gaussian_filter(probabilities, (20.0, 20.0, 0), mode="nearest")
    initial = smooth.argmax(2).astype(np.uint8)
    fields = np.stack([signed_distance(initial == kind) for kind in range(CLASS_COUNT)])
    class_means = np.stack([np.mean([shape.mean_rgb for shape in library.macro
                                     if shape.material == kind], axis=0)
                            for kind in range(CLASS_COUNT)])
    macro_low = gaussian_filter(smooth @ class_means, (18.0, 18.0, 0))
    macro_source = np.full((SIZE, SIZE), -1, np.int32)
    micro_source = np.full((SIZE, SIZE), -1, np.int32)
    placements: list[Placement] = []
    active_source: tuple[int, int] | None = None

    for _ in range(MACRO_ITERATIONS):
        current = fields.argmax(0)
        actual = np.stack([gaussian_filter((current == kind).astype(np.float32), 44.0,
                                           mode="nearest") for kind in range(CLASS_COUNT)], -1)
        deficit = (probabilities - actual) / np.maximum(probabilities, 0.08)
        desired_mass = probabilities.sum((0, 1))
        actual_mass = np.bincount(current.ravel(), minlength=CLASS_COUNT)
        material = int(np.argmax((desired_mass - actual_mass) /
                                 np.maximum(desired_mass, 96.0)))
        # Sample a high-deficit interior location instead of repeatedly taking
        # the anchored corner maximum and clipping every transformed shape.
        location_weight = np.maximum(deficit[..., material], 0.0) + \
                          0.20 * probabilities[..., material]
        margin = np.ones((SIZE, SIZE), np.float32)
        margin[:24] *= np.linspace(0.12, 1.0, 24)[:, None]
        margin[-24:] *= np.linspace(1.0, 0.12, 24)[:, None]
        margin[:, :24] *= np.linspace(0.12, 1.0, 24)[None]
        margin[:, -24:] *= np.linspace(1.0, 0.12, 24)[None]
        location_weight *= margin
        location_weight /= max(float(location_weight.sum()), 1.0e-8)
        cy, cx = divmod(int(rng.choice(SIZE * SIZE, p=location_weight.ravel())), SIZE)
        geometry = _geometry_at(height, slope, cy, cx)
        indexes = library._candidate_indexes("macro", material, geometry, active_source,
                                             rng, MACRO_SCORE_CANDIDATES)
        if not len(indexes):
            continue
        candidate_data = []
        scores = []
        current_mass = mass_error(current, probabilities)
        for index in indexes:
            shape = library.macro[int(index)]
            scale = float(rng.uniform(0.65, 1.6))
            target_size = float(np.clip(np.sqrt(shape.area) * scale, 96, 224))
            angle = float(rng.uniform(0, 360))
            aspect = float(rng.uniform(0.80, 1.25))
            elastic = float(rng.uniform(3.0, 10.0))
            source_rgb = library.clean_tile(shape.source_x, shape.source_y)
            mask, low_rgb = _transform(shape, (cy, cx), target_size, angle, aspect,
                                       elastic, rng, source_rgb)
            if mask.sum() < 48:
                continue
            sdf = signed_distance(mask)
            trial_fields = fields.copy(); trial_fields[material] = np.maximum(trial_fields[material], sdf)
            trial = trial_fields.argmax(0)
            mass = mass_error(trial, probabilities)
            geometry_cost = float(np.mean((shape.geometry - geometry) ** 2))
            if active_source is None:
                source_cost = 0.0; colour_cost = 0.0
            else:
                distance = np.hypot(shape.source_x - active_source[0], shape.source_y - active_source[1])
                source_cost = 1.0 - np.exp(-distance / 2.0)
                active_shapes = [p for p in placements if (p.source_x, p.source_y) == active_source]
                active_material = active_shapes[-1].material if active_shapes else material
                reference = class_means[active_material]
                colour_cost = float(np.linalg.norm((shape.mean_rgb - reference) / 255.0))
            scores.append(14.0 * (mass - current_mass) + geometry_cost +
                          0.55 * source_cost + 0.25 * colour_cost + 0.15 * (1 - shape.confidence))
            candidate_data.append((shape, sdf, mask, low_rgb, scale, angle, aspect, elastic))
        if not candidate_data:
            continue
        choice = soft_choice(np.asarray(scores), rng, temperature=0.45)
        shape, sdf, mask, low_rgb, scale, angle, aspect, elastic = candidate_data[choice]
        fields[material] = np.maximum(fields[material], sdf)
        if low_rgb is not None:
            alpha = gaussian_filter(mask.astype(np.float32), 3.0)[..., None]
            macro_low = macro_low * (1 - alpha) + low_rgb.astype(np.float32) * alpha
        source_id = library.coordinates.index((shape.source_x, shape.source_y))
        macro_source[mask] = source_id
        active_source = (shape.source_x, shape.source_y)
        library.selected_sources.append(active_source)
        placements.append(Placement(material, shape.source_x, shape.source_y, cx, cy,
                                    scale, angle, aspect, elastic, mask))

    fields = _calibrate_fields(fields, probabilities, north_labels, east_labels, 32,
                               hard_edges=hard_edges)
    macro_labels = fields.argmax(0).astype(np.uint8)
    boundary = np.zeros((SIZE, SIZE), bool)
    boundary[:, 1:] |= macro_labels[:, 1:] != macro_labels[:, :-1]
    boundary[1:] |= macro_labels[1:] != macro_labels[:-1]
    distance_to_boundary = distance_transform_edt(~boundary)
    transition_density = np.exp(-distance_to_boundary / 13.0) + 0.08
    transition_density /= transition_density.sum()

    for _ in range(MICRO_ITERATIONS):
        flat = int(rng.choice(SIZE * SIZE, p=transition_density.ravel()))
        cy, cx = divmod(flat, SIZE)
        current = fields.argmax(0)
        background = int(current[cy, cx])
        actual = np.stack([gaussian_filter((current == kind).astype(np.float32), 24.0,
                                           mode="nearest") for kind in range(CLASS_COUNT)], -1)
        # Only the two classes that form this local macro boundary may be added,
        # plus any class the target probability field explicitly wants here.
        y0, y1 = max(cy - 12, 0), min(cy + 13, SIZE)
        x0, x1 = max(cx - 12, 0), min(cx + 13, SIZE)
        local_counts = np.bincount(macro_labels[y0:y1, x0:x1].ravel(), minlength=CLASS_COUNT)
        allowed = {int(kind) for kind in np.argsort(local_counts)[::-1][:2]
                   if local_counts[kind] > 0}
        allowed |= {kind for kind in range(CLASS_COUNT)
                    if probabilities[cy, cx, kind] > THIRD_MATERIAL_THRESHOLD}
        allowed.discard(background)
        if not allowed:
            continue
        desire = probabilities[cy, cx] - actual[cy, cx]
        material = max(allowed, key=lambda kind: float(desire[kind]))
        # Nothing to add if the background already meets the local quota for
        # every admissible class; leave the macro boundary untouched.
        if desire[material] <= 0.0:
            continue
        geometry = _geometry_at(height, slope, cy, cx, 24)
        active = library.selected_sources[-1] if library.selected_sources else None
        indexes = library._candidate_indexes("micro", material, geometry, active,
                                             rng, MICRO_SCORE_CANDIDATES)
        if not len(indexes):
            continue
        current_mass = mass_error(current, probabilities)
        scores, candidates = [], []
        for index in indexes:
            shape = library.micro[int(index)]
            size = float(rng.uniform(12, 64))
            angle = float(rng.uniform(0, 360))
            aspect = float(rng.uniform(0.72, 1.35))
            elastic = float(rng.uniform(1.0, 5.0))
            mask, _ = _transform(shape, (cy, cx), size, angle, aspect, elastic, rng)
            if mask.sum() < 5:
                continue
            sdf = signed_distance(mask)
            offset = float(np.max(fields[:, cy, cx]) - sdf[cy, cx] + 0.8)
            support = sdf > -2.0
            trial_fields = fields.copy()
            trial_fields[material, support] = np.maximum(
                trial_fields[material, support], sdf[support] + offset)
            trial = trial_fields.argmax(0)
            # A placement is only admissible if it does not worsen the material
            # quota; the pass proposes up to MICRO_SCORE_CANDIDATES edits and
            # accepts only the useful ones instead of forcing 72 additions.
            mass_delta = mass_error(trial, probabilities) - current_mass
            if mass_delta >= 0.0:
                continue
            context_cost = float((1.0 - shape.context[background]) +
                                 0.5 * (1.0 - shape.context[material]))
            scores.append(10.0 * mass_delta +
                          0.35 * context_cost + 0.2 * np.mean((shape.geometry - geometry) ** 2))
            candidates.append((shape, sdf, support, offset, mask))
        if not candidates:
            continue
        choice = soft_choice(np.asarray(scores), rng, temperature=0.65,
                             finalists=min(8, len(scores)))
        shape, sdf, support, offset, mask = candidates[choice]
        fields[material, support] = np.maximum(fields[material, support], sdf[support] + offset)
        micro_source[mask] = library.coordinates.index((shape.source_x, shape.source_y))
        library.selected_sources.append((shape.source_x, shape.source_y))

    fields = _calibrate_fields(fields, probabilities, north_labels, east_labels, 5,
                               hard_edges=hard_edges)
    labels = fields.argmax(0).astype(np.uint8)
    if hard_edges:
        labels[0] = north_labels[-1]
        labels[:, -1] = east_labels[:, 0]
    desired = probabilities.sum((0, 1))
    actual_macro = np.bincount(macro_labels.ravel(), minlength=CLASS_COUNT)
    actual_final = np.bincount(labels.ravel(), minlength=CLASS_COUNT)
    macro_low = np.clip(macro_low, 0, 255).astype(np.uint8)
    metrics = {
        "source_tiles": len(library.coordinates),
        "macro_exemplars": len(library.macro),
        "micro_exemplars": len(library.micro),
        "macro_placements": len(placements),
        "desired_mass": desired.tolist(),
        "macro_mass": actual_macro.tolist(),
        "final_mass": actual_final.tolist(),
        "macro_mass_error": mass_error(macro_labels, probabilities),
        "final_mass_error": mass_error(labels, probabilities),
        "selected_style_sources": [list(source) for source in
                                   dict.fromkeys(reversed(library.selected_sources))],
    }
    return ShapeSynthesisResult(initial, macro_labels, labels, macro_low, macro_source,
                                micro_source, transition_density.astype(np.float32),
                                probabilities, placements, metrics)
