#!/usr/bin/env python3
"""Regenerate a map as novel structure plus real multiscale material detail.

The command writes every checkpoint to ``/tmp/fullregen`` by default:

  00 destination DEM fields (no RGB access)
  01 DEM-conditioned material probabilities
  02 seeded base material layout
  03 transformed short shape phrases
  04 newly traced gullies/streaks and final semantic layout
  05 three independently quilted material canvases
  06 macro RGB composition
  07 macro + mesostructure
  08 final native-resolution result
  09 reference-only audit and novelty/quality metrics

Destination imagery is excluded from every training and texture bank. It is
opened only after synthesis, in ``write_reference_audit``, so it cannot acquire
positional authority over the generated image.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from scipy.cluster.vq import kmeans2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from diagnose_macro_structure import compare_images  # noqa: E402
from procedural_gap_demo import (  # noqa: E402
    CLASS_COLORS, colour_labels,
)
from synthesize_macro_layout import (  # noqa: E402
    LayoutResult, StructureLibrary, assemble_dem, build_structure_library,
    correlated_noise, generate_layout, oriented_macro_variation,
    write_layout_checkpoints,
)
from terrain_synth import clean_source, tile_rgb  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
CLASS_COUNT = 3
CLASS_NAMES = ("rock", "grass", "snow")


def log(*values: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *values, flush=True)


@dataclass
class TextureBank:
    rgb: np.ndarray
    orientation: np.ndarray
    descriptors: np.ndarray
    state: np.ndarray
    state_centres: np.ndarray
    source_id: np.ndarray
    source_coordinates: list[tuple[int, int]]


def _fill_material(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        raise ValueError("cannot fill an absent material")
    if mask.all():
        return rgb
    # Nearest-neighbour filling creates radial Voronoi wedges around every
    # rejected island.  Normalized convolution instead extends only the broad
    # colour of valid material into holes; genuine native detail is retained
    # wherever the requested material was actually observed.
    values = rgb.astype(np.float32)
    mass = gaussian_filter(mask.astype(np.float32), 7.0, mode="nearest")
    numerator = gaussian_filter(values * mask[..., None], (7.0, 7.0, 0), mode="nearest")
    smooth = numerator / np.maximum(mass[..., None], 1.0e-4)
    result = values.copy()
    result[~mask] = smooth[~mask]
    # A crop can have a corner beyond the normalized filter's reach.
    missing = (~mask) & (mass < 1.0e-4)
    if missing.any():
        nearest = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        result[missing] = values[nearest[0][missing], nearest[1][missing]]
    return np.clip(result, 0, 255).astype(np.uint8)


def _structure_orientation(rgb: np.ndarray) -> float:
    luminance = rgb.astype(np.float32) @ np.array((.299, .587, .114), np.float32)
    gy, gx = np.gradient(gaussian_filter(luminance, 1.2))
    jxx, jyy = float(np.mean(gx * gx)), float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    return float(np.mod(0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2, np.pi))


def _patch_descriptor(rgb: np.ndarray) -> np.ndarray:
    """Colour, contrast, directional coherence, and frequency character."""
    values = rgb.astype(np.float32)
    luminance = values @ np.array((.299, .587, .114), np.float32)
    low = gaussian_filter(luminance, 12.0)
    fine = luminance - gaussian_filter(luminance, 2.0)
    gy, gx = np.gradient(gaussian_filter(luminance, 1.2))
    jxx, jyy = float(np.mean(gx * gx)), float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / max(jxx + jyy, 1.0e-6)
    return np.array((
        *(values.mean((0, 1)) / 255.0),
        luminance.std() / 64.0,
        np.hypot(gx, gy).mean() / 32.0,
        coherence,
        low.std() / 48.0,
        fine.std() / 32.0,
    ), np.float32)


def _cluster_states(descriptors: np.ndarray, requested: int,
                    seed: int) -> tuple[np.ndarray, np.ndarray]:
    count = min(requested, max(1, len(descriptors) // 5))
    if count == 1:
        return np.zeros(len(descriptors), np.uint8), descriptors.mean(0, keepdims=True)
    scale = np.maximum(descriptors.std(0), 0.08)
    normalized = (descriptors - descriptors.mean(0)) / scale
    centres_normalized, states = kmeans2(
        normalized, count, iter=30, minit="++", rng=np.random.default_rng(seed))
    centres = centres_normalized * scale + descriptors.mean(0)
    # Stable semantic order: dark states first, then brighter variants.
    luminance = centres[:, :3] @ np.array((.299, .587, .114), np.float32)
    order = np.argsort(luminance)
    inverse = np.empty_like(order); inverse[order] = np.arange(len(order))
    return inverse[states].astype(np.uint8), centres[order].astype(np.float32)


def build_texture_banks(library: StructureLibrary, dataset: Path, patch: int,
                        maximum: int = 320) -> dict[int, TextureBank]:
    """Build pure-material patch banks from target-excluding training tiles."""
    rows: dict[int, list[np.ndarray]] = {k: [] for k in range(CLASS_COUNT)}
    masks: dict[int, list[np.ndarray]] = {k: [] for k in range(CLASS_COUNT)}
    angles: dict[int, list[float]] = {k: [] for k in range(CLASS_COUNT)}
    ids: dict[int, list[int]] = {k: [] for k in range(CLASS_COUNT)}
    coordinates = library.coordinates
    stride = max(24, (256 - patch) // 2)
    for source_id, (x, y) in enumerate(coordinates):
        rgb = tile_rgb(dataset, x, y)
        if rgb is None:
            continue
        clean, confidence, _ = clean_source(rgb)
        labels = colour_labels(clean)
        for oy in range(0, 256 - patch + 1, stride):
            for ox in range(0, 256 - patch + 1, stride):
                crop = clean[oy:oy + patch, ox:ox + patch]
                valid = confidence[oy:oy + patch, ox:ox + patch] >= 247
                for material in range(CLASS_COUNT):
                    if len(rows[material]) >= maximum:
                        continue
                    mask = (labels[oy:oy + patch, ox:ox + patch] == material) & valid
                    if float(mask.mean()) < 0.62:
                        continue
                    rows[material].append(crop.copy())
                    masks[material].append(mask)
                    angles[material].append(0.0)  # calculated after same-material completion
                    ids[material].append(source_id)
        if all(len(rows[k]) >= maximum for k in range(CLASS_COUNT)):
            break
    banks: dict[int, TextureBank] = {}
    rng = np.random.default_rng(90210)
    for material in range(CLASS_COUNT):
        if not rows[material]:
            raise RuntimeError(f"no {CLASS_NAMES[material]} texture patches found")
        completed: list[np.ndarray] = []
        completed_angles: list[float] = []
        # Fill rejected parts with observed pixels of the same material from
        # unrelated crops.  This avoids both cross-material contamination and
        # the smooth/radial holes produced by nearest or broad inpainting.
        for base_index, (base, base_mask) in enumerate(zip(rows[material], masks[material])):
            result = base.copy()
            covered = base_mask.copy()
            order = rng.permutation(len(rows[material]))
            for donor_index in order:
                if donor_index == base_index:
                    continue
                donor_rgb = rows[material][int(donor_index)]
                donor_mask = masks[material][int(donor_index)]
                # Source masks often occupy similar crop coordinates (snow in
                # an upper corner, grass below a ridge). Random translation
                # removes that positional bias and lets observed same-material
                # detail cover the whole destination motif.
                shift = tuple(int(value) for value in rng.integers(-patch // 2,
                                                                   patch // 2 + 1, 2))
                donor_rgb = np.roll(donor_rgb, shift, axis=(0, 1))
                donor_mask = np.roll(donor_mask, shift, axis=(0, 1))
                add = (~covered) & donor_mask
                if add.any():
                    result[add] = donor_rgb[add]
                    covered |= add
                if covered.all():
                    break
            if not covered.all():
                result = _fill_material(result, covered)
            completed.append(result)
            completed_angles.append(_structure_orientation(result))
        descriptor_array = np.asarray([_patch_descriptor(row) for row in completed], np.float32)
        requested_states = (8, 6, 6)[material]
        states, state_centres = _cluster_states(
            descriptor_array, requested_states, 7100 + material)
        banks[material] = TextureBank(np.asarray(completed, np.uint8),
                                      np.asarray(completed_angles, np.float32),
                                      descriptor_array, states, state_centres,
                                      np.asarray(ids[material], np.int32), coordinates)
        log(f"{CLASS_NAMES[material]} bank: {len(rows[material])} patches from "
            f"{len(set(ids[material]))} source tiles, {len(state_centres)} latent states")
    return banks


def _vseam(error: np.ndarray) -> np.ndarray:
    """True on the new-patch side of a minimum-cost vertical seam."""
    height, width = error.shape
    cost = error.astype(np.float32).copy()
    back = np.zeros((height, width), np.int16)
    for y in range(1, height):
        previous = cost[y - 1]
        for x in range(width):
            lo, hi = max(0, x - 1), min(width, x + 2)
            parent = lo + int(np.argmin(previous[lo:hi]))
            back[y, x] = parent
            cost[y, x] += previous[parent]
    result = np.zeros((height, width), bool)
    x = int(np.argmin(cost[-1]))
    for y in range(height - 1, -1, -1):
        result[y, x + 1:] = True
        x = int(back[y, x])
    return result


def _axis_origins(extent: int, patch: int, step: int) -> list[int]:
    values = list(range(0, max(extent - patch + 1, 1), step))
    values.append(max(extent - patch, 0))
    return sorted(set(values))


def generate_material_state_fields(layout: LayoutResult,
                                   banks: dict[int, TextureBank],
                                   seed: int, output: Path
                                   ) -> dict[int, np.ndarray]:
    """Generate coherent latent subclasses without source-space coordinates."""
    fields: dict[int, np.ndarray] = {}
    preview = np.zeros((*layout.labels.shape, 3), np.uint8)
    legend: dict[str, object] = {}
    colour_rng = np.random.default_rng(3317)
    offset = 0
    for material in range(CLASS_COUNT):
        centres = banks[material].state_centres
        state_scores = []
        luma = centres[:, :3] @ np.array((.299, .587, .114), np.float32)
        luma_z = (luma - luma.mean()) / max(float(luma.std()), .05)
        contrast = centres[:, 3]
        median_contrast = max(float(np.median(contrast)), .05)
        for state in range(len(centres)):
            phase = correlated_noise(
                layout.labels.shape, seed + material * 1009 + state * 97,
                (max(min(layout.labels.shape) / 9, 10),
                 max(min(layout.labels.shape) / 22, 4),
                 max(min(layout.labels.shape) / 55, 2)))
            score = 0.82 * phase
            score += 0.75 * luma_z[state] * layout.lithology
            score -= 1.05 * np.abs(centres[state, 5] - layout.structure_coherence)
            desired_contrast = median_contrast * layout.local_contrast
            score -= 0.55 * np.abs(centres[state, 3] - desired_contrast)
            state_scores.append(score)
        state_field = np.stack(state_scores).argmax(0).astype(np.uint8)
        fields[material] = state_field
        colours = colour_rng.integers(35, 245, (len(centres), 3), dtype=np.uint8)
        selected = layout.labels == material
        preview[selected] = colours[state_field[selected]]
        legend[CLASS_NAMES[material]] = [
            {"state": int(index), "mean_rgb": (centre[:3] * 255).tolist(),
             "luma_std": float(centre[3] * 64),
             "gradient_energy": float(centre[4] * 32),
             "coherence": float(centre[5]),
             "low_band_std": float(centre[6] * 48),
             "fine_band_std": float(centre[7] * 32)}
            for index, centre in enumerate(centres)]
        offset += len(centres)
    Image.fromarray(preview, "RGB").save(output / "05_material_state_field.png")
    np.savez_compressed(output / "05_material_state_fields.npz",
                        **{CLASS_NAMES[k]: value for k, value in fields.items()})
    (output / "05_material_state_legend.json").write_text(
        json.dumps(legend, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return fields


def _transform_motif(rgb: np.ndarray, angle: float, scale: float) -> np.ndarray:
    """Rotate/scale a square motif with reflected sampling and no empty corners."""
    size = rgb.shape[0]
    yy, xx = np.indices((size, size), dtype=np.float32)
    centre = (size - 1) / 2
    dx, dy = (xx - centre) / scale, (yy - centre) / scale
    cosine, sine = np.cos(angle), np.sin(angle)
    source_x = cosine * dx + sine * dy + centre
    source_y = -sine * dx + cosine * dy + centre
    channels = [map_coordinates(rgb[..., channel].astype(np.float32),
                                (source_y, source_x), order=1, mode="reflect")
                for channel in range(3)]
    return np.clip(np.stack(channels, -1), 0, 255).astype(np.uint8)


def quilt_material(bank: TextureBank, height: int, width: int, patch: int,
                   overlap: int, target_state: np.ndarray,
                   target_orientation: np.ndarray,
                   target_coherence: np.ndarray,
                   target_feature_scale: np.ndarray,
                   target_contrast: np.ndarray, seed: int,
                   candidates: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Quilt one material canvas with orientation and source-reuse control."""
    if height < patch or width < patch:
        raise ValueError("output must be at least one patch")
    rng = np.random.default_rng(seed)
    canvas = np.zeros((height, width, 3), np.uint8)
    owner = np.full((height, width), -1, np.int16)
    known = np.zeros((height, width), bool)
    step = patch - overlap
    ys, xs = _axis_origins(height, patch, step), _axis_origins(width, patch, step)
    recent_sources: list[int] = []
    for row_index, oy in enumerate(ys):
        for ox in xs:
            ty = min(target_orientation.shape[0] - 1,
                     int((oy + patch / 2) / height * target_orientation.shape[0]))
            tx = min(target_orientation.shape[1] - 1,
                     int((ox + patch / 2) / width * target_orientation.shape[1]))
            wanted_state = int(target_state[ty, tx])
            pool = np.flatnonzero(bank.state == wanted_state)
            if not len(pool):
                pool = np.arange(len(bank.rgb))
            count = min(candidates, len(pool))
            indexes = rng.choice(pool, count, replace=False)
            coherence = float(target_coherence[ty, tx])
            wanted_contrast = float(np.median(bank.descriptors[:, 3]) * target_contrast[ty, tx])
            costs = 260.0 * np.abs(bank.descriptors[indexes, 5] - coherence)
            costs += 150.0 * np.abs(bank.descriptors[indexes, 3] - wanted_contrast)
            source_ids = bank.source_id[indexes]
            if recent_sources:
                costs += 180.0 * np.isin(source_ids, recent_sources[-3:])
            # Rotate every candidate into the newly generated formation field;
            # scale maps the learned phrase into a 64..192 px destination motif.
            angle = float(target_orientation[ty, tx]) - bank.orientation[indexes]
            scale = float(np.clip(target_feature_scale[ty, tx] / max(patch, 1), .52, 1.52))
            transformed = np.asarray([
                _transform_motif(bank.rgb[int(index)], float(rotation), scale)
                for index, rotation in zip(indexes, angle)], np.uint8)
            patches = transformed.astype(np.float32)
            if ox > 0:
                existing = canvas[oy:oy + patch, ox:ox + overlap].astype(np.float32)
                costs += np.mean((patches[:, :, :overlap] - existing) ** 2, axis=(1, 2, 3))
            if oy > 0:
                existing = canvas[oy:oy + overlap, ox:ox + patch].astype(np.float32)
                costs += np.mean((patches[:, :overlap] - existing) ** 2, axis=(1, 2, 3))
            finalists = np.argsort(costs)[:min(3, len(costs))]
            shifted = costs[finalists] - costs[finalists[0]]
            weights = np.exp(-shifted / max(float(np.std(shifted)), 40.0))
            weights /= weights.sum()
            chosen_local = int(rng.choice(finalists, p=weights))
            chosen = int(indexes[chosen_local])
            candidate = transformed[chosen_local]
            take = np.ones((patch, patch), bool)
            if ox > 0:
                error = np.mean((canvas[oy:oy + patch, ox:ox + overlap].astype(np.float32) -
                                 candidate[:, :overlap].astype(np.float32)) ** 2, axis=2)
                take[:, :overlap] = _vseam(error)
            if oy > 0:
                error = np.mean((canvas[oy:oy + overlap, ox:ox + patch].astype(np.float32) -
                                 candidate[:overlap].astype(np.float32)) ** 2, axis=2)
                take[:overlap] &= _vseam(error.T).T
            region = canvas[oy:oy + patch, ox:ox + patch]
            region[take] = candidate[take]
            owner_region = owner[oy:oy + patch, ox:ox + patch]
            owner_region[take] = bank.source_id[chosen]
            known[oy:oy + patch, ox:ox + patch] |= take
            recent_sources.append(int(bank.source_id[chosen]))
        if row_index % 6 == 0 or row_index == len(ys) - 1:
            log(f"    quilt row {row_index + 1}/{len(ys)}")
    if not known.all():
        nearest = distance_transform_edt(~known, return_distances=False, return_indices=True)
        canvas[~known] = canvas[nearest[0][~known], nearest[1][~known]]
        owner[~known] = owner[nearest[0][~known], nearest[1][~known]]
    return canvas, owner


def _resize_rgb(array: np.ndarray, size: tuple[int, int],
                resampling=Image.Resampling.BILINEAR) -> np.ndarray:
    return np.asarray(Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB").resize(size, resampling))


def _resize_float_channel(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(array.astype(np.float32), "F").resize(
        size, Image.Resampling.BILINEAR), np.float32)


def _soft_material_weights(labels: np.ndarray) -> np.ndarray:
    weights = np.stack([gaussian_filter((labels == material).astype(np.float32), 1.35,
                                        mode="nearest") for material in range(CLASS_COUNT)], -1)
    return weights / np.maximum(weights.sum(2, keepdims=True), 1.0e-6)


def compose_frequency_checkpoints(output: Path, layout: LayoutResult,
                                  material_paths: list[Path],
                                  final_size: tuple[int, int],
                                  banks: dict[int, TextureBank],
                                  state_fields: dict[int, np.ndarray],
                                  seed: int) -> tuple[Path, Path, Path]:
    """Compose independently phased macro, meso, and micro donor bands."""
    width, height = final_size
    labels = layout.labels
    weights_small = _soft_material_weights(labels)
    weights = np.stack([_resize_float_channel(weights_small[..., material], (width, height))
                        for material in range(CLASS_COUNT)], -1)
    weights /= np.maximum(weights.sum(2, keepdims=True), 1.0e-6)
    macro = np.zeros((height, width, 3), np.float32)
    meso_delta = np.zeros_like(macro)
    micro_delta = np.zeros_like(macro)
    layout_size = (labels.shape[1], labels.shape[0])
    quarter_size = (max(1, width // 4), max(1, height // 4))
    for material, path in enumerate(material_paths):
        canvas = np.asarray(Image.open(path).convert("RGB"), np.float32)
        macro_small = _resize_rgb(canvas, layout_size, Image.Resampling.BOX)
        macro_full = _resize_rgb(macro_small, (width, height)).astype(np.float32)
        quarter = _resize_rgb(canvas, quarter_size, Image.Resampling.BOX)
        quarter_full = _resize_rgb(quarter, (width, height)).astype(np.float32)
        weight = weights[..., material, None]
        macro += weight * macro_full
        meso_delta += weight * (quarter_full - macro_full)
        micro_delta += weight * (canvas - quarter_full)
    # A smoothly varying grade target is shared by whole state neighborhoods.
    # This removes block exposure jumps without normalizing away a pale slab or
    # dark vegetation family every 32 pixels.
    state_macro_small = np.zeros((*labels.shape, 3), np.float32)
    for material in range(CLASS_COUNT):
        centres = banks[material].state_centres[:, :3] * 255
        material_mean = np.median(banks[material].descriptors[:, :3] * 255, axis=0)
        colours = .74 * centres + .26 * material_mean[None]
        raw = colours[state_fields[material]]
        support = (labels == material).astype(np.float32)
        mass = gaussian_filter(support, 1.8, mode="nearest")
        numerator = gaussian_filter(raw * support[..., None], (1.8, 1.8, 0), mode="nearest")
        smooth = numerator / np.maximum(mass[..., None], 1.0e-5)
        state_macro_small += weights_small[..., material, None] * smooth
    state_macro = _resize_rgb(state_macro_small, (width, height)).astype(np.float32)
    macro = .72 * macro + .28 * state_macro
    Image.fromarray(np.clip(state_macro_small, 0, 255).astype(np.uint8), "RGB").save(
        output / "06_material_state_grade.png")
    variation_small = oriented_macro_variation(layout.structure_orientation, seed + 909)
    variation = _resize_float_channel(variation_small, (width, height))
    coherence = _resize_float_channel(layout.structure_coherence, (width, height))
    macro *= np.clip(1.0 + (0.055 + 0.045 * coherence[..., None]) * variation[..., None],
                     0.72, 1.30)
    Image.fromarray(np.clip((variation_small * .22 + .5) * 255, 0, 255).astype(np.uint8),
                    "L").save(output / "06_synthetic_macro_variation.png")
    # Generated broad state layers.  They deliberately retain meaningful
    # material-internal contrast instead of grading every patch to one mean.
    lithology = _resize_float_channel(layout.lithology, (width, height))
    vegetation = _resize_float_channel(layout.vegetation_density, (width, height))
    grass_weight = weights[..., 1]
    rock_weight = weights[..., 0]
    snow_weight = weights[..., 2]
    vegetation_darkening = np.clip(0.44 * vegetation * grass_weight, 0, .44)
    macro *= (1.0 - vegetation_darkening[..., None])
    # Rock lithology moves coherently between brown-grey, neutral, and blue-grey.
    rock_shift = np.clip(lithology * rock_weight, -1.8, 1.8)
    macro[..., 0] *= np.clip(1.0 + .075 * rock_shift, .82, 1.18)
    macro[..., 1] *= np.clip(1.0 + .015 * rock_shift, .88, 1.12)
    macro[..., 2] *= np.clip(1.0 - .070 * rock_shift, .82, 1.18)
    dirty_snow = _resize_float_channel(
        np.clip(-layout.lithology * .45, 0, 1), (width, height))
    macro *= 1.0 - (0.16 * np.maximum(dirty_snow, 0) * snow_weight)[..., None]
    Image.fromarray(np.clip(vegetation_darkening * 255 / .44, 0, 255).astype(np.uint8),
                    "L").save(output / "06_vegetation_darkening.png")
    macro_u8 = np.clip(np.rint(macro), 0, 255).astype(np.uint8)
    meso_u8 = np.clip(np.rint(macro + meso_delta), 0, 255).astype(np.uint8)
    final_u8 = np.clip(np.rint(macro + meso_delta + micro_delta), 0, 255).astype(np.uint8)
    macro_path = output / "06_macro_rgb.png"
    meso_path = output / "07_macro_plus_mesostructure.png"
    final_path = output / "08_final_regeneration.png"
    Image.fromarray(macro_u8, "RGB").save(macro_path)
    Image.fromarray(meso_u8, "RGB").save(meso_path)
    Image.fromarray(final_u8, "RGB").save(final_path)
    return macro_path, meso_path, final_path


def assemble_reference(dataset: Path,
                       bounds: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Reference-only audit helper. Never call before generation is complete."""
    x0, x1, y0, y1 = bounds
    reference = np.zeros(((y1 - y0) * 256, (x1 - x0) * 256, 3), np.uint8)
    for j, y in enumerate(range(y0, y1)):
        for i, x in enumerate(range(x0, x1)):
            rgb = tile_rgb(dataset, x, y)
            if rgb is None:
                raise FileNotFoundError(f"missing audit imagery {x}/{y}")
            reference[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = rgb
    return reference, colour_labels(reference)


def write_reference_audit(output: Path, dataset: Path,
                          bounds: tuple[int, int, int, int], final_path: Path,
                          generated_labels: np.ndarray) -> dict[str, object]:
    log("Synthesis complete; opening destination imagery for the separate audit…")
    reference, reference_labels = assemble_reference(dataset, bounds)
    reference_path = output / "09_audit_reference.png"
    labels_path = output / "09_audit_reference_labels.png"
    generated_labels_path = output / "09_audit_generated_labels.png"
    Image.fromarray(reference, "RGB").save(reference_path)
    Image.fromarray(CLASS_COLORS[reference_labels], "RGB").save(labels_path)
    Image.fromarray(CLASS_COLORS[generated_labels], "RGB").save(generated_labels_path)
    metrics = compare_images(final_path, reference_path, generated_labels_path, labels_path)
    (output / "09_quality_novelty_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def write_pyramids(output: Path, paths: list[Path]) -> None:
    directory = output / "pyramids"
    directory.mkdir(parents=True, exist_ok=True)
    for path in paths:
        image = Image.open(path).convert("RGB")
        for size in (128, 64, 32, 16):
            image.resize((size, size), Image.Resampling.BOX).resize(
                (512, 512), Image.Resampling.NEAREST).save(
                    directory / f"{path.stem}_{size}.png")


def write_checkpoint_readme(output: Path) -> None:
    (output / "README.md").write_text(
        "# Full terrain regeneration checkpoints\n\n"
        "The destination imagery is excluded from all training and donor banks. It is opened "
        "only for checkpoint 09, after checkpoint 08 already exists.\n\n"
        "- `00_*`: destination DEM derivatives; no RGB input.\n"
        "- `01_dem_material_probabilities.png`: learned DEM-conditioned material prior.\n"
        "- `02_seeded_base_layout.png`: new correlated-phase grass/rock/snow topology.\n"
        "- `03_*`: transformed short SDF shape phrases and target-space source audit.\n"
        "- `04_streak_*`: new downhill gullies/snow tongues and final categorical layout.\n"
        "- `04_structure_*`: generated orientation, coherence, feature scale, contrast, "
        "lithology, drainage proposal, and selected vegetation density.\n"
        "- `05_material_state_*`: 8 rock, 6 grass, and 6 snow latent-state fields and descriptors.\n"
        "- `05_material_{rock,grass,snow}.png`: independently transformed/quilted motif canvases.\n"
        "- `06_macro_rgb.png`: new macro composition and smoothly varying neighborhood grades.\n"
        "- `07_macro_plus_mesostructure.png`: 16–128 px donor structure restored.\n"
        "- `08_final_regeneration.png`: native donor residual detail restored.\n"
        "- `09_audit_*`: reference data loaded after synthesis plus quality/novelty metrics.\n"
        "- `pyramids/`: 128/64/32/16 px distance-view comparisons.\n\n"
        "The acceptance target is high distributional structure with low shifted spatial "
        "correlation. See `09_quality_novelty_metrics.json` and `diagnostic_cli_metrics.json`.\n",
        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "alps-data" / "trn-alps-16km")
    parser.add_argument("--tiles", type=int, nargs=4, default=(16, 32, 0, 16),
                        metavar=("X0", "X1", "Y0", "Y1"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/fullregen"))
    parser.add_argument("--seed", type=int, default=1515)
    parser.add_argument("--layout-tile-size", type=int, default=32)
    parser.add_argument("--training-tiles", type=int, default=192)
    parser.add_argument("--placements", type=int, default=180)
    parser.add_argument("--streaks", type=int, default=36)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--bank-size", type=int, default=320)
    parser.add_argument("--candidates", type=int, default=14)
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()
    bounds = tuple(args.tiles)
    x0, x1, y0, y1 = bounds
    excluded = {(x, y) for y in range(y0, y1) for x in range(x0, x1)}
    args.output.mkdir(parents=True, exist_ok=True)

    log("Checkpoint 0: building target-excluding statistical/shape library")
    library = build_structure_library(args.dataset, excluded, args.training_tiles, args.seed + 17)
    log("Checkpoint 0: reading destination DEM only")
    height, slope = assemble_dem(args.dataset, bounds, args.layout_tile_size)
    log("Checkpoints 1–4: generating probabilities, topology, phrases, and streaks")
    layout = generate_layout(height, slope, library, args.seed, args.placements, args.streaks)
    write_layout_checkpoints(args.output, layout)

    final_width, final_height = (x1 - x0) * 256, (y1 - y0) * 256
    log("Checkpoint 5: building target-excluding texture banks")
    banks = build_texture_banks(library, args.dataset, args.patch, args.bank_size)
    state_fields = generate_material_state_fields(layout, banks, args.seed + 707,
                                                  args.output)
    material_paths: list[Path] = []
    source_use: dict[str, object] = {}
    for material in range(CLASS_COUNT):
        log(f"Checkpoint 5: quilting {CLASS_NAMES[material]}")
        canvas, owner = quilt_material(
            banks[material], final_height, final_width, args.patch, args.overlap,
            state_fields[material], layout.structure_orientation,
            layout.structure_coherence, layout.feature_scale,
            layout.local_contrast,
            args.seed + 1009 * (material + 1), args.candidates)
        path = args.output / f"05_material_{CLASS_NAMES[material]}.png"
        Image.fromarray(canvas, "RGB").save(path)
        material_paths.append(path)
        used, counts = np.unique(owner[owner >= 0], return_counts=True)
        source_use[CLASS_NAMES[material]] = [
            {"source": list(banks[material].source_coordinates[int(source_id)]),
             "pixels": int(count)} for source_id, count in zip(used, counts)]
        owner_preview = ((owner.astype(np.int32) * 67) % 255).astype(np.uint8)
        Image.fromarray(owner_preview, "L").resize(
            (layout.labels.shape[1], layout.labels.shape[0]), Image.Resampling.NEAREST).save(
                args.output / f"05_material_{CLASS_NAMES[material]}_source_map.png")
    (args.output / "05_texture_source_use.json").write_text(
        json.dumps(source_use, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    log("Checkpoints 6–8: composing independent macro, meso, and micro bands")
    macro_path, meso_path, final_path = compose_frequency_checkpoints(
        args.output, layout, material_paths, (final_width, final_height),
        banks, state_fields, args.seed)
    pyramid_inputs = [macro_path, meso_path, final_path]
    if not args.skip_audit:
        metrics = write_reference_audit(args.output, args.dataset, bounds,
                                        final_path, layout.labels)
        pyramid_inputs.append(args.output / "09_audit_reference.png")
        log("audit:", json.dumps(metrics["summary"], sort_keys=True))
    write_pyramids(args.output, pyramid_inputs)
    write_checkpoint_readme(args.output)
    log(f"all checkpoints written to {args.output}")


if __name__ == "__main__":
    main()
