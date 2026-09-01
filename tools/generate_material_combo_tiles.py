#!/usr/bin/env python3
"""Generate the five pass15 material showcase tiles in one atlas build.

The held-out target RGB is never sampled during generation.  Its DEM supplies
terrain geometry; all visible colour/detail comes from cleaned donor tiles.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import procedural_gap_demo as demo  # noqa: E402
from terrain_shape_synth import (CLASS_COUNT, SIZE, ShapeSynthesisResult,
                                 WholeMapShapeLibrary, mass_error,
                                 synthesize_macro_micro)  # noqa: E402


PRESETS = ("full_snow", "snow_rock", "full_rock", "rock_grass", "full_grass")
PURE_KIND = {
    "full_snow": demo.CLASS_SNOW,
    "full_rock": demo.CLASS_ROCK,
    "full_grass": demo.CLASS_GRASS,
}
TRANSITION_KINDS = {
    "snow_rock": (demo.CLASS_SNOW, demo.CLASS_ROCK),
    "rock_grass": (demo.CLASS_ROCK, demo.CLASS_GRASS),
}


def preset_probabilities(name: str, height: np.ndarray, seed: int) -> np.ndarray:
    """Return one-hot pure fields or a terrain-warped left/right transition."""
    if name in PURE_KIND:
        result = np.zeros((SIZE, SIZE, CLASS_COUNT), np.float32)
        result[..., PURE_KIND[name]] = 1.0
        return result
    left, right = TRANSITION_KINDS[name]
    rng = np.random.default_rng(seed)
    noise = gaussian_filter(rng.normal(size=(SIZE, SIZE)), 28.0, mode="reflect")
    noise /= max(float(noise.std()), 1.0e-5)
    terrain = gaussian_filter(height.astype(np.float32), 22.0, mode="nearest")
    terrain = (terrain - terrain.mean()) / max(float(terrain.std()), 1.0e-5)
    _yy, xx = np.indices((SIZE, SIZE), dtype=np.float32)
    signed = (xx - SIZE * 0.5 - 17.0 * noise - 8.0 * terrain) / 16.0
    blend = 1.0 / (1.0 + np.exp(-np.clip(signed, -12.0, 12.0)))
    result = np.full((SIZE, SIZE, CLASS_COUNT), 0.004, np.float32)
    result[..., left] += 0.988 * (1.0 - blend)
    result[..., right] += 0.988 * blend
    return result / result.sum(axis=2, keepdims=True)


def uniform_result(library: WholeMapShapeLibrary, kind: int,
                   probabilities: np.ndarray) -> ShapeSynthesisResult:
    """Represent a genuinely pure tile without injecting quota-zero micro islands."""
    labels = np.full((SIZE, SIZE), kind, np.uint8)
    class_rgb = np.mean([shape.mean_rgb for shape in library.macro
                         if shape.material == kind], axis=0)
    macro_low = np.broadcast_to(class_rgb, (SIZE, SIZE, 3)).astype(np.uint8).copy()
    source = np.full((SIZE, SIZE), -1, np.int32)
    transition = np.zeros((SIZE, SIZE), np.float32)
    metrics = {
        "source_tiles": len(library.coordinates),
        "macro_exemplars": len(library.macro),
        "micro_exemplars": len(library.micro),
        "macro_placements": 0,
        "desired_mass": probabilities.sum((0, 1)).tolist(),
        "macro_mass": np.bincount(labels.ravel(), minlength=CLASS_COUNT).tolist(),
        "final_mass": np.bincount(labels.ravel(), minlength=CLASS_COUNT).tolist(),
        "macro_mass_error": mass_error(labels, probabilities),
        "final_mass_error": mass_error(labels, probabilities),
        "selected_style_sources": [],
        "preset_is_exact_uniform": True,
    }
    return ShapeSynthesisResult(labels.copy(), labels.copy(), labels, macro_low,
                                source.copy(), source.copy(), transition,
                                probabilities, [], metrics)


def write_contact_sheet(output: Path) -> None:
    title_height = 28
    canvas = Image.new("RGB", (SIZE * len(PRESETS), SIZE + title_height), "black")
    draw = ImageDraw.Draw(canvas)
    labels = ("full snow", "snow → rock", "full rock", "rock → grass", "full grass")
    for column, (preset, title) in enumerate(zip(PRESETS, labels)):
        tile = Image.open(output / preset / "texture_reconstruction_target.png").convert("RGB")
        canvas.paste(tile, (column * SIZE, title_height))
        draw.text((column * SIZE + 7, 7), title, fill="white")
    canvas.save(output / "material_combo_contact_sheet.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "alps-data" / "trn-alps-16km")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, nargs=2, default=(23, 8))
    parser.add_argument("--west-donor", type=int, nargs=2, default=(24, 8))
    parser.add_argument("--north-donor", type=int, nargs=2, default=(23, 7))
    parser.add_argument("--shape-tile-limit", type=int,
                        help="optional development cap; omit for the full map")
    parser.add_argument("--texture-only", action="store_true",
                        help="reuse each preset's saved labels/metrics and rerender RGB only")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    target = demo.load_sample(args.dataset, *args.target)
    west = demo.load_sample(args.dataset, *args.west_donor)
    north = demo.load_sample(args.dataset, *args.north_donor)
    for donor in (west, north):
        donor.colour_class = demo.colour_labels(donor.rgb)

    print("Selecting whole-map high-purity material exemplar sources…", flush=True)
    material_exemplars = demo.load_material_exemplar_sources(
        args.dataset, exclude=frozenset({tuple(args.target)}))

    library = None if args.texture_only else WholeMapShapeLibrary(
        args.dataset, demo.colour_labels, exclude={tuple(args.target)},
        max_tiles=args.shape_tile_limit)
    if args.texture_only:
        for preset in PRESETS:
            destination = args.output / preset
            class_rgb = np.asarray(Image.open(
                destination / "constrained_patch_classes.png").convert("RGB"))
            difference = class_rgb[:, :, None, :].astype(np.int32) - \
                demo.CLASS_COLORS[None, None, :, :].astype(np.int32)
            labels = np.sum(difference * difference, axis=3).argmin(axis=2).astype(np.uint8)
            metrics = json.loads((destination / "macro_micro_metrics.json").read_text(
                encoding="utf-8"))
            preferred = [tuple(int(value) for value in source)
                         for source in metrics["texture_source_coordinates"]]
            texture_sources = demo.load_texture_sources(
                args.dataset, target.x, target.y, preferred)
            macro_low = np.asarray(Image.open(
                destination / "macro_low_frequency_filled.png").convert("RGB"))
            demo.write_texture_outputs(destination, target, west, north, labels,
                                       texture_sources, macro_low, snap_edges=False,
                                       material_exemplars=material_exemplars)
            demo.write_audit_outputs(destination, target, north, west, refined=True)
            Image.open(destination / "texture_reconstruction_target.png").save(
                args.output / f"{preset}.png")
        write_contact_sheet(args.output)
        print(f"rerendered five preset textures in {args.output}", flush=True)
        return

    assert library is not None
    manifest: dict[str, object] = {
        "target": list(args.target),
        "west_donor": list(args.west_donor),
        "north_donor": list(args.north_donor),
        "source_tiles": len(library.coordinates),
        "presets": {},
    }

    for preset_index, preset in enumerate(PRESETS):
        print(f"\n=== generating {preset} ===", flush=True)
        destination = args.output / preset
        destination.mkdir(parents=True, exist_ok=True)
        probabilities = preset_probabilities(preset, target.height, 1515 + preset_index * 101)
        library.selected_sources.clear()
        if preset in PURE_KIND:
            shape_result = uniform_result(library, PURE_KIND[preset], probabilities)
        else:
            shape_result = synthesize_macro_micro(
                library, target.height, target.slope, probabilities,
                north.colour_class, west.colour_class,
                seed=1515 + preset_index * 101, hard_edges=False)
        demo.write_macro_micro_diagnostics(destination, shape_result, north, west)

        preferred = library.texture_source_coordinates(tuple(args.target))
        metrics_path = destination / "macro_micro_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["preset"] = preset
        metrics["texture_source_coordinates"] = [
            [int(value) for value in source] for source in preferred]
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
        texture_sources = demo.load_texture_sources(
            args.dataset, target.x, target.y, preferred)
        macro_low = demo.compose_macro_low_background(
            shape_result.macro_low_rgb, shape_result.macro_source_map >= 0,
            shape_result.labels, texture_sources, seed=1515 + preset_index * 101)
        Image.fromarray(macro_low, "RGB").save(destination / "macro_low_frequency_filled.png")
        demo.write_outputs(destination, target, west, north, shape_result.labels,
                           probabilities, ROOT / "textures/runtime/terrain_micro_albedo.png",
                           texture_sources, macro_low, snap_edges=False,
                           material_exemplars=material_exemplars)
        demo.write_audit_outputs(destination, target, north, west, refined=True)
        Image.open(destination / "texture_reconstruction_target.png").save(
            args.output / f"{preset}.png")
        manifest["presets"][preset] = {
            "directory": preset,
            "final_tile": f"{preset}.png",
            "class_pixels": np.bincount(shape_result.labels.ravel(),
                                         minlength=CLASS_COUNT).tolist(),
            "texture_source_coordinates": metrics["texture_source_coordinates"],
        }

    write_contact_sheet(args.output)
    (args.output / "combo_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote five presets to {args.output}", flush=True)


if __name__ == "__main__":
    main()
