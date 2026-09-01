#!/usr/bin/env python3
"""Measure material-shape aggregation across a complete terrain tile grid.

The output deliberately separates raw connectivity from visual aggregation.
A one-pixel bridge can turn a whole material into one connected component while
still looking like a fragmented web.  The primary ``aggregation_score`` instead
measures variance in locally blurred class occupancy at several radii:

    A(sigma) = Var(G_sigma * class_mask) / (density * (1 - density))

Values near zero mean uniformly interspersed pixels; larger values mean the
material collects into coherent neighbourhoods at that scale.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, label
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from procedural_gap_demo import (  # noqa: E402
    CLASS_COLORS,
    CLASS_NAMES,
    GUTTER,
    SIZE,
    boundary_length,
    colour_labels,
    load_sample,
)

AGGREGATION_SIGMAS = (2, 4, 8, 16, 32)
AGGREGATION_WEIGHTS = np.asarray((1, 2, 3, 3, 2), dtype=np.float64)
COMPONENT_THRESHOLDS = (2, 8, 32, 128)
CORE_RADII = (1, 2, 4)


def disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return xx * xx + yy * yy <= radius * radius


def aggregation_curve(mask: np.ndarray) -> np.ndarray:
    """Return normalized local-density variance at each configured scale."""
    values = mask.astype(np.float32)
    density = float(values.mean())
    denominator = density * (1.0 - density)
    if denominator < 1.0e-6:
        return np.zeros(len(AGGREGATION_SIGMAS), dtype=np.float32)
    return np.asarray([
        gaussian_filter(values, sigma, mode="nearest").var() / denominator
        for sigma in AGGREGATION_SIGMAS
    ], dtype=np.float32)


def component_metrics(mask: np.ndarray) -> dict[str, float | int]:
    components, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(components.ravel())[1:].astype(np.float64)
    material_pixels = int(mask.sum())
    output: dict[str, float | int] = {
        "component_count": int(count),
        "component_density_per_4096px": float(count * 4096.0 / mask.size),
    }
    for threshold in COMPONENT_THRESHOLDS:
        output[f"components_ge_{threshold}px"] = int(np.count_nonzero(areas >= threshold))
    if material_pixels and areas.size:
        output["largest_component_fraction"] = float(areas.max() / material_pixels)
        output["effective_component_area"] = float(np.sum(areas * areas) / material_pixels)
    else:
        output["largest_component_fraction"] = 0.0
        output["effective_component_area"] = 0.0
    return output


def material_metrics(labels: np.ndarray, kind: int) -> dict[str, float | int]:
    mask = labels == kind
    density = float(mask.mean())
    result: dict[str, float | int] = {
        "density": density,
        "class_boundary_length": int(boundary_length(mask)),
        **component_metrics(mask),
    }
    curve = aggregation_curve(mask)
    for sigma, value in zip(AGGREGATION_SIGMAS, curve):
        result[f"aggregation_sigma_{sigma}"] = float(value)
    result["aggregation_score"] = float(np.average(curve, weights=AGGREGATION_WEIGHTS))

    distance = distance_transform_edt(mask)
    interior = distance[mask]
    result["thickness_q50"] = float(np.quantile(interior, 0.50)) if interior.size else 0.0
    result["thickness_q90"] = float(np.quantile(interior, 0.90)) if interior.size else 0.0
    result["thickness_q99"] = float(np.quantile(interior, 0.99)) if interior.size else 0.0
    for radius in CORE_RADII:
        core = binary_erosion(mask, structure=disk(radius))
        result[f"core_retention_r{radius}"] = float(core.sum() / max(int(mask.sum()), 1))
        core_components = component_metrics(core)
        result[f"core_effective_area_r{radius}"] = float(
            core_components["effective_component_area"])
    return result


def terrain_metrics(sample) -> dict[str, float]:
    height = sample.height
    slope = sample.slope
    return {
        "height_mean": float(height.mean()),
        "height_min": float(height.min()),
        "height_max": float(height.max()),
        "height_range": float(height.max() - height.min()),
        "height_std": float(height.std()),
        "slope_mean": float(slope.mean()),
        "slope_std": float(slope.std()),
        "slope_p90": float(np.quantile(slope, 0.90)),
    }


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def correlation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    predictors = ("height_mean", "height_range", "height_std", "slope_mean",
                  "slope_std", "slope_p90", "density")
    outcomes = ("aggregation_score", "effective_component_area",
                "largest_component_fraction", "thickness_q90",
                "component_density_per_4096px")
    output: list[dict[str, object]] = []
    for material in CLASS_NAMES:
        selected = [row for row in rows if row["material"] == material]
        for outcome in outcomes:
            y = np.asarray([float(row[outcome]) for row in selected])
            for predictor in predictors:
                x = np.asarray([float(row[predictor]) for row in selected])
                pearson = pearsonr(x, y).statistic if np.std(x) and np.std(y) else np.nan
                spearman = spearmanr(x, y).statistic if np.std(x) and np.std(y) else np.nan
                output.append({
                    "material": material,
                    "outcome": outcome,
                    "predictor": predictor,
                    "pearson_r": float(pearson),
                    "spearman_rho": float(spearman),
                    "tile_count": len(selected),
                })
    return output


def binned_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    features = ("height_mean", "height_range", "height_std", "slope_mean", "density")
    output: list[dict[str, object]] = []
    for material in CLASS_NAMES:
        selected = [row for row in rows if row["material"] == material]
        for feature in features:
            values = np.asarray([float(row[feature]) for row in selected])
            edges = np.quantile(values, np.linspace(0.0, 1.0, 6))
            for bin_index in range(5):
                low, high = float(edges[bin_index]), float(edges[bin_index + 1])
                include = ((values >= low) & (values <= high)) if bin_index == 4 else \
                    ((values >= low) & (values < high))
                group = [row for row, keep in zip(selected, include) if keep]
                if not group:
                    continue
                output.append({
                    "material": material,
                    "feature": feature,
                    "quintile": bin_index + 1,
                    "lower": low,
                    "upper": high,
                    "tile_count": len(group),
                    "mean_density": float(np.mean([float(row["density"]) for row in group])),
                    "mean_aggregation_score": float(np.mean([
                        float(row["aggregation_score"]) for row in group])),
                    "mean_effective_component_area": float(np.mean([
                        float(row["effective_component_area"]) for row in group])),
                    "mean_thickness_q90": float(np.mean([
                        float(row["thickness_q90"]) for row in group])),
                })
    return output


def heat_colour(value: float) -> tuple[int, int, int]:
    """Small blue/cyan/yellow perceptual-ish ramp without plotting dependencies."""
    value = float(np.clip(value, 0.0, 1.0))
    stops = ((26, 35, 126), (38, 139, 173), (83, 201, 120), (245, 225, 76))
    position = value * (len(stops) - 1)
    index = min(int(position), len(stops) - 2)
    fraction = position - index
    return tuple(round(stops[index][channel] * (1.0 - fraction) +
                       stops[index + 1][channel] * fraction) for channel in range(3))


def save_metric_atlas(path: Path, rows: list[dict[str, object]], grid_size: int) -> None:
    by_tile_material = {(int(row["x"]), int(row["y"]), str(row["material"])): row
                        for row in rows}
    panels = [
        ("mean elevation", None, "height_mean"),
        ("elevation range", None, "height_range"),
        ("mean slope", None, "slope_mean"),
        ("rock density", "rock", "density"),
        ("grass density", "grass", "density"),
        ("snow density", "snow", "density"),
        ("rock togetherness", "rock", "aggregation_score"),
        ("grass togetherness", "grass", "aggregation_score"),
        ("snow togetherness", "snow", "aggregation_score"),
    ]
    cell, title_height, gap = 12, 28, 18
    panel_size = grid_size * cell
    width = 3 * panel_size + 4 * gap
    height = 3 * (panel_size + title_height) + 4 * gap
    image = Image.new("RGB", (width, height), (18, 18, 20))
    draw = ImageDraw.Draw(image)
    for panel_index, (title, material, metric) in enumerate(panels):
        panel_x = gap + (panel_index % 3) * (panel_size + gap)
        panel_y = gap + (panel_index // 3) * (panel_size + title_height + gap)
        draw.text((panel_x, panel_y), title, fill="white")
        values = np.empty((grid_size, grid_size), dtype=np.float64)
        for y in range(grid_size):
            for x in range(grid_size):
                row = by_tile_material[(x, y, material or "rock")]
                values[y, x] = float(row[metric])
        low, high = float(np.quantile(values, 0.02)), float(np.quantile(values, 0.98))
        span = max(high - low, 1.0e-9)
        for y in range(grid_size):
            for x in range(grid_size):
                colour = heat_colour((values[y, x] - low) / span)
                x0 = panel_x + x * cell
                y0 = panel_y + title_height + y * cell
                draw.rectangle((x0, y0, x0 + cell - 1, y0 + cell - 1), fill=colour)
        draw.text((panel_x, panel_y + 13), f"{low:.3g} .. {high:.3g} (2-98%)", fill=(190, 190, 190))
    image.save(path)


def save_reference_layout(path: Path, mosaic: np.ndarray,
                          target_x: int, target_y: int, grid_size: int) -> None:
    """Save the north/target/east categorical view used by the gap demo."""
    if target_y <= 0 or target_x >= grid_size - 1:
        raise ValueError("reference target must have a north and east neighbour")

    def tile(x: int, y: int) -> np.ndarray:
        return mosaic[y * SIZE:(y + 1) * SIZE, x * SIZE:(x + 1) * SIZE]

    canvas = np.zeros((SIZE * 2, SIZE * 2, 3), dtype=np.uint8)
    canvas[:SIZE, :SIZE] = CLASS_COLORS[tile(target_x, target_y - 1)]
    canvas[SIZE:, :SIZE] = CLASS_COLORS[tile(target_x, target_y)]
    canvas[SIZE:, SIZE:] = CLASS_COLORS[tile(target_x + 1, target_y)]
    Image.fromarray(canvas, "RGB").save(path)


def write_summary(path: Path, rows: list[dict[str, object]], correlations: list[dict[str, object]],
                  grid_size: int) -> None:
    lines = [
        "# Full-map material togetherness analysis",
        "",
        f"Grid: {grid_size}×{grid_size} tiles; each tile is {SIZE}×{SIZE} pixels.",
        "",
        "`aggregation_score` is the weighted multiscale normalized variance of local material density ",
        "at sigma 2/4/8/16/32 px. Higher means the material forms coherent neighbourhoods rather ",
        "than a uniformly interspersed web. It is intentionally not equivalent to raw connectedness.",
        "",
        "## Strongest relationships",
        "",
    ]
    for material in CLASS_NAMES:
        selected = [row for row in correlations
                    if row["material"] == material and row["outcome"] == "aggregation_score" and
                    row["predictor"] != "density"]
        selected.sort(key=lambda row: abs(float(row["spearman_rho"])), reverse=True)
        lines.append(f"- {material}: " + ", ".join(
            f"{row['predictor']} rho={float(row['spearman_rho']):+.3f}"
            for row in selected[:3]))
    lines += [
        "",
        "## Files",
        "",
        "- `material_classes_32x32.png`: full-resolution categorical mosaic.",
        "- `tile_material_metrics.csv`: one row per tile and material.",
        "- `metric_correlations.csv`: Pearson and Spearman relationships.",
        "- `metric_quintiles.csv`: terrain/density-conditioned quintile summaries.",
        "- `tile_metric_atlas.png`: spatial overview of terrain, density, and togetherness.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "alps-data" / "trn-alps-16km")
    parser.add_argument("--terrain-dataset", type=Path, default=None,
                        help="dataset providing unchanged TRN geometry when imagery-only dataset is analysed")
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--output", type=Path,
                        default=Path("/tmp/terrain-material-analysis"))
    parser.add_argument("--reference-target", type=int, nargs=2, default=(23, 8),
                        metavar=("X", "Y"),
                        help="target coordinate for the north/target/east reference preview")
    args = parser.parse_args()
    if args.level != 5:
        raise ValueError("this analysis currently uses the level-5 loader")
    args.output.mkdir(parents=True, exist_ok=True)
    terrain_dataset = args.terrain_dataset or args.dataset

    grid_pixels = args.grid_size * SIZE
    mosaic = np.empty((grid_pixels, grid_pixels), dtype=np.uint8)
    rows: list[dict[str, object]] = []
    total = args.grid_size * args.grid_size
    for index, (y, x) in enumerate(((y, x) for y in range(args.grid_size)
                                    for x in range(args.grid_size)), start=1):
        sample = load_sample(terrain_dataset, x, y)
        if args.dataset.resolve() != terrain_dataset.resolve():
            imagery_path = args.dataset / "imagery" / str(args.level) / str(x) / f"{y}.png"
            with Image.open(imagery_path) as source:
                sample.rgb = np.asarray(source.convert("RGB"))[
                    GUTTER:-GUTTER, GUTTER:-GUTTER].copy()
        labels = colour_labels(sample.rgb)
        mosaic[y * SIZE:(y + 1) * SIZE, x * SIZE:(x + 1) * SIZE] = labels
        terrain = terrain_metrics(sample)
        categorical_boundary = int(boundary_length(labels))
        for kind, material in enumerate(CLASS_NAMES):
            rows.append({
                "x": x,
                "y": y,
                "material": material,
                **terrain,
                "categorical_boundary_length": categorical_boundary,
                **material_metrics(labels, kind),
            })
        if index == 1 or index % 32 == 0 or index == total:
            print(f"analysed {index}/{total} tiles", flush=True)

    print("writing full-resolution material mosaic", flush=True)
    Image.fromarray(CLASS_COLORS[mosaic], "RGB").save(
        args.output / f"material_classes_{args.grid_size}x{args.grid_size}.png")
    save_reference_layout(args.output / f"reference_{args.reference_target[0]}_{args.reference_target[1]}.png",
                          mosaic, *args.reference_target, args.grid_size)
    save_csv(args.output / "tile_material_metrics.csv", rows)
    correlations = correlation_rows(rows)
    save_csv(args.output / "metric_correlations.csv", correlations)
    save_csv(args.output / "metric_quintiles.csv", binned_rows(rows))
    save_metric_atlas(args.output / "tile_metric_atlas.png", rows, args.grid_size)
    write_summary(args.output / "README.md", rows, correlations, args.grid_size)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
