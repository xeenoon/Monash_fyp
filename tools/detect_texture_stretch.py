#!/usr/bin/env python3
"""Detect and highlight top-down texture stretching on a terrain tile.

Terrain imagery is a planar top-down (X/Z) projection: ``imagery_uv`` in
terrain.vert is a straight scale-bias of the interior grid coordinate,
independent of height (see shaders/terrain.vert). On a slope, that flat UV
footprint is stretched over more real 3-D surface area than the same
footprint on flat ground, so the texture visibly stretches along the slope
direction.

For a planar triangle with unit normal n (Y up), that stretch factor is
exactly ``1 / |n.y|``: projecting a tilted triangle straight down onto the
ground plane shrinks its area by a factor of ``|n.y|``, so covering the same
ground-plane/UV footprint takes ``1 / |n.y|`` times as much triangle surface,
stretching the texture flowing across it by that same factor. A vertical
face has ``n.y == 0`` and an unbounded stretch factor.

This tool rebuilds the interior mesh the same way terrain_grid.c
triangulates it, computes that ratio per triangle, and writes the tile's
imagery back out with a translucent red overlay on every triangle at or
above --threshold.

Usage:
    python3 tools/detect_texture_stretch.py trn/tiles/1/0/0.trn \\
        --output stretch.png --threshold 1.5 --opacity 0.25
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from terrain_tiles import Tile, read_tile  # noqa: E402

GridCoord = tuple[int, int]
Triangle = tuple[GridCoord, GridCoord, GridCoord]


def triangle_stretch_factors(tile: Tile) -> tuple[list[tuple[Triangle, float]], int]:
    """Per-interior-triangle top-down stretch factor, 1 / |normal.y|.

    Mirrors the (i01, i00, i11) / (i00, i10, i11) quad diagonal
    terrain_grid.c uses, restricted to the interior samples actually drawn
    -- the gutter ring exists only to seed elevation-texture normals (see
    docs/offline_terrain_tile_format.md) and is not part of the mesh.
    Triangles touching a no-data sample or degenerate (zero-area) are
    skipped.
    """
    g = tile.gutter
    segments = tile.width - 2 * g - 1
    if segments <= 0 or segments != tile.height - 2 * g - 1:
        raise ValueError(f"{tile.key}: interior grid must be a non-empty square")
    west, south, east, north = tile.extent
    dx = (east - west) / segments
    dz = (north - south) / segments
    heights = tile.decoded_heights()

    def valid(coord: GridCoord) -> bool:
        row, col = coord
        return tile.is_valid(row * tile.width + col)

    def position(coord: GridCoord) -> tuple[float, float, float]:
        row, col = coord
        return (col * dx, heights[row * tile.width + col], row * dz)

    triangles: list[tuple[Triangle, float]] = []
    for row in range(g, g + segments):
        for col in range(g, g + segments):
            i00, i01 = (row, col), (row + 1, col)
            i10, i11 = (row, col + 1), (row + 1, col + 1)
            for tri in ((i01, i00, i11), (i00, i10, i11)):
                if not all(valid(corner) for corner in tri):
                    continue
                p0, p1, p2 = (position(corner) for corner in tri)
                e1 = tuple(b - a for a, b in zip(p0, p1))
                e2 = tuple(b - a for a, b in zip(p0, p2))
                nx = e1[1] * e2[2] - e1[2] * e2[1]
                ny = e1[2] * e2[0] - e1[0] * e2[2]
                nz = e1[0] * e2[1] - e1[1] * e2[0]
                magnitude = math.sqrt(nx * nx + ny * ny + nz * nz)
                if magnitude < 1e-9:
                    continue
                cos_theta = abs(ny) / magnitude
                stretch = math.inf if cos_theta < 1e-6 else 1.0 / cos_theta
                triangles.append((tri, stretch))
    return triangles, segments


def load_imagery(dataset_root: Path, tile: Tile) -> Image.Image:
    if tile.imagery:
        raise ValueError(f"{tile.key}: inline imagery is not supported, only external imagery_uri tiles")
    image = Image.open(dataset_root / tile.imagery_uri)
    return image.convert("RGBA" if "A" in image.getbands() else "RGB").convert("RGBA")


def highlight_stretched_triangles(image: Image.Image, tile: Tile,
                                  triangles: list[tuple[Triangle, float]], segments: int,
                                  threshold: float, opacity: float) -> tuple[Image.Image, int]:
    """Composite a translucent red overlay over every triangle >= threshold.

    Triangle grid coordinates map onto imagery pixels the same way
    terrain_tiles.py's ``_slice_image`` built the raster: the interior
    [0, segments] samples land on the [gutter, size - gutter) pixel
    sub-rectangle, one-to-one with the gutter used for height samples.
    """
    g = tile.gutter
    img_w, img_h = image.size
    inner_w, inner_h = img_w - 2 * g, img_h - 2 * g
    if inner_w <= 0 or inner_h <= 0:
        raise ValueError(f"{tile.key}: imagery ({img_w}x{img_h}) too small for gutter {g}")

    def pixel(coord: GridCoord) -> tuple[float, float]:
        row, col = coord
        u = (col - g) / segments
        v = (row - g) / segments
        return (g + u * inner_w, g + v * inner_h)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    alpha = max(0, min(255, round(255 * opacity)))
    flagged = 0
    for tri, stretch in triangles:
        if stretch < threshold:
            continue
        flagged += 1
        draw.polygon([pixel(corner) for corner in tri], fill=(255, 0, 0, alpha))
    return Image.alpha_composite(image, overlay), flagged


def analyze(tile_path: Path, dataset_root: Path, threshold: float, opacity: float
           ) -> tuple[Image.Image, dict]:
    tile = read_tile(tile_path)
    triangles, segments = triangle_stretch_factors(tile)
    if not triangles:
        raise ValueError(f"{tile.key}: no valid interior triangles")
    image = load_imagery(dataset_root, tile)
    result, flagged = highlight_stretched_triangles(image, tile, triangles, segments,
                                                     threshold, opacity)
    finite = [stretch for _, stretch in triangles if math.isfinite(stretch)]
    stats = {
        "key": (tile.key.level, tile.key.x, tile.key.y),
        "triangle_count": len(triangles),
        "flagged_count": flagged,
        "flagged_fraction": flagged / len(triangles),
        "max_stretch": max(finite) if finite else math.inf,
        "vertical_count": len(triangles) - len(finite),
        "imagery_path": dataset_root / tile.imagery_uri,
    }
    return result, stats


def _format_summary(label: str, stats: dict, threshold: float) -> str:
    max_stretch = ("vertical" if math.isinf(stats["max_stretch"])
                   else f"{stats['max_stretch']:.2f}x")
    return (f"{label}: {stats['flagged_count']}/{stats['triangle_count']} triangles "
            f"({100 * stats['flagged_fraction']:.1f}%) stretched >= {threshold}x "
            f"(max {max_stretch}, {stats['vertical_count']} vertical)")


def _default_dataset_root(tile_path: Path) -> Path:
    # dataset layout is <root>/tiles/<level>/<x>/<y>.trn
    parents = tile_path.resolve().parents
    if len(parents) < 4 or parents[2].name != "tiles":
        raise ValueError(f"cannot infer dataset root from {tile_path}; pass --dataset-root")
    return parents[3]


def _run_dataset(dataset_root: Path, threshold: float, opacity: float, in_place: bool) -> int:
    tile_paths = sorted(dataset_root.glob("tiles/**/*.trn"))
    if not tile_paths:
        raise ValueError(f"no .trn tiles found under {dataset_root}/tiles")
    if not in_place:
        raise ValueError("processing a whole dataset requires --in-place "
                          "(pass a single .trn tile with --output for a one-off render)")
    total_triangles = total_flagged = 0
    for tile_path in tile_paths:
        result, stats = analyze(tile_path, dataset_root, threshold, opacity)
        result.convert(Image.open(stats["imagery_path"]).mode).save(stats["imagery_path"], "PNG")
        total_triangles += stats["triangle_count"]
        total_flagged += stats["flagged_count"]
        print(_format_summary(f"tile {'/'.join(map(str, stats['key']))}", stats, threshold))
    print(f"processed {len(tile_paths)} tiles: {total_flagged}/{total_triangles} triangles "
          f"({100 * total_flagged / total_triangles:.1f}%) stretched >= {threshold}x")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tile", type=Path,
                        help="path to a single .trn tile, or a dataset root directory "
                             "(containing tiles/) to process every tile in it")
    parser.add_argument("--dataset-root", type=Path,
                        help="dataset root imagery_uri is resolved against "
                             "(default: the tile argument itself in directory mode, else "
                             "inferred from <root>/tiles/L/X/Y.trn)")
    parser.add_argument("--output", type=Path,
                        help="output PNG (single-tile mode only)")
    parser.add_argument("--in-place", action="store_true",
                        help="overwrite each tile's own imagery file instead of --output; "
                             "required in directory mode")
    parser.add_argument("--threshold", type=float, default=1.5,
                        help="stretch factor at/above which a triangle is highlighted (default: 1.5)")
    parser.add_argument("--opacity", type=float, default=0.25,
                        help="red overlay opacity, 0-1 (default: 0.25)")
    try:
        args = parser.parse_args()
        if args.tile.is_dir():
            dataset_root = args.dataset_root or args.tile
            return _run_dataset(dataset_root, args.threshold, args.opacity, args.in_place)

        dataset_root = args.dataset_root or _default_dataset_root(args.tile)
        result, stats = analyze(args.tile, dataset_root, args.threshold, args.opacity)
        if args.in_place:
            output = stats["imagery_path"]
            result = result.convert(Image.open(output).mode)
        elif args.output:
            output = args.output
        else:
            raise ValueError("pass --output, or --in-place to overwrite the tile's own imagery")
        output.parent.mkdir(parents=True, exist_ok=True)
        result.save(output, "PNG")
        print(_format_summary(f"tile {'/'.join(map(str, stats['key']))}", stats, args.threshold))
        print(f"wrote {output}")
        return 0
    except (OSError, ValueError) as error:
        print(f"detect_texture_stretch: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
