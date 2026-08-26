#!/usr/bin/env python3
"""Install coordinate-labelled shadow-free Alpine imagery masters.

The 16 km dataset stores level-5 imagery as 256-pixel interiors with a
one-pixel neighbour gutter.  This tool maps the numbered edited master images
to the coordinate order produced by the swatch export, scales each master to
1024 square, archives it, and replaces just its corresponding 4 x 4 level-5
tile interiors.  Imagery outside those masters stays untouched.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image

from terrain_tiles import Key, _resize_srgb, _slice_image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "alps-data" / "trn-alps-16km"
COORDINATE_NAME = re.compile(
    r"^E(?P<east_min>\d+)-(?P<east_max>\d+)_N(?P<north_min>\d+)-(?P<north_max>\d+)\.png$")
MASTER_SIZE = 1024
TILE_SIZE = 256
GUTTER = 1
DATASET_EAST_MIN = 2639000
DATASET_NORTH_MAX = 1168000


def coordinate_masters(directory: Path) -> list[Path]:
    masters = sorted(directory.glob("E*_N*.png"))
    if not masters:
        raise ValueError(f"no coordinate-labelled PNG masters in {directory}")
    return masters


def numbered_masters(directory: Path, count: int) -> list[Path]:
    files = [directory / f"{index}.png" for index in range(1, count + 1)]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ValueError("missing numbered master images: " + ", ".join(missing))
    return files


def origin_from_name(path: Path) -> tuple[int, int]:
    match = COORDINATE_NAME.match(path.name)
    if not match:
        raise ValueError(f"invalid coordinate master name: {path.name}")
    east_min = int(match["east_min"])
    east_max = int(match["east_max"])
    north_min = int(match["north_min"])
    north_max = int(match["north_max"])
    if east_max - east_min != 2000 or north_max - north_min != 2000:
        raise ValueError(f"{path.name}: expected a 2 km by 2 km master")
    return ((east_min - DATASET_EAST_MIN) // 500,
            (DATASET_NORTH_MAX - north_max) // 500)


def source_interior(imagery_root: Path, tile_x: int, tile_y: int) -> Image.Image:
    """Return one unchanged 256-pixel tile interior, excluding its gutter."""
    with Image.open(imagery_root / str(tile_x) / f"{tile_y}.png") as image:
        rgb = image.convert("RGB")
        return rgb.crop((GUTTER, GUTTER, GUTTER + TILE_SIZE, GUTTER + TILE_SIZE))


def pixel_from_dataset(imagery_root: Path,
                       masters: dict[tuple[int, int], tuple[Image.Image, int, int]],
                       global_x: int, global_y: int,
                       cache: dict[tuple[int, int], Image.Image]):
    """Get a level-5 pixel, using the edited master where it covers the point."""
    master_tile = (global_x // TILE_SIZE, global_y // TILE_SIZE)
    if master_tile in masters:
        master, master_x0, master_y0 = masters[master_tile]
        return master.getpixel((global_x - master_x0 * TILE_SIZE,
                                global_y - master_y0 * TILE_SIZE))
    tile_x = max(0, min(31, global_x // TILE_SIZE))
    tile_y = max(0, min(31, global_y // TILE_SIZE))
    key = (tile_x, tile_y)
    if key not in cache:
        cache[key] = source_interior(imagery_root, *key)
    return cache[key].getpixel((global_x - tile_x * TILE_SIZE,
                                global_y - tile_y * TILE_SIZE))


def write_guttered_tile(source_imagery_root: Path, output_imagery_root: Path,
                        masters: dict[tuple[int, int], tuple[Image.Image, int, int]],
                        tile_x: int, tile_y: int,
                        source_cache: dict[tuple[int, int], Image.Image]) -> None:
    output = Image.new("RGB", (TILE_SIZE + 2 * GUTTER, TILE_SIZE + 2 * GUTTER))
    for local_y in range(-GUTTER, TILE_SIZE + GUTTER):
        global_y = max(0, min(31 * TILE_SIZE + TILE_SIZE - 1, tile_y * TILE_SIZE + local_y))
        for local_x in range(-GUTTER, TILE_SIZE + GUTTER):
            global_x = max(0, min(31 * TILE_SIZE + TILE_SIZE - 1, tile_x * TILE_SIZE + local_x))
            output.putpixel((local_x + GUTTER, local_y + GUTTER),
                            pixel_from_dataset(source_imagery_root, masters, global_x, global_y,
                                               source_cache))
    output_path = output_imagery_root / str(tile_x) / f"{tile_y}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)


def rebuild_ancestor_imagery(level5_root: Path) -> None:
    """Regenerate levels 0--4 from the completed level-5 imagery mosaic.

    Edited shadow-free masters replace level-5 interiors. Ancestors must be
    regenerated from those replacements too: retaining the original imagery at
    lower levels makes the quadtree switch between different colour datasets at
    an LOD boundary.
    """
    if level5_root.name != "5":
        raise ValueError(f"expected a level-5 imagery directory, got {level5_root}")
    finest_size = TILE_SIZE * 32
    finest = Image.new("RGB", (finest_size, finest_size))
    for tile_y in range(32):
        for tile_x in range(32):
            finest.paste(source_interior(level5_root, tile_x, tile_y),
                         (tile_x * TILE_SIZE, tile_y * TILE_SIZE))

    imagery_root = level5_root.parent
    for level in range(5):
        count = 1 << level
        level_image = _resize_srgb(finest, (TILE_SIZE * count, TILE_SIZE * count))
        for tile_y in range(count):
            for tile_x in range(count):
                tile = _slice_image(level_image, Key(level, tile_x, tile_y),
                                    TILE_SIZE, GUTTER)
                path = imagery_root / str(level) / str(tile_x) / f"{tile_y}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                tile.save(path)
        print(f"rebuilt imagery level {level} from level-5 shadow-free imagery")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--masters-dir", type=Path, default=Path.home() / "Downloads",
                        help="directory containing 1.png through N.png")
    parser.add_argument("--swatches-dir", type=Path,
                        default=Path("/tmp/alps-16km-texture-swatches"),
                        help="coordinate-labelled 1024-pixel swatches, defining the order")
    parser.add_argument("--base-dir", type=Path, default=None,
                        help="persistent directory for coordinate-labelled 1024-pixel masters")
    parser.add_argument("--source-imagery-root", type=Path, default=None,
                        help="level-5 imagery used for unchanged pixels and edge gutters")
    parser.add_argument("--output-imagery-root", type=Path, default=None,
                        help="level-5 imagery directory to update (defaults to the source root)")
    parser.add_argument("--rebuild-ancestors-only", action="store_true",
                        help="regenerate imagery levels 0--4 from the existing level-5 imagery")
    args = parser.parse_args()

    source_imagery_root = args.source_imagery_root or args.dataset / "imagery" / "5"
    output_imagery_root = args.output_imagery_root or source_imagery_root
    if not source_imagery_root.is_dir():
        raise ValueError(f"missing source level-5 imagery directory: {source_imagery_root}")

    if args.rebuild_ancestors_only:
        rebuild_ancestor_imagery(output_imagery_root)
        return

    coordinate_files = coordinate_masters(args.swatches_dir)
    edited_files = numbered_masters(args.masters_dir, len(coordinate_files))
    base_dir = args.base_dir or args.dataset / "shadowfree-masters"
    base_dir.mkdir(parents=True, exist_ok=True)

    masters: dict[tuple[int, int], tuple[Image.Image, int, int]] = {}
    master_details: list[tuple[Path, Path, int, int]] = []
    for coordinate_file, edited_file in zip(coordinate_files, edited_files):
        master_x0, master_y0 = origin_from_name(coordinate_file)
        with Image.open(edited_file) as image:
            master = image.convert("RGB").resize((MASTER_SIZE, MASTER_SIZE), Image.Resampling.LANCZOS)
        base_path = base_dir / coordinate_file.name
        master.save(base_path)
        for tile_y in range(master_y0, master_y0 + 4):
            for tile_x in range(master_x0, master_x0 + 4):
                if (tile_x, tile_y) in masters:
                    raise ValueError(f"overlapping masters at level-5 tile {tile_x}/{tile_y}")
                masters[(tile_x, tile_y)] = (master, master_x0, master_y0)
        master_details.append((edited_file, base_path, master_x0, master_y0))

    # Refresh the one-pixel gutter of adjacent original tiles as well.  Their
    # 256-pixel interiors remain untouched, while both sides of every boundary
    # agree when a shadow-free master meets regular red-highlighted imagery.
    tiles_to_write = set(masters)
    for tile_x, tile_y in masters:
        for delta_x, delta_y in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbour = (tile_x + delta_x, tile_y + delta_y)
            if 0 <= neighbour[0] < 32 and 0 <= neighbour[1] < 32:
                tiles_to_write.add(neighbour)

    source_cache: dict[tuple[int, int], Image.Image] = {}
    for tile_x, tile_y in sorted(tiles_to_write, key=lambda key: (key[1], key[0])):
        write_guttered_tile(source_imagery_root, output_imagery_root, masters,
                            tile_x, tile_y, source_cache)

    for edited_file, base_path, master_x0, master_y0 in master_details:
        print(f"{edited_file.name} -> {base_path.name} -> level-5 tiles {master_x0}:{master_x0 + 3}, {master_y0}:{master_y0 + 3}")
    rebuild_ancestor_imagery(output_imagery_root)


if __name__ == "__main__":
    main()
