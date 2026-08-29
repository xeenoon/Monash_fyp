#!/usr/bin/env python3
"""Install colour-graded AI shadow-removal masters as a new sibling dataset.

Builds alps-data/trn-alps-16km-graded/ next to alps-data/trn-alps-16km/: the
graded masters only change colour, not geometry or the quadtree, so tiles/
and manifest.json are symlinked from the source dataset rather than copied,
and only imagery/ is regenerated (the same source_interior/write_guttered_tile
/rebuild_ancestor_imagery machinery tools/integrate_shadowfree_alps.py uses to
install the original AI shadow-removed masters, reused here for the graded
set instead of duplicating it).

Usage:
    python3 tools/build_graded_alps_dataset.py --masters-dir /tmp/claude-ae54
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

from integrate_shadowfree_alps import (
    MASTER_SIZE,
    origin_from_name,
    rebuild_ancestor_imagery,
    write_guttered_tile,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = ROOT / "alps-data" / "trn-alps-16km"
DEFAULT_OUTPUT = ROOT / "alps-data" / "trn-alps-16km-graded"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masters-dir", type=Path, required=True,
                        help="directory of coordinate-named graded masters (E<...>_N<...>.png)")
    parser.add_argument("--source-dataset", type=Path, default=SOURCE_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    master_files = sorted(path for path in args.masters_dir.glob("E*_N*.png")
                          if not path.stem.endswith("_compare"))
    if not master_files:
        raise SystemExit(f"no coordinate-named masters (E<...>_N<...>.png) in {args.masters_dir}")

    output = args.output
    if output.exists():
        raise SystemExit(f"{output} already exists -- remove it first")
    output.mkdir(parents=True)

    (output / "tiles").symlink_to(args.source_dataset / "tiles", target_is_directory=True)
    (output / "manifest.json").symlink_to(args.source_dataset / "manifest.json")

    print(f"copying imagery/ ({args.source_dataset / 'imagery'} -> {output / 'imagery'}) ...")
    shutil.copytree(args.source_dataset / "imagery", output / "imagery")

    shutil.copytree(args.masters_dir, output / "graded-masters",
                     ignore=shutil.ignore_patterns("*_compare.png", "*.py", "__pycache__"))

    masters: dict[tuple[int, int], tuple[Image.Image, int, int]] = {}
    for path in master_files:
        master_x0, master_y0 = origin_from_name(path)
        with Image.open(path) as image:
            master = image.convert("RGB").resize((MASTER_SIZE, MASTER_SIZE), Image.Resampling.LANCZOS)
        for tile_y in range(master_y0, master_y0 + 4):
            for tile_x in range(master_x0, master_x0 + 4):
                if (tile_x, tile_y) in masters:
                    raise SystemExit(f"overlapping masters at level-5 tile {tile_x}/{tile_y}")
                masters[(tile_x, tile_y)] = (master, master_x0, master_y0)

    # Refresh the one-pixel gutter of adjacent tiles too, same reasoning as
    # integrate_shadowfree_alps.py: both sides of a boundary must agree.
    tiles_to_write = set(masters)
    for tile_x, tile_y in masters:
        for delta_x, delta_y in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbour = (tile_x + delta_x, tile_y + delta_y)
            if 0 <= neighbour[0] < 32 and 0 <= neighbour[1] < 32:
                tiles_to_write.add(neighbour)

    imagery_root = output / "imagery" / "5"
    source_cache: dict[tuple[int, int], Image.Image] = {}
    for tile_x, tile_y in sorted(tiles_to_write, key=lambda key: (key[1], key[0])):
        write_guttered_tile(imagery_root, imagery_root, masters, tile_x, tile_y, source_cache)

    rebuild_ancestor_imagery(imagery_root)
    print(f"built graded dataset at {output} "
          f"({len(master_files)} masters, {len(tiles_to_write)} level-5 tiles touched)")


if __name__ == "__main__":
    main()
