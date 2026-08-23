#!/usr/bin/env python3
"""Build strict natural-terrain masks from official Swiss vector geometry.

The output is an 8-bit PNG aligned pixel-for-pixel with every source imagery
COG: 255 means natural terrain, 0 means infrastructure.  It is intentionally
conservative.  Roads, rails, buildings, and optional infrastructure begin as
vector geometry, are rasterised, then dilated in metres.  Dilation is performed
after rasterisation so the same rule works for lines and polygons without
depending on source-specific attribute schemas.

Inputs must be vector datasets with CRS metadata, preferably LV95 / EPSG:2056:
  --roads FILE[:LAYER]          12 m exclusion on either side
  --railways FILE[:LAYER]       10 m exclusion on either side
  --buildings FILE[:LAYER]       6 m outward exclusion
  --infrastructure FILE[:LAYER]  8 m outward exclusion (repeatable)

Use the official swissTLM3D road/rail/other-transport layers and
swissBUILDINGS3D building footprints. The program requires gdalinfo and
gdal_rasterize; it never downloads vectors itself, so their
version and licence can be recorded explicitly by the caller.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps

Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class Source:
    value: str
    buffer_m: float

    def command_args(self) -> list[str]:
        # A colon selects an OGR layer.  Linux paths used by this project do not
        # contain colons; this keeps the command interface compact and explicit.
        path, separator, layer = self.value.rpartition(":")
        return [path or self.value] + (["-l", layer] if separator else [])


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def image_layout(path: Path) -> tuple[int, int, float, float, float, float, float]:
    info = json.loads(subprocess.check_output(["gdalinfo", "-json", str(path)], text=True))
    width, height = info["size"]
    corners = info["cornerCoordinates"]
    west, north = corners["upperLeft"]
    east, south = corners["lowerRight"]
    if not west < east or not south < north:
        raise ValueError(f"{path}: invalid georeferenced extent")
    metres_per_pixel = max((east - west) / width, (north - south) / height)
    return width, height, west, south, east, north, metres_per_pixel


def rasterise_source(source: Source, layout: tuple[int, int, float, float, float, float, float],
                     destination: Path) -> None:
    width, height, west, south, east, north, _ = layout
    command = ["gdal_rasterize", "-of", "GTiff", "-ot", "Byte", "-init", "255",
               "-burn", "0", "-at", "-te", str(west), str(south), str(east), str(north),
               "-ts", str(width), str(height), "-a_srs", "EPSG:2056"]
    path_args = source.command_args()
    if len(path_args) == 3:
        command += path_args[1:]
    command += [path_args[0], str(destination)]
    run(command)


def dilate_exclusion(mask: Image.Image, radius_pixels: int) -> Image.Image:
    if radius_pixels <= 0:
        return mask
    # MaxFilter expands white. Invert first so it expands the black exclusion.
    size = 2 * radius_pixels + 1
    return ImageOps.invert(ImageOps.invert(mask).filter(ImageFilter.MaxFilter(size)))


def make_mask(image_path: Path, output: Path, sources: list[Source]) -> float:
    layout = image_layout(image_path)
    work = output.with_suffix(".work.tif")
    result = Image.new("L", layout[:2], 255)
    try:
        for index, source in enumerate(sources):
            layer = work.with_name(f"{work.stem}-{index}.tif")
            rasterise_source(source, layout, layer)
            with Image.open(layer) as raw:
                radius = math.ceil(source.buffer_m / layout[-1])
                excluded = dilate_exclusion(raw.convert("L"), radius)
                result = ImageChops.darker(result, excluded)
            layer.unlink()
        output.parent.mkdir(parents=True, exist_ok=True)
        result.save(output, "PNG", optimize=False, compress_level=9)
    finally:
        work.unlink(missing_ok=True)
    histogram = result.histogram()
    return histogram[0] / (result.width * result.height)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True,
                        help="source cache written by download_swiss_alps.py")
    result.add_argument("--roads", required=True, metavar="FILE[:LAYER]")
    result.add_argument("--railways", required=True, metavar="FILE[:LAYER]")
    result.add_argument("--buildings", required=True, metavar="FILE[:LAYER]")
    result.add_argument("--infrastructure", action="append", default=[],
                        metavar="FILE[:LAYER]")
    result.add_argument("--max-masked-fraction", type=float, default=0.005)
    return result


def main() -> int:
    args = parser().parse_args()
    required = ("gdalinfo", "gdal_rasterize")
    missing = [tool for tool in required if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"missing GDAL tools: {', '.join(missing)}")
    sources = [Source(args.roads, 12), Source(args.railways, 10),
               Source(args.buildings, 6)] + [Source(value, 8) for value in args.infrastructure]
    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summaries = []
    for entry in manifest["accepted"]:
        image_path = args.dataset / entry["imagery"]
        mask_path = args.dataset / "masks" / (image_path.stem + ".natural.png")
        fraction = make_mask(image_path, mask_path, sources)
        entry["natural_mask"] = str(mask_path.relative_to(args.dataset))
        entry["masked_fraction"] = fraction
        entry["strictly_usable"] = fraction <= args.max_masked_fraction
        summaries.append({"tile_id": entry["tile_id"], "masked_fraction": fraction,
                          "strictly_usable": entry["strictly_usable"]})
    manifest["infrastructure_mask"] = {
        "method": "official vectors rasterised at source imagery resolution; conservative pixel dilation",
        "sources": [{"source": source.value, "buffer_m": source.buffer_m} for source in sources],
        "max_masked_fraction": args.max_masked_fraction,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.dataset / "mask-summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(summaries)} natural masks in {args.dataset / 'masks'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"mask_swiss_infrastructure: {error}", file=sys.stderr)
        raise SystemExit(1)
