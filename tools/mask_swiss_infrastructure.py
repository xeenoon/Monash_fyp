#!/usr/bin/env python3
"""Build strict natural-terrain masks from official Swiss vector geometry.

Each output is an 8-bit georeferenced GeoTIFF aligned pixel-for-pixel with its
source imagery COG: 255 means natural terrain and 0 means infrastructure. Roads,
rails, buildings, and optional infrastructure begin as vector geometry, are
rasterised, then conservatively dilated with a square kernel in whole pixels.

Inputs must be spatial vector datasets with CRS metadata. The imagery must be a
north-up, square-pixel LV95 / EPSG:2056 raster:
  --roads FILE[:LAYER]          12 m exclusion on either side
  --railways FILE[:LAYER]       10 m exclusion on either side
  --buildings FILE[:LAYER]       6 m outward exclusion
  --infrastructure FILE[:LAYER]  8 m outward exclusion (repeatable)

Use the official swissTLM3D road/rail/other-transport layers and
swissBUILDINGS3D building footprints. Optional --*-where arguments accept OGR
SQL attribute filters, for example to omit tunnels. The program never downloads
vectors itself, so their version and licence remain explicit caller inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageFilter, ImageOps

Image.MAX_IMAGE_PIXELS = None
EXPECTED_EPSG = 2056


@dataclass(frozen=True)
class RasterLayout:
    width: int
    height: int
    west: float
    south: float
    east: float
    north: float
    metres_per_pixel: float

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


@dataclass(frozen=True)
class Source:
    value: str
    buffer_m: float
    where: str | None = None

    def dataset_and_layer(self) -> tuple[str, str | None]:
        # A double colon explicitly separates any OGR datasource (including a
        # VSI URL or database connection string) from its layer.
        dataset, separator, layer = self.value.rpartition("::")
        if separator and dataset and layer:
            return dataset, layer
        # Prefer an existing path verbatim, which permits colons in local paths.
        # Otherwise split only when the prefix is itself an existing path. OGR
        # connection strings such as PG:dbname=x therefore remain untouched.
        if Path(self.value).exists():
            return self.value, None
        dataset, separator, layer = self.value.rpartition(":")
        if separator and dataset and layer and Path(dataset).exists():
            return dataset, layer
        return self.value, None


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def command_json(command: list[str]) -> dict[str, Any]:
    output = subprocess.check_output(command, text=True)
    value = json.loads(output)
    if not isinstance(value, dict):
        raise ValueError(f"{command[0]} returned a non-object JSON document")
    return value


def image_layout(path: Path) -> RasterLayout:
    info = command_json(["gdalinfo", "-json", str(path)])
    size = info.get("size")
    transform = info.get("geoTransform")
    wkt = info.get("coordinateSystem", {}).get("wkt", "")
    if not isinstance(size, list) or len(size) != 2 or not all(
            isinstance(value, int) and value > 0 for value in size):
        raise ValueError(f"{path}: invalid raster dimensions")
    if not isinstance(transform, list) or len(transform) != 6 or not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in transform):
        raise ValueError(f"{path}: missing or invalid geotransform")
    if not re.search(rf'ID\["EPSG"\s*,\s*{EXPECTED_EPSG}\s*\]', wkt):
        raise ValueError(f"{path}: expected EPSG:{EXPECTED_EPSG} imagery CRS")
    origin_x, pixel_x, rotate_x, origin_y, rotate_y, pixel_y = transform
    tolerance = max(abs(pixel_x), abs(pixel_y), 1.0) * 1e-10
    if abs(rotate_x) > tolerance or abs(rotate_y) > tolerance:
        raise ValueError(f"{path}: rotated or skewed rasters are not supported")
    if pixel_x <= 0.0 or pixel_y >= 0.0:
        raise ValueError(f"{path}: expected a north-up geotransform")
    if not math.isclose(pixel_x, -pixel_y, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{path}: pixels must be square to buffer in metres")
    width, height = size
    east = origin_x + pixel_x * width
    south = origin_y + pixel_y * height
    if not origin_x < east or not south < origin_y:
        raise ValueError(f"{path}: invalid georeferenced extent")
    return RasterLayout(width, height, origin_x, south, east, origin_y, pixel_x)


def validate_source(source: Source) -> dict[str, Any]:
    dataset, layer = source.dataset_and_layer()
    command = ["ogrinfo", "-ro", "-so", "-json"]
    if source.where:
        command += ["-where", source.where]
    command.append(dataset)
    if layer:
        command.append(layer)
    info = command_json(command)
    spatial_layers = []
    for description in info.get("layers", []):
        geometry_fields = description.get("geometryFields", [])
        if not geometry_fields:
            continue
        for geometry in geometry_fields:
            wkt = geometry.get("coordinateSystem", {}).get("wkt", "")
            if not wkt:
                raise ValueError(
                    f"{source.value}: layer {description.get('name')} has no CRS metadata")
            if not re.search(rf'ID\["EPSG"\s*,\s*{EXPECTED_EPSG}\s*\]', wkt):
                raise ValueError(
                    f"{source.value}: layer {description.get('name')} must use "
                    f"EPSG:{EXPECTED_EPSG}")
        spatial_layers.append({
            "name": description.get("name"),
            "feature_count": description.get("featureCount"),
            "geometry_types": [geometry.get("type") for geometry in geometry_fields],
            "crs_sha256": hashlib.sha256(
                geometry_fields[0]["coordinateSystem"]["wkt"].encode("utf-8")
            ).hexdigest(),
        })
    if not spatial_layers:
        selected = f" layer {layer}" if layer else ""
        raise ValueError(f"{dataset}:{selected} contains no spatial layer with CRS metadata")
    if layer is None and len(spatial_layers) != 1:
        raise ValueError(
            f"{dataset}: select one layer explicitly with FILE::LAYER")
    return {
        "source": source.value,
        "dataset": dataset,
        "layer": layer,
        "where": source.where,
        "buffer_m": source.buffer_m,
        "driver": info.get("driverShortName"),
        "layers": spatial_layers,
    }


def rasterise_source(source: Source, layout: RasterLayout, destination: Path) -> None:
    # gdal_rasterize iterates an entire large OpenFileGDB layer even for a tiny
    # output extent. Spatially extracting first uses the source index and turns
    # a multi-second national scan into a bounded local operation.
    clipped = destination.with_suffix(".gpkg")
    dataset, layer = source.dataset_and_layer()
    clip_command = [
        "ogr2ogr", "--quiet", "-f", "GPKG", "-nln", "features",
        "-nlt", "GEOMETRY",
        "-spat", str(layout.west), str(layout.south),
        str(layout.east), str(layout.north),
    ]
    if source.where:
        clip_command += ["-where", source.where]
    clip_command += [str(clipped), dataset]
    if layer:
        clip_command.append(layer)
    run(clip_command)

    command = [
        "gdal_rasterize", "--quiet", "-of", "GTiff", "-ot", "Byte",
        "-init", "255", "-burn", "0", "-at",
        "-te", str(layout.west), str(layout.south), str(layout.east), str(layout.north),
        "-ts", str(layout.width), str(layout.height),
        "-a_srs", f"EPSG:{EXPECTED_EPSG}",
        "-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=2",
    ]
    command += ["-l", "features", str(clipped), str(destination)]
    run(command)


def dilate_exclusion(mask: Image.Image, radius_pixels: int) -> Image.Image:
    if radius_pixels <= 0:
        return mask.copy()
    # MaxFilter expands white. Invert first so it expands black exclusions. A
    # square kernel intentionally over-excludes diagonal corners by up to sqrt(2).
    size = 2 * radius_pixels + 1
    return ImageOps.invert(ImageOps.invert(mask).filter(ImageFilter.MaxFilter(size)))


def make_mask(image_path: Path, output: Path, sources: list[Source],
              layout: RasterLayout | None = None) -> float:
    layout = layout or image_layout(image_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = Image.new("L", layout.size, 255)
    try:
        with tempfile.TemporaryDirectory(prefix=f".{output.stem}-", dir=output.parent) as raw_work:
            work = Path(raw_work)
            for index, source in enumerate(sources):
                layer_path = work / f"source-{index}.tif"
                radius = math.ceil(source.buffer_m / layout.metres_per_pixel)
                padded = RasterLayout(
                    layout.width + 2 * radius,
                    layout.height + 2 * radius,
                    layout.west - radius * layout.metres_per_pixel,
                    layout.south - radius * layout.metres_per_pixel,
                    layout.east + radius * layout.metres_per_pixel,
                    layout.north + radius * layout.metres_per_pixel,
                    layout.metres_per_pixel,
                )
                rasterise_source(source, padded, layer_path)
                with Image.open(layer_path) as raw:
                    excluded = dilate_exclusion(raw.convert("L"), radius)
                    cropped = excluded.crop((radius, radius,
                                             radius + layout.width,
                                             radius + layout.height))
                    combined = ImageChops.darker(result, cropped)
                    result.close()
                    excluded.close()
                    cropped.close()
                    result = combined

            histogram = result.histogram()
            fraction = histogram[0] / (result.width * result.height)
            pixels = work / "combined.png"
            result.save(pixels, "PNG", optimize=False, compress_level=1)
            temporary_output = work / output.name
            run([
                "gdal_translate", "--quiet", "-of", "GTiff", "-ot", "Byte",
                "-a_srs", f"EPSG:{EXPECTED_EPSG}",
                "-a_ullr", str(layout.west), str(layout.north),
                str(layout.east), str(layout.south),
                "-colorinterp", "gray", "-co", "TILED=YES",
                "-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=2",
                "-co", "BIGTIFF=IF_SAFER", str(pixels), str(temporary_output),
            ])
            os.replace(temporary_output, output)
            return fraction
    finally:
        result.close()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(serialized)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True,
                        help="source cache written by download_swiss_alps.py")
    result.add_argument("--roads", required=True, metavar="FILE[:LAYER]")
    result.add_argument("--roads-where", metavar="OGR_SQL")
    result.add_argument("--railways", required=True, metavar="FILE[:LAYER]")
    result.add_argument("--railways-where", metavar="OGR_SQL")
    result.add_argument("--buildings", required=True, metavar="FILE[:LAYER]")
    result.add_argument("--buildings-where", metavar="OGR_SQL")
    result.add_argument("--infrastructure", action="append", default=[],
                        metavar="FILE[:LAYER]")
    result.add_argument("--infrastructure-where", action="append", default=[],
                        metavar="OGR_SQL")
    result.add_argument("--infrastructure-buffer-m", action="append", default=[],
                        type=float, metavar="METRES")
    result.add_argument("--source-version",
                        help="vector release identifier recorded in the manifest")
    result.add_argument("--source-license",
                        help="vector licence identifier recorded in the manifest")
    result.add_argument("--max-masked-fraction", type=float, default=0.005)
    result.add_argument("--workers", type=int, default=4,
                        help="concurrent imagery validation and mask jobs (default: 4)")
    result.add_argument("--dry-run", action="store_true",
                        help="validate tools, vectors, manifest, and imagery without writing")
    return result


def manifest_jobs(dataset: Path, manifest: dict[str, Any], workers: int = 1) -> list[
        tuple[dict[str, Any], Path, Path, RasterLayout]]:
    accepted = manifest.get("accepted")
    if not isinstance(accepted, list):
        raise ValueError("manifest.json: accepted must be an array")
    root = dataset.resolve()
    jobs = []
    outputs: set[Path] = set()
    for index, entry in enumerate(accepted):
        if not isinstance(entry, dict):
            raise ValueError(f"manifest.json: accepted[{index}] must be an object")
        tile_id = entry.get("tile_id")
        imagery = entry.get("imagery")
        if not isinstance(tile_id, str) or not tile_id:
            raise ValueError(f"manifest.json: accepted[{index}] has no tile_id")
        if not isinstance(imagery, str) or not imagery:
            raise ValueError(f"manifest.json: accepted[{index}] has no imagery path")
        relative = Path(imagery)
        image_path = (dataset / relative).resolve()
        if relative.is_absolute() or not image_path.is_relative_to(root):
            raise ValueError(f"manifest.json: unsafe imagery path {imagery!r}")
        if not image_path.is_file():
            raise ValueError(f"{image_path}: imagery file does not exist")
        mask_path = dataset / "masks" / (relative.stem + ".natural.tif")
        if mask_path in outputs:
            raise ValueError(f"manifest.json: duplicate mask output {mask_path}")
        outputs.add(mask_path)
        jobs.append((entry, image_path, mask_path))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        layouts = executor.map(image_layout, (job[1] for job in jobs))
        return [(entry, image_path, mask_path, layout)
                for (entry, image_path, mask_path), layout in zip(jobs, layouts)]


def main() -> int:
    args = parser().parse_args()
    required = ("gdalinfo", "gdal_rasterize", "gdal_translate", "ogr2ogr", "ogrinfo")
    missing = [tool for tool in required if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"missing GDAL tools: {', '.join(missing)}")
    if not math.isfinite(args.max_masked_fraction) or not (
            0.0 <= args.max_masked_fraction <= 1.0):
        raise ValueError("--max-masked-fraction must be finite and between 0 and 1")
    if not 1 <= args.workers <= 16:
        raise ValueError("--workers must be between 1 and 16")
    if args.infrastructure_where and len(args.infrastructure_where) != len(
            args.infrastructure):
        raise ValueError("provide one --infrastructure-where per --infrastructure, "
                         "using an empty value when no filter is required")
    if args.infrastructure_buffer_m and len(args.infrastructure_buffer_m) != len(
            args.infrastructure):
        raise ValueError("provide one --infrastructure-buffer-m per --infrastructure")
    if any(not math.isfinite(value) or value < 0.0
           for value in args.infrastructure_buffer_m):
        raise ValueError("--infrastructure-buffer-m must be finite and non-negative")
    infrastructure_where = args.infrastructure_where or [None] * len(args.infrastructure)
    infrastructure_buffers = args.infrastructure_buffer_m or [8.0] * len(
        args.infrastructure)
    sources = [
        Source(args.roads, 12, args.roads_where),
        Source(args.railways, 10, args.railways_where),
        Source(args.buildings, 6, args.buildings_where),
    ] + [Source(value, buffer_m, where) for value, where, buffer_m in zip(
        args.infrastructure, infrastructure_where, infrastructure_buffers)]

    if not args.dataset.is_dir():
        raise ValueError(f"{args.dataset}: dataset directory does not exist")
    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")

    print(f"validating {len(sources)} vector sources", flush=True)
    source_metadata = [validate_source(source) for source in sources]
    jobs = manifest_jobs(args.dataset, manifest, args.workers)
    if args.dry_run:
        print(f"preflight passed for {len(jobs)} imagery tiles; no files written")
        return 0

    def mask_job(job: tuple[dict[str, Any], Path, Path, RasterLayout]) -> float:
        _, image_path, mask_path, layout = job
        return make_mask(image_path, mask_path, sources, layout)

    summaries = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        fractions = executor.map(mask_job, jobs)
        for index, ((entry, _, mask_path, _), fraction) in enumerate(
                zip(jobs, fractions), 1):
            print(f"[{index}/{len(jobs)}] masked {entry['tile_id']}", flush=True)
            entry["natural_mask"] = str(mask_path.relative_to(args.dataset))
            entry["masked_fraction"] = fraction
            entry["strictly_usable"] = fraction <= args.max_masked_fraction
            summaries.append({
                "tile_id": entry["tile_id"],
                "masked_fraction": fraction,
                "strictly_usable": entry["strictly_usable"],
            })

    gdal_version = subprocess.check_output(["gdalinfo", "--version"], text=True).strip()
    manifest["infrastructure_mask"] = {
        "method": "all-touched vector rasterisation followed by conservative square-pixel dilation",
        "output": "georeferenced 8-bit GeoTIFF; 255 natural, 0 infrastructure",
        "crs": f"EPSG:{EXPECTED_EPSG}",
        "sources": source_metadata,
        "source_version": args.source_version,
        "source_license": args.source_license,
        "max_masked_fraction": args.max_masked_fraction,
        "gdal_version": gdal_version,
    }
    write_json_atomic(args.dataset / "mask-summary.json", summaries)
    write_json_atomic(manifest_path, manifest)
    print(f"wrote {len(summaries)} natural masks in {args.dataset / 'masks'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError,
            json.JSONDecodeError) as error:
        print(f"mask_swiss_infrastructure: {error}", file=sys.stderr)
        raise SystemExit(1)
