#!/usr/bin/env python3
"""Unit and GDAL-backed integration tests for Swiss infrastructure masks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


def load_tool(path: Path):
    name = "mask_swiss_infrastructure_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MaskUnitTests(unittest.TestCase):
    def test_dilation_expands_black_with_square_kernel(self) -> None:
        source = Image.new("L", (7, 7), 255)
        source.putpixel((3, 3), 0)
        result = TOOL.dilate_exclusion(source, 1)
        black = {(x, y) for y in range(7) for x in range(7)
                 if result.getpixel((x, y)) == 0}
        self.assertEqual(black, {(x, y) for y in range(2, 5) for x in range(2, 5)})
        source.close()
        result.close()

    def test_source_parser_handles_layer_colons_and_connection_strings(self) -> None:
        with tempfile.TemporaryDirectory(prefix="terrain-mask-source-") as raw_root:
            root = Path(raw_root)
            dataset = root / "roads:2026.gpkg"
            dataset.touch()
            self.assertEqual(
                TOOL.Source(str(dataset), 12).dataset_and_layer(), (str(dataset), None))
            plain = root / "roads.gpkg"
            plain.touch()
            self.assertEqual(
                TOOL.Source(f"{plain}:roads", 12).dataset_and_layer(),
                (str(plain), "roads"))
            self.assertEqual(
                TOOL.Source("PG:dbname=terrain", 12).dataset_and_layer(),
                ("PG:dbname=terrain", None))
            self.assertEqual(
                TOOL.Source("PG:dbname=terrain::roads", 12).dataset_and_layer(),
                ("PG:dbname=terrain", "roads"))

    def test_image_layout_accepts_only_square_north_up_lv95(self) -> None:
        base = {
            "size": [100, 50],
            "coordinateSystem": {"wkt": 'PROJCRS["LV95",ID["EPSG",2056]]'},
            "geoTransform": [2600000.0, 2.0, 0.0, 1200100.0, 0.0, -2.0],
        }
        with mock.patch.object(TOOL, "command_json", return_value=base):
            layout = TOOL.image_layout(Path("image.tif"))
        self.assertEqual(layout.size, (100, 50))
        self.assertEqual((layout.west, layout.south, layout.east, layout.north),
                         (2600000.0, 1200000.0, 2600200.0, 1200100.0))
        self.assertEqual(layout.metres_per_pixel, 2.0)

        invalid = dict(base)
        invalid["geoTransform"] = [2600000.0, 2.0, 0.0, 1200100.0, 0.0, -3.0]
        with mock.patch.object(TOOL, "command_json", return_value=invalid):
            with self.assertRaisesRegex(ValueError, "pixels must be square"):
                TOOL.image_layout(Path("image.tif"))
        invalid = dict(base)
        invalid["coordinateSystem"] = {
            "wkt": 'GEOGCRS["WGS 84",ID["EPSG",4326]]'}
        with mock.patch.object(TOOL, "command_json", return_value=invalid):
            with self.assertRaisesRegex(ValueError, "expected EPSG:2056"):
                TOOL.image_layout(Path("image.tif"))

    def test_atomic_json_has_no_temporary_residue(self) -> None:
        with tempfile.TemporaryDirectory(prefix="terrain-mask-json-") as raw_root:
            path = Path(raw_root) / "manifest.json"
            TOOL.write_json_atomic(path, {"answer": 42})
            self.assertEqual(json.loads(path.read_text()), {"answer": 42})
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_vector_sources_must_be_lv95(self) -> None:
        info = {
            "driverShortName": "GeoJSON",
            "layers": [{
                "name": "features",
                "featureCount": 1,
                "geometryFields": [{
                    "type": "LineString",
                    "coordinateSystem": {
                        "wkt": 'GEOGCRS["WGS 84",ID["EPSG",4326]]'},
                }],
            }],
        }
        with mock.patch.object(TOOL, "command_json", return_value=info):
            with self.assertRaisesRegex(ValueError, "must use EPSG:2056"):
                TOOL.validate_source(TOOL.Source("features.geojson", 5))


@unittest.skipUnless(all(shutil.which(tool) for tool in (
    "gdal_create", "gdalinfo", "gdal_rasterize", "gdal_translate", "ogr2ogr", "ogrinfo"
)), "GDAL command-line tools are required")
class MaskIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="terrain-mask-integration-")
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        imagery = self.dataset / "imagery" / "synthetic.tif"
        imagery.parent.mkdir(parents=True)
        subprocess.run([
            "gdal_create", "--quiet", "-of", "GTiff", "-ot", "Byte",
            "-outsize", "100", "100", "-bands", "1", "-burn", "127",
            "-a_srs", "EPSG:2056", "-a_ullr", "2600000", "1200100",
            "2600100", "1200000", str(imagery),
        ], check=True)
        (self.dataset / "manifest.json").write_text(json.dumps({
            "accepted": [{"tile_id": "synthetic", "imagery": "imagery/synthetic.tif"}],
            "rejected": [],
            "bytes": imagery.stat().st_size,
        }), encoding="utf-8")

        geojson = self.root / "features.geojson"
        geojson.write_text(json.dumps({
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"include": 1},
                 "geometry": {"type": "LineString", "coordinates": [
                     [2600010, 1200050.5], [2600090, 1200050.5]]}},
                {"type": "Feature", "properties": {"include": 0},
                 "geometry": {"type": "LineString", "coordinates": [
                     [2600010, 1200080.5], [2600090, 1200080.5]]}},
                # The geometry itself is outside the image, but its 12 m road
                # buffer must cross the western edge of the tile.
                {"type": "Feature", "properties": {"include": 1},
                 "geometry": {"type": "LineString", "coordinates": [
                     [2599996, 1200040], [2599996, 1200060]]}},
            ],
        }), encoding="utf-8")
        self.vector = self.root / "features.gpkg"
        subprocess.run([
            "ogr2ogr", "-f", "GPKG", "-nln", "features", "-a_srs", "EPSG:2056",
            str(self.vector), str(geojson),
        ], check=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def command(self, *extra: str) -> list[str]:
        source = f"{self.vector}:features"
        return [
            sys.executable, str(TOOL_PATH), "--dataset", str(self.dataset),
            "--roads", source, "--roads-where", "include = 1",
            "--railways", source, "--railways-where", "include = 1",
            "--buildings", source, "--buildings-where", "include = 1",
            "--infrastructure", source,
            "--infrastructure-where", "include = 0 AND include = 1",
            "--infrastructure-buffer-m", "4",
            "--workers", "2",
            *extra,
        ]

    def test_dry_run_is_read_only(self) -> None:
        result = subprocess.run(self.command("--dry-run"), check=True,
                                capture_output=True, text=True)
        self.assertIn("preflight passed for 1 imagery tiles", result.stdout)
        self.assertFalse((self.dataset / "masks").exists())
        manifest = json.loads((self.dataset / "manifest.json").read_text())
        self.assertNotIn("infrastructure_mask", manifest)

    def test_end_to_end_output_is_binary_georeferenced_and_filtered(self) -> None:
        subprocess.run(self.command(), check=True)
        output = self.dataset / "masks" / "synthetic.natural.tif"
        self.assertTrue(output.is_file())
        info = json.loads(subprocess.check_output(
            ["gdalinfo", "-json", str(output)], text=True))
        self.assertEqual(info["size"], [100, 100])
        self.assertEqual(info["geoTransform"],
                         [2600000.0, 1.0, 0.0, 1200100.0, 0.0, -1.0])
        self.assertIn('ID["EPSG",2056]', info["coordinateSystem"]["wkt"])
        with Image.open(output) as mask:
            self.assertEqual(mask.mode, "L")
            get_data = getattr(mask, "get_flattened_data", mask.getdata)
            self.assertEqual(set(get_data()), {0, 255})
            self.assertEqual(mask.getpixel((50, 49)), 0)
            self.assertEqual(mask.getpixel((0, 49)), 0)
            # The excluded test feature is 30 m away, beyond the 12 m buffer.
            self.assertEqual(mask.getpixel((50, 19)), 255)
            fraction = mask.histogram()[0] / 10_000
        self.assertGreater(fraction, 0.15)
        self.assertLess(fraction, 0.30)

        manifest = json.loads((self.dataset / "manifest.json").read_text())
        entry = manifest["accepted"][0]
        self.assertEqual(entry["natural_mask"], "masks/synthetic.natural.tif")
        self.assertTrue(math.isclose(entry["masked_fraction"], fraction))
        self.assertFalse(entry["strictly_usable"])
        self.assertEqual(manifest["infrastructure_mask"]["crs"], "EPSG:2056")
        self.assertEqual(manifest["infrastructure_mask"]["sources"][0]["where"],
                         "include = 1")
        self.assertEqual(manifest["infrastructure_mask"]["sources"][-1]["buffer_m"], 4.0)
        self.assertEqual(list((self.dataset / "masks").glob(".*")), [])

    def test_failed_generation_preserves_existing_output_and_cleans_work(self) -> None:
        output = self.dataset / "masks" / "existing.tif"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"existing-mask")
        layout = TOOL.image_layout(self.dataset / "imagery" / "synthetic.tif")
        sources = [TOOL.Source(f"{self.vector}:features", 12),
                   TOOL.Source(str(self.root / "missing.gpkg"), 10)]
        with self.assertRaises(subprocess.CalledProcessError):
            TOOL.make_mask(self.dataset / "imagery" / "synthetic.tif",
                           output, sources, layout)
        self.assertEqual(output.read_bytes(), b"existing-mask")
        self.assertEqual(list(output.parent.glob(".existing-*")), [])

    def test_invalid_fraction_is_rejected_before_vector_access(self) -> None:
        command = self.command("--max-masked-fraction", "nan")
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be finite and between 0 and 1", result.stderr)
        self.assertFalse((self.dataset / "masks").exists())

    def test_invalid_worker_count_is_rejected_before_vector_access(self) -> None:
        result = subprocess.run(self.command("--workers", "17"),
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--workers must be between 1 and 16", result.stderr)
        self.assertFalse((self.dataset / "masks").exists())


def main() -> None:
    unittest.main(argv=[sys.argv[0]], verbosity=2)


if __name__ == "__main__":
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--tool", type=Path, required=True)
    parsed = arguments.parse_args()
    TOOL_PATH = parsed.tool.resolve()
    TOOL = load_tool(TOOL_PATH)
    main()
