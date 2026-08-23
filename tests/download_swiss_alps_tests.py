#!/usr/bin/env python3
"""Focused unit tests for the Swiss Alps cache downloader."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


def load_tool(path: Path):
    name = "download_swiss_alps_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DownloaderTests(unittest.TestCase):
    def test_transient_url_timeout_is_retried(self) -> None:
        request = TOOL.urllib.request.Request("https://example.invalid/data")
        response = object()
        with mock.patch.object(TOOL.urllib.request, "urlopen",
                               side_effect=[TimeoutError(), response]) as urlopen, \
                mock.patch.object(TOOL.time, "sleep") as sleep:
            self.assertIs(TOOL.open_with_retry(request, attempts=2), response)
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_terrain_metrics_do_not_smooth_away_low_pixels(self) -> None:
        with tempfile.TemporaryDirectory(prefix="terrain-download-dem-") as raw_root:
            path = Path(raw_root) / "dem.tif"
            image = Image.new("F", (500, 500), 2200.0)
            image.putpixel((249, 249), 1799.0)
            image.putpixel((250, 250), 2400.0)
            image.save(path)

            minimum, median, relief = TOOL.terrain_metrics(path)

        self.assertEqual(minimum, 1799.0)
        self.assertEqual(median, 2200.0)
        self.assertEqual(relief, 601.0)
        self.assertFalse(TOOL.natural_remote(minimum, median, relief, 1800.0, 180.0))

    def test_nan_samples_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory(prefix="terrain-download-dem-") as raw_root:
            path = Path(raw_root) / "dem.tif"
            image = Image.new("F", (5, 1))
            image.putdata([float("nan"), float("-inf"), 2000.0,
                           2300.0, float("inf")])
            image.save(path)
            self.assertEqual(TOOL.terrain_metrics(path), (2000.0, 2300.0, 300.0))

    def test_remote_thresholds_are_inclusive(self) -> None:
        self.assertTrue(TOOL.natural_remote(1800.0, 1950.0, 180.0, 1800.0, 180.0))
        self.assertFalse(TOOL.natural_remote(1799.99, 2200.0, 500.0, 1800.0, 180.0))
        self.assertFalse(TOOL.natural_remote(2000.0, 1949.99, 500.0, 1800.0, 180.0))
        self.assertFalse(TOOL.natural_remote(2000.0, 2200.0, 179.99, 1800.0, 180.0))

    def test_json_writes_are_atomic_and_leave_no_residue(self) -> None:
        with tempfile.TemporaryDirectory(prefix="terrain-download-json-") as raw_root:
            path = Path(raw_root) / "manifest.json"
            TOOL.write_json(path, {"bytes": 42})
            self.assertEqual(json.loads(path.read_text()), {"bytes": 42})
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_resume_validation_checks_configuration_files_and_bytes(self) -> None:
        args = argparse.Namespace(regions=["valais"], minimum_height=1800.0,
                                  minimum_relief=180.0)
        selection = TOOL.selection_config(args)
        with tempfile.TemporaryDirectory(prefix="terrain-download-state-") as raw_root:
            root = Path(raw_root)
            (root / "dem").mkdir()
            (root / "imagery").mkdir()
            (root / "dem/a.tif").write_bytes(b"dem")
            (root / "imagery/a.tif").write_bytes(b"image")
            state = {
                "accepted": [{"tile_id": "a", "dem": "dem/a.tif",
                              "imagery": "imagery/a.tif"}],
                "rejected": [], "bytes": 8, "selection": selection,
            }
            TOOL.validate_resume_state(root, state, selection, 100)
            with self.assertRaisesRegex(RuntimeError, "selection settings"):
                TOOL.validate_resume_state(root, state, {**selection, "gsd_m": 1.0}, 100)
            state["bytes"] = 7
            with self.assertRaisesRegex(RuntimeError, "actual assets total 8"):
                TOOL.validate_resume_state(root, state, selection, 100)

    def test_parallel_fetch_exposes_completed_partials_atomically(self) -> None:
        with tempfile.TemporaryDirectory(prefix="terrain-download-fetch-") as raw_root:
            root = Path(raw_root)
            complete = root / "complete.tif"
            complete.write_bytes(b"already complete")
            destination = root / "nested" / "asset.tif"

            def fake_curl(command: list[str], check: bool) -> None:
                self.assertTrue(check)
                self.assertIn("--parallel", command)
                outputs = [command[index + 1] for index, value in enumerate(command)
                           if value == "--output"]
                self.assertEqual(outputs, [str(destination) + ".part"])
                Path(outputs[0]).write_bytes(b"downloaded")

            with mock.patch.object(TOOL.subprocess, "run", side_effect=fake_curl):
                TOOL.fetch_many_with_curl([
                    ("https://example.invalid/complete", complete),
                    ("https://example.invalid/asset", destination),
                ], workers=3)

            self.assertEqual(complete.read_bytes(), b"already complete")
            self.assertEqual(destination.read_bytes(), b"downloaded")
            self.assertFalse(destination.with_suffix(".tif.part").exists())

    def test_discovery_catalogue_is_cached_and_reloaded(self) -> None:
        tile_id = "2600-1200"
        asset = {"gsd": 2.0, "href": "https://example.invalid/a.tif"}
        dem_item = {"id": f"dem_{tile_id}", "assets": {"dem": asset}}
        image_item = {"id": f"image_{tile_id}", "assets": {"image": asset}}
        pair = TOOL.Pair(tile_id, dem_item, image_item, asset, asset)
        with tempfile.TemporaryDirectory(prefix="terrain-download-cache-") as raw_root:
            root = Path(raw_root)
            with mock.patch.object(TOOL, "discover_pairs", return_value=[pair]) as discover:
                first = TOOL.cached_pairs(root, ["valais"], 3)
                second = TOOL.cached_pairs(root, ["valais"], 3)
            self.assertEqual(discover.call_count, 1)
            self.assertEqual([value.tile_id for value in first], [tile_id])
            self.assertEqual([value.tile_id for value in second], [tile_id])
            self.assertTrue((root / "discovery.json").is_file())


def main() -> None:
    unittest.main(argv=[sys.argv[0]], verbosity=2)


if __name__ == "__main__":
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--tool", type=Path, required=True)
    parsed = arguments.parse_args()
    TOOL = load_tool(parsed.tool.resolve())
    main()
