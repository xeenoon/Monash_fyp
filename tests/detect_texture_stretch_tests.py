#!/usr/bin/env python3
"""Unit and integration tests for tools/detect_texture_stretch.py."""

from __future__ import annotations

import argparse
import importlib.util
import math
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOL_PATH = ROOT / "tools" / "detect_texture_stretch.py"
DEFAULT_TERRAIN_TOOL_PATH = ROOT / "tools" / "terrain_tiles.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TOOL = None
TERRAIN_TOOL = None


def make_tile(segments, gutter, dx, dz, height_fn, *, all_valid=True):
    """A synthetic in-memory Tile: heights follow height_fn(row, col)."""
    width = height = segments + 1 + 2 * gutter
    west, south = 0.0, 0.0
    east, north = dx * segments, dz * segments
    heights = [height_fn(row, col) for row in range(height) for col in range(width)]
    valid = [all_valid] * (width * height)
    payload = struct.pack(f"<{len(heights)}f", *heights)
    validity, valid_count = TERRAIN_TOOL._validity_bytes(valid)
    key = TERRAIN_TOOL.Key(0, 0, 0)
    return TERRAIN_TOOL.Tile(
        key, None, width, height, gutter, TERRAIN_TOOL.HEIGHT_F32,
        0.0, 0.0, 1.0, valid_count, (west, south, east, north),
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        TERRAIN_TOOL.HAS_IMAGERY, payload, validity, b"",
        "EPSG:2056", "{}", "imagery/0/0/0.png")


class TriangleStretchFactorTests(unittest.TestCase):
    def test_flat_plane_has_unit_stretch_everywhere(self) -> None:
        tile = make_tile(4, 1, 10.0, 10.0, lambda row, col: 100.0)
        triangles, segments = TOOL.triangle_stretch_factors(tile)
        self.assertEqual(segments, 4)
        self.assertEqual(len(triangles), 4 * 4 * 2)
        for _, stretch in triangles:
            self.assertAlmostEqual(stretch, 1.0, places=9)

    def test_uniform_ramp_matches_the_closed_form_1_over_cos_theta(self) -> None:
        slope_per_step, dx = 15.0, 10.0  # rise 15m per 10m step -> slope 1.5
        tile = make_tile(4, 1, dx, dx, lambda row, col: slope_per_step * col)
        triangles, _ = TOOL.triangle_stretch_factors(tile)
        expected = math.sqrt(1.0 + (slope_per_step / dx) ** 2)  # sqrt(1+1.5^2)
        self.assertGreater(expected, 1.5)
        for _, stretch in triangles:
            self.assertAlmostEqual(stretch, expected, places=6)

    def test_gentler_ramp_stays_below_1_5x(self) -> None:
        slope_per_step, dx = 5.0, 10.0  # slope 0.5 -> stretch sqrt(1.25) ~= 1.118
        tile = make_tile(4, 1, dx, dx, lambda row, col: slope_per_step * col)
        triangles, _ = TOOL.triangle_stretch_factors(tile)
        for _, stretch in triangles:
            self.assertLess(stretch, 1.5)

    def test_vertical_face_reports_infinite_stretch(self) -> None:
        # A "cliff": one column jumps by a huge height relative to its
        # horizontal spacing, i.e. the triangle normal is ~horizontal.
        tile = make_tile(4, 1, 1e-6, 1e-6, lambda row, col: 1000.0 * col)
        triangles, _ = TOOL.triangle_stretch_factors(tile)
        self.assertTrue(all(math.isinf(stretch) for _, stretch in triangles))

    def test_no_data_sample_drops_every_triangle_touching_it(self) -> None:
        tile = make_tile(4, 1, 10.0, 10.0, lambda row, col: 100.0)
        index = 3 * tile.width + 3  # an interior corner shared by up to 6 triangles
        bits = bytearray(tile.validity)
        bits[index >> 3] &= ~(1 << (index & 7))
        tile.validity = bytes(bits)
        tile.valid_count -= 1
        triangles, _ = TOOL.triangle_stretch_factors(tile)
        touched = {corner for tri, _ in triangles for corner in tri}
        self.assertNotIn((3, 3), touched)
        self.assertEqual(len(triangles), 4 * 4 * 2 - 6)

    def test_gutter_ring_is_excluded_from_the_mesh(self) -> None:
        # Height differs wildly outside the interior; if the gutter leaked
        # into the mesh, some triangle would report a huge stretch.
        def height_fn(row, col):
            interior = range(1, 1 + 4 + 1)
            if row in interior and col in interior:
                return 100.0
            return 100.0 + 1e6  # gutter ring: absurd height, must not be used

        tile = make_tile(4, 1, 10.0, 10.0, height_fn)
        triangles, _ = TOOL.triangle_stretch_factors(tile)
        for _, stretch in triangles:
            self.assertAlmostEqual(stretch, 1.0, places=6)


class DatasetRootInferenceTests(unittest.TestCase):
    def test_infers_root_two_levels_above_tiles(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tile_path = Path(raw) / "dataset" / "tiles" / "1" / "0" / "0.trn"
            tile_path.parent.mkdir(parents=True)
            tile_path.touch()
            self.assertEqual(TOOL._default_dataset_root(tile_path),
                             (Path(raw) / "dataset").resolve())

    def test_raises_when_layout_does_not_match(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tile_path = Path(raw) / "somewhere" / "0.trn"
            tile_path.parent.mkdir(parents=True)
            tile_path.touch()
            with self.assertRaises(ValueError):
                TOOL._default_dataset_root(tile_path)


class IntegrationTests(unittest.TestCase):
    """Builds a real one-tile dataset (flat west half, steep east ramp)
    through terrain_tiles.py and runs the CLI end to end against it."""

    SAMPLES = 9        # -> segments = 8, interior grid is 9x9
    GUTTER = 1
    SPAN = 80.0        # extent side length -> dx = dz = 10m per step
    IMAGERY_SIZE = 80  # -> 10px per grid step, easy pixel arithmetic
    FLAT_COLOR = (60, 160, 60)
    RAMP_START_COL = 4
    RISE_PER_STEP = 40.0  # slope 4.0 over dx=10 -> stretch sqrt(17) ~= 4.12

    def _height(self, row, col):
        if col < self.RAMP_START_COL:
            return 100.0
        return 100.0 + self.RISE_PER_STEP * (col - self.RAMP_START_COL)

    def _build_dataset(self, root: Path) -> Path:
        dem = Image.new("F", (self.SAMPLES, self.SAMPLES))
        dem.putdata([self._height(row, col)
                     for row in range(self.SAMPLES) for col in range(self.SAMPLES)])
        dem_path = root / "dem.tif"
        dem.save(dem_path)
        imagery = Image.new("RGB", (self.SAMPLES, self.SAMPLES), self.FLAT_COLOR)
        imagery_path = root / "imagery.tif"
        imagery.save(imagery_path)
        dataset = root / "dataset"
        subprocess.run([sys.executable, str(DEFAULT_TERRAIN_TOOL_PATH), "build",
                        "--dem", str(dem_path), "--imagery", str(imagery_path),
                        "--output", str(dataset),
                        "--extent", "0", "0", str(self.SPAN), str(self.SPAN),
                        "--profile", "EPSG:2056", "--levels", "1",
                        "--samples", str(self.SAMPLES),
                        "--imagery-size", str(self.IMAGERY_SIZE),
                        "--gutter", str(self.GUTTER), "--height-encoding", "f32"],
                       check=True, capture_output=True, text=True)
        return dataset

    def _pixel_for(self, col: int, row: int, img_size: int) -> tuple[int, int]:
        segments = self.SAMPLES - 1
        inner = img_size - 2 * self.GUTTER
        u = (col - self.GUTTER) / segments
        v = (row - self.GUTTER) / segments
        return (round(self.GUTTER + u * inner), round(self.GUTTER + v * inner))

    def test_flat_side_is_untouched_and_steep_side_is_reddened(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dataset = self._build_dataset(root)
            tile_path = dataset / "tiles" / "0" / "0" / "0.trn"
            output = root / "stretch.png"
            result = subprocess.run(
                [sys.executable, str(DEFAULT_TOOL_PATH), str(tile_path),
                 "--output", str(output), "--threshold", "1.5", "--opacity", "0.25"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("stretched >=", result.stdout)

            with Image.open(output) as image:
                img_w, _ = image.size
                flat_x, flat_y = self._pixel_for(1, 4, img_w)
                steep_x, steep_y = self._pixel_for(7, 4, img_w)
                flat = ImageStat.Stat(image.crop((flat_x - 1, flat_y - 1, flat_x + 2, flat_y + 2))
                                       .convert("RGB")).mean
                steep = ImageStat.Stat(image.crop((steep_x - 1, steep_y - 1, steep_x + 2, steep_y + 2))
                                        .convert("RGB")).mean

            # Flat region: no triangle reaches the threshold, colour is untouched.
            for channel, original in zip(flat, self.FLAT_COLOR):
                self.assertAlmostEqual(channel, original, delta=2)
            # Steep region: every covering triangle is >= threshold, so the
            # 25% red blend applies: 0.75*original + 0.25*(255,0,0).
            expected_red = 0.75 * self.FLAT_COLOR[0] + 0.25 * 255
            expected_green = 0.75 * self.FLAT_COLOR[1]
            self.assertAlmostEqual(steep[0], expected_red, delta=6)
            self.assertAlmostEqual(steep[1], expected_green, delta=6)
            self.assertGreater(steep[0] - flat[0], 40)
            self.assertLess(steep[1] - flat[1], -20)

    def test_raising_the_threshold_flags_fewer_triangles(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dataset = self._build_dataset(root)
            tile_path = dataset / "tiles" / "0" / "0" / "0.trn"

            def run(threshold: str) -> str:
                output = root / f"out-{threshold}.png"
                result = subprocess.run(
                    [sys.executable, str(DEFAULT_TOOL_PATH), str(tile_path),
                     "--output", str(output), "--threshold", threshold],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout

            lenient = run("1.5")
            strict = run("10.0")

            def flagged_count(stdout: str) -> int:
                return int(stdout.split(":", 1)[1].strip().split("/")[0])

            self.assertGreater(flagged_count(lenient), flagged_count(strict))

    def test_in_place_single_tile_overwrites_its_own_imagery(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dataset = self._build_dataset(root)
            tile_path = dataset / "tiles" / "0" / "0" / "0.trn"
            imagery_path = dataset / "imagery" / "0" / "0" / "0.png"
            with Image.open(imagery_path) as before_image:
                original_mode = before_image.mode
                before = ImageStat.Stat(before_image.convert("RGB")).mean

            result = subprocess.run(
                [sys.executable, str(DEFAULT_TOOL_PATH), str(tile_path), "--in-place"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            with Image.open(imagery_path) as after_image:
                self.assertEqual(after_image.mode, original_mode)
                after = ImageStat.Stat(after_image.convert("RGB")).mean
            self.assertNotEqual(before, after)  # the steep half should now be reddened

    def test_directory_mode_processes_every_tile_and_requires_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dataset = self._build_dataset(root)

            without_flag = subprocess.run(
                [sys.executable, str(DEFAULT_TOOL_PATH), str(dataset)],
                capture_output=True, text=True)
            self.assertNotEqual(without_flag.returncode, 0)
            self.assertIn("--in-place", without_flag.stderr)

            imagery_path = dataset / "imagery" / "0" / "0" / "0.png"
            before = ImageStat.Stat(Image.open(imagery_path).convert("RGB")).mean

            result = subprocess.run(
                [sys.executable, str(DEFAULT_TOOL_PATH), str(dataset), "--in-place"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("processed 1 tiles", result.stdout)

            after = ImageStat.Stat(Image.open(imagery_path).convert("RGB")).mean
            self.assertNotEqual(before, after)

    def test_missing_dataset_root_produces_a_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dataset = self._build_dataset(root)
            tile_path = dataset / "tiles" / "0" / "0" / "0.trn"
            result = subprocess.run(
                [sys.executable, str(DEFAULT_TOOL_PATH), str(tile_path),
                 "--dataset-root", str(root / "does-not-exist"),
                 "--output", str(root / "out.png")],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("detect_texture_stretch:", result.stderr)


def main() -> None:
    unittest.main(argv=[sys.argv[0]], verbosity=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", type=Path, default=DEFAULT_TOOL_PATH)
    parser.add_argument("--terrain-tool", type=Path, default=DEFAULT_TERRAIN_TOOL_PATH)
    parsed, remaining = parser.parse_known_args()
    DEFAULT_TOOL_PATH = parsed.tool.resolve()
    DEFAULT_TERRAIN_TOOL_PATH = parsed.terrain_tool.resolve()
    TERRAIN_TOOL = load_module(DEFAULT_TERRAIN_TOOL_PATH, "terrain_tiles_under_test")
    TOOL = load_module(DEFAULT_TOOL_PATH, "detect_texture_stretch_under_test")
    sys.argv = [sys.argv[0], *remaining]
    main()
