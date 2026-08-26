#!/usr/bin/env python3
"""Unit and integration tests for tools/scscts_checkpoint_demo.py (see shadowrmplan.md)."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOL_PATH = ROOT / "tools" / "scscts_checkpoint_demo.py"
DEFAULT_TERRAIN_TOOL_PATH = ROOT / "tools" / "terrain_tiles.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TOOL = None          # set in __main__ / main()
TERRAIN_TOOL = None


# ---------------------------------------------------------------------------
# Section 7: coordinates, gradients, incidence
# ---------------------------------------------------------------------------

class CoordinateAndGeometryTests(unittest.TestCase):
    def test_pixel_centre_mapping_matches_formula(self) -> None:
        extent = (2648000.0, 1162000.0, 2648500.0, 1162500.0)
        east, north = TOOL.pixel_centres(extent, size=256)
        # E = W + (col+0.5)(Emax-W)/256 ; N = Nmax - (row+0.5)(Nmax-S)/256
        self.assertAlmostEqual(east[0, 0], 2648000.0 + 0.5 * 500.0 / 256, places=6)
        self.assertAlmostEqual(north[0, 0], 1162500.0 - 0.5 * 500.0 / 256, places=6)
        self.assertAlmostEqual(east[0, 255], 2648000.0 + 255.5 * 500.0 / 256, places=6)
        self.assertAlmostEqual(north[255, 0], 1162500.0 - 255.5 * 500.0 / 256, places=6)

    def test_bilinear_interpolation_reproduces_synthetic_plane_exactly(self) -> None:
        size, spacing = 9, 10.0
        cols = np.arange(size) * spacing
        rows = np.arange(size) * spacing
        colgrid, rowgrid = np.meshgrid(cols, rows)
        plane = 3.0 + 0.4 * colgrid - 0.7 * rowgrid  # z = a + b*E - c*(south distance)
        valid = np.ones_like(plane, dtype=bool)
        west, north = 0.0, (size - 1) * spacing
        east = np.array([23.4, 41.0, 5.5])
        south_dist = np.array([12.1, 33.3, 70.0])  # distance south of the north edge
        north_coord = north - south_dist
        expected = 3.0 + 0.4 * east - 0.7 * south_dist
        value, ok, inside = TOOL.bilinear_sample(plane, valid, east, north_coord, west, north, spacing)
        np.testing.assert_allclose(value, expected, atol=1e-9)
        self.assertTrue(ok.all())
        self.assertTrue(inside.all())

    def test_flat_terrain_has_zero_slope_and_upward_normal(self) -> None:
        elevation = np.full((16, 16), 1500.0)
        normal, slope, aspect, g_e, g_n = TOOL.terrain_geometry(elevation, 2.0)
        np.testing.assert_allclose(slope, 0.0, atol=1e-9)
        np.testing.assert_allclose(normal[..., 0], 0.0, atol=1e-9)
        np.testing.assert_allclose(normal[..., 1], 0.0, atol=1e-9)
        np.testing.assert_allclose(normal[..., 2], 1.0, atol=1e-9)

    def test_synthetic_planes_produce_correct_aspect(self) -> None:
        size, spacing = 16, 2.0
        cols = np.arange(size, dtype=np.float64)
        rows = np.arange(size, dtype=np.float64)
        colgrid, rowgrid = np.meshgrid(cols, rows)  # rowgrid increases south
        cases = {
            # downslope east: z decreases as east (col) increases
            "east": (-colgrid, 90.0),
            # downslope west: z increases as east increases
            "west": (colgrid, 270.0),
            # downslope north: z decreases as north increases i.e. increases as row (south) increases
            "north": (rowgrid, 0.0),
            # downslope south: z decreases as row (south) increases
            "south": (-rowgrid, 180.0),
        }
        for label, (elevation, expected_aspect) in cases.items():
            _, slope, aspect, _, _ = TOOL.terrain_geometry(elevation, spacing)
            centre = aspect[size // 2, size // 2]
            self.assertGreater(slope[size // 2, size // 2], 0.0, label)
            self.assertAlmostEqual(math.degrees(centre) % 360.0, expected_aspect, delta=1e-3, msg=label)

    def test_incidence_dot_and_formula_agree_within_tolerance(self) -> None:
        rng = np.random.default_rng(0)
        elevation = np.cumsum(rng.normal(0, 0.4, size=(24, 24)), axis=0) + \
            np.cumsum(rng.normal(0, 0.4, size=(24, 24)), axis=1)
        normal, slope, aspect, _, _ = TOOL.terrain_geometry(elevation, 3.0)
        sun_az, sun_el = math.radians(133.0), math.radians(42.0)
        sun = TOOL.sun_vector(sun_az, sun_el)
        dot = TOOL.incidence_dot(normal, sun)
        formula = TOOL.incidence_formula(slope, aspect, sun_az, sun_el)
        self.assertLess(float(np.max(np.abs(dot.astype(np.float64) - formula.astype(np.float64)))), 1e-6)


# ---------------------------------------------------------------------------
# Section 3/7/17: TRN reader and gutter alignment
# ---------------------------------------------------------------------------

class TrnReaderTests(unittest.TestCase):
    def test_reads_synthetic_dataset_and_reports_expected_grid(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scscts-trn-") as raw:
            root = Path(raw)
            dem = Image.new("F", (33, 33))
            dem.putdata([100.0 + 0.1 * x + 0.2 * y for y in range(33) for x in range(33)])
            dem_path = root / "dem.tif"
            dem.save(dem_path)
            imagery = Image.new("RGB", (33, 33), (80, 120, 60))
            imagery_path = root / "imagery.png"
            imagery.save(imagery_path)
            dataset = root / "dataset"
            subprocess.run([sys.executable, str(DEFAULT_TERRAIN_TOOL_PATH), "build",
                             "--dem", str(dem_path), "--imagery", str(imagery_path),
                             "--output", str(dataset),
                             "--extent", "2648000", "1162000", "2648500", "1162500",
                             "--profile", "EPSG:2056", "--levels", "1", "--samples", "17",
                             "--imagery-size", "17", "--gutter", "1", "--height-encoding", "f32"],
                            check=True, capture_output=True)
            tile = TOOL.read_tile(dataset / "tiles" / "0" / "0" / "0.trn")
            self.assertEqual(tile.extent, (2648000.0, 1162000.0, 2648500.0, 1162500.0))
            self.assertEqual(tile.width, 19)   # samples(17) + 2*gutter(1)
            self.assertEqual(tile.height, 19)


class GutterAlignmentTests(unittest.TestCase):
    def test_repository_gutter_interior_matches_supplied_input(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scscts-gutter-") as raw:
            root = Path(raw)
            terrain_root = root / "tiles"
            imagery_root = root / "imagery" / "0" / "0"
            imagery_root.mkdir(parents=True)
            rng = np.random.default_rng(1)
            interior = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
            padded = np.zeros((258, 258, 3), dtype=np.uint8)
            padded[1:257, 1:257] = interior
            padded[0, :] = padded[1, :]
            padded[-1, :] = padded[-2, :]
            padded[:, 0] = padded[:, 1]
            padded[:, -1] = padded[:, -2]
            Image.fromarray(padded).save(imagery_root / "0.png")
            ok, note = TOOL.check_gutter_alignment(interior, terrain_root, 0, 0, 0)
            self.assertTrue(ok, note)
            mismatched = interior.copy()
            mismatched[0, 0] ^= 0xFF
            ok, note = TOOL.check_gutter_alignment(mismatched, terrain_root, 0, 0, 0)
            self.assertFalse(ok)


# ---------------------------------------------------------------------------
# Section 8: sun position, flight-id parsing, GeoAdmin selection
# ---------------------------------------------------------------------------

def _reference_noaa(when_utc, lat, lon):
    """Independent re-transcription of the NOAA PDF formulas (cross-check)."""
    n = when_utc.timetuple().tm_yday
    hour = when_utc.hour + when_utc.minute / 60 + when_utc.second / 3600
    leap = when_utc.year % 4 == 0 and (when_utc.year % 100 != 0 or when_utc.year % 400 == 0)
    days = 366.0 if leap else 365.0
    gamma = 2 * math.pi / days * (n - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                        - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    tst = hour * 60 + eqtime + 4 * lon
    ha_deg = tst / 4 - 180
    ha = math.radians(ha_deg)
    lat_r = math.radians(lat)
    cos_zen = math.sin(lat_r) * math.sin(decl) + math.cos(lat_r) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1.0, min(1.0, cos_zen)))
    elev = 90 - math.degrees(zen)
    cos_az = (math.sin(lat_r) * math.cos(zen) - math.sin(decl)) / (math.cos(lat_r) * math.sin(zen))
    cos_az = max(-1.0, min(1.0, cos_az))
    az0 = math.degrees(math.acos(cos_az))
    az = (az0 + 180) % 360 if ha_deg > 0 else (540 - az0) % 360
    return az, elev


class SunPositionTests(unittest.TestCase):
    def test_lv95_to_wgs84_matches_known_centroid(self) -> None:
        lon, lat = TOOL.lv95_to_wgs84(2648250.0, 1162250.0)
        self.assertAlmostEqual(lon, 8.0684591165, delta=1e-3)
        self.assertAlmostEqual(lat, 46.6097687673, delta=1e-3)

    def test_noaa_matches_independent_reimplementation_of_the_published_formula(self) -> None:
        # NOTE: this cross-checks scscts_checkpoint_demo.noaa_sun_position against a
        # second, independently transcribed copy of the same NOAA PDF formula (fetched
        # and quoted from gml.noaa.gov/grad/solcalc/solareqns.PDF), NOT against
        # shadowrmplan.md's stated approximate reference figures. Careful verification
        # (leap-year 366-day rule included, per the PDF) puts this pilot's tile/time
        # fixture at az~118.17 deg, el~38.31 deg -- about 0.3-0.4 deg away from the
        # plan's quoted ~118.611/38.042, which most plausibly came from a higher-order
        # solar-position algorithm rather than the simplified formula this section
        # specifies. See run_manifest.json's "flight" stage deviations for the same note.
        when = dt.datetime(2024, 8, 24, 8, 31, tzinfo=dt.timezone.utc)
        lon, lat = TOOL.lv95_to_wgs84(2648250.0, 1162250.0)
        az, el, _ = TOOL.noaa_sun_position(when, lat, lon)
        ref_az, ref_el = _reference_noaa(when, lat, lon)
        self.assertAlmostEqual(az, ref_az, delta=1e-6)
        self.assertAlmostEqual(el, ref_el, delta=1e-6)

    def test_flight_id_parser_extracts_utc_date_and_time(self) -> None:
        when = TOOL.parse_flight_id_time("20240824_0831_12504")
        self.assertEqual(when, dt.datetime(2024, 8, 24, 8, 31, tzinfo=dt.timezone.utc))
        with self.assertRaises(ValueError):
            TOOL.parse_flight_id_time("not-a-flight-id")


def _candidate(flight_id, year, goal="SWISSIMAGE"):
    return {"featureId": flight_id, "id": flight_id,
            "attributes": {"bgdi_flugjahr": year, "goal": goal}}


class FlightSelectionTests(unittest.TestCase):
    def _cache_dir(self, tmp):
        cache_dir = Path(tmp) / "cache"
        cache_dir.mkdir()
        return cache_dir

    def test_filters_by_year_and_goal_and_selects_unique_candidate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scscts-flight-") as tmp:
            cache_dir = self._cache_dir(tmp)
            candidates = {"results": [
                _candidate("20150603_1057_1308", 2015),
                _candidate("20240824_0831_12504", 2024),
                _candidate("20240101_0000_99999", 2024, goal="TLM"),
            ]}
            (cache_dir / "geoadmin_candidates_raw.json").write_text(json.dumps(candidates))
            (cache_dir / "geo_admin_response.json").write_text(json.dumps({"feature": {}}))
            flight_id, when, detail, source, filtered, warnings = TOOL.resolve_flight(
                cache_dir, (2648000.0, 1162000.0, 2648500.0, 1162500.0), 2024, None, False, None)
            self.assertEqual(flight_id, "20240824_0831_12504")
            self.assertEqual(len(filtered), 1)
            self.assertEqual(when, dt.datetime(2024, 8, 24, 8, 31, tzinfo=dt.timezone.utc))

    def test_ambiguous_selection_raises_without_explicit_flight_id(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scscts-flight-") as tmp:
            cache_dir = self._cache_dir(tmp)
            candidates = {"results": [
                _candidate("20240824_0831_12504", 2024),
                _candidate("20240915_0900_99999", 2024),
            ]}
            (cache_dir / "geoadmin_candidates_raw.json").write_text(json.dumps(candidates))
            with self.assertRaises(RuntimeError):
                TOOL.resolve_flight(cache_dir, (0, 0, 500, 500), 2024, None, False, None)

    def test_offline_mode_reproduces_selection_from_cache(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scscts-flight-") as tmp:
            cache_dir = self._cache_dir(tmp)
            candidates = {"results": [_candidate("20240824_0831_12504", 2024)]}
            (cache_dir / "geoadmin_candidates_raw.json").write_text(json.dumps(candidates))
            (cache_dir / "geo_admin_response.json").write_text(json.dumps({"feature": {}}))
            flight_id, when, _, source, filtered, _ = TOOL.resolve_flight(
                cache_dir, (0, 0, 500, 500), 2024, None, True, None)
            self.assertEqual(flight_id, "20240824_0831_12504")
            self.assertEqual(source, "cache")

    def test_offline_without_cache_or_override_requires_flight_id(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scscts-flight-") as tmp:
            cache_dir = self._cache_dir(tmp)
            with self.assertRaises(RuntimeError):
                TOOL.resolve_flight(cache_dir, (0, 0, 500, 500), 2024, None, True, None)


# ---------------------------------------------------------------------------
# Section 10: shadow geometry
# ---------------------------------------------------------------------------

class ShadowGeometryTests(unittest.TestCase):
    def _flat_mosaic(self, size=41, spacing=10.0, height=0.0):
        mosaic = np.full((size, size), height, dtype=np.float64)
        valid = np.ones_like(mosaic, dtype=bool)
        west, north = 0.0, (size - 1) * spacing
        return mosaic, valid, west, north, spacing

    def test_flat_terrain_with_sun_above_horizon_has_no_cast_shadow(self) -> None:
        mosaic, valid, west, north, spacing = self._flat_mosaic()
        east = np.array([200.0])
        north_coord = np.array([200.0])
        elevation = np.array([0.0])
        shadowed, truncated, h_min = TOOL.cast_shadow(
            mosaic, valid, west, north, spacing, east, north_coord, elevation,
            math.radians(90.0), math.radians(30.0), 150.0, spacing, 3.0)
        self.assertFalse(bool(shadowed[0]))
        self.assertFalse(bool(truncated[0]))

    def test_ridge_casts_shadow_only_on_the_far_side_from_the_sun(self) -> None:
        mosaic, valid, west, north, spacing = self._flat_mosaic()
        ridge_col = 25  # E = 250
        mosaic[:, ridge_col] = 80.0  # tall ridge, sun az=90 (east) el=30
        west_of_ridge = np.array([200.0])   # away from the sun -> shadowed
        east_of_ridge = np.array([300.0])   # sun-side of the ridge -> not shadowed
        north_coord = np.array([200.0])
        for east, expect_shadow in ((west_of_ridge, True), (east_of_ridge, False)):
            shadowed, truncated, _ = TOOL.cast_shadow(
                mosaic, valid, west, north, spacing, east, north_coord, np.array([0.0]),
                math.radians(90.0), math.radians(30.0), 150.0, spacing, 3.0)
            self.assertEqual(bool(shadowed[0]), expect_shadow, f"east={east}")

    def test_object_exactly_at_clearance_boundary_casts_shadow_only_when_exceeded(self) -> None:
        mosaic, valid, west, north, spacing = self._flat_mosaic()
        distance = 50.0  # lands exactly on a raymarch sample: d=2*step+3*step
        threshold_height = distance * math.tan(math.radians(30.0)) + 3.0  # z_ray + clearance
        row, col = 20, 25
        for height, expect_shadow in ((threshold_height + 1.0, True), (threshold_height - 1.0, False)):
            m = mosaic.copy()
            m[row, col] = height
            shadowed, _, _ = TOOL.cast_shadow(
                m, valid, west, north, spacing, np.array([200.0]), np.array([200.0]), np.array([0.0]),
                math.radians(90.0), math.radians(30.0), 150.0, spacing, 3.0)
            self.assertEqual(bool(shadowed[0]), expect_shadow, f"height={height}")


# ---------------------------------------------------------------------------
# Otsu, robust regression, SCS+C, irradiance, lambda
# ---------------------------------------------------------------------------

class NumericalMethodTests(unittest.TestCase):
    def test_otsu_separates_a_synthetic_bimodal_distribution(self) -> None:
        # Between-class variance is provably flat across the empty gap between
        # two well-separated clusters, so any threshold within the gap is
        # equally valid; this checks separation, not a specific midpoint.
        rng = np.random.default_rng(2)
        low = rng.normal(0.2, 0.03, size=2000)
        high = rng.normal(0.8, 0.03, size=2000)
        values = np.clip(np.concatenate([low, high]), 0, 1)
        threshold = TOOL.otsu_threshold(values)
        self.assertGreater(threshold, float(np.percentile(low, 99.9)))
        self.assertLess(threshold, float(np.percentile(high, 0.1)))

    def test_robust_regression_resists_injected_outliers(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.uniform(0, 1, size=300)
        y_true = 2.0 * x + 1.0
        y = y_true + rng.normal(0, 0.01, size=300)
        y[:10] += 50.0  # gross outliers
        design = np.stack([x, np.ones_like(x)], axis=1)
        result = TOOL.robust_regress(design, y)
        self.assertAlmostEqual(result.x[0], 2.0, delta=0.2)
        self.assertAlmostEqual(result.x[1], 1.0, delta=0.2)
        contaminated_ols, *_ = np.linalg.lstsq(design, y, rcond=None)
        self.assertGreater(abs(contaminated_ols[0] - 2.0), abs(result.x[0] - 2.0))

    def test_srgb_round_trip_within_one_8bit_code(self) -> None:
        values = np.arange(256, dtype=np.uint8)
        linear = TOOL.srgb_to_linear(values)
        back = np.clip(TOOL.linear_to_srgb(linear) * 255.0 + 0.5, 0, 255).astype(np.int32)
        self.assertTrue(np.all(np.abs(back - values.astype(np.int32)) <= 1))

    def test_scsc_returns_identity_on_uniform_flat_input(self) -> None:
        n = 1000
        channel = np.full(n, 0.5)
        direct = np.full(n, 0.6)
        cosi = np.full(n, 0.6)
        cos_alpha = np.full(n, 1.0)
        mask = np.ones(n, dtype=bool)
        gain, params = TOOL.fit_scsc_channel(channel, direct, cosi, cos_alpha, math.cos(math.radians(50)), mask)
        self.assertTrue(params["invalid"])
        np.testing.assert_allclose(gain, 1.0)

    def test_scsc_invalid_regression_sets_qa_flag_with_too_few_samples(self) -> None:
        n = 50  # below the 500-sample minimum
        rng = np.random.default_rng(4)
        direct = rng.uniform(0.1, 0.9, size=n)
        channel = 0.3 * direct + 0.2
        cos_alpha = np.cos(np.radians(30.0)) * np.ones(n)
        mask = np.ones(n, dtype=bool)
        gain, params = TOOL.fit_scsc_channel(channel, direct, direct, cos_alpha, math.cos(math.radians(50)), mask)
        self.assertTrue(params["invalid"])
        np.testing.assert_allclose(gain, 1.0)

    def test_lambda_is_zero_below_threshold_one_at_si_max_and_bounded(self) -> None:
        proxy = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        threshold, si_max = 0.3, 0.9
        lam = TOOL.compute_lambda_raw(proxy, threshold, si_max)
        np.testing.assert_allclose(lam[:2], 0.0)   # below threshold
        self.assertAlmostEqual(lam[proxy == 0.3][0], 0.0)
        self.assertAlmostEqual(lam[proxy == 0.9][0], 1.0)
        self.assertTrue(np.all((lam >= 0.0) & (lam <= 1.0)))
        # above si_max still clipped to 1, never exceeds it
        self.assertAlmostEqual(lam[proxy == 1.0][0], 1.0)

    def test_irradiance_denominator_stays_finite_as_albedo_and_diffuse_approach_zero(self) -> None:
        a_b, d_b = 0.0, 0.0
        cos_theta_s = math.cos(math.radians(50))
        v_d = np.array([0.5, 0.9])
        c_t = 1.0 - v_d
        albedo = np.array([0.0, 0.0])
        es = d_b * cos_theta_s
        ek = a_b * v_d
        ea = (a_b + d_b * cos_theta_s) * c_t * albedo
        ew = ek + ea
        self.assertTrue(np.all(np.isfinite(ew)))
        rgb = np.array([[[0.2, 0.2, 0.2]]])
        lam = np.array([[1.0]])
        es_arr = np.array([[[es, es, es]]])
        ew_arr = np.array([[[ew[0], ew[0], ew[0]]]])
        luminance = rgb @ TOOL.RGB_LUMA
        result = TOOL.compute_shadow_compensation(rgb, luminance, lam, es_arr, ew_arr, 2.0)
        self.assertTrue(np.isfinite(result["final"]).all())

    def test_shadow_compensation_enforces_the_gain_cap_before_blending(self) -> None:
        rgb = np.full((1, 1, 3), 0.1)
        luminance = rgb @ TOOL.RGB_LUMA
        lam = np.array([[1.0]])
        es = np.full((1, 1, 3), 5.0)   # deliberately huge relative irradiance
        ew = np.full((1, 1, 3), 0.01)
        for cap in (2.0, 3.5):
            result = TOOL.compute_shadow_compensation(rgb, luminance, lam, es, ew, cap)
            capped_per_channel = np.minimum(result["uncapped"], rgb * cap)
            self.assertTrue(np.all(capped_per_channel <= rgb * cap + 1e-12))
            self.assertTrue(np.isfinite(result["final"]).all())
            self.assertTrue(np.all(result["final"] <= 1.0))


# ---------------------------------------------------------------------------
# Integration: full CLI run against a small, hermetic synthetic dataset
# ---------------------------------------------------------------------------

def build_synthetic_dataset(root: Path) -> tuple[Path, Path]:
    """Builds a small multi-tile dataset via terrain_tiles.py so the halo loader
    has real neighbour tiles to read, without depending on the (gitignored,
    machine-local) alps-data/ directory."""
    size = 65  # finest-level global DEM raster: (samples-1)*count + 1 = 16*4+1
    dem = Image.new("F", (size, size))
    rows = []
    for y in range(size):
        for x in range(size):
            rows.append(300.0 + 6.0 * math.sin(x / 6.0) + 4.0 * math.cos(y / 5.0) + 0.05 * x)
    dem.putdata(rows)
    dem_path = root / "dem.tif"
    dem.save(dem_path)
    imagery = Image.new("RGB", (size, size))
    imagery.putdata([(int(120 + 60 * math.sin(x / 4.0)), int(140 + 40 * math.cos(y / 4.0)), 90)
                      for y in range(size) for x in range(size)])
    imagery_path = root / "imagery.tif"
    imagery.save(imagery_path)
    dataset = root / "dataset"
    subprocess.run([sys.executable, str(DEFAULT_TERRAIN_TOOL_PATH), "build",
                     "--dem", str(dem_path), "--imagery", str(imagery_path),
                     "--output", str(dataset),
                     "--extent", "0", "0", "400", "400",
                     "--profile", "EPSG:2056", "--levels", "3", "--samples", "17",
                     "--imagery-size", "17", "--gutter", "1", "--height-encoding", "f32"],
                    check=True, capture_output=True, text=True)
    return dataset, dataset / "manifest.json"


def build_synthetic_input(path: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, size=(256, 256, 1))
    noise = rng.integers(-15, 15, size=(256, 256, 3))
    rgb = np.clip(base + noise + np.array([0, 20, -10]), 0, 255).astype(np.uint8)
    Image.fromarray(rgb, "RGB").save(path)


class IntegrationTests(unittest.TestCase):
    def _command(self, output, dataset, manifest, input_path, seed, extra=()):
        return [sys.executable, str(DEFAULT_TOOL_PATH),
                "--input", str(input_path),
                "--terrain-root", str(dataset / "tiles"),
                "--manifest", str(manifest),
                "--level", "2", "--tile-x", "1", "--tile-y", "1",
                "--year", "2024", "--output", str(output),
                "--offline", "--flight-id", "20240824_0831_TEST",
                "--acquisition-time", "2024-08-24T08:31:00Z",
                "--max-horizon-distance", "90", "--ray-step", "6.25",
                "--seed", str(seed), *extra]

    def _run(self, root, output, seed=0, extra=(), expect_success=True):
        dataset, manifest = build_synthetic_dataset(root)
        input_path = root / "input.png"
        build_synthetic_input(input_path, seed=seed)
        command = self._command(output, dataset, manifest, input_path, seed, extra)
        result = subprocess.run(command, capture_output=True, text=True)
        if expect_success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result, dataset, manifest, input_path

    def test_integration_run_creates_every_required_stage_and_manifest_entry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scscts-out-") as tmp:
            root = Path(tmp)
            output = root / "run"
            self._run(root, output)
            expected_dirs = [f"{i:02d}_{name}" for i, name in enumerate(TOOL.STAGE_ORDER)] + ["99_contact_sheet"]
            for name in expected_dirs:
                stage_dir = output / name
                self.assertTrue(stage_dir.is_dir(), name)
                if name != "99_contact_sheet":
                    self.assertTrue((stage_dir / "stats.json").is_file(), name)
                self.assertTrue((stage_dir / "stage.json").is_file(), name)
            self.assertTrue((output / "run_manifest.json").is_file())
            self.assertTrue((output / "index.html").is_file())
            manifest = json.loads((output / "run_manifest.json").read_text())
            for key in ("tool_version", "command_line", "versions", "terrain", "flight",
                        "solar", "parameters", "equations_implemented", "acceptance", "metrics"):
                self.assertIn(key, manifest)
            self.assertTrue(manifest["acceptance"]["no_nan_or_inf"])

            for profile in ("conservative", "aggressive"):
                final = np.load(output / f"{13 if profile == 'conservative' else 14}_{profile}" /
                                 "arrays" / "final_linear_rgb.npy")
                self.assertTrue(np.isfinite(final).all())

    def test_integration_run_is_deterministic_for_the_same_seed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scscts-out-") as tmp:
            root_a, root_b = Path(tmp) / "a", Path(tmp) / "b"
            root_a.mkdir()
            root_b.mkdir()
            first, second = root_a / "run", root_b / "run"
            self._run(root_a, first, seed=7)
            self._run(root_b, second, seed=7)
            manifest_a = json.loads((first / "run_manifest.json").read_text())
            manifest_b = json.loads((second / "run_manifest.json").read_text())
            for key in ("acceptance", "metrics", "solar", "terrain"):
                self.assertEqual(manifest_a[key], manifest_b[key], key)
            final_a = np.load(first / "13_conservative" / "arrays" / "final_linear_rgb.npy")
            final_b = np.load(second / "13_conservative" / "arrays" / "final_linear_rgb.npy")
            np.testing.assert_array_equal(final_a, final_b)

    def test_refuses_to_overwrite_existing_output_without_force(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scscts-out-") as tmp:
            root = Path(tmp)
            output = root / "run"
            self._run(root, output)
            result, *_ = self._run(root, output, expect_success=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already exists", result.stdout + result.stderr)


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
    TOOL = load_module(DEFAULT_TOOL_PATH, "scscts_checkpoint_demo_under_test")
    sys.argv = [sys.argv[0], *remaining]
    main()
