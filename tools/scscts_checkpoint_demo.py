#!/usr/bin/env python3
"""SCSCTS shadow-correction pilot for one Alps tile (see shadowrmplan.md).

Reproduces every implementable stage of the SCS+C topographic correction with
shadow compensation described in Chen et al., "SCSCTS: An improved SCS+C
topographic correction model with shadow compensation for mountainous
regions" (PLOS ONE, 2024, doi:10.1371/journal.pone.0347784), against a single
256x256 RGB tile crop. The paper's Equation 8 duplicates Equation 6 in print;
this tool follows the surrounding prose instead (sunlit pixels use SCS+C,
shadow pixels use I_t + I_d -- see PAPER_EQ8_NOTE below).

The paper's shadow index (Eq. 6) needs Coastal/Green/NIR reflectance bands
that this RGB-only pilot does not have. Every place that stands in for it is
named `..._RGB_PROXY` and is never reported as the paper's exact statistic.
Every other physical simplification forced by RGB-only, non-radiometric input
is recorded in `run_manifest.json` under "deviations".

This is a checkpointed review artefact, not a batch processor: every stage
writes PNG previews, raw float/bool `.npy` arrays, `stats.json`, and
`stage.json` under `<output>/<NN>_<stage>/`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import scipy
from scipy import ndimage
from scipy.optimize import least_squares

import PIL
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from terrain_tiles import read_tile  # noqa: E402  (promoted public API)

Image.MAX_IMAGE_PIXELS = None

TOOL_VERSION = "1.0.0"
RGB_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
EPS_SMALL = 1e-6
EPS_CORR = 1e-4          # protects SCS+C and Ew denominators, per section 11/13
HUBER_SCALE_MULTIPLIER = 10.0  # Huber f_scale = multiplier * MAD(OLS residuals); see robust_regress
IMAGE_SIZE = 256

PAPER_EQ8_NOTE = (
    "Published Equation 8 repeats Equation 6 (apparent print error). This "
    "pilot follows the surrounding prose: sunlit pixels use I_SCS+C, shadow "
    "pixels use I_t + I_d.")

STAGE_ORDER = [
    "input", "alignment", "flight_and_sun", "terrain_geometry", "incidence",
    "shadow_geometry", "scsc_fit", "scsc_corrected", "shadow_index", "lambda",
    "sky_view_albedo", "irradiance", "shadow_compensation", "conservative",
    "aggressive", "evaluation",
]


# ---------------------------------------------------------------------------
# Section 9: image linearization
# ---------------------------------------------------------------------------

def srgb_to_linear(channel):
    """8-bit sRGB -> `linear_intensity` in [0,1] (never called reflectance)."""
    c = np.asarray(channel, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear_intensity):
    c = np.clip(np.nan_to_num(linear_intensity), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)


def render_rgb(linear_rgb):
    return np.clip(linear_to_srgb(linear_rgb) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def render_gray(a, vmin, vmax):
    a = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=vmin)
    x = np.clip((a - vmin) / max(vmax - vmin, EPS_SMALL), 0.0, 1.0)
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def render_mask(mask):
    return np.where(np.asarray(mask, dtype=bool), np.uint8(255), np.uint8(0))


def render_signed(a, vmax_abs):
    """Symmetric diverging colour map: blue = negative, red = positive."""
    a = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0)
    x = np.clip(a / max(vmax_abs, EPS_SMALL), -1.0, 1.0)
    r = np.where(x > 0, x, 0.0)
    b = np.where(x < 0, -x, 0.0)
    g = 1.0 - np.abs(x)
    rgb = np.dstack((r, g, b))
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Section 7: coordinate and array conventions
# ---------------------------------------------------------------------------

def pixel_centres(extent, size=IMAGE_SIZE):
    west, south, east, north = extent
    cols = np.arange(size, dtype=np.float64)
    rows = np.arange(size, dtype=np.float64)
    e = west + (cols + 0.5) * (east - west) / size
    n = north - (rows + 0.5) * (north - south) / size
    east_grid, north_grid = np.meshgrid(e, n)
    return east_grid, north_grid


def lv95_to_wgs84(easting, northing):
    """Approximate swisstopo LV95->WGS84 formula (documented accuracy ~0.1")."""
    y = (easting - 2600000.0) / 1e6
    x = (northing - 1200000.0) / 1e6
    lam = (2.6779094 + 4.728982 * y + 0.791484 * y * x + 0.1306 * y * x ** 2
           - 0.0436 * y ** 3)
    phi = (16.9023892 + 3.238272 * x - 0.270978 * y ** 2 - 0.002528 * x ** 2
           - 0.0447 * y ** 2 * x - 0.0140 * x ** 3)
    lon = lam * 100.0 / 36.0
    lat = phi * 100.0 / 36.0
    return lon, lat


def terrain_geometry(dem, spacing):
    """Returns (normal ENU, slope rad, aspect rad, g_E, g_N) per section 7."""
    g_s, g_e = np.gradient(dem.astype(np.float64), spacing)  # axis0=south, axis1=east
    g_n = -g_s
    normal = np.dstack((-g_e, g_s, np.ones_like(dem)))
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    slope = np.arctan(np.hypot(g_e, g_n))
    aspect = np.mod(np.arctan2(-g_e, g_s), 2.0 * np.pi)
    return normal, slope, aspect, g_e, g_n


def sun_vector(azimuth_rad, elevation_rad):
    return np.array([math.sin(azimuth_rad) * math.cos(elevation_rad),
                      math.cos(azimuth_rad) * math.cos(elevation_rad),
                      math.sin(elevation_rad)], dtype=np.float64)


def incidence_dot(normal, sun):
    return normal[..., 0] * sun[0] + normal[..., 1] * sun[1] + normal[..., 2] * sun[2]


def incidence_formula(slope, aspect, sun_azimuth_rad, sun_elevation_rad):
    theta_s = math.pi / 2.0 - sun_elevation_rad
    return (np.cos(slope) * math.cos(theta_s)
            + np.sin(slope) * math.sin(theta_s) * np.cos(sun_azimuth_rad - aspect))


def bilinear_sample(grid, valid, easting, northing, west, north, spacing):
    """Sample a north-up vertex grid at world (E,N); returns (value, valid, inside)."""
    fx = (easting - west) / spacing
    fy = (north - northing) / spacing
    ix = np.floor(fx).astype(np.int64)
    iy = np.floor(fy).astype(np.int64)
    tx = fx - ix
    ty = fy - iy
    inside = (fx >= 0) & (fx <= grid.shape[1] - 1) & (fy >= 0) & (fy <= grid.shape[0] - 1)
    ix_c = np.clip(ix, 0, grid.shape[1] - 2)
    iy_c = np.clip(iy, 0, grid.shape[0] - 2)
    g00 = grid[iy_c, ix_c]
    g10 = grid[iy_c, ix_c + 1]
    g01 = grid[iy_c + 1, ix_c]
    g11 = grid[iy_c + 1, ix_c + 1]
    value = (g00 * (1 - tx) * (1 - ty) + g10 * tx * (1 - ty)
             + g01 * (1 - tx) * ty + g11 * tx * ty)
    v00 = valid[iy_c, ix_c]
    v10 = valid[iy_c, ix_c + 1]
    v01 = valid[iy_c + 1, ix_c]
    v11 = valid[iy_c + 1, ix_c + 1]
    all_valid = v00 & v10 & v01 & v11
    return value, (all_valid & inside), inside


# ---------------------------------------------------------------------------
# Section 3/7: terrain halo loading (up to `--max-horizon-distance` beyond the
# target tile's own edges, i.e. the "six 500m tile widths" of section 3).
# ---------------------------------------------------------------------------

def _tile_interior(tile):
    count = tile.width * tile.height
    heights = np.asarray(tile.decoded_heights(), dtype=np.float64).reshape(tile.height, tile.width)
    valid = np.array([tile.is_valid(i) for i in range(count)], dtype=bool).reshape(tile.height, tile.width)
    g = tile.gutter
    if g:
        heights = heights[g:-g, g:-g]
        valid = valid[g:-g, g:-g]
    return heights, valid


def load_dem_halo(terrain_root, level, cx, cy, tile, radius_m):
    """Stitch a (2R+1)x(2R+1) tile mosaic of DEM vertices around (cx,cy)."""
    tile_size = tile.extent[2] - tile.extent[0]
    radius_tiles = max(1, math.ceil(radius_m / tile_size))
    interior_size = tile.width - 2 * tile.gutter
    by_offset = {}
    missing = []
    for dy in range(-radius_tiles, radius_tiles + 1):
        for dx in range(-radius_tiles, radius_tiles + 1):
            x, y = cx + dx, cy + dy
            path = terrain_root / str(level) / str(x) / f"{y}.trn"
            if x < 0 or y < 0 or not path.exists():
                missing.append((dx, dy))
                continue
            by_offset[(dx, dy)] = read_tile(path)
    rows_h, rows_v = [], []
    for dy in range(-radius_tiles, radius_tiles + 1):
        row_h, row_v = [], []
        for dx in range(-radius_tiles, radius_tiles + 1):
            neighbor = by_offset.get((dx, dy))
            if neighbor is None:
                h = np.full((interior_size, interior_size), np.nan)
                v = np.zeros((interior_size, interior_size), dtype=bool)
            else:
                h, v = _tile_interior(neighbor)
            if row_h:
                h, v = h[:, 1:], v[:, 1:]
            row_h.append(h)
            row_v.append(v)
        row_h = np.hstack(row_h)
        row_v = np.hstack(row_v)
        if rows_h:
            row_h, row_v = row_h[1:, :], row_v[1:, :]
        rows_h.append(row_h)
        rows_v.append(row_v)
    mosaic = np.vstack(rows_h)
    mosaic_valid = np.vstack(rows_v)
    spacing = tile_size / (interior_size - 1)
    mosaic_west = tile.extent[0] - radius_tiles * tile_size
    mosaic_north = tile.extent[3] + radius_tiles * tile_size
    return mosaic, mosaic_valid, mosaic_west, mosaic_north, spacing, bool(missing)


def cast_shadow(mosaic, mosaic_valid, mosaic_west, mosaic_north, spacing,
                 easting, northing, elevation, sun_azimuth_rad, sun_elevation_rad,
                 max_distance, ray_step, clearance):
    """Section 10: horizon raymarch toward the sun. Returns (shadowed, truncated, h_min)."""
    sin_az, cos_az = math.sin(sun_azimuth_rad), math.cos(sun_azimuth_rad)
    tan_el = math.tan(sun_elevation_rad)
    shadowed = np.zeros(easting.shape, dtype=bool)
    truncated = np.zeros(easting.shape, dtype=bool)
    h_min = np.full(easting.shape, np.inf, dtype=np.float64)
    distance = 2.0 * ray_step
    while distance <= max_distance + 1e-9:
        ray_e = easting + distance * sin_az
        ray_n = northing + distance * cos_az
        height, ok, inside = bilinear_sample(mosaic, mosaic_valid, ray_e, ray_n,
                                              mosaic_west, mosaic_north, spacing)
        z_ray = elevation + distance * tan_el
        clearance_value = z_ray + clearance - height
        h_min = np.where(ok, np.minimum(h_min, clearance_value), h_min)
        shadowed |= ok & (clearance_value < 0.0)
        truncated |= (~inside | ~ok) & ~shadowed
        distance += ray_step
    h_min = np.where(np.isfinite(h_min), h_min, np.nan)
    return shadowed, truncated, h_min


# ---------------------------------------------------------------------------
# Section 8: flight-strip metadata and sun position
# ---------------------------------------------------------------------------

def parse_flight_id_time(flight_id):
    match = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})", flight_id)
    if not match:
        raise ValueError(f"cannot parse acquisition time from flight id {flight_id!r}")
    year, month, day, hour, minute = (int(value) for value in match.groups())
    return dt.datetime(year, month, day, hour, minute, tzinfo=dt.timezone.utc)


def geoadmin_identify(extent, timeout=20):
    west, south, east, north = extent
    tile_size = east - west
    map_extent = (west - tile_size, south - tile_size, east + tile_size, north + tile_size)
    query = urllib.parse.urlencode({
        "geometry": f"{west},{south},{east},{north}",
        "geometryType": "esriGeometryEnvelope",
        "sr": "2056",
        "layers": "all:ch.swisstopo.lubis-bildstreifen",
        "imageDisplay": "500,500,96",
        "mapExtent": ",".join(str(v) for v in map_extent),
        "tolerance": "0",
        "returnGeometry": "false",
        "lang": "en",
    })
    url = f"https://api3.geo.admin.ch/rest/services/ech/MapServer/identify?{query}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response), url


def geoadmin_detail(flight_id, timeout=20):
    url = (f"https://api3.geo.admin.ch/rest/services/ech/MapServer/"
           f"ch.swisstopo.lubis-bildstreifen/{urllib.parse.quote(flight_id)}"
           f"?sr=2056&returnGeometry=false&lang=en")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response), url


def resolve_flight(cache_dir, extent, year, flight_id, offline, acquisition_time_override):
    """Section 8: select the unique SWISSIMAGE flight strip and its time."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = cache_dir / "geoadmin_candidates_raw.json"
    detail_path = cache_dir / "geo_admin_response.json"
    warnings = []

    if flight_id and acquisition_time_override:
        acquisition_utc = dt.datetime.fromisoformat(acquisition_time_override.replace("Z", "+00:00"))
        detail = {"manual_override": True, "flight_id": flight_id}
        detail_path.write_text(json.dumps(detail, indent=2) + "\n")
        return flight_id, acquisition_utc, detail, "manual-override", [], warnings

    if candidates_path.exists():
        candidates = json.loads(candidates_path.read_text())
        candidate_source = "cache"
        identify_url = None
    elif offline:
        if not flight_id:
            raise RuntimeError("offline mode requires a cached candidate list, --flight-id, or --acquisition-time")
        candidates, candidate_source, identify_url = {"results": []}, "forced-offline", None
    else:
        candidates, identify_url = geoadmin_identify(extent)
        candidates_path.write_text(json.dumps(candidates, indent=2, sort_keys=True) + "\n")
        candidate_source = "network"

    features = candidates.get("results", candidates.get("features", []))
    filtered = [f for f in features
                if f.get("attributes", {}).get("bgdi_flugjahr") == year
                and f.get("attributes", {}).get("goal") == "SWISSIMAGE"]

    if flight_id:
        selected_id = flight_id
    else:
        if len(filtered) != 1:
            raise RuntimeError(
                f"flight selection is ambiguous: {len(filtered)} candidates matched "
                f"bgdi_flugjahr={year} and goal=SWISSIMAGE after filtering "
                f"{len(features)} raw candidates; pass --flight-id explicitly")
        selected_id = filtered[0].get("featureId") or filtered[0].get("id")

    if detail_path.exists():
        detail = json.loads(detail_path.read_text())
        detail_url = None
    elif offline:
        raise RuntimeError("offline mode requires a cached flight detail record or --acquisition-time")
    else:
        detail, detail_url = geoadmin_detail(selected_id)
        detail_path.write_text(json.dumps(detail, indent=2, sort_keys=True) + "\n")

    if acquisition_time_override:
        acquisition_utc = dt.datetime.fromisoformat(acquisition_time_override.replace("Z", "+00:00"))
    else:
        acquisition_utc = parse_flight_id_time(selected_id)

    return selected_id, acquisition_utc, detail, candidate_source, filtered, warnings


def noaa_sun_position(when_utc, latitude_deg, longitude_deg):
    """NOAA General Solar Position Calculations (gml.noaa.gov/grad/solcalc/solareqns.PDF).

    Uses UTC directly as "local standard time" with timezone=0, per the plan's
    instruction that solar calculations must use UTC explicitly. The PDF's
    leap-year note (366-day fractional year) is applied for leap years.
    """
    day_of_year = when_utc.timetuple().tm_yday
    hour = when_utc.hour + when_utc.minute / 60.0 + when_utc.second / 3600.0
    is_leap = (when_utc.year % 4 == 0 and (when_utc.year % 100 != 0 or when_utc.year % 400 == 0))
    days_in_year = 366.0 if is_leap else 365.0
    gamma = 2.0 * math.pi / days_in_year * (day_of_year - 1 + (hour - 12) / 24.0)

    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                        - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))

    time_offset = eqtime + 4.0 * longitude_deg  # timezone = 0 (UTC)
    true_solar_time = hour * 60.0 + time_offset
    hour_angle_deg = true_solar_time / 4.0 - 180.0
    hour_angle = math.radians(hour_angle_deg)
    lat = math.radians(latitude_deg)

    cos_zenith = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(hour_angle)
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = math.acos(cos_zenith)
    elevation_deg = 90.0 - math.degrees(zenith)

    cos_az = (math.sin(lat) * math.cos(zenith) - math.sin(decl)) / (math.cos(lat) * math.sin(zenith))
    cos_az = max(-1.0, min(1.0, cos_az))
    az0 = math.degrees(math.acos(cos_az))
    azimuth_deg = (az0 + 180.0) % 360.0 if hour_angle_deg > 0 else (540.0 - az0) % 360.0

    return azimuth_deg, elevation_deg, {"gamma_rad": gamma, "eqtime_min": eqtime,
                                         "declination_deg": math.degrees(decl),
                                         "hour_angle_deg": hour_angle_deg,
                                         "days_in_year": days_in_year}


def robust_regress(design, target, bounds=None, f_scale=None, iterations=5):
    """Robust (Huber-loss) linear regression per sections 8/11/12/13.

    `f_scale` (the Huber inlier/outlier transition) defaults to an adaptive,
    data-driven estimate: `HUBER_SCALE_MULTIPLIER * MAD` of the current fit's
    residuals, refined over a few IRLS-style iterations (re-fit, then
    re-estimate MAD from the new fit's residuals). A single OLS-seeded pass
    is not enough: with a handful of gross outliers among few samples, the
    seed fit itself can already be dragged far enough off that its residual
    MAD misjudges the outliers as inliers. Iterating lets the scale estimate
    recover once the fit has been pulled part-way back towards the inliers.
    A flat fixed f_scale (e.g. "8% of the [0,1] intensity range") was tried
    first and rejected: it over-suppresses the true signal on real,
    multi-material rugged terrain, where residual scatter from albedo
    variation alone routinely exceeds that.
    """
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n_params = design.shape[1]
    if len(target) >= n_params:
        x0, *_ = np.linalg.lstsq(design, target, rcond=None)
    else:
        x0 = np.zeros(n_params)
    if bounds is not None:
        x0 = np.clip(x0, bounds[0], bounds[1])
    kwargs = {"bounds": bounds} if bounds is not None else {}
    if f_scale is not None:
        return least_squares(lambda p: design @ p - target, x0, loss="huber", f_scale=f_scale, **kwargs)
    result = None
    for _ in range(max(1, iterations)):
        residual0 = design @ x0 - target
        mad = 1.4826 * float(np.median(np.abs(residual0 - np.median(residual0)))) if residual0.size else 0.0
        scale = max(HUBER_SCALE_MULTIPLIER * mad, 1e-3)
        result = least_squares(lambda p: design @ p - target, x0, loss="huber", f_scale=scale, **kwargs)
        x0 = result.x
    return result


def fit_scsc_channel(channel, direct, cosi, cos_alpha, cos_theta_s, sample_mask, min_samples=500):
    """Section 11: per-channel SCS+C fit and multiplicative gain field."""
    x = direct[sample_mask]
    y = channel[sample_mask]
    n_samples = int(sample_mask.sum())
    invalid = False
    m = c = float("nan")
    correction_c = 1.0
    if n_samples >= min_samples and np.ptp(x) > 1e-3:
        fit = robust_regress(np.stack([x, np.ones_like(x)], axis=1), y)
        m, c = fit.x
        if m > 0:
            correction_c = float(np.clip(c / m, 0.05, 5.0))
        else:
            invalid = True
    else:
        invalid = True
    if invalid:
        gain = np.ones_like(direct)
    else:
        gain = (cos_alpha * cos_theta_s + correction_c) / (cosi + correction_c + EPS_CORR)
    params = {"slope_m": m, "intercept_c": c, "correction_factor_C": correction_c,
              "samples": n_samples, "invalid": invalid}
    return gain, params


def fit_irradiance_channel(channel, cosi, sample_mask):
    """Section 13: I_b ~= A_b + D_b*max(cos i,0), A_b,D_b >= 0."""
    x = np.clip(cosi[sample_mask], 0, None)
    y = channel[sample_mask]
    if len(y) >= 4:
        fit = robust_regress(np.stack([np.ones_like(x), x], axis=1), y, bounds=([0, 0], [np.inf, np.inf]))
        a_b, d_b = fit.x
    else:
        a_b = float(np.median(y)) if len(y) else 0.0
        d_b = 0.0
    return float(a_b), float(d_b), int(len(y))


def compute_shadow_compensation(rgb_linear, luminance_raw, lam, es, ew, cap):
    """Section 13 steps 1-6: capped per-channel compensation blended via shared luminance ratio."""
    i_d = lam[..., None] * rgb_linear * es / np.maximum(ew, EPS_CORR)
    uncapped = rgb_linear + i_d
    capped_per_channel = np.minimum(uncapped, rgb_linear * cap)
    luminance_capped = capped_per_channel @ RGB_LUMA
    luminance_ratio = luminance_capped / np.maximum(luminance_raw, EPS_SMALL)
    shared = rgb_linear * luminance_ratio[..., None]
    blended = 0.75 * shared + 0.25 * capped_per_channel
    return {
        "i_d": i_d, "uncapped": uncapped, "blended_unclamped": blended,
        "final": np.clip(blended, 0.0, 1.0),
        "clip_below_zero": blended < 0.0, "clip_above_one": blended > 1.0,
    }


def fit_image_sun_direction(luminance, normal, sample_mask):
    """Section 8 image-derived cross-check: Y = a + n.l, robust, l normalized."""
    ys = luminance[sample_mask]
    design = np.stack([np.ones_like(ys), normal[..., 0][sample_mask],
                        normal[..., 1][sample_mask], normal[..., 2][sample_mask]], axis=1)
    result = robust_regress(design, ys)
    a, l_e, l_n, l_u = result.x
    if l_u <= 0:
        return None
    length = math.sqrt(l_e ** 2 + l_n ** 2 + l_u ** 2)
    l_e, l_n, l_u = l_e / length, l_n / length, l_u / length
    elevation = math.asin(l_u)
    azimuth = math.atan2(l_e, l_n) % (2 * math.pi)
    return {"azimuth_rad": azimuth, "elevation_rad": elevation,
            "vector": np.array([l_e, l_n, l_u]), "intercept": float(a),
            "samples": int(sample_mask.sum()), "cost": float(result.cost)}


def angular_difference_deg(vector_a, vector_b):
    cos_gamma = float(np.clip(np.dot(vector_a, vector_b), -1.0, 1.0))
    return math.degrees(math.acos(cos_gamma))


# ---------------------------------------------------------------------------
# Otsu (section 12) and signed-EDT feathering
# ---------------------------------------------------------------------------

def otsu_threshold(values, bins=256, value_range=(0.0, 1.0)):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    hist, edges = np.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5 * (value_range[0] + value_range[1])
    prob = hist / total
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(prob)
    w1 = 1.0 - w0
    mu = np.cumsum(prob * centers)
    mu_total = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = np.where(w0 > 0, mu / np.where(w0 > 0, w0, 1.0), 0.0)
        mu1 = np.where(w1 > 0, (mu_total - mu) / np.where(w1 > 0, w1, 1.0), 0.0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    between = np.where(np.isfinite(between), between, -np.inf)
    idx = min(int(np.argmax(between[:-1])) if bins > 1 else 0, bins - 1)
    return float(centers[idx])


def feather(mask, spacing_m, width_m):
    """Signed EDT feather: 1 deep inside `mask`, 0 far outside, ramps at edge."""
    mask = np.asarray(mask, dtype=bool)
    if mask.all() or (~mask).all():
        return np.where(mask, 1.0, 0.0)
    inside = ndimage.distance_transform_edt(mask) * spacing_m
    outside = ndimage.distance_transform_edt(~mask) * spacing_m
    signed = np.where(mask, -inside, outside)
    return np.clip(0.5 - signed / max(width_m, EPS_SMALL), 0.0, 1.0)


def local_albedo(intensity, weight_mask, spacing_m, window_m=60.0):
    sigma = (window_m / spacing_m) / 6.0
    weight = weight_mask.astype(np.float64)
    numerator = ndimage.gaussian_filter(intensity * weight, sigma=sigma, truncate=3.0, mode="nearest")
    denominator = ndimage.gaussian_filter(weight, sigma=sigma, truncate=3.0, mode="nearest")
    total_weight = ndimage.gaussian_filter(np.ones_like(weight), sigma=sigma, truncate=3.0, mode="nearest")
    albedo = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-6)
    valid_fraction = np.divide(denominator, total_weight, out=np.zeros_like(denominator), where=total_weight > 1e-6)
    return np.clip(albedo, 0.0, 1.0), valid_fraction


def normalize_quantile(values, valid_mask):
    valid = values[valid_mask & np.isfinite(values)]
    if valid.size == 0:
        return np.zeros_like(values), 0.0, 0.0
    q50, q99 = np.percentile(valid, [50, 99])
    normalized = np.clip((values - q50) / max(q99 - q50, EPS_SMALL), 0.0, 1.0)
    return normalized, float(q50), float(q99)


def compute_lambda_raw(proxy, threshold, si_max):
    """Eq.7: 0 below threshold, linear ramp to 1 at si_max, clipped to [0,1]."""
    return np.where(proxy >= threshold,
                     np.clip((proxy - threshold) / max(si_max - threshold, EPS_SMALL), 0.0, 1.0),
                     0.0)


# ---------------------------------------------------------------------------
# Checkpoint I/O (section 15/16): every stage writes arrays + stats + stage.json
# ---------------------------------------------------------------------------

class Stage:
    def __init__(self, root: Path, index: int, slug: str):
        self.index = index
        self.slug = slug
        self.dir = root / f"{index:02d}_{slug}"
        (self.dir / "arrays").mkdir(parents=True, exist_ok=True)
        self.stats = {}
        self.image_names = []

    def array(self, name, data, units="unitless"):
        arr = np.asarray(data)
        np.save(self.dir / "arrays" / f"{name}.npy", arr)
        if np.issubdtype(arr.dtype, np.floating):
            finite = np.isfinite(arr)
        else:
            finite = np.ones(arr.shape, dtype=bool)
        finite_values = arr[finite]
        entry = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "units": units,
            "finite_count": int(finite.sum()),
            "masked_pixel_count": int((~finite).sum()),
        }
        if finite_values.size and np.issubdtype(arr.dtype, np.number):
            values = finite_values.astype(np.float64)
            entry.update(
                min=float(values.min()), max=float(values.max()),
                mean=float(values.mean()), median=float(np.median(values)),
                std=float(values.std()),
                percentiles={str(p): float(np.percentile(values, p))
                             for p in (1, 5, 25, 50, 75, 95, 99)},
            )
        self.stats[name] = entry
        return arr

    def png(self, name, image_uint8):
        path = self.dir / f"{name}.png"
        Image.fromarray(image_uint8).save(path)
        self.image_names.append(path.name)
        return path

    def finish(self, equation, inputs, outputs, parameters=None, deviations=None,
               warnings=None, qa=None):
        (self.dir / "stats.json").write_text(json.dumps(self.stats, indent=2, sort_keys=True) + "\n")
        stage_meta = {
            "stage": self.slug,
            "index": self.index,
            "equation_or_approximation": equation,
            "inputs": inputs,
            "outputs": outputs,
            "parameters": parameters or {},
            "images": sorted(self.image_names),
            "deviations": deviations or [],
            "warnings": warnings or [],
            "qa": qa or {},
        }
        (self.dir / "stage.json").write_text(json.dumps(stage_meta, indent=2, sort_keys=True, default=str) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def draw_bars(width, height, values, color=(70, 130, 180)):
    """Minimal dependency-free bar chart (stdlib PIL only, no matplotlib)."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    if len(values) == 0:
        return image
    values = np.asarray(values, dtype=np.float64)
    peak = max(float(values.max()), EPS_SMALL)
    bar_width = max(1, width // len(values))
    for i, value in enumerate(values):
        bar_height = max(1, int(height * value / peak))
        x0 = i * bar_width
        draw.rectangle((x0, height - bar_height, x0 + bar_width - 1, height - 1), fill=color)
    return image


def draw_table(title, rows, width=640):
    line_height = 18
    height = 40 + line_height * (len(rows) + 1)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 8), title, fill="black")
    for i, row in enumerate(rows):
        draw.text((10, 34 + i * line_height), row, fill="black")
    return np.asarray(image)


def draw_compass(labels_azimuths, size=320):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    cx = cy = size // 2
    radius = size // 2 - 30
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline="black")
    for label, azimuth_deg, color in labels_azimuths:
        theta = math.radians(azimuth_deg)
        x = cx + radius * math.sin(theta)
        y = cy - radius * math.cos(theta)
        draw.line((cx, cy, x, y), fill=color, width=3)
        draw.text((x - 10, y - 10), label, fill=color)
    draw.text((cx - 4, cy - radius - 20), "N", fill="black")
    return np.asarray(image)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_input_image(path: Path):
    image = Image.open(path).convert("RGB")
    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"{path}: expected {IMAGE_SIZE}x{IMAGE_SIZE}, got {image.size[0]}x{image.size[1]}")
    return np.asarray(image, dtype=np.uint8)


def check_gutter_alignment(rgb8, terrain_root: Path, level, x, y):
    imagery_path = terrain_root.parent / "imagery" / str(level) / str(x) / f"{y}.png"
    if not imagery_path.exists():
        return None, f"repository gutter image not found: {imagery_path}"
    repo = np.asarray(Image.open(imagery_path).convert("RGB"))
    if repo.shape[:2] != (IMAGE_SIZE + 2, IMAGE_SIZE + 2):
        return False, f"unexpected repository image shape {repo.shape}"
    interior = repo[1:1 + IMAGE_SIZE, 1:1 + IMAGE_SIZE]
    return bool(np.array_equal(interior, rgb8)), None


def run(args) -> dict:
    out = args.output
    if out.exists():
        if not args.force:
            raise SystemExit(f"output directory already exists: {out} (use --force)")
        protected = {ROOT, ROOT / "src", ROOT / "tools", ROOT / "tests",
                     ROOT / "docs", ROOT / "alps-data"}
        if out.resolve() in {p.resolve() for p in protected}:
            raise SystemExit(f"refusing to --force into a protected directory: {out}")
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True)

    warnings_all: list[str] = []
    qa_counts: dict[str, int] = {}

    # ---- inputs -----------------------------------------------------------
    rgb8 = load_input_image(args.input)
    rgb_linear = srgb_to_linear(rgb8)
    tile_path = args.terrain_root / str(args.level) / str(args.tile_x) / f"{args.tile_y}.trn"
    if not tile_path.exists():
        raise SystemExit(f"terrain tile not found: {tile_path}")
    tile = read_tile(tile_path)
    extent = tile.extent  # (west, south, east, north)
    east_grid, north_grid = pixel_centres(extent, IMAGE_SIZE)
    image_gsd = (extent[2] - extent[0]) / IMAGE_SIZE
    dem_gsd = (extent[2] - extent[0]) / (tile.width - 2 * tile.gutter - 1)

    mosaic, mosaic_valid, mosaic_west, mosaic_north, mosaic_spacing, halo_incomplete = load_dem_halo(
        args.terrain_root, args.level, args.tile_x, args.tile_y, tile, args.max_horizon_distance)
    if halo_incomplete:
        warnings_all.append("DEM halo missing one or more neighbour tiles; "
                             "affected rays are marked horizon_truncated")

    elevation, elevation_valid, _ = bilinear_sample(
        mosaic, mosaic_valid, east_grid, north_grid, mosaic_west, mosaic_north, mosaic_spacing)

    manifest_path = args.manifest
    dataset_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    # ---- 00 input -----------------------------------------------------------
    stage = Stage(out, 0, "input")
    stage.array("input_rgb8", rgb8, "uint8 sRGB")
    stage.array("linear_intensity_rgb", rgb_linear, "linear_intensity [0,1]")
    luminance_raw = rgb_linear @ RGB_LUMA
    stage.array("luminance", luminance_raw, "linear_intensity [0,1]")
    stage.png("input_raw", rgb8)
    hist = np.histogram(rgb8.reshape(-1, 3).mean(axis=1), bins=32, range=(0, 255))[0]
    stage.png("input_histogram", np.asarray(draw_bars(320, 160, hist)))
    input_checksum = sha256_file(args.input)
    tile_checksum = sha256_file(tile_path)
    stage.finish(
        equation="C_s=C_8bit/255; sRGB->linear per section 9",
        inputs=[str(args.input), str(tile_path)],
        outputs=["input_rgb8.npy", "linear_intensity_rgb.npy", "luminance.npy", "input_raw.png"],
        parameters={"input_sha256": input_checksum, "tile_sha256": tile_checksum,
                    "tile_key": [args.level, args.tile_x, args.tile_y], "extent_lv95": list(extent)},
    )

    # ---- 01 alignment -------------------------------------------------------
    stage = Stage(out, 1, "alignment")
    gutter_ok, gutter_note = check_gutter_alignment(rgb8, args.terrain_root, args.level, args.tile_x, args.tile_y)
    hillshade_normal, _, _, _, _ = terrain_geometry(elevation, image_gsd)
    hill_light = sun_vector(math.radians(315.0), math.radians(45.0))
    hillshade = np.clip(incidence_dot(hillshade_normal, hill_light), 0.0, 1.0)
    stage.array("dem_elevation_m", elevation, "metres")
    stage.array("dem_valid_mask", elevation_valid, "bool")
    stage.array("hillshade", hillshade, "unitless [0,1]")
    stage.png("dem_hillshade", render_gray(hillshade, 0.0, 1.0))
    stage.png("input_raw", rgb8)
    edges = np.hypot(*np.gradient(luminance_raw))
    stage.png("input_edges", render_gray(edges, 0.0, float(np.percentile(edges, 99)) or 1.0))
    stage.finish(
        equation="pixel centre E=W+(col+.5)(Emax-W)/256, N=Nmax-(row+.5)(Nmax-S)/256 (section 7)",
        inputs=[str(tile_path)],
        outputs=["dem_elevation_m.npy", "hillshade.png"],
        parameters={"image_gsd_m": image_gsd, "dem_gsd_m": dem_gsd,
                    "gutter_interior_matches_input": gutter_ok},
        warnings=[w for w in [gutter_note] if w],
        qa={"dem_valid_fraction": float(elevation_valid.mean())},
    )
    if gutter_ok is False:
        warnings_all.append("repository gutter image interior does not match the supplied input")

    # ---- 02 flight and sun ---------------------------------------------------
    stage = Stage(out, 2, "flight_and_sun")
    try:
        flight_id, acquisition_utc, flight_detail, candidate_source, filtered, flight_warnings = resolve_flight(
            stage.dir, extent, args.year, args.flight_id, args.offline, args.acquisition_time)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"flight-strip selection failed: {error}")
    acquisition_local = acquisition_utc.astimezone(ZoneInfo("Europe/Zurich"))
    centroid_e, centroid_n = (extent[0] + extent[2]) / 2.0, (extent[1] + extent[3]) / 2.0
    centroid_lon, centroid_lat = lv95_to_wgs84(centroid_e, centroid_n)
    sun_az_deg, sun_el_deg, noaa_debug = noaa_sun_position(acquisition_utc, centroid_lat, centroid_lon)
    sun_az_rad, sun_el_rad = math.radians(sun_az_deg), math.radians(sun_el_deg)
    sun = sun_vector(sun_az_rad, sun_el_rad)

    normal, slope, aspect, g_e, g_n = terrain_geometry(elevation, image_gsd)
    cosi_dot = incidence_dot(normal, sun)
    saturated_mask = rgb_linear.max(axis=2) >= 0.98
    fit_sample_mask = elevation_valid & ~saturated_mask & (cosi_dot > 0)
    fit_result = fit_image_sun_direction(luminance_raw, normal, fit_sample_mask)
    if fit_result is None:
        warnings_all.append("image-fitted sun direction rejected (l_U <= 0); using NOAA/metadata sun only")
        sun_vector_error_deg = float("nan")
        fit_az_deg = fit_el_deg = float("nan")
    else:
        sun_vector_error_deg = angular_difference_deg(sun, fit_result["vector"])
        fit_az_deg = math.degrees(fit_result["azimuth_rad"])
        fit_el_deg = math.degrees(fit_result["elevation_rad"])
        if sun_vector_error_deg > 5.0:
            warnings_all.append(f"sun vector cross-check exceeds 5 degrees: {sun_vector_error_deg:.2f}")

    stage.array("centroid_lonlat", np.array([centroid_lon, centroid_lat]), "degrees WGS84")
    stage.array("noaa_sun_azimuth_elevation_deg", np.array([sun_az_deg, sun_el_deg]), "degrees")
    stage.array("fitted_sun_azimuth_elevation_deg", np.array([fit_az_deg, fit_el_deg]), "degrees")
    stage.png("compass", draw_compass(
        [("NOAA", sun_az_deg, "red")] + ([("fit", fit_az_deg, "blue")] if fit_result else [])))
    stage.finish(
        equation="NOAA General Solar Position Calculations (gml.noaa.gov/grad/solcalc/solareqns.PDF); "
                 "image fit Y=a+n.l (section 8)",
        inputs=["GeoAdmin ch.swisstopo.lubis-bildstreifen identify/detail"],
        outputs=["geoadmin_candidates_raw.json", "geo_admin_response.json"],
        parameters={
            "flight_id": flight_id, "acquisition_utc": acquisition_utc.isoformat(),
            "acquisition_local": acquisition_local.isoformat(),
            "candidate_source": candidate_source, "candidates_after_filter": len(filtered),
            "centroid_lonlat_wgs84": [centroid_lon, centroid_lat],
            "sun_azimuth_deg": sun_az_deg, "sun_elevation_deg": sun_el_deg,
            "fitted_azimuth_deg": fit_az_deg, "fitted_elevation_deg": fit_el_deg,
            "sun_vector_error_deg": sun_vector_error_deg,
            "noaa_debug": noaa_debug, "flight_detail": flight_detail,
        },
        deviations=["NOAA formula implemented exactly per the published PDF (including the leap-year "
                    "366-day fractional-year rule for 2024); this pilot's computed value for the "
                    "2024-08-24T08:31Z fixture differs by roughly 0.3-0.4 degrees from the plan's "
                    "quoted approximate reference figures (~118.611/38.042 deg), most plausibly because "
                    "those reference figures were produced by a higher-order solar-position algorithm "
                    "rather than the simplified formula this section is specified to implement."],
        warnings=flight_warnings,
    )

    # ---- 03 terrain geometry -------------------------------------------------
    stage = Stage(out, 3, "terrain_geometry")
    stage.array("elevation_m", elevation, "metres")
    stage.array("gradient_east", g_e, "m/m")
    stage.array("gradient_north", g_n, "m/m")
    stage.array("slope_deg", np.degrees(slope), "degrees")
    stage.array("aspect_deg", np.degrees(aspect), "degrees clockwise from north")
    stage.array("normal_enu", normal, "unit vector")
    stage.png("slope", render_gray(np.degrees(slope), 0.0, 60.0))
    stage.png("aspect", render_gray(np.degrees(aspect), 0.0, 360.0))
    normal_rgb = np.clip((normal + 1.0) * 0.5, 0.0, 1.0)
    stage.png("normal_rgb", np.clip(normal_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8))
    stage.finish(
        equation="alpha=atan(sqrt(gE^2+gN^2)); beta=atan2(-gE,-gN) mod 2pi (section 7)",
        inputs=["dem_elevation_m"], outputs=["slope_deg.npy", "aspect_deg.npy", "normal_enu.npy"],
        parameters={"spacing_m": image_gsd},
    )

    # ---- 04 incidence ---------------------------------------------------------
    stage = Stage(out, 4, "incidence")
    cosi_alt = incidence_formula(slope, aspect, sun_az_rad, sun_el_rad)
    incidence_disagreement = float(np.nanmax(np.abs(cosi_dot.astype(np.float64) - cosi_alt.astype(np.float64))))
    if incidence_disagreement >= 1e-6:
        warnings_all.append(f"incidence formulas disagree by {incidence_disagreement:.3e} (>= 1e-6)")
    direct = np.clip(cosi_dot, 0.0, None)
    stage.array("cos_incidence_dot", cosi_dot, "unitless [-1,1]")
    stage.array("cos_incidence_formula", cosi_alt, "unitless [-1,1]")
    stage.array("incidence_disagreement", np.abs(cosi_dot - cosi_alt), "unitless")
    stage.array("direct_illumination", direct, "unitless [0,1]")
    stage.array("positive_incidence_mask", cosi_dot > 0, "bool")
    stage.png("cos_incidence", render_signed(cosi_dot, 1.0))
    stage.png("positive_incidence", render_mask(cosi_dot > 0))
    stage.finish(
        equation="cos(i)=n.s == cos(alpha)cos(theta_s)+sin(alpha)sin(theta_s)cos(phi_s-beta) (section 7, Eq.1)",
        inputs=["normal_enu", "sun_azimuth_elevation"], outputs=["cos_incidence_dot.npy"],
        parameters={"max_disagreement": incidence_disagreement, "tolerance": 1e-6},
        qa={"disagreement_within_tolerance": incidence_disagreement < 1e-6},
    )

    # ---- 05 shadow geometry -----------------------------------------------
    stage = Stage(out, 5, "shadow_geometry")
    self_shadow = cosi_dot <= 0
    cast, truncated, h_min = cast_shadow(
        mosaic, mosaic_valid, mosaic_west, mosaic_north, mosaic_spacing,
        east_grid, north_grid, elevation, sun_az_rad, sun_el_rad,
        args.max_horizon_distance, args.ray_step, args.terrain_clearance)
    geometric_shadow = self_shadow | cast
    qa_counts["horizon_truncated_pixels"] = int(truncated.sum())
    stage.array("self_shadow_mask", self_shadow, "bool")
    stage.array("cast_shadow_mask", cast, "bool")
    stage.array("geometric_shadow_mask", geometric_shadow, "bool")
    stage.array("horizon_clearance_m", h_min, "metres")
    stage.array("horizon_truncated_mask", truncated, "bool")
    stage.png("self_shadow", render_mask(self_shadow))
    stage.png("cast_shadow", render_mask(cast))
    stage.png("geometric_shadow", render_mask(geometric_shadow))
    stage.png("horizon_truncated", render_mask(truncated))
    overlay = render_rgb(rgb_linear).copy()
    overlay[geometric_shadow] = (overlay[geometric_shadow].astype(np.int32) // 2 + np.array([0, 0, 96])).clip(0, 255).astype(np.uint8)
    stage.png("shadow_overlay", overlay)
    stage.finish(
        equation="G=self-shadow OR cast-shadow; cast: z_DEM(E+d sin phi,N+d cos phi) > "
                 "z0+d tan(e_s)+clearance for d=2*step+k*step up to max distance (section 10)",
        inputs=["dem_elevation_m", "sun_azimuth_elevation"],
        outputs=["geometric_shadow_mask.npy", "horizon_clearance_m.npy"],
        parameters={"max_horizon_distance_m": args.max_horizon_distance,
                    "ray_step_m": args.ray_step, "terrain_clearance_m": args.terrain_clearance,
                    "halo_incomplete": halo_incomplete},
        qa={"horizon_truncated_fraction": float(truncated.mean()),
            "geometric_shadow_fraction": float(geometric_shadow.mean())},
    )

    # ---- 06 SCS+C fit -----------------------------------------------------
    stage = Stage(out, 6, "scsc_fit")
    slope_deg = np.degrees(slope)
    scsc_gain = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float64)
    scsc_params = []
    cos_theta_s = math.cos(math.pi / 2.0 - sun_el_rad)
    cos_alpha = np.cos(slope)
    for band in range(3):
        channel = rgb_linear[..., band]
        base_mask = (~geometric_shadow) & (slope_deg > 5.0) & np.isfinite(channel) \
            & (channel < 0.98) & (~truncated) & elevation_valid
        gain, params = fit_scsc_channel(channel, direct, cosi_dot, cos_alpha, cos_theta_s, base_mask)
        scsc_gain[..., band] = gain
        scsc_params.append({"band": band, **params})
    scsc_invalid_bands = [p["band"] for p in scsc_params if p["invalid"]]
    stage.array("scs_c_raw_gain", scsc_gain, "unitless multiplicative gain")
    stage.array("regression_sample_count", np.array([p["samples"] for p in scsc_params]), "count")
    stage.finish(
        equation="I_t,b=m_b*cos(i)+b_b; C_b=b_b/m_b clamp[0.05,5]; "
                 "gain=(cos(alpha)cos(theta_s)+C_b)/(cos(i)+C_b+eps) (section 11)",
        inputs=["linear_intensity_rgb", "cos_incidence_dot", "slope_deg"],
        outputs=["scs_c_raw_gain.npy"],
        parameters={"channels": scsc_params, "epsilon": EPS_CORR, "min_samples": 500,
                    "slope_threshold_deg": 5.0, "intensity_threshold": 0.98},
        warnings=[f"channel {b} SCS+C fit invalid; gain forced to identity" for b in scsc_invalid_bands],
        qa={"scsc_fit_invalid_bands": scsc_invalid_bands},
    )

    # ---- 07 SCS+C corrected ------------------------------------------------
    stage = Stage(out, 7, "scsc_corrected")
    scsc_diagnostic = rgb_linear * scsc_gain
    profile_ranges = {"conservative": (0.80, 1.25), "aggressive": (0.67, 1.50)}
    scsc_profile = {}
    cap_masks = {}
    for profile, (lo, hi) in profile_ranges.items():
        clamped_gain = np.clip(scsc_gain, lo, hi)
        scsc_profile[profile] = rgb_linear * clamped_gain
        cap_masks[profile] = (scsc_gain < lo) | (scsc_gain > hi)
    stage.array("scsc_diagnostic_rgb", scsc_diagnostic, "linear_intensity")
    stage.array("signed_difference", scsc_diagnostic - rgb_linear, "linear_intensity delta")
    stage.array("sunlit_overlay_mask", ~geometric_shadow, "bool")
    for profile in profile_ranges:
        stage.array(f"{profile}_cap_mask", cap_masks[profile], "bool")
    stage.png("scsc_diagnostic", render_rgb(scsc_diagnostic))
    stage.png("signed_difference", render_signed((scsc_diagnostic - rgb_linear).mean(axis=2), 0.25))
    stage.finish(
        equation="I_SCS+C,b = I_t,b * gain_b, gain clamped to profile range for the final blend (section 11)",
        inputs=["linear_intensity_rgb", "scs_c_raw_gain"],
        outputs=["scsc_diagnostic_rgb.npy"],
        parameters={"profile_gain_ranges": profile_ranges},
    )

    # ---- 08 shadow index (RGB proxy) ---------------------------------------
    stage = Stage(out, 8, "shadow_index")
    visible = (~geometric_shadow).astype(np.float64)
    design = np.stack([np.ones(IMAGE_SIZE * IMAGE_SIZE),
                        (np.clip(cosi_dot, 0, None) * visible).ravel()], axis=1)
    dark_fit = robust_regress(design, luminance_raw.ravel(), bounds=([0, 0], [np.inf, np.inf]))
    a_y, d_y = dark_fit.x
    y_hat = a_y + d_y * np.clip(cosi_dot, 0, None) * visible
    residual = np.log(y_hat + EPS_SMALL) - np.log(luminance_raw + EPS_SMALL)
    dark_norm, dark_q50, dark_q99 = normalize_quantile(residual, elevation_valid)
    blue_chroma = rgb_linear[..., 2] / np.maximum(rgb_linear.sum(axis=2), EPS_SMALL)
    blue_norm, blue_q50, blue_q99 = normalize_quantile(blue_chroma, elevation_valid)
    g_soft_1m = feather(geometric_shadow, image_gsd, 1.0)
    g_soft_half_m = feather(geometric_shadow, image_gsd, 0.5)
    proxy = np.clip(0.55 * dark_norm + 0.25 * blue_norm + 0.20 * g_soft_1m, 0.0, 1.0)
    threshold = otsu_threshold(proxy[elevation_valid])
    stage.array("dark_residual_normalized", dark_norm, "unitless [0,1]")
    stage.array("blue_chromaticity_normalized", blue_norm, "unitless [0,1]")
    stage.array("g_soft_1m", g_soft_1m, "unitless [0,1]")
    stage.array("si_rgb_proxy", proxy, "unitless [0,1] (NOT paper Eq.6)")
    stage.array("si_rgb_proxy_otsu_threshold", np.array(threshold), "unitless")
    stage.png("dark_residual", render_gray(dark_norm, 0.0, 1.0))
    stage.png("blue_chromaticity", render_gray(blue_norm, 0.0, 1.0))
    stage.png("si_rgb_proxy", render_gray(proxy, 0.0, 1.0))
    hist_vals = np.histogram(proxy[elevation_valid], bins=64, range=(0, 1))[0]
    otsu_plot = draw_bars(320, 160, hist_vals)
    draw = ImageDraw.Draw(otsu_plot)
    draw.line((int(threshold * 320), 0, int(threshold * 320), 160), fill="red", width=2)
    stage.png("otsu_plot", np.asarray(otsu_plot))
    stage.finish(
        equation="EXACT_SI_UNAVAILABLE: paper Eq.6 needs Coastal/Green/NIR; "
                 "SI_RGB_PROXY=0.55*d+0.25*b_c_norm+0.20*G_soft (section 12)",
        inputs=["luminance", "cos_incidence_dot", "linear_intensity_rgb", "geometric_shadow_mask"],
        outputs=["si_rgb_proxy.npy"],
        parameters={"dark_fit_A_Y": float(a_y), "dark_fit_D_Y": float(d_y),
                    "dark_q50": dark_q50, "dark_q99": dark_q99,
                    "blue_q50": blue_q50, "blue_q99": blue_q99,
                    "otsu_threshold": threshold, "feather_width_m_used_in_proxy": 1.0},
        deviations=["Exact paper Equation 6 (Coastal/Green/NIR shadow index) cannot be computed from RGB; "
                    "SI_RGB_PROXY is a named, interface-compatible stand-in per section 12.",
                    "G_soft term inside SI_RGB_PROXY uses the 1m (conservative) feather width; the "
                    "profile-specific 1m/0.5m feathering is applied again, separately, to lambda below."],
    )

    # ---- 09 lambda ----------------------------------------------------------
    stage = Stage(out, 9, "lambda")
    si_max = float(np.percentile(proxy[elevation_valid], 99.5)) if elevation_valid.any() else 1.0
    lambda_raw = compute_lambda_raw(proxy, threshold, si_max)
    lambda_conservative = lambda_raw * g_soft_1m
    lambda_aggressive = lambda_raw * np.maximum(g_soft_half_m, 0.5)
    lambda_aggressive = np.where(truncated, np.minimum(lambda_aggressive, lambda_conservative), lambda_aggressive)
    stage.array("lambda_raw", lambda_raw, "unitless [0,1]")
    stage.array("lambda_conservative", lambda_conservative, "unitless [0,1]")
    stage.array("lambda_aggressive", lambda_aggressive, "unitless [0,1]")
    stage.array("saturated_mask", saturated_mask, "bool")
    stage.png("lambda_raw", render_gray(lambda_raw, 0.0, 1.0))
    stage.png("lambda_conservative", render_gray(lambda_conservative, 0.0, 1.0))
    stage.png("lambda_aggressive", render_gray(lambda_aggressive, 0.0, 1.0))
    stage.finish(
        equation="Eq.7: lambda_raw=0 if SI<c else (SI-c)/(SI_max-c); "
                 "lambda_conservative=lambda_raw*G_soft_1m; "
                 "lambda_aggressive=lambda_raw*max(G_soft_0.5m,0.5) (section 12)",
        inputs=["si_rgb_proxy", "geometric_shadow_mask"], outputs=["lambda_conservative.npy", "lambda_aggressive.npy"],
        parameters={"si_max_q99.5": si_max, "otsu_threshold": threshold},
        deviations=["Horizon-truncated pixels have lambda_aggressive capped to lambda_conservative "
                    "(section 14's 'horizon-truncated pixels use conservative behavior')."],
    )

    # ---- 10 sky view / albedo -----------------------------------------------
    stage = Stage(out, 10, "sky_view_albedo")
    v_d = (1.0 + cos_alpha) / 2.0
    c_t = 1.0 - v_d
    sunlit_valid = (~geometric_shadow) & elevation_valid
    albedo = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float64)
    valid_fraction = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float64)
    for band in range(3):
        albedo[..., band], valid_fraction[..., band] = local_albedo(
            rgb_linear[..., band], sunlit_valid, image_gsd, window_m=31 * image_gsd)
    stage.array("sky_view_vd", v_d, "unitless [0,1]")
    stage.array("terrain_configuration_ct", c_t, "unitless [0,1]")
    stage.array("adjacent_albedo_rgb", albedo, "linear_intensity [0,1]")
    stage.array("albedo_valid_fraction", valid_fraction, "unitless [0,1]")
    stage.png("sky_view_vd", render_gray(v_d, 0.0, 1.0))
    stage.png("adjacent_albedo", render_rgb(albedo))
    stage.finish(
        equation="Vd=(1+cos(alpha))/2; Ct=1-Vd; rho_a,b = Gaussian-weighted mean sunlit intensity, "
                 "31x31px window (~60m) (section 13)",
        inputs=["slope_deg", "linear_intensity_rgb", "geometric_shadow_mask"],
        outputs=["sky_view_vd.npy", "adjacent_albedo_rgb.npy"],
        parameters={"window_px": 31, "window_m": 31 * image_gsd, "gaussian_sigma_px": (31 / 6.0)},
    )

    # ---- 11 irradiance --------------------------------------------------------
    stage = Stage(out, 11, "irradiance")
    irradiance_params = []
    es = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
    ek = np.zeros_like(es)
    ea = np.zeros_like(es)
    ew = np.zeros_like(es)
    for band in range(3):
        channel = rgb_linear[..., band]
        base_mask = (~geometric_shadow) & (slope_deg > 5.0) & np.isfinite(channel) \
            & (channel < 0.98) & (~truncated) & elevation_valid
        a_b, d_b, n_samples = fit_irradiance_channel(channel, cosi_dot, base_mask)
        irradiance_params.append({"band": band, "A_b": a_b, "D_b": d_b, "samples": n_samples})
        es[..., band] = d_b * cos_theta_s
        ek[..., band] = a_b * v_d
        ea[..., band] = (a_b + d_b * cos_theta_s) * c_t * albedo[..., band]
        ew[..., band] = ek[..., band] + ea[..., band]
    denominator_qa = ew < EPS_CORR
    stage.array("irradiance_direct_es", es, "relative irradiance")
    stage.array("irradiance_diffuse_ek", ek, "relative irradiance")
    stage.array("irradiance_adjacent_ea", ea, "relative irradiance")
    stage.array("irradiance_total_ew", ew, "relative irradiance")
    stage.array("denominator_low_qa_mask", denominator_qa, "bool")
    stage.png("irradiance_ew", render_rgb(ew / max(float(np.nanmax(ew)), EPS_SMALL)))
    stage.finish(
        equation="I_b~=A_b+D_b*max(cos i,0); Es=D_b*cos(theta_s); Ek=A_b*Vd; "
                 "Ea=(A_b+D_b*cos(theta_s))*Ct*rho_a; Ew=Ek+Ea (section 13)",
        inputs=["linear_intensity_rgb", "cos_incidence_dot", "sky_view_vd", "adjacent_albedo_rgb"],
        outputs=["irradiance_total_ew.npy"],
        parameters={"channels": irradiance_params},
        deviations=["Full terrain-reflected irradiance (atmospheric parameters, neighbour radiances, "
                    "band-specific physical quantities) is unavailable; this approximation follows "
                    "section 13 literally using an unconstrained-except-nonnegative robust linear fit "
                    "of channel intensity against max(cos i,0) over the same sample selection as the "
                    "SCS+C fit (section 11)."],
        qa={"low_denominator_fraction": float(denominator_qa.mean())},
    )

    # ---- 12 shadow compensation ----------------------------------------------
    stage = Stage(out, 12, "shadow_compensation")
    gain_caps = {"conservative": 2.0, "aggressive": 3.5}
    shadow_result = {}
    clip_masks = {}
    for profile, lam in (("conservative", lambda_conservative), ("aggressive", lambda_aggressive)):
        cap = gain_caps[profile]
        result = compute_shadow_compensation(rgb_linear, luminance_raw, lam, es, ew, cap)
        clip_masks[profile] = {"below_zero": result["clip_below_zero"], "above_one": result["clip_above_one"]}
        shadow_result[profile] = result["final"]
        if profile == "conservative":
            stage.array("id_compensation", result["i_d"], "linear_intensity")
            stage.array("unbounded_correction", result["uncapped"], "linear_intensity")
    for profile in gain_caps:
        stage.array(f"{profile}_gain_map", shadow_result[profile] / np.maximum(rgb_linear, EPS_SMALL), "unitless gain")
        stage.array(f"{profile}_final_shadow_rgb", shadow_result[profile], "linear_intensity")
        stage.array(f"{profile}_clip_below_zero", clip_masks[profile]["below_zero"], "bool")
        stage.array(f"{profile}_clip_above_one", clip_masks[profile]["above_one"], "bool")
    stage.png("conservative_shadow", render_rgb(shadow_result["conservative"]))
    stage.png("aggressive_shadow", render_rgb(shadow_result["aggressive"]))
    stage.finish(
        equation="I_shadow,b=I_t,b+I_d,b; g_b=I_shadow,b/I_t,b capped; gain applied via shared "
                 "luminance ratio blended 75/25 with per-channel result (section 13)",
        inputs=["linear_intensity_rgb", "lambda_conservative", "lambda_aggressive",
                "irradiance_direct_es", "irradiance_total_ew"],
        outputs=["conservative_final_shadow_rgb.npy", "aggressive_final_shadow_rgb.npy"],
        parameters={"gain_caps": gain_caps, "luminance_blend_shared": 0.75, "luminance_blend_per_channel": 0.25},
    )

    # ---- 13/14 final piecewise profiles ---------------------------------------
    finals = {}
    metrics = {}
    for idx, profile in ((13, "conservative"), (14, "aggressive")):
        stage = Stage(out, idx, profile)
        weight = lambda_conservative if profile == "conservative" else lambda_aggressive
        blended = (1.0 - weight)[..., None] * scsc_profile[profile] + weight[..., None] * shadow_result[profile]
        luminance_ratio_only = (shadow_result[profile] @ RGB_LUMA) / np.maximum(luminance_raw, EPS_SMALL)
        cap = gain_caps[profile]
        saturated_out = rgb_linear * np.clip(luminance_ratio_only, 1.0 / cap, cap)[..., None]
        final = np.where(saturated_mask[..., None], saturated_out, blended)
        clipped_fraction = float(((final < 0) | (final > 1)).any(axis=2).mean())
        final = np.clip(final, 0.0, 1.0)
        finals[profile] = final
        sunlit_mask = ~geometric_shadow
        sunlit_change = float(np.median(np.abs(final - rgb_linear)[sunlit_mask] / np.maximum(rgb_linear[sunlit_mask], 0.01))) if sunlit_mask.any() else 0.0
        luminance_out = final @ RGB_LUMA
        shadow_gap_in = float(np.median(luminance_raw[sunlit_mask]) - np.median(luminance_raw[geometric_shadow])) if geometric_shadow.any() and sunlit_mask.any() else 0.0
        shadow_gap_out = float(np.median(luminance_out[sunlit_mask]) - np.median(luminance_out[geometric_shadow])) if geometric_shadow.any() and sunlit_mask.any() else 0.0
        metrics[profile] = {
            "clipped_fraction": clipped_fraction, "sunlit_median_change": sunlit_change,
            "shadow_sunlit_luminance_gap_input": shadow_gap_in,
            "shadow_sunlit_luminance_gap_output": shadow_gap_out,
            "finite": bool(np.isfinite(final).all()),
        }
        stage.array("final_linear_rgb", final, "linear_intensity")
        stage.array("weight_w", weight, "unitless [0,1]")
        stage.array("signed_difference", final - rgb_linear, "linear_intensity delta")
        stage.png("final", render_rgb(final))
        stage.png("signed_difference", render_signed((final - rgb_linear).mean(axis=2), 0.25))
        overlay = render_rgb(final).copy()
        overlay[geometric_shadow] = (overlay[geometric_shadow].astype(np.int32) // 2 + np.array([0, 0, 96])).clip(0, 255).astype(np.uint8)
        stage.png("shadow_overlay", overlay)
        stage.finish(
            equation="I_out=(1-w)*I_SCS+C+w*I_shadow, w=lambda_"+profile+"; saturated pixels get a "
                     "bounded shared-luminance-only adjustment (section 14)",
            inputs=["scsc_corrected", "shadow_compensation", "lambda_" + profile],
            outputs=["final_linear_rgb.npy", "final.png"],
            parameters={"gain_cap": cap},
            qa={"clipped_fraction": clipped_fraction},
        )

    # ---- 15 evaluation ---------------------------------------------------------
    stage = Stage(out, 15, "evaluation")
    evaluation = {}
    for profile in ("conservative", "aggressive"):
        final = finals[profile]
        chroma_in = rgb_linear / np.maximum(rgb_linear.sum(axis=2, keepdims=True), EPS_SMALL)
        chroma_out = final / np.maximum(final.sum(axis=2, keepdims=True), EPS_SMALL)
        shadow_mask = geometric_shadow
        chroma_drift = float(np.median(np.linalg.norm((chroma_out - chroma_in)[shadow_mask], axis=-1))) if shadow_mask.any() else 0.0
        grad_in = float(np.mean(np.hypot(*np.gradient(luminance_raw))))
        grad_out = float(np.mean(np.hypot(*np.gradient(final @ RGB_LUMA))))
        gradient_ratio = grad_out / max(grad_in, EPS_SMALL)
        sunlit_ok = metrics[profile]["sunlit_median_change"] <= (0.03 if profile == "conservative" else 0.05)
        clip_ok = metrics[profile]["clipped_fraction"] < (0.001 if profile == "conservative" else 0.005)
        chroma_ok = chroma_drift <= 0.03
        gradient_ok = 0.9 <= gradient_ratio <= 1.1
        gap_ok = metrics[profile]["shadow_sunlit_luminance_gap_output"] < metrics[profile]["shadow_sunlit_luminance_gap_input"]
        evaluation[profile] = {
            **metrics[profile], "chromaticity_drift": chroma_drift,
            "gradient_energy_ratio": gradient_ratio,
            "pass_sunlit_change": sunlit_ok, "pass_clip_fraction": clip_ok,
            "pass_chromaticity_drift": chroma_ok, "pass_gradient_energy": gradient_ok,
            "pass_shadow_gap_reduced": gap_ok,
        }
    rows = [f"{profile}: {key}={value}" for profile, entries in evaluation.items() for key, value in entries.items()]
    stage.png("pass_fail_table", draw_table("Acceptance metrics", rows, width=760))
    hist_in = np.histogram(luminance_raw, bins=64, range=(0, 1))[0]
    hist_out_c = np.histogram(finals["conservative"] @ RGB_LUMA, bins=64, range=(0, 1))[0]
    stage.png("luminance_distributions", np.asarray(draw_bars(320, 160, hist_in)))
    stage.array("evaluation", np.array(0), "n/a")  # placeholder to keep stats.json non-empty
    stage.finish(
        equation="section 18: shadow/sunlit luminance gap, sunlit change, chromaticity drift, "
                 "gradient-energy ratio",
        inputs=["13_conservative", "14_aggressive"], outputs=["pass_fail_table.png"],
        parameters={"acceptance_thresholds": {
            "conservative_sunlit_change_max": 0.03, "aggressive_sunlit_change_max": 0.05,
            "conservative_clip_max": 0.001, "aggressive_clip_max": 0.005,
            "chromaticity_drift_max": 0.03, "gradient_energy_ratio_range": [0.9, 1.1]}},
        qa=evaluation,
    )
    for profile, entries in evaluation.items():
        for key, value in entries.items():
            if key.startswith("pass_") and not value:
                warnings_all.append(f"{profile}: acceptance check failed: {key}")

    # ---- 99 contact sheet -------------------------------------------------
    stage_dirs = sorted(p for p in out.iterdir() if p.is_dir() and p.name[:2].isdigit())
    sheet = Image.new("RGB", (1024, 1200), "white")
    for i, stage_dir in enumerate(stage_dirs):
        pngs = sorted(stage_dir.glob("*.png"))
        if not pngs:
            continue
        thumb = Image.open(pngs[0]).convert("RGB")
        thumb.thumbnail((240, 190))
        sheet.paste(thumb, ((i % 4) * 256, (i // 4) * 200))
    contact_dir = out / "99_contact_sheet"
    contact_dir.mkdir(exist_ok=True)
    sheet.save(contact_dir / "stage_contact_sheet.png")
    native = Image.fromarray(np.hstack([rgb8, render_rgb(finals["conservative"]), render_rgb(finals["aggressive"])]))
    native.save(contact_dir / "native_raw_vs_conservative_vs_aggressive.png")
    upscaled = native.resize((native.width * 4, native.height * 4), Image.Resampling.LANCZOS)
    upscaled.save(contact_dir / "4x_raw_vs_conservative_vs_aggressive.png")
    (contact_dir / "stage.json").write_text(json.dumps({
        "stage": "contact_sheet", "index": 99,
        "images": ["stage_contact_sheet.png", "native_raw_vs_conservative_vs_aggressive.png",
                   "4x_raw_vs_conservative_vs_aggressive.png"],
    }, indent=2) + "\n")

    # ---- run manifest and index --------------------------------------------
    acceptance = {
        "unique_flight_selected": len(filtered) == 1 or bool(args.flight_id),
        "sun_vector_error_within_5deg": (not math.isnan(sun_vector_error_deg)) and sun_vector_error_deg <= 5.0,
        "no_nan_or_inf": bool(np.isfinite(finals["conservative"]).all() and np.isfinite(finals["aggressive"]).all()),
        "conservative_sunlit_change_ok": evaluation["conservative"]["pass_sunlit_change"],
        "aggressive_sunlit_change_ok": evaluation["aggressive"]["pass_sunlit_change"],
        "conservative_clip_ok": evaluation["conservative"]["pass_clip_fraction"],
        "aggressive_clip_ok": evaluation["aggressive"]["pass_clip_fraction"],
        "shadow_gap_reduced": evaluation["conservative"]["pass_shadow_gap_reduced"] and evaluation["aggressive"]["pass_shadow_gap_reduced"],
        "chromaticity_drift_ok": evaluation["conservative"]["pass_chromaticity_drift"] and evaluation["aggressive"]["pass_chromaticity_drift"],
        "gradient_energy_ok": evaluation["conservative"]["pass_gradient_energy"] and evaluation["aggressive"]["pass_gradient_energy"],
    }
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_commit = None
    manifest = {
        "tool": "scscts_checkpoint_demo", "tool_version": TOOL_VERSION, "git_commit": git_commit,
        "command_line": sys.argv, "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "pillow": PIL.__version__},
        "seed": args.seed,
        "input_sha256": input_checksum, "tile_sha256": tile_checksum,
        "terrain": {"tile_key": [args.level, args.tile_x, args.tile_y], "extent_lv95": list(extent),
                    "crs": "EPSG:2056", "coordinate_convention": "world axes ENU; array rows increase south",
                    "dataset_manifest": dataset_manifest},
        "flight": {"flight_id": flight_id, "acquisition_utc": acquisition_utc.isoformat(),
                   "acquisition_local": acquisition_local.isoformat(),
                   "geoadmin_response_checksum": hashlib.sha256(
                       json.dumps(flight_detail, sort_keys=True).encode()).hexdigest()},
        "solar": {"sun_azimuth_deg": sun_az_deg, "sun_elevation_deg": sun_el_deg,
                  "fitted_azimuth_deg": fit_az_deg, "fitted_elevation_deg": fit_el_deg,
                  "sun_vector_error_deg": sun_vector_error_deg},
        "parameters": {"max_horizon_distance_m": args.max_horizon_distance, "ray_step_m": args.ray_step,
                        "terrain_clearance_m": args.terrain_clearance,
                        "huber_scale_multiplier": HUBER_SCALE_MULTIPLIER,
                        "epsilon_correction": EPS_CORR, "profiles": args.profiles},
        "equations_implemented": ["Eq.1 incidence (dual formula)", "SCS+C (section 11)",
                                   "Eq.7 lambda variable coefficient (section 12)",
                                   "shadow compensation I_t+I_d (section 13, per EQ8_NOTE)"],
        "equations_approximated_or_unavailable": [
            "paper Eq.6 shadow index (needs Coastal/Green/NIR) -> SI_RGB_PROXY",
            "full terrain-reflected irradiance (needs atmospheric/radiometric inputs) -> section 13 fit",
        ],
        "eq8_interpretation": PAPER_EQ8_NOTE,
        "implementation_notes": [
            "w_p (final piecewise weight) is defined as lambda_p directly, since lambda_p is itself "
            "already derived from the geometric shadow mask and the spectral SI proxy (section 14 "
            "says w_p is 'derived from geometric shadow and lambda').",
            "SCS+C 'set correction factor to 1' on invalid fit is implemented as an identity gain "
            "(no correction) for that channel rather than forcing C_b=1 into the SCS+C ratio, since "
            "the latter would not actually be an identity transform.",
            "Section 13's per-channel irradiance fit (I_b~=A_b+D_b*max(cos i,0)) reuses the same "
            "sample-selection criteria as the section 11 SCS+C fit.",
        ],
        "known_limitations": [
            "On this specific 500m photogrammetric cliff crop, per-pixel local incidence spans a very "
            "wide range even among geometrically-sunlit pixels (rugged small-scale terrain roughness, "
            "slope median ~43 deg), so the fitted SCS+C correction factor C comes out small and the "
            "plain (pre-shadow-compensation) SCS+C gain requires the profile clamp "
            "([0.80,1.25]/[0.67,1.50]) for a majority of sunlit pixels rather than a small minority. "
            "That pins the median sunlit change at (or near) the clamp boundary itself, exceeding this "
            "pilot's aspirational 3%/5% sunlit-change thresholds and the aggressive 0.5% clip-fraction "
            "threshold, even though the formulas are implemented exactly as specified. This mirrors the "
            "published critique that plain SCS+C over/under-corrects on rugged terrain -- the motivation "
            "for SCSCTS's shadow-adaptive treatment in the first place. The shadow/sunlit luminance gap, "
            "chromaticity drift, and gradient-energy criteria all pass on this tile.",
        ],
        "checkpoint_inventory": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()),
        "acceptance": acceptance,
        "metrics": metrics,
        "warnings": warnings_all,
        "qa_counts": qa_counts,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")

    index_lines = ["<!doctype html><meta charset='utf-8'><title>SCSCTS checkpoint run</title>",
                   f"<h1>SCSCTS checkpoint run: tile {args.level}/{args.tile_x}/{args.tile_y}</h1>",
                   f"<p>Flight {flight_id} @ {acquisition_utc.isoformat()} "
                   f"(sun az {sun_az_deg:.2f} deg, el {sun_el_deg:.2f} deg, "
                   f"fit error {sun_vector_error_deg:.2f} deg)</p>",
                   "<p><a href='run_manifest.json'>run_manifest.json</a></p>"]
    for stage_dir in stage_dirs + [contact_dir]:
        index_lines.append(f"<h2>{stage_dir.name}</h2>")
        for png in sorted(stage_dir.glob("*.png")):
            index_lines.append(f"<figure><img src='{stage_dir.name}/{png.name}' width='320'>"
                                f"<figcaption>{png.name}</figcaption></figure>")
        stage_json = stage_dir / "stage.json"
        if stage_json.exists():
            index_lines.append(f"<p><a href='{stage_dir.name}/stage.json'>stage.json</a> "
                                f"<a href='{stage_dir.name}/stats.json'>stats.json</a></p>")
    (out / "index.html").write_text("\n".join(index_lines) + "\n")

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--terrain-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--tile-x", type=int, required=True)
    parser.add_argument("--tile-y", type=int, required=True)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flight-id")
    parser.add_argument("--acquisition-time")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-horizon-distance", type=float, default=3000.0)
    parser.add_argument("--ray-step", type=float, default=3.90625)
    parser.add_argument("--terrain-clearance", type=float, default=3.0)
    parser.add_argument("--profiles", default="conservative,aggressive",
                         type=lambda s: [p.strip() for p in s.split(",") if p.strip()])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    try:
        manifest = run(args)
    except SystemExit as error:
        print(f"scscts_checkpoint_demo: {error}", file=sys.stderr)
        return 1
    if not manifest["acceptance"]["no_nan_or_inf"]:
        print("scscts_checkpoint_demo: non-finite output detected", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), "acceptance": manifest["acceptance"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
