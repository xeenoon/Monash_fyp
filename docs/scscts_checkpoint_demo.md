# SCSCTS shadow-correction pilot (`tools/scscts_checkpoint_demo.py`)

A checkpointed, single-tile pilot of the SCS+C topographic correction with
shadow compensation from:

- Chen et al., *"SCSCTS: An improved SCS+C topographic correction model with
  shadow compensation for mountainous regions"*, PLOS ONE, 2024,
  [doi:10.1371/journal.pone.0347784](https://doi.org/10.1371/journal.pone.0347784).
- Supporting shadow-index method: Yang et al., *"A Correction Method of NDVI
  Topographic Shadow Effect for Rugged Terrain"*,
  [doi:10.1109/JSTARS.2022.3193419](https://doi.org/10.1109/JSTARS.2022.3193419).
- Sun position: [NOAA General Solar Position Calculations](https://gml.noaa.gov/grad/solcalc/solareqns.PDF).
- Flight metadata: [SWISSIMAGE RS](https://www.swisstopo.admin.ch/en/orthoimage-swissimage-rs) via
  the [GeoAdmin identify API](https://docs.geo.admin.ch/access-data/identify-features.html).

It is a review artefact, not a batch processor: every stage writes PNG
previews, raw `.npy` arrays, `stats.json`, and `stage.json` so a reviewer can
inspect every intermediate quantity without re-running anything.

## Why this exists, and what it is not

The paper needs Coastal/Green/NIR reflectance and calibrated irradiance. This
pilot has one 256x256 8-bit sRGB crop and a DEM. Every place that stands in
for a paper quantity it cannot compute exactly is named `..._RGB_PROXY` (or
documented as `EXACT_..._UNAVAILABLE`) and is never reported as the paper's
statistic. All such deviations, plus every fitted parameter and constant, are
recorded in `run_manifest.json`.

The published paper's Equation 8 duplicates Equation 6 in print. This tool
follows the surrounding prose instead:

```
I_out = I_SCS+C          sunlit pixel
I_out = I_t + I_d        shadow pixel
```

## Coordinate and array conventions

- World axes: East, North, Up. EPSG:2056 axis order is easting, northing.
- Image/DEM array rows increase **south**; columns increase **east**.
- Solar azimuth is clockwise from true north; elevation is above the horizon.
- Internal angles are radians; JSON reports both radians and degrees.
- Pixel centre `(row, col)` of a 256x256 crop over `extent=(W,S,E,N)`:

  ```
  E = W + (col+0.5)(E-W)/256
  N = N - (row+0.5)(N-S)/256
  ```

- Terrain normal from east/south gradients `g_E, g_S` (`g_N = -g_S`):

  ```
  n = [-g_E, g_S, 1] / ||...||
  slope  alpha = atan(sqrt(g_E^2 + g_N^2))
  aspect beta  = atan2(-g_E, -g_N) mod 2*pi   (clockwise from north)
  cos(i) = n . s == cos(alpha)cos(theta_s) + sin(alpha)sin(theta_s)cos(phi_s - beta)
  ```

  Both incidence formulas are computed and asserted to agree within `1e-6`
  every run (checkpoint `04_incidence`).

## Pipeline stages

| # | Stage | What it computes |
|---|-------|-------------------|
| 00 | `input` | sRGB->linear (`linear_intensity`, never "reflectance"), luminance, checksums |
| 01 | `alignment` | pixel-centre/gutter validation against the repo's `imagery/{level}/{x}/{y}.png`, DEM hillshade |
| 02 | `flight_and_sun` | GeoAdmin flight-strip selection, NOAA sun position, image-fitted sun cross-check |
| 03 | `terrain_geometry` | gradients, slope, aspect, ENU normal |
| 04 | `incidence` | dual-formula cos(i), disagreement map, positive-incidence mask |
| 05 | `shadow_geometry` | self-shadow, cast-shadow raymarch, horizon clearance, truncation QA |
| 06 | `scsc_fit` | per-channel `I_t = m*cos(i) + b` robust fit, correction factor `C=b/m` |
| 07 | `scsc_corrected` | diagnostic SCS+C RGB, signed difference, profile cap masks |
| 08 | `shadow_index` | `SI_RGB_PROXY` (dark residual + blue chromaticity + feathered shadow mask), Otsu threshold |
| 09 | `lambda` | Eq.7 variable coefficient, conservative (1m feather) / aggressive (0.5m, floor 0.5) |
| 10 | `sky_view_albedo` | `V_d=(1+cos(alpha))/2`, `C_t=1-V_d`, Gaussian-weighted (~60m) adjacent albedo |
| 11 | `irradiance` | per-channel `I_b~=A_b+D_b*max(cos i,0)`, `E_s, E_k, E_a, E_w` |
| 12 | `shadow_compensation` | `I_d`, gain-capped compensation blended via shared luminance ratio |
| 13/14 | `conservative` / `aggressive` | final piecewise blend, saturated-pixel handling |
| 15 | `evaluation` | shadow/sunlit luminance gap, chromaticity drift, gradient-energy ratio, pass/fail table |
| 99 | `contact_sheet` | stage thumbnails, native and 4x raw/conservative/aggressive comparison |

`index.html` at the run root links every stage's images and JSON so a
reviewer can browse the whole run without a notebook server.

## CLI

```sh
python3 tools/scscts_checkpoint_demo.py \
  --input path/to/256x256.png \
  --terrain-root alps-data/trn-alps-16km/tiles \
  --manifest alps-data/trn-alps-16km/manifest.json \
  --level 5 --tile-x 18 --tile-y 11 --year 2024 \
  --output out/scscts_tile_5_18_11
```

Useful flags: `--flight-id` / `--acquisition-time` to override or bypass
GeoAdmin lookup entirely (combine both for a fully offline, deterministic
run); `--offline` to forbid network access (requires a cached response, or
the override pair above); `--max-horizon-distance`, `--ray-step`,
`--terrain-clearance` to tune the cast-shadow raymarch; `--profiles` to
select which of `conservative,aggressive` to produce; `--seed` for
determinism; `--force` to replace an existing output directory (refused for
a short list of protected repo paths).

Exit code is nonzero for invalid input dimensions, a missing terrain tile, an
ambiguous flight-strip match, or any non-finite output.

## Known implementation decisions worth knowing about

- **Robust regression scale.** Every Huber-loss fit (image sun-direction,
  SCS+C, dark residual, per-channel irradiance) uses an adaptive `f_scale =
  10 * MAD(residuals)`, refined over 5 IRLS-style iterations, instead of a
  fixed constant. A fixed small scale (e.g. "8% of the linear-intensity
  range") looked reasonable in isolation but badly over-suppressed the true
  signal on this real, multi-material rugged tile, and a single-pass
  MAD estimate is itself fooled when a few gross outliers dominate a small
  sample (see `tests/scscts_checkpoint_demo_tests.py::test_robust_regression_resists_injected_outliers`).
- **`w_p` (final blend weight).** Section 14 says the piecewise weight is
  "derived from geometric shadow and lambda"; this implementation sets
  `w_p = lambda_p` directly, since `lambda_p` itself already folds in the
  geometric shadow mask (via its feather term) and the spectral proxy.
- **Invalid SCS+C fit.** "Set the correction factor to 1" is implemented as
  an identity multiplicative gain for that channel, not literally `C_b=1`
  plugged into the SCS+C ratio (which would not itself be an identity
  transform).
- **NOAA solar position.** Implemented exactly per the published PDF,
  including its leap-year 366-day fractional-year rule (2024 is a leap
  year). For this pilot's `2024-08-24T08:31Z` fixture this pipeline computes
  azimuth/elevation roughly 0.3-0.4 degrees away from shadowrmplan.md's
  quoted approximate reference figures (~118.611/38.042 deg vs. this
  implementation's ~118.17/38.31 deg); the formula was independently
  verified against NOAA's own PDF text, so the discrepancy most plausibly
  traces to the reference figures having been produced by a higher-order
  solar-position algorithm rather than the simplified formula this section
  specifies implementing. `run_manifest.json` records this.
- **On this specific real Alps cliff tile**, the pilot's aspirational 3%/5%
  "sunlit pixels change by no more than..." and the aggressive 0.5%
  clip-fraction acceptance thresholds are not met, even with the formulas
  implemented exactly as specified: the crop's local incidence angle varies
  enormously pixel-to-pixel (median slope ~43 degrees), so the fitted SCS+C
  correction factor `C` is small and the profile gain clamp
  (`[0.80,1.25]`/`[0.67,1.50]`) ends up active for a majority of
  geometrically-sunlit pixels rather than a small minority. This mirrors a
  known, published limitation of plain SCS+C on rugged terrain -- the
  motivation for SCSCTS's shadow-adaptive extension in the first place. The
  shadow/sunlit luminance-gap reduction, chromaticity-drift, and
  gradient-energy-ratio criteria all pass on this tile. See
  `run_manifest.json`'s `known_limitations` and `acceptance` fields for the
  measured numbers.

## Testing

```sh
python3 tests/scscts_checkpoint_demo_tests.py
ctest --test-dir build --output-on-failure -R scscts
```

The integration tests build their own small synthetic multi-tile dataset via
`tools/terrain_tiles.py build` (the same pattern as `tests/phase3_tests.py`)
so they do not depend on the gitignored, machine-local `alps-data/` dataset.
Running the pilot against the real Alps dataset (as in the CLI example
above) is a manual, exploratory step, not part of the hermetic test suite.
