# Grading the AI shadow-removal masters

`alps-data/trn-alps-16km/shadowfree-masters/` are 19 hand-selected 2km×2km
tiles that were run through an external AI shadow-removal tool
(`tools/integrate_shadowfree_alps.py` installs them into the runtime
imagery pyramid). Two defects showed up after installing them and are fixed
by `tools/grade_shadowfree_masters.py`, in this order:

1. **Every tile came out too bright.** `shaders/terrain.frag`'s own neutral
   rock reference (`material_macro_reference_luminance(7u)` = 0.192 linear,
   ~0.47 sRGB) is what the shader itself treats as "normal rock" — the raw
   AI output measured close to double that in sunlit patches. This read as
   pale/chalky/washed-out once auto-exposure was fixed to stop masking it
   (fixed exposure was crushing everything dark enough to hide the problem).
2. **Neighbouring tiles disagree with each other**, in two separable ways:
   average brightness/colour, and how much fine surface detail each one
   carries (a smooth tile sitting next to a densely-cracked one). The
   underlying AI-synthesized detail also doesn't always geometrically
   continue across a shared border — each tile was processed independently
   with no cross-tile consistency guarantee.

Everything below was worked out and validated on one deliberately worst-case
2×2 cluster (picked by an automated seam-mismatch scorer, see the "finding
the worst cluster" section) before running on the full 19-tile set — each
stage is a real, separable fix for a real, separately-diagnosed problem, not
a bundle of guesses.

## Running it

```
python3 tools/grade_shadowfree_masters.py \
    --source-dir alps-data/trn-alps-16km/shadowfree-masters \
    --output-dir <somewhere>

python3 tools/build_graded_alps_dataset.py \
    --masters-dir <somewhere> \
    --output alps-data/trn-alps-16km-graded-v2
```

The first command grades and cross-matches all masters (a few seconds for
19 tiles). The second installs them into a full runtime dataset: it copies
`UNSHADOWED_ALPS_DIR`'s `imagery/` and overwrites just the level-5 tiles the
masters cover (plus their 1px neighbour gutter), then rebuilds ancestor
levels 0–4 from the new level-5 mosaic. `tiles/` and `manifest.json` are
symlinked from the source dataset since geometry never changes here.

Point the renderer at the result with `TERRAIN_DATASET=alps-data/trn-alps-16km-graded-v2`,
or make it the `ATLAS=NEW` default (`GRADED_ALPS_DIR` in `CMakeLists.txt` /
`src/main.c`'s terrain-selection block) once you're happy with it.

## Finding the worst cluster (diagnostic, not part of the pipeline)

Before building the real fix, `tools/` doesn't have this as a committed
script (it lived in a scratch dir during development) — the method, for
future reference: parse each master's `E<min>-<max>_N<min>-<max>.png`
filename for its real-world bounding box, find every 2×2 block of four
masters whose boxes tile a 4km×4km square, and score each cluster by
comparing 12px-wide strips on either side of its four internal seams (mean
RGB difference, `band` = pixels from each edge). Highest score = worst
seam. This is also how `grade_shadowfree_masters.py`'s adjacency detection
works in general (`find_adjacency`), just generalized from "2×2 clusters"
to "every pair that shares a border," since the real 19-tile set isn't a
clean grid.

## The four stages

### A. Shared gamma, not per-tile

Pool every master's sub-highlight (`luminance < 0.75`, i.e. "probably rock
or grass, not snow") pixels into one set, take the median, solve for the
gamma that maps it to the 0.47 sRGB target. Apply that **one** gamma to
every tile identically.

The first version of this computed gamma **per tile** from that tile's own
median. That's wrong for two reasons, one obvious in hindsight: a per-tile
statistic lets neighbouring tiles land on different gammas whenever their
rock/snow mix differs, which directly creates a brightness seam that wasn't
there before. The other reason took longer to find: several masters are
mostly snow (>75% of pixels), and snow is legitimately supposed to be
bright (the shader's own snow reference is ~0.91 sRGB) — matching a
whole-tile mean to the *rock* target on a snow-dominated tile computes an
absurd gamma (we saw up to 5.3) that crushes the small rock portion to
near-black while barely touching the snow. Pooling only sub-highlight
pixels *across the whole dataset* fixes both: one gamma structurally can't
disagree with itself at a border, and snow never enters the statistic that
decides it.

A `HIGHLIGHT_LOW`–`HIGHLIGHT_HIGH` (0.75–0.92) smoothstep protects anything
already near the snow reference from the correction entirely, tapering the
gamma's effect to zero as a pixel's own luminance approaches "definitely
snow." Local contrast (small unsharp mask), saturation, and a gentle
midtone S-curve are applied per-pixel afterward, same function everywhere.

### B. Whole-tile gain compensation (global least-squares, not a sweep)

Grading a tile with the shared gamma removes cross-tile disagreement from
*that specific step*, but the masters still disagree with each other going
in — the AI tool's independent per-tile output already had inconsistent
brightness and detail density before any of this touched it (confirmed by
scoring the *ungraded* masters the same way: nonzero, non-negligible seam
scores even with no grading applied at all).

For every adjacent pair, measure the mean disagreement in a strip along
their shared border (12px band for brightness, 24px for texture-energy —
texture-energy is noisier and needs the wider sample). Then solve, across
**every edge in the dataset simultaneously**, for one whole-tile correction
per tile that best cancels all of that disagreement at once. This is the
"gain compensation" step orthomosaic/panorama stitching pipelines use for
radiometric normalization (e.g. Brown & Lowe 2007) — a small least-squares
problem: `correction[a] - correction[b] ≈ -offset[(a,b)]` for every edge
`(a,b)`, solved with `numpy.linalg.lstsq`. It's under-determined by exactly
one free constant per connected component of the adjacency graph (shifting
every tile in a component by the same amount changes no pairwise
agreement); `lstsq`'s minimum-norm solution handles that by centering each
component's corrections around zero rather than needing an arbitrarily
chosen anchor tile. The full 19-tile set is one connected component, so
this is one solve, not per-region.

Two independent solves, same math, different measurement:

- **Additive RGB correction** (closes the brightness/colour gap).
- **Multiplicative gain on each tile's detail layer**, `image − small_blur`
  (closes the texture-energy gap) — solved in log-space since gain is
  multiplicative and must stay positive.

**Why not a sequential sweep** ("start top-left, match its neighbours, step
right, repeat"): a sequential pass accumulates error along whatever path
you sweep, and the result depends on where you start and which direction
you go — tile 19 in the sweep order inherits the accumulated slop from
tiles 1–18. Solving every constraint jointly doesn't have a sweep order to
depend on, and costs about the same to compute (this is a system with as
many unknowns as tiles and as many equations as edges — 19 and 23 here,
trivial for `lstsq`).

Because the correction is a single constant applied to the **whole tile**,
not just its border, it cannot introduce a new seam on its own — at worst
it leaves the *average* disagreement at zero without fixing the disagreement
at the literal boundary pixel, which is what stage C is for.

### C. Residual feather

A whole-tile constant matches the *average* border disagreement, not the
exact boundary row/column — there's usually a small residual left right at
the seam, and that reads as a thin hard line even when the whole-tile
statistics now agree. A short (~40–64px) additive feather for brightness and
a matching local detail-gain feather (fading multiplicative correction on
the detail layer, closing the residual specifically) mop this up.

This is deliberately the *second* fix, not the *only* one — an earlier
attempt used feathering alone with no whole-tile correction underneath, and
it had to hide a much bigger gap (the full per-tile-gamma disagreement)
within a fixed narrow margin, which just moved the hard edge to the margin
boundary instead of removing it. With stage B already closing most of the
gap, the feather here only needs to polish what's left.

### D. Multi-band blend (properly tapering)

Stages A–C fix *statistics* (mean colour, mean detail density). They can't
fix **geometry** — a crevasse or drainage line that should cross a border
and just doesn't, because the AI tool regenerated fine detail independently
on each side with no promise the two sides' features line up. A
cross-correlation check for a simple translational misalignment between
tiles was tried and came back weak (~3σ peak, not a confident detection) —
this isn't a registration error, it's genuinely different synthesized
detail, so there's no clean shift that reconnects it.

The fix is a band-pass approximation of Laplacian-pyramid blending: decompose
each side's edge region into frequency bands (successive box-blur residuals,
radii `[2, 6, 18, 54, 162]`, plus a coarsest residual), and blend each band
between the two tiles using a mask specific to that band — low frequencies
blend over a wide margin, high frequencies over a narrow one, avoiding the
"double vision" ghosting a single flat cross-fade produces on sharp
features. Since tiles don't truly overlap, each side's edge band is
mirror-extended past its real boundary first, giving the blend real (if
approximate) content on both sides of the seam to work with — accurate
right at the seam where blend weight is highest, and mattering less
further out where its weight fades toward zero anyway.

**The bug worth remembering:** the first version blurred a hard 0/1 seam
mask by each band's own radius, inside a small fixed window. The coarsest
band's radius (up to `162 * 3 = 486`) was far wider than that window
(`64px` at the time), so its mask never actually reached 0 or 1 before the
code spliced back into the unmodified tile — producing a *second*, harder
line at the edge of the blend region, not at the seam itself. The fix:
explicitly cap every band's transition width at the blend window's own
half-size (`taper`, 200px), via `smoothstep(taper - width, taper + width, y)`
with `width = min(radius * 1.5, taper)`. This *guarantees* every band hits
exactly its asymptote by the window edge regardless of blur radius, so the
blend is provably continuous where it rejoins the unblended tile, not just
"usually close enough."

## Known limitations

- **Corner triple-points.** A tile with both an east and a south neighbour
  gets blended against each independently; where their blend windows
  overlap near the shared corner, the second blend applied there partially
  overrides the first's effect in that small region. Not solved here — a
  true N-way corner blend was judged out of scope given how small the
  affected area is (a `taper`-px corner square, `200×200` out of a
  `1024×1024` tile) relative to the win from doing the rest of this at all.
- **Whole-tile correction assumes tiles are describing genuinely comparable
  material.** If a future master is legitimately supposed to look different
  from its neighbours (different rock type, not a grading artifact), this
  pipeline will still try to match it to them. Nothing here distinguishes
  "artifact" from "real geological difference" except the sub-highlight/
  highlight-protection split for snow.
- **Verification is score-based, not exhaustive-visual.** The pipeline
  prints a brightness/texture-energy score per edge (see `seam_score_rgb`
  / `seam_score_texture` in the script) and a worst-case summary — it does
  not render every seam and check it by eye. Spot-check new output before
  trusting it, the same way the worst-cluster swatches were eyeballed
  during development.

## Tuning knobs (all at the top of `tools/grade_shadowfree_masters.py`)

| constant | meaning | validated value |
|---|---|---|
| `TARGET_LUMINANCE` | grading target, derived from `shaders/terrain.frag`'s rock reference | ~0.47 sRGB |
| `HIGHLIGHT_LOW` / `HIGHLIGHT_HIGH` | snow-protection band | 0.75 / 0.92 |
| `BORDER_BAND` / `TEXTURE_BORDER_BAND` | strip width for measuring disagreement | 12px / 24px |
| `RGB_FEATHER_MARGIN` / `DETAIL_FEATHER_MARGIN` | stage C margins | 40px / 64px |
| `BLEND_TAPER` | stage D blend window half-height | 200px |
| `BLEND_RADII` | stage D band-pass levels | `[2, 6, 18, 54, 162]` |

If a future dataset needs different values (much larger/smaller tiles,
different source imagery scale), re-derive them the same way these were:
build one deliberately worst-case cluster, score it before and after, and
look at the result at native resolution zoomed into the actual seam — not
just the downscaled overview.
