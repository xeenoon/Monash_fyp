# Swiss Alps 20 GiB source cache

`tools/download_swiss_alps.py` builds a resumable source cache from the official
swisstopo STAC service. It intentionally uses the 2 m COG assets for both
`swissALTI3D` and SWISSIMAGE:

- the original 10 cm orthophotos are about 45 MB per square kilometre in the
  existing sample, whereas the matching 2 m orthophoto is about 160 KB;
- `tools/terrain_tiles.py` later reduces colour to 256-pixel per-tile PNGs, so
  the 10 cm source would mostly be discarded;
- the 2 m DEM is about 1.2 MB per square kilometre and is enough for a broad
  Alpine corpus. Hero terrain can be acquired at 0.5 m separately.

The cache is not a renderer dataset. The C runtime only opens one contiguous
`.trn` pyramid and its external PNG imagery; it has no geographic catalogue for
disconnected regions. After acquisition, group accepted COGs into contiguous
LV95 regions, mosaic/crop each group, and run `tools/terrain_tiles.py build` on
each aligned pair. Supporting several regional `.trn` roots is a future runtime
change.

## Selection rule

The script enumerates the Valais, Bern/Uri, Ticino, and Graubünden high-Alps
through STAC, then admits a cell only when its 2 m DEM has all of:

- minimum elevation of at least 1,800 m;
- median elevation of at least 1,950 m;
- at least 180 m local relief.

This deliberately drops towns, low valleys, agricultural land, and nearly all
road corridors. The downloader evaluates every DEM pixel (it does not smooth a
thumbnail before applying the minimum-height rule). The remaining strict
exclusion stage is implemented by
`tools/mask_swiss_infrastructure.py`. Give it official swissTLM3D road, rail,
and other-transport vectors plus swissBUILDINGS3D footprints. It rasterises the
features on the exact imagery grid and grows exclusions by 12 m (roads), 10 m
(rail), 6 m (buildings), or 8 m (other infrastructure). Its georeferenced,
8-bit GeoTIFF output is white for natural terrain and black for excluded
infrastructure. Because the masks retain the source CRS and geotransform, GDAL
can mosaic them directly with the source rasters.

Cells with more than 0.5% black pixels are marked `strictly_usable: false`.
For cells that are retained, mosaic the masks with the source rasters and pass
the resulting mask to `terrain_tiles.py --natural-mask`. Black mask samples
become `.trn` no-data, creating a real mesh hole rather than merely hiding a
road with a texture.

## Commands

The following first command only queries STAC metadata and `Content-Length`
headers, writing a candidate manifest. It downloads no imagery or DEM data.

```sh
python3 tools/download_swiss_alps.py plan \
  --output alps-data/large-2m --budget-gib 20
```

The following performs the actual resumable acquisition and is intentionally
not run automatically by this repository:

```sh
python3 tools/download_swiss_alps.py download \
  --output alps-data/large-2m --budget-gib 20
```

After downloading the official vectors, run the final infrastructure pass:

```sh
python3 tools/mask_swiss_infrastructure.py \
  --dataset alps-data/large-2m \
  --roads vectors/roads.gpkg::roads \
  --roads-where "(KUNSTBAUTE IS NULL OR KUNSTBAUTE NOT IN (1000,1100,1200)) AND (OBJEKTART IS NULL OR OBJEKTART NOT IN (15,16,17))" \
  --railways vectors/railways.gpkg::railways \
  --railways-where "KUNSTBAUTE IS NULL OR KUNSTBAUTE NOT IN (800,900)" \
  --buildings vectors/buildings.gpkg::buildings \
  --infrastructure vectors/roads.gpkg::roads \
  --infrastructure-where "(KUNSTBAUTE IS NULL OR KUNSTBAUTE NOT IN (1000,1100,1200)) AND OBJEKTART IN (15,16,17)" \
  --infrastructure-buffer-m 4 \
  --infrastructure vectors/lifts_and_stations.gpkg::infrastructure \
  --infrastructure-where "" \
  --infrastructure-buffer-m 8 \
  --source-version swissTLM3D-2.4 \
  --source-license "Swisstopo Open Government Data" \
  --workers 8
```

Run the same command with `--dry-run` first to validate every vector source,
layer, CRS, imagery file, and grid without writing masks or changing the
manifest. Attribute filters use OGR SQL and are recorded in the manifest. The
example road filter omits tunnels and underpasses whose geometry is not visible
on the terrain surface; confirm its coded values against the catalogue for the
specific swissTLM3D release in use. For OGR URLs and connection strings that
already contain colons, use a double colon before the layer name, for example
`PG:dbname=terrain::roads`. When several optional infrastructure inputs are
filtered, provide one `--infrastructure-where` for each input in the same order;
use an empty string for an unfiltered input. Optional infrastructure buffers
default to 8 m; repeat `--infrastructure-buffer-m` once per input to override
them.

`TLM_STRASSE` also contains narrow paths and signed hiking routes. Applying the
12 m road buffer to the entire layer is deliberately severe. For a less wasteful
policy, filter those object classes out of `--roads` and pass them separately as
an optional infrastructure layer with a smaller `--infrastructure-buffer-m`.
Choose and document the value for the intended corpus. Do not silently treat the
whole feature class as motor roads.

Vector sources must use LV95 / EPSG:2056. For each imagery cell the tool first
uses the vector dataset's spatial index to extract a small local subset, then
rasterises a canvas padded by the configured buffer. The padding is important:
a road just outside a tile can still exclude pixels inside its edge.

`manifest.json` is atomically rewritten after every completed download batch,
and curl uses an adjacent `.part` file for resume. The script never exceeds the
requested budget of accepted assets, although concurrent screened DEMs can
briefly add up to one batch. DEMs and imagery are transferred in bounded curl
HTTP/2 batches. The first run also writes `discovery.json`; later resumes load
that deterministic STAC candidate catalogue locally. Use
`--refresh-discovery` only when intentionally replacing it with a fresh
catalogue snapshot.
