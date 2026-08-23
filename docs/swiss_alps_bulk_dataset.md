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
road corridors. The remaining strict exclusion stage is implemented by
`tools/mask_swiss_infrastructure.py`. Give it official swissTLM3D road, rail,
and other-transport vectors plus swissBUILDINGS3D footprints. It rasterises the
features on the exact imagery grid and grows exclusions by 12 m (roads), 10 m
(rail), 6 m (buildings), or 8 m (other infrastructure). Its PNG output is white
for natural terrain and black for excluded infrastructure.

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
  --roads vectors/roads.gpkg:roads \
  --railways vectors/railways.gpkg:railways \
  --buildings vectors/buildings.gpkg:buildings \
  --infrastructure vectors/lifts_and_stations.gpkg:infrastructure
```

`manifest.json` is rewritten after every accepted/rejected cell, and curl uses
an adjacent `.part` file for resume. The script never exceeds 20 GiB of accepted
assets, although temporary screened DEMs can briefly add roughly one tile.
