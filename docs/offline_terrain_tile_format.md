# Offline terrain tile format (`.trn` version 1)

The offline pipeline is `tools/terrain_tiles.py`. Inputs must already describe
the same extent in the same documented CRS/profile. Reprojection is intentionally
an offline GDAL/PROJ concern; neither the renderer nor `terrain_tile_load` needs
those libraries.

Tiles use XYZ addressing: `x` increases east, `y` increases south, and `(0,0)`
is the north-west tile. A complete dataset contains `4^level` tiles at every
level. Height samples are vertex samples, so adjacent interiors share their edge
sample. At least one sample beyond that edge is stored as a gutter. Imagery uses
pixel areas and has the same gutter width.

## Binary layout

Every integer and IEEE-754 value is little-endian. Files are not native C struct
dumps. The fixed 224-byte header is followed, without padding, by:

1. height samples (`R16_UNORM` or float32);
2. a validity bitset (one bit per sample, low bit first, `1` means valid);
3. optional inline imagery bytes;
4. UTF-8 profile, source/version JSON, and imagery URI strings (not NUL-ended).

The header records each payload length and CRC-32 of the complete payload. It
also stores the tile and parent key, west/south/east/north extent, valid sample
count, decoded minimum/range, geometric error, and double-precision local-to-
world transform. `R16_UNORM` decodes as:

```text
height_m = min_height_m + encoded / 65535 * height_range_m
```

No-data is represented only by the validity bitset. A numeric height value is
never reserved as a no-data sentinel. Imagery is normally an external PNG URI,
allowing it to be loaded/uploaded independently; its mips are filtered in linear
light and converted back to sRGB.

### Version 1 header offsets

| Offset | Type | Field |
|---:|---|---|
| 0 | `u32` | magic `TRN1` (`0x314e5254`) |
| 4 | `u16` | version (`1`) |
| 6 | `u16` | flags: parent, imagery, no-data |
| 8 | `u32` | header size (`224`) |
| 12 | `3 x u32` | level, x, y |
| 24 | `3 x u32` | parent level, x, y; all `UINT32_MAX` for root |
| 36 | `4 x u16` | sample width, height, gutter, height encoding |
| 44 | `3 x f32` | minimum, range, geometric error in metres |
| 56 | `u32` | valid sample count |
| 60 | `6 x u32` | height, validity, inline imagery, profile, source, URI byte counts |
| 84 | `u32` | payload CRC-32 |
| 88 | `4 x f64` | west, south, east, north extent |
| 120 | `9 x f64` | column-major local-to-world rotation |
| 192 | `3 x f64` | local-to-world translation |
| 216 | 8 bytes | reserved, written as zero |

Readers reject unknown versions and flags. Version 1 writers emit the external
imagery URI relative to the dataset root; a pager resolves it against the root
that owns `manifest.json`.

## Commands

```sh
python3 tools/terrain_tiles.py build \
  --dem dem.tif --imagery imagery.tif --output tiles \
  --extent WEST SOUTH EAST NORTH --profile EPSG:2056 \
  --levels 3 --samples 65 --imagery-size 256 --gutter 1

python3 tools/terrain_tiles.py validate tiles
python3 tools/terrain_tiles.py inspect tiles --level 2 --output atlas.png
```

The validator checks payload sizes/checksums, decoded ranges, explicit validity,
parent availability, neighbour height and imagery gutters, exact child-extent
unions, non-increasing geometric error, and parent/child imagery orientation.
The inspection atlas labels tile keys and elevation ranges, places imagery next
to decoded height, marks interior/gutter boundaries in red, and checkerboards
no-data samples.
