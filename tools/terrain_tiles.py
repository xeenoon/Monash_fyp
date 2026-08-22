#!/usr/bin/env python3
"""Build, validate, and inspect deterministic offline terrain tile datasets.

Inputs must already cover the same extent in the profile supplied with
``--profile``. Geospatial reprojection belongs at this offline boundary (for
example, with gdalwarp); the renderer and .trn loader deliberately do not link
GDAL or PROJ.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageStat

Image.MAX_IMAGE_PIXELS = None

MAGIC = 0x314E5254  # TRN1, little-endian
VERSION = 1
HEADER_BYTES = 224
ROOT_KEY = 0xFFFFFFFF
HAS_PARENT = 1 << 0
HAS_IMAGERY = 1 << 1
HAS_NODATA = 1 << 2
HEIGHT_R16 = 1
HEIGHT_F32 = 2


@dataclass(frozen=True, order=True)
class Key:
    level: int
    x: int
    y: int


@dataclass
class Tile:
    key: Key
    parent: Key | None
    width: int
    height: int
    gutter: int
    encoding: int
    minimum: float
    height_range: float
    geometric_error: float
    valid_count: int
    extent: tuple[float, float, float, float]
    transform: tuple[float, ...]
    flags: int
    heights: bytes
    validity: bytes
    imagery: bytes
    profile: str
    source: str
    imagery_uri: str

    def decoded_heights(self) -> list[float]:
        count = self.width * self.height
        if self.encoding == HEIGHT_F32:
            return list(struct.unpack(f"<{count}f", self.heights))
        encoded = struct.unpack(f"<{count}H", self.heights)
        if self.height_range == 0.0:
            return [self.minimum] * count
        return [self.minimum + self.height_range * value / 65535.0
                for value in encoded]

    def is_valid(self, index: int) -> bool:
        return bool(self.validity[index >> 3] & (1 << (index & 7)))


def _put_u16(header: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<H", header, offset, value)


def _put_u32(header: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", header, offset, value)


def _encode_tile(tile: Tile) -> bytes:
    profile = tile.profile.encode("utf-8")
    source = tile.source.encode("utf-8")
    imagery_uri = tile.imagery_uri.encode("utf-8")
    payload = b"".join((tile.heights, tile.validity, tile.imagery,
                        profile, source, imagery_uri))
    header = bytearray(HEADER_BYTES)
    _put_u32(header, 0, MAGIC)
    _put_u16(header, 4, VERSION)
    _put_u16(header, 6, tile.flags)
    _put_u32(header, 8, HEADER_BYTES)
    _put_u32(header, 12, tile.key.level)
    _put_u32(header, 16, tile.key.x)
    _put_u32(header, 20, tile.key.y)
    parent = tile.parent or Key(ROOT_KEY, ROOT_KEY, ROOT_KEY)
    _put_u32(header, 24, parent.level)
    _put_u32(header, 28, parent.x)
    _put_u32(header, 32, parent.y)
    _put_u16(header, 36, tile.width)
    _put_u16(header, 38, tile.height)
    _put_u16(header, 40, tile.gutter)
    _put_u16(header, 42, tile.encoding)
    struct.pack_into("<fff", header, 44, tile.minimum, tile.height_range,
                     tile.geometric_error)
    _put_u32(header, 56, tile.valid_count)
    for offset, value in zip((60, 64, 68, 72, 76, 80),
                             (len(tile.heights), len(tile.validity),
                              len(tile.imagery), len(profile), len(source),
                              len(imagery_uri))):
        _put_u32(header, offset, value)
    _put_u32(header, 84, zlib.crc32(payload))
    struct.pack_into("<4d", header, 88, *tile.extent)
    struct.pack_into("<12d", header, 120, *tile.transform)
    return bytes(header) + payload


def _read_tile(path: Path) -> Tile:
    data = path.read_bytes()
    if len(data) < HEADER_BYTES:
        raise ValueError(f"{path}: truncated header")
    if struct.unpack_from("<I", data, 0)[0] != MAGIC:
        raise ValueError(f"{path}: bad magic")
    if struct.unpack_from("<H", data, 4)[0] != VERSION:
        raise ValueError(f"{path}: unsupported version")
    if struct.unpack_from("<I", data, 8)[0] != HEADER_BYTES:
        raise ValueError(f"{path}: unsupported header size")
    flags = struct.unpack_from("<H", data, 6)[0]
    key = Key(*struct.unpack_from("<III", data, 12))
    parent_values = struct.unpack_from("<III", data, 24)
    parent = None if parent_values == (ROOT_KEY,) * 3 else Key(*parent_values)
    width, height, gutter, encoding = struct.unpack_from("<HHHH", data, 36)
    minimum, height_range, error = struct.unpack_from("<fff", data, 44)
    valid_count = struct.unpack_from("<I", data, 56)[0]
    sizes = struct.unpack_from("<6I", data, 60)
    expected_crc = struct.unpack_from("<I", data, 84)[0]
    extent = struct.unpack_from("<4d", data, 88)
    transform = struct.unpack_from("<12d", data, 120)
    if sum(sizes) != len(data) - HEADER_BYTES:
        raise ValueError(f"{path}: payload sizes do not match file size")
    payload = data[HEADER_BYTES:]
    if zlib.crc32(payload) != expected_crc:
        raise ValueError(f"{path}: payload checksum mismatch")
    chunks: list[bytes] = []
    cursor = 0
    for size in sizes:
        chunks.append(payload[cursor:cursor + size])
        cursor += size
    heights, validity, imagery, profile, source, imagery_uri = chunks
    return Tile(key, parent, width, height, gutter, encoding, minimum,
                height_range, error, valid_count, extent, transform, flags,
                heights, validity, imagery, profile.decode("utf-8"),
                source.decode("utf-8"), imagery_uri.decode("utf-8"))


def _validity_bytes(valid: Iterable[bool]) -> tuple[bytes, int]:
    values = list(valid)
    result = bytearray((len(values) + 7) // 8)
    count = 0
    for index, value in enumerate(values):
        if value:
            result[index >> 3] |= 1 << (index & 7)
            count += 1
    return bytes(result), count


def _load_height_grid(path: Path, output_size: int,
                      nodata: float | None) -> tuple[list[float], list[bool]]:
    source = Image.open(path).convert("F")
    get_data = getattr(source, "get_flattened_data", source.getdata)
    raw = list(get_data())
    valid = [math.isfinite(value) and (nodata is None or value != nodata)
             for value in raw]
    numerator = Image.new("F", source.size)
    numerator.putdata([value if ok else 0.0 for value, ok in zip(raw, valid)])
    weights = Image.new("F", source.size)
    weights.putdata([1.0 if ok else 0.0 for ok in valid])
    numerator = numerator.resize((output_size, output_size), Image.Resampling.BILINEAR)
    weights = weights.resize((output_size, output_size), Image.Resampling.BILINEAR)
    get_num = getattr(numerator, "get_flattened_data", numerator.getdata)
    get_weight = getattr(weights, "get_flattened_data", weights.getdata)
    result: list[float] = []
    result_valid: list[bool] = []
    for value, weight in zip(get_num(), get_weight()):
        ok = weight > 0.0
        result.append(value / weight if ok else 0.0)
        result_valid.append(ok)
    return result, result_valid


def _linear_channel(channel: Image.Image) -> Image.Image:
    return channel.point(
        lambda value: value / 3294.6 if value <= 10.31475 else
        ((value / 255.0 + 0.055) / 1.055) ** 2.4,
        "F")


def _encode_srgb(channel: Image.Image) -> Image.Image:
    get_data = getattr(channel, "get_flattened_data", channel.getdata)
    encoded = bytearray()
    for value in get_data():
        value = max(0.0, min(1.0, value))
        srgb = 12.92 * value if value <= 0.0031308 else \
            1.055 * value ** (1.0 / 2.4) - 0.055
        encoded.append(round(srgb * 255.0))
    result = Image.new("L", channel.size)
    result.frombytes(bytes(encoded))
    return result


def _resize_srgb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    rgb = image.convert("RGB")
    channels = []
    for band in range(3):
        channel = rgb.getchannel(band)
        linear = _linear_channel(channel)
        linear = linear.resize(size, Image.Resampling.LANCZOS)
        channels.append(_encode_srgb(linear))
    return Image.merge("RGB", channels)


def _slice_grid(values: list, valid: list[bool], global_size: int,
                key: Key, segments: int, gutter: int,
                finest_level: int) -> tuple[list, list[bool]]:
    level_step = 1 << (finest_level - key.level)
    start_x = key.x * segments * level_step
    start_y = key.y * segments * level_step
    result = []
    result_valid = []
    for local_y in range(-gutter, segments + 1 + gutter):
        y = max(0, min(global_size - 1, start_y + local_y * level_step))
        for local_x in range(-gutter, segments + 1 + gutter):
            x = max(0, min(global_size - 1, start_x + local_x * level_step))
            index = y * global_size + x
            result.append(values[index])
            result_valid.append(valid[index])
    return result, result_valid


def _slice_image(image: Image.Image, key: Key, tile_size: int,
                 gutter: int) -> Image.Image:
    start_x, start_y = key.x * tile_size, key.y * tile_size
    output = Image.new("RGB", (tile_size + 2 * gutter,) * 2)
    pixels = output.load()
    source = image.load()
    for y in range(-gutter, tile_size + gutter):
        sy = max(0, min(image.height - 1, start_y + y))
        for x in range(-gutter, tile_size + gutter):
            sx = max(0, min(image.width - 1, start_x + x))
            pixels[x + gutter, y + gutter] = source[sx, sy]
    return output


def _tile_extent(root: tuple[float, float, float, float], key: Key
                 ) -> tuple[float, float, float, float]:
    west, south, east, north = root
    count = 1 << key.level
    x_span, y_span = (east - west) / count, (north - south) / count
    return (west + key.x * x_span, north - (key.y + 1) * y_span,
            west + (key.x + 1) * x_span, north - key.y * y_span)


def _height_payload(values: list[float], valid: list[bool], encoding: int
                    ) -> tuple[bytes, float, float]:
    usable = [value for value, ok in zip(values, valid) if ok]
    minimum = min(usable) if usable else 0.0
    maximum = max(usable) if usable else 0.0
    height_range = maximum - minimum
    if encoding == HEIGHT_F32:
        return struct.pack(f"<{len(values)}f", *values), minimum, height_range
    scale = 65535.0 / height_range if height_range else 0.0
    encoded = [round((value - minimum) * scale) if ok and height_range else 0
               for value, ok in zip(values, valid)]
    return struct.pack(f"<{len(values)}H", *encoded), minimum, height_range


def build(args: argparse.Namespace) -> None:
    if not 1 <= args.levels <= 32 or args.samples < 3 or args.gutter < 1:
        raise ValueError("1 <= levels <= 32, samples >= 3, and gutter >= 1 are required")
    if args.imagery_size < 1:
        raise ValueError("imagery-size must be positive")
    if (args.samples - 1) % 2:
        raise ValueError("samples must be odd so parent and child grids nest exactly")
    if args.root_geometric_error is not None and \
            (not math.isfinite(args.root_geometric_error) or
             args.root_geometric_error < 0.0):
        raise ValueError("root-geometric-error must be finite and non-negative")
    west, south, east, north = args.extent
    if not all(math.isfinite(value) for value in args.extent) or \
            not west < east or not south < north:
        raise ValueError("extent must be finite WEST SOUTH EAST NORTH bounds")
    if not args.profile:
        raise ValueError("profile must not be empty")
    if args.samples + 2 * args.gutter > 65535:
        raise ValueError("height tile dimensions exceed the format limit")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    finest = args.levels - 1
    segments = args.samples - 1
    finest_grid_size = segments * (1 << finest) + 1
    heights, valid = _load_height_grid(Path(args.dem), finest_grid_size,
                                       args.nodata)
    source_image = Image.open(args.imagery)
    root_extent = tuple(args.extent)
    root_error = args.root_geometric_error
    if root_error is None:
        root_error = max(root_extent[2] - root_extent[0],
                         root_extent[3] - root_extent[1]) / segments
    encoding = HEIGHT_F32 if args.height_encoding == "f32" else HEIGHT_R16
    source = json.dumps({"dataset": args.source, "version": args.source_version},
                        sort_keys=True, separators=(",", ":"))

    finest_imagery_size = args.imagery_size * (1 << finest)
    finest_imagery = _resize_srgb(
        source_image, (finest_imagery_size, finest_imagery_size))
    source_image.close()
    imagery_levels: dict[int, Image.Image] = {finest: finest_imagery}
    for level in range(finest):
        size = args.imagery_size * (1 << level)
        imagery_levels[level] = _resize_srgb(finest_imagery, (size, size))

    for level in range(args.levels):
        count = 1 << level
        for y in range(count):
            for x in range(count):
                key = Key(level, x, y)
                samples, sample_valid = _slice_grid(
                    heights, valid, finest_grid_size, key, segments,
                    args.gutter, finest)
                height_payload, minimum, height_range = _height_payload(
                    samples, sample_valid, encoding)
                validity, valid_count = _validity_bytes(sample_valid)
                extent = _tile_extent(root_extent, key)
                error = root_error / count
                image = _slice_image(imagery_levels[level], key,
                                     args.imagery_size, args.gutter)
                image_path = output / "imagery" / str(level) / str(x) / f"{y}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(image_path, "PNG", optimize=False, compress_level=9)
                image_uri = image_path.relative_to(output).as_posix()
                flags = HAS_IMAGERY
                parent = None
                if level:
                    flags |= HAS_PARENT
                    parent = Key(level - 1, x // 2, y // 2)
                if valid_count != len(sample_valid):
                    flags |= HAS_NODATA
                centre_x = (extent[0] + extent[2]) * 0.5
                centre_y = (extent[1] + extent[3]) * 0.5
                transform = (1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1.0,
                             centre_x, minimum, -centre_y)
                tile = Tile(key, parent, args.samples + 2 * args.gutter,
                            args.samples + 2 * args.gutter, args.gutter,
                            encoding, minimum, height_range, error, valid_count,
                            extent, transform, flags, height_payload, validity,
                            b"", args.profile, source, image_uri)
                tile_path = output / "tiles" / str(level) / str(x) / f"{y}.trn"
                tile_path.parent.mkdir(parents=True, exist_ok=True)
                tile_path.write_bytes(_encode_tile(tile))

    manifest = {
        "format": "terrain-tile-dataset",
        "version": VERSION,
        "profile": args.profile,
        "axis_convention": "XYZ: x east, y south; y=0 is north",
        "extent": list(root_extent),
        "levels": args.levels,
        "height_samples": args.samples,
        "imagery_size": args.imagery_size,
        "gutter": args.gutter,
        "height_encoding": args.height_encoding,
        "root_geometric_error_m": root_error,
        "source": json.loads(source),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"built {sum(4 ** level for level in range(args.levels))} tiles in {output}")


def _close_height(a: float, b: float, first: Tile, second: Tile) -> bool:
    tolerance = 1e-5
    if first.encoding == HEIGHT_R16:
        tolerance += first.height_range / 65535.0
    if second.encoding == HEIGHT_R16:
        tolerance += second.height_range / 65535.0
    return abs(a - b) <= tolerance


def _image(dataset: Path, tile: Tile) -> Image.Image:
    if tile.imagery:
        raise ValueError("inline imagery inspection is not implemented")
    return Image.open(dataset / tile.imagery_uri).convert("RGB")


def validate_dataset(dataset: Path) -> list[Tile]:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    paths = sorted((dataset / "tiles").glob("*/*/*.trn"),
                   key=lambda path: tuple(map(int, path.parts[-3:-1])) +
                   (int(path.stem),))
    tiles = [_read_tile(path) for path in paths]
    by_key = {tile.key: tile for tile in tiles}
    expected_count = sum(4 ** level for level in range(manifest["levels"]))
    if len(tiles) != expected_count or len(by_key) != len(tiles):
        raise ValueError(f"expected {expected_count} unique tiles, found {len(tiles)}")

    for tile in tiles:
        count = tile.width * tile.height
        bytes_per_height = 2 if tile.encoding == HEIGHT_R16 else 4
        if tile.encoding not in (HEIGHT_R16, HEIGHT_F32):
            raise ValueError(f"{tile.key}: invalid height encoding")
        if len(tile.heights) != count * bytes_per_height or \
                len(tile.validity) != (count + 7) // 8:
            raise ValueError(f"{tile.key}: invalid raster byte count")
        actual_valid = sum(tile.is_valid(index) for index in range(count))
        if actual_valid != tile.valid_count:
            raise ValueError(f"{tile.key}: valid sample count mismatch")
        decoded = tile.decoded_heights()
        usable = [value for index, value in enumerate(decoded) if tile.is_valid(index)]
        if usable:
            tolerance = tile.height_range / 65535.0 + 1e-5 \
                if tile.encoding == HEIGHT_R16 else 1e-5
            if min(usable) < tile.minimum - tolerance or \
                    max(usable) > tile.minimum + tile.height_range + tolerance:
                raise ValueError(f"{tile.key}: decoded height outside metadata range")
        if tile.profile != manifest["profile"]:
            raise ValueError(f"{tile.key}: profile differs from manifest")
        if tile.parent is None:
            if tile.key.level != 0 or tile.flags & HAS_PARENT:
                raise ValueError(f"{tile.key}: invalid root parent")
        elif tile.parent not in by_key or not tile.flags & HAS_PARENT:
            raise ValueError(f"{tile.key}: parent missing")
        if not (dataset / tile.imagery_uri).is_file():
            raise ValueError(f"{tile.key}: imagery is missing")

    for tile in tiles:
        g = tile.gutter
        interior = tile.width - 2 * g
        decoded = tile.decoded_heights()
        east = by_key.get(Key(tile.key.level, tile.key.x + 1, tile.key.y))
        if east:
            other = east.decoded_heights()
            for y in range(interior):
                for k in range(g):
                    ai = (g + y) * tile.width + g + interior + k
                    bi = (g + y) * east.width + g + 1 + k
                    if tile.is_valid(ai) != east.is_valid(bi) or \
                            (tile.is_valid(ai) and not _close_height(
                                decoded[ai], other[bi], tile, east)):
                        raise ValueError(f"{tile.key}: east height gutter mismatch")
            a_image, b_image = _image(dataset, tile), _image(dataset, east)
            image_interior = a_image.width - 2 * g
            for y in range(image_interior):
                for k in range(g):
                    if a_image.getpixel((g + image_interior + k, g + y)) != \
                            b_image.getpixel((g + k, g + y)):
                        raise ValueError(f"{tile.key}: east imagery gutter mismatch")
        north = by_key.get(Key(tile.key.level, tile.key.x, tile.key.y - 1))
        if north:
            other = north.decoded_heights()
            for x in range(interior):
                for k in range(g):
                    ai = (g - 1 - k) * tile.width + g + x
                    bi = (g + interior - 2 - k) * north.width + g + x
                    if tile.is_valid(ai) != north.is_valid(bi) or \
                            (tile.is_valid(ai) and not _close_height(
                                decoded[ai], other[bi], tile, north)):
                        raise ValueError(f"{tile.key}: north height gutter mismatch")
            a_image, b_image = _image(dataset, tile), _image(dataset, north)
            image_interior = a_image.height - 2 * g
            for x in range(image_interior):
                for k in range(g):
                    if a_image.getpixel((g + x, g - 1 - k)) != \
                            b_image.getpixel((g + x, g + image_interior - 1 - k)):
                        raise ValueError(f"{tile.key}: north imagery gutter mismatch")

    for parent in (tile for tile in tiles if tile.key.level + 1 < manifest["levels"]):
        children = [by_key[Key(parent.key.level + 1, parent.key.x * 2 + dx,
                               parent.key.y * 2 + dy)]
                    for dy in range(2) for dx in range(2)]
        union = (min(child.extent[0] for child in children),
                 min(child.extent[1] for child in children),
                 max(child.extent[2] for child in children),
                 max(child.extent[3] for child in children))
        if union != parent.extent:
            raise ValueError(f"{parent.key}: child extents do not union to parent")
        if any(child.geometric_error > parent.geometric_error for child in children):
            raise ValueError(f"{parent.key}: child geometric error increased")
        parent_heights = parent.decoded_heights()
        segments = parent.width - 2 * parent.gutter - 1
        if segments % 2:
            raise ValueError(f"{parent.key}: height grid does not nest")
        for y in range(segments + 1):
            fine_y = y * 2
            child_y = 0 if fine_y <= segments else 1
            local_y = fine_y if child_y == 0 else fine_y - segments
            for x in range(segments + 1):
                fine_x = x * 2
                child_x = 0 if fine_x <= segments else 1
                local_x = fine_x if child_x == 0 else fine_x - segments
                child = children[child_y * 2 + child_x]
                child_heights = child.decoded_heights()
                pi = (parent.gutter + y) * parent.width + parent.gutter + x
                ci = ((child.gutter + local_y) * child.width +
                      child.gutter + local_x)
                if parent.is_valid(pi) != child.is_valid(ci) or \
                        (parent.is_valid(pi) and not _close_height(
                            parent_heights[pi], child_heights[ci], parent, child)):
                    raise ValueError(f"{parent.key}: parent/child height mismatch")
        # Orientation check: the four XYZ children must form NW, NE, SW, SE.
        g = parent.gutter
        size = _image(dataset, parent).width - 2 * g
        mosaic = Image.new("RGB", (size * 2, size * 2))
        for child in children:
            image = _image(dataset, child).crop((g, g, g + size, g + size))
            mosaic.paste(image, ((child.key.x & 1) * size,
                                 (child.key.y & 1) * size))
        expected = _resize_srgb(mosaic, (size, size))
        actual = _image(dataset, parent).crop((g, g, g + size, g + size))
        difference = ImageStat.Stat(Image.frombytes(
            "RGB", actual.size,
            bytes(abs(a - b) for a, b in zip(actual.tobytes(), expected.tobytes()))))
        if max(difference.mean) > 8.0:
            raise ValueError(f"{parent.key}: child imagery orientation disagrees")
    return tiles


def validate(args: argparse.Namespace) -> None:
    tiles = validate_dataset(Path(args.dataset))
    print(f"validated {len(tiles)} tiles in {args.dataset}")


def inspect(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    tiles = validate_dataset(dataset)
    level = args.level if args.level is not None else max(tile.key.level for tile in tiles)
    selected = [tile for tile in tiles if tile.key.level == level]
    if not selected:
        raise ValueError(f"level {level} is not present")
    count = 1 << level
    card = args.card_size
    atlas = Image.new("RGB", (count * card * 2, count * card), "#20242a")
    draw = ImageDraw.Draw(atlas)
    for tile in selected:
        g = tile.gutter
        imagery = _image(dataset, tile)
        imagery = imagery.resize((card, card), Image.Resampling.NEAREST)
        x0, y0 = tile.key.x * card * 2, tile.key.y * card
        atlas.paste(imagery, (x0, y0))
        decoded = tile.decoded_heights()
        grey = bytearray()
        for index, height in enumerate(decoded):
            if not tile.is_valid(index):
                grey.append(255 if (index // tile.width + index % tile.width) & 1 else 0)
            elif tile.height_range:
                grey.append(round(255 * (height - tile.minimum) / tile.height_range))
            else:
                grey.append(127)
        height_image = Image.frombytes("L", (tile.width, tile.height), bytes(grey)).convert("RGB")
        height_image = height_image.resize((card, card), Image.Resampling.NEAREST)
        atlas.paste(height_image, (x0 + card, y0))
        gutter_px = max(1, round(card * g / tile.width))
        for offset in (0, card):
            draw.rectangle((x0 + offset + gutter_px, y0 + gutter_px,
                            x0 + offset + card - gutter_px - 1,
                            y0 + card - gutter_px - 1), outline="#ff3b30")
        label = (f"{level}/{tile.key.x}/{tile.key.y}  "
                 f"{tile.minimum:.1f}..{tile.minimum + tile.height_range:.1f}m  "
                 f"valid {tile.valid_count}/{tile.width * tile.height}")
        draw.rectangle((x0, y0, min(atlas.width, x0 + len(label) * 6 + 4), y0 + 12),
                       fill="#000000")
        draw.text((x0 + 2, y0 + 1), label, fill="white")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(output, "PNG")
    print(f"wrote level {level} inspection atlas to {output}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="build a complete XYZ pyramid")
    build_parser.add_argument("--dem", required=True)
    build_parser.add_argument("--imagery", required=True)
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--extent", type=float, nargs=4, required=True,
                              metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    build_parser.add_argument("--profile", required=True,
                              help="documented CRS/profile shared by both inputs")
    build_parser.add_argument("--levels", type=int, default=3)
    build_parser.add_argument("--samples", type=int, default=65,
                              help="height samples per tile excluding gutters")
    build_parser.add_argument("--imagery-size", type=int, default=256)
    build_parser.add_argument("--gutter", type=int, default=1)
    build_parser.add_argument("--height-encoding", choices=("f32", "r16"), default="f32")
    build_parser.add_argument("--nodata", type=float)
    build_parser.add_argument("--source", default="unspecified")
    build_parser.add_argument("--source-version", default="unspecified")
    build_parser.add_argument(
        "--root-geometric-error", type=float,
        help="root error in metres; default assumes profile extent units are metres")
    build_parser.set_defaults(function=build)
    validate_parser = commands.add_parser("validate", help="validate a complete dataset")
    validate_parser.add_argument("dataset")
    validate_parser.set_defaults(function=validate)
    inspect_parser = commands.add_parser("inspect", help="write a labelled debug atlas")
    inspect_parser.add_argument("dataset")
    inspect_parser.add_argument("--output", required=True)
    inspect_parser.add_argument("--level", type=int)
    inspect_parser.add_argument("--card-size", type=int, default=192)
    inspect_parser.set_defaults(function=inspect)
    return result


def main() -> int:
    try:
        args = parser().parse_args()
        args.function(args)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"terrain_tiles: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
