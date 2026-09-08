#!/usr/bin/env python3
"""Render a dungeon_cave_dump.c binary blob as a top-down PNG.

Usage: python3 tools/preview_cave.py cave_dump.bin out.png

The blob format (little-endian, written by tools/dungeon_cave_dump.c):
  uint32 width, uint32 height, float cell_size, float origin_x, float origin_z,
  float values[width*height],
  float spawn_x, float spawn_z, float exit_x, float exit_z,
  uint32 puddle_count, then per puddle: float center_x, center_z, radius,
  uint32 door_count, then per door: float ax, az, bx, bz, uint32 lock_kind.

Field values render as grayscale (white = open floor, black = rock); spawn is
a green dot, exit a blue dot, puddles translucent cyan discs, and each locked
doorway the blocker segment it seals (amber = pin tumbler, magenta = vault
dial). This is how layouts get inspected -- never by screenshotting the
running game.
"""
import argparse
import struct

import numpy as np
from PIL import Image, ImageDraw


def load(path):
    with open(path, "rb") as f:
        width, height = struct.unpack("<II", f.read(8))
        cell_size, origin_x, origin_z = struct.unpack("<fff", f.read(12))
        values = np.frombuffer(f.read(4 * width * height), dtype="<f4").reshape(height, width)
        spawn_x, spawn_z, exit_x, exit_z = struct.unpack("<ffff", f.read(16))
        (puddle_count,) = struct.unpack("<I", f.read(4))
        puddles = [struct.unpack("<fff", f.read(12)) for _ in range(puddle_count)]
        (door_count,) = struct.unpack("<I", f.read(4))
        doors = [struct.unpack("<ffffI", f.read(20)) for _ in range(door_count)]
    return {
        "width": width, "height": height, "cell_size": cell_size,
        "origin": (origin_x, origin_z), "values": values,
        "spawn": (spawn_x, spawn_z), "exit": (exit_x, exit_z), "puddles": puddles,
        "doors": doors,
    }


def world_to_pixel(cave, x, z):
    ox, oz = cave["origin"]
    return (x - ox) / cave["cell_size"], (z - oz) / cave["cell_size"]


def render(cave, scale=4):
    values = np.clip(cave["values"], 0.0, 1.0)
    gray = (values * 255.0).astype("uint8")
    image = Image.fromarray(gray, mode="L").convert("RGB")
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    draw = ImageDraw.Draw(image, "RGBA")

    def dot(world_xz, color, radius_px=5):
        px, pz = world_to_pixel(cave, *world_xz)
        px, pz = px * scale, pz * scale
        draw.ellipse([px - radius_px, pz - radius_px, px + radius_px, pz + radius_px], fill=color)

    for cx, cz, radius in cave["puddles"]:
        px, pz = world_to_pixel(cave, cx, cz)
        px, pz = px * scale, pz * scale
        r = radius / cave["cell_size"] * scale
        draw.ellipse([px - r, pz - r, px + r, pz + r], fill=(60, 160, 220, 140))
    lock_colors = {1: (240, 176, 60, 255), 2: (222, 96, 220, 255)}
    for ax, az, bx, bz, lock in cave["doors"]:
        pax, paz = world_to_pixel(cave, ax, az)
        pbx, pbz = world_to_pixel(cave, bx, bz)
        draw.line([pax * scale, paz * scale, pbx * scale, pbz * scale],
                  fill=lock_colors.get(lock, (200, 200, 200, 255)), width=max(2, scale))
    dot(cave["spawn"], (60, 220, 90, 255), radius_px=6)
    dot(cave["exit"], (70, 120, 255, 255), radius_px=6)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", help="binary dump from dungeon_cave_dump")
    parser.add_argument("out", help="output PNG path")
    parser.add_argument("--scale", type=int, default=4, help="pixels per field cell")
    args = parser.parse_args()
    cave = load(args.dump)
    render(cave, scale=args.scale).save(args.out)
    print(f"wrote {args.out} ({cave['width']}x{cave['height']} corners, "
          f"{len(cave['puddles'])} puddles, {len(cave['doors'])} doors)")


if __name__ == "__main__":
    main()
