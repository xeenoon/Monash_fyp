#!/usr/bin/env python3
"""Checks the baked padlock against what the engine assumes about it.

The runtime asset is generated (tools/export_padlock_runtime.py) rather than
checked in, so a fresh clone does not have it and this SKIPS -- the same
condition under which dungeon_scene falls back to the generated mechanism. What
it must never do is pass quietly on an asset that is present but wrong: every
assertion here is something src/dungeon_scene.c or src/gltf_scene.c would
otherwise discover at run time, on a frame, in front of a player.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

CLIPS = ("Pick_Insert", "Pick_Withdraw", "Pick_Jiggle_Loop", "Pin_01_Pop", "Pin_02_Pop",
         "Pin_03_Pop", "Pin_04_Pop", "Unlock_All_Four_Set")
PINS = ("PIN_01", "PIN_02", "PIN_03", "PIN_04")
# DUNGEON_PADLOCK_MAX_PRIMITIVES in src/dungeon_scene.h: the per-door draw
# reservation. More primitives than this and the lock would be drawn truncated.
MAX_PRIMITIVES = 32
# The inspection window, in metres, from tools/export_padlock_runtime.py -- in
# the EXPORTED frame, where the exporter's source (x, y, z) became (x, z, -y).
# Nothing may be left inside it: that space is the sightline to the pins.
UNIT = 0.022
WINDOW_MIN = (-1.20 * UNIT, 0.60 * UNIT, 0.40 * UNIT)
WINDOW_MAX = (1.20 * UNIT, 2.20 * UNIT, 3.00 * UNIT)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def node_matrix(node: dict) -> list[list[float]]:
    tx, ty, tz = node.get("translation", (0.0, 0.0, 0.0))
    x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    sx, sy, sz = node.get("scale", (1.0, 1.0, 1.0))
    rotation = [
        [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
        [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
        [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)],
    ]
    scale = (sx, sy, sz)
    return [[rotation[c][r] * scale[c] for r in range(3)] for c in range(3)] + [[tx, ty, tz]]


def multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    out = [[sum(a[k][r] * b[c][k] for k in range(3)) for r in range(3)] for c in range(3)]
    out.append([sum(a[k][r] * b[3][k] for k in range(3)) + a[3][r] for r in range(3)])
    return out


def world_matrices(gltf: dict) -> list[list[list[float]]]:
    parent = {}
    for index, node in enumerate(gltf["nodes"]):
        for child in node.get("children", []):
            parent[child] = index
    world: dict[int, list[list[float]]] = {}

    def resolve(index: int):
        if index not in world:
            local = node_matrix(gltf["nodes"][index])
            world[index] = multiply(resolve(parent[index]), local) if index in parent else local
        return world[index]

    return [resolve(i) for i in range(len(gltf["nodes"]))]


def read_floats(gltf: dict, buffer: bytes, accessor_index: int) -> list[list[float]]:
    accessor = gltf["accessors"][accessor_index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    return [list(struct.unpack_from(f"<{components}f", buffer, start + i * components * 4))
            for i in range(accessor["count"])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    gltf_path = args.runtime / "padlock.gltf"
    if not gltf_path.is_file():
        print(f"padlock asset tests skipped: no {gltf_path} "
              "(run tools/export_padlock_runtime.py)")
        return 0
    manifest = json.loads((args.runtime / "manifest.json").read_text(encoding="utf-8"))
    for name, entry in manifest["files"].items():
        path = args.runtime / name
        assert path.is_file(), path
        assert path.stat().st_size == entry["bytes"], path
        assert md5(path) == entry["md5"], f"{path} does not match the manifest"

    gltf = json.loads(gltf_path.read_text(encoding="utf-8"))
    buffer = (args.runtime / gltf["buffers"][0]["uri"]).read_bytes()

    # The importer refuses all three of these, so an asset carrying one would
    # take the whole scene down at load rather than degrade.
    assert "skins" not in gltf, "the runtime importer has no skinning"
    assert not any("skin" in node for node in gltf["nodes"])
    for mesh in gltf["meshes"]:
        for primitive in mesh["primitives"]:
            assert "targets" not in primitive, "morph targets are phase 2"
            assert primitive.get("mode", 4) == 4
            assert "TEXCOORD_0" in primitive["attributes"], "the atlas would not map"

    # One material, one atlas: the mechanism is 19 draws, and a per-part
    # material set would be a descriptor set and a pipeline bind per part.
    assert len(gltf["materials"]) == 1, gltf["materials"]
    material = gltf["materials"][0]
    assert "baseColorTexture" in material["pbrMetallicRoughness"]
    assert "metallicRoughnessTexture" in material["pbrMetallicRoughness"]
    assert "normalTexture" in material
    for image in gltf["images"]:
        assert (args.runtime / image["uri"]).is_file(), image["uri"]

    primitives = sum(len(mesh["primitives"]) for mesh in gltf["meshes"])
    assert primitives <= MAX_PRIMITIVES, \
        f"{primitives} primitives, over the {MAX_PRIMITIVES} reserved per door"

    names = [node.get("name", "") for node in gltf["nodes"]]
    for required in PINS + ("PICK", "ROOT", "SHACKLE", "CORE"):
        assert required in names, f"the engine looks up {required} by name"

    clips = {animation["name"]: animation for animation in gltf["animations"]}
    assert set(CLIPS) <= set(clips), sorted(set(CLIPS) - set(clips))
    for name, animation in clips.items():
        for sampler in animation["samplers"]:
            assert sampler.get("interpolation", "LINEAR") in ("LINEAR", "STEP"), name

    # Layering is the contract the whole presentation rests on: a pin pops while
    # the pick keeps working, so a pin's clip must touch that pin's stack and
    # NOTHING else. Blender bakes every bone into every action; the exporter
    # strips the channels that never leave rest, and this is what proves it.
    for pin_index, pin in enumerate(PINS, start=1):
        animated = {names[channel["target"]["node"]]
                    for channel in clips[f"Pin_0{pin_index}_Pop"]["channels"]}
        assert animated <= {pin, f"PIN_DRIVER_0{pin_index}", f"PIN_SPRING_0{pin_index}"}, \
            f"Pin_0{pin_index}_Pop also moves {sorted(animated)}"
        assert pin in animated
    # The pick's clips touch the pick and nothing else. The idle LOOP especially:
    # it runs the whole time a lock is being worked, and the pin stacks are not
    # its to move -- where a pin sits is the puzzle's answer, and a clip nudging
    # it is a permanent wobble over a readout measured in millimetres.
    for clip in ("Pick_Insert", "Pick_Withdraw", "Pick_Jiggle_Loop"):
        animated = {names[channel["target"]["node"]] for channel in clips[clip]["channels"]}
        assert animated == {"PICK"}, f"{clip} moves {sorted(animated)}"

    # Every pin must MOVE in its own pop clip: dungeon_scene reads the travel
    # off the clip and shows an unset pin as a fraction of it.
    for pin_index, pin in enumerate(PINS, start=1):
        channel = next(c for c in clips[f"Pin_0{pin_index}_Pop"]["channels"]
                       if names[c["target"]["node"]] == pin and c["target"]["path"] == "translation")
        values = read_floats(gltf, buffer,
                             clips[f"Pin_0{pin_index}_Pop"]["samplers"][channel["sampler"]]["output"])
        travel = max(abs(b - a) for a, b in zip(values[0], values[-1]))
        assert travel > 1e-4, f"{pin} does not move in its own pop clip"

    world = world_matrices(gltf)
    pin_positions = [world[names.index(pin)][3] for pin in PINS]
    # The pins are spaced along the model's +Z, PIN_01 furthest along it. That
    # is what padlock_placement maps to the camera's LEFT, so pin 0 -- the
    # puzzle's first pin -- is the leftmost on screen.
    depths = [position[2] for position in pin_positions]
    assert depths == sorted(depths, reverse=True), f"the pin row is out of order: {depths}"
    for position in pin_positions:
        assert abs(position[1] - pin_positions[0][1]) < 1e-4, "the pins are not on one line"

    # The window has to be empty: that space is the sightline to the pins, and
    # the export is the only thing that can put something back in it. The
    # exporter ray-casts as well; this is the cheap check that the file on disk
    # is the one that passed.
    trespass = []
    for index, node in enumerate(gltf["nodes"]):
        if "mesh" not in node or "PICK" in names[index]:
            continue  # the pick tunnels down the keyway through the window
        matrix = world[index]
        for primitive in gltf["meshes"][node["mesh"]]["primitives"]:
            accessor = gltf["accessors"][primitive["attributes"]["POSITION"]]
            low = [1e9] * 3
            high = [-1e9] * 3
            for corner in ((x, y, z) for x in (accessor["min"][0], accessor["max"][0])
                           for y in (accessor["min"][1], accessor["max"][1])
                           for z in (accessor["min"][2], accessor["max"][2])):
                for axis in range(3):
                    value = sum(matrix[c][axis] * corner[c] for c in range(3)) + matrix[3][axis]
                    low[axis] = min(low[axis], value)
                    high[axis] = max(high[axis], value)
            # A bounding box overlapping the window is not proof of trespass --
            # a wall around the hole overlaps it too -- so only a box whose
            # CENTRE sits inside counts, which catches a part left whole.
            centre = [(low[a] + high[a]) * 0.5 for a in range(3)]
            if all(WINDOW_MIN[a] < centre[a] < WINDOW_MAX[a] for a in range(3)):
                trespass.append(names[index])
    assert not trespass, f"left standing in the inspection window: {sorted(set(trespass))}"

    print(f"padlock asset tests passed ({primitives} primitives, {len(clips)} clips, "
          f"{manifest['geometry']['triangles']} triangles)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
