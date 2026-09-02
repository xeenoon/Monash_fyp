#!/usr/bin/env python3
"""Convert the pinned, position-only wall torch OBJ to an embedded glTF."""

import argparse
import base64
import json
import struct
from pathlib import Path


def parse_obj(path: Path):
    positions = []
    indices = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if not fields:
            continue
        if fields[0] == "v":
            positions.append(tuple(float(value) for value in fields[1:4]))
        elif fields[0] == "f":
            face = [int(value.split("/", 1)[0]) - 1 for value in fields[1:]]
            for corner in range(1, len(face) - 1):
                indices.extend((face[0], face[corner], face[corner + 1]))
    if not positions or not indices or max(indices) >= len(positions):
        raise ValueError("OBJ has incomplete or invalid geometry")
    return positions, indices


def convert(source: Path, destination: Path):
    positions, indices = parse_obj(source)
    position_bytes = b"".join(struct.pack("<3f", *position) for position in positions)
    index_bytes = b"".join(struct.pack("<H", index) for index in indices)
    payload = position_bytes + index_bytes
    mins = [min(position[axis] for position in positions) for axis in range(3)]
    maxs = [max(position[axis] for position in positions) for axis in range(3)]
    document = {
        "asset": {"version": "2.0", "generator": "convert_wall_torch.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Wall torch"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1,
                                       "material": 0}]}],
        "materials": [{"name": "Dark iron and wood", "pbrMetallicRoughness": {
            "baseColorFactor": [0.16, 0.07, 0.025, 1.0],
            "metallicFactor": 0.3, "roughnessFactor": 0.72}}],
        "buffers": [{"byteLength": len(payload), "uri":
                     "data:application/octet-stream;base64," +
                     base64.b64encode(payload).decode("ascii")}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes),
             "target": 34962},
            {"buffer": 0, "byteOffset": len(position_bytes), "byteLength": len(index_bytes),
             "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions),
             "type": "VEC3", "min": mins, "max": maxs},
            {"bufferView": 1, "componentType": 5123, "count": len(indices),
             "type": "SCALAR", "min": [min(indices)], "max": [max(indices)]},
        ],
    }
    destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)


if __name__ == "__main__":
    main()
