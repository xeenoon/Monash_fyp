#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runtime = root / "runtime"
        banks = (("rock", 9), ("grass", 6))
        manifests = {}
        expected_ids = []
        for bank, count in banks:
            source = root / bank
            materials = []
            for index in range(count):
                material_id = f"{bank}_{index}"
                expected_ids.append(material_id)
                materials.append({"id": material_id, "width_m": 1.5 + index})
                target = source / material_id
                target.mkdir(parents=True)
                y, x = np.mgrid[:32, :32]
                albedo = np.stack((70 + x * 3, 90 + y * 2,
                                   80 + ((x + y + index) % 8) * 8), axis=-1).astype(np.uint8)
                Image.fromarray(albedo, "RGB").save(target / f"{material_id}_albedo.png")
                Image.new("RGB", (32, 32), (128, 128, 255)).save(target / "normal_gl.png")
                Image.new("L", (32, 32), 150 + index).save(target / "roughness.png")
                Image.fromarray((x * 8).astype(np.uint8), "L").save(target / "height.png")
                Image.new("L", (32, 32), 220 - index).save(target / "_ao.png")
            manifest = root / f"{bank}.json"
            manifest.write_text(json.dumps({"materials": materials}), encoding="utf-8")
            manifests[bank] = (manifest, source)
        subprocess.run([sys.executable, str(args.tool),
                        "--rock-manifest", str(manifests["rock"][0]),
                        "--rock-source", str(manifests["rock"][1]),
                        "--grass-manifest", str(manifests["grass"][0]),
                        "--grass-source", str(manifests["grass"][1]),
                        "--output", str(runtime),
                        "--cell-size", "16", "--gutter", "2"], check=True)
        for name, mode in (("terrain_micro_albedo.png", "RGB"),
                           ("terrain_micro_normal.png", "RGB"),
                           ("terrain_micro_ormh.png", "RGBA")):
            with Image.open(runtime / name) as image:
                assert image.size == (80, 80)
                assert image.mode == mode
        metadata = json.loads((runtime / "atlas.json").read_text())
        assert [entry["id"] for entry in metadata["materials"]] == expected_ids
        assert metadata["banks"] == {
            "rock": {"first": 0, "count": 9},
            "grass": {"first": 9, "count": 6},
        }
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
