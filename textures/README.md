# Terrain micro materials

This library contains the nine CC0 rock materials listed in
[`manifest.json`](manifest.json). The large original packages are deliberately
ignored by Git under `source/`; fetch both banks and build the compact runtime atlas:

```sh
python3 tools/download_terrain_materials.py
python3 tools/download_terrain_materials.py \
  --manifest textures/grass_manifest.json \
  --source textures/grass-source
python3 tools/build_terrain_micro_atlas.py
```

Poly Haven source maps are fetched at 4K, ambientCG's public JPG bundles at 8K,
and PolyScan at 4K because its advertised 8K download endpoints currently
return 404 while the 4K packages are public. Generated runtime maps use all
nine rocks followed by all six grass materials in a guttered 4 x 4 atlas:

- `runtime/terrain_micro_albedo.png`: neutral high-pass luminance only;
- `runtime/terrain_micro_normal.png`: OpenGL tangent normals;
- `runtime/terrain_micro_ormh.png`: AO, roughness, zero metallic, height.

At runtime the shader samples the classified bank once at authored scale. The
golden terrain owns 80% of the result; scan luminance, normal, roughness, and AO
are each capped at a 20% contribution. Scan albedo remains neutral, so golden
RGB continues to own colour.

All source licenses are CC0. Provider/page provenance remains in the manifest.

## Alpine grass candidates

[`grass_manifest.json`](grass_manifest.json) records six CC0 grass/vegetated
ground candidates selected against the checked-in Swiss source's labelled grass
palette. That source has median sRGB `(69, 79, 56)` and an inter-percentile
range from roughly `(47, 56, 41)` to `(93, 100, 79)`: dark, desaturated and
olive-grey rather than lawn green.

Fetch the selected 4K PBR sets, then rebuild the combined runtime atlas:

```sh
python3 tools/download_terrain_materials.py \
  --manifest textures/grass_manifest.json \
  --source textures/grass-source
```

The three ambientCG grass sets bracket the dark green end of the Alpine swatch.
Sparse Grass adds soil/root breakup; Grass Path 3 and Forest Ground 01 cover the
drier, less saturated vegetation visible in the orthophoto. The full originals
are ignored by Git under `grass-source/`.
