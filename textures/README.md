# Terrain micro materials

This library contains the nine CC0 rock materials listed in
[`manifest.json`](manifest.json). The large original packages are deliberately
ignored by Git under `source/`; fetch all three banks and build the compact runtime atlas:

```sh
python3 tools/download_terrain_materials.py
python3 tools/download_terrain_materials.py \
  --manifest textures/grass_manifest.json \
  --source textures/grass-source
python3 tools/download_terrain_materials.py \
  --manifest textures/snow_manifest.json \
  --source textures/snow-source
python3 tools/build_terrain_micro_atlas.py
```

Poly Haven source maps are fetched at 4K, ambientCG's public JPG bundles at 8K,
and PolyScan at 4K because its advertised 8K download endpoints currently
return 404 while the 4K packages are public. Generated runtime maps use all
nine rocks, six grass materials, and six snow materials in a guttered 5 x 5 atlas:

- `runtime/terrain_micro_albedo.png`: linear material luminance for the coarse layer;
- `runtime/terrain_micro_normal.png`: OpenGL tangent normals;
- `runtime/terrain_micro_ormh.png`: AO, roughness, fine high-pass luminance, height.

At runtime the shader samples the classified bank at authored scale and again
at a coarse material-macro scale. The coarse layer contributes normalized,
achromatic structure and a restrained normal contribution, while roughness and
AO remain authored-scale. Terrain imagery remains the sole RGB source, so the
rock scans cannot tint cliffs brown/red. Fine scan luminance also remains neutral.

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
