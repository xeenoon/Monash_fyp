# Medusa guardian

The game's guardian is a petrified oracle growing out of five serpent coils,
with eleven fanged faces in a jade/violet crown, two temple tendrils, slit eyes
and bronze ribs. It faces local +Z, stands on Y=0 and is authored in metres.

`python3 tools/build_medusa_guardian.py` rebuilds the checked-in glTF, binary
and six 1024px PBR maps using NumPy and Pillow. There are no downloaded assets.
Tentacles use Catmull-Rom 3D centreline curves with parallel-transport frames,
tapered ring extrusion, closed caps, smooth normals and arc-length UVs.
The sixteen-second `Serpentine` idle clip gives every head independent searching
turns and staggered, violent double strikes: 42 cm neck extensions in 80–100 ms,
wide jaw opening, 45 ms jaw slams, recoil and rapid tongue flicks. Neck extrusion
scales from its fixed root, while reciprocal head scaling keeps faces in proportion.
The head hierarchy stays attached throughout. Keys are baked at 60 Hz to preserve
the sharp timing. Two temple tendrils sway gently between the striking heads.
These are visual hisses; this asset does not add a sound effect. Albedo, tangent normals,
occlusion and roughness use the same PBR pipeline as the dungeon and player.
Amber eyes turn red during a chase. Gameplay and collision remain unchanged.

Inspect it under the game's torch lighting with:

```sh
TERRAIN_SCENE=torch_lab2 DUNGEON_GUARDIAN_STUDY=1 \
TERRAIN_LAB_POS="3.3 1.8 5.4" TERRAIN_LAB_YAW=265 TERRAIN_LAB_PITCH=-9 \
./build-release/terrain_renderer
```

WASD and the lab's free camera work as usual. For a separate studio render:
`blender -b --python tools/render_medusa_preview.py -- /tmp/medusa-preview.png`.
Run `python3 tests/medusa_asset_tests.py` to check geometry, winding, maps,
draw budget and animation loops.
