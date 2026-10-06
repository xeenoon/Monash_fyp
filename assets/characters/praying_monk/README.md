# Praying monk

Seated NPC design matching the game's muted brown cloth and rounded 3D characters.
A deep hood hides the face completely; a long silver beard reaches the floor.
The legs are crossed and the hands meet in prayer. The pose is static.

- `praying_monk.blend`: editable cloak, hood, hands, folded legs, individual beard locks, and preview studio.
- `praying_monk.glb`: portable model with embedded cloth normal map.
- `praying_monk.gltf`, `praying_monk.bin`, `cloth_normal.png`: separate-file export for the game's loader.
- `previews/`: front, three-quarter, rear, and elevated game-angle renders of the actual geometry.
- `asset_report.json`: source mesh counts and dimensions.

The export has one mesh with 12 material primitives. Coordinates are metres,
Y up, facing +Z, with the lowest geometry at ground level. This is a detailed
static asset, without a rig, animations, collision mesh, or LODs. The game
places him in a reserved corner of each generated starting room, with a
runtime collision radius and a paginated conversation opened with E. Preview
images use studio lighting rather than the game's torch lighting.

The materials reuse the project's existing cloth normal map. Other colors
and roughness values are neutral material parameters; lighting is not baked.

Rebuild from the repository root:

```sh
blender --background --threads 6 --python tools/build_praying_monk.py
```

The source and glTF exports were checked for finite geometry, normalized
normals, valid indices, texture references, and preserved bounds after GLB
re-import. These checks do not establish runtime performance or a collision
free cloth simulation.

The runtime glTF, buffer, and texture are checked in explicitly. The larger
editable Blender file and studio previews can be regenerated with the script.
