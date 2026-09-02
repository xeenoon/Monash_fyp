# Dungeon materials

The example dungeon uses three photo-scanned Poly Haven materials. All are
CC0-1.0 and retain their source page, physical width, selected channels, and
checksums in `manifest.json`.

Reproduce the checked-in runtime files with:

```sh
python3 tools/download_dungeon_materials.py
```

The static mesh shader consumes the files without repacking: `albedo` is sRGB,
`normal` is the OpenGL tangent convention, and `orm` is Poly Haven's ARM image
(red ambient occlusion, green roughness, blue metallic). Displacement is not
downloaded because the dungeon renderer does not currently consume it.
