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

The moss layer uses the user-provided `runtime/moss.jpeg`, copied unchanged
from Downloads. Its checksum and provenance are in `local_materials`; it is
separate from the CC0 Poly Haven set and the downloader preserves it.
The sparse underlayer uses a two-metre scale with rotated sample blending.
Bent leaf geometry adds volume on floors and lower walls, using random JPEG
crops, geometric normals and two-sided foliage lighting. Both stay below
the roof cap. No normal or opacity scan is implied by the supplied JPEG.
