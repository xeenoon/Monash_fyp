# Dungeon torch shading

Wall vertex normals point into open floor, opposite the solid-side displacement
direction. Tangent handedness follows that normal. Material normals stay in
the geometric hemisphere; they are never flipped toward the camera. Flipping
them based on the view previously produced sharp black patches on walls.

`dungeon_shadow.c` builds a preorder BVH from rendered wall triangles, including
bulges and plateau caps, plus transformed closed-door triangles. A storage
buffer at set 0 binding 6 carries the nodes. Misses skip to the node's subtree
end, so traversal needs no fixed stack or 64-blocker limit. Geometry is uploaded
on level creation and when the closed-door mask changes; camera movement only
updates the camera-to-dungeon translation. The dungeon uses small local world
coordinates, independent of the terrain's large-world coordinates.

As with collision, a door stops blocking when it unlocks; its brief sinking
animation is cosmetic. The GPU buffer is replaced only after outstanding
readers complete. Mounted torch heads are placed along a verified ray to the
actual wall mesh rather than the simplified collision boundary.

Mesh, wall, moss, puddle glints, and puddle reflections share the ray test in
`dungeon_shadow.glsl`. Rays use a 3 mm surface offset and stop 3 mm before the
light. Puddle alpha is radial coverage times Fresnel, so the existing lit floor
is transmitted rather than overwritten by opaque near-black colour.

The `point_shadow` dump record identifies the selected light (the player by
default), blocking BVH leaf, hit distance, visibility, unshadowed contribution,
and surface/light positions. A leaf of -1 means unobstructed. The diagnostic
light index occupies `point_light_options.w`. Records are emitted for contributing
lights only; use the surface normal record to diagnose a zero cosine term.

`dungeon_triangle_shadows` checks finite-height rays, both triangle sides,
more than 64 walls, and exact hits on the generated mesh. The mesh regression
also checks that wall normals point toward open floor.
