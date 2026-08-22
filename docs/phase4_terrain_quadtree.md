# Phase 4: global terrain quadtree

The runtime now treats the Phase 3 dataset as a paged quadtree instead of
loading only `0/0/0.trn`. `TerrainRuntime` owns the Vulkan-facing resources;
`TerrainQuadtree` owns only keys, state, selection, budgets, and opaque payload
pointers. This keeps the selection and lifecycle rules testable without a GPU.

## Frame workflow

1. Finish any upload tickets whose fences have signalled.
2. Frustum-test resident nodes and calculate screen-space geometric error.
3. Request a node's four children when it crosses the split threshold.
4. Keep drawing the parent until the entire child quad is GPU-resident.
5. Load and upload only the configured number of tiles for this frame.
6. Build a multi-tile draw list using one shared normalized grid.
7. Mark expired sibling quads for eviction.
8. After frame submission has waited for the preceding fence, destroy resources
   that are absent from the newly submitted draw list.

Missing files become `unavailable`, which naturally identifies the leaves of a
dataset without maintaining a second manifest parser in the renderer. A
generation counter rejects completions from cancelled or superseded requests.

## LOD and fallback

Screen-space error is:

```text
focal_px = 0.5 * viewport_height / tan(0.5 * vertical_fov)
error_px = geometric_error_m * focal_px / max(range_to_bound, 1 metre)
```

The default split/merge thresholds are 3 px and 2 px. Their gap supplies
hysteresis. Replacement is atomic at sibling-quad granularity: four children
replace one parent, never an arbitrary parent/child mixture. Consequently slow
or absent child data always leaves valid parent terrain on screen.

Each child also stores its general parent UV scale/bias. The current atomic
fallback draws parent geometry directly; the mapping is retained for later
shader-side inherited imagery/elevation without changing the pager contract.

## Geometry and culling

All projected tiles share one `65x65` normalized grid (or whatever interior
sample size the root declares). Per-tile elevation is an R16_UNORM texture and
the vertex shader applies horizontal span, elevation range, camera-relative
transform, and skirt depth through 128 bytes of push constants. Imagery and
height gutters are excluded with texel-centre scale/bias values.

CPU culling uses all eight elevation/skirt-aware world-space corners against
the five infinite-perspective frustum planes. Horizon culling is intentionally
disabled for the current projected EPSG:2056 dataset; it becomes valid only
after the renderer uses ECEF bounds and camera coordinates.

Press F7 to colour tiles by LOD. Runtime logging reports known, resident, and
drawn tile counts plus accounted CPU/GPU memory.

## Rocky provenance

Phase 4 is a plain-C adaptation of Rocky's terrain architecture. Each adapted
function also contains an adjacent source comment.

| Local code | Rocky source | Reason for adaptation |
|---|---|---|
| `terrain_tile_key_child`, `terrain_tile_key_quadrant` | `src/rocky/TileKey.cpp` | Preserve Rocky's quadtree addressing without its Profile/C++ dependencies. |
| `terrain_tile_parent_uv_scale_bias` | `TerrainTileNode.cpp::scaleBias` | Store the same fallback concept in four shader-ready floats; flip to this dataset's top-left image convention. |
| `select_node` | `TerrainTileNode::accept` | Preserve all-four-children replacement and parent fallback without VSG traversal. |
| `tile_visible` | `SurfaceNode::isVisible` | Preserve the tight eight-corner test using the renderer's camera basis. |
| `terrain_quadtree_schedule_evictions` | `TerrainTilePager::update` | Preserve sibling-quad lifetime while replacing VSG's tracker with explicit budgets/frame age. |
| `terrain_grid_create` | `GeometryPool::createIndices` and skirt macros | Reuse Rocky's pooled surface/skirt topology with shader extrusion markers. |
| `load_node`, `upload_node` split | `TerrainTilePager::requestLoadData` / `requestMergeData` | Keep immutable CPU loading separate from owner-controlled GPU publication. |

Rocky is MIT licensed. Its notice is preserved in
[`third_party_notices.md`](third_party_notices.md).
