#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mesh.h"

/* "TRN1" when read as a little-endian uint32_t. The file encoding is always
   little-endian; it is not a dump of TerrainTileHeader's native C layout. */
#define TERRAIN_TILE_MAGIC        UINT32_C(0x314e5254)
#define TERRAIN_TILE_VERSION      UINT16_C(1)
#define TERRAIN_TILE_HEADER_BYTES UINT32_C(224)
#define TERRAIN_TILE_ROOT_KEY     UINT32_MAX

typedef enum {
    TERRAIN_TILE_HEIGHT_R16_UNORM = 1,
    TERRAIN_TILE_HEIGHT_F32       = 2,
} TerrainTileHeightEncoding;

enum {
    TERRAIN_TILE_HAS_PARENT  = 1u << 0,
    TERRAIN_TILE_HAS_IMAGERY = 1u << 1,
    TERRAIN_TILE_HAS_NODATA  = 1u << 2,
};

typedef struct {
    uint32_t level, x, y;             /* XYZ: y=0 is the profile's north edge */
    uint32_t parent_level, parent_x, parent_y;

    uint16_t sample_width;            /* Includes the height gutter. */
    uint16_t sample_height;
    uint16_t gutter;
    uint16_t height_encoding;

    float min_height_m;               /* R16 decode offset; valid minimum. */
    float height_range_m;             /* R16 decode scale; max = min + range. */
    float geometric_error_m;
    uint32_t valid_sample_count;

    /* Extent in the named profile: west, south, east, north. The profile may
       be geographic or projected, so these are deliberately doubles. */
    double extent[4];

    LocalToWorldTransform local_to_world;

    uint32_t flags;
    uint32_t height_bytes;
    uint32_t validity_bytes;           /* One bit/sample; 1 means valid. */
    uint32_t imagery_bytes;            /* Zero when imagery_uri is external. */
    uint32_t profile_bytes;
    uint32_t source_bytes;
    uint32_t imagery_uri_bytes;
    uint32_t payload_crc32;
} TerrainTileHeader;

/* TerrainTile retains the renderer's C inheritance convention: Mesh is first,
   so TerrainTile* may be cast to Mesh*. Offline raster data and metadata follow
   it and can later feed quadtree-specific mesh generation/displacement. */
typedef struct {
    Mesh base;
    TerrainTileHeader header;

    void *height_data;
    uint8_t *validity;
    uint8_t *imagery;
    char *profile;
    char *source;
    char *imagery_uri;

    bool owns_offline_data;
} TerrainTile;

_Static_assert(offsetof(TerrainTile, base) == 0,
               "TerrainTile must retain Mesh inheritance");

typedef enum {
    TERRAIN_TILE_OK = 0,
    TERRAIN_TILE_INVALID_ARGUMENT,
    TERRAIN_TILE_IO_ERROR,
    TERRAIN_TILE_TRUNCATED,
    TERRAIN_TILE_BAD_MAGIC,
    TERRAIN_TILE_UNSUPPORTED_VERSION,
    TERRAIN_TILE_INVALID_FORMAT,
    TERRAIN_TILE_CHECKSUM_MISMATCH,
    TERRAIN_TILE_OUT_OF_MEMORY,
} TerrainTileResult;

/* Loads and validates a .trn asset without GDAL/PROJ or GPU work. On success,
   base.local_to_world is ready and base.texture_path contains the serialized
   dataset-root-relative imagery URI. The pager must resolve that URI before
   upload. Mesh geometry remains the responsibility of terrain generation.
   `out` need not be initialised, but must not already own a loaded tile. */
TerrainTileResult terrain_tile_load(const char *path, TerrainTile *out);
void terrain_tile_unload(TerrainTile *tile);

bool terrain_tile_sample_valid(const TerrainTile *tile, uint32_t x, uint32_t y);
float terrain_tile_height(const TerrainTile *tile, uint32_t x, uint32_t y);
const char *terrain_tile_result_string(TerrainTileResult result);
