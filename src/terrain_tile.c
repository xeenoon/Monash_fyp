#include "terrain_tile.h"

#include "byte_utils.h"
#include "checksum_utils.h"
#include "file_utils.h"
#include "size_utils.h"
#include "str_utils.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

enum {
    OFF_MAGIC = 0,
    OFF_VERSION = 4,
    OFF_FLAGS = 6,
    OFF_HEADER_BYTES = 8,
    OFF_LEVEL = 12,
    OFF_X = 16,
    OFF_Y = 20,
    OFF_PARENT_LEVEL = 24,
    OFF_PARENT_X = 28,
    OFF_PARENT_Y = 32,
    OFF_SAMPLE_WIDTH = 36,
    OFF_SAMPLE_HEIGHT = 38,
    OFF_GUTTER = 40,
    OFF_HEIGHT_ENCODING = 42,
    OFF_MIN_HEIGHT = 44,
    OFF_HEIGHT_RANGE = 48,
    OFF_GEOMETRIC_ERROR = 52,
    OFF_VALID_SAMPLE_COUNT = 56,
    OFF_HEIGHT_BYTES = 60,
    OFF_VALIDITY_BYTES = 64,
    OFF_IMAGERY_BYTES = 68,
    OFF_PROFILE_BYTES = 72,
    OFF_SOURCE_BYTES = 76,
    OFF_IMAGERY_URI_BYTES = 80,
    OFF_PAYLOAD_CRC32 = 84,
    OFF_EXTENT = 88,
    OFF_ROTATION = 120,
    OFF_TRANSLATION = 192,
};

static bool finite_header(const TerrainTileHeader *h) {
    if (!isfinite(h->min_height_m) || !isfinite(h->height_range_m) ||
        h->height_range_m < 0.0f || !isfinite(h->geometric_error_m) ||
        h->geometric_error_m < 0.0f)
        return false;
    for (unsigned i = 0; i < 4; ++i)
        if (!isfinite(h->extent[i])) return false;
    if (!(h->extent[0] < h->extent[2]) || !(h->extent[1] < h->extent[3]))
        return false;
    for (unsigned column = 0; column < 3; ++column)
        for (unsigned row = 0; row < 3; ++row)
            if (!isfinite(h->local_to_world.rotation[column][row])) return false;
    return isfinite(h->local_to_world.translation.x) &&
           isfinite(h->local_to_world.translation.y) &&
           isfinite(h->local_to_world.translation.z);
}

TerrainTileResult terrain_tile_load(const char *path, TerrainTile *out) {
    if (!path || !out) return TERRAIN_TILE_INVALID_ARGUMENT;

    TerrainTile tile = {0};
    uint8_t *file_data;
    size_t file_size;
    FileReadResult file_result = file_read_all(path, &file_data, &file_size);
    if (file_result == FILE_READ_OUT_OF_MEMORY) return TERRAIN_TILE_OUT_OF_MEMORY;
    if (file_result != FILE_READ_OK) return TERRAIN_TILE_IO_ERROR;
    if (file_size < TERRAIN_TILE_HEADER_BYTES) {
        free(file_data);
        return TERRAIN_TILE_TRUNCATED;
    }

    const uint8_t *raw_header = file_data;
    if (byte_read_u32_le(raw_header + OFF_MAGIC) != TERRAIN_TILE_MAGIC) {
        free(file_data);
        return TERRAIN_TILE_BAD_MAGIC;
    }
    if (byte_read_u16_le(raw_header + OFF_VERSION) != TERRAIN_TILE_VERSION) {
        free(file_data);
        return TERRAIN_TILE_UNSUPPORTED_VERSION;
    }
    if (byte_read_u32_le(raw_header + OFF_HEADER_BYTES) != TERRAIN_TILE_HEADER_BYTES) {
        free(file_data);
        return TERRAIN_TILE_INVALID_FORMAT;
    }

    TerrainTileHeader *h = &tile.header;
    h->flags = byte_read_u16_le(raw_header + OFF_FLAGS);
    h->level = byte_read_u32_le(raw_header + OFF_LEVEL);
    h->x = byte_read_u32_le(raw_header + OFF_X);
    h->y = byte_read_u32_le(raw_header + OFF_Y);
    h->parent_level = byte_read_u32_le(raw_header + OFF_PARENT_LEVEL);
    h->parent_x = byte_read_u32_le(raw_header + OFF_PARENT_X);
    h->parent_y = byte_read_u32_le(raw_header + OFF_PARENT_Y);
    h->sample_width = byte_read_u16_le(raw_header + OFF_SAMPLE_WIDTH);
    h->sample_height = byte_read_u16_le(raw_header + OFF_SAMPLE_HEIGHT);
    h->gutter = byte_read_u16_le(raw_header + OFF_GUTTER);
    h->height_encoding = byte_read_u16_le(raw_header + OFF_HEIGHT_ENCODING);
    h->min_height_m = byte_read_f32_le(raw_header + OFF_MIN_HEIGHT);
    h->height_range_m = byte_read_f32_le(raw_header + OFF_HEIGHT_RANGE);
    h->geometric_error_m = byte_read_f32_le(raw_header + OFF_GEOMETRIC_ERROR);
    h->valid_sample_count = byte_read_u32_le(raw_header + OFF_VALID_SAMPLE_COUNT);
    h->height_bytes = byte_read_u32_le(raw_header + OFF_HEIGHT_BYTES);
    h->validity_bytes = byte_read_u32_le(raw_header + OFF_VALIDITY_BYTES);
    h->imagery_bytes = byte_read_u32_le(raw_header + OFF_IMAGERY_BYTES);
    h->profile_bytes = byte_read_u32_le(raw_header + OFF_PROFILE_BYTES);
    h->source_bytes = byte_read_u32_le(raw_header + OFF_SOURCE_BYTES);
    h->imagery_uri_bytes = byte_read_u32_le(raw_header + OFF_IMAGERY_URI_BYTES);
    h->payload_crc32 = byte_read_u32_le(raw_header + OFF_PAYLOAD_CRC32);
    for (unsigned i = 0; i < 4; ++i)
        h->extent[i] = byte_read_f64_le(raw_header + OFF_EXTENT + i * 8u);
    for (unsigned column = 0; column < 3; ++column)
        for (unsigned row = 0; row < 3; ++row)
            h->local_to_world.rotation[column][row] =
                byte_read_f64_le(raw_header + OFF_ROTATION + (column * 3u + row) * 8u);
    h->local_to_world.translation.x = byte_read_f64_le(raw_header + OFF_TRANSLATION);
    h->local_to_world.translation.y = byte_read_f64_le(raw_header + OFF_TRANSLATION + 8);
    h->local_to_world.translation.z = byte_read_f64_le(raw_header + OFF_TRANSLATION + 16);

    uint64_t sample_count64 = (uint64_t)h->sample_width * h->sample_height;
    uint32_t bytes_per_height = h->height_encoding == TERRAIN_TILE_HEIGHT_R16_UNORM ? 2u :
                                h->height_encoding == TERRAIN_TILE_HEIGHT_F32 ? 4u : 0u;
    uint64_t expected_height = sample_count64 * bytes_per_height;
    uint64_t expected_validity = (sample_count64 + 7u) / 8u;
    uint64_t tiles_on_axis = h->level <= 31u ? UINT64_C(1) << h->level : 0u;
    uint32_t known_flags = TERRAIN_TILE_HAS_PARENT | TERRAIN_TILE_HAS_IMAGERY |
                           TERRAIN_TILE_HAS_NODATA;
    bool parent_ok = h->level == 0
        ? !(h->flags & TERRAIN_TILE_HAS_PARENT) &&
          h->parent_level == TERRAIN_TILE_ROOT_KEY &&
          h->parent_x == TERRAIN_TILE_ROOT_KEY && h->parent_y == TERRAIN_TILE_ROOT_KEY
        : (h->flags & TERRAIN_TILE_HAS_PARENT) &&
          h->parent_level + 1u == h->level && h->parent_x == h->x / 2u &&
          h->parent_y == h->y / 2u;
    if (!bytes_per_height || !sample_count64 || !tiles_on_axis ||
        h->x >= tiles_on_axis || h->y >= tiles_on_axis || h->gutter == 0 ||
        h->sample_width <= h->gutter * 2u || h->sample_height <= h->gutter * 2u ||
        h->height_bytes != expected_height || h->validity_bytes != expected_validity ||
        h->valid_sample_count > sample_count64 || (h->flags & ~known_flags) ||
        !parent_ok || !finite_header(h) || h->profile_bytes == 0 ||
        ((h->flags & TERRAIN_TILE_HAS_IMAGERY) != 0) !=
            (h->imagery_bytes != 0 || h->imagery_uri_bytes != 0)) {
        free(file_data);
        return TERRAIN_TILE_INVALID_FORMAT;
    }

    size_t payload_size = 0;
    if (!size_add_checked(&payload_size, h->height_bytes) ||
        !size_add_checked(&payload_size, h->validity_bytes) ||
        !size_add_checked(&payload_size, h->imagery_bytes) ||
        !size_add_checked(&payload_size, h->profile_bytes) ||
        !size_add_checked(&payload_size, h->source_bytes) ||
        !size_add_checked(&payload_size, h->imagery_uri_bytes)) {
        free(file_data);
        return TERRAIN_TILE_INVALID_FORMAT;
    }
    size_t available_payload = file_size - TERRAIN_TILE_HEADER_BYTES;
    if (payload_size > available_payload) {
        free(file_data);
        return TERRAIN_TILE_TRUNCATED;
    }
    if (payload_size < available_payload) {
        free(file_data);
        return TERRAIN_TILE_INVALID_FORMAT;
    }

    const uint8_t *payload = file_data + TERRAIN_TILE_HEADER_BYTES;
    if (checksum_crc32(payload, payload_size) != h->payload_crc32) {
        free(file_data);
        return TERRAIN_TILE_CHECKSUM_MISMATCH;
    }

    const uint8_t *cursor = payload;
    tile.height_data = malloc(h->height_bytes);
    tile.validity = malloc(h->validity_bytes);
    tile.imagery = h->imagery_bytes ? malloc(h->imagery_bytes) : NULL;
    if (!tile.height_data || !tile.validity || (h->imagery_bytes && !tile.imagery)) {
        free(file_data);
        free(tile.height_data); free(tile.validity); free(tile.imagery);
        return TERRAIN_TILE_OUT_OF_MEMORY;
    }
    memcpy(tile.height_data, cursor, h->height_bytes); cursor += h->height_bytes;
    memcpy(tile.validity, cursor, h->validity_bytes); cursor += h->validity_bytes;
    if (h->imagery_bytes) {
        memcpy(tile.imagery, cursor, h->imagery_bytes);
        cursor += h->imagery_bytes;
    }
    tile.profile = str_dup_n((const char *)cursor, h->profile_bytes);
    cursor += h->profile_bytes;
    tile.source = str_dup_n((const char *)cursor, h->source_bytes);
    cursor += h->source_bytes;
    tile.imagery_uri = str_dup_n((const char *)cursor, h->imagery_uri_bytes);
    free(file_data);
    if (!tile.profile || !tile.source || !tile.imagery_uri) {
        tile.owns_offline_data = true;
        terrain_tile_unload(&tile);
        return TERRAIN_TILE_OUT_OF_MEMORY;
    }

    uint32_t counted_valid = 0;
    for (uint64_t i = 0; i < sample_count64; ++i)
        counted_valid += (tile.validity[i >> 3] >> (i & 7u)) & 1u;
    bool samples_in_range = true;
    float range_tolerance = fmaxf(1e-5f, h->height_range_m * 1e-6f);
    for (uint32_t y = 0; samples_in_range && y < h->sample_height; ++y) {
        for (uint32_t x = 0; x < h->sample_width; ++x) {
            if (!terrain_tile_sample_valid(&tile, x, y)) continue;
            float height = terrain_tile_height(&tile, x, y);
            if (!isfinite(height) || height < h->min_height_m - range_tolerance ||
                height > h->min_height_m + h->height_range_m + range_tolerance) {
                samples_in_range = false;
                break;
            }
        }
    }
    if (counted_valid != h->valid_sample_count ||
        (!!(h->flags & TERRAIN_TILE_HAS_NODATA) != (counted_valid != sample_count64)) ||
        !samples_in_range) {
        tile.owns_offline_data = true;
        terrain_tile_unload(&tile);
        return TERRAIN_TILE_INVALID_FORMAT;
    }

    tile.base.local_to_world = h->local_to_world;
    tile.base.texture_path = tile.imagery_uri[0] ? tile.imagery_uri : NULL;
    tile.owns_offline_data = true;
    *out = tile;
    return TERRAIN_TILE_OK;
}

void terrain_tile_unload(TerrainTile *tile) {
    if (!tile) return;
    if (tile->owns_offline_data) {
        free(tile->height_data);
        free(tile->validity);
        free(tile->imagery);
        free(tile->profile);
        free(tile->source);
        free(tile->imagery_uri);
    }
    *tile = (TerrainTile){0};
}

bool terrain_tile_sample_valid(const TerrainTile *tile, uint32_t x, uint32_t y) {
    if (!tile || !tile->validity || x >= tile->header.sample_width ||
        y >= tile->header.sample_height)
        return false;
    size_t index = (size_t)y * tile->header.sample_width + x;
    return ((tile->validity[index >> 3] >> (index & 7u)) & 1u) != 0;
}

float terrain_tile_height(const TerrainTile *tile, uint32_t x, uint32_t y) {
    if (!terrain_tile_sample_valid(tile, x, y)) return NAN;
    size_t index = (size_t)y * tile->header.sample_width + x;
    const uint8_t *data = tile->height_data;
    if (tile->header.height_encoding == TERRAIN_TILE_HEIGHT_R16_UNORM) {
        uint16_t encoded = byte_read_u16_le(data + index * 2u);
        return tile->header.min_height_m +
               tile->header.height_range_m * ((float)encoded / 65535.0f);
    }
    return byte_read_f32_le(data + index * 4u);
}

const char *terrain_tile_result_string(TerrainTileResult result) {
    switch (result) {
    case TERRAIN_TILE_OK: return "success";
    case TERRAIN_TILE_INVALID_ARGUMENT: return "invalid argument";
    case TERRAIN_TILE_IO_ERROR: return "I/O error";
    case TERRAIN_TILE_TRUNCATED: return "truncated tile";
    case TERRAIN_TILE_BAD_MAGIC: return "bad tile magic";
    case TERRAIN_TILE_UNSUPPORTED_VERSION: return "unsupported tile version";
    case TERRAIN_TILE_INVALID_FORMAT: return "invalid tile format";
    case TERRAIN_TILE_CHECKSUM_MISMATCH: return "tile checksum mismatch";
    case TERRAIN_TILE_OUT_OF_MEMORY: return "out of memory";
    }
    return "unknown terrain tile error";
}
