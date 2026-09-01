#include "bake_sample.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bake_clean.h"
#include "bake_image.h"
#include "bake_pipeline.h"
#include "bake_resample.h"
#include "terrain_tile.h"

static int path_for(char *out, size_t size, const char *dataset,
                    const char *kind, int x, int y, const char *extension) {
    int written = snprintf(out, size, "%s/%s/5/%d/%d.%s",
                           dataset, kind, x, y, extension);
    return written >= 0 && (size_t)written < size;
}

static bool load_rgb(const char *path, BakeTerrainSample *sample) {
    BakeImage raw = {0};
    if (!bake_image_load(path, &raw)) return false;
    int gutter = raw.width == 258 && raw.height == 258 ? 1 : 0;
    int width = raw.width - 2 * gutter, height = raw.height - 2 * gutter;
    uint32_t *rgb = malloc((size_t)width * height * sizeof(*rgb));
    if (!rgb) {
        bake_image_free(&raw);
        return false;
    }
    for (int y = 0; y < height; ++y)
        memcpy(rgb + (size_t)y * width,
               raw.pixels + (size_t)(y + gutter) * raw.width + gutter,
               (size_t)width * sizeof(*rgb));
    bake_image_free(&raw);
    sample->width = width;
    sample->height = height;
    sample->rgb = rgb;
    return true;
}

static bool load_dem(BakeGpu *gpu, const char *path, BakeTerrainSample *sample) {
    TerrainTile tile;
    TerrainTileResult result = terrain_tile_load(path, &tile);
    if (result != TERRAIN_TILE_OK) return false;
    int source_width = tile.header.sample_width - 2 * tile.header.gutter;
    int source_height = tile.header.sample_height - 2 * tile.header.gutter;
    size_t source_count = (size_t)source_width * source_height;
    size_t target_count = (size_t)sample->width * sample->height;
    float *source = malloc(source_count * sizeof(*source));
    sample->elevation = malloc(target_count * sizeof(*sample->elevation));
    sample->slope = malloc(target_count * sizeof(*sample->slope));
    sample->gradient_x = malloc(target_count * sizeof(*sample->gradient_x));
    sample->gradient_y = malloc(target_count * sizeof(*sample->gradient_y));
    if (!source || !sample->elevation || !sample->slope ||
        !sample->gradient_x || !sample->gradient_y) {
        free(source);
        terrain_tile_unload(&tile);
        return false;
    }
    for (int y = 0; y < source_height; ++y)
        for (int x = 0; x < source_width; ++x)
            source[(size_t)y * source_width + x] = terrain_tile_height(
                &tile, (uint32_t)x + tile.header.gutter,
                (uint32_t)y + tile.header.gutter);
    double extent_width = tile.header.extent[2] - tile.header.extent[0];
    float spacing = (float)(extent_width / (double)(source_width - 1) * 0.5);
    terrain_tile_unload(&tile);

    BakeBuffer source_buffer = bake_buffer_host(gpu, source_count * sizeof(float));
    BakeBuffer target_buffer = bake_buffer_host(gpu, target_count * sizeof(float));
    memcpy(source_buffer.mapped, source, source_count * sizeof(float));
    bake_resample_bilinear_gpu(gpu, &source_buffer, &target_buffer,
                               source_width, source_height,
                               sample->width, sample->height);
    memcpy(sample->elevation, target_buffer.mapped, target_count * sizeof(float));
    bake_buffer_destroy(gpu, &target_buffer);
    bake_buffer_destroy(gpu, &source_buffer);
    free(source);

    for (int y = 0; y < sample->height; ++y) {
        int ym = y > 0 ? y - 1 : y, yp = y + 1 < sample->height ? y + 1 : y;
        float dy_scale = (y > 0 && y + 1 < sample->height) ? 0.5f : 1.0f;
        for (int x = 0; x < sample->width; ++x) {
            int xm = x > 0 ? x - 1 : x, xp = x + 1 < sample->width ? x + 1 : x;
            float dx_scale = (x > 0 && x + 1 < sample->width) ? 0.5f : 1.0f;
            size_t i = (size_t)y * sample->width + x;
            float gx = (sample->elevation[(size_t)y * sample->width + xp] -
                        sample->elevation[(size_t)y * sample->width + xm]) *
                       dx_scale / spacing;
            float gy = (sample->elevation[(size_t)yp * sample->width + x] -
                        sample->elevation[(size_t)ym * sample->width + x]) *
                       dy_scale / spacing;
            sample->gradient_x[i] = gx;
            sample->gradient_y[i] = gy;
            sample->slope[i] = hypotf(gx, gy);
        }
    }
    return true;
}

bool bake_sample_load(BakeGpu *gpu, const char *dataset, int tile_x, int tile_y,
                      bool clean_rgb, BakeTerrainSample *sample) {
    if (!gpu || !dataset || !sample) return false;
    memset(sample, 0, sizeof(*sample));
    sample->tile_x = tile_x;
    sample->tile_y = tile_y;
    char path[1024];
    if (!path_for(path, sizeof(path), dataset, "imagery", tile_x, tile_y, "png") ||
        !load_rgb(path, sample)) goto fail;
    if (!path_for(path, sizeof(path), dataset, "tiles", tile_x, tile_y, "trn") ||
        !load_dem(gpu, path, sample)) goto fail;

    size_t count = (size_t)sample->width * sample->height;
    size_t bytes = count * sizeof(uint32_t);
    sample->labels = malloc(bytes);
    if (!sample->labels) goto fail;
    BakeBuffer source = bake_buffer_host(gpu, bytes);
    BakeBuffer cleaned = bake_buffer_host(gpu, bytes);
    BakeBuffer confidence = bake_buffer_host(gpu, bytes);
    BakeBuffer hard = bake_buffer_host(gpu, bytes);
    BakeBuffer labels = bake_buffer_host(gpu, bytes);
    memcpy(source.mapped, sample->rgb, bytes);
    if (clean_rgb) {
        // The pass17 oracle uses SciPy's float64 Gaussian coefficients during
        // this one-time source cleanup.  Keep that exact preparation on the
        // CPU; classification and all iterative synthesis remain GPU-backed.
        bake_clean_cpu(source.mapped, cleaned.mapped, confidence.mapped,
                       hard.mapped, sample->width, sample->height);
    } else {
        memcpy(cleaned.mapped, source.mapped, bytes);
        for (size_t i = 0; i < count; ++i) {
            ((uint32_t *)confidence.mapped)[i] = 255;
            ((uint32_t *)hard.mapped)[i] = 0;
        }
    }
    struct { uint32_t width, height; } push = {
        (uint32_t)sample->width, (uint32_t)sample->height};
    BakePipeline classify = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_classify.comp.spv", 2, sizeof(push));
    BakeBuffer bindings[2] = {cleaned, labels};
    bake_dispatch(gpu, &classify, bindings, 2, &push, sizeof(push),
                  (uint32_t)(sample->width + 7) / 8,
                  (uint32_t)(sample->height + 7) / 8, 1);
    bake_pipeline_destroy(gpu, &classify);
    memcpy(sample->rgb, cleaned.mapped, bytes);
    memcpy(sample->labels, labels.mapped, bytes);
    size_t repaired = 0, hard_count = 0;
    for (size_t i = 0; i < count; ++i) {
        repaired += ((uint32_t *)confidence.mapped)[i] < 255;
        hard_count += ((uint32_t *)hard.mapped)[i] != 0;
    }
    sample->repair_fraction = (float)((double)repaired / (double)count);
    sample->hard_fraction = (float)((double)hard_count / (double)count);
    bake_buffer_destroy(gpu, &labels);
    bake_buffer_destroy(gpu, &hard);
    bake_buffer_destroy(gpu, &confidence);
    bake_buffer_destroy(gpu, &cleaned);
    bake_buffer_destroy(gpu, &source);
    return true;
fail:
    bake_sample_free(sample);
    return false;
}

void bake_sample_free(BakeTerrainSample *sample) {
    if (!sample) return;
    free(sample->gradient_y);
    free(sample->gradient_x);
    free(sample->slope);
    free(sample->elevation);
    free(sample->labels);
    free(sample->rgb);
    memset(sample, 0, sizeof(*sample));
}
