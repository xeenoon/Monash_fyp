// Headless Vulkan terrain baker: real imagery + .trn heightmap -> five PNGs.
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include "bake_clean.h"
#include "bake_gpu.h"
#include "bake_image.h"
#include "bake_jfa.h"
#include "bake_pipeline.h"
#include "bake_resample.h"
#include "terrain_tile.h"

#ifndef BAKE_DATASET_DIR
#define BAKE_DATASET_DIR "."
#endif

static int make_directory(const char *path) {
#ifdef _WIN32
    return _mkdir(path) == 0 || errno == EEXIST;
#else
    return mkdir(path, 0775) == 0 || errno == EEXIST;
#endif
}

static int join(char *out, size_t size, const char *a, const char *b) {
    int n = snprintf(out, size, "%s/%s", a, b);
    return n >= 0 && (size_t)n < size;
}

static int load_interior_image(const char *path, BakeImage *out) {
    BakeImage raw = {0};
    if (!bake_image_load(path, &raw)) return 0;
    int border = raw.width == 258 && raw.height == 258 ? 1 : 0;
    int width = raw.width - border * 2, height = raw.height - border * 2;
    if (!bake_image_alloc(width, height, out)) { bake_image_free(&raw); return 0; }
    for (int y = 0; y < height; ++y)
        memcpy(out->pixels + (size_t)y * width,
               raw.pixels + (size_t)(y + border) * raw.width + border,
               (size_t)width * sizeof(uint32_t));
    bake_image_free(&raw);
    return 1;
}

static int load_height(const char *path, float **out, int *width, int *height,
                       float *threshold) {
    TerrainTile tile;
    TerrainTileResult result = terrain_tile_load(path, &tile);
    if (result != TERRAIN_TILE_OK) {
        fprintf(stderr, "terrain_bake: cannot load height tile '%s': %s\n", path,
                terrain_tile_result_string(result));
        return 0;
    }
    int interior_w = tile.header.sample_width - 2 * tile.header.gutter;
    int interior_h = tile.header.sample_height - 2 * tile.header.gutter;
    float *values = malloc((size_t)interior_w * interior_h * sizeof(float));
    if (!values) { terrain_tile_unload(&tile); return 0; }
    float lo = INFINITY, hi = -INFINITY;
    for (int y = 0; y < interior_h; ++y) for (int x = 0; x < interior_w; ++x) {
        float v = terrain_tile_height(&tile, (uint32_t)x + tile.header.gutter,
                                      (uint32_t)y + tile.header.gutter);
        if (!isfinite(v)) v = tile.header.min_height_m;
        values[y * interior_w + x] = v;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    terrain_tile_unload(&tile);
    *threshold = lo + (hi - lo) * 0.52f;
    *width = interior_w;
    *height = interior_h;
    *out = values;
    return 1;
}

static int isolate_material(BakeGpu *gpu, const BakeImage *source, uint32_t material,
                            uint32_t *atlas_plane) {
    size_t count = (size_t)source->width * source->height;
    size_t bytes = count * sizeof(uint32_t);
    BakeBuffer input = bake_buffer_host(gpu, bytes), cleaned = bake_buffer_host(gpu, bytes);
    BakeBuffer confidence = bake_buffer_host(gpu, bytes), hard = bake_buffer_host(gpu, bytes);
    BakeBuffer labels = bake_buffer_host(gpu, bytes), mask = bake_buffer_host(gpu, bytes);
    BakeBuffer owners = bake_buffer_host(gpu, bytes), isolated = bake_buffer_host(gpu, bytes);
    memcpy(input.mapped, source->pixels, bytes);
    bake_clean_gpu(gpu, &input, &cleaned, &confidence, &hard, source->width, source->height);
    struct { uint32_t width, height; } size = {(uint32_t)source->width,
                                               (uint32_t)source->height};
    uint32_t gx = (uint32_t)(source->width + 7) / 8,
             gy = (uint32_t)(source->height + 7) / 8;
    BakePipeline classify = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_classify.comp.spv", 2, sizeof(size));
    BakeBuffer cb[2] = {cleaned, labels};
    bake_dispatch(gpu, &classify, cb, 2, &size, sizeof(size), gx, gy, 1);
    bake_pipeline_destroy(gpu, &classify);
    struct { uint32_t width, height, material; } mp = {
        (uint32_t)source->width, (uint32_t)source->height, material};
    BakePipeline masks = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_material_mask.comp.spv", 2, sizeof(mp));
    BakeBuffer mb[2] = {labels, mask};
    bake_dispatch(gpu, &masks, mb, 2, &mp, sizeof(mp), gx, gy, 1);
    bake_pipeline_destroy(gpu, &masks);
    size_t valid = 0;
    for (size_t i = 0; i < count; ++i) valid += ((uint32_t *)mask.mapped)[i] == 0;
    if (!valid) {
        fprintf(stderr, "terrain_bake: exemplar has no pixels of material %u\n", material);
        return 0;
    }
    bake_jfa_gpu(gpu, &mask, &owners, source->width, source->height);
    BakePipeline gather = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_gather.comp.spv", 3, sizeof(size));
    BakeBuffer gb[3] = {cleaned, owners, isolated};
    bake_dispatch(gpu, &gather, gb, 3, &size, sizeof(size), gx, gy, 1);
    bake_pipeline_destroy(gpu, &gather);
    memcpy(atlas_plane, isolated.mapped, bytes);
    printf("  exemplar material %u: %.1f%% native pixels, remainder JFA-filled\n",
           material, 100.0 * (double)valid / (double)count);
    bake_buffer_destroy(gpu, &isolated); bake_buffer_destroy(gpu, &owners);
    bake_buffer_destroy(gpu, &mask); bake_buffer_destroy(gpu, &labels);
    bake_buffer_destroy(gpu, &hard); bake_buffer_destroy(gpu, &confidence);
    bake_buffer_destroy(gpu, &cleaned); bake_buffer_destroy(gpu, &input);
    return 1;
}

int main(int argc, char **argv) {
    const char *dataset = BAKE_DATASET_DIR, *output_dir = "baked_materials";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--dataset") && i + 1 < argc) dataset = argv[++i];
        else if (!strcmp(argv[i], "--output") && i + 1 < argc) output_dir = argv[++i];
        else { fprintf(stderr, "usage: %s [--dataset DIR] [--output DIR]\n", argv[0]); return 2; }
    }
    if (!make_directory(output_dir)) {
        fprintf(stderr, "terrain_bake: cannot create output directory '%s'\n", output_dir);
        return 1;
    }
    const char *relative[3] = {"imagery/5/1/2.png", "imagery/5/6/16.png",
                               "imagery/5/10/24.png"};
    BakeImage source[3] = {{0}}; char path[1024];
    for (int i = 0; i < 3; ++i) {
        if (!join(path, sizeof(path), dataset, relative[i]) ||
            !load_interior_image(path, &source[i])) return 1;
    }
    if (source[0].width != source[1].width || source[0].width != source[2].width ||
        source[0].height != source[1].height || source[0].height != source[2].height) {
        fprintf(stderr, "terrain_bake: exemplar dimensions differ\n"); return 1;
    }
    int width = source[0].width, height = source[0].height;
    if (!join(path, sizeof(path), dataset, "tiles/5/23/8.trn")) return 1;
    float *height_values = NULL, threshold = 0.0f;
    int height_width = 0, height_height = 0;
    if (!load_height(path, &height_values, &height_width, &height_height,
                     &threshold)) return 1;

    BakeGpu gpu;
    if (!bake_gpu_init(&gpu)) return 1;
    printf("terrain_bake: %s, imagery %dx%d, height %dx%d -> %dx%d, "
           "transition threshold %.1fm\n", gpu.device_name, width, height,
           height_width, height_height, width, height, threshold);
    size_t count = (size_t)width * height, bytes = count * sizeof(uint32_t);
    BakeBuffer atlas = bake_buffer_host(&gpu, bytes * 3);
    for (uint32_t material = 0; material < 3; ++material)
        if (!isolate_material(&gpu, &source[material], material,
                              (uint32_t *)atlas.mapped + count * material)) return 1;
    size_t height_source_count = (size_t)height_width * height_height;
    BakeBuffer height_source = bake_buffer_host(
        &gpu, height_source_count * sizeof(float));
    BakeBuffer heightmap = bake_buffer_host(&gpu, count * sizeof(float));
    memcpy(height_source.mapped, height_values,
           height_source_count * sizeof(float));
    bake_resample_bilinear_gpu(&gpu, &height_source, &heightmap,
                               height_width, height_height, width, height);
    BakeBuffer outputs[BAKE_PRESET_COUNT];
    for (int i = 0; i < BAKE_PRESET_COUNT; ++i) outputs[i] = bake_buffer_host(&gpu, bytes);
    bake_pipeline_gpu(&gpu, &atlas, &heightmap, outputs, width, height,
                      threshold, 151500u);
    BakeImage image = {width, height, NULL};
    for (int preset = 0; preset < BAKE_PRESET_COUNT; ++preset) {
        image.pixels = outputs[preset].mapped;
        char filename[1024], leaf[128];
        snprintf(leaf, sizeof(leaf), "%s.png", bake_preset_names[preset]);
        if (!join(filename, sizeof(filename), output_dir, leaf) ||
            !bake_image_write_png(filename, &image)) return 1;
        printf("  wrote %s\n", filename);
    }
    for (int i = 0; i < BAKE_PRESET_COUNT; ++i) bake_buffer_destroy(&gpu, &outputs[i]);
    bake_buffer_destroy(&gpu, &heightmap);
    bake_buffer_destroy(&gpu, &height_source);
    bake_buffer_destroy(&gpu, &atlas);
    bake_gpu_destroy(&gpu);
    for (int i = 0; i < 3; ++i) bake_image_free(&source[i]);
    free(height_values);
    return 0;
}
