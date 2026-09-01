#include "bake_pipeline.h"

#include <stdlib.h>
#include <string.h>

#include "bake_bands.h"
#include "bake_patchmatch.h"
#include "bake_quilt.h"
#include "bake_relayer.h"
#include "bake_transition.h"

const char *const bake_preset_names[BAKE_PRESET_COUNT] = {
    "full_snow", "snow_rock", "full_rock", "rock_grass", "full_grass"};

static uint32_t pure_kind(int preset) {
    return preset == BAKE_FULL_SNOW ? 2u : (preset == BAKE_FULL_GRASS ? 1u : 0u);
}

void bake_pipeline_cpu(const uint32_t *atlas, const float *heightmap,
                       uint32_t *outputs, int width, int height,
                       float threshold, uint32_t seed) {
    size_t count = (size_t)width * height, samples = count * 3;
    float *bands = malloc(samples * 3 * 4 * sizeof(float));
    float *low = bands, *mid = low + samples * 3, *meso = mid + samples * 3;
    float *fine = meso + samples * 3;
    float *rgb = malloc(samples * sizeof(float));
    uint32_t *labels = malloc(count * sizeof(uint32_t));
    int32_t *sdf = malloc(count * sizeof(int32_t));
    uint32_t *owners = malloc(count * sizeof(uint32_t));
    uint32_t *owner_scratch = malloc(count * sizeof(uint32_t));
    uint32_t *quilt = malloc(count * sizeof(uint32_t));
    for (int material = 0; material < 3; ++material) {
        BakeBandsCpu b = {rgb, low + samples * material, mid + samples * material,
                          meso + samples * material, fine + samples * material};
        bake_bands_cpu(atlas + count * material, b, width, height);
    }
    for (int preset = 0; preset < BAKE_PRESET_COUNT; ++preset) {
        if (preset == BAKE_SNOW_ROCK) {
            bake_transition_cpu(heightmap, atlas, labels, sdf, owners, quilt,
                                width, height, threshold, 0, 2, seed + (uint32_t)preset);
        } else if (preset == BAKE_ROCK_GRASS) {
            bake_transition_cpu(heightmap, atlas, labels, sdf, owners, quilt,
                                width, height, threshold, 1, 0, seed + (uint32_t)preset);
        } else {
            uint32_t kind = pure_kind(preset);
            for (size_t i = 0; i < count; ++i) labels[i] = kind;
            bake_quilt_cpu(atlas, labels, quilt, owners, width, height,
                           seed + (uint32_t)preset);
            bake_patchmatch_cpu(atlas, labels, owners, owner_scratch, width, height,
                                (seed + (uint32_t)preset) ^ 0xa511e9b3u, 4);
            for (size_t i = 0; i < count; ++i) quilt[i] = atlas[owners[i]];
        }
        bake_relayer_cpu(quilt, labels, owners, low, mid, meso, fine,
                         outputs + count * preset, width, height);
    }
    free(quilt); free(owner_scratch); free(owners); free(sdf); free(labels);
    free(rgb); free(bands);
}

void bake_pipeline_gpu(BakeGpu *gpu, const BakeBuffer *atlas,
                       const BakeBuffer *heightmap, BakeBuffer outputs[BAKE_PRESET_COUNT],
                       int width, int height, float threshold, uint32_t seed) {
    size_t count = (size_t)width * height, ubytes = count * sizeof(uint32_t);
    size_t plane_fbytes = count * 3 * sizeof(float), atlas_fbytes = plane_fbytes * 3;
    BakeBuffer low = bake_buffer_host(gpu, atlas_fbytes), mid = bake_buffer_host(gpu, atlas_fbytes);
    BakeBuffer meso = bake_buffer_host(gpu, atlas_fbytes), fine = bake_buffer_host(gpu, atlas_fbytes);
    BakeBuffer plane = bake_buffer_host(gpu, ubytes), rgb = bake_buffer_host(gpu, plane_fbytes);
    BakeBuffer plow = bake_buffer_host(gpu, plane_fbytes), pmid = bake_buffer_host(gpu, plane_fbytes);
    BakeBuffer pmeso = bake_buffer_host(gpu, plane_fbytes), pfine = bake_buffer_host(gpu, plane_fbytes);
    for (int material = 0; material < 3; ++material) {
        memcpy(plane.mapped, (const uint32_t *)atlas->mapped + count * material, ubytes);
        bake_bands_gpu(gpu, &plane, &rgb, &plow, &pmid, &pmeso, &pfine, width, height);
        memcpy((float *)low.mapped + count * 3 * material, plow.mapped, plane_fbytes);
        memcpy((float *)mid.mapped + count * 3 * material, pmid.mapped, plane_fbytes);
        memcpy((float *)meso.mapped + count * 3 * material, pmeso.mapped, plane_fbytes);
        memcpy((float *)fine.mapped + count * 3 * material, pfine.mapped, plane_fbytes);
    }
    BakeBuffer labels = bake_buffer_host(gpu, ubytes), sdf = bake_buffer_host(gpu, ubytes);
    BakeBuffer owners = bake_buffer_host(gpu, ubytes), quilt = bake_buffer_host(gpu, ubytes);
    for (int preset = 0; preset < BAKE_PRESET_COUNT; ++preset) {
        if (preset == BAKE_SNOW_ROCK) {
            bake_transition_gpu(gpu, heightmap, atlas, &labels, &sdf, &owners, &quilt,
                                width, height, threshold, 0, 2, seed + (uint32_t)preset);
        } else if (preset == BAKE_ROCK_GRASS) {
            bake_transition_gpu(gpu, heightmap, atlas, &labels, &sdf, &owners, &quilt,
                                width, height, threshold, 1, 0, seed + (uint32_t)preset);
        } else {
            uint32_t kind = pure_kind(preset);
            for (size_t i = 0; i < count; ++i) ((uint32_t *)labels.mapped)[i] = kind;
            bake_quilt_gpu(gpu, atlas, &labels, &quilt, &owners, width, height,
                           seed + (uint32_t)preset);
            bake_patchmatch_gpu(gpu, atlas, &labels, &owners, &quilt, width, height,
                                (seed + (uint32_t)preset) ^ 0xa511e9b3u, 4);
        }
        bake_relayer_gpu(gpu, &quilt, &labels, &owners, &low, &mid, &meso, &fine,
                         &outputs[preset], width, height);
    }
    bake_buffer_destroy(gpu, &quilt); bake_buffer_destroy(gpu, &owners);
    bake_buffer_destroy(gpu, &sdf); bake_buffer_destroy(gpu, &labels);
    bake_buffer_destroy(gpu, &pfine); bake_buffer_destroy(gpu, &pmeso);
    bake_buffer_destroy(gpu, &pmid); bake_buffer_destroy(gpu, &plow);
    bake_buffer_destroy(gpu, &rgb); bake_buffer_destroy(gpu, &plane);
    bake_buffer_destroy(gpu, &fine); bake_buffer_destroy(gpu, &meso);
    bake_buffer_destroy(gpu, &mid); bake_buffer_destroy(gpu, &low);
}
