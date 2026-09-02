#include "bake_pass17_relayer.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bake_filter.h"
#include "bake_random.h"
#include "bake_seam.h"

enum {
    MATERIALS = 3,
    CHANNELS = 3,
    EX_PATCH = 80,
    EX_BATCH = 9,
    EX_MAX_ATTEMPTS = 1024,
};

static float exemplar_min_purity(int kind) {
    return kind == 0 ? .85f : .94f;
}

static int exemplar_max_foreign_component(int kind) {
    return kind == 0 ? 256 : 128;
}

// Fine (high-frequency) restoration cap per material. Rock/snow stay low to
// avoid the embossed grit the user rejected; grass needs far more fine energy
// or the quilt base reads as washed-out pale streaks (contrast deficit vs the
// pass17 golden). BAKE_GRASS == 1.
static float exemplar_fine_cap(int kind) {
    return kind == 1 ? 1.5f : .55f;
}

static uint8_t byte_channel(uint32_t pixel, int channel) {
    return (uint8_t)(pixel >> (channel * 8));
}

static float pixel_luma(const float *rgb) {
    return (rgb[0] * .2126f + rgb[1] * .7152f + rgb[2] * .0722f) / 255.0f;
}

// A bright, desaturated pixel inside a grass crop is a sunlit bare/rock/snow
// speck the classifier still labels grass. Left in the exemplar the fine band
// amplifies it into a white streak, so we detect and remove it.
static bool pixel_is_bright(const float *rgb) {
    const float lum = .299f * rgb[0] + .587f * rgb[1] + .114f * rgb[2];
    const float mx = fmaxf(rgb[0], fmaxf(rgb[1], rgb[2]));
    const float mn = fminf(rgb[0], fminf(rgb[1], rgb[2]));
    return lum > 150.0f && (mx - mn) / fmaxf(mx, 1.0e-5f) < .22f;
}

// Black out those specks the same way we mask trees: replace each with the
// crop's grass mean so it contributes no bright detail. Grass only (kind 1);
// snow/rock legitimately contain bright pixels.
static void neutralize_bright_grass(float *crop, int pixels) {
    double sum[3] = {0};
    int count = 0;
    for (int i = 0; i < pixels; ++i) {
        if (pixel_is_bright(crop + i * 3)) continue;
        for (int c = 0; c < 3; ++c) sum[c] += crop[i * 3 + c];
        ++count;
    }
    if (!count) return;
    const float mean[3] = {(float)(sum[0] / count), (float)(sum[1] / count),
                           (float)(sum[2] / count)};
    for (int i = 0; i < pixels; ++i)
        if (pixel_is_bright(crop + i * 3))
            for (int c = 0; c < 3; ++c) crop[i * 3 + c] = mean[c];
}

static bool filter_layers(BakeGpu *gpu, const float *input, float *output,
                          int width, int height, int channels, int layers,
                          float sigma) {
    const size_t values = (size_t)width * height * channels * layers;
    BakeBuffer source = bake_buffer_host(gpu, values * sizeof(float));
    BakeBuffer result = bake_buffer_host(gpu, values * sizeof(float));
    BakeBuffer scratch = bake_buffer_host(gpu, values * sizeof(float));
    if (!source.mapped || !result.mapped || !scratch.mapped) return false;
    memcpy(source.mapped, input, values * sizeof(float));
    bake_gaussian_batch_gpu(gpu, &source, &result, &scratch, width, height,
                            channels, layers, sigma, 4.0f, 0);
    memcpy(output, result.mapped, values * sizeof(float));
    bake_buffer_destroy(gpu, &scratch);
    bake_buffer_destroy(gpu, &result);
    bake_buffer_destroy(gpu, &source);
    return true;
}

static void rgb_seam_cost(const float *existing, const float *candidate,
                          int patch, float *cost) {
    for (int y = 0; y < patch; ++y) {
        const int ym = y > 0 ? y - 1 : y;
        const int yp = y + 1 < patch ? y + 1 : y;
        const float yscale = y > 0 && y + 1 < patch ? .5f : 1.0f;
        for (int x = 0; x < patch; ++x) {
            const int xm = x > 0 ? x - 1 : x;
            const int xp = x + 1 < patch ? x + 1 : x;
            const float xscale = x > 0 && x + 1 < patch ? .5f : 1.0f;
            const size_t p = ((size_t)y * patch + x) * 3;
            float colour = 0.0f;
            for (int c = 0; c < 3; ++c) {
                const float delta = (existing[p + c] - candidate[p + c]) / 255.0f;
                colour += delta * delta;
            }
            colour /= 3.0f;
            const float egx = (pixel_luma(existing + ((size_t)y * patch + xp) * 3) -
                               pixel_luma(existing + ((size_t)y * patch + xm) * 3)) *
                              xscale;
            const float egy = (pixel_luma(existing + ((size_t)yp * patch + x) * 3) -
                               pixel_luma(existing + ((size_t)ym * patch + x) * 3)) *
                              yscale;
            const float cgx = (pixel_luma(candidate + ((size_t)y * patch + xp) * 3) -
                               pixel_luma(candidate + ((size_t)y * patch + xm) * 3)) *
                              xscale;
            const float cgy = (pixel_luma(candidate + ((size_t)yp * patch + x) * 3) -
                               pixel_luma(candidate + ((size_t)ym * patch + x) * 3)) *
                              yscale;
            const float dx = egx - cgx, dy = egy - cgy;
            cost[(size_t)y * patch + x] = colour + .5f * (dx * dx + dy * dy);
        }
    }
}

static int largest_foreign_component(const BakeTerrainSample *source, int kind,
                                     int source_x, int source_y) {
    uint8_t visited[EX_PATCH * EX_PATCH] = {0};
    int queue[EX_PATCH * EX_PATCH];
    int largest = 0;
    for (int y = 0; y < EX_PATCH; ++y) {
        for (int x = 0; x < EX_PATCH; ++x) {
            const int start = y * EX_PATCH + x;
            if (visited[start] ||
                source->labels[(size_t)(source_y + y) * source->width +
                               source_x + x] == (uint32_t)kind)
                continue;
            int head = 0, tail = 0;
            visited[start] = 1;
            queue[tail++] = start;
            while (head < tail) {
                const int local = queue[head++];
                const int cx = local % EX_PATCH, cy = local / EX_PATCH;
                static const int dx[4] = {-1, 1, 0, 0};
                static const int dy[4] = {0, 0, -1, 1};
                for (int direction = 0; direction < 4; ++direction) {
                    const int nx = cx + dx[direction], ny = cy + dy[direction];
                    if (nx < 0 || nx >= EX_PATCH || ny < 0 || ny >= EX_PATCH)
                        continue;
                    const int next = ny * EX_PATCH + nx;
                    if (visited[next] ||
                        source->labels[(size_t)(source_y + ny) * source->width +
                                       source_x + nx] == (uint32_t)kind)
                        continue;
                    visited[next] = 1;
                    queue[tail++] = next;
                }
            }
            if (tail > largest) largest = tail;
            if (largest > exemplar_max_foreign_component(kind)) return largest;
        }
    }
    return largest;
}

typedef struct {
    float energy;
    int candidate;
} ExemplarEnergy;

static int exemplar_energy_compare(const void *left, const void *right) {
    const ExemplarEnergy *a = left, *b = right;
    if (a->energy < b->energy) return -1;
    if (a->energy > b->energy) return 1;
    return a->candidate - b->candidate;
}

static bool build_material_exemplar(
    BakeGpu *gpu, const BakeTerrainSample *sources, int source_count, int kind,
    BakePcg64 *rng, int width, int height, float *exemplar) {
    int usable[16], usable_count = 0;
    for (int source = 0; source < source_count; ++source) {
        size_t matching = 0;
        const size_t pixels = (size_t)sources[source].width * sources[source].height;
        for (size_t p = 0; p < pixels; ++p)
            matching += sources[source].labels[p] == (uint32_t)kind;
        if ((double)matching / pixels >= .5) usable[usable_count++] = source;
    }
    if (!usable_count) return false;
    const size_t output_pixels = (size_t)width * height;
    memset(exemplar, 0, output_pixels * 3 * sizeof(float));
    uint8_t *filled = calloc(output_pixels, 1);
    const size_t patch_values = (size_t)EX_PATCH * EX_PATCH * 3;
    float *crops = malloc(EX_BATCH * patch_values * sizeof(float));
    float *smooth_2 = malloc(EX_BATCH * patch_values * sizeof(float));
    float *smooth_12 = malloc(EX_BATCH * patch_values * sizeof(float));
    float *cost = malloc((size_t)EX_PATCH * EX_PATCH * sizeof(float));
    uint8_t *known = malloc((size_t)EX_PATCH * EX_PATCH);
    uint8_t *take = malloc((size_t)EX_PATCH * EX_PATCH);
    if (!filled || !crops || !smooth_2 || !smooth_12 || !cost || !known || !take)
        return false;
    int origins[8], origin_count = 0;
    for (int value = 0; value <= width - EX_PATCH; value += EX_PATCH - 20)
        origins[origin_count++] = value;
    if (!origin_count || origins[origin_count - 1] != width - EX_PATCH)
        origins[origin_count++] = width - EX_PATCH;
    for (int yi = 0; yi < origin_count; ++yi) {
        for (int xi = 0; xi < origin_count; ++xi) {
            float purity[EX_BATCH];
            int accepted = 0;
            for (int attempt = 0;
                 attempt < EX_MAX_ATTEMPTS && accepted < EX_BATCH; ++attempt) {
                const int source = usable[bake_pcg64_bounded(
                    rng, 0, (uint32_t)usable_count)];
                const int sy = (int)bake_pcg64_bounded(
                    rng, 0, (uint32_t)(sources[source].height - EX_PATCH + 1));
                const int sx = (int)bake_pcg64_bounded(
                    rng, 0, (uint32_t)(sources[source].width - EX_PATCH + 1));
                size_t matching = 0;
                for (int y = 0; y < EX_PATCH; ++y)
                    for (int x = 0; x < EX_PATCH; ++x)
                        matching += sources[source].labels[
                            (size_t)(sy + y) * sources[source].width + sx + x] ==
                            (uint32_t)kind;
                const float candidate_purity =
                    (float)((double)matching / (EX_PATCH * EX_PATCH));
                if (candidate_purity < exemplar_min_purity(kind) ||
                    largest_foreign_component(&sources[source], kind, sx, sy) >
                        exemplar_max_foreign_component(kind))
                    continue;
                float *crop = crops + (size_t)accepted * patch_values;
                for (int y = 0; y < EX_PATCH; ++y) {
                    for (int x = 0; x < EX_PATCH; ++x) {
                        const size_t source_pixel =
                            (size_t)(sy + y) * sources[source].width + sx + x;
                        const uint32_t pixel = sources[source].rgb[source_pixel];
                        const size_t target = ((size_t)y * EX_PATCH + x) * 3;
                        for (int c = 0; c < 3; ++c)
                            crop[target + c] = byte_channel(pixel, c);
                    }
                }
                if (kind == 1) neutralize_bright_grass(crop, EX_PATCH * EX_PATCH);
                purity[accepted++] = candidate_purity;
            }
            if (!accepted) {
                fprintf(stderr,
                        "material %d: no %.0f%%-pure exemplar crop at %d,%d\n",
                        kind, 100.0f * exemplar_min_purity(kind), origins[xi],
                        origins[yi]);
                return false;
            }
            if (!filter_layers(gpu, crops, smooth_2, EX_PATCH, EX_PATCH, 3,
                               accepted, 2.0f) ||
                !filter_layers(gpu, crops, smooth_12, EX_PATCH, EX_PATCH, 3,
                               accepted, 12.0f))
                return false;
            ExemplarEnergy ranked[EX_BATCH];
            for (int candidate = 0; candidate < accepted; ++candidate) {
                float sum = 0.0f;
                const size_t base = (size_t)candidate * patch_values;
                for (size_t i = 0; i < patch_values; ++i) {
                    const float delta = smooth_2[base + i] - smooth_12[base + i];
                    sum += delta * delta;
                }
                ranked[candidate] = (ExemplarEnergy){
                    sqrtf(sum / (float)patch_values), candidate};
            }
            qsort(ranked, (size_t)accepted, sizeof(ranked[0]),
                  exemplar_energy_compare);
            // The median retains ordinary native texture without systematically
            // selecting the roughest valid crop.
            const int best = ranked[accepted / 2].candidate;
            (void)purity;
            float existing[EX_PATCH * EX_PATCH * 3];
            for (int y = 0; y < EX_PATCH; ++y) {
                for (int x = 0; x < EX_PATCH; ++x) {
                    const size_t global = (size_t)(origins[yi] + y) * width +
                                          origins[xi] + x;
                    const size_t local = (size_t)y * EX_PATCH + x;
                    memcpy(existing + local * 3, exemplar + global * 3,
                           3 * sizeof(float));
                    known[local] = filled[global];
                }
            }
            const float *candidate = crops + (size_t)best * patch_values;
            rgb_seam_cost(existing, candidate, EX_PATCH, cost);
            if (!bake_quilt_take_mask_from_cost(
                    cost, known, EX_PATCH, 20, origins[xi] > 0,
                    origins[yi] > 0, take))
                return false;
            for (int y = 0; y < EX_PATCH; ++y) {
                for (int x = 0; x < EX_PATCH; ++x) {
                    const size_t global = (size_t)(origins[yi] + y) * width +
                                          origins[xi] + x;
                    const size_t local = (size_t)y * EX_PATCH + x;
                    if (take[local])
                        memcpy(exemplar + global * 3, candidate + local * 3,
                               3 * sizeof(float));
                    filled[global] = 1;
                }
            }
        }
    }
    free(take);
    free(known);
    free(cost);
    free(smooth_12);
    free(smooth_2);
    free(crops);
    free(filled);
    return true;
}

static int float_compare(const void *left, const void *right) {
    const float a = *(const float *)left, b = *(const float *)right;
    return (a > b) - (a < b);
}

static float median(float *values, int count) {
    qsort(values, (size_t)count, sizeof(*values), float_compare);
    return count & 1 ? values[count / 2]
                     : .5f * (values[count / 2 - 1] + values[count / 2]);
}

static float rms_difference(const float *first, const float *second,
                            size_t count) {
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float delta = first[i] - second[i];
        sum += delta * delta;
    }
    return sqrtf(sum / (float)count);
}

static float material_rms_difference(const float *first, const float *second,
                                     const uint32_t *labels, int kind,
                                     size_t pixels) {
    double sum = 0.0;
    size_t count = 0;
    for (size_t p = 0; p < pixels; ++p) {
        if (labels[p] != (uint32_t)kind) continue;
        for (int c = 0; c < 3; ++c) {
            const float delta = first[p * 3 + c] - second[p * 3 + c];
            sum += (double)delta * delta;
            ++count;
        }
    }
    return count ? (float)sqrt(sum / (double)count) : 0.0f;
}

static float required_residual_gain(float target, float base, float residual,
                                    float maximum) {
    const float missing_squared = target * target - base * base;
    if (missing_squared <= 0.0f || residual <= 1.0e-5f) return 0.0f;
    return fminf(sqrtf(missing_squared) / residual, maximum);
}

bool bake_pass17_material_relayer_gpu(
    BakeGpu *gpu, const uint32_t *base, const uint32_t *labels,
    const BakeTerrainSample *texture_donors, int texture_donor_count,
    const BakeTerrainSample *material_sources[3],
    const int material_source_counts[3], uint32_t *output,
    int width, int height) {
    if (!gpu || !base || !labels || !texture_donors || texture_donor_count <= 0 ||
        !output || width != 256 || height != 256)
        return false;
    const size_t pixels = (size_t)width * height;
    const size_t values = pixels * 3;
    const size_t donor_values = (size_t)texture_donor_count * values;
    float *donor_rgb = malloc(donor_values * sizeof(float));
    float *donor_2 = malloc(donor_values * sizeof(float));
    float *donor_12 = malloc(donor_values * sizeof(float));
    float *donor_32 = malloc(donor_values * sizeof(float));
    if (!donor_rgb || !donor_2 || !donor_12 || !donor_32) return false;
    for (int donor = 0; donor < texture_donor_count; ++donor) {
        for (size_t p = 0; p < pixels; ++p) {
            for (int c = 0; c < 3; ++c)
                donor_rgb[((size_t)donor * pixels + p) * 3 + c] =
                    byte_channel(texture_donors[donor].rgb[p], c);
        }
    }
    if (!filter_layers(gpu, donor_rgb, donor_2, width, height, 3,
                       texture_donor_count, 2.0f) ||
        !filter_layers(gpu, donor_rgb, donor_12, width, height, 3,
                       texture_donor_count, 12.0f) ||
        !filter_layers(gpu, donor_rgb, donor_32, width, height, 3,
                       texture_donor_count, 32.0f))
        return false;
    float *mid = calloc(values, sizeof(float));
    float *meso = calloc(values, sizeof(float));
    float *fine = calloc(values, sizeof(float));
    float *exemplars = calloc(MATERIALS * values, sizeof(float));
    float *exemplar_2 = malloc(MATERIALS * values * sizeof(float));
    float *exemplar_12 = malloc(MATERIALS * values * sizeof(float));
    float *exemplar_32 = malloc(MATERIALS * values * sizeof(float));
    float *masks = malloc(MATERIALS * pixels * sizeof(float));
    float *mask_mid = malloc(MATERIALS * pixels * sizeof(float));
    float *mask_meso = malloc(MATERIALS * pixels * sizeof(float));
    if (!mid || !meso || !fine || !exemplars || !exemplar_2 || !exemplar_12 ||
        !exemplar_32 || !masks || !mask_mid || !mask_meso)
        return false;
    BakePcg64 rng;
    if (!bake_pcg64_pass17_seed(&rng, 1515)) return false;
    int active_material[MATERIALS] = {0};
    for (size_t p = 0; p < pixels; ++p)
        if (labels[p] < MATERIALS) active_material[labels[p]] = 1;
    for (int kind = 0; kind < MATERIALS; ++kind) {
        if (active_material[kind] &&
            !build_material_exemplar(
                gpu, material_sources[kind], material_source_counts[kind], kind,
                &rng, width, height, exemplars + (size_t)kind * values))
            return false;
        for (size_t p = 0; p < pixels; ++p)
            masks[(size_t)kind * pixels + p] = labels[p] == (uint32_t)kind;
    }
    if (!filter_layers(gpu, exemplars, exemplar_2, width, height, 3,
                       MATERIALS, 2.0f) ||
        !filter_layers(gpu, exemplars, exemplar_12, width, height, 3,
                       MATERIALS, 12.0f) ||
        !filter_layers(gpu, exemplars, exemplar_32, width, height, 3,
                       MATERIALS, 32.0f) ||
        !filter_layers(gpu, masks, mask_mid, width, height, 1,
                       MATERIALS, 1.0f) ||
        !filter_layers(gpu, masks, mask_meso, width, height, 1,
                       MATERIALS, 3.0f))
        return false;
    float material_colours[3][3];
    float colour_keep[3];
    for (int kind = 0; kind < MATERIALS; ++kind) {
        if (!active_material[kind]) {
            colour_keep[kind] = 0.0f;
            memset(material_colours[kind], 0, sizeof(material_colours[kind]));
            continue;
        }
        const float *exemplar = exemplars + (size_t)kind * values;
        double colour_sum[3] = {0};
        for (size_t p = 0; p < pixels; ++p)
            for (int c = 0; c < 3; ++c) colour_sum[c] += exemplar[p * 3 + c];
        const float maximum = fmaxf((float)colour_sum[0],
                                    fmaxf((float)colour_sum[1],
                                          (float)colour_sum[2]));
        const float minimum = fminf((float)colour_sum[0],
                                    fminf((float)colour_sum[1],
                                          (float)colour_sum[2]));
        colour_keep[kind] = fminf(fmaxf(((maximum - minimum) /
                                         fmaxf(maximum, 1.0e-5f) - .05f) * 3.5f,
                                        .10f),
                                  1.0f);
        for (int c = 0; c < 3; ++c) {
            float means[64];
            int count = 0;
            for (int donor = 0; donor < texture_donor_count; ++donor) {
                double sum = 0.0;
                int matching = 0;
                for (size_t p = 0; p < pixels; ++p) {
                    if (texture_donors[donor].labels[p] != (uint32_t)kind) continue;
                    sum += byte_channel(texture_donors[donor].rgb[p], c);
                    ++matching;
                }
                if (matching >= 256) means[count++] = (float)(sum / matching);
            }
            if (!count) {
                for (int donor = 0; donor < texture_donor_count; ++donor) {
                    double sum = 0.0;
                    for (size_t p = 0; p < pixels; ++p)
                        sum += byte_channel(texture_donors[donor].rgb[p], c);
                    means[count++] = (float)(sum / pixels);
                }
            }
            material_colours[kind][c] = median(means, count);
        }
    }
    float *base_f = malloc(values * sizeof(float));
    float *base_2 = malloc(values * sizeof(float));
    float *base_12 = malloc(values * sizeof(float));
    float *base_32 = malloc(values * sizeof(float));
    float *colour_grade = calloc(values, sizeof(float));
    if (!base_f || !base_2 || !base_12 || !base_32 || !colour_grade) return false;
    for (size_t p = 0; p < pixels; ++p)
        for (int c = 0; c < 3; ++c)
            base_f[p * 3 + c] = byte_channel(base[p], c);
    if (!filter_layers(gpu, base_f, base_2, width, height, 3, 1, 2.0f) ||
        !filter_layers(gpu, base_f, base_12, width, height, 3, 1, 12.0f) ||
        !filter_layers(gpu, base_f, base_32, width, height, 3, 1, 32.0f))
        return false;
    for (int kind = 0; kind < MATERIALS; ++kind) {
        if (!active_material[kind]) continue;
        double current[3] = {0};
        int count = 0;
        for (size_t p = 0; p < pixels; ++p) {
            if (labels[p] != (uint32_t)kind) continue;
            for (int c = 0; c < 3; ++c) current[c] += base_12[p * 3 + c];
            ++count;
        }
        for (int c = 0; c < 3; ++c) {
            const float mean = count ? (float)(current[c] / count)
                                     : material_colours[kind][c];
            const float shift = fminf(fmaxf(material_colours[kind][c] - mean,
                                            -48.0f),
                                      48.0f);
            for (size_t p = 0; p < pixels; ++p)
                colour_grade[p * 3 + c] +=
                    mask_meso[(size_t)kind * pixels + p] * shift;
        }
    }
    float gains[MATERIALS][3] = {{0}};
    for (int kind = 0; kind < MATERIALS; ++kind) {
        if (!active_material[kind]) continue;
        float target_mid[64], target_meso[64], target_fine[64];
        int target_count = 0;
        for (int donor = 0; donor < texture_donor_count; ++donor) {
            size_t matching = 0;
            for (size_t p = 0; p < pixels; ++p)
                matching += texture_donors[donor].labels[p] == (uint32_t)kind;
            if (matching < 256) continue;
            const size_t offset = (size_t)donor * values;
            target_mid[target_count] = material_rms_difference(
                donor_2 + offset, donor_12 + offset,
                texture_donors[donor].labels, kind, pixels);
            target_meso[target_count] = material_rms_difference(
                donor_12 + offset, donor_32 + offset,
                texture_donors[donor].labels, kind, pixels);
            target_fine[target_count] = material_rms_difference(
                donor_rgb + offset, donor_2 + offset,
                texture_donors[donor].labels, kind, pixels);
            ++target_count;
        }
        if (!target_count) continue;
        const float base_mid = material_rms_difference(
            base_2, base_12, labels, kind, pixels);
        const float base_meso = material_rms_difference(
            base_12, base_32, labels, kind, pixels);
        const float base_fine = material_rms_difference(
            base_f, base_2, labels, kind, pixels);
        const size_t offset = (size_t)kind * values;
        const float exemplar_mid = rms_difference(
            exemplar_2 + offset, exemplar_12 + offset, values);
        const float exemplar_meso = rms_difference(
            exemplar_12 + offset, exemplar_32 + offset, values);
        const float exemplar_fine = rms_difference(
            exemplars + offset, exemplar_2 + offset, values);
        gains[kind][0] = required_residual_gain(
            median(target_mid, target_count), base_mid, exemplar_mid, 1.0f);
        gains[kind][1] = required_residual_gain(
            median(target_meso, target_count), base_meso, exemplar_meso, 1.0f);
        gains[kind][2] = required_residual_gain(
            median(target_fine, target_count), base_fine, exemplar_fine,
            exemplar_fine_cap(kind));
        printf("    relayer material=%d gains mid=%.3f meso=%.3f fine=%.3f\n",
               kind, gains[kind][0], gains[kind][1], gains[kind][2]);
    }
    for (int kind = 0; kind < MATERIALS; ++kind) {
        for (size_t p = 0; p < pixels; ++p) {
            float bands[3][3];
            for (int c = 0; c < 3; ++c) {
                const size_t q = ((size_t)kind * pixels + p) * 3 + c;
                bands[0][c] = exemplar_2[q] - exemplar_12[q];
                bands[1][c] = exemplar_12[q] - exemplar_32[q];
                bands[2][c] = exemplars[q] - exemplar_2[q];
            }
            for (int band = 0; band < 3; ++band) {
                const float mean = (bands[band][0] + bands[band][1] +
                                    bands[band][2]) / 3.0f;
                for (int c = 0; c < 3; ++c)
                    bands[band][c] = mean +
                        colour_keep[kind] * (bands[band][c] - mean);
            }
            const float mid_weight = mask_mid[(size_t)kind * pixels + p];
            const float meso_weight = mask_meso[(size_t)kind * pixels + p];
            const float fine_weight = masks[(size_t)kind * pixels + p];
            for (int c = 0; c < 3; ++c) {
                mid[p * 3 + c] += mid_weight * gains[kind][0] * bands[0][c];
                meso[p * 3 + c] += meso_weight * gains[kind][1] * bands[1][c];
                fine[p * 3 + c] += fine_weight * gains[kind][2] * bands[2][c];
            }
        }
    }
    for (size_t p = 0; p < pixels; ++p) {
        uint32_t packed = 0xff000000u;
        for (int c = 0; c < 3; ++c) {
            float value = base_f[p * 3 + c] + .80f * colour_grade[p * 3 + c] +
                          mid[p * 3 + c] + meso[p * 3 + c] +
                          fine[p * 3 + c];
            if (value < 0.0f) value = 0.0f;
            if (value > 255.0f) value = 255.0f;
            packed |= (uint32_t)(uint8_t)value << (c * 8);
        }
        output[p] = packed;
    }
    free(colour_grade);
    free(base_32);
    free(base_12);
    free(base_2);
    free(base_f);
    free(mask_meso);
    free(mask_mid);
    free(masks);
    free(exemplar_32);
    free(exemplar_12);
    free(exemplar_2);
    free(exemplars);
    free(fine);
    free(meso);
    free(mid);
    free(donor_32);
    free(donor_12);
    free(donor_2);
    free(donor_rgb);
    return true;
}
