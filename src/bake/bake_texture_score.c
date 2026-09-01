#include "bake_texture_score.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

enum { THUMB = 8, MAX_ANCHORS = 11 };

typedef struct {
    float value;
    int index;
} RankedValue;

static int ranked_compare(const void *left, const void *right) {
    const RankedValue *a = left;
    const RankedValue *b = right;
    if (a->value < b->value) return -1;
    if (a->value > b->value) return 1;
    return a->index - b->index;
}

static float srgb_linear(float value) {
    return value <= 0.04045f ? value / 12.92f
                             : powf((value + 0.055f) / 1.055f, 2.4f);
}

static float lab_function(float value) {
    const float delta = 6.0f / 29.0f;
    return value > delta * delta * delta
               ? cbrtf(value)
               : value / (3.0f * delta * delta) + 4.0f / 29.0f;
}

static void pixel_lab(uint32_t pixel, float result[3]) {
    const float red = srgb_linear((float)(pixel & 255u) / 255.0f);
    const float green = srgb_linear((float)((pixel >> 8) & 255u) / 255.0f);
    const float blue = srgb_linear((float)((pixel >> 16) & 255u) / 255.0f);
    const float x = (red * .4124564f + green * .3575761f + blue * .1804375f) /
                    .95047f;
    const float y = red * .2126729f + green * .7151522f + blue * .0721750f;
    const float z = (red * .0193339f + green * .1191920f + blue * .9503041f) /
                    1.08883f;
    const float fx = lab_function(x), fy = lab_function(y), fz = lab_function(z);
    result[0] = 116.0f * fy - 16.0f;
    result[1] = 500.0f * (fx - fy);
    result[2] = 200.0f * (fy - fz);
}

static float channel(uint32_t pixel, int component) {
    return (float)((pixel >> (component * 8)) & 255u) / 255.0f;
}

static float luma(uint32_t pixel) {
    return channel(pixel, 0) * .2126f + channel(pixel, 1) * .7152f +
           channel(pixel, 2) * .0722f;
}

static int rounded_axis(int value, int extent) {
    return (int)lrintf((float)value * (float)(extent - 1) / 7.0f);
}

static float axial_error(float first, float second) {
    return 0.5f * fabsf(atan2f(sinf(2.0f * (first - second)),
                              cosf(2.0f * (first - second))));
}

static float semantic_patch_error(const BakeTerrainSample *donor,
                                  const uint32_t *target_labels, int width,
                                  int target_x, int target_y, int source_x,
                                  int source_y, int patch,
                                  const float density[3]) {
    float weights[3];
    for (int kind = 0; kind < 3; ++kind)
        weights[kind] = 1.0f / sqrtf(fmaxf(density[kind], 0.05f));
    double mismatch = 0.0, total = 0.0;
    for (int y = 0; y < patch; ++y) {
        for (int x = 0; x < patch; ++x) {
            const uint32_t target =
                target_labels[(size_t)(target_y + y) * width + target_x + x];
            const uint32_t source = donor->labels[
                (size_t)(source_y + y) * donor->width + source_x + x];
            const float weight = weights[target < 3 ? target : 0];
            total += weight;
            if (source != target) mismatch += weight;
        }
    }
    return (float)(mismatch / fmax(total, 1.0e-6));
}

typedef struct {
    int donor;
    float expected_x, expected_y;
} CoherenceAnchor;

static int coherence_anchors(const BakeTextureScoreInput *input, int target_x,
                             int target_y, int patch, int level,
                             CoherenceAnchor anchors[MAX_ANCHORS]) {
    static const int fractions[5] = {0, 1, 2, 3, 4};
    int count = 0;
    for (int i = 0; i < 5; ++i) {
        const int local = (int)lrintf((float)fractions[i] * (patch - 1) / 4.0f);
        if (target_x > 0) {
            const int y = target_y + local, x = target_x - 1;
            const size_t p = (size_t)y * input->width + x;
            if (input->known[p]) {
                anchors[count++] = (CoherenceAnchor){
                    input->donor_map[p], (float)input->source_x[p] + 1.0f,
                    (float)input->source_y[p] - local};
            }
        }
    }
    for (int i = 0; i < 5; ++i) {
        const int local = (int)lrintf((float)fractions[i] * (patch - 1) / 4.0f);
        if (target_y > 0) {
            const int y = target_y - 1, x = target_x + local;
            const size_t p = (size_t)y * input->width + x;
            if (input->known[p]) {
                anchors[count++] = (CoherenceAnchor){
                    input->donor_map[p], (float)input->source_x[p] - local,
                    (float)input->source_y[p] + 1.0f};
            }
        }
    }
    if (level > 0) {
        const int y = target_y + patch / 2, x = target_x + patch / 2;
        const size_t p = (size_t)y * input->width + x;
        if (input->known[p]) {
            anchors[count++] = (CoherenceAnchor){
                input->donor_map[p], (float)input->source_x[p] - patch / 2,
                (float)input->source_y[p] - patch / 2};
        }
    }
    return count;
}

static float coherence_for(const BakeResidualCandidate *candidate,
                           const CoherenceAnchor *anchors, int anchor_count,
                           int patch) {
    if (!anchor_count) return 0.0f;
    float values[MAX_ANCHORS];
    for (int i = 0; i < anchor_count; ++i) {
        const float dx = candidate->source_x - anchors[i].expected_x;
        const float dy = candidate->source_y - anchors[i].expected_y;
        values[i] = fminf(hypotf(dx, dy) / fmaxf((float)patch, 1.0f), 3.0f) +
                    1.5f * (candidate->donor != anchors[i].donor);
    }
    for (int i = 1; i < anchor_count; ++i) {
        const float value = values[i];
        int j = i;
        while (j > 0 && values[j - 1] > value) {
            values[j] = values[j - 1];
            --j;
        }
        values[j] = value;
    }
    const int keep = (anchor_count + 1) / 2;
    float sum = 0.0f;
    for (int i = 0; i < keep; ++i) sum += values[i];
    return sum / keep;
}

static float boundary_descriptor_for(const BakeTextureDatabase *database,
                                     const BakeTextureScoreInput *input,
                                     int candidate, int target_x, int target_y,
                                     int patch) {
    const int band = patch / 3 < 16 ? patch / 3 : 16;
    float total = 0.0f;
    int terms = 0;
    if (target_y == 0) {
        float context_sum = 0.0f, edge_sum = 0.0f, gradient_sum = 0.0f;
        for (int cy = 0; cy < 4; ++cy) {
            const int by = (int)lrintf((float)cy * (band - 1) / 3.0f);
            for (int ax = 0; ax < THUMB; ++ax) {
                const int px = rounded_axis(ax, patch);
                const uint32_t neighbour = input->north->rgb[
                    (size_t)(input->north->height - band + by) *
                        input->north->width +
                    target_x + px];
                for (int c = 0; c < 3; ++c) {
                    const size_t q = ((size_t)candidate * 4 * THUMB +
                                      cy * THUMB + ax) * 3 + c;
                    const float delta =
                        (float)database->north_context[q] / 255.0f -
                        channel(neighbour, c);
                    context_sum += delta * delta;
                }
            }
        }
        for (int ax = 0; ax < THUMB; ++ax) {
            const int px = rounded_axis(ax, patch);
            const uint32_t edge = input->north->rgb[
                (size_t)(input->north->height - 1) * input->north->width +
                target_x + px];
            const uint32_t before = input->north->rgb[
                (size_t)(input->north->height - 2) * input->north->width +
                target_x + px];
            for (int c = 0; c < 3; ++c) {
                const size_t q = ((size_t)candidate * THUMB + ax) * 3 + c;
                float delta = ((float)database->top_edge[q] -
                               (float)((edge >> (c * 8)) & 255u)) /
                              255.0f;
                edge_sum += delta * delta;
                const int neighbour_gradient =
                    (int)((edge >> (c * 8)) & 255u) -
                    (int)((before >> (c * 8)) & 255u);
                delta = ((float)database->top_gradient[q] -
                         (float)neighbour_gradient) /
                        255.0f;
                gradient_sum += delta * delta;
            }
        }
        float term = context_sum / (4.0f * THUMB * 3.0f) / 0.04f +
                     2.0f * edge_sum / (THUMB * 3.0f) / 0.04f +
                     gradient_sum / (THUMB * 3.0f) / 0.02f;
        if (!database->north_valid[candidate]) term = 3.0f;
        total += term;
        ++terms;
    }
    if (target_x + patch == input->width) {
        float context_sum = 0.0f, edge_sum = 0.0f, gradient_sum = 0.0f;
        for (int ay = 0; ay < THUMB; ++ay) {
            const int py = rounded_axis(ay, patch);
            for (int cx = 0; cx < 4; ++cx) {
                const int bx = (int)lrintf((float)cx * (band - 1) / 3.0f);
                const uint32_t neighbour = input->east->rgb[
                    (size_t)(target_y + py) * input->east->width + bx];
                for (int c = 0; c < 3; ++c) {
                    const size_t q = ((size_t)candidate * THUMB * 4 +
                                      ay * 4 + cx) * 3 + c;
                    const float delta =
                        (float)database->east_context[q] / 255.0f -
                        channel(neighbour, c);
                    context_sum += delta * delta;
                }
            }
        }
        for (int ay = 0; ay < THUMB; ++ay) {
            const int py = rounded_axis(ay, patch);
            const uint32_t edge = input->east->rgb[
                (size_t)(target_y + py) * input->east->width];
            const uint32_t after = input->east->rgb[
                (size_t)(target_y + py) * input->east->width + 1];
            for (int c = 0; c < 3; ++c) {
                const size_t q = ((size_t)candidate * THUMB + ay) * 3 + c;
                float delta = ((float)database->right_edge[q] -
                               (float)((edge >> (c * 8)) & 255u)) /
                              255.0f;
                edge_sum += delta * delta;
                const int neighbour_gradient =
                    (int)((after >> (c * 8)) & 255u) -
                    (int)((edge >> (c * 8)) & 255u);
                delta = ((float)database->right_gradient[q] -
                         (float)neighbour_gradient) /
                        255.0f;
                gradient_sum += delta * delta;
            }
        }
        float term = context_sum / (THUMB * 4.0f * 3.0f) / 0.04f +
                     2.0f * edge_sum / (THUMB * 3.0f) / 0.04f +
                     gradient_sum / (THUMB * 3.0f) / 0.02f;
        if (!database->east_valid[candidate]) term = 3.0f;
        total += term;
        ++terms;
    }
    return terms ? total / terms : 0.0f;
}

void bake_texture_descriptor_scores(const BakeTextureDatabase *database,
                                    const BakeTextureScoreInput *input,
                                    int target_x, int target_y, int level,
                                    float *descriptor, float *coherence,
                                    float *room, float target_density[3]) {
    const int patch = database->patch;
    float class_count[3] = {0};
    double height_sum = 0.0, height_squared = 0.0, slope_sum = 0.0;
    double jxx = 0.0, jyy = 0.0, jxy = 0.0;
    for (int y = 0; y < patch; ++y) {
        for (int x = 0; x < patch; ++x) {
            const size_t p = (size_t)(target_y + y) * input->width + target_x + x;
            const int kind = input->target_labels[p] < 3 ? input->target_labels[p] : 0;
            class_count[kind] += 1.0f;
            const double h = input->target->elevation[p];
            height_sum += h;
            height_squared += h * h;
            slope_sum += input->target->slope[p];
            const double gx = input->target->gradient_x[p];
            const double gy = input->target->gradient_y[p];
            jxx += gx * gx;
            jyy += gy * gy;
            jxy += gx * gy;
        }
    }
    const double area = (double)patch * patch;
    for (int kind = 0; kind < 3; ++kind)
        target_density[kind] = class_count[kind] / (float)area;
    const double mean = height_sum / area;
    const float terrain[3] = {
        (float)mean,
        (float)sqrt(fmax(height_squared / area - mean * mean, 0.0)),
        (float)(slope_sum / area)};
    jxx /= area;
    jyy /= area;
    jxy /= area;
    const float target_angle = .5f * (float)atan2(2.0 * jxy, jxx - jyy);
    const float target_anisotropy =
        (float)(hypot(jxx - jyy, 2.0 * jxy) / fmax(jxx + jyy, 1.0e-6));
    float class_weight[3];
    for (int kind = 0; kind < 3; ++kind)
        class_weight[kind] = 1.0f / sqrtf(fmaxf(target_density[kind], 0.05f));
    float thumb_total = 0.0f;
    uint32_t target_thumb[64];
    for (int y = 0; y < THUMB; ++y) {
        for (int x = 0; x < THUMB; ++x) {
            const int py = target_y + rounded_axis(y, patch);
            const int px = target_x + rounded_axis(x, patch);
            const uint32_t kind = input->target_labels[(size_t)py * input->width + px];
            target_thumb[y * THUMB + x] = kind;
            thumb_total += class_weight[kind < 3 ? kind : 0];
        }
    }
    CoherenceAnchor anchors[MAX_ANCHORS];
    const int anchor_count = coherence_anchors(input, target_x, target_y, patch,
                                               level, anchors);
    for (int i = 0; i < database->count; ++i) {
        float mismatch = 0.0f;
        for (int p = 0; p < 64; ++p) {
            const int kind = target_thumb[p] < 3 ? (int)target_thumb[p] : 0;
            if (database->thumbnail[(size_t)i * 64 + p] != target_thumb[p])
                mismatch += class_weight[kind];
        }
        const float semantic = mismatch / fmaxf(thumb_total, 1.0e-6f);
        float density_error = 0.0f;
        for (int kind = 0; kind < 3; ++kind) {
            const float delta = database->density[i * 3 + kind] - target_density[kind];
            density_error += delta * delta;
        }
        density_error /= 3.0f;
        const float terrain_scale[3] = {100.0f, 50.0f, 0.20f};
        float terrain_error = 0.0f;
        for (int k = 0; k < 3; ++k) {
            const float delta = (database->terrain[i * 3 + k] - terrain[k]) /
                                terrain_scale[k];
            terrain_error += delta * delta;
        }
        terrain_error /= 3.0f;
        float orientation = 2.0f * axial_error(database->terrain_angle[i * 2],
                                               target_angle) /
                            (float)M_PI;
        orientation = orientation * orientation *
                      sqrtf(fmaxf(database->terrain_angle[i * 2 + 1] *
                                      target_anisotropy,
                                  0.0f));
        coherence[i] = coherence_for(&database->records[i], anchors,
                                     anchor_count, patch);
        const float base_x = database->records[i].source_x - target_x;
        const float base_y = database->records[i].source_y - target_y;
        const float overflow_x = fmaxf(-base_x, 0.0f) + fmaxf(base_x, 0.0f);
        const float overflow_y = fmaxf(-base_y, 0.0f) + fmaxf(base_y, 0.0f);
        room[i] = (overflow_x + overflow_y) / fmaxf((float)patch, 1.0f);
        const float boundary = boundary_descriptor_for(
            database, input, i, target_x, target_y, patch);
        descriptor[i] = 8.0f * semantic + 1.5f * density_error +
                        .20f * terrain_error + .35f * orientation + coherence[i] +
                        room[i] + 12.0f * boundary;
    }
}

int bake_texture_shortlist(const float *descriptor, const float *coherence,
                           const float *room, int database_count, int level,
                           int *shortlist) {
    if (database_count <= 0) return 0;
    RankedValue *ranked = malloc((size_t)database_count * sizeof(*ranked));
    RankedValue *local = level ? malloc((size_t)database_count * sizeof(*local)) : NULL;
    if (!ranked || (level && !local)) {
        free(local);
        free(ranked);
        return 0;
    }
    for (int i = 0; i < database_count; ++i) {
        ranked[i] = (RankedValue){descriptor[i], i};
        if (local) local[i] = (RankedValue){coherence[i] + .5f * room[i], i};
    }
    qsort(ranked, (size_t)database_count, sizeof(*ranked), ranked_compare);
    int count = 0;
    if (!level) {
        count = database_count < 48 ? database_count : 48;
        for (int i = 0; i < count; ++i) shortlist[i] = ranked[i].index;
    } else {
        qsort(local, (size_t)database_count, sizeof(*local), ranked_compare);
        const int local_count = database_count < 32 ? database_count : 32;
        const int global_count = database_count < 16 ? database_count : 16;
        for (int i = 0; i < local_count; ++i) shortlist[count++] = local[i].index;
        for (int i = 0; i < global_count; ++i) {
            const int candidate = ranked[i].index;
            int found = 0;
            for (int j = 0; j < count; ++j) found |= shortlist[j] == candidate;
            if (!found) shortlist[count++] = candidate;
        }
        // np.unique sorts the concatenated candidate indexes.
        for (int i = 1; i < count; ++i) {
            const int value = shortlist[i];
            int j = i;
            while (j > 0 && shortlist[j - 1] > value) {
                shortlist[j] = shortlist[j - 1];
                --j;
            }
            shortlist[j] = value;
        }
    }
    free(local);
    free(ranked);
    return count;
}

static float region_error(const uint32_t *first, int first_stride,
                          const uint32_t *second, int second_stride, int width,
                          int height, float *gradient_error) {
    double colour = 0.0, gradient = 0.0;
    for (int y = 0; y < height; ++y) {
        const int ym = y > 0 ? y - 1 : y;
        const int yp = y + 1 < height ? y + 1 : y;
        const float ys = y > 0 && y + 1 < height ? .5f : 1.0f;
        for (int x = 0; x < width; ++x) {
            const int xm = x > 0 ? x - 1 : x;
            const int xp = x + 1 < width ? x + 1 : x;
            const float xs = x > 0 && x + 1 < width ? .5f : 1.0f;
            const uint32_t a = first[(size_t)y * first_stride + x];
            const uint32_t b = second[(size_t)y * second_stride + x];
            for (int c = 0; c < 3; ++c) {
                const float delta = channel(a, c) - channel(b, c);
                colour += delta * delta;
            }
            const float agx = (luma(first[(size_t)y * first_stride + xp]) -
                               luma(first[(size_t)y * first_stride + xm])) * xs;
            const float agy = (luma(first[(size_t)yp * first_stride + x]) -
                               luma(first[(size_t)ym * first_stride + x])) * ys;
            const float bgx = (luma(second[(size_t)y * second_stride + xp]) -
                               luma(second[(size_t)y * second_stride + xm])) * xs;
            const float bgy = (luma(second[(size_t)yp * second_stride + x]) -
                               luma(second[(size_t)ym * second_stride + x])) * ys;
            const float dx = agx - bgx, dy = agy - bgy;
            gradient += dx * dx + dy * dy;
        }
    }
    const double pixels = (double)width * height;
    *gradient_error = (float)(gradient / pixels / 0.02);
    return (float)(colour / (pixels * 3.0) / 0.04);
}

static float boundary_full_error(const BakeTextureScoreInput *input,
                                 const BakeResidualCandidate *candidate,
                                 int target_x, int target_y, int patch) {
    const BakeTerrainSample *donor = &input->donors[candidate->donor];
    const int sx = candidate->source_x, sy = candidate->source_y;
    const int band = patch / 3 < 16 ? patch / 3 : 16;
    float errors[8];
    int count = 0;
    if (target_y == 0) {
        double edge = 0.0, gradient = 0.0;
        for (int x = 0; x < patch; ++x) {
            const uint32_t known = input->north->rgb[
                (size_t)(input->north->height - 1) * input->north->width +
                target_x + x];
            const uint32_t before = input->north->rgb[
                (size_t)(input->north->height - 2) * input->north->width +
                target_x + x];
            const uint32_t source = donor->rgb[(size_t)sy * donor->width + sx + x];
            const uint32_t after = donor->rgb[(size_t)(sy + 1) * donor->width + sx + x];
            for (int c = 0; c < 3; ++c) {
                float delta = channel(known, c) - channel(source, c);
                edge += delta * delta;
                delta = (channel(known, c) - channel(before, c)) -
                        (channel(after, c) - channel(source, c));
                gradient += delta * delta;
            }
        }
        errors[count++] = (float)(2.0 * edge / (patch * 3.0) / 0.04);
        errors[count++] = (float)(gradient / (patch * 3.0) / 0.02);
        if (sy < band) {
            errors[count++] = 3.0f;
        } else {
            float gradient_error;
            const uint32_t *known = input->north->rgb +
                (size_t)(input->north->height - band) * input->north->width + target_x;
            const uint32_t *source = donor->rgb + (size_t)(sy - band) * donor->width + sx;
            errors[count++] = region_error(known, input->north->width, source,
                                           donor->width, patch, band,
                                           &gradient_error);
            errors[count++] = gradient_error;
        }
    }
    if (target_x + patch == input->width) {
        double edge = 0.0, gradient = 0.0;
        for (int y = 0; y < patch; ++y) {
            const uint32_t known = input->east->rgb[
                (size_t)(target_y + y) * input->east->width];
            const uint32_t after = input->east->rgb[
                (size_t)(target_y + y) * input->east->width + 1];
            const uint32_t source = donor->rgb[
                (size_t)(sy + y) * donor->width + sx + patch - 1];
            const uint32_t before = donor->rgb[
                (size_t)(sy + y) * donor->width + sx + patch - 2];
            for (int c = 0; c < 3; ++c) {
                float delta = channel(known, c) - channel(source, c);
                edge += delta * delta;
                delta = (channel(after, c) - channel(known, c)) -
                        (channel(source, c) - channel(before, c));
                gradient += delta * delta;
            }
        }
        errors[count++] = (float)(2.0 * edge / (patch * 3.0) / 0.04);
        errors[count++] = (float)(gradient / (patch * 3.0) / 0.02);
        if (sx + patch + band > donor->width) {
            errors[count++] = 3.0f;
        } else {
            // East regions are strided columns, so materialize the small band.
            uint32_t known[256 * 16], source[256 * 16];
            for (int y = 0; y < patch; ++y) {
                memcpy(known + (size_t)y * band,
                       input->east->rgb + (size_t)(target_y + y) * input->east->width,
                       (size_t)band * sizeof(uint32_t));
                memcpy(source + (size_t)y * band,
                       donor->rgb + (size_t)(sy + y) * donor->width + sx + patch,
                       (size_t)band * sizeof(uint32_t));
            }
            float gradient_error;
            errors[count++] = region_error(known, band, source, band, band,
                                           patch, &gradient_error);
            errors[count++] = gradient_error;
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) sum += errors[i];
    return count ? sum / count : 0.0f;
}

void bake_texture_base_scores(const BakeTextureDatabase *database,
                              const BakeTextureScoreInput *input,
                              const float *coherence, const float *room,
                              const float target_density[3], int target_x,
                              int target_y, int level, const int *shortlist,
                              int shortlist_count,
                              BakeResidualCandidate *candidates,
                              float *base_scores) {
    const int patch = database->patch;
    const int overlap = patch - (level == 0 ? 32 : level == 1 ? 16 : 8);
    int donor_counts[64] = {0}, active_id = -1, active_count = 0;
    double lab_sum[3] = {0};
    int lab_count = 0;
    for (int y = 0; y < patch; ++y) {
        for (int x = 0; x < patch; ++x) {
            const int overlap_pixel = (target_x > 0 && x < overlap) ||
                                      (target_y > 0 && y < overlap);
            if (!overlap_pixel) continue;
            const size_t p = (size_t)(target_y + y) * input->width + target_x + x;
            const int donor = input->donor_map[p];
            if (donor >= 0 && donor < input->donor_count) ++donor_counts[donor];
            float lab[3];
            pixel_lab(input->generated[p], lab);
            for (int c = 0; c < 3; ++c) lab_sum[c] += lab[c];
            ++lab_count;
        }
    }
    for (int donor = 0; donor < input->donor_count; ++donor) {
        if (donor_counts[donor] > active_count) {
            active_count = donor_counts[donor];
            active_id = donor;
        }
    }
    float active_position[2] = {0}, active_lab[3] = {0};
    if (active_id >= 0) {
        double sx_sum = 0.0, sy_sum = 0.0;
        int count = 0;
        for (int y = 0; y < patch; ++y) {
            for (int x = 0; x < patch; ++x) {
                const int overlap_pixel = (target_x > 0 && x < overlap) ||
                                          (target_y > 0 && y < overlap);
                if (!overlap_pixel) continue;
                const size_t p = (size_t)(target_y + y) * input->width + target_x + x;
                if (input->donor_map[p] == active_id) {
                    sx_sum += input->source_x[p];
                    sy_sum += input->source_y[p];
                    ++count;
                }
            }
        }
        active_position[0] = input->donors[active_id].tile_x +
                             (float)(sx_sum / count / 256.0);
        active_position[1] = input->donors[active_id].tile_y +
                             (float)(sy_sum / count / 256.0);
    }
    if (lab_count) {
        for (int c = 0; c < 3; ++c) active_lab[c] = (float)(lab_sum[c] / lab_count);
    }
    for (int i = 0; i < shortlist_count; ++i) {
        const int database_index = shortlist[i];
        const BakeResidualCandidate candidate = database->records[database_index];
        const BakeTerrainSample *donor = &input->donors[candidate.donor];
        candidates[i] = candidate;
        const float label = semantic_patch_error(
            donor, input->target_labels, input->width, target_x, target_y,
            candidate.source_x, candidate.source_y, patch, target_density);
        const float boundary = boundary_full_error(input, &candidate, target_x,
                                                   target_y, patch);
        double reuse_sum = 0.0;
        for (int y = 0; y < patch; ++y) {
            for (int x = 0; x < patch; ++x) {
                const size_t target =
                    (size_t)(target_y + y) * input->width + target_x + x;
                const int source_x = candidate.source_x + x;
                const int source_y = candidate.source_y + y;
                const size_t source =
                    ((size_t)candidate.donor * donor->height + source_y) *
                        donor->width +
                    source_x;
                const int same = input->known[target] &&
                                 input->donor_map[target] == candidate.donor &&
                                 input->source_x[target] == source_x &&
                                 input->source_y[target] == source_y;
                int usage = input->source_usage[source] - same;
                if (usage < 0) usage = 0;
                if (usage > 2) usage = 2;
                reuse_sum += usage;
            }
        }
        const float reuse = (float)(reuse_sum / ((double)patch * patch));
        float style = 0.0f;
        if (active_id >= 0) {
            const float dx = database->source_position[database_index * 2] -
                             active_position[0];
            const float dy = database->source_position[database_index * 2 + 1] -
                             active_position[1];
            const float saturated_source = 1.0f - expf(-hypotf(dx, dy) / 2.0f);
            float colour_distance = 0.0f, material_distance = 0.0f;
            for (int c = 0; c < 3; ++c) {
                const float lab_delta = database->mean_lab[database_index * 3 + c] -
                                        active_lab[c];
                colour_distance += lab_delta * lab_delta;
                const float material_delta =
                    database->density[database_index * 3 + c] - target_density[c];
                material_distance += material_delta * material_delta;
            }
            style = .60f * saturated_source +
                    .25f * sqrtf(colour_distance) / 35.0f +
                    .15f * sqrtf(material_distance);
        }
        base_scores[i] = 30.0f * label + 2.5f * coherence[database_index] +
                         4.0f * room[database_index] + 32.0f * boundary +
                         2.0f * reuse + 2.0f * style;
    }
}
