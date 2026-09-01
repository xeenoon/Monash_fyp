// Development driver for the exact pass17 texture stage. Frozen labels and
// macro RGB are stage oracles only; all donor selection, residual extraction,
// scoring, seams, and ordered commits execute in the C/GPU implementation.
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

#include "bake_gpu.h"
#include "bake_image.h"
#include "bake_ordered_quilt.h"
#include "bake_pass17_relayer.h"
#include "bake_pass17_sources.h"
#include "bake_random.h"
#include "bake_sample.h"
#include "bake_texture_database.h"
#include "bake_texture_score.h"

#ifndef BAKE_DATASET_DIR
#define BAKE_DATASET_DIR "."
#endif

enum { SIZE = 256, DONOR_COUNT = 14, MAX_DATABASE = 12600 };

static const char *preset_names[] = {
    "full_snow", "snow_rock", "full_rock", "rock_grass", "full_grass"};

static int make_directory(const char *path) {
#ifdef _WIN32
    return _mkdir(path) == 0 || errno == EEXIST;
#else
    return mkdir(path, 0775) == 0 || errno == EEXIST;
#endif
}

static int join(char *out, size_t capacity, const char *a, const char *b) {
    const int count = snprintf(out, capacity, "%s/%s", a, b);
    return count >= 0 && (size_t)count < capacity;
}

static int parse_preset(const char *name) {
    for (int i = 0; i < 5; ++i)
        if (!strcmp(name, preset_names[i])) return i;
    return -1;
}

static int decode_labels(const BakeImage *image, uint32_t *labels) {
    static const int colours[3][3] = {
        {112, 112, 112}, {48, 142, 57}, {245, 245, 245}};
    if (image->width != SIZE || image->height != SIZE) return 0;
    for (int i = 0; i < SIZE * SIZE; ++i) {
        const uint32_t pixel = image->pixels[i];
        int best = 0, best_error = 1 << 30;
        for (int kind = 0; kind < 3; ++kind) {
            int error = 0;
            for (int channel = 0; channel < 3; ++channel) {
                const int delta = (int)((pixel >> (channel * 8)) & 255u) -
                                  colours[kind][channel];
                error += delta * delta;
            }
            if (error < best_error) {
                best_error = error;
                best = kind;
            }
        }
        labels[i] = (uint32_t)best;
    }
    return 1;
}

static int irregular_origins(int patch, int stride, BakePcg64 *rng,
                             int origins[64]) {
    const int last = SIZE - patch;
    const int low = (int)lroundf(stride * .72f);
    const int high = (int)lroundf(stride * 1.28f) + 1;
    int count = 1;
    origins[0] = 0;
    while (origins[count - 1] < last) {
        const int advance = (int)bake_pcg64_bounded(
            rng, (uint32_t)(low > 1 ? low : 1),
            (uint32_t)(high > 2 ? high : 2));
        int following = origins[count - 1] + advance;
        if (following > last) following = last;
        if (following == origins[count - 1]) break;
        origins[count++] = following;
    }
    return count;
}

static int float_compare(const void *left, const void *right) {
    const float a = *(const float *)left, b = *(const float *)right;
    return (a > b) - (a < b);
}

static uint8_t clipped_byte(float value) {
    if (value <= 0.0f) return 0;
    if (value >= 255.0f) return 255;
    return (uint8_t)value;
}

static void apply_level(uint32_t *generated, float *band,
                        const float *selected_rms, int selection_count,
                        int level) {
    float sum = 0.0f;
    for (int i = 0; i < SIZE * SIZE * 3; ++i) sum += band[i] * band[i];
    const float current_rms = sqrtf(sum / (float)(SIZE * SIZE * 3));
    float *ordered = malloc((size_t)selection_count * sizeof(*ordered));
    memcpy(ordered, selected_rms, (size_t)selection_count * sizeof(*ordered));
    qsort(ordered, (size_t)selection_count, sizeof(*ordered), float_compare);
    float desired_rms;
    if (selection_count & 1)
        desired_rms = ordered[selection_count / 2];
    else
        desired_rms = .5f * (ordered[selection_count / 2 - 1] +
                             ordered[selection_count / 2]);
    free(ordered);
    const float native_gain[3] = {1.15f, 1.10f, 1.00f};
    const float ratio = desired_rms / fmaxf(current_rms, 1.0e-5f);
    const float gain = native_gain[level] * fminf(1.6f, ratio);
    for (int p = 0; p < SIZE * SIZE; ++p) {
        uint32_t result = 0xff000000u;
        for (int channel = 0; channel < 3; ++channel) {
            const uint8_t original = (uint8_t)(generated[p] >> (channel * 8));
            band[p * 3 + channel] *= gain;
            result |= (uint32_t)clipped_byte(original + band[p * 3 + channel]) <<
                      (channel * 8);
        }
        generated[p] = result;
    }
    printf("    band RMS current=%.6f desired=%.6f gain=%.6f\n",
           current_rms, desired_rms, gain);
}

static void compare_level(const char *reference_path, const uint32_t *pixels) {
    BakeImage reference = {0};
    if (!bake_image_load(reference_path, &reference)) {
        fprintf(stderr, "cannot load level oracle %s\n", reference_path);
        return;
    }
    uint64_t absolute = 0;
    size_t exact = 0, count = 0;
    unsigned maximum = 0;
    for (int y = 1; y < SIZE; ++y) {
        for (int x = 0; x < SIZE - 1; ++x) {
            const size_t p = (size_t)y * SIZE + x;
            int pixel_exact = 1;
            for (int c = 0; c < 3; ++c) {
                int delta = (int)((pixels[p] >> (c * 8)) & 255u) -
                            (int)((reference.pixels[p] >> (c * 8)) & 255u);
                if (delta < 0) delta = -delta;
                pixel_exact &= delta == 0;
                absolute += (unsigned)delta;
                if ((unsigned)delta > maximum) maximum = (unsigned)delta;
                ++count;
            }
            exact += pixel_exact;
        }
    }
    printf("    oracle interior exact=%8.4f%% MAE=%8.4f max=%u\n",
           100.0 * (double)exact / ((SIZE - 1.0) * (SIZE - 1.0)),
           (double)absolute / count, maximum);
    bake_image_free(&reference);
}

static void destroy_state(BakeGpu *gpu, BakeOrderedState *state) {
    bake_buffer_destroy(gpu, &state->source_usage);
    bake_buffer_destroy(gpu, &state->source_y);
    bake_buffer_destroy(gpu, &state->source_x);
    bake_buffer_destroy(gpu, &state->donor_map);
    bake_buffer_destroy(gpu, &state->level_known);
    bake_buffer_destroy(gpu, &state->level_band);
}

static int run_preset(BakeGpu *gpu, const char *dataset, const char *reference_root,
                      const char *output_dir, int preset, int level_count) {
    char preset_reference[1024], path[1024], leaf[256];
    if (!join(preset_reference, sizeof(preset_reference), reference_root,
              preset_names[preset]))
        return 0;
    BakeImage labels_image = {0}, macro = {0};
    if (!join(path, sizeof(path), preset_reference, "constrained_patch_classes.png") ||
        !bake_image_load(path, &labels_image) ||
        !join(path, sizeof(path), preset_reference, "macro_low_frequency_filled.png") ||
        !bake_image_load(path, &macro)) {
        fprintf(stderr, "%s: missing pass17 stage oracle\n", preset_names[preset]);
        return 0;
    }
    uint32_t *target_labels = malloc((size_t)SIZE * SIZE * sizeof(*target_labels));
    if (!target_labels || !decode_labels(&labels_image, target_labels) ||
        macro.width != SIZE || macro.height != SIZE)
        return 0;

    BakeTerrainSample target = {0}, north = {0}, east = {0};
    BakeTerrainSample donors[DONOR_COUNT];
    memset(donors, 0, sizeof(donors));
    if (!bake_sample_load(gpu, dataset, 23, 8, false, &target) ||
        !bake_sample_load(gpu, dataset, 23, 7, false, &north) ||
        !bake_sample_load(gpu, dataset, 24, 8, false, &east)) {
        fprintf(stderr, "%s: cannot load target/neighbours\n", preset_names[preset]);
        return 0;
    }
    size_t donor_count = 0;
    const BakeTileCoordinate *coordinates =
        bake_pass17_texture_sources(preset, &donor_count);
    if (!coordinates || donor_count != DONOR_COUNT) return 0;
    for (int i = 0; i < DONOR_COUNT; ++i) {
        printf("  donor %2d: %d/%d\n", i, coordinates[i].x, coordinates[i].y);
        if (!bake_sample_load(gpu, dataset, coordinates[i].x, coordinates[i].y,
                              true, &donors[i])) {
            fprintf(stderr, "cannot load donor %d/%d\n", coordinates[i].x,
                    coordinates[i].y);
            return 0;
        }
    }

    const size_t pixels = (size_t)SIZE * SIZE;
    BakeBuffer atlas = bake_buffer_host(
        gpu, DONOR_COUNT * pixels * sizeof(uint32_t));
    for (int donor = 0; donor < DONOR_COUNT; ++donor)
        memcpy((uint32_t *)atlas.mapped + (size_t)donor * pixels,
               donors[donor].rgb, pixels * sizeof(uint32_t));
    BakeOrderedState state = {
        bake_buffer_host(gpu, pixels * 3 * sizeof(float)),
        bake_buffer_host(gpu, pixels * sizeof(uint32_t)),
        bake_buffer_host(gpu, pixels * sizeof(int32_t)),
        bake_buffer_host(gpu, pixels * sizeof(int32_t)),
        bake_buffer_host(gpu, pixels * sizeof(int32_t)),
        bake_buffer_host(gpu, DONOR_COUNT * pixels * sizeof(int32_t))};
    memset(state.donor_map.mapped, 0xff, pixels * sizeof(int32_t));
    memset(state.source_x.mapped, 0xff, pixels * sizeof(int32_t));
    memset(state.source_y.mapped, 0xff, pixels * sizeof(int32_t));
    memset(state.source_usage.mapped, 0,
           DONOR_COUNT * pixels * sizeof(int32_t));
    uint32_t *known = malloc(pixels * sizeof(*known));
    for (size_t i = 0; i < pixels; ++i) known[i] = 1;
    BakeTextureScoreInput score_input = {
        &target, &north, &east, donors, DONOR_COUNT, target_labels,
        macro.pixels, known, state.donor_map.mapped, state.source_x.mapped,
        state.source_y.mapped, state.source_usage.mapped, SIZE, SIZE};
    float *descriptor = malloc(MAX_DATABASE * sizeof(float));
    float *coherence = malloc(MAX_DATABASE * sizeof(float));
    float *room = malloc(MAX_DATABASE * sizeof(float));
    if (!descriptor || !coherence || !room) return 0;
    BakePcg64 rng;
    if (!bake_pcg64_pass17_seed(&rng, 151500)) return 0;
    static const int patches[3] = {96, 48, 24};
    static const int strides[3] = {32, 16, 8};
    snprintf(leaf, sizeof(leaf), "%s_selections.txt", preset_names[preset]);
    if (!join(path, sizeof(path), output_dir, leaf)) return 0;
    FILE *selection_log = fopen(path, "wb");
    if (!selection_log) return 0;
    for (int level = 0; level < level_count; ++level) {
        const int patch = patches[level], stride = strides[level];
        BakeTextureDatabase database;
        printf("%s: texture level %d patch=%d stride=%d\n",
               preset_names[preset], level + 1, patch, stride);
        if (!bake_texture_database_build(donors, DONOR_COUNT, patch, 8,
                                         &database))
            return 0;
        printf("    database records=%d\n", database.count);
        int origins[64];
        const int origin_count = irregular_origins(patch, stride, &rng, origins);
        const int placement_count = origin_count * origin_count;
        float *selected_rms = malloc((size_t)placement_count * sizeof(float));
        memset(state.level_band.mapped, 0, pixels * 3 * sizeof(float));
        memset(state.level_known.mapped, 0, pixels * sizeof(uint32_t));
        int placement = 0;
        for (int yi = 0; yi < origin_count; ++yi) {
            for (int xi = 0; xi < origin_count; ++xi, ++placement) {
                const int target_y = origins[yi], target_x = origins[xi];
                float target_density[3];
                bake_texture_descriptor_scores(
                    &database, &score_input, target_x, target_y, level,
                    descriptor, coherence, room, target_density);
                int shortlist[48];
                const int shortlist_count = bake_texture_shortlist(
                    descriptor, coherence, room, database.count, level,
                    shortlist);
                BakeResidualCandidate candidates[48];
                float base_scores[48];
                bake_texture_base_scores(
                    &database, &score_input, coherence, room, target_density,
                    target_x, target_y, level, shortlist, shortlist_count,
                    candidates, base_scores);
                int selected = -1;
                if (!bake_ordered_residual_placement_gpu(
                        gpu, &atlas, SIZE, SIZE, target_x, target_y, candidates,
                        base_scores, shortlist_count, patch, stride, level, &rng,
                        &state, SIZE, SIZE, &selected,
                        &selected_rms[placement], NULL))
                    return 0;
                const BakeResidualCandidate winner = candidates[selected];
                fprintf(selection_log,
                        "level=%d placement=%d target=%d,%d db=%d donor=%d "
                        "source=%d,%d\n",
                        level + 1, placement + 1, target_x, target_y,
                        winner.database_index, winner.donor, winner.source_x,
                        winner.source_y);
                if ((placement + 1) % origin_count == 0) {
                    printf("\r    placements %3d/%3d", placement + 1,
                           placement_count);
                    fflush(stdout);
                }
            }
        }
        putchar('\n');
        apply_level(macro.pixels, state.level_band.mapped, selected_rms,
                    placement_count, level);
        free(selected_rms);
        snprintf(leaf, sizeof(leaf), "%s_level_%d.png", preset_names[preset],
                 level + 1);
        if (!join(path, sizeof(path), output_dir, leaf)) return 0;
        BakeImage output = {SIZE, SIZE, macro.pixels};
        if (!bake_image_write_png(path, &output)) return 0;
        snprintf(leaf, sizeof(leaf),
                 "texture_synthesis/level_%d_%dpx_rgb.png", level + 1, patch);
        if (join(path, sizeof(path), preset_reference, leaf))
            compare_level(path, macro.pixels);
        bake_texture_database_free(&database);
    }
    fclose(selection_log);
    if (level_count == 3) {
        BakeTerrainSample material_samples[3][6];
        const BakeTerrainSample *material_sets[3] = {
            material_samples[0], material_samples[1], material_samples[2]};
        int material_counts[3] = {0};
        memset(material_samples, 0, sizeof(material_samples));
        puts("  loading material exemplar sources");
        for (int kind = 0; kind < 3; ++kind) {
            size_t count = 0;
            const BakeTileCoordinate *sources =
                bake_pass17_material_sources(kind, &count);
            if (!sources || count > 6) return 0;
            material_counts[kind] = (int)count;
            for (size_t source = 0; source < count; ++source) {
                if (!bake_sample_load(gpu, dataset, sources[source].x,
                                      sources[source].y, true,
                                      &material_samples[kind][source]))
                    return 0;
            }
        }
        uint32_t *relayered = malloc(pixels * sizeof(*relayered));
        if (!relayered || !bake_pass17_material_relayer_gpu(
                              gpu, macro.pixels, target_labels, donors,
                              DONOR_COUNT, material_sets, material_counts,
                              relayered, SIZE, SIZE))
            return 0;
        memcpy(macro.pixels, relayered, pixels * sizeof(*relayered));
        free(relayered);
        for (int kind = 0; kind < 3; ++kind)
            for (int source = 0; source < material_counts[kind]; ++source)
                bake_sample_free(&material_samples[kind][source]);
        puts("  material microtexture relayer complete");
    }
    snprintf(leaf, sizeof(leaf), "%s.png", preset_names[preset]);
    if (!join(path, sizeof(path), output_dir, leaf)) return 0;
    BakeImage output = {SIZE, SIZE, macro.pixels};
    if (!bake_image_write_png(path, &output)) return 0;
    printf("  wrote texture-stage output %s\n", path);

    free(room);
    free(coherence);
    free(descriptor);
    free(known);
    destroy_state(gpu, &state);
    bake_buffer_destroy(gpu, &atlas);
    for (int i = 0; i < DONOR_COUNT; ++i) bake_sample_free(&donors[i]);
    bake_sample_free(&east);
    bake_sample_free(&north);
    bake_sample_free(&target);
    free(target_labels);
    bake_image_free(&macro);
    bake_image_free(&labels_image);
    return 1;
}

int main(int argc, char **argv) {
    const char *dataset = BAKE_DATASET_DIR;
    const char *reference = NULL;
    const char *output = "pass17_texture_gpu";
    int preset = BAKE_PASS17_FULL_GRASS;
    int level_count = 3;
    double choice_bias = 0.00025;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--dataset") && i + 1 < argc)
            dataset = argv[++i];
        else if (!strcmp(argv[i], "--reference") && i + 1 < argc)
            reference = argv[++i];
        else if (!strcmp(argv[i], "--output") && i + 1 < argc)
            output = argv[++i];
        else if (!strcmp(argv[i], "--preset") && i + 1 < argc)
            preset = parse_preset(argv[++i]);
        else if (!strcmp(argv[i], "--levels") && i + 1 < argc)
            level_count = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--choice-bias") && i + 1 < argc)
            choice_bias = strtod(argv[++i], NULL);
        else {
            fprintf(stderr,
                    "usage: %s --reference PASS17_DIR [--dataset DIR] "
                    "[--output DIR] [--preset NAME] [--levels 1..3] "
                    "[--choice-bias 0..0.01]\n",
                    argv[0]);
            return 2;
        }
    }
    if (!reference || preset < 0 || level_count < 1 || level_count > 3 ||
        choice_bias < 0.0 || choice_bias > 0.01 || !make_directory(output))
        return 2;
    bake_ordered_quilt_set_choice_bias(choice_bias);
    BakeGpu gpu;
    if (!bake_gpu_init(&gpu)) return 1;
    printf("bake_pass17_texture: %s\n", gpu.device_name);
    const int ok = run_preset(&gpu, dataset, reference, output, preset,
                              level_count);
    bake_gpu_destroy(&gpu);
    return ok ? 0 : 1;
}
