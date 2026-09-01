#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "bake_random.h"

int main(void) {
    static const uint64_t raw_expected[] = {
        UINT64_C(10602601213757432165), UINT64_C(16144275291680980389),
        UINT64_C(6172815167445010881), UINT64_C(11173449323747326261),
        UINT64_C(17142356096191657664)};
    static const uint32_t int_expected[] = {
        0,57,48,87,59,33,88,60,62,92,97,42,9,24,83,89,82,67,65,31};
    BakePcg64 rng;
    if (!bake_pcg64_pass17_seed(&rng, 1515)) return 1;
    for (size_t i = 0; i < sizeof(raw_expected) / sizeof(raw_expected[0]); ++i) {
        uint64_t got = bake_pcg64_raw(&rng);
        if (got != raw_expected[i]) {
            fprintf(stderr, "raw %zu: got %" PRIu64 ", expected %" PRIu64 "\n",
                    i, got, raw_expected[i]);
            return 1;
        }
    }
    if (!bake_pcg64_pass17_seed(&rng, 1515)) return 1;
    for (size_t i = 0; i < sizeof(int_expected) / sizeof(int_expected[0]); ++i) {
        uint32_t got = bake_pcg64_bounded(&rng, 0, 100);
        if (got != int_expected[i]) {
            fprintf(stderr, "integer %zu: got %u, expected %u\n",
                    i, got, int_expected[i]);
            return 1;
        }
    }
    static const uint32_t origin_advance[] = {29, 34, 30, 34, 33};
    static const double mixed_double[] = {
        0.09752365953720044, 0.5964647271334528, 0.5937313603842023,
        0.43643292885440454, 0.9124665555256252, 0.18150529282495487,
        0.5955537990171146, 0.6346194436580093, 0.17304925837448637};
    if (!bake_pcg64_pass17_seed(&rng, 151500)) return 1;
    for (size_t i = 0; i < sizeof(origin_advance) / sizeof(origin_advance[0]); ++i) {
        uint32_t got = bake_pcg64_bounded(&rng, 23, 42);
        if (got != origin_advance[i]) {
            fprintf(stderr, "mixed integer %zu: got %u, expected %u\n", i,
                    got, origin_advance[i]);
            return 1;
        }
    }
    for (size_t i = 0; i < sizeof(mixed_double) / sizeof(mixed_double[0]); ++i) {
        double got = bake_pcg64_double(&rng);
        if (got != mixed_double[i]) {
            fprintf(stderr, "mixed double %zu: got %.17g, expected %.17g\n",
                    i, got, mixed_double[i]);
            return 1;
        }
    }
    if (bake_pcg64_pass17_seed(&rng, 999)) return 1;
    puts("bake_random_tests: OK (NumPy PCG64 raw and bounded streams)");
    return 0;
}
