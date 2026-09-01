// NumPy default_rng/PCG64 compatibility for pass17's fixed showcase seeds.
#ifndef BAKE_RANDOM_H
#define BAKE_RANDOM_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    uint64_t state_hi, state_lo;
    uint64_t increment_hi, increment_lo;
    uint32_t cached_u32;
    bool has_cached_u32;
} BakePcg64;

// Initializes one of the pass17 production streams (1515 + preset*101 or
// 151500). Returns false for an unknown seed rather than silently diverging.
bool bake_pcg64_pass17_seed(BakePcg64 *rng, uint64_t seed);
uint64_t bake_pcg64_raw(BakePcg64 *rng);
uint32_t bake_pcg64_u32(BakePcg64 *rng);
uint32_t bake_pcg64_bounded(BakePcg64 *rng, uint32_t low, uint32_t high);
double bake_pcg64_double(BakePcg64 *rng);

#endif
