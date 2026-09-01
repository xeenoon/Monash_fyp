#include "bake_random.h"

#include <stddef.h>

typedef struct {
    uint64_t seed, state_hi, state_lo, increment_hi, increment_lo;
} Pass17Seed;

// Values are NumPy PCG64 states immediately after default_rng(seed). NumPy's
// next_uint64 advances once before applying XSL-RR, which bake_pcg64_raw mirrors.
static const Pass17Seed seeds[] = {
    {1515u,   0xdad9219c0ba267fdULL, 0x19024505c006ded6ULL,
              0xc758c75969664668ULL, 0xfb54dcbfd590bf45ULL},
    {1616u,   0xfa039e77566876ecULL, 0x178b2a3d7d42c20bULL,
              0x18268053846ab153ULL, 0xef7028681ee11575ULL},
    {1717u,   0x173f973b52f0e5dcULL, 0x713cd1dac9afc22dULL,
              0x2b90fd98696ed2e5ULL, 0xc823cc3a014089b9ULL},
    {1818u,   0x27643e4f06adbe12ULL, 0xd6778ff5cc2a7de0ULL,
              0xf0fe0ce9c71ca44bULL, 0x85ff86861fe3e4b7ULL},
    {1919u,   0x597aebf3b75d7430ULL, 0xb5d5306702de048cULL,
              0x7d6ce07b6105aa5aULL, 0x39e2a1c40fdeb9b9ULL},
    {151500u, 0xb35abae2447f1ac2ULL, 0xe47520e90e044db9ULL,
              0x743ce8466a9abc05ULL, 0xfdc2c1c02c7a03f5ULL},
};

static void multiply64(uint64_t a, uint64_t b, uint64_t *hi, uint64_t *lo) {
    uint64_t a0 = (uint32_t)a, a1 = a >> 32;
    uint64_t b0 = (uint32_t)b, b1 = b >> 32;
    uint64_t p0 = a0 * b0, p1 = a0 * b1;
    uint64_t p2 = a1 * b0, p3 = a1 * b1;
    uint64_t middle = (p0 >> 32) + (uint32_t)p1 + (uint32_t)p2;
    *lo = (p0 & 0xffffffffULL) | (middle << 32);
    *hi = p3 + (p1 >> 32) + (p2 >> 32) + (middle >> 32);
}

bool bake_pcg64_pass17_seed(BakePcg64 *rng, uint64_t seed) {
    if (!rng) return false;
    for (size_t i = 0; i < sizeof(seeds) / sizeof(seeds[0]); ++i) {
        if (seeds[i].seed != seed) continue;
        rng->state_hi = seeds[i].state_hi;
        rng->state_lo = seeds[i].state_lo;
        rng->increment_hi = seeds[i].increment_hi;
        rng->increment_lo = seeds[i].increment_lo;
        rng->cached_u32 = 0;
        rng->has_cached_u32 = false;
        return true;
    }
    return false;
}

static uint64_t rotate_right(uint64_t value, unsigned rotation) {
    rotation &= 63u;
    return (value >> rotation) | (value << ((0u - rotation) & 63u));
}

uint64_t bake_pcg64_raw(BakePcg64 *rng) {
    // PCG_DEFAULT_MULTIPLIER_128 =
    // 0x2360ed051fc65da44385df649fccf645, modulo 2^128.
    const uint64_t multiplier_hi = 0x2360ed051fc65da4ULL;
    const uint64_t multiplier_lo = 0x4385df649fccf645ULL;
    uint64_t product_hi, product_lo;
    multiply64(rng->state_lo, multiplier_lo, &product_hi, &product_lo);
    product_hi += rng->state_hi * multiplier_lo;
    product_hi += rng->state_lo * multiplier_hi;
    uint64_t old_lo = product_lo;
    product_lo += rng->increment_lo;
    product_hi += rng->increment_hi + (product_lo < old_lo);
    rng->state_hi = product_hi;
    rng->state_lo = product_lo;
    return rotate_right(product_hi ^ product_lo, (unsigned)(product_hi >> 58));
}

uint32_t bake_pcg64_u32(BakePcg64 *rng) {
    if (rng->has_cached_u32) {
        rng->has_cached_u32 = false;
        return rng->cached_u32;
    }
    uint64_t raw = bake_pcg64_raw(rng);
    rng->cached_u32 = (uint32_t)(raw >> 32);
    rng->has_cached_u32 = true;
    return (uint32_t)raw;
}

uint32_t bake_pcg64_bounded(BakePcg64 *rng, uint32_t low, uint32_t high) {
    if (high <= low) return low;
    uint32_t range = high - low;
    // Lemire's method, as used by NumPy's bounded_uint32 implementation.
    uint64_t product = (uint64_t)bake_pcg64_u32(rng) * range;
    uint32_t leftover = (uint32_t)product;
    if (leftover < range) {
        uint32_t threshold = (uint32_t)(0u - range) % range;
        while (leftover < threshold) {
            product = (uint64_t)bake_pcg64_u32(rng) * range;
            leftover = (uint32_t)product;
        }
    }
    return low + (uint32_t)(product >> 32);
}

double bake_pcg64_double(BakePcg64 *rng) {
    return (double)(bake_pcg64_raw(rng) >> 11) * 0x1.0p-53;
}
