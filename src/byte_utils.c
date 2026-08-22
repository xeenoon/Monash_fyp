#include "byte_utils.h"

#include <string.h>

uint16_t byte_read_u16_le(const void *bytes) {
    const uint8_t *p = bytes;
    return (uint16_t)((uint16_t)p[0] | (uint16_t)p[1] << 8);
}

uint32_t byte_read_u32_le(const void *bytes) {
    const uint8_t *p = bytes;
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

uint64_t byte_read_u64_le(const void *bytes) {
    const uint8_t *p = bytes;
    return (uint64_t)byte_read_u32_le(p) |
           (uint64_t)byte_read_u32_le(p + 4) << 32;
}

float byte_read_f32_le(const void *bytes) {
    uint32_t bits = byte_read_u32_le(bytes);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

double byte_read_f64_le(const void *bytes) {
    uint64_t bits = byte_read_u64_le(bytes);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}
