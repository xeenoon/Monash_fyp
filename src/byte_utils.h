#pragma once

#include <stdint.h>

/* Decode fixed-width little-endian values from possibly unaligned bytes.
   These functions are host-endian independent. */
uint16_t byte_read_u16_le(const void *bytes);
uint32_t byte_read_u32_le(const void *bytes);
uint64_t byte_read_u64_le(const void *bytes);
float byte_read_f32_le(const void *bytes);
double byte_read_f64_le(const void *bytes);
