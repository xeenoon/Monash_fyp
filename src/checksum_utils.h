#pragma once

#include <stddef.h>
#include <stdint.h>

/* Standard IEEE CRC-32, delegated to the system zlib implementation. */
uint32_t checksum_crc32(const void *data, size_t size);
