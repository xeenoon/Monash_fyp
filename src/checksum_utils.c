#include "checksum_utils.h"

#include <limits.h>
#include <zlib.h>

uint32_t checksum_crc32(const void *data, size_t size)
{
	const Bytef *cursor = data;
	uLong checksum = crc32(0L, Z_NULL, 0);
	while (size)
	{
		uInt chunk = size > UINT_MAX ? UINT_MAX : (uInt)size;
		checksum = crc32(checksum, cursor, chunk);
		cursor += chunk;
		size -= chunk;
	}
	return (uint32_t)checksum;
}
