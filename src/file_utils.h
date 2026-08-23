#pragma once

#include <stddef.h>
#include <stdint.h>

typedef enum
{
	FILE_READ_OK = 0,
	FILE_READ_INVALID_ARGUMENT,
	FILE_READ_OPEN_FAILED,
	FILE_READ_SEEK_FAILED,
	FILE_READ_TOO_LARGE,
	FILE_READ_OUT_OF_MEMORY,
	FILE_READ_FAILED,
} FileReadResult;

/* Reads a complete file into owned memory. On failure, outputs are reset to
   NULL/zero. Even an empty successful file returns a freeable data pointer. */
FileReadResult file_read_all(const char *path, uint8_t **data_out, size_t *size_out);
const char *file_read_result_string(FileReadResult result);
