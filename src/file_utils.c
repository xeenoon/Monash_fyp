#include "file_utils.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

FileReadResult file_read_all(const char *path, uint8_t **data_out,
                             size_t *size_out) {
    if (!path || !data_out || !size_out) return FILE_READ_INVALID_ARGUMENT;
    *data_out = NULL;
    *size_out = 0;

    FILE *file = fopen(path, "rb");
    if (!file) return FILE_READ_OPEN_FAILED;
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return FILE_READ_SEEK_FAILED;
    }
    long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return FILE_READ_SEEK_FAILED;
    }
    if ((uintmax_t)length > SIZE_MAX) {
        fclose(file);
        return FILE_READ_TOO_LARGE;
    }

    size_t size = (size_t)length;
    uint8_t *data = malloc(size ? size : 1u);
    if (!data) {
        fclose(file);
        return FILE_READ_OUT_OF_MEMORY;
    }
    if (fread(data, 1, size, file) != size) {
        free(data);
        fclose(file);
        return FILE_READ_FAILED;
    }
    fclose(file);
    *data_out = data;
    *size_out = size;
    return FILE_READ_OK;
}

const char *file_read_result_string(FileReadResult result) {
    switch (result) {
    case FILE_READ_OK: return "success";
    case FILE_READ_INVALID_ARGUMENT: return "invalid argument";
    case FILE_READ_OPEN_FAILED: return "could not open file";
    case FILE_READ_SEEK_FAILED: return "could not determine file size";
    case FILE_READ_TOO_LARGE: return "file is too large";
    case FILE_READ_OUT_OF_MEMORY: return "out of memory";
    case FILE_READ_FAILED: return "could not read file";
    }
    return "unknown file-read error";
}
