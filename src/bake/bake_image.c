#include "bake_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#include "stb_image.h"

bool bake_image_load(const char *path, BakeImage *image) {
    int width = 0, height = 0, channels = 0;
    stbi_uc *data = stbi_load(path, &width, &height, &channels, 4);
    if (!data) {
        fprintf(stderr, "bake_image: cannot load '%s': %s\n", path,
                stbi_failure_reason());
        return false;
    }
    image->width = width;
    image->height = height;
    image->pixels = malloc((size_t)width * height * sizeof(uint32_t));
    if (!image->pixels) {
        stbi_image_free(data);
        return false;
    }
    memcpy(image->pixels, data, (size_t)width * height * 4);
    stbi_image_free(data);
    return true;
}

bool bake_image_alloc(int width, int height, BakeImage *image) {
    image->width = width;
    image->height = height;
    image->pixels = calloc((size_t)width * height, sizeof(uint32_t));
    return image->pixels != NULL;
}

void bake_image_free(BakeImage *image) {
    free(image->pixels);
    image->pixels = NULL;
    image->width = image->height = 0;
}

bool bake_image_write_ppm(const char *path, const BakeImage *image) {
    FILE *file = fopen(path, "wb");
    if (!file) {
        fprintf(stderr, "bake_image: cannot open '%s' for writing\n", path);
        return false;
    }
    fprintf(file, "P6\n%d %d\n255\n", image->width, image->height);
    size_t count = (size_t)image->width * image->height;
    for (size_t i = 0; i < count; ++i) {
        const uint8_t *rgba = (const uint8_t *)&image->pixels[i];
        fwrite(rgba, 1, 3, file);  // R, G, B (drop A)
    }
    fclose(file);
    return true;
}

static void put_be32(uint8_t out[4], uint32_t value) {
    out[0] = (uint8_t)(value >> 24); out[1] = (uint8_t)(value >> 16);
    out[2] = (uint8_t)(value >> 8); out[3] = (uint8_t)value;
}

static bool png_chunk(FILE *file, const char type[4], const void *data, uint32_t size) {
    uint8_t be[4]; put_be32(be, size);
    if (fwrite(be, 1, 4, file) != 4 || fwrite(type, 1, 4, file) != 4) return false;
    if (size && fwrite(data, 1, size, file) != size) return false;
    uLong crc = crc32(0L, Z_NULL, 0);
    crc = crc32(crc, (const Bytef *)type, 4);
    if (size) crc = crc32(crc, data, size);
    put_be32(be, (uint32_t)crc);
    return fwrite(be, 1, 4, file) == 4;
}

bool bake_image_write_png(const char *path, const BakeImage *image) {
    if (!path || !image || !image->pixels || image->width <= 0 || image->height <= 0)
        return false;
    FILE *file = fopen(path, "wb");
    if (!file) return false;
    const uint8_t signature[8] = {137,80,78,71,13,10,26,10};
    bool ok = fwrite(signature, 1, 8, file) == 8;
    uint8_t ihdr[13];
    put_be32(ihdr, (uint32_t)image->width); put_be32(ihdr + 4, (uint32_t)image->height);
    ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = ihdr[11] = ihdr[12] = 0;
    ok = ok && png_chunk(file, "IHDR", ihdr, sizeof(ihdr));
    size_t row = (size_t)image->width * 4 + 1;
    size_t raw_size = row * (size_t)image->height;
    uint8_t *raw = malloc(raw_size);
    if (!raw) ok = false;
    if (raw) {
        for (int y = 0; y < image->height; ++y) {
            raw[(size_t)y * row] = 0;
            memcpy(raw + (size_t)y * row + 1,
                   (const uint8_t *)image->pixels + (size_t)y * image->width * 4,
                   (size_t)image->width * 4);
        }
        uLongf compressed_size = compressBound(raw_size);
        uint8_t *compressed = malloc(compressed_size);
        if (!compressed || compress2(compressed, &compressed_size, raw, raw_size,
                                     Z_BEST_SPEED) != Z_OK || compressed_size > UINT32_MAX)
            ok = false;
        else
            ok = ok && png_chunk(file, "IDAT", compressed, (uint32_t)compressed_size);
        free(compressed); free(raw);
    }
    ok = ok && png_chunk(file, "IEND", NULL, 0);
    ok = fclose(file) == 0 && ok;
    if (!ok) fprintf(stderr, "bake_image: failed writing PNG '%s'\n", path);
    return ok;
}
