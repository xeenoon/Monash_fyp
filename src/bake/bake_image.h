// Image I/O for the terrain baker: load PNG/JPEG tiles into a flat RGBA8 buffer
// (one uint32 per pixel, matching the std430 layout the compute shaders index),
// and write results back out.  Kept separate from bake_gpu so the GPU harness
// has no image-codec dependency.
#ifndef BAKE_IMAGE_H
#define BAKE_IMAGE_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    int width;
    int height;
    uint32_t *pixels;  // width*height, RGBA8 packed (byte 0 = R … byte 3 = A)
} BakeImage;

// Load via stb_image, forced to 4 channels.  Returns false on failure.
bool bake_image_load(const char *path, BakeImage *image);

// Allocate an uninitialised image of the given size.
bool bake_image_alloc(int width, int height, BakeImage *image);

void bake_image_free(BakeImage *image);

// Write a binary PPM (P6, RGB — alpha dropped).  Dependency-free and viewable;
// final tiles move to PNG at the orchestration stage.
bool bake_image_write_ppm(const char *path, const BakeImage *image);

// Write an 8-bit RGBA PNG using the project's existing zlib dependency.
bool bake_image_write_png(const char *path, const BakeImage *image);

#endif  // BAKE_IMAGE_H
