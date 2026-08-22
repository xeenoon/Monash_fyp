/* Single translation unit that compiles the stb_image implementation.
   Everywhere else includes stb_image.h for declarations only. */
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"
