/* Single translation unit that compiles the stb_image implementation.
   Everywhere else includes stb_image.h for declarations only. PNG for every
   authored texture (albedo/ORM/normal/terrain imagery); HDR (Radiance
   .hdr, via stbi_loadf) for the Phase B IBL environment map
   (src/environment.c). Any STBI_ONLY_* define restricts the compiled
   implementation to exactly that set of formats -- omitting HDR here made
   stbi_loadf silently fail on every .hdr file. */
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_HDR
#include "stb_image.h"
