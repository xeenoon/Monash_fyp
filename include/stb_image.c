/* Single translation unit that compiles the stb_image implementation.
   Everywhere else includes stb_image.h for declarations only.  Static glTF
   assets may legitimately carry PNG or JPEG images, while the environment
   uses Radiance HDR, so keep stb's normal decoder set enabled. */
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
