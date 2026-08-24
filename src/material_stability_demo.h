#pragma once

#include "mesh.h"

struct Renderer;

/* Procedural Phase-D comparison asset.  Panels share the exact same Blue
 * Metal Plate maps; only their per-draw Phase C/D controls differ. */
typedef struct {
    Mesh panels[3]; /* left, right, and a small D-only cavity fixture */
    Texture albedo, orm, normal, cavity;
    VkDescriptorSet neutral_set, cavity_set;
} MaterialStabilityDemo;

bool material_stability_demo_create(struct Renderer *renderer, MaterialStabilityDemo *out);
void material_stability_demo_destroy(struct Renderer *renderer, MaterialStabilityDemo *demo);
