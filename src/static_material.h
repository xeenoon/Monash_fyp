#pragma once

#include <stdbool.h>

/* Static mesh controls.  glTF roughnessFactor remains a separate multiplier:
 * it is not folded into roughness_bias. */
typedef struct {
    float normal_strength;
    float ao_strength;
    float cavity_strength;
    float roughness_bias;
    float displacement_scale;
} StaticMaterialParameters;

float static_material_authored_roughness(float orm_roughness, float gltf_factor,
                                         float roughness_bias);
float static_material_curvature_floor(float variance, float curvature_strength);
float static_material_effective_roughness(float authored, float curvature_floor);
float static_material_visibility(float sample, float strength);

