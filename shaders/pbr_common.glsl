/* Compact non-anisotropic Default Lit helpers for forward shading.
   The function names identify their Unreal BRDF.ush/ShadingModels.ush
   counterparts; the implementation is intentionally limited to the Quarry
   material contract rather than copying Unreal's shader framework. */
#ifndef PBR_COMMON_GLSL
#define PBR_COMMON_GLSL

const float PBR_PI = 3.14159265358979323846;

float pbr_pow5(float value) {
    float value2 = value * value;
    return value2 * value2 * value;
}

/* BRDF.ush::F_Schlick. Unreal limits F90 for very dark F0 values so values
   below two percent behave as material shadowing instead of bright Fresnel. */
vec3 ue_f_schlick(vec3 F0, float VoH) {
    float fresnel = pbr_pow5(1.0 - clamp(VoH, 0.0, 1.0));
    float F90 = clamp(50.0 * F0.g, 0.0, 1.0);
    return vec3(F90 * fresnel) + (1.0 - fresnel) * F0;
}

/* BRDF.ush::D_GGX. a2 is alpha squared, or perceptualRoughness^4. */
float ue_d_ggx(float a2, float NoH) {
    float d = (NoH * a2 - NoH) * NoH + 1.0;
    return a2 / max(PBR_PI * d * d, 1e-7);
}

/* BRDF.ush::Vis_SmithJointApprox. This is already G/(4 NoL NoV), so it
   must not be divided by the Cook-Torrance denominator a second time. */
float ue_vis_smith_joint_approx(float a2, float NoV, float NoL) {
    float alpha = sqrt(a2);
    float smith_v = NoL * (NoV * (1.0 - alpha) + alpha);
    float smith_l = NoV * (NoL * (1.0 - alpha) + alpha);
    return 0.5 / max(smith_v + smith_l, 1e-6);
}

/* BRDF.ush::Diffuse_Lambert. DiffuseColor already excludes metallic energy. */
vec3 ue_diffuse_lambert(vec3 diffuse_color) {
    return diffuse_color / PBR_PI;
}

struct UeBxdfEnergyTerms {
    vec3 compensation;
    vec3 specular_energy;
};

/* Analytic form of Unreal's directional GGX energy lookup. It replaces the
   engine-owned LUT while retaining the same inputs and roles for this compact
   forward path: x is single-scatter directional albedo and y its Fresnel
   contribution. roughness is clamped by the material caller. */
vec2 ue_ggx_energy_lookup(float roughness, float NoV) {
    float r = max(roughness, 1e-4);
    float c = clamp(NoV, 0.0, 1.0);
    float single_scatter = 1.0 - clamp(
        pow(r, c / r) * ((r * c + 0.0266916) / (0.466495 + c)),
        0.0, 1.0);
    float fresnel_scatter = pbr_pow5(1.0 - c) *
        pow(2.36651 * pow(c, 4.7703 * r) + 0.0387332, r);
    return vec2(clamp(single_scatter, 1e-4, 1.0),
                clamp(fresnel_scatter, 0.0, 1.0));
}

/* ShadingEnergyConservation.ush::ComputeFresnelEnergyTerms, reduced to the
   Schlick GGX case used by DefaultLitBxDF. */
UeBxdfEnergyTerms ue_compute_ggx_spec_energy_terms(
    float roughness, float NoV, vec3 F0) {
    vec2 lookup = ue_ggx_energy_lookup(roughness, NoV);
    vec3 F90 = vec3(clamp(50.0 * F0.g, 0.0, 1.0));
    UeBxdfEnergyTerms terms;
    terms.compensation = vec3(1.0) + F0 * ((1.0 - lookup.x) / lookup.x);
    terms.specular_energy = terms.compensation *
        (lookup.x * F0 + lookup.y * (F90 - F0));
    return terms;
}

/* ShadingEnergyConservation.ush::ComputeEnergyPreservation. This reserves for
   diffuse only the energy not reflected by the complete specular lobe. */
vec3 ue_compute_energy_preservation(UeBxdfEnergyTerms terms) {
    return clamp(vec3(1.0) - terms.specular_energy, vec3(0.0), vec3(1.0));
}

/* ShadingEnergyConservation.ush::ComputeEnergyConservation. The multiplier
   restores energy lost by the single-scattering GGX approximation. */
vec3 ue_compute_energy_conservation(UeBxdfEnergyTerms terms) {
    return terms.compensation;
}

struct UeDefaultLit {
    vec3 diffuse;
    vec3 specular;
};

/* The pre-Phase-A Quarry BRDF, retained verbatim in structure for an in-app
   A/B comparison. Unlike the UE path this uses Schlick-GGX masking and no
   diffuse-energy or multiple-scattering compensation. */
UeDefaultLit legacy_quarry_bxdf(vec3 base_color, float metallic,
                                float roughness, vec3 N, vec3 V, vec3 L) {
    vec3 H = normalize(L + V);
    float NoL = max(dot(N, L), 0.0);
    float NoV = max(dot(N, V), 0.0);
    float alpha = roughness * roughness;
    float a2 = alpha * alpha;
    float NoH = max(dot(N, H), 0.0);
    float denominator = max(PBR_PI *
        pow(NoH * NoH * (a2 - 1.0) + 1.0, 2.0), 1e-5);
    float D = a2 / denominator;
    float k = roughness + 1.0;
    k = k * k / 8.0;
    float G = NoL / mix(NoL, 1.0, k) *
              NoV / mix(NoV, 1.0, k);
    vec3 F0 = mix(vec3(0.04), base_color, metallic);
    vec3 F = F0 + (1.0 - F0) *
             pbr_pow5(1.0 - max(dot(H, V), 0.0));

    UeDefaultLit lighting;
    lighting.specular = D * G * F / max(4.0 * NoL * NoV, 1e-5);
    lighting.diffuse = (1.0 - metallic) * base_color / PBR_PI;
    return lighting;
}

/* ShadingModels.ush::DefaultLitBxDF, reduced to one punctual directional light,
   isotropic GGX, Lambert diffuse, and opaque reflection. The caller applies
   light radiance, NoL, and shadow visibility in the existing forward pass. */
UeDefaultLit ue_default_lit_bxdf(vec3 diffuse_color, vec3 F0,
                                float roughness, vec3 N, vec3 V, vec3 L) {
    UeDefaultLit lighting;
    lighting.diffuse = vec3(0.0);
    lighting.specular = vec3(0.0);

    float NoL = clamp(dot(N, L), 0.0, 1.0);
    if (NoL <= 0.0) return lighting;

    /* Match DefaultLitBxDF's two-sided-safe NoV handling without allowing a
       zero denominator at grazing angles. */
    float NoV = clamp(abs(dot(N, V)) + 1e-5, 1e-5, 1.0);
    vec3 half_sum = V + L;
    float half_length2 = dot(half_sum, half_sum);
    vec3 H = half_length2 > 1e-10 ? half_sum * inversesqrt(half_length2) : N;
    float NoH = clamp(dot(N, H), 0.0, 1.0);
    float VoH = clamp(dot(V, H), 0.0, 1.0);
    float a2 = roughness * roughness;
    a2 *= a2;

    float D = ue_d_ggx(a2, NoH);
    float Vis = ue_vis_smith_joint_approx(a2, NoV, NoL);
    vec3 F = ue_f_schlick(F0, VoH);
    UeBxdfEnergyTerms energy =
        ue_compute_ggx_spec_energy_terms(roughness, NoV, F0);

    lighting.diffuse = ue_diffuse_lambert(diffuse_color) *
        ue_compute_energy_preservation(energy);
    lighting.specular = (D * Vis) * F *
        ue_compute_energy_conservation(energy);
    return lighting;
}

/* Distance attenuation for a local light that is not a point.
 *
 * The windowed inverse square is the standard one: a smooth cutoff at the
 * light's radius, because a hard `if (distance < radius)` leaves a visible
 * edge on the floor. What is added here is a SOURCE RADIUS.
 *
 * A torch flame is roughly a 15 cm blob, and a wall-mounted one sits about
 * 17 cm off the wall behind it. Treated as a point, the inverse square then
 * makes that wall 2.5x brighter than the flame itself -- measurably, not as
 * an impression -- so the fire reads as a dark smudge in front of a hot spot,
 * which is the exact opposite of what fire looks like. Softening the near
 * field with 1/(d + r)^2 fixes it where it is wrong and leaves the rest of
 * the room alone: at 17 cm a 12 cm source drops the surface to a third of the
 * point-light value, while at three metres it is still 92% of it.
 *
 * `source_radius` of 0 reproduces the old point behaviour exactly. */
float local_light_attenuation(float distance_to_light, float light_radius, float source_radius) {
    float normalized_distance = distance_to_light / max(light_radius, 1e-4);
    float window = max(1.0 - pow(normalized_distance, 4.0), 0.0);
    float softened = distance_to_light + max(source_radius, 0.0);
    return window * window / max(softened * softened, 0.01);
}

#endif
