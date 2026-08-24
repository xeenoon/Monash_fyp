/* Phase D has deliberately separate geometry and normal-map filters. */
struct MaterialDetailResult {
    float authored_roughness;
    float effective_roughness;
    float geometric_variance;
    float geometric_floor;
    float filtered_normal_length;
    float mip_variance;
    float mip_kernel;
    float mip_roughness;
};

float material_authored_roughness(float orm_roughness, float factor, float bias) {
    return clamp(orm_roughness * factor + bias, 0.045, 1.0);
}

float material_normal_mip_variance(float filtered_normal_length, float normal_strength) {
    float len = clamp(filtered_normal_length, 1e-4, 1.0);
    float variance = max((1.0 - len) / len, 0.0);
    return variance * normal_strength * normal_strength;
}

MaterialDetailResult material_detail_evaluate(vec3 geometric_normal,
                                               float filtered_normal_length,
                                               float normal_strength,
                                               float authored_roughness,
                                               bool phase_d_enabled) {
    MaterialDetailResult r;
    r.authored_roughness = authored_roughness;
    vec3 dx = dFdx(geometric_normal), dy = dFdy(geometric_normal);
    r.geometric_variance = max(dot(dx, dx), dot(dy, dy));
    r.geometric_floor = clamp(pow(max(r.geometric_variance, 0.0), frame.material_curvature.z) *
                               frame.material_curvature.x + frame.material_curvature.y, 0.0, 1.0);
    r.filtered_normal_length = clamp(filtered_normal_length, frame.material_normal_filter.z, 1.0);
    r.mip_variance = material_normal_mip_variance(r.filtered_normal_length, normal_strength);
    r.mip_kernel = min(r.mip_variance * frame.material_normal_filter.x, frame.material_normal_filter.y);
    float authored_alpha = authored_roughness * authored_roughness;
    r.mip_roughness = sqrt(clamp(authored_alpha + r.mip_kernel, authored_alpha, 1.0));
    /* Branching protects Phase C if curvature bias is configured later. */
    r.effective_roughness = phase_d_enabled
        ? max(authored_roughness, max(r.geometric_floor, r.mip_roughness))
        : authored_roughness;
    return r;
}

float material_visibility(float value, float strength) {
    return mix(1.0, clamp(value, 0.0, 1.0), clamp(strength, 0.0, 1.0));
}
