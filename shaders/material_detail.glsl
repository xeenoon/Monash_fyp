/* D1 material-detail stability.  Derivatives are deliberately evaluated only
 * after the final mapped world-space normal has been constructed. */
float material_authored_roughness(float orm_roughness, float factor, float bias) {
    return clamp(orm_roughness * factor + bias, 0.045, 1.0);
}
float material_curvature_floor(vec3 mapped_normal, float strength, out float variance) {
    vec3 dx = dFdx(mapped_normal);
    vec3 dy = dFdy(mapped_normal);
    variance = max(dot(dx, dx), dot(dy, dy));
    return clamp(sqrt(max(variance, 0.0)) * max(strength, 0.0), 0.0, 1.0);
}
float material_visibility(float value, float strength) {
    return mix(1.0, clamp(value, 0.0, 1.0), clamp(strength, 0.0, 1.0));
}
