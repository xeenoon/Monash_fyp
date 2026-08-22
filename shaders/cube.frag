#version 450

layout(location = 0) in vec3 normal;
layout(location = 0) out vec4 out_color;

void main() {
    vec3 light_direction = normalize(vec3(0.6, 1.0, 0.8));
    float diffuse = max(dot(normalize(normal), light_direction), 0.0);
    float lighting = 0.28 + 0.72 * diffuse;
    out_color = vec4(vec3(0.68) * lighting, 1.0);
}
