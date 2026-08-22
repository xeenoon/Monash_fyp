#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

layout(push_constant) uniform Camera {
    mat4 view_projection;
} camera;

layout(location = 0) out vec3 normal;

void main() {
    gl_Position = camera.view_projection * vec4(in_position, 1.0);
    normal = in_normal;
}
