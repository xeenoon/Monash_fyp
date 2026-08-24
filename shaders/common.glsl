/* Shared shader definitions. Every shader that includes this file is rebuilt
   when it changes (tracked via glslc -MD depfiles in CMakeLists.txt). */
#ifndef COMMON_GLSL
#define COMMON_GLSL

/* Descriptor set 0: per-frame data. Matches FrameUniforms in renderer.h
   (std140: every member vec4-aligned, tail padded). */
layout(set = 0, binding = 0) uniform FrameUniforms {
    mat4  projection;
    mat4  view;
    mat4  view_projection;
    mat4  inverse_view_projection;
    mat4  previous_projection;
    mat4  previous_view;
    mat4  previous_view_projection;
    mat4  local_to_camera_relative;
    mat4  previous_local_to_camera_relative;
    vec4  sun_direction;
    float time;
    float near_plane;
    float debug_view;
    float relight_strength;
    mat4  shadow_view_projection[4];
    vec4  shadow_splits;
    vec4  shadow_parameters;
    vec4  sun_radiance;
    vec4  atmosphere_radii;       /* bottom/top/camera altitude km/sun radius */
    vec4  atmosphere_rayleigh;    /* scattering RGB, density exponential scale */
    vec4  atmosphere_mie_scatter; /* scattering RGB, density exponential scale */
    vec4  atmosphere_mie_extinct; /* extinction RGB, phase g */
    vec4  atmosphere_absorption;  /* ozone extinction RGB */
    vec4  atmosphere_ground;      /* albedo RGB, multiple-scattering factor */
    vec4  atmosphere_options;     /* aerial max km, debug slice, reserved */
    vec4  temporal_parameters;    /* history valid, dt, sRGB swapchain, reserved */
    vec4  temporal_jitter;        /* current NDC xy, previous NDC xy */
    vec4  shader_dump;            /* dump enabled (x), reserved */
} frame;

layout(set = 0, binding = 1) uniform sampler2DArrayShadow shadow_map;
layout(set = 0, binding = 2) uniform sampler2DArray shadow_map_raw;

/* Diffuse sky IBL (set 0, binding 4). Matches EnvironmentUniforms in
   renderer.h; independent of FrameUniforms, updated once at load rather than
   per frame (see environment.c). */
layout(set = 0, binding = 4) uniform EnvironmentUniforms {
    vec4 sh[9];       /* 3rd-order SH, RGB in .xyz; cosine-lobe + 1/pi baked in */
    vec4 env_params;  /* x: enabled, y: diffuse intensity, z/w: reserved */
} env;

/* 3rd-order real-SH reconstruction of the diffuse (Lambertian) response to the
   environment along N. The basis below must match environment_project_sh9's
   projection basis in environment.c exactly (same normalization constants,
   same term order) -- env.sh[] already has the cosine-lobe convolution and
   the 1/pi Lambertian normalization baked in, so this is directly usable as
   `indirect_diffuse * environment_irradiance(N)`, no further constants. */
vec3 environment_irradiance(vec3 N) {
    vec3 result =
        env.sh[0].xyz * 0.282095 +
        env.sh[1].xyz * (0.488603 * N.y) +
        env.sh[2].xyz * (0.488603 * N.z) +
        env.sh[3].xyz * (0.488603 * N.x) +
        env.sh[4].xyz * (1.092548 * N.x * N.y) +
        env.sh[5].xyz * (1.092548 * N.y * N.z) +
        env.sh[6].xyz * (0.315392 * (3.0 * N.z * N.z - 1.0)) +
        env.sh[7].xyz * (1.092548 * N.x * N.z) +
        env.sh[8].xyz * (0.546274 * (N.x * N.x - N.y * N.y));
    return max(result, vec3(0.0));
}

/* Infinite reversed-Z helpers. UV follows Vulkan framebuffer orientation, so
   the projection's single Y flip is not repeated here. The reconstructed
   position is camera-relative by design; absolute world positions stay double
   precision on the CPU. */
float linear_view_depth(float depth) {
    return frame.near_plane / max(depth, 1e-20);
}

vec3 reconstruct_camera_relative(vec2 uv, float depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 world = frame.inverse_view_projection * clip;
    return world.xyz / world.w;
}

#endif
