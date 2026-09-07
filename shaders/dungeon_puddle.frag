#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "pbr_common.glsl"
#include "dungeon_noise.glsl"
#include "shader_dump.glsl"

/* Fake dungeon puddle reflections (Phase 5). Puddles are cosmetic and cheap
   on purpose: no separate reflection render target, no floor/prop/player
   reflected -- only the nearest wall, using the exact same blocker segment
   array the point-light occlusion test already uploads (see mesh.frag's
   point_light_occluded). It's mostly dark anyway; torch glints do the real
   work of making it read as water. Alpha-blended, depth-write off (see
   create_scene_pipeline_ex's alpha_blend path in renderer.c), untextured --
   this batch has no material set bound. */
layout(early_fragment_tests) in;

layout(location = 0) in vec2 uv; /* x: 0 at centre .. 1 at rim (radial fade) */
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec3 camera_relative_position;
layout(location = 4) in vec4 current_clip;
layout(location = 5) in vec4 previous_clip;
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_motion;

layout(push_constant) uniform DrawData {
    mat4 local_to_camera_relative;
    vec4 geometry;
    vec4 elevation_uv;
    vec4 material_factors;
    vec4 debug;
} draw;

/* Marches the blocker segment array and returns the distance (metres) along
   `direction` (world XZ, normalized) from `origin` to the nearest crossing,
   or a large sentinel if none is hit. Mirrors mesh.frag's
   segment_crosses_blocker geometry but solves for distance instead of a
   boolean, since the puddle needs to know *where* it hit to shade it. */
float raycast_walls_xz(vec2 origin, vec2 direction, out vec2 hit_point, out int hit_index) {
    float best_t = 1e6;
    hit_point = origin;
    hit_index = -1;
    int blocker_count = clamp(int(frame.point_light_options.w + 0.5), 0, 64);
    for (int i = 0; i < blocker_count; ++i) {
        vec4 blocker = frame.point_light_blocker_xz[i];
        vec2 a = blocker.xy, b = blocker.zw;
        vec2 d2 = b - a;
        float denom = direction.x * d2.y - direction.y * d2.x;
        if (abs(denom) < 1e-9)
            continue;
        vec2 e = a - origin;
        float t = (e.x * d2.y - e.y * d2.x) / denom;
        float u = (e.x * direction.y - e.y * direction.x) / denom;
        if (t > 0.05 && t < best_t && u >= 0.0 && u <= 1.0) {
            best_t = t;
            hit_point = origin + direction * t;
            hit_index = i;
        }
    }
    return best_t;
}

void main() {
    float time = frame.time;
    vec2 world_xz = camera_relative_position.xz +
        vec2(frame.shader_dump.y, frame.shader_dump.w);

    /* Two scrolling fbm layers for ripples: a rippled normal perturbing an
       otherwise flat pool, animated so the surface isn't static. */
    vec2 flow_a = world_xz * 1.3 + vec2(time * 0.06, time * 0.04);
    vec2 flow_b = world_xz * 2.7 - vec2(time * 0.03, time * 0.05);
    float ripple_a = dungeon_fbm(flow_a);
    float ripple_b = dungeon_fbm(flow_b);
    float ripple_height = ripple_a * 0.6 + ripple_b * 0.4;
    vec3 N = normalize(vec3(dFdxFine(ripple_height) * -1.5, 1.0, dFdyFine(ripple_height) * -1.5));

    vec3 V = normalize(-camera_relative_position);
    float NoV = max(dot(N, V), 0.0);
    float roughness = 0.05;
    vec3 F0 = vec3(0.02);
    float fresnel = pow(1.0 - NoV, 5.0);
    fresnel = mix(0.02, 1.0, fresnel);

    /* Fake wall reflection: reflect the view about the rippled normal,
       project to the ground plane, and see which wall segment that ray
       would hit. Walls only, exactly as scoped -- no floor/prop/player
       reflection, and it is dark down here anyway.

       The blocker array (frame.point_light_blocker_xz) is written camera-
       relative, same as point light positions -- see
       dungeon_scene_write_light_blockers and dungeon_scene_write_lights.
       The ray must be cast in that same space, so the origin here is
       camera_relative_position.xz, NOT the true-world world_xz used for the
       ripple noise above; mixing the two silently makes every ray miss. */
    vec3 R = reflect(-V, N);
    vec2 reflect_dir = length(R.xz) > 1e-4 ? normalize(R.xz) : vec2(1.0, 0.0);
    vec2 hit_point;
    int hit_index;
    float hit_distance =
        raycast_walls_xz(camera_relative_position.xz, reflect_dir, hit_point, hit_index);
    vec3 reflection_color = vec3(0.0);
    if (hit_index >= 0) {
        /* Shade the hit wall as a dark rock tint, lit by the same point
           lights the rest of the scene uses, attenuated to the hit point --
           not the puddle's own position -- so the reflection dims with
           distance from the torch near the wall, not near the water. */
        vec3 rock_tint = vec3(0.05, 0.045, 0.04);
        int point_light_count = clamp(int(frame.point_light_options.x + 0.5), 0, 16);
        for (int i = 0; i < point_light_count; ++i) {
            vec4 position_radius = frame.point_light_position_radius[i];
            float distance_to_hit = length(position_radius.xz - hit_point);
            float normalized_distance = distance_to_hit / max(position_radius.w, 1e-4);
            float window = max(1.0 - pow(normalized_distance, 4.0), 0.0);
            float attenuation = window * window / max(distance_to_hit * distance_to_hit, 0.01);
            vec4 color_intensity = frame.point_light_color_intensity[i];
            reflection_color += rock_tint * color_intensity.rgb * color_intensity.w * attenuation;
        }
        /* Distant hits fade toward black rather than snapping the reflection
           off at the last blocker -- keeps far corridors reading as dark
           water instead of a hard-edged mirror. */
        reflection_color *= exp(-hit_distance * 0.15);
    }

    /* Torch glints: direct GGX highlights are what actually sell "puddle" in
       a scene this dark -- the shared environment cube contributes almost
       nothing indoors (see environment_lighting.glsl's no-HDR/low-intensity
       fallback), so it is deliberately not sampled here. */
    vec3 specular_glints = vec3(0.0);
    int point_light_count = clamp(int(frame.point_light_options.x + 0.5), 0, 16);
    for (int i = 0; i < point_light_count; ++i) {
        vec4 position_radius = frame.point_light_position_radius[i];
        vec3 to_light = position_radius.xyz - camera_relative_position;
        float distance2 = dot(to_light, to_light);
        float distance_to_light = sqrt(max(distance2, 1e-8));
        vec3 local_L = to_light / distance_to_light;
        float normalized_distance = distance_to_light / max(position_radius.w, 1e-4);
        float window = max(1.0 - pow(normalized_distance, 4.0), 0.0);
        float attenuation = window * window / max(distance2, 0.01);
        UeDefaultLit bxdf = ue_default_lit_bxdf(vec3(0.0), F0, roughness, N, V, local_L);
        vec4 color_intensity = frame.point_light_color_intensity[i];
        specular_glints += bxdf.specular * color_intensity.rgb * color_intensity.w * attenuation;
    }

    vec3 base_color = vec3(0.03, 0.028, 0.035); /* near-black, faint cool tint */
    vec3 final_color = base_color + reflection_color * fresnel + specular_glints;

    /* Rim fade: blend into the surrounding floor over the outer ~20% of the
       fan radius using the packed radial coordinate, plus a touch of noise
       so the edge isn't a perfect circle. */
    float rim_noise = dungeon_fbm(world_xz * 3.0) * 0.08;
    float alpha = 1.0 - smoothstep(0.8 + rim_noise, 1.0 + rim_noise, uv.x);

    out_color = vec4(final_color, alpha);
    vec2 current_uv = current_clip.xy / current_clip.w * 0.5 + 0.5;
    vec2 previous_uv = previous_clip.xy / previous_clip.w * 0.5 + 0.5;
    out_motion = previous_uv - current_uv;

    /* See SHADER_DUMP_LEGEND["dungeon_puddle"] in renderer.c for the f0..f19 layout. */
    shader_dump(DUMP_SHADER_DUNGEON_PUDDLE,
                vec4(alpha, fresnel, hit_distance, roughness),
                vec4(base_color, alpha),
                vec4(N, NoV),
                vec4(reflection_color, length(specular_glints)),
                vec4(final_color, ripple_height));
}
