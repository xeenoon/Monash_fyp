#include "temporal.h"

#include <math.h>

float temporal_halton(uint32_t index, uint32_t base) {
    if (base < 2u) return 0.0f;
    float result = 0.0f;
    float fraction = 1.0f;
    while (index) {
        fraction /= (float)base;
        result += fraction * (float)(index % base);
        index /= base;
    }
    return result;
}

vec2s temporal_jitter_ndc(uint64_t frame_index, uint32_t width, uint32_t height) {
    if (!width || !height) return (vec2s){{0.0f, 0.0f}};
    /* Eight samples are enough to break the grid while returning regularly to
       a known phase. A half-pixel box keeps silhouettes well behaved. */
    uint32_t sample = (uint32_t)(frame_index & 7u) + 1u;
    float pixel_x = temporal_halton(sample, 2u) - 0.5f;
    float pixel_y = temporal_halton(sample, 3u) - 0.5f;
    return (vec2s){{2.0f * pixel_x / (float)width,
                    2.0f * pixel_y / (float)height}};
}

mat4s temporal_jitter_projection(mat4s projection, vec2s jitter_ndc) {
    /* cglm stores mat[col][row]. With this RH projection clip.w=-view.z, so a
       negative z-column coefficient produces the requested positive NDC shift. */
    projection.raw[2][0] -= jitter_ndc.x;
    projection.raw[2][1] -= jitter_ndc.y;
    return projection;
}

static float wrapped_angle_delta(float a, float b) {
    float delta = fmodf(fabsf(a - b), 360.0f);
    return delta > 180.0f ? 360.0f - delta : delta;
}

bool temporal_camera_cut(WorldPosition previous, WorldPosition current,
                         float previous_yaw, float current_yaw,
                         float previous_pitch, float current_pitch,
                         double teleport_distance_m) {
    double dx = current.x - previous.x;
    double dy = current.y - previous.y;
    double dz = current.z - previous.z;
    double limit = teleport_distance_m > 0.0 ? teleport_distance_m : 1.0;
    if (dx * dx + dy * dy + dz * dz > limit * limit) return true;
    return wrapped_angle_delta(previous_yaw, current_yaw) > 45.0f ||
           fabsf(previous_pitch - current_pitch) > 45.0f;
}

float temporal_exposure_target(float average_luminance, float middle_grey,
                               float minimum_exposure, float maximum_exposure) {
    float luminance = fmaxf(average_luminance, 1e-5f);
    float target = middle_grey / luminance;
    return fminf(fmaxf(target, minimum_exposure), maximum_exposure);
}

float temporal_adapt_exposure(float current_exposure, float target_exposure,
                              float delta_seconds, float brighten_speed,
                              float darken_speed) {
    if (!(delta_seconds > 0.0f)) return current_exposure;
    float speed = target_exposure > current_exposure ? brighten_speed : darken_speed;
    float weight = 1.0f - expf(-fmaxf(speed, 0.0f) * delta_seconds);
    return current_exposure + (target_exposure - current_exposure) * weight;
}
