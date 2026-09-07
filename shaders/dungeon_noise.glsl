#ifndef DUNGEON_NOISE_GLSL
#define DUNGEON_NOISE_GLSL

/* Small self-contained 2D noise for the dungeon surface permutations (moss
   threshold, puddle ripples/masks). Deliberately separate from terrain's
   material_hash in terrain.frag -- that one is an integer atlas-cell picker,
   this is continuous gradient noise, and nothing outside the two dungeon
   fragment shaders needs it. */

vec2 dungeon_hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

/* Gradient (Perlin-style) noise, range approximately [-1, 1]. */
float dungeon_noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = dot(dungeon_hash2(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0));
    float b = dot(dungeon_hash2(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0));
    float c = dot(dungeon_hash2(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0));
    float d = dot(dungeon_hash2(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

/* 3-octave fractal Brownian motion, range approximately [-1, 1]. */
float dungeon_fbm(vec2 p) {
    float sum = 0.0;
    float amplitude = 0.5;
    for (int octave = 0; octave < 3; ++octave) {
        sum += amplitude * dungeon_noise(p);
        p *= 2.02;
        amplitude *= 0.5;
    }
    return sum;
}

#endif
