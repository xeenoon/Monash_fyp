#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "dungeon_noise.glsl"
#include "shader_dump.glsl"

/* Torch flame: one billboard quad per fixture (oriented by flame_billboard()
   in dungeon_scene.c), drawn additively with depth writes off. Everything
   visible here is procedural in quad UV space, so the asset stays the
   untextured OBJ it was.

   The model is a DENSITY FIELD: one common burning mass at the fuel that
   splits into tongues as it rises. Two earlier versions of this were wrong in
   opposite directions and both are worth recording, because the failure modes
   are the obvious ones to walk into.

   The first was a single noise-decorated teardrop. Closed, symmetric and
   topologically constant, it read instantly as a candle sprite however much
   noise was layered onto its edge.

   The second was four independent lobes rising from the same point. That
   fixed the topology but produced three clean petals: each tongue had its own
   crisp outline and its own white stripe down the middle, so it looked like
   glowing leaves stuck together. Fire does not do that either.

   What the reference photograph actually shows, and what this builds:

     - the bottom half is ONE chaotic mass. Tongues share a root and only
       separate higher up (`split`), so it is one fire dividing rather than
       several fires sharing a coordinate;
     - the silhouette is domain-warped, and the freedom to deform grows
       steeply with height, so tips are ragged and roots are not;
     - large ELONGATED holes are carved through the body with a low vertical
       frequency; high-frequency carving only produces television static;
     - temperature comes from its own turbulent field rather than from
       distance to a centreline, so the white-hot regions are islands instead
       of a stripe through every tongue;
     - emission goes as density SQUARED while coverage does not, so the fire
       is luminous in its core and translucent at its edges.

   The quad is authored bottom-up: uv.y 0 is inside the torch head, uv.y 1 is
   the top of the plume. Horizontal distances are scaled by the quad's aspect
   so the shape is a property of this shader and not of whatever quad size
   dungeon_scene.c picks.

   There is deliberately no glow or halo term. The soft spread of light around
   a fire is the renderer's bloom pass (bloom_downsample.comp), fed by the
   above-threshold radiance emitted here.

   The BRIGHTNESS is not decided here either. dungeon_light_flicker() on the
   CPU decides it once per fixture per frame and hands the same number to the
   point light and to this shader (geometry.w), because a flame that flares a
   frame out of step with the light it casts reads as two effects rather than
   one. That number drives the plume's HEIGHT as well as its brightness: a
   torch that gutters visibly shrinks. */

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_motion;

layout(push_constant) uniform DrawData {
    mat4 local_to_camera_relative;
    vec4 geometry;     /* flame tint RGB, flicker intensity scale */
    vec4 elevation_uv; /* time s, fixture phase s, quad width m, quad height m */
    vec4 material;     /* lean (sway X) in quad heights, reserved, core radiance, reserved */
    vec4 debug;
} draw;

/* The fuel line in uv.y: where the top of the torch head sits. It is a
   quarter of the way UP the quad, not at its bottom edge, because the flame
   has to continue DOWNWARD around the head -- fire wraps its fuel. With the
   fuel line at the quad's bottom there is nowhere for that to happen and the
   flame gets cut off square where it meets the fixture, which is exactly what
   the first attempt at a collar did. dungeon_scene.c's FLAME_BASE_DROP_M must
   agree with this: it is FLAME_BASE * FLAME_HEIGHT_M. */
const float FLAME_BASE = 0.26;
const float FLAME_TIP = 0.94;
const int FLAME_LOBES = 4;
const int EMBER_COUNT = 7;

float flame_hash(float x) {
    return fract(sin(x * 91.7) * 43758.5453);
}

/* A 1-D band of the shared gradient noise, for signals that vary in time
   only. The second coordinate is an arbitrary fixed lane. */
float flame_signal(float x) {
    return dungeon_noise(vec2(x, 17.31));
}

/* Time for a parcel of hot gas leaving the fuel to reach height y, under
   v(y) = sqrt(v0^2 + 2*a*y): buoyancy accelerates it as it rises.
   The algebraically equivalent 2y / (v0 + sqrt(...)) is used rather than
   (sqrt(...) - v0) / a because the latter is a difference of nearly equal
   numbers as y approaches zero, which is exactly where the flame is
   attached and where the error would show. */
float travel_time(float y) {
    const float initial_speed = 0.35;
    const float buoyancy = 1.8;
    return (2.0 * y) /
           (initial_speed + sqrt(initial_speed * initial_speed + 2.0 * buoyancy * max(y, 0.0)));
}

void main() {
    /* The same clock dungeon_light_flicker() runs on, phase included, so the
       plume and the light it casts are two views of one fire. */
    float t = draw.elevation_uv.x + draw.elevation_uv.y * 37.0;
    float aspect = draw.elevation_uv.z / max(draw.elevation_uv.w, 1e-4);
    float flicker = draw.geometry.w;

    /* x in quad-height units, so the constants below are aspect independent. */
    float x = (uv.x - 0.5) * aspect;
    float y = uv.y;

    /* --- three flow fields -------------------------------------------
       Distinct frequencies AND distinct rates. Driving the silhouette, the
       internal heat and the edges from one field makes the bright core bend
       as a coherent object when the flame leans, which reads as a solid
       thing being waved about rather than gas burning. Roughly: macro around
       a hertz, structure two to five, edges faster still. */
    /* THE flow coordinate. Everything large-scale below is a function of tau
       rather than of t, which is the difference between a shape that deforms
       in place and one that advects: a disturbance born at the fuel at time
       t0 appears at height y at t0 + T(y), so surfaces of constant tau travel
       upward, accelerating as they go. Scrolling a noise field gives features
       that move at one speed regardless of height; this does not. */
    /* Height above the fuel, normalised: 0 at the head, 1 at the quad top.
       Parcels of gas start burning at the fuel line, so this -- not raw uv.y
       -- is what the rise, the turbulence and the splitting are measured
       against. Below the fuel line it pins to 0, which is what lets the
       collar below inherit the root's behaviour. */
    float above_h = clamp((y - FLAME_BASE) / (1.0 - FLAME_BASE), 0.0, 1.0);
    float tau = t - travel_time(above_h);

    float macro = dungeon_fbm(vec2(x * 1.5, above_h * 1.2 - tau * 1.1));
    float structure = dungeon_fbm(vec2(x * 4.0, above_h * 2.0 - tau * 2.6));
    float edge = dungeon_fbm(vec2(x * 9.0, above_h * 6.0 - tau * 5.2));
    /* Elongated cavities, domain-warped by the macro flow so they drift with
       the fire instead of being a fixed lattice it slides through. The low
       VERTICAL frequency is what makes them long tears; raise it and the
       carve degenerates into television static. */
    float cavity = dungeon_fbm(vec2(x * 4.5 + macro * 0.9, above_h * 1.3 - tau * 1.8));
    float cavity_fine = dungeon_fbm(vec2(x * 7.5 - macro * 1.3, above_h * 2.2 - tau * 2.9));

    /* Turbulence is substantial even at the root. An amplitude that starts at
       zero down there leaves a persistent bulb at the fuel with an animated
       flame perched on top of it -- which is what an earlier version did, and
       it was the largest remaining tell once the motion was right. */
    float turbulence = clamp(0.35 + 0.65 * smoothstep(0.0, 0.65, above_h), 0.0, 1.0);
    float wind = 0.12 * sin(tau * 1.7) + 0.05 * sin(tau * 4.1 + 1.7);
    float warped_x = x - (macro * 0.17 + edge * 0.030) * turbulence;
    warped_x -= wind * pow(above_h, 1.5);
    warped_x -= draw.material.x * above_h; /* the CPU's lateral sway, on top */

    float plume = 0.70 + 0.42 * clamp((flicker - 0.45) / 0.90, 0.0, 1.0);
    /* Tongues share a root and separate only as they rise: one fire dividing,
       not several fires sharing a coordinate. */
    float split = smoothstep(0.12, 0.62, above_h);

    /* --- tongues ------------------------------------------------------- */
    float density = 0.0;
    float local_width = 1.0;
    for (int lobe = 0; lobe < FLAME_LOBES; ++lobe) {
        float s1 = flame_hash(float(lobe) + 1.0 + draw.elevation_uv.y * 3.0);
        float s2 = flame_hash(float(lobe) + 11.0 + draw.elevation_uv.y * 3.0);
        float s3 = flame_hash(float(lobe) + 23.0 + draw.elevation_uv.y * 3.0);
        float base_x = (s1 - 0.5) * 0.20 * split;
        float span = (FLAME_TIP - FLAME_BASE) * plume * (0.42 + 0.62 * s2);
        float h = clamp(above_h * (1.0 - FLAME_BASE) / span, 0.0, 1.0);
        /* On the flow clock, so a bend enters at the fuel and rides upward
           rather than the whole tongue swinging at once. */
        float lobe_time = tau * (0.85 + 0.5 * s3) + s1 * 6.28;
        float lean = (0.16 * sin(lobe_time * 1.6 + s2 * 5.0) +
                      0.09 * sin(lobe_time * 3.3 + s1 * 4.0)) * pow(h, 1.6) * split;
        float profile = (0.13 + 0.09 * s3) * pow(max(h, 0.0), 0.45) *
                            pow(max(1.0 - h, 0.0), 0.75) + 0.010;
        /* And the WIDTH travels too, which is what actually reads as
           billowing: a parcel swells low down, rises, necks, and either
           merges or tears off. Animating width per-height instead just makes
           the outline breathe in place. Exponential so it stays positive and
           expands and contracts asymmetrically. */
        float billow = 0.70 * flame_signal(lobe_time * 1.15 + 31.7) +
                       0.22 * flame_signal(lobe_time * 2.31 + 11.2) +
                       0.08 * flame_signal(lobe_time * 4.67 + 73.4);
        profile *= exp(0.32 * smoothstep(0.05, 0.80, above_h) * billow);
        /* A narrowing tongue becomes unstable: it whips harder and tears up.
           Without this a thin branch stretches into a clean geometric noodle
           and hangs there -- a spline, not a wisp of burning gas. */
        float thin = 1.0 - smoothstep(0.030, 0.105, profile);
        float lateral = warped_x - base_x - lean - edge * 0.055 * thin;
        float lobe_density = 1.0 - abs(lateral) / max(profile, 1e-4);
        lobe_density *= 1.0 - smoothstep(0.52, 1.0, h); /* burns out at the tip */
        /* Each tongue waxes and wanes on its own slow cycle, so the
           arrangement is not fixed for the life of the fixture. */
        lobe_density *= 0.76 + 0.31 * sin(lobe_time * 0.9 + s2 * 6.0);
        lobe_density *= 1.0 - thin * max(edge, 0.0) * 0.70;
        lobe_density = clamp(lobe_density, 0.0, 1.0);
        if (lobe_density > density)
            local_width = profile;
        density = max(density, lobe_density);
    }

    /* The common mass at the fuel. Warped and carved like everything else --
       a body term left unperturbed is precisely the stable bulb above. */
    float body_h = clamp(above_h * (1.0 - FLAME_BASE) /
                            ((FLAME_TIP - FLAME_BASE) * plume * 0.55), 0.0, 1.0);
    float body_billow = 0.70 * flame_signal(tau * 1.05 + 5.3) +
                        0.25 * flame_signal(tau * 2.11 + 41.9);
    float body_profile = (0.26 * pow(max(body_h, 0.0), 0.45) *
                              pow(max(1.0 - body_h, 0.0), 0.85) + 0.02) *
                         exp(0.26 * smoothstep(0.05, 0.80, above_h) * body_billow);
    float body = clamp(1.0 - abs(warped_x - macro * 0.05) / max(body_profile, 1e-4), 0.0, 1.0);
    body *= 1.0 - split * 0.62;
    if (body > density)
        local_width = body_profile;
    density = max(density, body);

    /* --- the collar around the head ------------------------------------
       Fire does not sit on top of its fuel, it wraps it. Below the top of the
       head the flame carries on, pinched to the head's own width and fading
       with depth, so tongues lick up the sides of the fixture. What was here
       before terminated the flame at the fuel line and covered the join with
       a bright horizontal band, which read as a flat glowing mat the fire
       bounced on -- wider than the head it was supposed to be part of. */
    float depth_below = max(FLAME_BASE - y, 0.0);
    float collar = exp(-pow(depth_below / 0.20, 2.0));
    float over_head = 1.0 - smoothstep(0.055, 0.105, abs(warped_x));
    float above_fuel = smoothstep(FLAME_BASE - 0.02, FLAME_BASE + 0.10, y);
    float root = mix(collar * over_head, 1.0, above_fuel);
    density *= root;

    /* --- cavities THROUGH the body, at every height -------------------- */
    density *= smoothstep(0.24, 0.66, cavity * 0.5 + 0.5);
    density *= clamp(0.70 + 0.70 * (cavity_fine * 0.5 + 0.5), 0.0, 1.25);
    density *= clamp(0.86 + 0.55 * edge, 0.0, 1.5);
    /* Nothing may reach the quad's top edge, or a tongue is cut off square. */
    density *= 1.0 - smoothstep(0.80, 1.0, y);
    density = clamp(density, 0.0, 1.0);

    /* --- temperature ---------------------------------------------------
       Its own field, only loosely tied to the shape. Deriving heat from
       distance to a tongue's centreline gives every tongue an identical
       orange/yellow/white cross-section, which is what made an earlier
       version look like a bundle of glowing petals. */
    float temperature = density * clamp(0.45 + 0.75 * (structure * 0.5 + 0.5) +
                                            0.35 * (edge * 0.5 + 0.5), 0.0, 1.3);
    vec3 flame_color = mix(vec3(1.00, 0.16, 0.01), vec3(1.00, 0.58, 0.08),
                           smoothstep(0.25, 0.65, temperature));
    flame_color = mix(flame_color, vec3(1.00, 0.93, 0.68), smoothstep(0.72, 1.0, temperature));
    /* density * temperature^2: luminous where it is both dense and hot, and
       translucent everywhere else. Note this is a SMALLER number than the
       density-squared it replaced -- typical values land near an eighth
       rather than a half -- so FLAME_CORE_RADIANCE moved with it. */
    vec3 emission = flame_color * (density * temperature * temperature * draw.material.z);

    /* --- burning fuel ---------------------------------------------------
       Charred, cracked, glowing in patches -- but only ON the head, under the
       collar, never spilling past its silhouette. Cracks are static in space
       because they are on solid fuel, and only pulse; and the whole term is
       barely flickered, because a torch head does not bob with the gas above
       it. */
    float on_head = collar * over_head * (1.0 - above_fuel);
    float crack = pow(max(1.0 - abs(dungeon_fbm(vec2(x * 26.0, y * 34.0 + 3.0))) * 5.0, 0.0), 2.0);
    float fuel_patch = smoothstep(0.15, 0.75, structure * 0.5 + 0.5);
    float ember_pulse = 0.62 + 0.38 * sin(t * 2.3 + x * 9.0);
    vec3 fuel_emission = on_head * ember_pulse *
                         (vec3(0.90, 0.12, 0.015) * crack * 1.2 +
                          vec3(1.00, 0.40, 0.080) * fuel_patch * 1.4);

    /* --- embers --------------------------------------------------------
       Small, dim most of the time, occasionally very bright, stretched along
       the direction of travel and cooling white -> orange -> deep red. Round
       dots of equal brightness read as game particles rather than sparks. */
    vec3 embers = vec3(0.0);
    float ember_coverage = 0.0;
    for (int i = 0; i < EMBER_COUNT; ++i) {
        float seed = flame_hash(float(i) + draw.elevation_uv.y * 13.0);
        float bright = flame_hash(float(i) * 3.1 + 5.0 + draw.elevation_uv.y * 7.0);
        float life = fract(t * (0.30 + 0.28 * seed) + seed);
        float ex = (seed - 0.5) * 0.13 + 0.09 * sin(t * 1.7 + float(i) * 2.2) * life;
        float ey = FLAME_BASE + 0.14 + life * 0.58;
        /* Stretched vertically: a few pixels of motion blur, not a streak. */
        vec2 to_ember = vec2(warped_x - ex, y - ey) / vec2(0.0035, 0.0075);
        float spark = exp(-dot(to_ember, to_ember));
        float cooling = (1.0 - life) * smoothstep(0.0, 0.10, life);
        vec3 ember_color = mix(vec3(1.0, 0.85, 0.55), vec3(1.0, 0.32, 0.04),
                               smoothstep(0.0, 0.45, life));
        ember_color = mix(ember_color, vec3(0.55, 0.045, 0.01), smoothstep(0.45, 1.0, life));
        embers += ember_color * spark * cooling * mix(1.5, 11.0, bright * bright);
        ember_coverage = max(ember_coverage, spark * cooling);
    }

    /* Coverage bounds where this quad may touch the motion attachment, and is
       deliberately a gentler curve than the emission above: a region can be
       luminous without being opaque, which is what lets the wall show through
       the low-density parts of the fire instead of a solid orange slab. */
    float coverage = max(max(smoothstep(0.18, 0.75, density) * 0.70, ember_coverage),
                         min(on_head * (crack + fuel_patch), 1.0) * 0.6);
    if (coverage < 0.004)
        discard;

    /* Premultiplied: the pipeline blends SRC_ALPHA/ONE, so an alpha of one
       makes this a straight add of already-masked radiance. */
    vec3 hdr = (emission + embers) * flicker + fuel_emission * mix(1.0, flicker, 0.35);
    hdr *= draw.geometry.rgb;
    out_color = vec4(hdr, 1.0);
    /* Animated every frame: reject temporal history rather than smear it. */
    out_motion = vec2(2.0);

    /* See SHADER_DUMP_LEGEND["dungeon_flame"] in renderer.c for the f0..f19
       layout. The shape is what is hard to judge by eye at thirty pixels
       across, so the density field, the temperature islands and the flicker
       the CPU handed down are all readable numerically. */
    shader_dump(DUMP_SHADER_DUNGEON_FLAME,
                vec4(density, temperature, cavity, coverage),
                vec4(on_head, ember_coverage, flicker, local_width),
                vec4(flame_color, structure),
                vec4(draw.geometry.rgb, warped_x),
                vec4(hdr, tau));
}
