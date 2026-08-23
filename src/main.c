#include <SDL3/SDL.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "camera.h"
#include "atmosphere.h"
#include "input.h"
#include "renderer.h"
#include "terrain_runtime.h"

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

int main(void) {
    if (!SDL_Init(SDL_INIT_VIDEO)) { fprintf(stderr, "SDL_Init: %s\n", SDL_GetError()); return EXIT_FAILURE; }
    SDL_Window *window = SDL_CreateWindow("Terrain Renderer", WINDOW_WIDTH, WINDOW_HEIGHT,
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!window) { fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError()); return EXIT_FAILURE; }
    if (!SDL_SetWindowRelativeMouseMode(window, true))
        fprintf(stderr, "Relative mouse mode unavailable: %s\n", SDL_GetError());

    Renderer renderer;
    renderer_init(&renderer, window);
    TerrainRuntime *terrain = terrain_runtime_create(&renderer, TRN_DIR, NULL);
    if (!terrain) {
        fprintf(stderr, "Could not create terrain quadtree from %s\n", TRN_DIR);
        renderer_shutdown(&renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return EXIT_FAILURE;
    }
    LocalToWorldTransform root_transform = terrain_runtime_root_transform(terrain);
    /* Overlook the ~1 km field from above one corner, looking down into it. */
    Camera camera = {
        .position = coordinate_local_to_world(&root_transform,
                                               (TileLocalPosition){-600, 500, -600}),
        .yaw = 45.0f,
        .pitch = -30.0f,
    };

    Input input = {0};
    uint64_t start_ticks = SDL_GetTicksNS();
    uint64_t previous_ticks = start_ticks;
    mat4s previous_projection = GLMS_MAT4_IDENTITY_INIT;
    mat4s previous_view = GLMS_MAT4_IDENTITY_INIT;
    mat4s previous_view_projection = GLMS_MAT4_IDENTITY_INIT;
    bool history_valid = false;
    unsigned debug_mode = 0;
    unsigned relight_mode = 1;
    unsigned sun_mode = 0;
    unsigned atmosphere_slice = 15;
    AtmosphereParameters atmosphere = atmosphere_earth();
    uint64_t terrain_frame = 0;
    uint64_t last_stats_log = 0;
    TerrainRuntimeStats previous_stats = {0};
    bool running = true;
    while (running) {
        input_poll(&input, window);
        if (input.quit) running = false;
        if (input.reload_shaders) renderer_reload_pipeline(&renderer);
        if (input.toggle_depth_debug)
            debug_mode = debug_mode == 1u ? 0u : 1u;
        if (input.toggle_lod_debug)
            debug_mode = debug_mode == 2u ? 0u : 2u;
        if (input.cycle_relighting) {
            relight_mode = (relight_mode + 1u) % 3u;
            const char *names[] = {"unlit map", "subtle relight", "material"};
            printf("Terrain imagery: %s\n", names[relight_mode]);
        }
        if (input.cycle_surface_debug)
            debug_mode = debug_mode >= 3u && debug_mode < 6u
                ? debug_mode + 1u : (debug_mode == 6u ? 0u : 3u);
        if (input.cycle_shadow_debug)
            debug_mode = debug_mode >= 7u && debug_mode < 11u
                ? debug_mode + 1u : (debug_mode == 11u ? 0u : 7u);
        if (input.cycle_atmosphere_debug)
            debug_mode = debug_mode >= 12u && debug_mode < 16u
                ? debug_mode + 1u : (debug_mode == 16u ? 0u : 12u);
        if (input.previous_atmosphere_slice && atmosphere_slice > 0u)
            --atmosphere_slice;
        if (input.next_atmosphere_slice && atmosphere_slice < 31u)
            ++atmosphere_slice;
        if (input.previous_atmosphere_slice || input.next_atmosphere_slice)
            printf("Atmosphere volume debug slice: %u/31\n", atmosphere_slice);
        if (input.cycle_sun) {
            sun_mode = (sun_mode + 1u) % 3u;
            const char *names[] = {"afternoon", "sunset", "high sun"};
            printf("Atmosphere sun: %s\n", names[sun_mode]);
        }

        uint64_t ticks = SDL_GetTicksNS();
        float dt = (float)(ticks - previous_ticks) / 1000000000.0f;
        previous_ticks = ticks;
        if (dt > 0.1f) dt = 0.1f;

        camera_update(&camera, input.move_forward, input.move_right,
                      input.look_dx, input.look_dy, input.sprint, dt);

        vec3s camera_forward_direction = camera_forward(&camera);
        TerrainQuadtreeView terrain_view = {
            .camera_world = camera.position,
            .forward = {camera_forward_direction.x, camera_forward_direction.y,
                        camera_forward_direction.z},
            .up = {0.0, 1.0, 0.0},
            .vertical_fov_radians = glm_rad(60.0f),
            .aspect = renderer_aspect(&renderer),
            .near_plane_m = CAMERA_NEAR_PLANE,
            .viewport_height_px = renderer.swapchain_extent.height,
        };
        terrain_runtime_update(terrain, &terrain_view, terrain_frame++);
        uint32_t terrain_draw_count = 0;
        const RendererDraw *terrain_draws =
            terrain_runtime_draws(terrain, &terrain_draw_count);

        mat4s projection = camera_projection(&camera, renderer_aspect(&renderer));
        mat4s view = camera_view(&camera);
        mat4s view_projection = glms_mat4_mul(projection, view);
        if (!history_valid) {
            previous_projection = projection;
            previous_view = view;
            previous_view_projection = view_projection;
            history_valid = true;
        }

        const vec3s sun_directions[3] = {
            {{-0.4f, -1.0f, -0.3f}},
            {{-1.0f, -0.08f, -0.15f}},
            {{-0.1f, -1.0f, -0.05f}},
        };
        vec3s sun_direction = glms_vec3_normalize(sun_directions[sun_mode]);
        ShadowCascadeConfig shadow_config =
            shadow_cascade_default_config(renderer_aspect(&renderer));
        ShadowCascadeSet shadow_cascades;
        if (!shadow_cascade_build(&shadow_config, camera.position,
                                  camera_forward_direction,
                                  (vec3s){{0.0f, 1.0f, 0.0f}},
                                  sun_direction, &shadow_cascades)) {
            fprintf(stderr, "Could not build sun shadow cascades\n");
            running = false;
            continue;
        }

        FrameUniforms frame = {
            .projection = projection,
            .view = view,
            .view_projection = view_projection,
            .inverse_view_projection = glms_mat4_inv(view_projection),
            .previous_projection = previous_projection,
            .previous_view = previous_view,
            .previous_view_projection = previous_view_projection,
            .local_to_camera_relative = GLMS_MAT4_IDENTITY_INIT,
            .previous_local_to_camera_relative = GLMS_MAT4_IDENTITY_INIT,
            .sun_direction = glms_vec4(sun_direction, 0.0f),
            .time = (float)(ticks - start_ticks) / 1000000000.0f,
            .near_plane = CAMERA_NEAR_PLANE,
            .debug_view = (float)debug_mode,
            .relight_strength = (const float[]){0.0f, 0.35f, 1.0f}[relight_mode],
            .shadow_splits = (vec4s){{
                shadow_config.split_m[0], shadow_config.split_m[1],
                shadow_config.split_m[2], shadow_config.split_m[3]}},
            .shadow_parameters = (vec4s){{0.35f, 1.75f, 1.0f, 1.25f}},
            .sun_radiance = (vec4s){{3.2f, 3.0f, 2.7f, 0.0f}},
            .atmosphere_radii = (vec4s){{
                atmosphere.bottom_radius_km, atmosphere.top_radius_km,
                fmaxf((float)camera.position.y * 0.001f, 0.001f),
                atmosphere.sun_angular_radius_rad}},
            .atmosphere_rayleigh = (vec4s){{
                atmosphere.rayleigh_scattering[0],
                atmosphere.rayleigh_scattering[1],
                atmosphere.rayleigh_scattering[2],
                atmosphere.rayleigh_density_exp_scale}},
            .atmosphere_mie_scatter = (vec4s){{
                atmosphere.mie_scattering[0], atmosphere.mie_scattering[1],
                atmosphere.mie_scattering[2], atmosphere.mie_density_exp_scale}},
            .atmosphere_mie_extinct = (vec4s){{
                atmosphere.mie_extinction[0], atmosphere.mie_extinction[1],
                atmosphere.mie_extinction[2], atmosphere.mie_phase_g}},
            .atmosphere_absorption = (vec4s){{
                atmosphere.absorption_extinction[0],
                atmosphere.absorption_extinction[1],
                atmosphere.absorption_extinction[2], 0.0f}},
            .atmosphere_ground = (vec4s){{
                atmosphere.ground_albedo[0], atmosphere.ground_albedo[1],
                atmosphere.ground_albedo[2],
                atmosphere.multiple_scattering_factor}},
            .atmosphere_options = (vec4s){{
                atmosphere.aerial_max_distance_km, (float)atmosphere_slice,
                0.0f, 0.0f}},
        };
        for (uint32_t i = 0; i < SHADOW_CASCADE_COUNT; ++i)
            frame.shadow_view_projection[i] = shadow_cascades.view_projection[i];
        renderer_draw_frame(&renderer, &frame, terrain_draws,
                            terrain_draw_count, input.resized);
        terrain_runtime_collect_evictions(terrain);

        TerrainRuntimeStats stats = terrain_runtime_stats(terrain);
        bool stats_changed = stats.resident_tiles != previous_stats.resident_tiles ||
                             stats.drawn_tiles != previous_stats.drawn_tiles;
        if (stats_changed &&
            (last_stats_log == 0 || ticks - last_stats_log >= UINT64_C(1000000000))) {
            printf("Terrain: known=%u resident=%u drawn=%u CPU=%.2f MiB GPU=%.2f MiB\n",
                   stats.known_tiles, stats.resident_tiles, stats.drawn_tiles,
                   (double)stats.cpu_bytes / (1024.0 * 1024.0),
                   (double)stats.gpu_bytes / (1024.0 * 1024.0));
            previous_stats = stats;
            last_stats_log = ticks;
        }

        previous_projection = projection;
        previous_view = view;
        previous_view_projection = view_projection;
    }

    renderer_wait_idle(&renderer);
    terrain_runtime_destroy(terrain);
    renderer_shutdown(&renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return EXIT_SUCCESS;
}
