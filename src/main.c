#include <SDL3/SDL.h>
#include <stdio.h>
#include <stdlib.h>

#include "camera.h"
#include "input.h"
#include "renderer.h"
#include "terrain_runtime.h"

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

int main(void) {
    if (!SDL_Init(SDL_INIT_VIDEO)) { fprintf(stderr, "SDL_Init: %s\n", SDL_GetError()); return EXIT_FAILURE; }
    SDL_Window *window = SDL_CreateWindow("Vulkan Cube", WINDOW_WIDTH, WINDOW_HEIGHT,
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!window) { fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError()); return EXIT_FAILURE; }
    if (!SDL_SetWindowRelativeMouseMode(window, true))
        fprintf(stderr, "Relative mouse mode unavailable: %s\n", SDL_GetError());

    Renderer renderer;
    renderer_init(&renderer, window);
    Terrain terrain = terrain_create(&renderer);
    /* Overlook the ~1 km field from above one corner, looking down into it. */
    Camera camera = {
        .position = coordinate_local_to_world(&terrain.base.local_to_world,
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
    mat4s previous_local_to_camera_relative = GLMS_MAT4_IDENTITY_INIT;
    bool history_valid = false;
    bool depth_debug = false;
    bool running = true;
    while (running) {
        input_poll(&input, window);
        if (input.quit) running = false;
        if (input.reload_shaders) renderer_reload_pipeline(&renderer);
        if (input.toggle_depth_debug) depth_debug = !depth_debug;

        uint64_t ticks = SDL_GetTicksNS();
        float dt = (float)(ticks - previous_ticks) / 1000000000.0f;
        previous_ticks = ticks;
        if (dt > 0.1f) dt = 0.1f;

        camera_update(&camera, input.move_forward, input.move_right,
                      input.look_dx, input.look_dy, input.sprint, dt);

        mat4s projection = camera_projection(&camera, renderer_aspect(&renderer));
        mat4s view = camera_view(&camera);
        mat4s view_projection = glms_mat4_mul(projection, view);
        mat4s local_to_camera_relative = coordinate_local_to_camera_relative(
            &terrain.base.local_to_world, camera.position);
        if (!history_valid) {
            previous_projection = projection;
            previous_view = view;
            previous_view_projection = view_projection;
            previous_local_to_camera_relative = local_to_camera_relative;
            history_valid = true;
        }

        FrameUniforms frame = {
            .projection = projection,
            .view = view,
            .view_projection = view_projection,
            .inverse_view_projection = glms_mat4_inv(view_projection),
            .previous_projection = previous_projection,
            .previous_view = previous_view,
            .previous_view_projection = previous_view_projection,
            .local_to_camera_relative = local_to_camera_relative,
            .previous_local_to_camera_relative = previous_local_to_camera_relative,
            /* Fixed afternoon sun until the sky phase drives it. */
            .sun_direction = glms_vec4(glms_vec3_normalize((vec3s){{-0.4f, -1.0f, -0.3f}}), 0.0f),
            .time = (float)(ticks - start_ticks) / 1000000000.0f,
            .near_plane = CAMERA_NEAR_PLANE,
            .depth_debug = depth_debug ? 1.0f : 0.0f,
        };
        renderer_draw_frame(&renderer, &frame, &terrain.base, input.resized);

        previous_projection = projection;
        previous_view = view;
        previous_view_projection = view_projection;
        previous_local_to_camera_relative = local_to_camera_relative;
    }

    renderer_wait_idle(&renderer);
    terrain_destroy(&renderer, &terrain);
    renderer_shutdown(&renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return EXIT_SUCCESS;
}
