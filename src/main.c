#include <SDL3/SDL.h>
#include <stdio.h>
#include <stdlib.h>

#include "camera.h"
#include "input.h"
#include "renderer.h"
#include "terrain.h"

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
    Camera camera = { .position = {{-600, 500, -600}}, .yaw = 45.0f, .pitch = -30.0f };

    Input input = {0};
    uint64_t previous_ticks = SDL_GetTicksNS();
    bool running = true;
    while (running) {
        input_poll(&input, window);
        if (input.quit) running = false;

        uint64_t ticks = SDL_GetTicksNS();
        float dt = (float)(ticks - previous_ticks) / 1000000000.0f;
        previous_ticks = ticks;
        if (dt > 0.1f) dt = 0.1f;

        camera_update(&camera, input.move_forward, input.move_right,
                      input.look_dx, input.look_dy, input.sprint, dt);

        mat4s view_projection = glms_mat4_mul(
            camera_projection(&camera, renderer_aspect(&renderer)),
            camera_view(&camera));
        renderer_draw_frame(&renderer, &view_projection, &terrain.base, input.resized);
    }

    renderer_wait_idle(&renderer);
    mesh_destroy(&renderer, &terrain.base);
    renderer_shutdown(&renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return EXIT_SUCCESS;
}
