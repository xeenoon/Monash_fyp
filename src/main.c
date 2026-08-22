#include <SDL3/SDL.h>
#include <stdio.h>
#include <stdlib.h>

#include "camera.h"
#include "cube.h"
#include "input.h"
#include "renderer.h"

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
    Cube cube = cube_create(&renderer);
    Camera camera = { .position = {{0, 0, 5}}, .yaw = -90.0f, .pitch = 0.0f };

    Input input = {0};
    Uint64 previous_ticks = SDL_GetTicksNS();
    bool running = true;
    while (running) {
        input_poll(&input, window);
        if (input.quit) running = false;

        Uint64 ticks = SDL_GetTicksNS();
        float dt = (float)(ticks - previous_ticks) / 1000000000.0f;
        previous_ticks = ticks;
        if (dt > 0.1f) dt = 0.1f;

        camera_update(&camera, input.move_forward, input.move_right,
                      input.look_dx, input.look_dy, input.sprint, dt);

        mat4s view_projection = glms_mat4_mul(
            camera_projection(&camera, renderer_aspect(&renderer)),
            camera_view(&camera));
        renderer_draw_frame(&renderer, &view_projection, &cube.base, input.resized);
    }

    renderer_wait_idle(&renderer);
    mesh_destroy(&renderer, &cube.base);
    renderer_shutdown(&renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return EXIT_SUCCESS;
}
