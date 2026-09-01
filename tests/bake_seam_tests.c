#include <stdint.h>
#include <stdio.h>

#include "bake_seam.h"

static int check(int width, int height, const int16_t *expected) {
    float cost[11 * 8];
    int16_t seam[11];
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            float value = (float)((x * 17 + y * 31 + x * y * 7) % 23) / 11.0f;
            cost[y * width + x] = x == width / 2 ? value * 0.25f : value;
        }
    if (!bake_minimum_vertical_seam(cost, width, height, seam)) return 1;
    for (int y = 0; y < height; ++y) {
        if (seam[y] != expected[y]) {
            fprintf(stderr, "%dx%d row %d: got %d, expected %d\n",
                    width, height, y, seam[y], expected[y]);
            return 1;
        }
    }
    return 0;
}

int main(void) {
    static const int16_t first[] = {2, 2, 2, 3, 2, 1, 2};
    static const int16_t second[] = {3, 4, 4, 3, 4, 5, 4, 3, 2, 2, 2};
    if (check(5, 7, first) || check(8, 11, second)) return 1;
    puts("bake_seam_tests: OK (pass17 dynamic-programming order)");
    return 0;
}
