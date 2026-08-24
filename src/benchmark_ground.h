#pragma once
#include "mesh.h"
struct Renderer;
typedef struct { Mesh mesh; Vertex vertices[4]; uint32_t indices[6]; } BenchmarkGround;
bool benchmark_ground_create(struct Renderer *renderer, BenchmarkGround *out, const Mesh *quarry);
void benchmark_ground_destroy(struct Renderer *renderer, BenchmarkGround *ground);
