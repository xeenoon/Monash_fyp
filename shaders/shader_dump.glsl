/* Shared per-fragment diagnostic sink (enabled in regular builds by default). Any fragment shader
   can #include this and call shader_dump() unconditionally: outside
   DEBUG_SHADER_DUMP builds it compiles to a no-op, so call sites never need
   their own #ifdef. Requires "common.glsl" to be included first (needs `frame`).

   Layout must stay byte-identical to struct DumpRecord in renderer.c (std430:
   six 16-byte vec4s = 96 bytes: meta + five generic vec4s). The CSV writer
   documents each shader's v0..v4 meaning; keep that legend in sync with the
   shader_dump() call in each instrumented .frag file. */
#ifndef SHADER_DUMP_GLSL
#define SHADER_DUMP_GLSL

/* Shader IDs for the CSV "shader" column. Kept in sync with shader_dump_name()
   in renderer.c. */
#define DUMP_SHADER_TERRAIN 0.0
#define DUMP_SHADER_MESH 1.0
#define DUMP_SHADER_CUBE 2.0
#define DUMP_SHADER_ATMOSPHERE_COMPOSITE 3.0
#define DUMP_SHADER_TONEMAP 4.0
#define DUMP_SHADER_TEMPORAL_RESOLVE 5.0
#define DUMP_SHADER_ENVIRONMENT_IBL 6.0
#define DUMP_SHADER_MATERIAL_DETAIL 7.0
#define DUMP_SHADER_DUNGEON_SURFACE 8.0
#define DUMP_SHADER_DUNGEON_PUDDLE 9.0
#define DUMP_SHADER_POINT_SHADOW 10.0

#ifdef DEBUG_SHADER_DUMP

struct DumpRecord {
    vec4 meta;               /* shader_id, frag_x, frag_y, reserved */
    vec4 v0, v1, v2, v3, v4; /* 20 generic floats, meaning documented per-shader */
};

layout(std430, set = 0, binding = 3) buffer DumpBuffer {
    uint dump_count;
    uint dump_capacity;
    uint dump_pad0;
    uint dump_pad1;
    DumpRecord dump_records[];
};

bool dump_enabled() {
    return frame.shader_dump.x > 0.5;
}

void shader_dump(float shader_id, vec4 v0, vec4 v1, vec4 v2, vec4 v3, vec4 v4) {
    if (!dump_enabled()) return;
    uint index = atomicAdd(dump_count, 1u);
    if (index >= dump_capacity) return;
    dump_records[index].meta = vec4(shader_id, gl_FragCoord.x, gl_FragCoord.y, 0.0);
    dump_records[index].v0 = v0;
    dump_records[index].v1 = v1;
    dump_records[index].v2 = v2;
    dump_records[index].v3 = v3;
    dump_records[index].v4 = v4;
}

#else

bool dump_enabled() {
    return false;
}

void shader_dump(float shader_id, vec4 v0, vec4 v1, vec4 v2, vec4 v3, vec4 v4) {
    /* no-op when shader-dump support is explicitly disabled */
}

#endif // DEBUG_SHADER_DUMP

#endif // SHADER_DUMP_GLSL
