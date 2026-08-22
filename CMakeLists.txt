cmake_minimum_required(VERSION 3.20)
project(terrain_renderer C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)
find_package(cglm REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL3 REQUIRED IMPORTED_TARGET sdl3)

find_program(GLSLC glslc REQUIRED)
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")

# Debug builds keep symbols and stay unoptimised for readable RenderDoc/validation
# output; every other config optimises. Single-config generator, so decide here.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(GLSLC_FLAGS -g -O0)
else()
    set(GLSLC_FLAGS -O)
endif()

# Explicit shader permutation list. Add permutations here rather than growing
# nested #ifdef combinations with no inventory. Shared includes (common.glsl)
# are picked up automatically via the per-shader depfile below, so editing an
# include rebuilds every dependent shader.
foreach(SHADER cube.vert cube.frag terrain.vert terrain.frag)
    add_custom_command(
        OUTPUT "${SHADER_OUTPUT_DIR}/${SHADER}.spv"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUTPUT_DIR}"
        COMMAND "${GLSLC}" ${GLSLC_FLAGS}
                -MD -MF "${SHADER_OUTPUT_DIR}/${SHADER}.spv.d"
                "${SHADER_DIR}/${SHADER}" -o "${SHADER_OUTPUT_DIR}/${SHADER}.spv"
        DEPFILE "${SHADER_OUTPUT_DIR}/${SHADER}.spv.d"
        DEPENDS "${SHADER_DIR}/${SHADER}"
        VERBATIM)
    list(APPEND SHADER_BINARIES "${SHADER_OUTPUT_DIR}/${SHADER}.spv")
endforeach()

add_custom_target(shaders ALL DEPENDS ${SHADER_BINARIES})

add_executable(terrain_renderer
    src/main.c
    src/input.c
    src/camera.c
    src/gpu_memory.c
    src/gpu_buffer.c
    src/upload.c
    src/mesh.c
    src/cube.c
    src/terrain.c
    src/texture.c
    src/renderer.c
    include/stb_image.c)
add_dependencies(terrain_renderer shaders)
target_include_directories(terrain_renderer PRIVATE include)
target_link_libraries(terrain_renderer PRIVATE Vulkan::Vulkan cglm::cglm PkgConfig::SDL3 m)
target_compile_definitions(terrain_renderer PRIVATE
    SHADER_DIR="${SHADER_OUTPUT_DIR}"
    ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets"
    CGLM_FORCE_DEPTH_ZERO_TO_ONE)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(terrain_renderer PRIVATE -Wall -Wextra -Wpedantic)
    # Vendored third-party header; not held to our warning bar.
    set_source_files_properties(include/stb_image.c PROPERTIES COMPILE_OPTIONS "-w")
endif()
