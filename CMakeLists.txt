cmake_minimum_required(VERSION 3.20)
project(vulkan_cube C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(cglm REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL3 REQUIRED IMPORTED_TARGET sdl3)

find_program(GLSLC glslc REQUIRED)
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")

foreach(SHADER cube.vert cube.frag)
    add_custom_command(
        OUTPUT "${SHADER_OUTPUT_DIR}/${SHADER}.spv"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUTPUT_DIR}"
        COMMAND "${GLSLC}" "${SHADER_DIR}/${SHADER}" -o "${SHADER_OUTPUT_DIR}/${SHADER}.spv"
        DEPENDS "${SHADER_DIR}/${SHADER}"
        VERBATIM)
    list(APPEND SHADER_BINARIES "${SHADER_OUTPUT_DIR}/${SHADER}.spv")
endforeach()

add_custom_target(shaders ALL DEPENDS ${SHADER_BINARIES})

add_executable(vulkan_cube
    src/main.c
    src/input.c
    src/camera.c
    src/mesh.c
    src/cube.c
    src/terrain.c
    src/renderer.c)
add_dependencies(vulkan_cube shaders)
target_link_libraries(vulkan_cube PRIVATE Vulkan::Vulkan cglm::cglm PkgConfig::SDL3 m)
target_compile_definitions(vulkan_cube PRIVATE
    SHADER_DIR="${SHADER_OUTPUT_DIR}"
    CGLM_FORCE_DEPTH_ZERO_TO_ONE)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(vulkan_cube PRIVATE -Wall -Wextra -Wpedantic)
endif()
