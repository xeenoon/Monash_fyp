cmake_minimum_required(VERSION 3.20)
project(terrain_renderer C)

include(CTest)

# Debug-only diagnostic: every instrumented fragment shader (see shaders/shader_dump.glsl)
# writes a per-fragment record buffer that the 'X' key dumps to a CSV file (cleared
# with 'C' or a clean build). Off by default; enable with -DDEBUG_SHADER_DUMP=ON.
# See renderer_dump_shader_data.
option(DEBUG_SHADER_DUMP "Enable the 'X'-key per-fragment shader data dump" OFF)
option(TERRAIN_STRETCH_OVERLAY
    "Compile the diagnostic red terrain texture-stretch overlay" OFF)
option(TERRAIN_GOLDEN
    "Boot the modern-render/legacy-sampling golden terrain dataset by default" ON)
option(TERRAIN_BAKE
    "Build the headless Vulkan compute terrain baker (terrain_bake) and its tests" OFF)
set(SHADER_DUMP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/debug_dumps")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)
find_package(cglm REQUIRED)
find_package(ZLIB REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL3 REQUIRED IMPORTED_TARGET sdl3)

find_program(GLSLC glslc REQUIRED)
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")

set(TERRAIN_DATASET_DIR "" CACHE PATH
    "Override the runtime terrain tile dataset (takes precedence over TERRAIN_GOLDEN)")
if(TERRAIN_DATASET_DIR)
    set(ACTIVE_TERRAIN_DATASET_DIR "${TERRAIN_DATASET_DIR}")
    set(ACTIVE_TERRAIN_DATASET_NAME "custom")
elseif(TERRAIN_GOLDEN)
    set(ACTIVE_TERRAIN_DATASET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/trn-golden")
    set(ACTIVE_TERRAIN_DATASET_NAME "modern legacy sampling/145a86b")
else()
    set(ACTIVE_TERRAIN_DATASET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/trn")
    set(ACTIVE_TERRAIN_DATASET_NAME "current")
endif()
message(STATUS "Terrain dataset: ${ACTIVE_TERRAIN_DATASET_NAME} (${ACTIVE_TERRAIN_DATASET_DIR})")

if(NOT EXISTS "${ACTIVE_TERRAIN_DATASET_DIR}/tiles/0/0/0.trn")
    message(WARNING
        "The selected runtime terrain tile is missing under "
        "'${ACTIVE_TERRAIN_DATASET_DIR}'. Generate or restore that dataset before running.")
endif()
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/textures/runtime/terrain_micro_albedo.png")
    message(WARNING
        "Terrain micro atlases are missing. Download both manifests as documented in "
        "textures/README.md, then run 'python3 tools/build_terrain_micro_atlas.py'.")
endif()

# Debug builds keep symbols and stay unoptimised for readable RenderDoc/validation
# output; every other config optimises. Single-config generator, so decide here.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(GLSLC_FLAGS -g -O0)
else()
    set(GLSLC_FLAGS -O)
endif()
if(DEBUG_SHADER_DUMP)
    list(APPEND GLSLC_FLAGS -DDEBUG_SHADER_DUMP)
endif()
if(TERRAIN_STRETCH_OVERLAY)
    list(APPEND GLSLC_FLAGS -DTERRAIN_STRETCH_OVERLAY)
endif()

# Explicit shader permutation list. Add permutations here rather than growing
# nested #ifdef combinations with no inventory. Shared includes (common.glsl)
# are picked up automatically via the per-shader depfile below, so editing an
# include rebuilds every dependent shader.
foreach(SHADER cube.vert cube.frag terrain.vert terrain.frag mesh.vert mesh.frag shadow.vert
               fullscreen.vert tonemap.frag atmosphere_composite.frag
               temporal_resolve.frag luminance_histogram.comp exposure.comp
               atmosphere_transmittance.comp
               atmosphere_multiscattering.comp atmosphere_skyview.comp
               atmosphere_aerial.comp
               environment_equirect_to_cube.comp environment_prefilter.comp)
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

# --- Headless Vulkan compute terrain baker (opt-in: -DTERRAIN_BAKE=ON) --------
if(TERRAIN_BAKE)
    set(BAKE_SHADER_BINARIES "")
    foreach(BAKE_SHADER bake_scale.comp bake_copy.comp bake_classify.comp
                        bake_gaussian.comp bake_jfa.comp bake_edt_horizontal.comp
                        bake_edt_vertical.comp bake_gaussian_batch.comp
                        bake_residual_extract.comp bake_residual_finish.comp
                        bake_residual_score.comp bake_residual_seam_cost.comp
                        bake_residual_commit.comp bake_green.comp
                        bake_clean_masks.comp bake_jfa_init.comp
                        bake_inpaint.comp bake_clean_finish.comp
                        bake_unpack.comp bake_bands.comp bake_subtract.comp
                        bake_quilt.comp bake_relayer.comp bake_patchmatch.comp
                        bake_gather.comp bake_transition_seed.comp
                        bake_transition_finish.comp bake_material_mask.comp
                        bake_resample.comp)
        add_custom_command(
            OUTPUT "${SHADER_OUTPUT_DIR}/${BAKE_SHADER}.spv"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUTPUT_DIR}"
            COMMAND "${GLSLC}" ${GLSLC_FLAGS}
                    -MD -MF "${SHADER_OUTPUT_DIR}/${BAKE_SHADER}.spv.d"
                    "${SHADER_DIR}/${BAKE_SHADER}" -o "${SHADER_OUTPUT_DIR}/${BAKE_SHADER}.spv"
            DEPFILE "${SHADER_OUTPUT_DIR}/${BAKE_SHADER}.spv.d"
            DEPENDS "${SHADER_DIR}/${BAKE_SHADER}"
            VERBATIM)
        list(APPEND BAKE_SHADER_BINARIES "${SHADER_OUTPUT_DIR}/${BAKE_SHADER}.spv")
    endforeach()
    add_custom_target(bake_shaders ALL DEPENDS ${BAKE_SHADER_BINARIES})

    add_library(bake_gpu STATIC
        src/bake/bake_gpu.c
        src/bake/bake_image.c
        src/bake/bake_filter.c
        src/bake/bake_jfa.c
        src/bake/bake_clean.c
        src/bake/bake_bands.c
        src/bake/bake_random.c
        src/bake/bake_pass17_sources.c
        src/bake/bake_pass17_relayer.c
        src/bake/bake_sample.c
        src/bake/bake_residual.c
        src/bake/bake_ordered_quilt.c
        src/bake/bake_texture_database.c
        src/bake/bake_texture_score.c
        src/bake/bake_seam.c
        src/bake/bake_quilt.c
        src/bake/bake_relayer.c
        src/bake/bake_patchmatch.c
        src/bake/bake_transition.c
        src/bake/bake_pipeline.c
        src/bake/bake_resample.c
        include/stb_image.c)
    target_include_directories(bake_gpu PUBLIC src src/bake PRIVATE include)
    target_link_libraries(bake_gpu PUBLIC core_utils Vulkan::Vulkan m)
    target_compile_definitions(bake_gpu PUBLIC BAKE_SHADER_DIR="${SHADER_OUTPUT_DIR}")
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(bake_gpu PRIVATE -Wall -Wextra -Wpedantic)
        set_source_files_properties(include/stb_image.c PROPERTIES COMPILE_OPTIONS "-w")
        # Bit-exact GPU match requires no fused multiply-add on the CPU side.
        set_source_files_properties(src/bake/bake_filter.c
            PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
        set_source_files_properties(src/bake/bake_clean.c
            PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
        set_source_files_properties(src/bake/bake_bands.c
            PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
        set_source_files_properties(src/bake/bake_relayer.c
            PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
        set_source_files_properties(src/bake/bake_resample.c
            PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    endif()

    add_executable(terrain_bake src/bake/bake_main.c src/terrain_tile.c)
    add_dependencies(terrain_bake bake_shaders)
    target_link_libraries(terrain_bake PRIVATE bake_gpu)
    target_compile_definitions(terrain_bake PRIVATE
        BAKE_DATASET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/alps-data/trn-alps-16km")

    add_executable(bake_pass17_compare src/bake/bake_compare_main.c)
    target_link_libraries(bake_pass17_compare PRIVATE bake_gpu)

    add_executable(bake_pass17_texture
        src/bake/bake_pass17_texture_main.c src/terrain_tile.c)
    add_dependencies(bake_pass17_texture bake_shaders)
    target_link_libraries(bake_pass17_texture PRIVATE bake_gpu)
    target_compile_definitions(bake_pass17_texture PRIVATE
        BAKE_DATASET_DIR="${TERRAIN_DATASET_DIR}")

    # `cmake --build . --target bake` runs the baker to produce images.
    add_custom_target(bake
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/baked_materials"
        COMMAND terrain_bake --output "${CMAKE_BINARY_DIR}/baked_materials"
        DEPENDS terrain_bake bake_shaders)

    if(BUILD_TESTING)
        add_executable(bake_gpu_tests tests/bake_gpu_tests.c)
        add_dependencies(bake_gpu_tests bake_shaders)
        target_link_libraries(bake_gpu_tests PRIVATE bake_gpu)
        add_test(NAME bake_gpu_roundtrip COMMAND bake_gpu_tests)
        set_tests_properties(bake_gpu_roundtrip PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(bake_image_tests tests/bake_image_tests.c)
        add_dependencies(bake_image_tests bake_shaders)
        target_link_libraries(bake_image_tests PRIVATE bake_gpu)
        add_test(NAME bake_image_roundtrip COMMAND bake_image_tests)
        set_tests_properties(bake_image_roundtrip PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(bake_classify_tests tests/bake_classify_tests.c)
        add_dependencies(bake_classify_tests bake_shaders)
        target_link_libraries(bake_classify_tests PRIVATE bake_gpu)
        add_test(NAME bake_classify_oracle COMMAND bake_classify_tests)
        set_tests_properties(bake_classify_oracle PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(bake_gaussian_tests tests/bake_gaussian_tests.c)
        add_dependencies(bake_gaussian_tests bake_shaders)
        target_link_libraries(bake_gaussian_tests PRIVATE bake_gpu)
        add_test(NAME bake_gaussian_bitexact COMMAND bake_gaussian_tests)
        set_tests_properties(bake_gaussian_bitexact PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(bake_jfa_tests tests/bake_jfa_tests.c)
        add_dependencies(bake_jfa_tests bake_shaders)
        target_link_libraries(bake_jfa_tests PRIVATE bake_gpu)
        add_test(NAME bake_jfa_bitexact COMMAND bake_jfa_tests)
        set_tests_properties(bake_jfa_bitexact PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(bake_clean_tests tests/bake_clean_tests.c)
        add_dependencies(bake_clean_tests bake_shaders)
        target_link_libraries(bake_clean_tests PRIVATE bake_gpu)
        add_test(NAME bake_clean_bitexact COMMAND bake_clean_tests)
        set_tests_properties(bake_clean_bitexact PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(bake_bands_tests tests/bake_bands_tests.c)
        add_dependencies(bake_bands_tests bake_shaders)
        target_link_libraries(bake_bands_tests PRIVATE bake_gpu)
        add_test(NAME bake_bands_bitexact COMMAND bake_bands_tests)
        set_tests_properties(bake_bands_bitexact PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(bake_quilt_tests tests/bake_quilt_tests.c)
        add_dependencies(bake_quilt_tests bake_shaders)
        target_link_libraries(bake_quilt_tests PRIVATE bake_gpu)
        add_test(NAME bake_quilt_bitexact COMMAND bake_quilt_tests)

        add_executable(bake_seam_tests tests/bake_seam_tests.c)
        target_link_libraries(bake_seam_tests PRIVATE bake_gpu)
        add_test(NAME bake_pass17_seam_order COMMAND bake_seam_tests)

        add_executable(bake_random_tests tests/bake_random_tests.c)
        target_link_libraries(bake_random_tests PRIVATE bake_gpu)
        add_test(NAME bake_pass17_random_stream COMMAND bake_random_tests)

        add_executable(bake_pass17_sources_tests tests/bake_pass17_sources_tests.c)
        target_link_libraries(bake_pass17_sources_tests PRIVATE bake_gpu)
        add_test(NAME bake_pass17_source_sets COMMAND bake_pass17_sources_tests)

        add_executable(bake_sample_tests tests/bake_sample_tests.c src/terrain_tile.c)
        add_dependencies(bake_sample_tests bake_shaders)
        target_link_libraries(bake_sample_tests PRIVATE bake_gpu)
        target_compile_definitions(bake_sample_tests PRIVATE
            BAKE_DATASET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/alps-data/trn-alps-16km")
        add_test(NAME bake_pass17_sample_loader COMMAND bake_sample_tests)
        set_tests_properties(bake_pass17_sample_loader PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(bake_residual_tests tests/bake_residual_tests.c)
        add_dependencies(bake_residual_tests bake_shaders)
        target_link_libraries(bake_residual_tests PRIVATE bake_gpu)
        add_test(NAME bake_pass17_residual_batch COMMAND bake_residual_tests)
        set_tests_properties(bake_pass17_residual_batch PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(bake_residual_score_tests tests/bake_residual_score_tests.c)
        add_dependencies(bake_residual_score_tests bake_shaders)
        target_link_libraries(bake_residual_score_tests PRIVATE bake_gpu)
        add_test(NAME bake_pass17_residual_scores COMMAND bake_residual_score_tests)
        set_tests_properties(bake_pass17_residual_scores PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(bake_residual_commit_tests tests/bake_residual_commit_tests.c)
        add_dependencies(bake_residual_commit_tests bake_shaders)
        target_link_libraries(bake_residual_commit_tests PRIVATE bake_gpu)
        add_test(NAME bake_pass17_ordered_commit COMMAND bake_residual_commit_tests)
        set_tests_properties(bake_pass17_ordered_commit PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(bake_ordered_quilt_tests tests/bake_ordered_quilt_tests.c)
        add_dependencies(bake_ordered_quilt_tests bake_shaders)
        target_link_libraries(bake_ordered_quilt_tests PRIVATE bake_gpu)
        add_test(NAME bake_pass17_ordered_scheduler COMMAND bake_ordered_quilt_tests)
        set_tests_properties(bake_pass17_ordered_scheduler PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(bake_texture_database_tests tests/bake_texture_database_tests.c)
        target_link_libraries(bake_texture_database_tests PRIVATE bake_gpu)
        add_test(NAME bake_pass17_texture_database COMMAND bake_texture_database_tests)
        set_tests_properties(bake_quilt_bitexact PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(bake_relayer_tests tests/bake_relayer_tests.c)
        add_dependencies(bake_relayer_tests bake_shaders)
        target_link_libraries(bake_relayer_tests PRIVATE bake_gpu)
        add_test(NAME bake_relayer_bitexact COMMAND bake_relayer_tests)
        set_tests_properties(bake_relayer_bitexact PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(bake_transition_tests tests/bake_transition_tests.c)
        add_dependencies(bake_transition_tests bake_shaders)
        target_link_libraries(bake_transition_tests PRIVATE bake_gpu)
        add_test(NAME bake_transition_bitexact COMMAND bake_transition_tests)
        set_tests_properties(bake_transition_bitexact PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(bake_pipeline_tests tests/bake_pipeline_tests.c)
        add_dependencies(bake_pipeline_tests bake_shaders)
        target_link_libraries(bake_pipeline_tests PRIVATE bake_gpu)
        add_test(NAME bake_pipeline_bitexact COMMAND bake_pipeline_tests)
        set_tests_properties(bake_pipeline_bitexact PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(bake_resample_tests tests/bake_resample_tests.c)
        add_dependencies(bake_resample_tests bake_shaders)
        target_link_libraries(bake_resample_tests PRIVATE bake_gpu)
        add_test(NAME bake_resample_bitexact COMMAND bake_resample_tests)
        set_tests_properties(bake_resample_bitexact PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

add_library(core_utils STATIC
    src/byte_utils.c
    src/checksum_utils.c
    src/file_utils.c
    src/path_utils.c
    src/size_utils.c
    src/str_utils.c)
target_include_directories(core_utils PUBLIC src)
target_link_libraries(core_utils PUBLIC ZLIB::ZLIB)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(core_utils PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(dungeon_level STATIC
    src/dungeon_level.c
    src/dungeon_grid.c
    src/dungeon_mesh.c)
target_include_directories(dungeon_level PUBLIC src)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(dungeon_level PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BUILD_TESTING)
    add_executable(dungeon_grid_tests tests/dungeon_grid_tests.c)
    target_link_libraries(dungeon_grid_tests PRIVATE dungeon_level)
    add_test(NAME dungeon_grid_compiler COMMAND dungeon_grid_tests)

    add_executable(dungeon_mesh_tests tests/dungeon_mesh_tests.c)
    target_link_libraries(dungeon_mesh_tests PRIVATE dungeon_level m)
    add_test(NAME dungeon_mesh_generation COMMAND dungeon_mesh_tests)
endif()

add_executable(terrain_renderer
    src/main.c
    src/input.c
    src/coordinate.c
    src/camera.c
    src/gpu_memory.c
    src/gpu_buffer.c
    src/upload.c
    src/mesh.c
    src/gltf_scene.c
    src/quarry.c
    src/benchmark_ground.c
    src/material_stability_demo.c
    src/terrain_tile.c
    src/terrain_grid.c
    src/terrain_quadtree.c
    src/cube.c
    src/terrain_runtime.c
    src/surface_detail.c
    src/shadow_cascade.c
    src/atmosphere.c
    src/environment.c
    src/temporal.c
    src/texture.c
    src/renderer.c
	 src/static_material.c
    third_party/mikktspace/mikktspace.c
    include/stb_image.c)
add_dependencies(terrain_renderer shaders)
target_include_directories(terrain_renderer PRIVATE include third_party/cgltf third_party/mikktspace)
target_link_libraries(terrain_renderer PRIVATE
    core_utils Vulkan::Vulkan cglm::cglm PkgConfig::SDL3 m)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets/env.hdr")
    message(WARNING
        "No IBL environment at assets/env.hdr -- the Quarry cliff will use the "
        "hemispheric ambient fallback (Phase B indirect lighting disabled).")
endif()

# Pinned decoder sources used by the strict static glTF importer.  cgltf:
# 85cd62382dfea638278962690cf515023f33ed00 (MIT); MikkTSpace:
# 3e895b49d05ea07e4c2133156cfa94369e19e409 (zlib).
target_compile_definitions(terrain_renderer PRIVATE
    SHADER_DIR="${SHADER_OUTPUT_DIR}"
    ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets"
    TEXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/textures"
    BENCHMARK_DIR="${CMAKE_CURRENT_SOURCE_DIR}/benchmark-assets"
    QUARRY_DIR="${CMAKE_CURRENT_SOURCE_DIR}/valley-of-the-ancient"
    TRN_DIR="${ACTIVE_TERRAIN_DATASET_DIR}"
    SHADOWED_TERRAIN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/alps-data/trn-alps-16km-shadowed"
    UNSHADOWED_ALPS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/alps-data/trn-alps-16km"
    GRADED_ALPS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/alps-data/trn-alps-16km-graded"
    ENV_HDR_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/env.hdr"
    CGLM_FORCE_DEPTH_ZERO_TO_ONE)
if(DEBUG_SHADER_DUMP)
    target_compile_definitions(terrain_renderer PRIVATE
        DEBUG_SHADER_DUMP
        SHADER_DUMP_PATH="${SHADER_DUMP_DIR}/shader_dump.txt")
    # A clean build wipes accumulated dumps, as does the in-app 'C' key.
    set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES "${SHADER_DUMP_DIR}")
endif()
if(TERRAIN_STRETCH_OVERLAY)
    target_compile_definitions(terrain_renderer PRIVATE TERRAIN_STRETCH_OVERLAY)
endif()

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(terrain_renderer PRIVATE -Wall -Wextra -Wpedantic)
    # Vendored third-party header; not held to our warning bar.
    set_source_files_properties(include/stb_image.c PROPERTIES COMPILE_OPTIONS "-w")
endif()

if(BUILD_TESTING)
    add_executable(utils_tests tests/utils_tests.c)
    target_link_libraries(utils_tests PRIVATE core_utils m)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(utils_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME reusable_utils COMMAND utils_tests)

    add_executable(phase2_tests
        tests/phase2_tests.c
        src/camera.c
        src/coordinate.c)
    target_include_directories(phase2_tests PRIVATE src)
    target_link_libraries(phase2_tests PRIVATE cglm::cglm m)
    target_compile_definitions(phase2_tests PRIVATE CGLM_FORCE_DEPTH_ZERO_TO_ONE)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(phase2_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME phase2_projection_and_precision COMMAND phase2_tests)

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME terrain_macro_colour_map
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/build_terrain_macro_tests.py"
                --tool "${CMAKE_CURRENT_SOURCE_DIR}/tools/build_terrain_macro.py")

    add_test(NAME texture_periodic_conversion
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/make_texture_tileable_tests.py"
                --tool "${CMAKE_CURRENT_SOURCE_DIR}/tools/make_texture_tileable.py")

    add_test(NAME terrain_micro_material_atlas
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/build_terrain_micro_atlas_tests.py"
                --tool "${CMAKE_CURRENT_SOURCE_DIR}/tools/build_terrain_micro_atlas.py")

    add_test(NAME terrain_texture_stretch_detection
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/detect_texture_stretch_tests.py"
                --tool "${CMAKE_CURRENT_SOURCE_DIR}/tools/detect_texture_stretch.py"
                --terrain-tool "${CMAKE_CURRENT_SOURCE_DIR}/tools/terrain_tiles.py")

    add_executable(phase3_tile_loader_tests
        tests/phase3_tile_loader_tests.c
        src/terrain_tile.c)
    target_include_directories(phase3_tile_loader_tests PRIVATE src)
    target_link_libraries(phase3_tile_loader_tests PRIVATE core_utils cglm::cglm m)
    target_compile_definitions(phase3_tile_loader_tests PRIVATE
        CGLM_FORCE_DEPTH_ZERO_TO_ONE)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(phase3_tile_loader_tests PRIVATE
            -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME phase3_offline_tile_format
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/phase3_tests.py"
                --tool "${CMAKE_CURRENT_SOURCE_DIR}/tools/terrain_tiles.py"
                --loader "$<TARGET_FILE:phase3_tile_loader_tests>")

    add_test(NAME swiss_infrastructure_masks
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/mask_swiss_infrastructure_tests.py"
                --tool "${CMAKE_CURRENT_SOURCE_DIR}/tools/mask_swiss_infrastructure.py")

    add_test(NAME swiss_alps_downloader
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/download_swiss_alps_tests.py"
                --tool "${CMAKE_CURRENT_SOURCE_DIR}/tools/download_swiss_alps.py")

    add_test(NAME scscts_shadow_correction_pilot
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/scscts_checkpoint_demo_tests.py"
                --tool "${CMAKE_CURRENT_SOURCE_DIR}/tools/scscts_checkpoint_demo.py"
                --terrain-tool "${CMAKE_CURRENT_SOURCE_DIR}/tools/terrain_tiles.py")

    add_executable(phase4_quadtree_tests
        tests/phase4_quadtree_tests.c
        src/terrain_quadtree.c
        src/terrain_grid.c)
    target_include_directories(phase4_quadtree_tests PRIVATE src include)
    target_link_libraries(phase4_quadtree_tests PRIVATE cglm::cglm m)
    target_compile_definitions(phase4_quadtree_tests PRIVATE
        CGLM_FORCE_DEPTH_ZERO_TO_ONE)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(phase4_quadtree_tests PRIVATE
            -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME phase4_global_terrain_quadtree COMMAND phase4_quadtree_tests)

    add_executable(phase5_surface_tests
        tests/phase5_surface_tests.c
        src/surface_detail.c)
    target_include_directories(phase5_surface_tests PRIVATE src)
    target_link_libraries(phase5_surface_tests PRIVATE m)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(phase5_surface_tests PRIVATE
            -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME phase5_terrain_surface_quality COMMAND phase5_surface_tests)

    add_executable(phase6_lighting_tests
        tests/phase6_lighting_tests.c
        src/shadow_cascade.c)
    target_include_directories(phase6_lighting_tests PRIVATE src)
    target_link_libraries(phase6_lighting_tests PRIVATE cglm::cglm m)
    target_compile_definitions(phase6_lighting_tests PRIVATE
        CGLM_FORCE_DEPTH_ZERO_TO_ONE)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(phase6_lighting_tests PRIVATE
            -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME phase6_hdr_lighting_and_shadows COMMAND phase6_lighting_tests)

    add_executable(phase7_atmosphere_tests
        tests/phase7_atmosphere_tests.c
        src/atmosphere.c)
    target_include_directories(phase7_atmosphere_tests PRIVATE src)
    target_link_libraries(phase7_atmosphere_tests PRIVATE m)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(phase7_atmosphere_tests PRIVATE
            -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME phase7_physical_sky_and_aerial_perspective
             COMMAND phase7_atmosphere_tests)

    add_executable(test_environment
        tests/environment_tests.c
        src/environment.c
        include/stb_image.c)
    target_include_directories(test_environment PRIVATE src include)
    target_link_libraries(test_environment PRIVATE m)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_environment PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME environment_sh9_projection COMMAND test_environment)

    add_executable(phase8_temporal_tests
        tests/phase8_temporal_tests.c
        src/temporal.c)
    target_include_directories(phase8_temporal_tests PRIVATE src)
    target_link_libraries(phase8_temporal_tests PRIVATE cglm::cglm m)
    target_compile_definitions(phase8_temporal_tests PRIVATE
        CGLM_FORCE_DEPTH_ZERO_TO_ONE)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(phase8_temporal_tests PRIVATE
            -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME phase8_motion_taa_and_exposure COMMAND phase8_temporal_tests)

    add_executable(phase_d_material_tests tests/phase_d_material_tests.c src/static_material.c)
    target_include_directories(phase_d_material_tests PRIVATE src)
    target_link_libraries(phase_d_material_tests PRIVATE m)
    add_test(NAME phase_d_material_policy COMMAND phase_d_material_tests)
endif()
