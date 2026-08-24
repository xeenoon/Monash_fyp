# CMake generated Testfile for 
# Source directory: /home/ccw100/terrain_gen
# Build directory: /home/ccw100/terrain_gen/build-dump
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test("reusable_utils" "/home/ccw100/terrain_gen/build-dump/utils_tests")
set_tests_properties("reusable_utils" PROPERTIES  _BACKTRACE_TRIPLES "/home/ccw100/terrain_gen/CMakeLists.txt;151;add_test;/home/ccw100/terrain_gen/CMakeLists.txt;0;")
add_test("phase2_projection_and_precision" "/home/ccw100/terrain_gen/build-dump/phase2_tests")
set_tests_properties("phase2_projection_and_precision" PROPERTIES  _BACKTRACE_TRIPLES "/home/ccw100/terrain_gen/CMakeLists.txt;163;add_test;/home/ccw100/terrain_gen/CMakeLists.txt;0;")
add_test("phase3_offline_tile_format" "/usr/bin/python3.14" "/home/ccw100/terrain_gen/tests/phase3_tests.py" "--tool" "/home/ccw100/terrain_gen/tools/terrain_tiles.py" "--loader" "/home/ccw100/terrain_gen/build-dump/phase3_tile_loader_tests")
set_tests_properties("phase3_offline_tile_format" PROPERTIES  _BACKTRACE_TRIPLES "/home/ccw100/terrain_gen/CMakeLists.txt;177;add_test;/home/ccw100/terrain_gen/CMakeLists.txt;0;")
add_test("swiss_infrastructure_masks" "/usr/bin/python3.14" "/home/ccw100/terrain_gen/tests/mask_swiss_infrastructure_tests.py" "--tool" "/home/ccw100/terrain_gen/tools/mask_swiss_infrastructure.py")
set_tests_properties("swiss_infrastructure_masks" PROPERTIES  _BACKTRACE_TRIPLES "/home/ccw100/terrain_gen/CMakeLists.txt;183;add_test;/home/ccw100/terrain_gen/CMakeLists.txt;0;")
add_test("swiss_alps_downloader" "/usr/bin/python3.14" "/home/ccw100/terrain_gen/tests/download_swiss_alps_tests.py" "--tool" "/home/ccw100/terrain_gen/tools/download_swiss_alps.py")
set_tests_properties("swiss_alps_downloader" PROPERTIES  _BACKTRACE_TRIPLES "/home/ccw100/terrain_gen/CMakeLists.txt;188;add_test;/home/ccw100/terrain_gen/CMakeLists.txt;0;")
add_test("phase4_global_terrain_quadtree" "/home/ccw100/terrain_gen/build-dump/phase4_quadtree_tests")
set_tests_properties("phase4_global_terrain_quadtree" PROPERTIES  _BACKTRACE_TRIPLES "/home/ccw100/terrain_gen/CMakeLists.txt;205;add_test;/home/ccw100/terrain_gen/CMakeLists.txt;0;")
add_test("phase5_terrain_surface_quality" "/home/ccw100/terrain_gen/build-dump/phase5_surface_tests")
set_tests_properties("phase5_terrain_surface_quality" PROPERTIES  _BACKTRACE_TRIPLES "/home/ccw100/terrain_gen/CMakeLists.txt;216;add_test;/home/ccw100/terrain_gen/CMakeLists.txt;0;")
add_test("phase6_hdr_lighting_and_shadows" "/home/ccw100/terrain_gen/build-dump/phase6_lighting_tests")
set_tests_properties("phase6_hdr_lighting_and_shadows" PROPERTIES  _BACKTRACE_TRIPLES "/home/ccw100/terrain_gen/CMakeLists.txt;229;add_test;/home/ccw100/terrain_gen/CMakeLists.txt;0;")
add_test("phase7_physical_sky_and_aerial_perspective" "/home/ccw100/terrain_gen/build-dump/phase7_atmosphere_tests")
set_tests_properties("phase7_physical_sky_and_aerial_perspective" PROPERTIES  _BACKTRACE_TRIPLES "/home/ccw100/terrain_gen/CMakeLists.txt;240;add_test;/home/ccw100/terrain_gen/CMakeLists.txt;0;")
add_test("environment_sh9_projection" "/home/ccw100/terrain_gen/build-dump/test_environment")
set_tests_properties("environment_sh9_projection" PROPERTIES  _BACKTRACE_TRIPLES "/home/ccw100/terrain_gen/CMakeLists.txt;252;add_test;/home/ccw100/terrain_gen/CMakeLists.txt;0;")
add_test("phase8_motion_taa_and_exposure" "/home/ccw100/terrain_gen/build-dump/phase8_temporal_tests")
set_tests_properties("phase8_motion_taa_and_exposure" PROPERTIES  _BACKTRACE_TRIPLES "/home/ccw100/terrain_gen/CMakeLists.txt;265;add_test;/home/ccw100/terrain_gen/CMakeLists.txt;0;")
add_test("phase_d_material_policy" "/home/ccw100/terrain_gen/build-dump/phase_d_material_tests")
set_tests_properties("phase_d_material_policy" PROPERTIES  _BACKTRACE_TRIPLES "/home/ccw100/terrain_gen/CMakeLists.txt;270;add_test;/home/ccw100/terrain_gen/CMakeLists.txt;0;")
