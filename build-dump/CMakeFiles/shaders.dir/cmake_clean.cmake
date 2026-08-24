file(REMOVE_RECURSE
  "CMakeFiles/shaders"
  "shaders/atmosphere_aerial.comp.spv"
  "shaders/atmosphere_composite.frag.spv"
  "shaders/atmosphere_multiscattering.comp.spv"
  "shaders/atmosphere_skyview.comp.spv"
  "shaders/atmosphere_transmittance.comp.spv"
  "shaders/cube.frag.spv"
  "shaders/cube.vert.spv"
  "shaders/environment_equirect_to_cube.comp.spv"
  "shaders/environment_prefilter.comp.spv"
  "shaders/exposure.comp.spv"
  "shaders/fullscreen.vert.spv"
  "shaders/luminance_histogram.comp.spv"
  "shaders/mesh.frag.spv"
  "shaders/mesh.vert.spv"
  "shaders/shadow.vert.spv"
  "shaders/temporal_resolve.frag.spv"
  "shaders/terrain.frag.spv"
  "shaders/terrain.vert.spv"
  "shaders/tonemap.frag.spv"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/shaders.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
