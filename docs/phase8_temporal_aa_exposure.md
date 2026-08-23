# Phase 8: motion vectors, temporal AA, and exposure

Phase 8 adds a temporal HDR frame between atmosphere composition and the final
display transform. Camera-relative world coordinates remain the authoritative
geometry contract; history never stores or reconstructs an absolute float
position.

## Frame workflow

1. An eight-sample Halton(2,3) sequence jitters the infinite reversed-Z
   projection by at most half a pixel.
2. Terrain renders HDR colour, reversed-Z depth, and `R16G16_SFLOAT` motion.
   Motion uses `previous_uv - current_uv`. The previous clip position applies
   the current-to-previous camera-origin translation before the previous
   view-projection matrix, so origin shifts do not appear as object motion.
3. Atmosphere, sun, aerial perspective, and terrain are composed into one
   unexposed RGBA16F image.
4. The temporal pass selects velocity at the nearest reversed-Z depth in a 3x3
   neighbourhood, reprojects colour and depth, rejects disocclusion, clamps
   compressed HDR history to the current neighbourhood, and writes the next
   colour/depth history pair.
5. A 256-bin log-luminance histogram covers the resolved HDR image. A second
   compute pass derives geometric-average luminance and adapts exposure with
   separate brighten and darken speeds.
6. The display pass applies exposure, the ACES-fitted curve, and display-space
   centred dither. An sRGB swapchain performs the sole transfer in hardware;
   a UNORM fallback receives the shader-encoded value instead.

Two full-resolution history colour/depth pairs ping-pong each frame. With one
frame in flight, their ownership and descriptor selection are deterministic.
Render-pass dependencies and explicit buffer barriers cover scene sampling,
history writes, histogram reduction, exposure writes, and display reads.

## History validity

History is reset after swapchain recreation, resize, shader reload, large camera
translation/rotation, sun/material/debug changes, and the first frame. The
renderer owns the final validity check because only it knows when out-of-date
swapchain images were replaced.

The terrain runtime also tracks whether each quadtree node was visible in the
immediately preceding frame. A newly introduced child, a returning parent, or
any other non-consecutive tile emits an off-screen motion sentinel. Its pixels
therefore reject history rather than pretending that two LOD surfaces describe
the same samples.

Depth rejection predicts the prior-frame depth of the current surface using
the previous camera-relative transform. This remains stable during translation;
comparing current and old linear depth directly would reject valid nearby
terrain whenever the camera moved.

## Controls and validation

- F4 cycles history weight, rejection, motion-vector, and clamp debug views.
- F5 reloads all graphics/compute pipelines and invalidates temporal history.
- All prior terrain, lighting, shadow, and atmosphere debug controls remain
  available; changing one resets history to avoid blending unlike views.

`phase8_temporal_tests` verifies the Halton sequence, sub-pixel projection
jitter, camera-cut policy, exposure target bounds, and independent adaptation
speeds. The complete CTest suite, shader compilation, `spirv-val`, and a live
Vulkan smoke run cover the integrated paths.
