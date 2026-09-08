#include <SDL3/SDL.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "atmosphere.h"
#include "benchmark_ground.h"
#include "camera.h"
#include "dungeon_scene.h"
#include "dungeon_camera.h"
#include "dungeon_harness.h"
#include "gltf_scene.h"
#include "input.h"
#include "material_stability_demo.h"
#include "quarry.h"
#include "renderer.h"
#include "surface_detail.h"
#include "temporal.h"
#include "terrain_runtime.h"

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

/* Compile-time terrain start camera. These are absolute WORLD coordinates (the
   `camera pos=` value printed in a shader dump / replay hint), NOT tile-local.
   Override any of them on the CMake command line, e.g.
   -DTERRAIN_START_YAW=85.6, to boot straight into a viewpoint that reproduces a
   bug so repeated dumps compare against the exact same frame. */
#ifndef TERRAIN_START_POS_X
#define TERRAIN_START_POS_X 2641555.948770
#endif
#ifndef TERRAIN_START_POS_Y
#define TERRAIN_START_POS_Y 2164.176855
#endif
#ifndef TERRAIN_START_POS_Z
#define TERRAIN_START_POS_Z -1160110.084046
#endif
#ifndef TERRAIN_START_YAW
#define TERRAIN_START_YAW 85.600f
#endif
#ifndef TERRAIN_START_PITCH
#define TERRAIN_START_PITCH -12.320f
#endif

typedef enum TerrainTextureSet {
	TERRAIN_TEXTURES_UNSHADOWED,
	TERRAIN_TEXTURES_SHADOWED,
} TerrainTextureSet;

static bool terrain_texture_set_from_arguments(int argc, char *argv[], TerrainTextureSet *out)
{
	*out = TERRAIN_TEXTURES_UNSHADOWED;
	for (int i = 1; i < argc; ++i)
	{
		if (strcmp(argv[i], "--terrain-textures=unshadowed") == 0)
			*out = TERRAIN_TEXTURES_UNSHADOWED;
		else if (strcmp(argv[i], "--terrain-textures=shadowed") == 0)
			*out = TERRAIN_TEXTURES_SHADOWED;
		else if (strncmp(argv[i], "--terrain-textures=", 19) == 0)
			return false;
	}
	return true;
}

typedef struct StretchOverlaySettings
{
	bool enabled;
	float threshold;
	float opacity;
} StretchOverlaySettings;

/* In a TERRAIN_STRETCH_OVERLAY build, --show-texture-stretch enables the live
   equivalent of tools/detect_texture_stretch.py's red overlay (terrain.frag).
   Threshold/opacity default to the tool's values and can be tuned via
   TERRAIN_STRETCH_THRESHOLD / TERRAIN_STRETCH_OPACITY. Ordinary builds retain
   the uniform ABI but cannot enable or render the highlight. */
static StretchOverlaySettings stretch_overlay_from_arguments(int argc, char *argv[])
{
	StretchOverlaySettings settings = {.enabled = false, .threshold = 1.5f, .opacity = 0.25f};
#ifdef TERRAIN_STRETCH_OVERLAY
	for (int i = 1; i < argc; ++i)
		if (strcmp(argv[i], "--show-texture-stretch") == 0)
			settings.enabled = true;
	const char *threshold = getenv("TERRAIN_STRETCH_THRESHOLD");
	if (threshold)
		settings.threshold = (float)atof(threshold);
	const char *opacity = getenv("TERRAIN_STRETCH_OPACITY");
	if (opacity)
		settings.opacity = (float)atof(opacity);
#else
	for (int i = 1; i < argc; ++i)
		if (strcmp(argv[i], "--show-texture-stretch") == 0)
			fprintf(stderr,
					"--show-texture-stretch is unavailable in this build; "
					"configure with -DTERRAIN_STRETCH_OVERLAY=ON\n");
#endif
	return settings;
}

static ShadowQualitySettings shadow_quality_from_environment(void)
{
	ShadowQualitySettings q = {.filter_mode = SHADOW_FILTER_PCF,
							   .resolution = 2048u,
							   .pcf_radius_texels = 1.75f,
							   .sun_angular_radius_rad = 0.004675f,
							   .blocker_search_m = 12.0f,
							   .max_filter_radius_texels = 24.0f};
	const char *mode = getenv("TERRAIN_SHADOW_FILTER");
	if (mode && strcmp(mode, "hard") == 0)
		q.filter_mode = SHADOW_FILTER_HARD;
	else if (mode && strcmp(mode, "pcss") == 0)
		q.filter_mode = SHADOW_FILTER_PCSS;
	else if (mode && strcmp(mode, "pcf") != 0)
	{
		fprintf(stderr, "TERRAIN_SHADOW_FILTER must be hard, pcf, or pcss\n");
		exit(EXIT_FAILURE);
	}
	const char *resolution = getenv("TERRAIN_SHADOW_RESOLUTION");
	if (resolution)
	{
		q.resolution = (uint32_t)strtoul(resolution, NULL, 10);
		if (q.resolution != 2048u && q.resolution != 4096u)
		{
			fprintf(stderr, "TERRAIN_SHADOW_RESOLUTION must be 2048 or 4096\n");
			exit(EXIT_FAILURE);
		}
	}
	return q;
}

/* The sole CPU packing contract for all static materials.  geometry retains
 * glTF base-colour RGBA; shadow state lives in debug.z, never in alpha. */
static DrawPushConstants
static_material_push(LocalToWorldTransform transform, WorldPosition camera_position,
					 const float base_color[4], float metallic, float gltf_roughness,
					 const StaticMaterialParameters *parameters, bool default_lit, bool diffuse_ibl,
					 bool specular_ibl, bool shadows, float curvature_strength)
{
	return (DrawPushConstants){
		.local_to_camera_relative =
			coordinate_local_to_camera_relative(&transform, camera_position),
		.geometry = {{base_color[0], base_color[1], base_color[2], base_color[3]}},
		.elevation_uv = {{gltf_roughness, parameters->normal_strength, parameters->ao_strength,
						  parameters->roughness_bias}},
		.material = {{metallic, default_lit ? 1.f : 0.f, diffuse_ibl ? 1.f : 0.f,
					  specular_ibl ? 1.f : 0.f}},
		.debug = {{parameters->cavity_strength, parameters->displacement_scale, shadows ? 1.f : 0.f,
				   curvature_strength}}};
}

int main(int argc, char *argv[])
{
	TerrainTextureSet terrain_textures;
	if (!terrain_texture_set_from_arguments(argc, argv, &terrain_textures))
	{
		fprintf(stderr, "Terrain textures must be unshadowed or shadowed. Example: "
				"--terrain-textures=shadowed\n");
		return EXIT_FAILURE;
	}
	StretchOverlaySettings stretch_overlay = stretch_overlay_from_arguments(argc, argv);
	if (stretch_overlay.enabled)
		printf("Texture stretch overlay: on (threshold %.2fx, opacity %.2f)\n",
			   stretch_overlay.threshold, stretch_overlay.opacity);
	const char *scene = getenv("TERRAIN_SCENE");
	bool use_terrain = !scene || strcmp(scene, "terrain") == 0;
	bool use_quarry = scene && strcmp(scene, "quarry") == 0;
	bool use_phase_d_demo = scene && strcmp(scene, "phase_d_demo") == 0;
	bool use_dungeon = scene && strcmp(scene, "dungeon") == 0;
#ifdef DEBUG_SHADER_DUMP
	const char *scene_name = use_terrain ? "terrain"
		: (use_dungeon ? "dungeon"
					   : (use_phase_d_demo ? "phase_d_demo" : (use_quarry ? "quarry" : "gltf")));
#endif
	const char *gltf_path = NULL;
	/* The large benchmark is intentionally not in git. Terrain is the normal
	 * no-argument scene; validate the benchmark only when explicitly selected. */
	if (scene && strcmp(scene, "coastal_cliff") == 0)
	{
		gltf_path = BENCHMARK_DIR "/coastal_cliff_01/coastal_cliff_01_4k.gltf";
		if (access(gltf_path, R_OK) != 0)
		{
			fprintf(stderr, "Coastal Cliff benchmark assets are missing. Download with:\n"
							"python3 tools/download_benchmark_assets.py\n");
			return EXIT_FAILURE;
		}
	}
	else if (scene && strcmp(scene, "gltf") == 0)
	{
		gltf_path = getenv("TERRAIN_GLTF_PATH");
		if (!gltf_path || !*gltf_path)
		{
			fprintf(stderr, "TERRAIN_SCENE=gltf requires TERRAIN_GLTF_PATH\n");
			return EXIT_FAILURE;
		}
	}
	else if (!use_terrain && !use_quarry && !use_phase_d_demo && !use_dungeon)
	{
		fprintf(stderr,
				"Supported scene selectors: terrain, dungeon, coastal_cliff, quarry, phase_d_demo, or gltf.\n");
		return EXIT_FAILURE;
	}
	if (!SDL_Init(SDL_INIT_VIDEO))
	{
		fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
		return EXIT_FAILURE;
	}
	SDL_Window *window = SDL_CreateWindow("Terrain Renderer", WINDOW_WIDTH, WINDOW_HEIGHT,
										  SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
	if (!window)
	{
		fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
		return EXIT_FAILURE;
	}
	if (use_phase_d_demo)
		SDL_SetWindowTitle(window, "Phase D — stabilized material detail");
	else if (use_dungeon)
		SDL_SetWindowTitle(window, "Dungeon Explorer");
	if (!SDL_SetWindowRelativeMouseMode(window, true))
		fprintf(stderr, "Relative mouse mode unavailable: %s\n", SDL_GetError());
	if (use_dungeon)
	{
		SDL_SetWindowRelativeMouseMode(window, false);
		SDL_ShowCursor();
	}

	Renderer renderer;
	ShadowQualitySettings shadow_quality = shadow_quality_from_environment();
	renderer_init(&renderer, window,
				  &(RendererConfig){.environment_path = getenv("TERRAIN_ENV_HDR"),
									.shadow_quality = shadow_quality});
	TerrainRuntime *terrain = NULL;
	Quarry quarry = {0};
	BenchmarkGround ground = {0};
	GltfScene gltf = {0};
	GltfLoadError gltf_error = {0};
	MaterialStabilityDemo demo = {0};
	DungeonScene dungeon = {0};
	GltfLoadResult load_result = GLTF_LOAD_OK;
	if (use_terrain)
	{
		/* Terrain defaults to the 16 km Swiss-Alps set -- the real dataset the
		   camera coordinates (world x ~2.64e6) live in and where the surface
		   artifacts reproduce. TRN_DIR is a tiny 1 km test pyramid whose world
		   origin is elsewhere, so booting there shows empty sky at these
		   coordinates. Override with TERRAIN_DATASET=... (e.g. TRN_DIR) or the
		   shadowed set via --terrain-textures=shadowed.

		   Within the unshadowed set, GRADED_ALPS_DIR (built by
		   tools/build_graded_alps_dataset.py) is UNSHADOWED_ALPS_DIR's imagery/
		   regenerated from colour-graded AI shadow-removal masters -- the
		   ungraded masters in shadowfree-masters/ read as washed-out/chalky in
		   full sun once exposure was fixed to stop masking it (macro_tint /
		   base_color luminance in sunlit rock was averaging ~1.8x the shader's
		   own neutral-rock reference, material_macro_reference_luminance(7u) in
		   shaders/terrain.frag). GRADED_ALPS_DIR shares tiles/ and manifest.json
		   with UNSHADOWED_ALPS_DIR via symlink (geometry is unchanged) and is now
		   the default; ATLAS=OLD reverts to the ungraded imagery for comparison. */
		const char *dataset_root = terrain_textures == TERRAIN_TEXTURES_SHADOWED
			? SHADOWED_TERRAIN_DIR : GRADED_ALPS_DIR;
		const char *atlas = "new";
		if (terrain_textures == TERRAIN_TEXTURES_UNSHADOWED)
		{
			const char *atlas_env = getenv("ATLAS");
			if (atlas_env && strcmp(atlas_env, "OLD") == 0)
			{
				dataset_root = UNSHADOWED_ALPS_DIR;
				atlas = "old";
			}
			const char *override_root = getenv("TERRAIN_DATASET");
			if (override_root && *override_root)
				dataset_root = override_root;
		}
		printf("Terrain textures: %s (atlas=%s)\n", terrain_textures == TERRAIN_TEXTURES_SHADOWED
			? "shadowed" : "unshadowed", atlas);
		TerrainRuntimeSettings settings = terrain_runtime_default_settings();
		settings.quadtree.split_threshold_px = 2.5f;
		settings.quadtree.merge_threshold_px = 1.75f;
		settings.quadtree.max_nodes = 8192;
		settings.quadtree.max_resident_tiles = 512;
		settings.quadtree.max_cpu_bytes = UINT64_C(512) * 1024u * 1024u;
		settings.quadtree.max_gpu_bytes = UINT64_C(1024) * 1024u * 1024u;
		settings.skirt_ratio = 0.01f;
		terrain = terrain_runtime_create(&renderer, dataset_root, &settings);
		if (!terrain)
			fprintf(stderr, "Could not create terrain quadtree from %s\n", dataset_root);
	}
	else if (use_phase_d_demo)
		load_result = GLTF_LOAD_OK;
	else if (use_dungeon)
	{
		DungeonLevelError dungeon_error = {0};
		if (!dungeon_scene_create(&renderer, &dungeon, &dungeon_error))
		{
			fprintf(stderr, "Could not load dungeon: %s\n", dungeon_error.message);
			load_result = GLTF_LOAD_INVALID;
		}
	}
	else if (use_quarry)
		load_result = quarry_create(&renderer, &quarry, QUARRY_DIR) ? GLTF_LOAD_OK
																	 : GLTF_LOAD_IO_ERROR;
	else
		load_result = gltf_scene_create(
						 &renderer, gltf_path,
						 &(GltfLoadOptions){
							 .use_metallic_roughness_red_as_occlusion =
								 strcmp(scene, "coastal_cliff") == 0,
							 /* Poly Haven's authored front faces +Z; the benchmark camera is
							  * placed on -Z, so turn only this named benchmark toward it. */
							 .placement = strcmp(scene, "coastal_cliff") == 0
										  ? coordinate_rotation_y(3.14159265358979323846,
																			  (WorldPosition){0})
										  : coordinate_identity_transform((WorldPosition){0})},
						 &gltf, &gltf_error);
	if ((use_terrain && !terrain) || load_result != GLTF_LOAD_OK)
	{
		if (!use_terrain && !use_dungeon)
			fprintf(stderr, "Could not load %s: %s\n", use_quarry ? QUARRY_DIR : gltf_path,
					use_quarry ? "Quarry loader failed" : gltf_error.message);
		renderer_shutdown(&renderer);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return EXIT_FAILURE;
	}
	if (use_quarry && !benchmark_ground_create(&renderer, &ground, &quarry.base))
	{
		fprintf(stderr, "Could not create Quarry benchmark ground\n");
		quarry_destroy(&renderer, &quarry);
		renderer_shutdown(&renderer);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return EXIT_FAILURE;
	}
	if (use_phase_d_demo && !material_stability_demo_create(&renderer, &demo))
	{
		fprintf(stderr, "Could not create Phase D demo; download benchmark assets first.\n");
		renderer_shutdown(&renderer);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return EXIT_FAILURE;
	}
	LocalToWorldTransform terrain_root = terrain
		? terrain_runtime_root_transform(terrain)
		: coordinate_identity_transform((WorldPosition){0});
	float terrain_span = terrain ? terrain_runtime_root_span(terrain) : 0.0f;
	(void)terrain_span;
	Camera camera = use_terrain
		? (Camera){.position = (WorldPosition){TERRAIN_START_POS_X, TERRAIN_START_POS_Y,
											   TERRAIN_START_POS_Z},
				   .yaw = TERRAIN_START_YAW,
				   .pitch = TERRAIN_START_PITCH}
		: (Camera){.position = use_dungeon
							 ? (WorldPosition){0.0, 22.0, -18.0}
							 : (use_phase_d_demo
								 ? (WorldPosition){0.0, 1.0, -18.0}
								 : (use_quarry ? (WorldPosition){0.0, 8.0, -25.0}
												: (WorldPosition){0.0, 6.0, -45.0})),
				   .yaw = 90.0f,
				   .pitch = use_dungeon ? -50.0f : (use_quarry ? -3.0f : -2.0f)};
	DungeonCamera dungeon_camera = {0};
	char dungeon_title[256] = {0};
	/* DUNGEON_SCRIPT drives the dungeon scene from a file instead of the
	   keyboard, so lock picking can be proved end-to-end in the real renderer.
	   Unset, none of this runs. */
	bool harness_script_error = false;
	DungeonHarness *harness = use_dungeon ? dungeon_harness_create(&harness_script_error) : NULL;
	if (harness_script_error)
		return 1;
	if (use_dungeon)
	{
		dungeon_camera_init(&dungeon_camera, dungeon.level.spawn);
		camera = dungeon_camera.camera;
	}

	/* Runtime override of the start camera in absolute WORLD coordinates -- the
	   exact triple printed as "camera pos=" in a shader dump. Unlike
	   TERRAIN_DUMP_POS (which is interpreted as tile-local and converted), these
	   are fed straight in, so pasting a dump's replay numbers reproduces its
	   framing. Lets us boot into a bug viewpoint without recompiling. */
	if (use_terrain)
	{
		if (getenv("TERRAIN_START_POS"))
		{
			double sx = 0, sy = 0, sz = 0;
			if (sscanf(getenv("TERRAIN_START_POS"), "%lf %lf %lf", &sx, &sy, &sz) == 3)
				camera.position = (WorldPosition){sx, sy, sz};
		}
		if (getenv("TERRAIN_START_YAW"))
			camera.yaw = (float)atof(getenv("TERRAIN_START_YAW"));
		if (getenv("TERRAIN_START_PITCH"))
			camera.pitch = (float)atof(getenv("TERRAIN_START_PITCH"));
	}
	if (use_terrain)
	{
		WorldPosition ro = coordinate_local_to_world(&terrain_root,
													 (TileLocalPosition){0.0f, 0.0f, 0.0f});
		printf("[boot] terrain_root world origin = %.3f %.3f %.3f  span=%.3f\n", ro.x, ro.y, ro.z,
			   terrain_span);
		printf("[boot] camera pos = %.3f %.3f %.3f  yaw=%.3f pitch=%.3f\n", camera.position.x,
			   camera.position.y, camera.position.z, camera.yaw, camera.pitch);
		fflush(stdout);
	}

	Input input = {.mouse_captured = !use_dungeon};
	/* Automated visual comparisons need to survive window focus and pointer
	   motion without drifting away from the replayed dump camera. */
	bool freeze_camera = getenv("TERRAIN_FREEZE_CAMERA") &&
		atoi(getenv("TERRAIN_FREEZE_CAMERA")) != 0;
	uint64_t start_ticks = SDL_GetTicksNS();
	uint64_t previous_ticks = start_ticks;
	mat4s previous_projection = GLMS_MAT4_IDENTITY_INIT;
	mat4s previous_view = GLMS_MAT4_IDENTITY_INIT;
	mat4s previous_view_projection = GLMS_MAT4_IDENTITY_INIT;
	vec2s previous_jitter = {{0.0f, 0.0f}};
	WorldPosition previous_camera_position = camera.position;
	float previous_camera_yaw = camera.yaw;
	float previous_camera_pitch = camera.pitch;
	bool history_valid = false;
	uint64_t temporal_frame = 0;
	bool demo_flyby =
		use_phase_d_demo && getenv("TERRAIN_DEMO_FLYBY") && atoi(getenv("TERRAIN_DEMO_FLYBY")) != 0;
	bool demo_split = use_phase_d_demo && !demo_flyby && getenv("TERRAIN_DEMO_SPLIT") &&
					  atoi(getenv("TERRAIN_DEMO_SPLIT")) != 0;
	if (demo_flyby)
		SDL_SetWindowTitle(window, "Phase C — raw normal detail");
	else if (demo_split)
		SDL_SetWindowTitle(window, "Phase D demo — LEFT: Phase C/raw | RIGHT: Phase D/stabilized");
	bool demo_flyby_phase_d = false;
	unsigned debug_mode = 0;
	if (getenv("TERRAIN_START_DEBUG_VIEW"))
		debug_mode = (unsigned)atoi(getenv("TERRAIN_START_DEBUG_VIEW"));
	/* Shadow-free footage is an illumination-neutral terrain source. Starting it
	   in the subtle 35% blend leaves a 65% unlit floor in every shadow, so it
	   cannot reproduce the contrast of the baked-light reference imagery. */
	unsigned relight_mode = terrain_textures == TERRAIN_TEXTURES_UNSHADOWED ? 2u : 1u;
	unsigned sun_mode = 0;
	unsigned atmosphere_slice = 15;
	/* F3 exposes the delivery sequence: legacy -> A (Default Lit) -> B1
	   (diffuse IBL) -> B2 (specular IBL) -> C (shadowing) -> legacy. */
	unsigned static_mesh_shading_mode = 5u;
	/* The terrain overview occupies a small part of a much darker sky frame;
	 * scene-average adaptation otherwise overexposes it and hides the material
	 * frequencies. E can still enable adaptation interactively. */
	bool auto_exposure_enabled = !use_terrain && !use_dungeon;
	AtmosphereParameters atmosphere = atmosphere_earth();
	bool running = true;
#ifdef DEBUG_SHADER_DUMP
	/* Automated capture for offline diagnosis. TERRAIN_DUMP_AFTER=N renders N
	   frames (letting tiles stream in), dumps once, then quits. Optional env:
	   TERRAIN_DUMP_SUN (0/1/2), TERRAIN_DUMP_YAW, TERRAIN_DUMP_PITCH,
	   TERRAIN_DUMP_POS="x y z" (tile-local metres). */
	const char *auto_after_env = getenv("TERRAIN_DUMP_AFTER");
	unsigned auto_dump_after = auto_after_env ? (unsigned)atoi(auto_after_env) : 0u;
	uint64_t rendered_frames = 0;
	if (getenv("TERRAIN_DUMP_SUN"))
		sun_mode = (unsigned)atoi(getenv("TERRAIN_DUMP_SUN")) % 3u;
	if (getenv("TERRAIN_DUMP_QUARRY_MODE"))
	{
		unsigned requested_mode = (unsigned)atoi(getenv("TERRAIN_DUMP_QUARRY_MODE"));
		if (requested_mode <= 5u)
			static_mesh_shading_mode = requested_mode;
	}
	if (getenv("TERRAIN_DUMP_YAW"))
		camera.yaw = (float)atof(getenv("TERRAIN_DUMP_YAW"));
	if (getenv("TERRAIN_DUMP_PITCH"))
		camera.pitch = (float)atof(getenv("TERRAIN_DUMP_PITCH"));
	if (getenv("TERRAIN_DUMP_POS"))
	{
		double x = 0, y = 0, z = 0;
		if (sscanf(getenv("TERRAIN_DUMP_POS"), "%lf %lf %lf", &x, &y, &z) == 3)
			camera.position = use_terrain
				? coordinate_local_to_world(
					  &terrain_root, (TileLocalPosition){(float)x, (float)y, (float)z})
				: (WorldPosition){x, y, z};
	}
#endif
	const vec3s sun_presets[3] = {
		{{-0.4f, -1.0f, -0.3f}},
		{{-1.0f, -0.08f, -0.15f}},
		{{-0.1f, -1.0f, -0.05f}},
	};
	vec3s initial_to_sun = glms_vec3_scale(glms_vec3_normalize(sun_presets[sun_mode]), -1.0f);
	float sun_azimuth = atan2f(initial_to_sun.z, initial_to_sun.x);
	float sun_orbit_angle = asinf(fmaxf(-1.0f, fminf(1.0f, initial_to_sun.y)));
	/* Runtime override of the start sun position (radians), the exact pair
	   printed in a shader dump's "sun azimuth=... orbit=..." line -- lets a
	   dump's sun be reproduced without recompiling, same idea as
	   TERRAIN_START_POS/YAW/PITCH above. F12 (hold) still rotates freely from
	   there at runtime. */
	if (getenv("TERRAIN_START_SUN_AZIMUTH"))
		sun_azimuth = (float)atof(getenv("TERRAIN_START_SUN_AZIMUTH"));
	if (getenv("TERRAIN_START_SUN_ORBIT"))
		sun_orbit_angle = (float)atof(getenv("TERRAIN_START_SUN_ORBIT"));
	while (running)
	{
		input_poll(&input, window);
#ifdef DEBUG_SHADER_DUMP
		/* Input is polled before the draw, so whether this frame should dump is
		   known in time to flip frame.shader_dump.x before renderer_draw_frame.
		   Predicts the same "last countdown frame" the post-draw auto-dump check
		   below fires on, since rendered_frames is unchanged in between. */
		bool dump_this_frame =
			input.dump_shader_data || (auto_dump_after && rendered_frames + 1 >= auto_dump_after);
#endif
		unsigned previous_debug_mode = debug_mode;
		if (input.quit)
			running = false;
		if (input.toggle_quarry_shading)
		{
			if (use_phase_d_demo)
				static_mesh_shading_mode = static_mesh_shading_mode >= 5u ? 4u : 5u;
			else
				static_mesh_shading_mode = (static_mesh_shading_mode + 1) % 6u;
			const char *names[] = {"legacy PBR",
								   "Phase A: Unreal Default Lit",
								   "Phase B1: Default Lit + diffuse sky IBL",
								   "Phase B2: Default Lit + diffuse and specular sky IBL",
								   "Phase C: B2 + cascaded shadows",
								   "Phase D: Phase C + material detail stability"};
			printf("Static-mesh renderer: %s\n", names[static_mesh_shading_mode]);
			if (use_phase_d_demo && !demo_split)
				SDL_SetWindowTitle(window, static_mesh_shading_mode >= 5u
											   ? "Phase D — stabilized material detail"
											   : "Phase C — raw normal detail");
			history_valid = false;
		}
		if (input.reload_shaders)
		{
			renderer_reload_pipeline(&renderer);
			history_valid = false;
		}
		if (input.cycle_temporal_debug)
		{
			debug_mode = debug_mode >= 17u && debug_mode < 20u ? debug_mode + 1u
															   : (debug_mode == 20u ? 0u : 17u);
			const char *names[] = {"history weight", "history rejection", "motion vectors",
								   "history clamp", "off"};
			printf("Temporal debug: %s\n", debug_mode >= 17u ? names[debug_mode - 17u] : names[4]);
			history_valid = false;
		}
		if (input.toggle_depth_debug)
			debug_mode = debug_mode == 1u ? 0u : 1u;
		if (input.toggle_lod_debug)
			debug_mode = debug_mode == 2u ? 0u : 2u;
		if (input.cycle_relighting)
		{
			relight_mode = (relight_mode + 1u) % 3u;
			const char *names[] = {"unlit map", "subtle relight", "material"};
			printf("Terrain imagery: %s\n", names[relight_mode]);
			history_valid = false;
		}
		if (input.cycle_surface_debug)
		{
			if (use_terrain)
			{
				debug_mode = debug_mode >= 3u && debug_mode < 6u ? debug_mode + 1u
														 : (debug_mode == 6u ? 0u : 3u);
				const char *names[] = {"Material normal", "Authored roughness",
									   "Macro colour", "Rock/grass blend", "off"};
				printf("Terrain material view: %s\n",
					   debug_mode >= 3u ? names[debug_mode - 3u] : names[4]);
			}
			else
			{
				debug_mode = debug_mode >= 3u && debug_mode < 6u
					? debug_mode + 1u
					: (debug_mode == 6u ? 21u : (debug_mode == 21u ? 0u : 3u));
				const char *names[] = {
					"Mapped normal", "Authored roughness", "Curvature roughness",
					"Effective roughness (R authored, G effective, B curvature)",
					"Phase D activation ×20", "off"};
				printf("Static-mesh material view: %s\n",
					   debug_mode >= 3u && debug_mode <= 6u ? names[debug_mode - 3u]
																	   : (debug_mode == 21u ? names[4] : names[5]));
			}
		}
		if (input.cycle_shadow_debug)
			debug_mode = debug_mode >= 7u && debug_mode < 11u ? debug_mode + 1u
															  : (debug_mode == 11u ? 0u : 7u);
		if (input.cycle_atmosphere_debug)
			debug_mode = debug_mode >= 12u && debug_mode < 16u ? debug_mode + 1u
															   : (debug_mode == 16u ? 0u : 12u);
		if (input.toggle_auto_exposure)
		{
			auto_exposure_enabled = !auto_exposure_enabled;
			printf("Auto-exposure: %s\n", auto_exposure_enabled ? "on" : "off");
		}
		if (debug_mode != previous_debug_mode)
			history_valid = false;
		if (input.previous_atmosphere_slice && atmosphere_slice > 0u)
			--atmosphere_slice;
		if (input.next_atmosphere_slice && atmosphere_slice < 31u)
			++atmosphere_slice;
		if (input.previous_atmosphere_slice || input.next_atmosphere_slice)
			printf("Atmosphere volume debug slice: %u/31\n", atmosphere_slice);
		uint64_t ticks = SDL_GetTicksNS();
		float dt = (float)(ticks - previous_ticks) / 1000000000.0f;
		previous_ticks = ticks;
		if (dt > 0.1f)
			dt = 0.1f;
		if (harness)
		{
			/* A fixed timestep, so scripted `wait` durations are exact and a
			   run reproduces regardless of what frame rate the machine hits. */
			dt = dungeon_harness_timestep(harness);
			if (!dungeon_harness_pre_frame(harness, &dungeon, &dungeon_camera, &input))
				running = false;
		}
		if (input.rotate_sun)
		{
			const float sun_rotation_speed = 0.35f;
			sun_orbit_angle = fmodf(sun_orbit_angle + sun_rotation_speed * dt, 2.0f * GLM_PI);
			history_valid = false;
		}

		if (!freeze_camera && !use_dungeon && input.mouse_captured)
			camera_update(&camera, input.move_forward, input.move_right, input.look_dx, input.look_dy,
						  input.sprint, dt);
		if (use_dungeon)
		{
			DungeonSession *session = &dungeon.session;
			bool picking = session->phase != DUNGEON_PHASE_EXPLORING;
			/* While a lock is up, the movement and orbit keys drive the lock
			 * instead -- the same keys, rebound by phase, exactly as the source
			 * game's GameplayPhase switch does. */
			if (picking)
			{
				if (session->phase == DUNGEON_PHASE_PIN_TUMBLER)
				{
					if (input.puzzle_left)
						dungeon_session_move_selection(session, -1);
					if (input.puzzle_right)
						dungeon_session_move_selection(session, 1);
					if (input.puzzle_up)
						dungeon_session_adjust(session, 1);
					if (input.puzzle_down)
						dungeon_session_adjust(session, -1);
					if (input.puzzle_confirm)
						dungeon_session_confirm(session);
				}
				else
				{
					/* The dial takes a direction outright: there is nothing to
					 * confirm, the fourth correct turn is what opens it. */
					if (input.puzzle_left)
						dungeon_session_turn(session, DUNGEON_DIAL_LEFT);
					else if (input.puzzle_right)
						dungeon_session_turn(session, DUNGEON_DIAL_RIGHT);
					else if (input.puzzle_up)
						dungeon_session_turn(session, DUNGEON_DIAL_UP);
					else if (input.puzzle_down)
						dungeon_session_turn(session, DUNGEON_DIAL_DOWN);
				}
				/* E steps back out as well as in: the key that got you into the
				 * lock is the one a player reaches for to leave it. */
				if (input.puzzle_cancel || input.interact)
					dungeon_session_cancel(session);
			}
			else if (input.interact)
				dungeon_session_interact(session, dungeon.player.position);

			if (!freeze_camera && !picking)
				dungeon_camera_orbit(&dungeon_camera, input.orbit_yaw, input.orbit_pitch, dt);
			float move_forward = picking ? 0.0f : input.move_forward;
			float move_right = picking ? 0.0f : input.move_right;
			if (!freeze_camera && dungeon_scene_update(&dungeon, move_forward, move_right,
											 dungeon_camera.camera.yaw, dt))
				printf("Dungeon exit reached\n");
			/* Ease the framing toward the lock, and follow a target blended the
			 * same amount, so the door -- not the player -- ends up centred. */
			dungeon_camera_focus(&dungeon_camera, picking,
								 dungeon_session_focus_facing_degrees(session),
								 DUNGEON_LOCK_CENTRE_Y_M, dt);
			float focus = dungeon_camera_focus_blend(&dungeon_camera);
			DungeonPoint lock = dungeon_session_focus_point(session);
			DungeonPoint follow = {
				dungeon.player.position.x + (lock.x - dungeon.player.position.x) * focus,
				dungeon.player.position.z + (lock.z - dungeon.player.position.z) * focus};
			dungeon_camera_update(&dungeon_camera, focus > 0.0f ? follow : dungeon.player.position,
								  dt);
			camera = dungeon_camera.camera;

			/* This renderer draws no text, so the window title is where status
			 * goes -- the same place the source game puts it. */
			char title[256];
			dungeon_session_status_text(session, title, sizeof(title));
			if (strcmp(title, dungeon_title) != 0)
			{
				snprintf(dungeon_title, sizeof(dungeon_title), "%s", title);
				SDL_SetWindowTitle(window, dungeon_title);
			}
		}
		if (demo_flyby)
		{
			float seconds = fmodf((float)(ticks - start_ticks) / 1000000000.0f, 22.0f);
			bool phase_d = seconds >= 10.0f;
			float replay = phase_d ? (seconds - 12.0f) / 8.0f : seconds / 8.0f;
			/* C: 0-8 replay, 8-10 far hold. D: 10-12 far hold, 12-20
			 * identical replay, 20-22 far hold. */
			replay = fminf(fmaxf(replay, 0.0f), 1.0f);
			float near_weight = 1.0f - fabsf(2.0f * replay - 1.0f);
			camera.position = (WorldPosition){0.0, 1.0, -30.0 + 18.0 * near_weight};
			camera.yaw = 90.f;
			camera.pitch = 0.f;
			if (phase_d != demo_flyby_phase_d)
			{
				demo_flyby_phase_d = phase_d;
				history_valid = false;
				printf("Phase D demo flyby: Phase %c\n", phase_d ? 'D' : 'C');
				SDL_SetWindowTitle(window, phase_d ? "Phase D — stabilized material detail"
												   : "Phase C — raw normal detail");
			}
			static_mesh_shading_mode = phase_d ? 5u : 4u;
			if (seconds < dt)
			{
				history_valid = false;
				printf("Phase D demo flyby: Phase C\n");
			}
		}
		bool camera_cut =
			history_valid &&
			temporal_camera_cut(previous_camera_position, camera.position, previous_camera_yaw,
								camera.yaw, previous_camera_pitch, camera.pitch, 2000.0);
		bool use_history = history_valid && !input.resized && !camera_cut;

		vec3s camera_forward_direction = camera_forward(&camera);
		const RendererDraw *terrain_draws = NULL;
		const RendererDraw *terrain_shadow_draws = NULL;
		uint32_t terrain_draw_count = 0;
		uint32_t terrain_shadow_draw_count = 0;
		if (use_terrain)
		{
			TerrainQuadtreeView terrain_view = {
				.camera_world = camera.position,
				.forward = {camera_forward_direction.x, camera_forward_direction.y,
							camera_forward_direction.z},
				.up = {0.0, 1.0, 0.0},
				.vertical_fov_radians = glm_rad(60.0f),
				.aspect = renderer_aspect(&renderer),
				.near_plane_m = CAMERA_NEAR_PLANE,
				.viewport_height_px = renderer.swapchain_extent.height,
			};
			terrain_runtime_update(terrain, &terrain_view, temporal_frame);
			terrain_draws = terrain_runtime_draws(terrain, &terrain_draw_count);
			terrain_shadow_draws =
				terrain_runtime_shadow_draws(terrain, &terrain_shadow_draw_count);
		}
		bool default_lit = static_mesh_shading_mode != 0u;
		bool diffuse_ibl_enabled = static_mesh_shading_mode >= 2u;
		bool specular_ibl_enabled = static_mesh_shading_mode >= 3u;
		bool shadows_enabled = static_mesh_shading_mode >= 4u;
		bool phase_d_enabled = static_mesh_shading_mode >= 5u;
		StaticMaterialParameters quarry_parameters = {
			.normal_strength = 1.0f,
			.ao_strength = 1.0f,
			.cavity_strength = phase_d_enabled && quarry.cavity_available ? 0.25f : 0.0f,
			.roughness_bias = 0.0f,
			.displacement_scale = phase_d_enabled && quarry.cavity_available ? 1.0f : 0.0f};
		const float white_rgba[4] = {1.f, 1.f, 1.f, 1.f};
		DrawPushConstants quarry_push = static_material_push(
			quarry.base.local_to_world, camera.position, white_rgba, quarry.metallic_factor, 1.f,
			&quarry_parameters, default_lit, diffuse_ibl_enabled, specular_ibl_enabled,
			shadows_enabled, phase_d_enabled ? 1.f : 0.f);
		RendererDraw quarry_draw = {.mesh = &quarry.base,
									.material_set = phase_d_enabled && quarry.cavity_available
														? quarry.cavity_material_set
														: quarry.base.material_set,
									.push = quarry_push,
									.static_mesh = true};
		DrawPushConstants ground_push = quarry_push;
		ground_push = static_material_push(ground.mesh.local_to_world, camera.position, white_rgba,
										   quarry.metallic_factor, 1.f, &quarry_parameters,
										   default_lit, diffuse_ibl_enabled, specular_ibl_enabled,
										   shadows_enabled, phase_d_enabled ? 1.f : 0.f);
		ground_push.debug.y = 0.0f; /* ground retains the neutral cavity descriptor */
		RendererDraw ground_draw = {.mesh = &ground.mesh, .push = ground_push, .static_mesh = true};
		RendererDraw dungeon_draws[DUNGEON_MAX_DRAWS] = {0};
		const RendererDraw *active_draws = use_terrain ? terrain_draws : &quarry_draw;
		uint32_t active_draw_count = use_terrain ? terrain_draw_count : 1u;
		const RendererDraw *active_shadow_draws =
			use_terrain ? terrain_shadow_draws : active_draws;
		uint32_t active_shadow_draw_count =
			use_terrain ? terrain_shadow_draw_count : active_draw_count;
		RendererDraw *allocated_draws = NULL;
		if (use_dungeon)
		{
			active_draw_count = dungeon_scene_draws(&dungeon, camera.position, dungeon_draws,
											 DUNGEON_MAX_DRAWS, &active_shadow_draw_count);
			active_draws = dungeon_draws;
			active_shadow_draws = dungeon_draws;
		}
		if (use_quarry)
		{
			allocated_draws = calloc(2, sizeof(*allocated_draws));
			if (!allocated_draws)
			{
				running = false;
				continue;
			}
			allocated_draws[0] = quarry_draw;
			allocated_draws[1] = ground_draw;
			active_draws = allocated_draws;
			active_shadow_draws = allocated_draws;
			active_draw_count = 2;
			active_shadow_draw_count = 2;
		}
		if (!use_terrain && !use_quarry && !use_dungeon)
		{
			active_draw_count = use_phase_d_demo ? (demo_split ? 2u : 1u) : gltf.primitive_count;
			allocated_draws = calloc(active_draw_count, sizeof(*allocated_draws));
			if (!allocated_draws)
			{
				fprintf(stderr, "Out of memory building glTF draw list\n");
				running = false;
				continue;
			}
			for (uint32_t i = 0; i < active_draw_count; ++i)
			{
				if (use_phase_d_demo)
				{
					/* Flyby submits the single centred hero; stationary mode retains
					 * a simultaneous C/D split for diagnostic inspection. */
					bool d = phase_d_enabled;
					if (demo_split)
						d = i != 0; /* optional simultaneous C/D inspection */
					StaticMaterialParameters p = {.normal_strength = 1.f,
												  .ao_strength = 1.f,
												  .cavity_strength = 0.f,
												  .displacement_scale = 0.f};
					allocated_draws[i] = (RendererDraw){
						.mesh = &demo.panels[i],
						/* Stability demo never binds cavity: it would confound C/D. */
						.material_set = demo.neutral_set,
						.static_mesh = true,
						.push = static_material_push(demo.panels[i].local_to_world, camera.position,
													 white_rgba, 1.f, 1.f, &p, true, true, true,
													 true, d ? 1.f : 0.f)};
					continue;
				}
				GltfPrimitive *primitive = &gltf.primitives[i];
				GltfMaterial *material = &gltf.materials[primitive->material_index];
				allocated_draws[i] = (RendererDraw){
					.mesh = &primitive->mesh,
					.material_set = material->descriptor_set,
					.static_mesh = true,
					.push = static_material_push(
						primitive->mesh.local_to_world, camera.position,
						material->base_color_factor, material->metallic_factor,
						material->roughness_factor,
						&(StaticMaterialParameters){.normal_strength = material->normal_scale,
													.ao_strength = material->occlusion_strength,
													.cavity_strength = 0.f,
													.roughness_bias = 0.f,
													.displacement_scale = 0.f},
						default_lit, diffuse_ibl_enabled, specular_ibl_enabled, shadows_enabled,
						phase_d_enabled ? 1.f : 0.f)};
			}
			active_draws = allocated_draws;
			active_shadow_draws = allocated_draws;
			active_shadow_draw_count = active_draw_count;
		}

		vec2s jitter = temporal_jitter_ndc(temporal_frame++, renderer.swapchain_extent.width,
										   renderer.swapchain_extent.height);
		mat4s projection = temporal_jitter_projection(
			use_dungeon ? dungeon_camera_projection(&dungeon_camera, renderer_aspect(&renderer))
						: camera_projection(&camera, renderer_aspect(&renderer)),
			jitter);
		mat4s view = camera_view(&camera);
		mat4s view_projection = glms_mat4_mul(projection, view);
		if (!use_history)
		{
			previous_projection = projection;
			previous_view = view;
			previous_view_projection = view_projection;
			previous_jitter = jitter;
			previous_camera_position = camera.position;
		}
		mat4s current_to_previous_camera = GLMS_MAT4_IDENTITY_INIT;
		current_to_previous_camera.raw[3][0] =
			(float)(camera.position.x - previous_camera_position.x);
		current_to_previous_camera.raw[3][1] =
			(float)(camera.position.y - previous_camera_position.y);
		current_to_previous_camera.raw[3][2] =
			(float)(camera.position.z - previous_camera_position.z);

		float sun_horizontal = cosf(sun_orbit_angle);
		vec3s to_sun = {{sun_horizontal * cosf(sun_azimuth), sinf(sun_orbit_angle),
						  sun_horizontal * sinf(sun_azimuth)}};
		vec3s sun_direction = glms_vec3_scale(to_sun, -1.0f);
		ShadowCascadeConfig shadow_config =
			shadow_cascade_default_config(renderer_aspect(&renderer));
		shadow_config.resolution = renderer.shadow_resolution;
		ShadowCascadeSet shadow_cascades;
		if (!shadow_cascade_build(&shadow_config, camera.position, camera_forward_direction,
								  (vec3s){{0.0f, 1.0f, 0.0f}}, sun_direction, &shadow_cascades))
		{
			fprintf(stderr, "Could not build sun shadow cascades\n");
			running = false;
			continue;
		}

		float shader_dump_enabled = 0.0f;
#ifdef DEBUG_SHADER_DUMP
		shader_dump_enabled = dump_this_frame ? 1.0f : 0.0f;
#endif
		/* Camera world phase lets side-projected terrain reconstruct a globally
		   continuous Y coordinate from each tile's camera-relative transform.
		   Keep the period equal to terrain_runtime.c's X/Z material phase. */
		const double material_phase_period_m = 4096.0;
		FrameUniforms frame = {
			.projection = projection,
			.view = view,
			.view_projection = view_projection,
			.inverse_view_projection = glms_mat4_inv(view_projection),
			.previous_projection = previous_projection,
			.previous_view = previous_view,
			.previous_view_projection = previous_view_projection,
			.local_to_camera_relative = GLMS_MAT4_IDENTITY_INIT,
			.previous_local_to_camera_relative = current_to_previous_camera,
			.sun_direction = glms_vec4(sun_direction, 0.0f),
			.time = (float)(ticks - start_ticks) / 1000000000.0f,
			.near_plane = CAMERA_NEAR_PLANE,
			.debug_view = (float)debug_mode,
			.relight_strength = (const float[]){0.0f, 0.35f, 1.0f}[relight_mode],
			.shadow_splits = (vec4s){{shadow_config.split_m[0], shadow_config.split_m[1],
									  shadow_config.split_m[2], shadow_config.split_m[3]}},
			.shadow_parameters = (vec4s){{0.35f, 1.75f, 1.0f, 1.25f}},
			.shadow_radii = (vec4s){{shadow_cascades.radius_m[0], shadow_cascades.radius_m[1],
									 shadow_cascades.radius_m[2], shadow_cascades.radius_m[3]}},
			.shadow_quality =
				(vec4s){{(float)shadow_quality.filter_mode, shadow_quality.pcf_radius_texels,
						 atmosphere.sun_angular_radius_rad, shadow_quality.blocker_search_m}},
			.shadow_pcss = (vec4s){{shadow_quality.max_filter_radius_texels,
									(float)renderer.shadow_resolution, 0.0f, 0.0f}},
			.sun_radiance = use_dungeon ? (vec4s){{0.22f, 0.20f, 0.18f, 0.0f}}
										 : (vec4s){{1.6f, 1.5f, 1.35f, 0.0f}},
			.atmosphere_radii = (vec4s){{atmosphere.bottom_radius_km, atmosphere.top_radius_km,
										 fmaxf((float)camera.position.y * 0.001f, 0.001f),
										 atmosphere.sun_angular_radius_rad}},
			.atmosphere_rayleigh =
				(vec4s){{atmosphere.rayleigh_scattering[0], atmosphere.rayleigh_scattering[1],
						 atmosphere.rayleigh_scattering[2], atmosphere.rayleigh_density_exp_scale}},
			.atmosphere_mie_scatter =
				(vec4s){{atmosphere.mie_scattering[0], atmosphere.mie_scattering[1],
						 atmosphere.mie_scattering[2], atmosphere.mie_density_exp_scale}},
			.atmosphere_mie_extinct =
				(vec4s){{atmosphere.mie_extinction[0], atmosphere.mie_extinction[1],
						 atmosphere.mie_extinction[2], atmosphere.mie_phase_g}},
			.atmosphere_absorption =
				(vec4s){{atmosphere.absorption_extinction[0], atmosphere.absorption_extinction[1],
						 atmosphere.absorption_extinction[2], 0.0f}},
			.atmosphere_ground =
				(vec4s){{atmosphere.ground_albedo[0], atmosphere.ground_albedo[1],
						 atmosphere.ground_albedo[2], atmosphere.multiple_scattering_factor}},
			.atmosphere_options = (vec4s){{atmosphere.aerial_max_distance_km,
										   (float)atmosphere_slice, use_dungeon ? 0.0f : 1.0f, 0.0f}},
			.temporal_parameters =
				(vec4s){{use_history ? 1.0f : 0.0f, dt, 0.0f, auto_exposure_enabled ? 1.0f : 0.0f}},
			.temporal_jitter = (vec4s){{jitter.x, jitter.y, previous_jitter.x, previous_jitter.y}},
			.shader_dump = (vec4s){{shader_dump_enabled,
									  surface_detail_phase(camera.position.x, material_phase_period_m),
									  surface_detail_phase(camera.position.y, material_phase_period_m),
									  surface_detail_phase(camera.position.z, material_phase_period_m)}},
			.material_curvature = (vec4s){{1.0f, 0.0f, 0.333f, 0.0f}},
			.material_normal_filter = (vec4s){{1.0f, 0.20f, 0.0001f, 0.0f}},
			.stretch_overlay = (vec4s){{stretch_overlay.enabled ? 1.0f : 0.0f,
										stretch_overlay.threshold, stretch_overlay.opacity, 0.0f}},
			.point_light_options =
				use_dungeon ? (vec4s){{0.0f, 0.25f, 0.25f, 0.0f}}
							: (vec4s){{0.0f, 1.0f, 1.0f, 0.0f}},
		};
		if (use_dungeon)
		{
			frame.point_light_options.x = (float)dungeon_scene_write_lights(
				&dungeon, camera.position, frame.point_light_position_radius,
				frame.point_light_color_intensity, MAX_POINT_LIGHTS);
			if (!dungeon_scene_prepare_shadows(&dungeon, &renderer)) {
				fprintf(stderr, "Could not build dungeon shadow geometry\n");
				break;
			}
			frame.point_shadow_origin = (vec4s){{(float)camera.position.x,
				(float)camera.position.y, (float)camera.position.z, (float)dungeon.shadow.count}};
		}
		for (uint32_t i = 0; i < SHADOW_CASCADE_COUNT; ++i)
			frame.shadow_view_projection[i] = shadow_cascades.view_projection[i];
#ifdef DEBUG_SHADER_DUMP
		/* Diagnostic knobs for the bias sweep. shadow_parameters.x is the receiver
		   normal-offset bias (metres); .y is the PCF radius in texels. */
		if (getenv("TERRAIN_SHADOW_NBIAS"))
			frame.shadow_parameters.x = (float)atof(getenv("TERRAIN_SHADOW_NBIAS"));
#endif
		renderer_draw_frame(&renderer, &frame, active_draws, active_draw_count, active_shadow_draws,
							active_shadow_draw_count, input.resized);
		if (harness)
			dungeon_harness_post_frame(harness, &renderer);
		free(allocated_draws);
		if (use_terrain)
			terrain_runtime_collect_evictions(terrain);
#ifdef DEBUG_SHADER_DUMP
		/* Dump reads the buffer the frame above just populated. Clear first so a
		   same-frame C+X starts a fresh file. */
		if (input.clear_shader_dump)
			renderer_clear_shader_dump(SHADER_DUMP_PATH);
		if (input.dump_shader_data)
		{
			char metadata[256];
			snprintf(metadata, sizeof(metadata),
					 "scene=%s static_mesh_mode=%u demo_split=%u flyby_replay=%u phase=%c "
					 "sun_azimuth=%.6f sun_orbit=%.6f (replay: TERRAIN_START_SUN_AZIMUTH=%.6f "
					 "TERRAIN_START_SUN_ORBIT=%.6f)",
					 scene_name,
					 static_mesh_shading_mode, demo_split ? 1u : 0u, demo_flyby_phase_d ? 1u : 0u,
					 phase_d_enabled ? 'D' : 'C', sun_azimuth, sun_orbit_angle, sun_azimuth,
					 sun_orbit_angle);
			renderer_dump_shader_data(&renderer, SHADER_DUMP_PATH, camera.position, camera.yaw,
									  camera.pitch, metadata);
		}
		if (auto_dump_after && ++rendered_frames >= auto_dump_after)
		{
			{
				char metadata[256];
				snprintf(metadata, sizeof(metadata),
						 "scene=%s static_mesh_mode=%u demo_split=%u flyby_replay=%u phase=%c "
						 "sun_azimuth=%.6f sun_orbit=%.6f (replay: TERRAIN_START_SUN_AZIMUTH=%.6f "
						 "TERRAIN_START_SUN_ORBIT=%.6f)",
						 scene_name,
						 static_mesh_shading_mode, demo_split ? 1u : 0u,
						 demo_flyby_phase_d ? 1u : 0u, phase_d_enabled ? 'D' : 'C', sun_azimuth,
						 sun_orbit_angle, sun_azimuth, sun_orbit_angle);
				renderer_dump_shader_data(&renderer, SHADER_DUMP_PATH, camera.position, camera.yaw,
										  camera.pitch, metadata);
			}
			running = false;
		}
#endif

		previous_projection = projection;
		previous_view = view;
		previous_view_projection = view_projection;
		previous_jitter = jitter;
		previous_camera_position = camera.position;
		previous_camera_yaw = camera.yaw;
		previous_camera_pitch = camera.pitch;
		history_valid = true;
	}

	renderer_wait_idle(&renderer);
	int harness_status = dungeon_harness_exit_code(harness);
	dungeon_harness_destroy(harness);
	if (use_terrain)
		terrain_runtime_destroy(terrain);
	else if (use_dungeon)
		dungeon_scene_destroy(&renderer, &dungeon);
	else if (use_quarry)
	{
		benchmark_ground_destroy(&renderer, &ground);
		quarry_destroy(&renderer, &quarry);
	}
	else if (use_phase_d_demo)
		material_stability_demo_destroy(&renderer, &demo);
	else
		gltf_scene_destroy(&renderer, &gltf);
	renderer_shutdown(&renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	/* A scripted run's exit status is its verdict, so it can be used as a test. */
	return harness_status ? EXIT_FAILURE : EXIT_SUCCESS;
}
