#include <SDL3/SDL.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "atmosphere.h"
#include "camera.h"
#include "input.h"
#include "gltf_scene.h"
#include "quarry.h"
#include "benchmark_ground.h"
#include "material_stability_demo.h"
#include "renderer.h"
#include "temporal.h"

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

static ShadowQualitySettings shadow_quality_from_environment(void)
{
	ShadowQualitySettings q = {.filter_mode = SHADOW_FILTER_PCF, .resolution = 2048u,
		.pcf_radius_texels = 1.75f, .sun_angular_radius_rad = 0.004675f,
		.blocker_search_m = 12.0f, .max_filter_radius_texels = 24.0f};
	const char *mode = getenv("TERRAIN_SHADOW_FILTER");
	if (mode && strcmp(mode, "hard") == 0) q.filter_mode = SHADOW_FILTER_HARD;
	else if (mode && strcmp(mode, "pcss") == 0) q.filter_mode = SHADOW_FILTER_PCSS;
	else if (mode && strcmp(mode, "pcf") != 0) { fprintf(stderr, "TERRAIN_SHADOW_FILTER must be hard, pcf, or pcss\n"); exit(EXIT_FAILURE); }
	const char *resolution = getenv("TERRAIN_SHADOW_RESOLUTION");
	if (resolution) { q.resolution = (uint32_t)strtoul(resolution, NULL, 10); if (q.resolution != 2048u && q.resolution != 4096u) { fprintf(stderr, "TERRAIN_SHADOW_RESOLUTION must be 2048 or 4096\n"); exit(EXIT_FAILURE); } }
	return q;
}

/* The sole CPU packing contract for all static materials.  geometry retains
 * glTF base-colour RGBA; shadow state lives in debug.z, never in alpha. */
static DrawPushConstants static_material_push(LocalToWorldTransform transform,
                                               WorldPosition camera_position,
                                               const float base_color[4], float metallic,
                                               float gltf_roughness,
                                               const StaticMaterialParameters *parameters,
                                               bool default_lit, bool diffuse_ibl,
                                               bool specular_ibl, bool shadows,
                                               float curvature_strength)
{
    return (DrawPushConstants){
        .local_to_camera_relative = coordinate_local_to_camera_relative(&transform, camera_position),
        .geometry = {{base_color[0], base_color[1], base_color[2], base_color[3]}},
        .elevation_uv = {{gltf_roughness, parameters->normal_strength, parameters->ao_strength,
                          parameters->roughness_bias}},
        .material = {{metallic, default_lit ? 1.f : 0.f, diffuse_ibl ? 1.f : 0.f,
                      specular_ibl ? 1.f : 0.f}},
        .debug = {{parameters->cavity_strength, parameters->displacement_scale,
                   shadows ? 1.f : 0.f, curvature_strength}}};
}

int main(void)
{
	const char *scene = getenv("TERRAIN_SCENE");
	bool use_quarry = scene && strcmp(scene, "quarry") == 0;
	bool use_phase_d_demo = scene && strcmp(scene, "phase_d_demo") == 0;
	const char *gltf_path = NULL;
	/* The large benchmark is intentionally not in git.  Check before Vulkan so
	 * a no-argument invocation never quietly falls back to Quarry. */
	if (!scene || strcmp(scene, "coastal_cliff") == 0)
	{
		gltf_path = BENCHMARK_DIR "/coastal_cliff_01/coastal_cliff_01_4k.gltf";
		if (access(gltf_path, R_OK) != 0)
		{
			fprintf(stderr, "Coastal Cliff benchmark assets are missing. Download with:\n"
					"python3 tools/download_benchmark_assets.py\n");
			return EXIT_FAILURE;
		}
	}
	else if (!use_quarry && !use_phase_d_demo && strcmp(scene, "gltf") == 0)
	{
		gltf_path = getenv("TERRAIN_GLTF_PATH");
		if (!gltf_path || !*gltf_path)
		{
			fprintf(stderr, "TERRAIN_SCENE=gltf requires TERRAIN_GLTF_PATH\n");
			return EXIT_FAILURE;
		}
	}
	else if (!use_quarry && !use_phase_d_demo)
	{
		fprintf(stderr, "Supported scene selectors: coastal_cliff, quarry, phase_d_demo, or gltf.\n");
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
	if (!SDL_SetWindowRelativeMouseMode(window, true))
		fprintf(stderr, "Relative mouse mode unavailable: %s\n", SDL_GetError());

	Renderer renderer;
	ShadowQualitySettings shadow_quality = shadow_quality_from_environment();
	renderer_init(&renderer, window, &(RendererConfig){.environment_path = getenv("TERRAIN_ENV_HDR"), .shadow_quality = shadow_quality});
	Quarry quarry = {0};
	BenchmarkGround ground = {0};
	GltfScene gltf = {0};
	GltfLoadError gltf_error = {0};
	MaterialStabilityDemo demo = {0};
	GltfLoadResult load_result = use_phase_d_demo ? GLTF_LOAD_OK : (use_quarry ?
		(quarry_create(&renderer, &quarry, QUARRY_DIR) ? GLTF_LOAD_OK : GLTF_LOAD_IO_ERROR) :
		gltf_scene_create(&renderer, gltf_path,
			&(GltfLoadOptions){.use_metallic_roughness_red_as_occlusion = !scene || strcmp(scene, "coastal_cliff") == 0,
				/* Poly Haven's authored front faces +Z; the benchmark camera is
				 * placed on -Z, so turn only this named benchmark toward it. */
				.placement = (!scene || strcmp(scene, "coastal_cliff") == 0)
					? coordinate_rotation_y(3.14159265358979323846, (WorldPosition){0})
					: coordinate_identity_transform((WorldPosition){0})}, &gltf, &gltf_error));
	if (load_result != GLTF_LOAD_OK)
	{
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
		quarry_destroy(&renderer, &quarry); renderer_shutdown(&renderer); SDL_DestroyWindow(window); SDL_Quit();
		return EXIT_FAILURE;
	}
	if (use_phase_d_demo && !material_stability_demo_create(&renderer, &demo))
	{
		fprintf(stderr, "Could not create Phase D demo; download benchmark assets first.\n");
		renderer_shutdown(&renderer); SDL_DestroyWindow(window); SDL_Quit(); return EXIT_FAILURE;
	}
	Camera camera = {
		.position = use_phase_d_demo ? (WorldPosition){0.0, 1.0, -18.0} : (use_quarry ? (WorldPosition){0.0, 8.0, -25.0} : (WorldPosition){0.0, 6.0, -45.0}),
		.yaw = 90.0f,
		.pitch = use_quarry ? -3.0f : -2.0f,
	};

	Input input = {.mouse_captured = true};
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
	bool demo_flyby = use_phase_d_demo && getenv("TERRAIN_DEMO_FLYBY") && atoi(getenv("TERRAIN_DEMO_FLYBY")) != 0;
	bool demo_split = use_phase_d_demo && !demo_flyby && getenv("TERRAIN_DEMO_SPLIT") && atoi(getenv("TERRAIN_DEMO_SPLIT")) != 0;
	if (demo_flyby)
		SDL_SetWindowTitle(window, "Phase C — raw normal detail");
	bool demo_flyby_phase_d = false;
	unsigned debug_mode = 0;
	unsigned relight_mode = 1;
	unsigned sun_mode = 0;
	unsigned atmosphere_slice = 15;
	/* F3 exposes the delivery sequence: legacy -> A (Default Lit) -> B1
	   (diffuse IBL) -> B2 (specular IBL) -> C (shadowing) -> legacy. */
	unsigned static_mesh_shading_mode = 5u;
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
			camera.position = (WorldPosition){x, y, z};
	}
#endif
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
			const char *names[] = {"legacy PBR", "Phase A: Unreal Default Lit",
								   "Phase B1: Default Lit + diffuse sky IBL",
								   "Phase B2: Default Lit + diffuse and specular sky IBL",
								   "Phase C: B2 + cascaded shadows",
								   "Phase D: Phase C + material detail stability"};
			printf("Static-mesh renderer: %s\n", names[static_mesh_shading_mode]);
			if (use_phase_d_demo)
				SDL_SetWindowTitle(window, static_mesh_shading_mode >= 5u ?
					"Phase D — stabilized material detail" : "Phase C — raw normal detail");
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
			debug_mode = debug_mode >= 3u && debug_mode < 6u ? debug_mode + 1u
															 : (debug_mode == 6u ? 0u : 3u);
			const char *names[] = {"Mapped normal", "Authored roughness", "Curvature roughness",
				"Effective roughness (R authored, G effective, B curvature)", "off"};
			printf("Static-mesh material view: %s\n", debug_mode >= 3u && debug_mode <= 6u ? names[debug_mode - 3u] : names[4]);
		}
		if (input.cycle_shadow_debug)
			debug_mode = debug_mode >= 7u && debug_mode < 11u ? debug_mode + 1u
															  : (debug_mode == 11u ? 0u : 7u);
		if (input.cycle_atmosphere_debug)
			debug_mode = debug_mode >= 12u && debug_mode < 16u ? debug_mode + 1u
															   : (debug_mode == 16u ? 0u : 12u);
		if (debug_mode != previous_debug_mode)
			history_valid = false;
		if (input.previous_atmosphere_slice && atmosphere_slice > 0u)
			--atmosphere_slice;
		if (input.next_atmosphere_slice && atmosphere_slice < 31u)
			++atmosphere_slice;
		if (input.previous_atmosphere_slice || input.next_atmosphere_slice)
			printf("Atmosphere volume debug slice: %u/31\n", atmosphere_slice);
		if (input.cycle_sun)
		{
			sun_mode = (sun_mode + 1u) % 3u;
			const char *names[] = {"afternoon", "sunset", "high sun"};
			printf("Atmosphere sun: %s\n", names[sun_mode]);
			history_valid = false;
		}

		uint64_t ticks = SDL_GetTicksNS();
		float dt = (float)(ticks - previous_ticks) / 1000000000.0f;
		previous_ticks = ticks;
		if (dt > 0.1f)
			dt = 0.1f;

		camera_update(&camera, input.move_forward, input.move_right, input.look_dx, input.look_dy,
					  input.sprint, dt);
		if (demo_flyby)
		{
			float seconds = fmodf((float)(ticks - start_ticks) / 1000000000.0f, 20.0f);
			bool phase_d = seconds >= 10.0f;
			float replay = fmodf(seconds, 10.0f) / 10.0f;
			/* Identical far -> near -> far camera path for C then D. */
			float near_weight = 1.0f - fabsf(2.0f * replay - 1.0f);
			camera.position = (WorldPosition){0.0, 1.0, -30.0 + 18.0 * near_weight};
			camera.yaw = 90.f; camera.pitch = 0.f;
			if (phase_d != demo_flyby_phase_d) {
				demo_flyby_phase_d = phase_d; history_valid = false;
				SDL_SetWindowTitle(window, phase_d ? "Phase D — stabilized material detail" :
					"Phase C — raw normal detail");
			}
			static_mesh_shading_mode = phase_d ? 5u : 4u;
			if (seconds < dt) history_valid = false;
		}
		bool camera_cut =
			history_valid &&
			temporal_camera_cut(previous_camera_position, camera.position, previous_camera_yaw,
								camera.yaw, previous_camera_pitch, camera.pitch, 2000.0);
		bool use_history = history_valid && !input.resized && !camera_cut;

		vec3s camera_forward_direction = camera_forward(&camera);
		bool default_lit = static_mesh_shading_mode != 0u;
		bool diffuse_ibl_enabled = static_mesh_shading_mode >= 2u;
		bool specular_ibl_enabled = static_mesh_shading_mode >= 3u;
		bool shadows_enabled = static_mesh_shading_mode >= 4u;
		bool phase_d_enabled = static_mesh_shading_mode >= 5u;
		StaticMaterialParameters quarry_parameters = {.normal_strength = 1.0f, .ao_strength = 1.0f,
			.cavity_strength = phase_d_enabled && quarry.cavity_available ? 0.25f : 0.0f, .roughness_bias = 0.0f,
			.displacement_scale = phase_d_enabled && quarry.cavity_available ? 1.0f : 0.0f};
		const float white_rgba[4] = {1.f, 1.f, 1.f, 1.f};
		DrawPushConstants quarry_push = static_material_push(
			quarry.base.local_to_world, camera.position, white_rgba, quarry.metallic_factor, 1.f,
			&quarry_parameters, default_lit, diffuse_ibl_enabled, specular_ibl_enabled,
			shadows_enabled, phase_d_enabled ? 1.f : 0.f);
		RendererDraw quarry_draw = {
			.mesh = &quarry.base, .material_set = phase_d_enabled && quarry.cavity_available ? quarry.cavity_material_set : quarry.base.material_set,
			.push = quarry_push, .static_mesh = true};
		DrawPushConstants ground_push = quarry_push;
		ground_push = static_material_push(ground.mesh.local_to_world, camera.position, white_rgba,
			quarry.metallic_factor, 1.f, &quarry_parameters, default_lit, diffuse_ibl_enabled,
			specular_ibl_enabled, shadows_enabled, phase_d_enabled ? 1.f : 0.f);
		ground_push.debug.y = 0.0f; /* ground retains the neutral cavity descriptor */
		RendererDraw ground_draw = {.mesh = &ground.mesh, .push = ground_push, .static_mesh = true};
		RendererDraw *active_draws = &quarry_draw;
		uint32_t active_draw_count = 1;
		RendererDraw *active_shadow_draws = &quarry_draw;
		uint32_t active_shadow_draw_count = 1;
		if (use_quarry) { active_draws = calloc(2, sizeof(*active_draws)); if (!active_draws) { running=false; continue; } active_draws[0]=quarry_draw; active_draws[1]=ground_draw; active_draw_count=2; }
		if (!use_quarry)
		{
			active_draw_count = use_phase_d_demo ? (demo_split ? 2u : 1u) : gltf.primitive_count;
			active_draws = calloc(active_draw_count, sizeof(*active_draws));
			if (!active_draws)
			{
				fprintf(stderr, "Out of memory building glTF draw list\n");
				running = false;
				continue;
			}
			for (uint32_t i = 0; i < active_draw_count; ++i)
			{
				if (use_phase_d_demo) {
					/* Flyby submits the single centred hero; stationary mode retains
					 * a simultaneous C/D split for diagnostic inspection. */
					bool d = phase_d_enabled;
					if (demo_split)
						d = i != 0; /* optional simultaneous C/D inspection */
					StaticMaterialParameters p = {.normal_strength=1.f,.ao_strength=1.f,
						.cavity_strength = i == 2 ? .25f : 0.f, .displacement_scale=d ? 1.f : 0.f};
					active_draws[i] = (RendererDraw){.mesh=&demo.panels[i],
						.material_set = d ? demo.cavity_set : demo.neutral_set, .static_mesh=true,
						.push=static_material_push(demo.panels[i].local_to_world,camera.position,white_rgba,
							1.f,1.f,&p,true,true,true,true,d ? 1.f : 0.f)};
					continue;
				}
				GltfPrimitive *primitive = &gltf.primitives[i];
				GltfMaterial *material = &gltf.materials[primitive->material_index];
				active_draws[i] = (RendererDraw){
					.mesh = &primitive->mesh,
					.material_set = material->descriptor_set,
					.static_mesh = true,
					.push = static_material_push(primitive->mesh.local_to_world, camera.position,
						material->base_color_factor, material->metallic_factor, material->roughness_factor,
						&(StaticMaterialParameters){.normal_strength = material->normal_scale,
							.ao_strength = material->occlusion_strength, .cavity_strength = 0.f,
							.roughness_bias = 0.f, .displacement_scale = 0.f}, default_lit,
						diffuse_ibl_enabled, specular_ibl_enabled, shadows_enabled,
						phase_d_enabled ? 1.f : 0.f)};
			}
			active_shadow_draws = active_draws;
			active_shadow_draw_count = active_draw_count;
		}

		vec2s jitter = temporal_jitter_ndc(temporal_frame++, renderer.swapchain_extent.width,
										   renderer.swapchain_extent.height);
		mat4s projection = temporal_jitter_projection(
			camera_projection(&camera, renderer_aspect(&renderer)), jitter);
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

		const vec3s sun_directions[3] = {
			{{-0.4f, -1.0f, -0.3f}},
			{{-1.0f, -0.08f, -0.15f}},
			{{-0.1f, -1.0f, -0.05f}},
		};
		vec3s sun_direction = glms_vec3_normalize(sun_directions[sun_mode]);
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
			.shadow_radii = (vec4s){{shadow_cascades.radius_m[0], shadow_cascades.radius_m[1], shadow_cascades.radius_m[2], shadow_cascades.radius_m[3]}},
			.shadow_quality = (vec4s){{(float)shadow_quality.filter_mode, shadow_quality.pcf_radius_texels, atmosphere.sun_angular_radius_rad, shadow_quality.blocker_search_m}},
			.shadow_pcss = (vec4s){{shadow_quality.max_filter_radius_texels, (float)renderer.shadow_resolution, 0.0f, 0.0f}},
			.sun_radiance = (vec4s){{1.6f, 1.5f, 1.35f, 0.0f}},
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
			.atmosphere_options =
				(vec4s){{atmosphere.aerial_max_distance_km, (float)atmosphere_slice, 0.0f, 0.0f}},
			.temporal_parameters = (vec4s){{use_history ? 1.0f : 0.0f, dt, 0.0f, 0.0f}},
			.temporal_jitter = (vec4s){{jitter.x, jitter.y, previous_jitter.x, previous_jitter.y}},
			.material_detail_settings = (vec4s){{1.0f, 0.0f, 0.333f, 0.0f}},
#ifdef DEBUG_SHADER_DUMP
			.shader_dump = (vec4s){{dump_this_frame ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f}},
#endif
		};
		for (uint32_t i = 0; i < SHADOW_CASCADE_COUNT; ++i)
			frame.shadow_view_projection[i] = shadow_cascades.view_projection[i];
#ifdef DEBUG_SHADER_DUMP
		/* Diagnostic knobs for the bias sweep. shadow_parameters.x is the receiver
		   normal-offset bias (metres); .y is the PCF radius in texels. */
		if (getenv("TERRAIN_SHADOW_NBIAS"))
			frame.shadow_parameters.x = (float)atof(getenv("TERRAIN_SHADOW_NBIAS"));
#endif
		renderer_draw_frame(&renderer, &frame, active_draws, active_draw_count,
							active_shadow_draws, active_shadow_draw_count, input.resized);
		free(active_draws);
#ifdef DEBUG_SHADER_DUMP
		/* Dump reads the buffer the frame above just populated. Clear first so a
		   same-frame C+X starts a fresh file. */
		if (input.clear_shader_dump)
			renderer_clear_shader_dump(SHADER_DUMP_PATH);
		if (input.dump_shader_data)
			renderer_dump_shader_data(&renderer, SHADER_DUMP_PATH, camera.position, camera.yaw,
									  camera.pitch);
		if (auto_dump_after && ++rendered_frames >= auto_dump_after)
		{
			renderer_dump_shader_data(&renderer, SHADER_DUMP_PATH, camera.position, camera.yaw,
									  camera.pitch);
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
	if (use_quarry)
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
	return EXIT_SUCCESS;
}
