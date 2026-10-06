"""Bake the Indiana Jones character study into a game-ready runtime asset.

    blender --background assets/characters/indiana_jones/indiana_jones.blend \
        --python tools/export_indiana_game.py

The study is 380k triangles across 19 materials, several of them flat colours
and the rest using physical-scale UV mapping the engine's glTF loader does not
reproduce. For the game it is collapsed to one posed (torch raised), decimated
mesh with ONE baked texture set:

  albedo  -- the studio materials' base colour, multiplied by baked ambient
             occlusion so seams, folds and the hat band read as grime, then
             graded toward a dusty, desaturated field look (see grade_albedo)
  normal  -- the high-detail surface baked onto the low mesh
  orm     -- R=1, G=baked roughness pushed toward matte, B=0

Output: assets/characters/indiana_jones/runtime/indiana.gltf (+ .bin, textures)
and runtime/manifest.json with the bounds and the torch head position the game
hangs the flame and light from.
"""
import bpy
import bmesh
import json
import math
import numpy as np
from mathutils import Vector
from pathlib import Path

ROOT = Path(bpy.data.filepath).resolve().parent
OUT = ROOT / "runtime"
OUT.mkdir(exist_ok=True)
TARGET_TRIANGLES = 42000
BAKE_SIZE = 2048


def log(*args):
    print("[indiana-game]", *args, flush=True)


scene = bpy.context.scene
scene.frame_set(1)
depsgraph = bpy.context.evaluated_depsgraph_get()
character = bpy.data.collections["INDIANA • character"]

# --- High: every character mesh, posed and subdivided, joined -------------
high_parts = []
torch_points = []
for obj in character.all_objects:
    if obj.type != "MESH" or obj.hide_render:
        continue
    evaluated = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
    mesh.transform(obj.matrix_world)
    part = bpy.data.objects.new("high_" + obj.name, mesh)
    scene.collection.objects.link(part)
    high_parts.append(part)
    if obj.name.startswith("Torch • linen"):
        torch_points.extend(v.co.copy() for v in mesh.vertices)
log("high parts", len(high_parts))
bpy.ops.object.select_all(action="DESELECT")
for part in high_parts:
    part.select_set(True)
bpy.context.view_layer.objects.active = high_parts[0]
bpy.ops.object.join()
high = bpy.context.view_layer.objects.active
high.name = "Indiana_high"
high_tris = sum(len(p.vertices) - 2 for p in high.data.polygons)
log("high triangles", high_tris)

# --- Low: decimated copy with a fresh atlas UV ------------------------------
low = high.copy()
low.data = high.data.copy()
low.name = "Indiana"
scene.collection.objects.link(low)
bpy.ops.object.select_all(action="DESELECT")
low.select_set(True)
bpy.context.view_layer.objects.active = low
# Weld the separate cages first so decimation does not open every seam.
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.remove_doubles(threshold=0.0004)
bpy.ops.object.mode_set(mode="OBJECT")
ratio = min(1.0, TARGET_TRIANGLES / max(high_tris, 1))
decimate = low.modifiers.new("decimate", "DECIMATE")
decimate.ratio = ratio
bpy.ops.object.modifier_apply(modifier=decimate.name)
low.data.materials.clear()
for layer in list(low.data.uv_layers):
    low.data.uv_layers.remove(layer)
low.data.uv_layers.new(name="atlas")
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.004)
bpy.ops.uv.pack_islands(margin=0.003)
bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.object.shade_smooth()
low_tris = sum(len(p.vertices) - 2 for p in low.data.polygons)
log("low triangles", low_tris)

# --- Bake ------------------------------------------------------------------
images = {}
for name, colour_space in (("albedo", "sRGB"), ("ao", "Non-Color"), ("normal", "Non-Color"),
                           ("rough", "Non-Color")):
    image = bpy.data.images.new("indiana_" + name, BAKE_SIZE, BAKE_SIZE, alpha=False,
                                float_buffer=False)
    image.colorspace_settings.name = colour_space
    images[name] = image
material = bpy.data.materials.new("Indiana • baked")
material.use_nodes = True
low.data.materials.append(material)
nodes = material.node_tree.nodes
bake_node = nodes.new("ShaderNodeTexImage")
nodes.active = bake_node

scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.render.bake.use_selected_to_active = True
scene.render.bake.cage_extrusion = 0.015
scene.render.bake.max_ray_distance = 0.05
scene.render.bake.margin = 6
bpy.ops.object.select_all(action="DESELECT")
high.select_set(True)
low.select_set(True)
bpy.context.view_layer.objects.active = low


def bake(kind, image, samples=1, **kwargs):
    bake_node.image = image
    scene.cycles.samples = samples
    log("baking", kind, image.name)
    bpy.ops.object.bake(type=kind, **kwargs)


bake("DIFFUSE", images["albedo"], pass_filter={"COLOR"})
bake("NORMAL", images["normal"], normal_space="TANGENT")
bake("ROUGHNESS", images["rough"])
# Occlusion from the LOW mesh itself is enough for crease grime and is far
# cheaper than tracing the 380k-triangle source.
scene.render.bake.use_selected_to_active = False
bake("AO", images["ao"], samples=48)


def pixels(image):
    data = np.empty(BAKE_SIZE * BAKE_SIZE * 4, dtype=np.float32)
    image.pixels.foreach_get(data)
    return data.reshape(BAKE_SIZE, BAKE_SIZE, 4)


def blur(channel, radius):
    """Separable box blur, repeated: a cheap Gaussian for grime masks."""
    out = channel
    for _ in range(3):
        for axis in (0, 1):
            kernel = np.ones(2 * radius + 1, dtype=np.float32) / (2 * radius + 1)
            out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), axis, out)
    return out


def value_noise(size, cells, seed):
    rng = np.random.default_rng(seed)
    grid = rng.random((cells + 1, cells + 1)).astype(np.float32)
    coords = np.linspace(0, cells, size, endpoint=False, dtype=np.float32)
    i = coords.astype(int)
    f = coords - i
    f = f * f * (3 - 2 * f)
    rows = grid[i][:, i] * (1 - f)[None, :] + grid[i][:, i + 1] * f[None, :]
    rows_next = grid[i + 1][:, i] * (1 - f)[None, :] + grid[i + 1][:, i + 1] * f[None, :]
    return rows * (1 - f)[:, None] + rows_next * f[:, None]


def grade_albedo(albedo, ao):
    """Field-worn rather than costume-fresh. Everything here works in the
    display-referred values Blender stores for an sRGB image."""
    rgb = albedo[..., :3]
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    # 1. Pull saturation down: the studio colours read as toy plastic.
    rgb = luma[..., None] + (rgb - luma[..., None]) * 0.72
    # 2. Crease grime from occlusion, softened so it reads as dirt, not shadow.
    occlusion = np.clip(blur(ao[..., 0], 2), 0.0, 1.0)
    rgb *= (0.55 + 0.45 * occlusion)[..., None]
    # 3. Large blotchy dust and sweat staining, plus fine speckle.
    big = value_noise(BAKE_SIZE, 9, 11)
    mid = value_noise(BAKE_SIZE, 37, 23)
    fine = value_noise(BAKE_SIZE, 260, 37)
    stain = 0.82 + 0.18 * (0.6 * big + 0.4 * mid)
    speck = 0.93 + 0.07 * fine
    rgb *= (stain * speck)[..., None]
    # 4. A dusty desert tint, strongest where the cloth is already pale.
    dust = np.array([0.62, 0.55, 0.44], dtype=np.float32)
    dust_mask = np.clip((0.5 * big + 0.5 * mid - 0.35) * 0.9, 0.0, 0.35) * np.clip(luma * 2.0, 0.2, 1.0)
    rgb = rgb * (1 - dust_mask[..., None]) + dust * dust_mask[..., None]
    # 5. Overall a little darker and lower contrast: torchlit, not studio-lit.
    rgb = np.clip(rgb * 0.9, 0.0, 1.0)
    out = albedo.copy()
    out[..., :3] = rgb
    out[..., 3] = 1.0
    return out


albedo = grade_albedo(pixels(images["albedo"]), pixels(images["ao"]))
images["albedo"].pixels.foreach_set(albedo.ravel())

rough = pixels(images["rough"])[..., 0]
# Nothing on him should gloss like vinyl: matte the cloth and leather.
rough = np.clip(0.35 + 0.65 * rough, 0.55, 1.0) * (0.95 + 0.05 * value_noise(BAKE_SIZE, 60, 5))
orm = np.ones((BAKE_SIZE, BAKE_SIZE, 4), dtype=np.float32)
orm[..., 1] = rough
orm[..., 2] = 0.0
images["orm"] = bpy.data.images.new("indiana_orm", BAKE_SIZE, BAKE_SIZE, alpha=False)
images["orm"].colorspace_settings.name = "Non-Color"
images["orm"].pixels.foreach_set(orm.ravel())

for name, filename, file_format in (("albedo", "indiana_albedo.jpg", "JPEG"),
                                    ("normal", "indiana_normal.png", "PNG"),
                                    ("orm", "indiana_orm.png", "PNG")):
    image = images[name]
    image.filepath_raw = str(OUT / filename)
    image.file_format = file_format
    image.save()
    log("wrote", filename)

# --- Final material and export ----------------------------------------------
nodes.clear()
links = material.node_tree.links
output = nodes.new("ShaderNodeOutputMaterial")
bsdf = nodes.new("ShaderNodeBsdfPrincipled")
links.new(bsdf.outputs[0], output.inputs[0])
albedo_node = nodes.new("ShaderNodeTexImage")
albedo_node.image = bpy.data.images.load(str(OUT / "indiana_albedo.jpg"))
links.new(albedo_node.outputs["Color"], bsdf.inputs["Base Color"])
orm_node = nodes.new("ShaderNodeTexImage")
orm_node.image = bpy.data.images.load(str(OUT / "indiana_orm.png"))
orm_node.image.colorspace_settings.name = "Non-Color"
separate = nodes.new("ShaderNodeSeparateColor")
links.new(orm_node.outputs["Color"], separate.inputs[0])
links.new(separate.outputs["Green"], bsdf.inputs["Roughness"])
links.new(separate.outputs["Blue"], bsdf.inputs["Metallic"])
normal_node = nodes.new("ShaderNodeTexImage")
normal_node.image = bpy.data.images.load(str(OUT / "indiana_normal.png"))
normal_node.image.colorspace_settings.name = "Non-Color"
normal_map = nodes.new("ShaderNodeNormalMap")
links.new(normal_node.outputs["Color"], normal_map.inputs["Color"])
links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

bpy.ops.object.select_all(action="DESELECT")
low.select_set(True)
bpy.context.view_layer.objects.active = low
bpy.ops.export_scene.gltf(
    filepath=str(OUT / "indiana.gltf"), export_format="GLTF_SEPARATE", use_selection=True,
    export_apply=True, export_animations=False, export_skins=False, export_morph=False,
    export_yup=True, export_cameras=False, export_lights=False, export_tangents=True,
    export_texcoords=True, export_normals=True, export_keep_originals=True,
    export_extras=False)


def to_gltf(v):
    """Blender Z-up to glTF Y-up."""
    return [round(v.x, 5), round(v.z, 5), round(-v.y, 5)]


corners = [low.matrix_world @ Vector(c) for c in low.bound_box]
lo = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
hi = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
torch_head = sum(torch_points, Vector()) / max(len(torch_points), 1)
manifest = {
    "source": "assets/characters/indiana_jones/indiana_jones.blend (frame 1, torch raised)",
    "triangles": low_tris,
    "source_triangles": high_tris,
    "bounds_min": to_gltf(lo),
    "bounds_max": to_gltf(hi),
    "torch_head": to_gltf(torch_head),
    "front": "Blender -Y, which glTF calls +Z",
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
log("manifest", manifest)
