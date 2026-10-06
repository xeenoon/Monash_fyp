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

The mesh is exported in BIND pose, skinned to the study's 55-bone rig (four
weights per vertex), with two looping clips authored here on top of the
study's raised-torch pose -- "Idle" (breathing, a slow sway of the torch arm)
and "Walk" (a striding gait that keeps the torch up) -- which the engine
blends by speed and skins on the CPU (src/game_character.c).

Output: assets/characters/indiana_jones/runtime/indiana.gltf (+ .bin, textures)
and runtime/manifest.json with the bounds, the torch head (bind space) and the
bone that holds it, and the ground speed the Walk clip covers at 1x.
"""
import bpy
import bmesh
from mathutils import Quaternion
import json
import math
import os
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
rig = bpy.data.objects["Indiana_Rig"]
# Meshes are taken in BIND pose: the engine skins them, and every pose --
# including the raised torch -- comes back through the clips.
rig.data.pose_position = "REST"
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
character = bpy.data.collections["INDIANA • character"]

# --- High: every character mesh, posed and subdivided, joined -------------
high_parts = []
torch_points = []
torch_votes = {}
for obj in character.all_objects:
    if obj.type != "MESH" or obj.hide_render:
        continue
    evaluated = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True,
                                           depsgraph=depsgraph)
    mesh.transform(obj.matrix_world)
    part = bpy.data.objects.new("high_" + obj.name, mesh)
    for group in obj.vertex_groups:
        part.vertex_groups.new(name=group.name)
    scene.collection.objects.link(part)
    high_parts.append(part)
    if obj.name.startswith("Torch • linen"):
        torch_points.extend(v.co.copy() for v in mesh.vertices)
        # Which bone carries the torch: the heaviest group on its head.
        for v in mesh.vertices:
            for g in v.groups:
                name = part.vertex_groups[g.group].name if g.group < len(part.vertex_groups) else None
                if name:
                    torch_votes[name] = torch_votes.get(name, 0.0) + g.weight
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
# The 700-odd source objects are folded into `high` now; keep them out of
# every later depsgraph evaluation (each frame change in the mocap retarget
# would otherwise re-skin all of them).
def find_layer(layer, name):
    if layer.collection.name == name:
        return layer
    for child in layer.children:
        found = find_layer(child, name)
        if found:
            return found
    return None


character_layer = find_layer(bpy.context.view_layer.layer_collection, character.name)
if character_layer:
    scene.collection.objects.link(rig)  # the rig stays in the view layer
    character_layer.exclude = True

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
# The study's garments are separate shells with mixed winding; welded and
# decimated, a third of the faces ended up pointing inward (the engine dump
# showed the hat top with normals facing the floor, so it rendered black, and
# the selected-to-active bake cast its rays the wrong way there too). Make the
# winding consistent and outward before anything is baked along it.
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode="OBJECT")
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


def bake_textures():
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
        """Field-worn rather than costume-fresh, without crushing it: the costume
        is dark leather and khaki, and the first grade multiplied enough
        darkening steps together that it rendered near black in torchlight.
        Values are the display-referred ones Blender stores for an sRGB image."""
        rgb = albedo[..., :3]
        weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        covered = rgb.max(axis=2) > 0.02  # atlas islands, not the empty margin
        luma = rgb @ weights
        # 1. Ease the saturation down a little: the studio colours read as toy.
        rgb = luma[..., None] + (rgb - luma[..., None]) * 0.8
        # 2. Gentle crease grime from occlusion.
        occlusion = np.clip(blur(ao[..., 0], 2), 0.0, 1.0)
        rgb *= (0.72 + 0.28 * occlusion)[..., None]
        # 3. Blotchy dust and sweat staining, plus fine speckle.
        big = value_noise(BAKE_SIZE, 9, 11)
        mid = value_noise(BAKE_SIZE, 37, 23)
        fine = value_noise(BAKE_SIZE, 260, 37)
        rgb *= ((0.9 + 0.1 * (0.6 * big + 0.4 * mid)) * (0.95 + 0.05 * fine))[..., None]
        # 4. Dust settles in a pale desert tint.
        dust = np.array([0.66, 0.58, 0.47], dtype=np.float32)
        dust_mask = np.clip((0.5 * big + 0.5 * mid - 0.4) * 0.6, 0.0, 0.2)
        rgb = rgb * (1 - dust_mask[..., None]) + dust * dust_mask[..., None]
        # 5. Level the whole costume so its median sits at a mid-dark value the
        #    torches can actually show, keeping relative contrast.
        median = float(np.median((rgb @ weights)[covered])) if covered.any() else 0.3
        rgb = np.clip(rgb * (0.36 / max(median, 1e-3)), 0.0, 1.0)
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



# Rebaking costs minutes and the decimation/UV unwrap is deterministic, so
# animation-only iterations can reuse the textures on disk:
# INDIANA_REBAKE=0 blender ... skips straight to rigging and export.
if os.environ.get("INDIANA_REBAKE", "1") != "0" or not (OUT / "indiana_albedo.jpg").exists():
    bake_textures()
else:
    log("reusing baked textures")

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

# --- Skin -------------------------------------------------------------------
bpy.ops.object.select_all(action="DESELECT")
low.select_set(True)
bpy.context.view_layer.objects.active = low
bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=4)
bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)
low.parent = rig
low.matrix_parent_inverse = rig.matrix_world.inverted()
armature = low.modifiers.new("Armature", "ARMATURE")
armature.object = rig
torch_bone = max(torch_votes, key=torch_votes.get) if torch_votes else "hand_r"
log("torch bone", torch_bone)

# --- Clips ------------------------------------------------------------------
rig.data.pose_position = "POSE"
scene.render.fps = 30
base_action = rig.animation_data.action
scene.frame_set(1)
bpy.context.view_layer.update()
bones = rig.pose.bones
base = {}
for pb in bones:
    if pb.rotation_mode != "QUATERNION":
        q = pb.rotation_euler.to_quaternion()
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = q
    base[pb.name] = (pb.rotation_quaternion.copy(), pb.location.copy())
rig.animation_data.action = None
for pb in bones:
    pb.rotation_quaternion, pb.location = base[pb.name][0].copy(), base[pb.name][1].copy()
bpy.context.view_layer.update()


def world_point(bone, tail=True):
    pb = bones[bone]
    return rig.matrix_world @ (pb.tail if tail else pb.head)


def forward_sign(bone, probe, axis="X"):
    """+1 if turning `bone` positively about its local `axis` moves `probe`
    forward (Blender -Y is the character's front), else -1."""
    pb = bones[bone]
    before = world_point(probe).y
    pb.rotation_quaternion = base[bone][0] @ Quaternion(Vector((1, 0, 0)) if axis == "X" else Vector((0, 0, 1)), 0.3)
    bpy.context.view_layer.update()
    after = world_point(probe).y
    pb.rotation_quaternion = base[bone][0].copy()
    bpy.context.view_layer.update()
    return 1.0 if after < before else -1.0


X = Vector((1, 0, 0))
Y = Vector((0, 1, 0))
thigh_sign = {side: forward_sign("thigh_" + side, "calf_" + side) for side in "lr"}
# A knee only folds backwards: positive bend must take the foot back.
knee_sign = {side: -forward_sign("calf_" + side, "foot_" + side) for side in "lr"}
arm_sign = forward_sign("upperarm_l", "hand_l")
log("signs", thigh_sign, knee_sign, arm_sign)


def key_pose(frame, offsets, pelvis_lift=0.0):
    """Key every bone: base pose times its offset rotations (local axes)."""
    for pb in bones:
        q = base[pb.name][0].copy()
        for axis, angle in offsets.get(pb.name, []):
            q = q @ Quaternion(axis, angle)
        pb.rotation_quaternion = q
        pb.keyframe_insert("rotation_quaternion", frame=frame)
        loc = base[pb.name][1].copy()
        if pb.name == "pelvis":
            loc.y += pelvis_lift  # the pelvis bone points up: local Y is height
        pb.location = loc
        pb.keyframe_insert("location", frame=frame)


# Mocap joint -> rig bone. Bones not listed keep the study's pose (the torch
# arm and both hands' grips) and simply ride on their parents.
MOCAP_MAP = {
    "LowerBack": "spine_01", "Spine": "spine_02", "Spine1": "spine_03",
    "Neck": "neck_01", "Head": "head",
    "LeftUpLeg": "thigh_l", "LeftLeg": "calf_l", "LeftFoot": "foot_l", "LeftToeBase": "ball_l",
    "RightUpLeg": "thigh_r", "RightLeg": "calf_r", "RightFoot": "foot_r", "RightToeBase": "ball_r",
    "LeftShoulder": "clavicle_l", "LeftArm": "upperarm_l", "LeftForeArm": "lowerarm_l",
}


phase_offsets = {}


def retarget_walk(path, clip_name="Walk"):
    """Bake one in-place gait cycle of a BVH walk onto the rig as "Walk".

    Direction retargeting: each mapped bone is swung so it points the way its
    mocap counterpart points, which is indifferent to the two skeletons having
    different rest poses (the BVH stands in a T, the study in an A). The pelvis
    takes the mocap hips' full rotation plus their bob and sway; forward travel
    and the walker's heading are removed so the clip plays in place facing the
    character's front. Returns the clip's ground speed in m/s.
    """
    before = set(bpy.data.objects)
    bpy.ops.import_anim.bvh(filepath=str(path), global_scale=1.0, axis_forward="-Z", axis_up="Y",
                            update_scene_fps=False, update_scene_duration=False)
    src = (set(bpy.data.objects) - before).pop()
    action = src.animation_data.action
    first, last = (int(v) for v in action.frame_range)

    def joint(f, name, tail=True):
        if scene.frame_current != f:
            scene.frame_set(f)
        pb = src.pose.bones[name]
        return src.matrix_world @ (pb.tail if tail else pb.head)

    to_rig = rig.matrix_world.to_3x3().inverted()

    # Gait cycle from the feet: the left-minus-right foot separation along the
    # direction of travel peaks once per cycle.
    travel = (joint(last, "Hips", False) - joint(first, "Hips", False))
    travel.z = 0
    forward = travel.normalized()
    signal = []
    for f in range(first, last + 1):
        signal.append(((joint(f, "LeftFoot", False) - joint(f, "RightFoot", False)) @ forward, f))
    # Extremes of the separation: left foot furthest ahead (+) and right foot
    # furthest ahead (-) alternate every half cycle. A cycle runs between two
    # extremes of the same sign; prefer one that does not start on the very
    # first extreme, where the walker may still be setting off.
    extremes = []
    for i in range(6, len(signal) - 6):
        value, frame = signal[i]
        window = [x[0] for x in signal[max(0, i - 25):i + 26]]
        sign = 1 if value == max(window) and value > 0 else (-1 if value == min(window) and value < 0 else 0)
        if sign and (not extremes or frame - extremes[-1][0] > 25):
            extremes.append((frame, sign))
    pairs = [(extremes[i][0], extremes[i + 2][0], extremes[i][1]) for i in range(len(extremes) - 2)
             if extremes[i][1] == extremes[i + 2][1]]
    peaks = [e[0] for e in extremes]
    # Start on the left foot when the data allows, so clips share a phase.
    left = [p_ for p_ in pairs if p_[2] > 0]
    chosen = (left[1] if len(left) > 1 else left[0]) if left else (pairs[1] if len(pairs) > 1 else pairs[0])
    start, end, start_sign = chosen
    # 0 when the cycle starts with the left foot forward, 0.5 otherwise: the
    # engine adds it to the shared gait phase so blended clips stay in step.
    phase_offsets[clip_name] = 0.0 if start_sign > 0 else 0.5
    log("mocap cycle frames", start, end, "of", first, last, "peaks", peaks)

    # Scale by leg length, and a heading correction that turns the travel
    # direction onto the character's front (Blender -Y).
    src_leg = (joint(start, "LeftUpLeg", False) - joint(start, "LeftFoot", False)).length
    rig_leg = (rig.data.bones["thigh_l"].head_local - rig.data.bones["foot_l"].head_local).length
    scale = rig_leg / src_leg
    heading = Quaternion((0, 0, 1), -math.atan2(forward.x, -forward.y))

    step = 120.0 / scene.render.fps
    frames = []
    f = float(start)
    while f < end - 1e-6:
        frames.append(f)
        f += step
    duration = (end - start) / 120.0
    distance = ((joint(end, "Hips", False) - joint(start, "Hips", False)) @ forward) * scale
    hips_mean = sum((joint(int(round(x)), "Hips", False) for x in frames), Vector()) / len(frames)

    rest = {b.name: b.matrix_local.copy() for b in rig.data.bones}
    order = [b.name for b in rig.data.bones]  # parents precede children
    src_rest_hips = src.data.bones["Hips"].matrix_local.to_quaternion()
    make_action(clip_name)
    for key, sf in enumerate(frames + [frames[0]]):  # repeat the first: a seamless loop
        fi = int(round(sf))
        scene.frame_set(fi)
        posed = {}
        for name in order:
            bone = rig.data.bones[name]
            pb = bones[name]
            parent = bone.parent.name if bone.parent else None
            # Where the bone would sit with an identity pose, given its posed parent.
            if parent:
                frame_mat = posed[parent] @ rest[parent].inverted() @ rest[name]
            else:
                frame_mat = rest[name].copy()
            source = next((s_ for s_, t in MOCAP_MAP.items() if t == name), None)
            if name == "pelvis":
                hips = src.pose.bones["Hips"]
                world_rot = (src.matrix_world.to_quaternion() @ hips.matrix.to_quaternion())
                delta = heading @ world_rot @ (src.matrix_world.to_quaternion() @ src_rest_hips).inverted()
                rot = rest[name].to_quaternion().inverted() @ delta @ rest[name].to_quaternion()
                offset = heading @ ((src.matrix_world @ hips.head) - hips_mean)
                offset = Vector((offset.x, 0.0, offset.z)) * scale  # no forward travel
                local = rest[name].to_3x3().inverted() @ offset
                pb.rotation_quaternion = rot
                pb.location = local
            elif source:
                d = to_rig @ (heading @ (joint(fi, source) - joint(fi, source, tail=False)))
                d_local = frame_mat.to_3x3().inverted() @ d
                pb.rotation_quaternion = Vector((0, 1, 0)).rotation_difference(d_local.normalized())
                pb.location = base[name][1]
            else:
                pb.rotation_quaternion = base[name][0]
                pb.location = base[name][1]
            basis = pb.rotation_quaternion.to_matrix().to_4x4()
            basis.translation = pb.location
            posed[name] = frame_mat @ basis
            pb.keyframe_insert("rotation_quaternion", frame=key + 1)
            pb.keyframe_insert("location", frame=key + 1)
    bpy.data.objects.remove(src)
    bpy.data.actions.remove(action)  # or it would export as a clip of its own
    speed = distance / duration
    log(clip_name, "cycle", len(frames), "keys,", round(duration, 3), "s,", round(distance, 3), "m")
    return speed


def make_action(name):
    action = bpy.data.actions.new(name)
    action.use_fake_user = True
    rig.animation_data.action = action
    return action


# Walk: one gait cycle of CMU motion capture (assets/animations), retargeted.
MOCAP = ROOT.parents[1] / "animations" / "cmu_35_01_walk.bvh"
walk_speed = retarget_walk(MOCAP, "Walk")
run_speed = retarget_walk(ROOT.parents[1] / "animations" / "cmu_16_35_jog.bvh", "Run")
log("run speed", run_speed)

IDLE_FRAMES = 90  # 3 s breath
make_action("Idle")
for f in range(0, IDLE_FRAMES + 1, 6):
    phase = 2 * math.pi * f / IDLE_FRAMES
    o = {
        "spine_02": [(X, math.radians(1.2) * math.sin(phase))],
        "spine_03": [(X, math.radians(1.0) * math.sin(phase))],
        "upperarm_r": [(X, math.radians(1.5) * math.sin(phase + 1.0))],
        "head": [(Y, math.radians(4) * math.sin(phase * 0.5))],
    }
    key_pose(f + 1, o, pelvis_lift=0.004 * math.sin(phase))

log("walk speed", walk_speed)
rig.animation_data.action = None
if base_action:
    bpy.data.actions.remove(base_action)  # its pose lives inside both clips now
for pb in bones:
    pb.rotation_quaternion, pb.location = base[pb.name][0].copy(), base[pb.name][1].copy()

bpy.ops.object.select_all(action="DESELECT")
low.select_set(True)
rig.select_set(True)
bpy.context.view_layer.objects.active = low
export_args = dict(
    filepath=str(OUT / "indiana.gltf"), export_format="GLTF_SEPARATE", use_selection=True,
    export_apply=True, export_animations=True, export_animation_mode="ACTIONS",
    export_force_sampling=True, export_skins=True, export_morph=False, export_yup=True,
    export_cameras=False, export_lights=False, export_tangents=True, export_texcoords=True,
    export_normals=True, export_keep_originals=True, export_extras=False)
bpy.ops.export_scene.gltf(**export_args)


def to_gltf(v):
    """Blender Z-up to glTF Y-up."""
    return [round(v.x, 5), round(v.z, 5), round(-v.y, 5)]


corners = [low.matrix_world @ Vector(c) for c in low.bound_box]
lo = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
hi = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
torch_head = sum(torch_points, Vector()) / max(len(torch_points), 1)
manifest = {
    "source": "assets/characters/indiana_jones/indiana_jones.blend (bind pose, skinned)",
    "triangles": low_tris,
    "source_triangles": high_tris,
    "bounds_min": to_gltf(lo),
    "bounds_max": to_gltf(hi),
    "torch_head": to_gltf(torch_head),
    "torch_bone": torch_bone,
    "walk_speed": round(walk_speed, 3),
    "run_speed": round(run_speed, 3),
    "walk_phase": phase_offsets.get("Walk", 0.0),
    "run_phase": phase_offsets.get("Run", 0.0),
    "clips": ["Idle", "Walk", "Run"],
    "front": "Blender -Y, which glTF calls +Z",
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
log("manifest", manifest)
