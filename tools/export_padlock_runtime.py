#!/usr/bin/env python3
"""Bake assets/dungeons/padlock/padlock_animated.blend down to a runtime glTF.

    blender -b assets/dungeons/padlock/padlock_animated.blend \
            --python tools/export_padlock_runtime.py -- --out assets/dungeons/padlock/runtime

The source is a render asset: 275 bone-parented objects, 964k triangles, 21
fully procedural materials and no textures at all.  None of that can be loaded
by the engine, so this tool produces the runtime form:

  * the tear-off variant is dropped (PHASE 2 -- see README), leaving the
    intact lock and the eight picking clips;
  * curves and text become meshes and every modifier is BAKED DOWN before the
    exporter runs.  Exporting with live Decimate modifiers made the glTF
    exporter re-evaluate them once per baked animation frame: the same export
    that takes 3 seconds with baked meshes did not finish in ten minutes;
  * each object is decimated to a triangle cap, then every object sharing a
    bone is joined, which turns 241 draws into one per animated bone;
  * the procedural materials are baked into ONE atlas (albedo / ORM / normal),
    because a shader cannot evaluate Blender's noise and voronoi trees and a
    per-part material set would cost a descriptor set and a draw per part;
  * animation channels that never leave the node's rest pose are stripped, so
    clips touch disjoint node sets and the engine can LAYER them -- playing
    Pin_02_Pop must not snap the pick back to rest.

The lock is exported at TRUE SIZE in metres (the source's 0.022 unit scale);
the game applies its own exaggeration when it places the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import bmesh  # type: ignore
import bpy  # type: ignore
from mathutils import Vector  # type: ignore

# Blender's unit scale in the source file: 1 Blender unit is 22 mm.
SOURCE_UNIT_SCALE = 0.022

# The inspection window, in source units (1 unit = 22 mm), in the source's own
# axes: X across the lock's width, Y along the keyway, Z up.
#
# The source is a CLOSED padlock, and a closed padlock does not show its pins:
# casting rays at the four key pins from 2000 directions, every direction that
# reaches all four is more than 14 degrees BELOW the lock -- they are visible up
# through the keyway and nowhere else, in the intact state and with the front
# plate peeled alike, since the pins live inside the plug and the tear does not
# open the plug. A lock whose pin heights cannot be read is not the puzzle this
# replaces: the dungeon's lock has always been drawn as a casing with a cutaway
# showing the pin stack in section (docs/dungeon_padlock.md).
#
# The first version of this cut the lock in half down its width. That did show
# the pins -- and left an open shoebox with half a shackle on top, because the
# half it removed included the entire front face the padlock is recognised by.
# So it is a WINDOW instead: a box cut out of the near wall over the mechanism,
# leaving the casing's frame, the face, the shackle and the silhouette intact.
# Wide enough that the mechanism is still in view at the angle the lock is
# mounted at, which the tool verifies by ray-casting at the pins rather than
# trusting these numbers.
# Which wall the window goes in follows from the angle the lock is mounted at.
# The pin row runs along the lock's DEPTH (Y here), so the sightline to each pin
# leaves the lock through the FACE, not the side wall: from the mounted angle a
# ray reaching the deepest pin crosses the front plane about half a unit off
# centre, well inside the face and nowhere near the side. The first attempt cut
# the side wall and every pin stayed buried behind the rear one.
#
# So the window is a box taken out of the face and the front of the plug,
# stopping short of the pins themselves -- there is half a unit of clearance
# between the front wall and the first pin to land it in.
WINDOW_MIN = (-1.20, -3.00, 0.60)
WINDOW_MAX = (1.20, -0.40, 2.20)

# The pick tunnels down the keyway, straight through the window box. It is a
# tool, not part of the lock, and cutting it would leave the player holding a
# stub.
WINDOW_KEEP_BONES = ("PICK",)

# Clips the runtime needs.  Full_Picking_Sequence is a preview edit of the
# others and Tear_Front_Plate is phase 2.
RUNTIME_ACTIONS = (
    "Pick_Insert",
    "Pick_Withdraw",
    "Pick_Jiggle_Loop",
    "Pin_01_Pop",
    "Pin_02_Pop",
    "Pin_03_Pop",
    "Pin_04_Pop",
    "Unlock_All_Four_Set",
)

# Collections and object prefixes that exist only for the still renders or for
# the tear-off, which the runtime does not have.
DROP_COLLECTIONS = (
    "STUDIO | cameras and lighting",
    "TEAR-OFF | plate, pry bar and released hardware",
)
DROP_PREFIXES = (
    "FRONT PLATE |",     # the pre-cut fractured plate; INTACT FRONT replaces it
    "REMOVED PLATE |",
    "PRY BAR |",
    "STUDIO |",
    # Stamped maker/serial text: 32k triangles of lettering that decimates into
    # noise and reads as nothing at runtime.  The baked normal map carries the
    # surface it sat on.
    "STAMP |",
)
# Bones whose geometry is phase 2.  They have to go from the ARMATURE too, not
# just lose their objects: Blender keys every bone into every action, so a bone
# left behind exports a channel into all eight clips, and a clip that carries
# channels it does not animate cannot be layered under another one.
DROP_BONES = ("PRY_BAR", "TORN_RIVET_1", "TORN_RIVET_2", "TORN_RIVET_3", "TORN_RIVET_4",
              "FRONT_CENTRE")


def log(message: str) -> None:
    print(f"[padlock] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="runtime asset directory")
    parser.add_argument("--triangle-cap", type=int, default=800,
                        help="per-object triangle cap before joining")
    parser.add_argument("--atlas", type=int, default=4096, help="baked atlas resolution")
    # Occlusion is the one ray-traced pass and it dominates the export: the
    # others are direct shader evaluations and take a sample each. 24 is clean
    # enough for a channel that only darkens crevices; raise it if the atlas
    # shows speckle in the deep recesses.
    parser.add_argument("--ao-samples", type=int, default=24)
    parser.add_argument("--rewrite-clips", action="store_true",
                        help="re-apply the clip channel policy to an export already in --out, "
                             "without re-running the bake. Which channels a clip carries lives "
                             "in the glTF, not in the atlas.")
    parser.add_argument("--skip-bake", action="store_true",
                        help="reuse the atlas already in --out. Only valid when the geometry "
                             "and the unwrap are unchanged -- the UVs are regenerated either "
                             "way, and a changed unwrap does not match an old atlas")
    return parser.parse_args(argv)


# ---------------------------------------------------------------- scene prep

def only(objects):
    """Selects exactly `objects`, making the first one active."""
    bpy.ops.object.select_all(action="DESELECT")
    active = None
    for obj in objects:
        obj.select_set(True)
        active = active or obj
    bpy.context.view_layer.objects.active = active
    return active


def prune() -> None:
    for name in DROP_COLLECTIONS:
        collection = bpy.data.collections.get(name)
        if not collection:
            continue
        for obj in list(collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    for obj in list(bpy.data.objects):
        if obj.name.startswith(DROP_PREFIXES) or obj.parent_bone in DROP_BONES:
            bpy.data.objects.remove(obj, do_unlink=True)
        elif obj.type in {"CAMERA", "LIGHT", "EMPTY"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for action in list(bpy.data.actions):
        if action.name not in RUNTIME_ACTIONS:
            bpy.data.actions.remove(action)
    log(f"pruned to {len(bpy.data.objects)} objects, {len(bpy.data.actions)} actions")


def drop_degenerate() -> None:
    """Drops parts that are still collapsed once the rig is at rest.

    Must run AFTER rest_the_rig: the source is saved mid-tear with the pick
    scaled to zero, and dropping zero-scaled objects before the pose is cleared
    deletes the pick.
    """
    for obj in list(bpy.data.objects):
        if obj.type != "MESH":
            continue
        if min(abs(component) for component in obj.matrix_world.to_scale()) < 1e-7:
            log(f"dropping collapsed {obj.name!r}")
            bpy.data.objects.remove(obj, do_unlink=True)


def drop_bones(armature) -> None:
    only([armature])
    bpy.ops.object.mode_set(mode="EDIT")
    for name in DROP_BONES:
        bone = armature.data.edit_bones.get(name)
        if bone:
            armature.data.edit_bones.remove(bone)
    bpy.ops.object.mode_set(mode="OBJECT")
    log(f"rig is {len(armature.data.bones)} bones after dropping the phase 2 controls")


def rest_the_rig(armature) -> None:
    """Puts the rig in its rest pose and clears the tear property and shape keys.

    Clearing the action is NOT enough: the source is saved part-way into
    Tear_Front_Plate, and a pose bone keeps whatever transform it was last
    evaluated to.  The file on disk holds PICK at scale zero -- the tear action
    hides the pick -- so an export that only dropped the action collapsed the
    whole lockpick to a point, and every clip then carried a PICK scale channel
    that "differed from rest" because rest itself was wrong.  The source's own
    README says to reset pose transforms before changing actions; this is that.

    The shape keys are all driven by `front_tear`; leaving them in would export
    morph targets and `weights` channels, which the runtime importer does not
    have and phase 1 does not need.
    """
    armature["front_tear"] = 0.0
    if armature.animation_data:
        armature.animation_data.action = None
    only([armature])
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.pose.transforms_clear()
    bpy.ops.object.mode_set(mode="OBJECT")
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.data.shape_keys:
            only([obj])
            obj.shape_key_clear()


def to_meshes() -> None:
    convertible = [o for o in bpy.data.objects if o.type in {"CURVE", "FONT", "SURFACE", "META"}]
    if convertible:
        only(convertible)
        bpy.ops.object.convert(target="MESH")
    log(f"converted {len(convertible)} curve/text objects to meshes")


def triangles_of(obj) -> int:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    if mesh is None:
        return 0
    count = sum(len(polygon.vertices) - 2 for polygon in mesh.polygons)
    evaluated.to_mesh_clear()
    return count


def bake_modifiers() -> None:
    """Applies every modifier, and gives every object its own mesh data.

    Applying is not an optimisation: the glTF exporter re-evaluates live
    modifiers once per baked animation frame, which at this object count does
    not terminate.

    The single-user pass is what makes the section cut below correct. Mirrored
    hardware in the source is linked duplicates -- one mesh datablock under two
    object transforms -- and cutting through a shared mesh cuts it in the first
    object's frame for both, which left the right-hand latch standing whole on
    the near side of a plane that had already removed its twin.
    """
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if meshes:
        only(meshes)
        for obj in meshes:
            obj.select_set(True)
        bpy.ops.object.make_single_user(object=True, obdata=True)
    for obj in meshes:
        only([obj])
        bpy.ops.object.convert(target="MESH")
    log(f"baked the modifier stacks of {len(meshes)} objects")


def section() -> int:
    """Cuts an inspection window through the near wall, over the mechanism.

    A boolean against a box rather than a plane bisect: a bisect can only take a
    whole half, and the half in front of the pins is also the half the padlock
    is recognised by. Only objects that actually reach into the window are cut,
    which keeps this to the casing, the face and the plug rather than running a
    solver over 200 screws.
    """
    cutter_mesh = bpy.data.meshes.new("WINDOW")
    cutter = bpy.data.objects.new("WINDOW", cutter_mesh)
    bpy.context.scene.collection.objects.link(cutter)
    mesh = bmesh.new()
    bmesh.ops.create_cube(mesh, size=1.0)
    for vertex in mesh.verts:
        for axis in range(3):
            vertex.co[axis] = (WINDOW_MAX[axis] if vertex.co[axis] > 0 else WINDOW_MIN[axis])
    mesh.to_mesh(cutter_mesh)
    mesh.free()

    def reaches_window(obj):
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            if all(WINDOW_MIN[a] - 1e-6 <= point[a] <= WINDOW_MAX[a] + 1e-6 for a in range(3)):
                return True
        # A wall can span the window without any corner inside it.
        low = [min((obj.matrix_world @ Vector(c))[a] for c in obj.bound_box) for a in range(3)]
        high = [max((obj.matrix_world @ Vector(c))[a] for c in obj.bound_box) for a in range(3)]
        return all(low[a] < WINDOW_MAX[a] and high[a] > WINDOW_MIN[a] for a in range(3))

    cut = 0
    removed = 0
    for obj in [o for o in bpy.data.objects if o.type == "MESH" and o is not cutter]:
        if obj.parent_bone in WINDOW_KEEP_BONES or not reaches_window(obj):
            continue
        before = len(obj.data.polygons)
        modifier = obj.modifiers.new("window", "BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        modifier.solver = "EXACT"
        only([obj])
        bpy.ops.object.convert(target="MESH")
        removed += before - len(obj.data.polygons)
        cut += 1
    bpy.data.objects.remove(cutter, do_unlink=True)
    empty = [o for o in bpy.data.objects if o.type == "MESH" and not o.data.polygons]
    for obj in empty:
        bpy.data.objects.remove(obj, do_unlink=True)
    log(f"cut the inspection window through {cut} objects "
        f"({removed:+d} faces, {len(empty)} left with nothing)")
    return removed


def check_pins_are_visible() -> None:
    """Rays at each pin from where the player's camera actually is.

    The window is the reason the mechanism can be read at all, so it is
    verified rather than assumed -- and verified from the mounted angle, since
    a window that clears a head-on view can still be blocked by its own edge at
    fifty degrees.

    The camera direction comes from DUNGEON_PADLOCK_FACE_TURN_DEGREES in
    src/dungeon_padlock_pose.h: the door normal is
    cos(t) * modelX + sin(t) * modelZ in the exported frame, which is
    (cos t, -sin t, 0) here, since the exporter maps source (x, y, z) to
    glTF (x, z, -y).
    """
    # Keep this in sync with DUNGEON_PADLOCK_FACE_TURN_DEGREES. The gameplay
    # view presents the broad lock face square to the player.
    turn = math.radians(90.0)
    direction = Vector((math.cos(turn), -math.sin(turn), 0.0)).normalized()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    pins = [o for o in bpy.data.objects
            if o.name.startswith("PIN 0") and "lower key pin" in o.name]
    if not pins:
        raise RuntimeError("no key pins left to look at")
    blocked = []
    for pin in pins:
        centre = pin.matrix_world.translation
        hit, _, _, _, obj, _ = bpy.context.scene.ray_cast(depsgraph, centre + direction * 5.0,
                                                          -direction)
        if hit and obj is not pin and "PIN 0" not in obj.name:
            blocked.append(f"{pin.name} is behind {obj.name}")
    if blocked:
        raise RuntimeError("the inspection window does not reach the pins:\n  " +
                           "\n  ".join(blocked))
    log(f"all {len(pins)} key pins are in view from the mounted angle")


def decimate(cap: int) -> int:
    """Caps every object's triangle count and bakes the decimation down."""
    total = 0
    for obj in [o for o in bpy.data.objects if o.type == "MESH"]:
        count = triangles_of(obj)
        if count > cap:
            modifier = obj.modifiers.new("runtime cap", "DECIMATE")
            modifier.ratio = cap / count
            modifier.use_collapse_triangulate = True
            only([obj])
            bpy.ops.object.convert(target="MESH")
        total += triangles_of(obj)
    log(f"{total} triangles after a {cap}-triangle per-object cap")
    return total


def join_per_bone() -> list:
    """One object per parent bone: 241 draws become one per animated bone."""
    groups: dict[str, list] = {}
    for obj in [o for o in bpy.data.objects if o.type == "MESH"]:
        groups.setdefault(obj.parent_bone or "", []).append(obj)
    joined = []
    for bone, members in sorted(groups.items()):
        if len(members) > 1:
            only(members)
            bpy.ops.object.join()
            survivor = bpy.context.view_layer.objects.active
        else:
            survivor = members[0]
        survivor.name = f"PADLOCK | {bone or 'UNPARENTED'}"
        joined.append(survivor)
    log(f"joined {len(joined)} bone groups")
    return joined


# ------------------------------------------------------------------- atlas

def surface_area(obj) -> float:
    matrix = obj.matrix_world
    area = 0.0
    for polygon in obj.data.polygons:
        corners = [matrix @ obj.data.vertices[i].co for i in polygon.vertices]
        for i in range(1, len(corners) - 1):
            area += (corners[i] - corners[0]).cross(corners[i + 1] - corners[0]).length * 0.5
    return area


def shelf_pack(sides: list[float]) -> list[tuple[float, float]] | None:
    """Places squares of the given sides into the unit square, tallest first."""
    order = sorted(range(len(sides)), key=lambda i: -sides[i])
    placements: list[tuple[float, float]] = [(0.0, 0.0)] * len(sides)
    x = y = shelf_height = 0.0
    for index in order:
        side = sides[index]
        if x + side > 1.0:
            x = 0.0
            y += shelf_height
            shelf_height = 0.0
        if y + side > 1.0:
            return None
        placements[index] = (x, y)
        x += side
        shelf_height = max(shelf_height, side)
    return placements


def atlas_unwrap(objects: list, margin_texels: float, atlas_size: int) -> None:
    """Smart-unwraps every object, then packs the objects into one UV square.

    Texel density follows each object's surface area, so the housing gets the
    resolution and a retaining screw does not.
    """
    for obj in objects:
        only([obj])
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.004,
                                 scale_to_bounds=True)
        bpy.ops.object.mode_set(mode="OBJECT")
    areas = [max(surface_area(obj), 1e-9) for obj in objects]
    scale = math.sqrt(0.86 / sum(areas))
    placements = None
    while scale > 1e-4:
        sides = [math.sqrt(area) * scale for area in areas]
        placements = shelf_pack(sides)
        if placements:
            break
        scale *= 0.94
    if not placements:
        raise RuntimeError("could not pack the atlas")
    inset = margin_texels / atlas_size
    for obj, (x, y), area in zip(objects, placements, areas):
        side = math.sqrt(area) * scale
        span = max(side - 2.0 * inset, 1e-5)
        for loop in obj.data.uv_layers.active.data:
            loop.uv.x = x + inset + loop.uv.x * span
            loop.uv.y = y + inset + loop.uv.y * span
    log(f"packed {len(objects)} objects into one UV atlas (cell scale {scale:.4f})")


# -------------------------------------------------------------------- bake

def principled_of(material):
    if not material.node_tree:
        return None
    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return None


def add_bake_targets(materials, image):
    nodes = []
    for material in materials:
        node = material.node_tree.nodes.new("ShaderNodeTexImage")
        node.image = image
        node.select = True
        material.node_tree.nodes.active = node
        nodes.append((material, node))
    return nodes


def drop_bake_targets(nodes):
    for material, node in nodes:
        material.node_tree.nodes.remove(node)


def bake_pass(objects, materials, image, bake_type: str, samples: int, margin: int, **settings):
    scene = bpy.context.scene
    scene.cycles.samples = samples
    scene.render.bake.margin = margin
    scene.render.bake.use_clear = True
    targets = add_bake_targets(materials, image)
    only(objects)
    for obj in objects:
        obj.select_set(True)
    bpy.ops.object.bake(type=bake_type, **settings)
    drop_bake_targets(targets)


def emission_override(materials, socket_name: str):
    """Rewires every material to emit one Principled input, for an EMIT bake.

    Cycles has no metallic bake, and the source materials drive metallic from
    node trees rather than constants in places.
    """
    saved = []
    for material in materials:
        tree = material.node_tree
        output = next(n for n in tree.nodes if n.type == "OUTPUT_MATERIAL")
        link = next((l for l in tree.links if l.to_node is output and l.to_socket.name == "Surface"),
                    None)
        # Read the socket NOW. Linking something else into Surface removes this
        # link -- an input takes one -- and reading from_socket off a removed
        # link afterwards restores nothing: the material is left with no
        # surface at all, and every later bake comes back blank. That is
        # exactly how the normal map baked flat.
        from_socket = link.from_socket if link else None
        principled = principled_of(material)
        emission = tree.nodes.new("ShaderNodeEmission")
        socket = principled.inputs[socket_name] if principled else None
        if socket is not None and socket.is_linked:
            tree.links.new(socket.links[0].from_socket, emission.inputs["Color"])
        else:
            value = float(socket.default_value) if socket is not None else 0.0
            emission.inputs["Color"].default_value = (value, value, value, 1.0)
        tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        saved.append((material, output, from_socket, emission))
    return saved


def restore_emission(saved):
    for material, output, from_socket, emission in saved:
        material.node_tree.nodes.remove(emission)
        if from_socket is not None:
            material.node_tree.links.new(from_socket, output.inputs["Surface"])


def bake_atlas(objects, out_dir: Path, size: int, ao_samples: int) -> dict:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.use_denoising = False
    scene.render.bake.use_selected_to_active = False
    materials = [m for m in bpy.data.materials if m.node_tree and principled_of(m)]
    log(f"baking {len(materials)} materials over {len(objects)} objects at {size}px")

    albedo = bpy.data.images.new("padlock_albedo", size, size)
    occlusion = bpy.data.images.new("padlock_ao", size, size, float_buffer=False, is_data=True)
    roughness = bpy.data.images.new("padlock_roughness", size, size, is_data=True)
    metallic = bpy.data.images.new("padlock_metallic", size, size, is_data=True)
    normal = bpy.data.images.new("padlock_normal", size, size, is_data=True)
    margin = max(4, size // 256)

    import numpy

    def baked(image, name, minimum_deviation):
        """Bakes are checked, not assumed: a blank map is a silent failure.

        The materials are rewired for the metallic pass, and a pass that comes
        back uniform means the rewiring took the surface with it -- which is
        not visible in the log, only in a lock that has lost its corrosion."""
        buffer = numpy.empty(size * size * 4, dtype=numpy.float32)
        image.pixels.foreach_get(buffer)
        deviation = float(buffer.reshape(-1, 4)[:, :3].std())
        log(f"  {name} done (deviation {deviation:.4f})")
        if deviation < minimum_deviation:
            raise RuntimeError(f"the {name} bake came back blank (deviation {deviation:.5f} < "
                               f"{minimum_deviation})")

    bake_pass(objects, materials, albedo, "DIFFUSE", 1, margin, pass_filter={"COLOR"})
    baked(albedo, "albedo", 0.01)
    bake_pass(objects, materials, roughness, "ROUGHNESS", 1, margin)
    baked(roughness, "roughness", 0.01)
    # The procedural bump -- corrosion, machining marks, the grain that is the
    # whole surface character of these materials -- lives in the shader, not in
    # the geometry, so this pass is where it survives decimation.
    bake_pass(objects, materials, normal, "NORMAL", 1, margin)
    baked(normal, "normal", 0.002)
    bake_pass(objects, materials, occlusion, "AO", ao_samples, margin)
    baked(occlusion, "occlusion", 0.01)
    # Last, because it rewires every material to emit one input and nothing
    # else may depend on the surface shader afterwards.
    saved = emission_override(materials, "Metallic")
    bake_pass(objects, materials, metallic, "EMIT", 1, margin)
    restore_emission(saved)
    log("  metallic done")

    # ORM, in the channel order the engine's importer expects: R occlusion
    # (GltfLoadOptions.use_metallic_roughness_red_as_occlusion), G roughness,
    # B metallic -- which is also glTF's own metallicRoughness layout.
    #
    # Through foreach_get/foreach_set and numpy, not image.pixels as a list: a
    # 4096 atlas is 67 million floats per channel, and reading three of those
    # into Python lists is several gigabytes of boxed floats before any of them
    # is combined.
    import numpy

    orm = bpy.data.images.new("padlock_orm", size, size, is_data=True)
    count = size * size * 4
    channels = []
    for image in (occlusion, roughness, metallic):
        buffer = numpy.empty(count, dtype=numpy.float32)
        image.pixels.foreach_get(buffer)
        channels.append(buffer)
    combined = numpy.empty(count, dtype=numpy.float32)
    combined[0::4] = channels[0][0::4]
    combined[1::4] = channels[1][0::4]
    combined[2::4] = channels[2][0::4]
    combined[3::4] = 1.0
    orm.pixels.foreach_set(combined)

    out_dir.mkdir(parents=True, exist_ok=True)
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 94
    albedo.filepath_raw = str(out_dir / "padlock_albedo.jpg")
    albedo.file_format = "JPEG"
    albedo.save()
    for image, name in ((orm, "padlock_orm.png"), (normal, "padlock_normal.png")):
        image.filepath_raw = str(out_dir / name)
        image.file_format = "PNG"
        image.save()
    log("atlas written")
    return {"albedo": "padlock_albedo.jpg", "orm": "padlock_orm.png",
            "normal": "padlock_normal.png"}


def single_material(objects, out_dir: Path, files: dict):
    """Replaces every source material with one that samples the baked atlas."""
    material = bpy.data.materials.new("PADLOCK | baked atlas")
    material.use_nodes = True
    # Solid metal: the runtime pipeline culls backfaces, and exporting
    # doubleSided would advertise a two-sided material the engine cannot honour.
    material.use_backface_culling = True
    tree = material.node_tree
    principled = principled_of(material)
    output = next(n for n in tree.nodes if n.type == "OUTPUT_MATERIAL")

    def image_node(name, colorspace):
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(str(out_dir / name), check_existing=True)
        node.image.colorspace_settings.name = colorspace
        return node

    base = image_node(files["albedo"], "sRGB")
    tree.links.new(base.outputs["Color"], principled.inputs["Base Color"])
    orm = image_node(files["orm"], "Non-Color")
    separate = tree.nodes.new("ShaderNodeSeparateColor")
    tree.links.new(orm.outputs["Color"], separate.inputs["Color"])
    tree.links.new(separate.outputs["Green"], principled.inputs["Roughness"])
    tree.links.new(separate.outputs["Blue"], principled.inputs["Metallic"])
    normal_image = image_node(files["normal"], "Non-Color")
    normal_map = tree.nodes.new("ShaderNodeNormalMap")
    tree.links.new(normal_image.outputs["Color"], normal_map.inputs["Color"])
    tree.links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])
    tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)
        # Vertex colours only exist to drive the procedural trees that are now
        # baked; exporting them would add two attributes the engine ignores.
        while obj.data.color_attributes:
            obj.data.color_attributes.remove(obj.data.color_attributes[0])
    log("materials collapsed to one baked atlas material")


# ------------------------------------------------------------------ export

REST = {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0]}

# Clips restricted to the nodes they are FOR.
#
# Pick_Jiggle_Loop is the idle loop that runs the whole time a lock is being
# worked, and the animator gave it small responses on all four pin stacks as
# well as the pick. In a render that is life; in the game it is a permanent
# wobble over a readout measured in millimetres -- and the pins are not the
# clip's to move anyway, since where a pin sits is the puzzle's answer (see
# dungeon_padlock_model_pose). Dropping the pin stacks from the loop leaves the
# pick doing what the loop is for, and leaves a pin still until a key moves it.
CLIP_NODES = {"Pick_Jiggle_Loop": {"PICK"}}


def restrict_clip_nodes(gltf: dict) -> dict:
    report = {}
    for animation in gltf.get("animations", []):
        allowed = CLIP_NODES.get(animation["name"])
        if not allowed:
            continue
        before = len(animation["channels"])
        animation["channels"] = [channel for channel in animation["channels"]
                                 if gltf["nodes"][channel["target"]["node"]].get("name") in allowed]
        report[animation["name"]] = (before, len(animation["channels"]))
    return report


def read_accessor(gltf: dict, buffer: bytes, index: int) -> list[list[float]]:
    import struct
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    if accessor["componentType"] != 5126:
        raise RuntimeError("animation data is expected to be float")
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride") or components * 4
    values = []
    for i in range(accessor["count"]):
        offset = start + i * stride
        values.append(list(struct.unpack_from(f"<{components}f", buffer, offset)))
    return values


def strip_rest_channels(gltf: dict, buffer: bytes) -> dict:
    """Drops channels that never leave the node's rest pose.

    Blender bakes every bone into every action, so each clip arrives with all
    28 bones keyed whether the animator touched them or not.  Layering
    Pin_02_Pop over Pick_Jiggle_Loop only works if a clip's channels are the
    parts it actually animates -- otherwise the pin clip snaps the pick, the
    core and the shackle back to rest underneath it.
    """
    report = {}
    for animation in gltf.get("animations", []):
        kept = []
        for channel in animation["channels"]:
            path = channel["target"]["path"]
            node = gltf["nodes"][channel["target"]["node"]]
            rest = node.get(path, REST[path])
            sampler = animation["samplers"][channel["sampler"]]
            values = read_accessor(gltf, buffer, sampler["output"])
            moves = any(any(abs(v - r) > 1e-6 for v, r in zip(value, rest)) for value in values)
            if moves:
                kept.append(channel)
        report[animation["name"]] = {"before": len(animation["channels"]), "after": len(kept),
                                     "nodes": sorted({gltf["nodes"][c["target"]["node"]]["name"]
                                                      for c in kept})}
        animation["channels"] = kept
    return report


def clip_duration(gltf: dict, buffer: bytes, animation: dict) -> float:
    end = 0.0
    for channel in animation["channels"]:
        times = read_accessor(gltf, buffer, animation["samplers"][channel["sampler"]]["input"])
        if times:
            end = max(end, times[-1][0])
    return end


def verify(gltf: dict) -> dict:
    """Fails loudly on anything the runtime importer cannot load."""
    problems = []
    if any("skin" in node for node in gltf["nodes"]):
        problems.append("a node references a skin; the runtime has no skinning")
    gltf.pop("skins", None)  # Blender emits an unreferenced 28-joint skin
    triangles = 0
    for mesh in gltf["meshes"]:
        for primitive in mesh["primitives"]:
            if primitive.get("mode", 4) != 4:
                problems.append(f"{mesh['name']}: not a triangle list")
            if "targets" in primitive:
                problems.append(f"{mesh['name']}: morph targets are phase 2")
            if "TEXCOORD_0" not in primitive["attributes"]:
                problems.append(f"{mesh['name']}: no TEXCOORD_0, the atlas would not map")
            for attribute in primitive["attributes"]:
                if attribute.startswith("COLOR") or attribute.startswith("JOINTS"):
                    problems.append(f"{mesh['name']}: unexpected {attribute}")
            indices = primitive.get("indices")
            triangles += (gltf["accessors"][indices]["count"] if indices is not None
                          else gltf["accessors"][primitive["attributes"]["POSITION"]]["count"]) // 3
    if len(gltf.get("materials", [])) != 1:
        problems.append(f"expected one baked material, found {len(gltf.get('materials', []))}")
    for animation in gltf.get("animations", []):
        for sampler in animation["samplers"]:
            if sampler.get("interpolation", "LINEAR") not in ("LINEAR", "STEP"):
                problems.append(f"{animation['name']}: {sampler['interpolation']} is unsupported")
    if problems:
        raise RuntimeError("runtime glTF is not loadable:\n  " + "\n  ".join(problems))
    return {"triangles": triangles, "nodes": len(gltf["nodes"]), "meshes": len(gltf["meshes"])}


def model_metrics(gltf: dict, buffer: bytes) -> dict:
    """Facts the engine's placement and its tests both need to agree on.

    `pin_travel` is the local translation a pin gains over its own Pop clip --
    the engine reads the same delta at load and uses it to show an UNSET pin's
    height, so the height readout is a fraction of the authored set travel
    rather than a second, invented constant.
    """
    names = [node.get("name", "") for node in gltf["nodes"]]
    metrics = {"pin_travel": {}, "pin_rest": {}}
    for pin in range(1, 5):
        bone = f"PIN_0{pin}"
        if bone not in names:
            continue
        index = names.index(bone)
        metrics["pin_rest"][bone] = gltf["nodes"][index].get("translation", [0.0, 0.0, 0.0])
        animation = next((a for a in gltf["animations"] if a["name"] == f"Pin_0{pin}_Pop"), None)
        if not animation:
            continue
        for channel in animation["channels"]:
            if channel["target"]["node"] == index and channel["target"]["path"] == "translation":
                values = read_accessor(gltf, buffer, animation["samplers"][channel["sampler"]]["output"])
                metrics["pin_travel"][bone] = [b - a for a, b in zip(values[0], values[-1])]
    return metrics


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(out_dir: Path, armature, files: dict, stats: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # True size: the source works in 22 mm units, the runtime works in metres.
    armature.scale = (SOURCE_UNIT_SCALE,) * 3
    path = out_dir / "padlock.gltf"
    bpy.ops.export_scene.gltf(
        filepath=str(path), export_format="GLTF_SEPARATE", export_apply=False,
        export_animations=True, export_animation_mode="ACTIONS", export_bake_animation=True,
        export_morph=False, export_skins=False, export_yup=True, export_cameras=False,
        export_lights=False, export_tangents=False, export_texcoords=True, export_normals=True,
        export_keep_originals=True, export_extras=False)
    gltf = json.loads(path.read_text(encoding="utf-8"))
    buffer = (out_dir / gltf["buffers"][0]["uri"]).read_bytes()
    stripped = strip_rest_channels(gltf, buffer)
    for name, counts in sorted(stripped.items()):
        log(f"  {name}: {counts['before']} -> {counts['after']} channels {counts['nodes']}")
    for name, (before, after) in sorted(restrict_clip_nodes(gltf).items()):
        log(f"  {name}: restricted to its own nodes, {before} -> {after} channels")
        stripped[name]["after"] = after
        stripped[name]["nodes"] = sorted(CLIP_NODES[name])
    stats.update(verify(gltf))
    metrics = model_metrics(gltf, buffer)
    clips = {a["name"]: round(clip_duration(gltf, buffer, a), 6) for a in gltf["animations"]}
    path.write_text(json.dumps(gltf, separators=(",", ":")), encoding="utf-8")

    manifest = {
        "source": "assets/dungeons/padlock/padlock_animated.blend",
        "generator": "tools/export_padlock_runtime.py",
        "units": "metres, true size",
        "phase_2_not_exported": ["Tear_Front_Plate", "FRONT PLATE", "REMOVED PLATE", "PRY BAR"],
        "clips": clips,
        "channels_per_clip": {name: counts["after"] for name, counts in stripped.items()},
        "animated_nodes": {name: counts["nodes"] for name, counts in stripped.items()},
        "geometry": stats,
        "metrics": metrics,
        "files": {},
    }
    for name in ["padlock.gltf", gltf["buffers"][0]["uri"]] + sorted(files.values()):
        file_path = out_dir / name
        manifest["files"][name] = {"md5": md5(file_path), "bytes": file_path.stat().st_size}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {path} ({stats['triangles']} triangles, {stats['nodes']} nodes, {len(clips)} clips)")


def rewrite_clips(out_dir: Path) -> int:
    """Re-applies the clip policy to an export that is already baked.

    Which channels a clip carries is decided in the glTF, not in the atlas or
    the geometry, so changing that policy does not need forty minutes of Cycles
    to be run again.
    """
    path = out_dir / "padlock.gltf"
    gltf = json.loads(path.read_text(encoding="utf-8"))
    buffer = (out_dir / gltf["buffers"][0]["uri"]).read_bytes()
    for name, (before, after) in sorted(restrict_clip_nodes(gltf).items()):
        log(f"  {name}: restricted to its own nodes, {before} -> {after} channels")
    stats = verify(gltf)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["clips"] = {a["name"]: round(clip_duration(gltf, buffer, a), 6)
                         for a in gltf["animations"]}
    manifest["channels_per_clip"] = {a["name"]: len(a["channels"]) for a in gltf["animations"]}
    manifest["animated_nodes"] = {
        a["name"]: sorted({gltf["nodes"][c["target"]["node"]]["name"] for c in a["channels"]})
        for a in gltf["animations"]}
    manifest["geometry"].update(stats)
    path.write_text(json.dumps(gltf, separators=(",", ":")), encoding="utf-8")
    for name in manifest["files"]:
        file_path = out_dir / name
        manifest["files"][name] = {"md5": md5(file_path), "bytes": file_path.stat().st_size}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"rewrote {path} without re-baking")
    return 0


def main() -> int:
    args = parse_args()
    out_dir = args.out.resolve()
    if args.rewrite_clips:
        return rewrite_clips(out_dir)
    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    prune()
    rest_the_rig(armature)
    drop_degenerate()
    drop_bones(armature)
    to_meshes()
    bake_modifiers()
    section()
    check_pins_are_visible()
    stats = {"triangles_capped": decimate(args.triangle_cap)}
    objects = join_per_bone()
    atlas_unwrap(objects, margin_texels=8.0, atlas_size=args.atlas)
    files = {"albedo": "padlock_albedo.jpg", "orm": "padlock_orm.png",
             "normal": "padlock_normal.png"}
    if not args.skip_bake:
        files = bake_atlas(objects, out_dir, args.atlas, args.ao_samples)
    single_material(objects, out_dir, files)
    export(out_dir, armature, files, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
