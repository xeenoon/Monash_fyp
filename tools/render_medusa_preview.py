"""Render the checked-in guardian for art review; run with Blender --background.

blender -b --python tools/render_medusa_preview.py -- /tmp/medusa-preview.png
This is a studio inspection, not a screenshot of the game lighting.
"""
from pathlib import Path
import sys
import bpy
from mathutils import Vector

root = Path(__file__).resolve().parents[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(root / "assets/characters/medusa/medusa.gltf"))
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 32
scene.cycles.use_denoising = True
scene.world = bpy.data.worlds.new("Dark studio")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.08,.10,.13,1)
scene.world.node_tree.nodes["Background"].inputs[1].default_value = .3

def aim(obj, target):
    obj.rotation_euler = (Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()

# glTF Y-up becomes Blender Z-up, glTF +Z faces Blender -Y.
bpy.ops.object.camera_add(location=(1.0,-4.8,2.35))
scene.camera = bpy.context.object
aim(scene.camera,(0,0,1.2))
scene.camera.data.type = 'ORTHO'
scene.camera.data.ortho_scale = 2.8
for name,pos,color,power,size in [
    ('Warm torch key',(-2,-3,4),(1,.73,.46),450,3),
    ('Cold stone fill',(2,-1,2),(.40,.73,1),180,2),
    ('Crown rim',(0,2,3),(.55,1,.72),500,2)]:
    light=bpy.data.lights.new(name,'AREA'); light.energy=power; light.color=color; light.shape='DISK'; light.size=size
    obj=bpy.data.objects.new(name,light); scene.collection.objects.link(obj); obj.location=pos; aim(obj,(0,0,1))
bpy.ops.mesh.primitive_plane_add(size=200)
mat=bpy.data.materials.new('Charcoal stage'); mat.diffuse_color=(.018,.025,.026,1)
bpy.context.object.data.materials.append(mat)
scene.render.resolution_x=1000; scene.render.resolution_y=1100; scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'
args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
scene.render.filepath=args[0] if args else '/tmp/medusa-preview.png'
# Optional seconds into the exported idle, for reviewing the exact game poses.
if len(args)>1:
    scene.frame_set(round(float(args[1])*scene.render.fps))
bpy.ops.render.render(write_still=True)
