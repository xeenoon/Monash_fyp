"""Review the delivered glTF vegetation with Blender, without editing assets.

blender -b --python tools/render_vegetation.py -- grove /tmp/vegetation-grove.png
Views: grove, ground, bark, or an asset id for an isolated full-height study.
"""
from pathlib import Path
import math
import random
import sys
import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/"assets/vegetation"
args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
view=args[0] if args else 'grove'
output=args[1] if len(args)>1 else str(PACK/"previews"/(view+".png"))
bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene
scene.render.engine='CYCLES'
scene.cycles.samples=24
scene.cycles.use_denoising=True
scene.render.resolution_x=1500
scene.render.resolution_y=1000
scene.render.resolution_percentage=100
scene.world=bpy.data.worlds.new('Daylight')
scene.world.use_nodes=True
nodes=scene.world.node_tree.nodes
env=nodes.new('ShaderNodeTexEnvironment'); env.image=bpy.data.images.load(str(ROOT/'assets/env.hdr'))
scene.world.node_tree.links.new(env.outputs['Color'],nodes['Background'].inputs['Color'])
nodes['Background'].inputs['Strength'].default_value=.45
scene.view_settings.view_transform='AgX'
scene.view_settings.exposure=1.0
rng=random.Random(82419)
prototypes={}

def plant(name,position=(0,0,0),scale=1,yaw=0,lod=0):
    key=(name,lod)
    if key not in prototypes:
        before=set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(PACK/'models'/f'{name}_lod{lod}.gltf'))
        objects=[o for o in bpy.data.objects if o not in before and o.type=='MESH']
        prototypes[key]=objects
        for obj in objects: obj.hide_render=True
    for original in prototypes[key]:
        obj=original.copy(); obj.data=original.data
        scene.collection.objects.link(obj)
        obj.hide_render=False
        # Blender's importer bakes glTF's Y-up conversion into each object.
        obj.location=position
        obj.rotation_euler.rotate_axis('Z',yaw)
        obj.scale*=scale
    return prototypes[key]

if view=='grove':
    for name,pos,scale in [
        ('oak_spreading',(-5,6,0),1),('pine_mature',(5,11,0),1),
        ('spruce_mature',(-1,13,0),1),('pine_windswept',(9,17,0),1),
        ('oak_tall',(-9,17,0),1),('spruce_young',(3,5,0),1),
        ('oak_sapling',(-1,5,0),1),('woodland_shrub',(-3,2,0),1),
        ('fallen_log',(0,2,0),1),('broken_stump',(4,3,0),1),
        ('fern_large',(-1,1,0),1.2),('fern_small',(2,1,0),.8)]:
        plant(name,pos,scale,rng.uniform(-.3,.3))
    for i in range(150):
        x=rng.uniform(-10,11); y=rng.uniform(-3,14)
        name=rng.choice(['grass_meadow_a','grass_meadow_b','grass_short','grass_dry','meadow_flowers'])
        plant(name,(x,y,0),rng.uniform(.8,1.3),rng.uniform(0,math.tau),0 if y<4 else 1)
    camera_pos=(17,-24,10); target=(-1,7,5); lens=48
elif view=='ground':
    for i in range(55):
        name=rng.choice(['grass_meadow_a','grass_meadow_b','grass_short','grass_dry','meadow_flowers'])
        plant(name,(rng.uniform(-1.5,1.5),rng.uniform(-.5,2),0),rng.uniform(.7,1.2),rng.uniform(0,math.tau))
    plant('fern_large',(-.6,1.3,0));plant('woodland_shrub',(1.1,2.5,0));plant('fallen_log',(0,2.3,0))
    camera_pos=(2.6,-3.5,1.65);target=(0,.8,.4);lens=48
elif view=='bark':
    plant('oak_spreading');plant('fern_small',(.7,-.5,0),.8)
    camera_pos=(1.6,-2.7,1.25);target=(0,0,.85);lens=62
else:
    objects=plant(view)
    height=max(max((o.matrix_world@Vector(c)).z for c in o.bound_box) for o in objects)
    # Leave vertical breathing room at the landscape render aspect ratio.
    camera_pos=(height*1.15,-height*2.35,height*1.1);target=(0,0,height*.50);lens=48
    if view=='laurel_giant':
        # Eye-level reference view exposes the low fork and hanging skirt.
        camera_pos=(height*1.15,-height*2.35,2.0)

bpy.ops.mesh.primitive_plane_add(size=200)
floor=bpy.context.object
mat=bpy.data.materials.new('Earth with low meadow cover');mat.use_nodes=True
nodes=mat.node_tree.nodes;links=mat.node_tree.links
bsdf=nodes.get('Principled BSDF');bsdf.inputs['Roughness'].default_value=.94
noise=nodes.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=3.4;noise.inputs['Detail'].default_value=5
ramp=nodes.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].color=(.025,.035,.010,1);ramp.color_ramp.elements[1].color=(.105,.095,.042,1)
links.new(noise.outputs['Fac'],ramp.inputs[0]);links.new(ramp.outputs[0],bsdf.inputs['Base Color'])
bump=nodes.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.25;bump.inputs['Distance'].default_value=.025
links.new(noise.outputs['Fac'],bump.inputs['Height']);links.new(bump.outputs[0],bsdf.inputs['Normal'])
floor.data.materials.append(mat)
bpy.ops.object.light_add(type='SUN',location=(5,-6,12))
sun=bpy.context.object;sun.data.energy=2.2;sun.data.angle=.10
sun.rotation_euler=(math.radians(25),math.radians(-20),math.radians(-35))
bpy.ops.object.camera_add(location=camera_pos)
scene.camera=bpy.context.object
scene.camera.rotation_euler=(Vector(target)-scene.camera.location).to_track_quat('-Z','Y').to_euler()
scene.camera.data.lens=lens
scene.render.image_settings.file_format='PNG';scene.render.filepath=output
Path(output).parent.mkdir(parents=True,exist_ok=True)
bpy.ops.render.render(write_still=True)
