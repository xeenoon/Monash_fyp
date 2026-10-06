"""Build the seated monk asset and review renders with Blender.

blender --background --threads 6 --python tools/build_praying_monk.py
Native geometry, Z up / -Y forward; glTF exports Y up / +Z forward.
"""
import json
import math
import random
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/characters/praying_monk'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'previews').mkdir(exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
random.seed(17)
character = bpy.data.collections.new('MONK | seated prayer')
bpy.context.scene.collection.children.link(character)


def material(name, color, roughness):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Roughness'].default_value = roughness
    return mat


cloth = material('Cloak | weathered umber wool', (.115, .079, .052), .94)
edge = material('Hems | worn warm brown', (.19, .136, .088), .96)
inner = material('Hood lining | deep charcoal', (.003, .0025, .002), 1)
skin = material('Hands | aged warm skin', (.39, .265, .18), .78)
nail = material('Nails | muted horn', (.42, .33, .25), .75)
hair = [material('Beard | silver %02d' % i,
                 tuple(v * (.72 + i*.055) for v in (.58, .55, .47)), .94)
        for i in range(7)]


def mesh(name, verts, faces, mat):
    data = bpy.data.meshes.new(name)
    data.from_pydata(verts, [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    character.objects.link(obj)
    obj.data.materials.append(mat)
    for poly in data.polygons:
        poly.use_smooth = True
    return obj


def surface(name, rows, mat, close=True):
    width = len(rows[0])
    faces = []
    for i in range(len(rows)-1):
        for j in range(width if close else width-1):
            k = (j+1) % width
            faces.append((i*width+j, i*width+k, (i+1)*width+k, (i+1)*width+j))
    obj = mesh(name, [p for row in rows for p in row], faces, mat)
    # Recalculate consistent outward normals on swept surfaces.
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(False)
    return obj


def tube(name, points, radii, mat, sides=12, steps=6, flatten=1, folds=0):
    points = [Vector(p) for p in points]
    padded = [2*points[0]-points[1], *points, 2*points[-1]-points[-2]]
    centers, rr = [], []
    for i in range(len(points)-1):
        a,b,c,d = padded[i:i+4]
        for j in range(steps):
            t = j/steps
            centers.append(.5*((2*b)+(-a+c)*t+(2*a-5*b+4*c-d)*t*t+(-a+3*b-3*c+d)*t*t*t))
            rr.append(radii[i]*(1-t)+radii[i+1]*t)
    centers.append(points[-1]); rr.append(radii[-1])
    rows=[]
    across=None
    for i, (p,r) in enumerate(zip(centers,rr)):
        direction=(centers[min(i+1,len(centers)-1)]-centers[max(i-1,0)]).normalized()
        if across is None:
            across=direction.cross(Vector((0,1,0)))
            if across.length < .01: across=direction.cross(Vector((1,0,0)))
        across=(across-direction*across.dot(direction)).normalized()
        other=direction.cross(across).normalized()
        row=[]
        for j in range(sides):
            a=math.tau*j/sides
            rj=r*(1+folds*math.cos(7*a+i*.10)+folds*.35*math.sin(11*a-i*.18))
            row.append(p+rj*(math.cos(a)*across+math.sin(a)*other*flatten))
        rows.append(row)
    obj=surface(name,rows,mat)
    # Close tube ends explicitly.
    data=obj.data
    verts=[v.co[:] for v in data.vertices]
    faces=[tuple(p.vertices) for p in data.polygons]
    faces.extend([tuple(reversed(range(sides))), tuple(range(len(verts)-sides,len(verts)))])
    replacement=bpy.data.meshes.new(name+' closed')
    replacement.from_pydata(verts,[],faces); replacement.materials.append(mat)
    obj.data=replacement
    bpy.data.meshes.remove(data)
    for p in replacement.polygons:p.use_smooth=True
    return obj


def oval(name, center, scale, mat, segments=32, rings=20):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, location=center)
    obj=bpy.context.object; obj.name=name; obj.scale=scale
    for col in list(obj.users_collection):col.objects.unlink(obj)
    character.objects.link(obj)
    obj.data.materials.append(mat)
    for p in obj.data.polygons:p.use_smooth=True
    obj.select_set(False)
    return obj


# Folded legs are actual crossed volumes, visible beneath the open front cloak.
for s in (-1,1):
    crossing_lift=.065 if s==-1 else 0
    tube('Folded leg %s'%s, [(s*.18,.06,.27),(s*.43,-.015,.19),(s*.37,-.27,.12),
                           (-s*.12,-.40,.115+crossing_lift)], [.18,.19,.145,.09], cloth, 36,10, .82,.065)
    oval('Bare foot %s'%s,(-s*.24,-.405,.105+crossing_lift),(.16,.072,.072),skin)
    for i in range(5):
        oval('Toe %s %d'%(s,i),(-s*(.29+i*.024),-.457+i*.006,.10+crossing_lift),
             (.033,.041-i*.003,.031-i*.002),skin,16,10)

# Continuous draped body; uneven broad folds retain a readable silhouette.
rows=[]
for i in range(45):
    t=i/44
    z=.045+.91*t
    rx=.55*(1-t)+.23*t+.05*math.sin(math.pi*t)
    ry=.34*(1-t)+.17*t
    row=[]
    for j in range(112):
        a=math.tau*j/112
        fold=(.020*math.cos(12*a+.65*t)+.008*math.cos(23*a-2*t))*(1-.45*t)
        # Lift the front hem to show the crossed shins.
        lift=.145*max(0,-math.sin(a))**8*(1-t)**5
        row.append(((rx+fold)*math.cos(a),.055+(ry+fold)*math.sin(a),
                    z+lift+.009*math.cos(12*a)*(1-t)**6))
    rows.append(row)
body=surface('Cloak | continuous gathered drape',rows,cloth)
tube('Cloak | rolled lower hem',rows[0]+[rows[0][0]],[.009]*113,edge,8,2)

# Shoulder mantle above the sleeves.
oval('Cloak | covered shoulder foundation',(0,.055,.805),(.32,.235,.205),cloth)
rows=[]
for i in range(22):
    t=i/21
    row=[]
    for j in range(96):
        a=math.tau*j/96
        r=.13+.285*math.sin(t*math.pi/2)
        row.append((r*math.cos(a),.035+r*.71*math.sin(a),
                    1.045-.31*t+.004*math.sin(a*11)*t))
    rows.append(row)
mantle=surface('Cowl | shoulder mantle',rows,cloth)
mantle.modifiers.new('Wool thickness','SOLIDIFY').thickness=.006
tube('Cowl | stitched edge',rows[-1]+[rows[-1][0]],[.007]*97,edge,8,2)

# Hood shell: open oval mouth, deep interior, closed rounded back.
rows=[]
for i in range(36):
    t=i/35
    shrink=max(.003,math.cos(t*math.pi/2))
    row=[]
    for j in range(96):
        a=math.tau*j/96
        peak=max(0,math.cos(a))
        row.append((.25*math.sin(a)*shrink*(1+.035*math.sin(7*a)),
                    -.205+.49*math.sin(t*math.pi/2)-.12*peak*(1-t)**3,
                    1.115+.31*math.cos(a)*shrink+.018*math.cos(5*a)*shrink))
    rows.append(row)
hood=surface('Hood | deep overhanging cloth',rows,cloth)
solid=hood.modifiers.new('Thick wool edge','SOLIDIFY');solid.thickness=.009
tube('Hood | heavy rolled opening',rows[0]+[rows[0][0]],[.012]*97,edge,10,2)
# Opaque dark lining prevents a face being exposed from any viewing direction.
lining=[[(x*.955,-.20+(y+.20)*.955,1.115+(z-1.115)*.955) for x,y,z in row] for row in rows]
surface('Hood | opaque inner lining',lining,inner)
oval('Hood | recessed shadow',(0,.075,1.115),(.165,.025,.205),inner)
for s in (-1,1):
    tube('Hood | falling side fold %s'%s,[(s*.22,-.12,1.19),(s*.205,-.15,1.02),
        (s*.18,-.20,.86),(s*.25,-.17,.78)],[.005,.034,.036,.002],cloth,16,8,.60)

# Sleeves sweep down to the elbows and back up to joined prayer hands.
for s in (-1,1):
    tube('Sleeve | bent praying arm %s'%s,[(s*.14,.035,.875),(s*.30,-.07,.70),
        (s*.29,-.18,.52),(s*.17,-.33,.65),(s*.07,-.44,.785)],
        [.10,.125,.105,.105,.068],cloth,40,10,1,.04)
    cuff=(s*.07,-.44,.785)
    direction=Vector((-s*.1,-.11,.135)).normalized()
    a=direction.cross(Vector((0,1,0))).normalized();b=direction.cross(a)
    ring=[Vector(cuff)+.069*(math.cos(j*math.tau/48)*a+math.sin(j*math.tau/48)*b)
          for j in range(49)]
    tube('Sleeve | cuff binding %s'%s,ring,[.009]*49,edge,8,2)
    tube('Wrist %s'%s,[(s*.068,-.437,.774),(s*.043,-.464,.837)], [.039,.034],skin,20,6)
    oval('Prayer palm %s'%s,(s*.024,-.465,.865),(.022,.056,.064),skin)
    for f in range(4):
        yy=-.503+f*.026
        height=[.097,.113,.106,.083][f]
        tube('Joined finger %s %d'%(s,f),[(s*.026,yy,.888),(s*.022,yy-.004,.93),
             (s*.014,yy-.003,.888+height)], [.014,.012,.0095],skin,12,5)
        oval('Fingernail %s %d'%(s,f),(s*.023,yy-.004,.876+height),
             (.004,.007,.012),nail,12,8)
    tube('Prayer thumb %s'%s,[(s*.044,-.493,.837),(s*.047,-.529,.876),
         (s*.014,-.535,.917)],[.018,.016,.011],skin,16,7)

# Overlapping tapered locks form the whole beard, with roots inside the hood.
for i in range(43):
    u=(i/42)*2-1
    phase=random.uniform(0,math.tau)
    length=random.uniform(.85,1)
    points=[];radii=[]
    for j in range(12):
        t=j/11*length
        x=u*.145*(.12+1.25*math.sin(math.pi*t)**.5)*(1-t)**.68+.014*math.sin(phase+8*t)*math.sin(math.pi*t)+.07*t**5
        y=.12-.77*math.sqrt(t)-.035*(1-u*u)*math.sin(math.pi*t)
        z=1.21-1.23*t
        points.append((x,y,z))
        radii.append((.017+.009*(1-abs(u)))*(.38+.9*math.sin(math.pi*t))*(1-j/11)**.65+.0007)
    tube('Beard | flowing lock %02d'%i,points,radii,hair[i%7],9,4,.62)
    for k in range(2):
        strand=[(x+(k-.5)*.009,y-.007,z) for x,y,z in points]
        tube('Beard | strand %02d %d'%(i,k),strand,
             [.0014*(1-j/11)+.00025 for j in range(12)],hair[(i+k+2)%7],5,3)

# Neutral material detail, shared with existing project cloth maps.
for mat in (cloth,edge):
    nodes=mat.node_tree.nodes;links=mat.node_tree.links
    tex=nodes.new('ShaderNodeTexImage')
    tex.image=bpy.data.images.load(str(ROOT/'textures/dungeon/runtime/cloth_normal.png'),check_existing=True)
    tex.image.colorspace_settings.name='Non-Color'
    normal=nodes.new('ShaderNodeNormalMap');normal.inputs['Strength'].default_value=.24
    links.new(tex.outputs['Color'],normal.inputs['Color'])
    links.new(normal.outputs['Normal'],nodes.get('Principled BSDF').inputs['Normal'])

# UVs and applied transforms make the static asset portable.
bpy.ops.object.select_all(action='DESELECT')
for obj in character.objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active=body
bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(angle_limit=1.15,island_margin=.01)
bpy.ops.object.mode_set(mode='OBJECT')
for image in bpy.data.images:
    if image.source=='FILE':image.pack()

# Ground the folded robe and preserve separate editable source pieces.
bottom=min((obj.matrix_world@v.co).z for obj in character.objects for v in obj.data.vertices)
for obj in character.objects:
    obj.location.z-=bottom
bpy.context.view_layer.update()
source_objects=list(character.objects)
# A single runtime mesh batches the many sculpted locks by their 12 materials.
bpy.ops.object.duplicate()
runtime_objects=list(bpy.context.selected_objects)
for obj in runtime_objects:
    bpy.context.view_layer.objects.active=obj
    for modifier in list(obj.modifiers):
        bpy.ops.object.modifier_apply(modifier=modifier.name)
bpy.context.view_layer.objects.active=bpy.context.selected_objects[0]
bpy.ops.object.join()
runtime=bpy.context.object;runtime.name='Praying monk | static runtime'
bpy.ops.export_scene.gltf(filepath=str(OUT/'praying_monk.glb'),export_format='GLB',
    use_selection=True,export_apply=True,export_animations=False,export_yup=True)
bpy.ops.export_scene.gltf(filepath=str(OUT/'praying_monk.gltf'),export_format='GLTF_SEPARATE',
    use_selection=True,export_apply=True,export_animations=False,export_yup=True)
bpy.data.objects.remove(runtime,do_unlink=True)
verts=[obj.matrix_world@v.co for obj in character.objects for v in obj.data.vertices]
report={'pose':'Seated cross-legged, palms joined in prayer; static',
        'face':'No facial geometry; opaque recessed hood lining',
        'bounds_blender':{'min':[min(v[i] for v in verts) for i in range(3)],
                          'max':[max(v[i] for v in verts) for i in range(3)]},
        'vertices':len(verts),'triangles':sum(len(p.vertices)-2 for o in character.objects for p in o.data.polygons),
        'mesh_objects':len(character.objects),'materials':len(bpy.data.materials),
        'runtime_axes':'Y up, +Z forward','animation':False}
(OUT/'asset_report.json').write_text(json.dumps(report,indent=2)+'\n')

# Studio is saved in its own collection; all preview images show the actual mesh.
bpy.ops.object.select_all(action='DESELECT')
scene=bpy.context.scene
scene.render.engine='CYCLES';scene.cycles.samples=40;scene.cycles.use_denoising=True
scene.world=bpy.data.worlds.new('Neutral dark studio');scene.world.use_nodes=True
scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.12,.14,.17,1)
scene.world.node_tree.nodes['Background'].inputs[1].default_value=.35
scene.render.resolution_x=1000;scene.render.resolution_y=1000;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'


def aim(obj,target):
    obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()


bpy.ops.mesh.primitive_plane_add(size=200)
floor=bpy.context.object;floor.name='Studio | ground'
floor.data.materials.append(material('Studio | charcoal',(.025,.029,.033),.94))
for name,pos,col,power,size in [('Key',(-2,-3,4),(1,.83,.66),350,3),
                              ('Fill',(2,-1,2),(.65,.79,1),110,2),
                              ('Rim',(0,2,3),(1,.91,.76),300,2)]:
    data=bpy.data.lights.new('Studio | '+name,'AREA');data.energy=power;data.color=col;data.shape='DISK';data.size=size
    obj=bpy.data.objects.new(data.name,data);scene.collection.objects.link(obj);obj.location=pos;aim(obj,(0,0,.7))
bpy.ops.object.camera_add(location=(1.9,-4,2.1))
camera=bpy.context.object;camera.name='Studio | camera';scene.camera=camera
camera.data.type='ORTHO';camera.data.ortho_scale=1.95
aim(camera,(0,-.08,.72))
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'praying_monk.blend'))
for name,pos in [('three_quarter',(1.9,-4,2.1)),('front',(0,-4,1.6)),('game_angle',(2,-3,4)),('rear',(1.8,4,2.2))]:
    camera.location=pos;aim(camera,(0,-.08,.72))
    scene.render.filepath=str(OUT/'previews'/f'{name}.png')
    bpy.ops.render.render(write_still=True)
print('MONK COMPLETE',json.dumps(report))
