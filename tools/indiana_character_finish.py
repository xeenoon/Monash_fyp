"""Small construction and surface finishing pass, reusable after rebuilding."""
import bpy,math
from mathutils import Vector
from mathutils.bvhtree import BVHTree

def apply(g):
 rig=g['rig'];col=g['character'];arm=rig.data
 arm.name='Indiana • 55 bone deformation skeleton'
 # Fit smooth nail plates to their own digit, avoiding cross-finger ray hits.
 from indiana_character_refine import frame
 for side in ['l','r']:
  hand=g['hands'][side];_,_,_,dorsal=frame(arm,side)
  for digit in ['thumb','index','middle','ring','pinky']:
   no=bpy.data.objects['Nail • '+side+' '+digit]
   b=arm.bones[digit+'_03_'+side];A=(b.tail_local-b.head_local).normalized()
   B=dorsal.cross(A).normalized();D=A.cross(B).normalized()
   if D.dot(dorsal)<0:D=-D
   groups={q.index for q in hand.vertex_groups if q.name.startswith(digit+'_')}
   ids={v.index for v in hand.data.vertices if sum(q.weight for q in v.groups if q.group in groups)>.3}
   faces=[p.vertices[:] for p in hand.data.polygons if all(i in ids for i in p.vertices)]
   tree=BVHTree.FromPolygons([v.co for v in hand.data.vertices],faces)
   length=min(b.length*.61,.014 if digit=='thumb' else .012)
   center=b.tail_local-A*(length*.60+.0025)
   hit=tree.ray_cast(center+D*.035,-D)
   if hit[0] is None:continue
   center=hit[0]+D*.0003
   width={'thumb':.011,'index':.0085,'middle':.009,'ring':.008,'pinky':.0065}[digit]
   for row in range(11):
    t=row/10
    for j in range(9):
     u=2*j/8-1;taper=.79+.21*math.sin(math.pi*t)**.4
     no.data.vertices[row*9+j].co=center+A*((t-.5)*length)+B*(u*width*.5*taper)-D*(.00075*u*u+.00025*(2*t-1)**2)
   # Keep the cuticle centered on the newly smoothed proximal edge.
   cut=bpy.data.objects.get('Cuticle • '+side+' '+digit)
   if cut:bpy.data.objects.remove(cut,do_unlink=True)
   no['plate_refined']=True
 for ob in col.objects:
  if ob.type!='MESH' or ob.get('collar_refined'):continue
  if ob.name.startswith(('Shirt collar -','Shirt collar 1','Jacket pointed collar')):
   # Preserve the neckline attachment while shortening the falling collar tips.
   top=max(v.co.z for v in ob.data.vertices)
   for v in ob.data.vertices:
    d=top-v.co.z;v.co.z+=d*.24;v.co.y+=d*.16
   ob['collar_refined']=True
 for side in ['l','r']:
  name='Shirt cuff • '+side
  if bpy.data.objects.get(name):continue
  b=arm.bones['lowerarm_'+side];w=arm.bones['hand_'+side].head_local
  A=(b.tail_local-b.head_local).normalized()
  B=A.cross(Vector((0,1,0))).normalized();D=A.cross(B).normalized()
  vs=[];fs=[];N=48
  for k,t in enumerate([-.055,-.053,-.011,-.009]):
   for j in range(N):
    a=2*math.pi*j/N;vs.append(w+A*t+B*(.029*math.cos(a))+D*(.022*math.sin(a)))
   if k:
    for j in range(N):fs.append(((k-1)*N+j,(k-1)*N+(j+1)%N,k*N+(j+1)%N,k*N+j))
  me=bpy.data.meshes.new(name);me.from_pydata(vs,[],fs);me.update()
  ob=bpy.data.objects.new(name,me);col.objects.link(ob);me.materials.append(g['shirtmat'])
  uv=me.uv_layers.new(name='UVMap')
  for p in me.polygons:
   for li in p.loop_indices:
    i=me.loops[li].vertex_index;u=i%N/N
    if p.index%N==N-1 and i%N==0:u=1
    uv.data[li].uv=(u,i//N/3)
   p.use_smooth=True
  ob.vertex_groups.new(name='lowerarm_'+side).add(list(range(len(vs))),1,'REPLACE')
  mod=ob.modifiers.new('Cotton thickness','SOLIDIFY');mod.thickness=.001
  mod=ob.modifiers.new('Pose deformation','ARMATURE');mod.object=rig
 # Surface grain is intentionally subtle at full-body distance, visible close up.
 for key,strength in [('skin',1.7),('leather',.65),('strapmat',.55),('wood',.20)]:
  for node in g[key].node_tree.nodes:
   if node.type=='NORMAL_MAP':node.inputs['Strength'].default_value=strength
 g['rig']['construction_revision']='v2: fitted cuffs and shortened collar points'
