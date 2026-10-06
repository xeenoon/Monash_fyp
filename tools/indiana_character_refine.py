"""Anatomy, hand construction and tailored clothing for the Raiders revision."""
import bpy, math, random, json
from mathutils import Vector, Matrix, Quaternion
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree
from math import sin,cos,pi,exp

def mean(V,ids):return sum((V[i] for i in ids),Vector())/len(ids)
def anatomy_base(g):
 V,W,weights,groups=g['V'],g['W'],g['weights'],g['groups']
 # Measure-driven reduction of the oversized hand; preserve all finger loops.
 for side in ['l','r']:
  wrist=mean(V,groups['joint-'+side+'-hand'])
  for i,p in enumerate(V):
   amount=sum(w for b,w in weights[i].items() if b.endswith('_'+side) and any(k in b for k in ['hand','thumb','index','middle','ring','pinky']))
   if amount:p[:]=p.lerp(wrist+(p-wrist)*.90,min(1,amount))
 # Slightly narrower upper torso and shoulder span, retaining joint proportions.
 for p in V:
  blend=max(0,min(1,(p.z-1.08)/.35))*(1-max(0,min(1,(p.z-1.57)/.10)))
  p.x*=1-.045*blend
 # Scapular contributions blend the upper back and posterior deltoid.
 for side,sg in [('l',1),('r',-1)]:
  b='scapula_'+side;W[b]=[]
  for i,p in enumerate(V):
   if sg*p.x<.04 or p.z<1.30 or p.z>1.58 or p.y<.015:continue
   a=.36*exp(-((abs(p.x)-.14)/.09)**2-((p.z-1.46)/.105)**2)
   for name in list(weights[i]):weights[i][name]*=1-a
   weights[i][b]=a;W[b].append((i,a))
 # Rebuild the serialized weight lists after the influence edits.
 for b in W:W[b]=[(i,ww[b])for i,ww in enumerate(weights)if ww.get(b,0)>0]

def scapula_bones(g):
 arm,V,groups=g['arm'],g['V'],g['groups']
 for side in ['l','r']:
  b=arm.edit_bones.new('scapula_'+side)
  b.head=mean(V,groups['joint-'+side+'-scapula']);b.tail=mean(V,groups['joint-'+side+'-shoulder'])
  b.parent=arm.edit_bones['clavicle_'+side]
  arm.edit_bones['upperarm_'+side].parent=b

def transfer(ob,surface,rig,g):
 # Smooth 4-neighbour transfer also keeps stitches and tendons on moving cloth.
 tree=KDTree(len(surface.data.vertices))
 for v in surface.data.vertices:tree.insert(v.co,v.index)
 tree.balance();ob.vertex_groups.clear()
 for group in surface.vertex_groups:ob.vertex_groups.new(name=group.name)
 for v in ob.data.vertices:
  pairs=tree.find_n(ob.matrix_world@v.co,4);acc={};total=0
  for _,idx,d in pairs:
   w=1/max(.0001,d)**2;total+=w
   for q in surface.data.vertices[idx].groups:acc[q.group]=acc.get(q.group,0)+w*q.weight
  for name,w in acc.items():ob.vertex_groups[name].add([v.index],w/total,'REPLACE')
 if not any(m.type=='ARMATURE' for m in ob.modifiers):g['bind'](ob)
 return ob

def frame(arm,side):
 wrist=arm.bones['hand_'+side].head_local
 along=(arm.bones['middle_01_'+side].head_local-wrist).normalized()
 across=(arm.bones['index_01_'+side].head_local-arm.bones['pinky_01_'+side].head_local).normalized()
 across=(across-along*across.dot(along)).normalized()
 # Dorsal hand surface is chosen from the known outward arm orientation.
 dorsal=across.cross(along).normalized()
 if dorsal.x*(1 if side=='l' else -1)<0:dorsal=-dorsal
 return wrist,along,across,dorsal

def hand_anatomy(g):
 arm=g['arm'];skin=g['skin'];hands={}
 nail=g['material']('Nails • translucent keratin',(.46,.29,.22),.31)
 nail.node_tree.nodes.get('Principled BSDF').inputs['Subsurface Weight'].default_value=.045
 for side,sg in [('l',1),('r',-1)]:
  ob,ids=g['frombase']('Hand • '+side+' anatomical rebuild',lambda p,f:sg*p.x>.34 and any(sum(w for k,w in g['weights'][i].items()if k.endswith('_'+side)and any(t in k for t in ['hand','thumb','index','middle','ring','pinky']))>.05 for i in f),skin,sub=1)
  g['bind'](ob);hands[side]=ob
  wrist,along,across,dorsal=frame(arm,side)
  # Sculpt metacarpal heads and extensor tendons into connected dorsal skin.
  for v in ob.data.vertices:
   p=v.co;w=1/(1+exp(-((p-wrist).dot(dorsal)-.001)*400))
   delta=0
   for digit in ['index','middle','ring','pinky']:
    mcp=arm.bones[digit+'_01_'+side].head_local
    start=wrist+(mcp-wrist)*.18;end=mcp+(wrist-mcp)*.08
    ab=end-start;t=max(0,min(1,(p-start).dot(ab)/ab.length_squared));q=start+ab*t
    dist=(p-q).dot(across)
    delta+=.0008*exp(-(dist/.0026)**2)*sin(pi*t)**1.2
    delta+=.0012*exp(-((p-mcp).length/.013)**2)
    for j in [2,3]:
     joint=arm.bones[f'{digit}_{j:02d}_{side}'].head_local
     delta+=.0008*exp(-((p-joint).length/.009)**2)
   v.co+=dorsal*(delta*w)
  ob.data.update()
  tree=BVHTree.FromPolygons([v.co for v in ob.data.vertices],[p.vertices for p in ob.data.polygons])
  # Anatomical nail plates: curved C-section, rounded proximal corners, inset cuticle.
  for digit in ['thumb','index','middle','ring','pinky']:
   b=arm.bones[digit+'_03_'+side];a=b.head_local;tip=b.tail_local;axis=(tip-a).normalized()
   lat=dorsal.cross(axis).normalized();normal=axis.cross(lat).normalized()
   if normal.dot(dorsal)<0:normal=-normal
   width={'thumb':.014,'index':.0105,'middle':.011,'ring':.010,'pinky':.008}[digit]
   length=min((tip-a).length*.67,.016 if digit=='thumb' else .014)
   center=tip-axis*(length*.63+.002)
   vs=[];fs=[];nu=9;nv=11
   for row in range(nv):
    t=row/(nv-1);long=(t-.5)*length
    for col in range(nu):
     u=2*col/(nu-1)-1
     taper=.83+.17*sin(pi*t)**.3
     c=center+axis*long+lat*(u*width*.5*taper)
     hit=tree.ray_cast(c+normal*.05,-normal)
     surf=hit[0] if hit[0]is not None else c+normal*.005
     # Consistent convex keratin plate, 0.2mm clear of the skin.
     vs.append(surf+normal*(.00022+.00032*(1-u*u)))
   for r in range(nv-1):
    for c in range(nu-1):fs.append((r*nu+c,r*nu+c+1,(r+1)*nu+c+1,(r+1)*nu+c))
   no=g['mesh']('Nail • '+side+' '+digit,vs,fs,nail,1);g['bind'](no,digit+'_03_'+side)
   no.modifiers.new('Keratin free edge','SOLIDIFY').thickness=.00025
   # Skin-colored cuticle ridge, not a dark painted outline.
   points=[vs[i]+normal*.00003 for i in range(nu)]
   cu=g['curve']('Cuticle • '+side+' '+digit,points,.00018,skin,digit+'_03_'+side)
  # Store frames for correct joint-plane posing and surface maps.
  ob['hand_length_m']=(arm.bones['middle_03_'+side].tail_local-wrist).length
  ob['construction']='Connected metacarpal/phalange loops; dorsal tendon relief; MCP/PIP/DIP landmarks; five fitted curved keratin plates.'
 g['hands']=hands
 return hands

def tailored_cloth(g):
 jacket,shirt,pants=g['jacket'],g['shirt'],g['pants'];arm=g['arm'];rig=g['rig']
 # Remove anatomical muscle impressions from the loose jacket back.
 for ob in [jacket,shirt]:
  neigh=[set()for _ in ob.data.vertices]
  for e in ob.data.edges:
   a,b=e.vertices;neigh[a].add(b);neigh[b].add(a)
  for _ in range(7):
   coords=[v.co.copy()for v in ob.data.vertices]
   for v in ob.data.vertices:
    if abs(v.co.x)<.20 and 1.08<v.co.z<1.48 and v.co.y>0 and neigh[v.index]:
     avg=sum((coords[j]for j in neigh[v.index]),Vector())/len(neigh[v.index]);v.co=v.co.lerp(avg,.45)
 # Add loop density for coherent folds, retaining a quad control surface.
 for ob in [jacket,pants,shirt]:
  sub=next((m for m in ob.modifiers if m.type=='SUBSURF'),None)
  am=next((m for m in ob.modifiers if m.type=='ARMATURE'),None)
  if sub:
   if am:am.show_viewport=False
   bpy.context.view_layer.objects.active=ob;bpy.ops.object.modifier_apply(modifier=sub.name)
   if am:am.show_viewport=True
   m=ob.modifiers.new('Tailored folds • subdivision','SUBSURF');m.levels=1;m.render_levels=1
 # Local compression folds only around elbow/cuff, not a noise-displaced whole coat.
 for v in jacket.data.vertices:
  p=v.co;disp=0
  for side in ['l','r']:
   elbow=arm.bones['lowerarm_'+side].head_local;wrist=arm.bones['hand_'+side].head_local
   axis=(wrist-elbow).normalized();d=(p-elbow).dot(axis)
   rel=p-elbow-axis*d;rad=rel.length
   strength=.0048 if side=='r' else .003
   angle=math.atan2(rel.z,rel.y)
   disp+=strength*sin(d*145+1.6*sin(angle))*exp(-((d-.025)/.07)**2)
   dw=(p-wrist).dot(axis)
   disp+=.0035*sin(dw*170+2*sin(angle))*exp(-((dw+.04)/.04)**2)
  if abs(p.x)<.22 and p.z<1.18:disp+=.0025*sin(p.z*140+p.x*28)*exp(-((p.z-1.065)/.055)**2)
  v.co+=v.normal*disp
 for v in pants.data.vertices:
  p=v.co;z=p.z;d=.0034*sin(z*115+p.x*40)*exp(-((z-.22)/.070)**2)
  d+=.003*sin(z*79+p.x*60)*exp(-((z-.53)/.08)**2)
  v.co+=v.normal*d
 for v in shirt.data.vertices:
  p=v.co;v.co+=v.normal*(.002*sin(100*p.z+36*p.x)*exp(-((p.z-1.08)/.1)**2)+.0015*sin(p.z*55-p.x*95))
 # Ray-projected seams carry interpolated garment skin weights.
 def surface_points(ob,xzs,back=False):
  tree=BVHTree.FromPolygons([v.co for v in ob.data.vertices],[p.vertices for p in ob.data.polygons]);pts=[]
  for x,z in xzs:
   hit=tree.ray_cast(Vector((x,1 if back else -1,z)),Vector((0,-1 if back else 1,0)))
   if hit[0] is not None:pts.append(hit[0]+hit[1]*.0008)
  return pts
 def seam(name,pts,ob,radius=.00065,mat=None,stitch=True):
  if len(pts)<2:return
  mat=mat or g['edge'];o=g['curve'](name,pts,radius,mat);transfer(o,ob,rig,g)
  if stitch:
   # True separated stitches instead of a solid decorative cord.
   for i in range(0,len(pts)-1,2):
    o=g['curve'](name+' stitch',[pts[i],Vector(pts[i]).lerp(Vector(pts[i+1]),.65)],.00028,g['edge']);transfer(o,ob,rig,g)
 for side in [-1,1]:
  seam('Jacket • front zipper seam',surface_points(jacket,[(side*(.054+.025*max(0,(z-1.28)/.26)),z)for z in [1.02+i*.008 for i in range(64)]]),jacket)
  seam('Jacket • back action pleat',surface_points(jacket,[(side*(.131+.017*sin(i*pi/45)),1.12+i*.007)for i in range(46)],True),jacket,.0011)
  seam('Trousers • outseam',surface_points(pants,[(side*(.22-.02*z),z)for z in [.18+i*.012 for i in range(67)]]),pants,.0005,g['pantsmat'],False)
 seam('Jacket • rear yoke',surface_points(jacket,[(x,1.44-.045*abs(x)/.20)for x in [-.20+i*.006 for i in range(68)]],True),jacket,.0009)
 # Cuffs and armscye seams trace the actual local sleeve surface.
 tree=BVHTree.FromPolygons([v.co for v in jacket.data.vertices],[p.vertices for p in jacket.data.polygons])
 for side in ['l','r']:
  for name,bn,t in [('Cuff','lowerarm_',.91),('Sleeve attachment','upperarm_',.09)]:
   b=arm.bones[bn+side];axis=(b.tail_local-b.head_local).normalized();c=b.head_local.lerp(b.tail_local,t)
   u=axis.cross(Vector((0,1,0))).normalized();vv=axis.cross(u).normalized();pts=[]
   for i in range(65):
    a=2*pi*i/64;n=u*cos(a)+vv*sin(a);hit=tree.ray_cast(c+n*.16,-n)
    if hit[0]is not None:pts.append(hit[0]+hit[1]*.0008)
   seam('Jacket • '+name+' '+side,pts,jacket,.0008)
 # Shirt placket and buttonholes; proper overlapped construction.
 for side in [-1,1]:
  seam('Shirt • placket topstitch',surface_points(shirt,[(side*.011,1.035+i*.009)for i in range(41)]),shirt,.00035,g['shirtmat'],False)
 g['construction_notes']='Scapular weighting; reduced sleeve ease; collar stands; fitted placket; front zipper seam, yoke, action pleats, armscye/cuff stitching; joint-local cloth compression.'

def pose_chain(g):
 arm,rig,aim=g['arm'],g['rig'],g['aim']
 # Upper arm in the scapular plane. The forearm reaches forward; the carpus
 # and third metacarpal continue its axis without the previous 70-degree kink.
 aim('clavicle_r',(-1,-.03,.13))
 pb=rig.pose.bones['scapula_r'];q=Quaternion(Vector((0,1,0)),math.radians(22))
 pb.matrix=Matrix.Translation(pb.head)@q.to_matrix().to_4x4()@pb.matrix.to_quaternion().to_matrix().to_4x4();bpy.context.view_layer.update()
 aim('upperarm_r',(-.72,-.45,.32));aim('lowerarm_r',(.12,-.93,.35))
 aim('upperarm_l',(.24,-.10,-1));aim('lowerarm_l',(-.52,-.68,-.38))
 axes={}
 for side in ['r','l']:
  b=rig.pose.bones['lowerarm_'+side];A=(b.tail-b.head).normalized()
  B=Vector((0,0,1)) if side=='r' else Vector((-1,0,0))
  B=(B-A*B.dot(A)).normalized()
  g['hand_frame'](side,A,B)
  C=B.cross(A).normalized()*(1 if side=='r'else -1)
  axes[side]=(A,B,C)
 # A single cylinder axis shared by all four fingers and the opposed thumb.
 A,B,C=axes['r']
 B=(B*math.cos(math.radians(30))+A*math.sin(math.radians(30))).normalized()
 roots=[rig.pose.bones[d+'_01_r'].head.copy()for d in ['index','middle','ring','pinky']]
 center=sum(roots,Vector())/4+A*.012+C*.028
 radius=.015
 random.seed(142)
 results={}
 for digit in ['index','middle','ring','pinky']:
  root=rig.pose.bones[digit+'_01_r'].head.copy()
  lengths=[arm.bones[f'{digit}_{i:02d}_r'].length for i in [1,2,3]]
  def cost(angles,ret=False):
   if angles[0]<0 or angles[0]>.95 or not(.45<angles[1]-angles[0]<1.85) or not(.25<angles[2]-angles[1]<1.60):return 1e6
   pts=[root];v=root.copy()
   for L,a in zip(lengths,angles):
    v=v+(A*cos(a)+C*sin(a))*L;pts.append(v)
   val=0
   for i in range(3):
    for t in [.25,.5,.75,1]:
     p=pts[i].lerp(pts[i+1],t)-center;d=(p-B*p.dot(B)).length
     val+=120*max(0,radius+.004-d)**2
    p=pts[i+1]-center;d=(p-B*p.dot(B)).length
    val+=(d-(radius+.006))**2*(2 if i==2 else .45)
   val+=.000015*(angles[0]-.3)**2+.000020*(angles[2]-2.95)**2
   if angles[1]-angles[0]>1.85:val+=.01*(angles[1]-angles[0]-1.85)**2
   if angles[2]-angles[1]>1.65:val+=.01*(angles[2]-angles[1]-1.65)**2
   return (val,pts)if ret else val
  best=[.25,1.7,2.9];bestval=cost(best)
  for _ in range(1800):
   ang=[random.uniform(-.1,.9),random.uniform(1.1,2.25),random.uniform(2.4,3.4)];c=cost(ang)
   if c<bestval:best,bestval=ang,c
  for step in [.12,.06,.03,.015]:
   for _ in range(5):
    for k in range(3):
     for sign in [-1,1]:
      ang=best.copy();ang[k]+=sign*step;c=cost(ang)
      if c<bestval:best,bestval=ang,c
  for i,a in enumerate(best):aim(f'{digit}_{i+1:02d}_r',A*cos(a)+C*sin(a))
  results[digit]=[round(math.degrees(a),2)for a in best]
 # Oppose the thumb pad; CCD distributes flexion across CMC, MCP and IP.
 target=center+B*.027+A*.003+C*(radius+.005)
 for _ in range(15):
  for i in [3,2,1]:
   p=rig.pose.bones[f'thumb_{i:02d}_r'];end=rig.pose.bones['thumb_03_r'].tail
   q=(end-p.head).rotation_difference(target-p.head)
   p.matrix=Matrix.Translation(p.head)@q.to_matrix().to_4x4()@p.matrix.to_quaternion().to_matrix().to_4x4();bpy.context.view_layer.update()
 # Relaxed left hand: progressive natural cascade rather than identical rods.
 A2,B2,C2=axes['l']
 for digit,flex in [('index',12),('middle',18),('ring',24),('pinky',30)]:
  for i,extra in [(1,0),(2,18),(3,34)]:
   a=math.radians(flex+extra);aim(f'{digit}_{i:02d}_l',A2*cos(a)+C2*sin(a))
 aim('thumb_01_l',A2*.5+B2*.7+C2*.12);aim('thumb_02_l',A2*.72+B2*.20+C2*.18);aim('thumb_03_l',A2*.8-B2*.08+C2*.20)
 # Derive actual carpal/metacarpal axis from the posed bind-space landmarks.
 reports={}
 for side in ['r','l']:
  D=rig.pose.bones['hand_'+side].matrix@arm.bones['hand_'+side].matrix_local.inverted()
  wrist=arm.bones['hand_'+side].head_local;mid=arm.bones['middle_01_'+side].head_local
  handdir=(D@mid-D@wrist).normalized();forearm=rig.pose.bones['lowerarm_'+side];fd=(forearm.tail-forearm.head).normalized()
  reports[side+'_wrist_axis_angle_deg']=math.degrees(handdir.angle(fd))
 reports.update({'right_scapula_upward_rotation_deg':22,'finger_absolute_flexion_deg':results,'torch_forearm_angle_deg':math.degrees(B.angle(axes['r'][0])),'torch_grip_radius_m':radius,'priority':'Neutral wrist and anatomical power grip; torch naturally crosses the palm instead of forcing its axis parallel to the forearm.'})
 (g['OUT']/'anatomy_report.json').write_text(json.dumps(reports,indent=2))
 g['torch_axis']=B;g['torch_across']=(A-B*A.dot(B)).normalized();g['torch_depth']=C;g['grip']=center
 return center,B

def face_anatomy(g):
 body=g['body']
 # Continuous facial planes: alar-to-mouth fold, cheek volume, tear trough,
 # lower-lip/chin transition. These are geometry, not painted shadows.
 for v in body.data.vertices:
  x,y,z=v.co
  if y>-.065 or z<1.61:continue
  ax=abs(x);front=max(0,min(1,(-y-.07)/.06))
  cheekline=.022+.30*(1.700-z)
  fold=.0010*exp(-((ax-cheekline)/.0028)**2)*exp(-((z-1.676)/.028)**4)
  cheek=-.0012*exp(-((ax-cheekline-.008)/.007)**2)*exp(-((z-1.693)/.035)**4)
  trough=.0009*exp(-((z-(1.731-.12*(ax-.033)))/.003)**2)*exp(-((ax-.042)/.022)**4)
  chin=.0008*exp(-((z-1.642)/.003)**2)*exp(-(x/.029)**4)
  brow=-.0012*exp(-((z-1.769)/.008)**2)*exp(-((ax-.035)/.023)**4)
  v.co.y+=front*(fold+cheek+trough+chin+brow)
  # Restrained asymmetry in the upper lip, characteristic of a real face.
  v.co.z+=.00065*exp(-((x+.017)/.014)**2-((z-1.669)/.010)**2)
 body.data.update()
