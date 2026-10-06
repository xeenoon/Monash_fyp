"""Unlit material atlases and distinct measured PBR surfaces for the character."""
import bpy, numpy as np, math, json
from pathlib import Path
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/characters/indiana_jones';TEX=OUT/'textures';SRC=OUT/'source/pbr'
TEX.mkdir(exist_ok=True)

def image_read(path):
 im=bpy.data.images.load(str(path),check_existing=True)
 if any(t in str(path)for t in ['Normal','Roughness','Metalness','Displacement']):im.colorspace_settings.name='Non-Color'
 w,h=im.size
 # Color sources are decoded by Blender's image buffer to scene-linear values.
 a=np.empty(w*h*4,dtype=np.float32);im.pixels.foreach_get(a)
 return a.reshape(h,w,4)[:,:,:3]

def save(name,data,color=False):
 h,w=data.shape[:2];rgba=np.ones((h,w,4),np.float32);vals=np.clip(data,0,1)
 if color:vals=np.where(vals<=.0031308,vals*12.92,1.055*np.power(vals,1/2.4)-.055)
 rgba[:,:,:3]=vals
 im=bpy.data.images.new(name,width=w,height=h,alpha=False,float_buffer=False)
 im.colorspace_settings.name='Non-Color'
 im.pixels.foreach_set(rgba.ravel());im.filepath_raw=str(TEX/(name+'.png'));im.file_format='PNG'
 im.save();bpy.data.images.remove(im)
 im=bpy.data.images.load(str(TEX/(name+'.png')),check_existing=False)
 im.colorspace_settings.name='sRGB' if color else 'Non-Color'
 return im

def node_material(mat,albedo,normal,orm,repeat=1,normal_strength=1):
 n=mat.node_tree.nodes;l=mat.node_tree.links;n.clear()
 out=n.new('ShaderNodeOutputMaterial');p=n.new('ShaderNodeBsdfPrincipled');l.new(p.outputs['BSDF'],out.inputs['Surface'])
 out.location=(650,50);p.location=(350,50)
 uv=n.new('ShaderNodeTexCoord');uv.location=(-950,0);mapping=n.new('ShaderNodeMapping');mapping.location=(-760,0);mapping.inputs['Scale'].default_value=(repeat,repeat,1);l.new(uv.outputs['UV'],mapping.inputs['Vector'])
 for im,label,y in [(albedo,'Unlit albedo',280),(normal,'Tangent micro-normal',0),(orm,'ORM — no baked lighting',-260)]:
  tx=n.new('ShaderNodeTexImage');tx.image=im;tx.label=label;tx.location=(-530,y);l.new(mapping.outputs['Vector'],tx.inputs['Vector'])
  if im==albedo:l.new(tx.outputs['Color'],p.inputs['Base Color'])
  elif im==normal:
   nm=n.new('ShaderNodeNormalMap');nm.location=(80,0);nm.inputs['Strength'].default_value=normal_strength;l.new(tx.outputs['Color'],nm.inputs['Color']);l.new(nm.outputs['Normal'],p.inputs['Normal'])
  else:
   sep=n.new('ShaderNodeSeparateColor');sep.location=(-140,-260);l.new(tx.outputs['Color'],sep.inputs[0]);l.new(sep.outputs['Green'],p.inputs['Roughness']);l.new(sep.outputs['Blue'],p.inputs['Metallic'])
 mat['lighting_baked']=False;mat['surface_revision']='v2 / distinct microstructure and roughness'
 return p

def skin_atlas(g,N=2048):
 # Rasterize the existing non-overlapping anatomical UV layout, retaining
 # facial and palmar regions rather than assigning noise to arbitrary islands.
 pos=np.zeros((N,N,3),np.float32);metric=np.ones((N,N,2),np.float32);valid=np.zeros((N,N),bool)
 for ob in [g['body'],*g['hands'].values()]:
  me=ob.data;me.calc_loop_triangles();uv=me.uv_layers.active.data
  for tr in me.loop_triangles:
   ids=list(tr.vertices);p=np.array([me.vertices[i].co[:]for i in ids],np.float32);t=np.array([uv[i].uv[:]for i in tr.loops],np.float32)
   low=np.maximum(0,np.floor(t.min(0)*N-.5).astype(int));high=np.minimum(N-1,np.ceil(t.max(0)*N-.5).astype(int))
   if np.any(high<low):continue
   den=(t[1,1]-t[2,1])*(t[0,0]-t[2,0])+(t[2,0]-t[1,0])*(t[0,1]-t[2,1])
   if abs(den)<1e-10:continue
   yy,xx=np.mgrid[low[1]:high[1]+1,low[0]:high[0]+1];u=(xx+.5)/N;v=(yy+.5)/N
   a=((t[1,1]-t[2,1])*(u-t[2,0])+(t[2,0]-t[1,0])*(v-t[2,1]))/den
   b=((t[2,1]-t[0,1])*(u-t[2,0])+(t[0,0]-t[2,0])*(v-t[2,1]))/den;c=1-a-b
   inside=(a>=-.0001)&(b>=-.0001)&(c>=-.0001);iy=yy[inside];ix=xx[inside]
   pos[iy,ix]=a[inside,None]*p[0]+b[inside,None]*p[1]+c[inside,None]*p[2];valid[iy,ix]=True
   du=t[1]-t[0];dv=t[2]-t[0];det=du[0]*dv[1]-du[1]*dv[0]
   if abs(det)>1e-10:
    dpdu=((p[1]-p[0])*dv[1]-(p[2]-p[0])*du[1])/det
    dpdv=(-(p[1]-p[0])*dv[0]+(p[2]-p[0])*du[0])/det
    metric[iy,ix]=[np.linalg.norm(dpdu),np.linalg.norm(dpdv)]
 # Dilate chart margins in position space for seamless filtering.
 for _ in range(12):
  for axis in [0,1]:
   for sh in [-1,1]:
    other=np.roll(valid,sh,axis);fill=~valid&other
    pos[fill]=np.roll(pos,sh,axis)[fill];metric[fill]=np.roll(metric,sh,axis)[fill];valid[fill]=True
 x,y,z=pos[:,:,0],pos[:,:,1],pos[:,:,2]
 rng=np.random.default_rng(91)
 def soft_noise(scale):
  noise=rng.normal(size=(N,N)).astype(np.float32)
  f=np.fft.fftfreq(N);fil=np.exp(-((f[:,None]*scale)**2+(f[None,:]*scale)**2))
  out=np.fft.ifft2(np.fft.fft2(noise)*fil).real.astype(np.float32);return out/(out.std()+1e-8)
 coarse=soft_noise(180);fine=soft_noise(7);pores=np.maximum(0,soft_noise(2.5)-1.0)
 face=(z>1.60)&(np.abs(x)<.13)&(y<-.045)
 mouth=np.exp(-((x/.031)**6+((z-1.666)/.008)**4))*face
 beard=np.exp(-((z-1.646)/.038)**4)*np.clip((-y-.06)/.045,0,1)*face*(1-mouth)
 beard*=np.clip((1.706-z)/.034,0,1)
 col=np.zeros((N,N,3),np.float32);col[:]=(.39,.225,.153)
 col*=1+(coarse*.037+fine*.009)[:,:,None]
 col=col*(1-mouth[:,:,None]*.48)+np.array((.35,.13,.103))*mouth[:,:,None]*.48
 col*=1-beard[:,:,None]*.16
 # Freckles, small pigment variation and follicle color are albedo, never shadows.
 follicle=np.clip(pores-.25,0,1)*beard
 col*=1-.34*follicle[:,:,None]
 browline=1.770+.004*np.sin(np.clip((np.abs(x)-.014)/.046,0,1)*np.pi)-.005*np.clip((np.abs(x)-.014)/.046,0,1)
 brow=np.exp(-((z-browline)/.0022)**4)*np.exp(-((np.abs(x)-.036)/.024)**8)*face
 col*=1-.74*(brow*np.clip(.75+.20*fine,0,1))[:,:,None]
 cheeks=np.exp(-((np.abs(x)-.048)/.025)**2-((z-1.706)/.027)**2)*face
 col[:,:,0]*=1+.055*cheeks;col[:,:,1]*=1-.025*cheeks
 rough=.48+.035*coarse+.055*np.minimum(pores,1)
 tzone=np.exp(-(x/.021)**4)*face;rough-=tzone*.11
 rough+=mouth*.025
 # Pores are shallow indentations. Wrinkles follow the facial anatomy.
 height=-.000035*pores+.000008*fine
 for z0 in [1.786,1.798,1.810]:
  line=z0+.0015*np.sin(x*100)
  height-=.000045*np.exp(-((z-line)/.0005)**2)*np.exp(-(x/.058)**8)*face
 for slope in [-.5,0,.6]:
  line=1.748+slope*(np.abs(x)-.058)
  height-=.000055*np.exp(-((z-line)/.00055)**2)*np.exp(-((np.abs(x)-.067)/.012)**4)*face
 height+=.000030*np.sin(x*5500+np.sin(z*1800))*mouth
 # Bone-local knuckle and palm crease masks, valid for both hands in bind pose.
 from indiana_character_refine import frame
 for side,ob in g['hands'].items():
  wrist,A,B,D=frame(g['arm'],side);mask=(x> .34)if side=='l'else(x<-.34)
  P=pos-np.array(wrist);al=np.einsum('ijk,k->ij',P,np.array(A));ac=np.einsum('ijk,k->ij',P,np.array(B));d=np.einsum('ijk,k->ij',P,np.array(D))
  palm=mask&(d<0);rough+=palm*.065
  # Major palmar flexion lines and thenar arc.
  for ln in [.037+.27*ac,.065-.2*ac,.034+4.2*(ac+.016)**2]:
   height-=.00007*np.exp(-((al-ln)/.0007)**2)*palm*np.exp(-(ac/.040)**6)
  for digit in ['index','middle','ring','pinky','thumb']:
   for j in [1,2,3]:
    b=g['arm'].bones[f'{digit}_{j:02d}_{side}'];axis=(b.tail_local-b.head_local).normalized();q=pos-np.array(b.head_local)
    t=np.einsum('ijk,k->ij',q,np.array(axis));dist=np.sum(q*q,axis=2)-t*t
    region=np.exp(-dist/.00012)*mask
    for offset in [-.002,0,.002]:height-=.000055*np.exp(-((t-offset)/.00050)**2)*region
  redness=np.clip((fine-.4)*.03,0,.05)*mask;col[:,:,0]*=1+redness
 dy,dx=np.gradient(height)
 # Convert height gradients to a tangent-space normal using UV chart metrics.
 nx=-dx*N/np.maximum(metric[:,:,0],.05);ny=-dy*N/np.maximum(metric[:,:,1],.05)
 normal=np.stack([nx,ny,np.ones_like(nx)],axis=-1);normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
 normal=normal*.5+.5
 orm=np.stack([np.ones_like(rough),np.clip(rough,.29,.68),np.zeros_like(rough)],axis=-1)
 a=save('skin_anatomical_albedo',col,True);n=save('skin_anatomical_normal',normal);r=save('skin_anatomical_orm',orm)
 p=node_material(g['skin'],a,n,r)
 p.inputs['Subsurface Weight'].default_value=.075;p.inputs['Subsurface Radius'].default_value=(1,.46,.24);p.inputs['Subsurface Scale'].default_value=.0018;p.inputs['IOR'].default_value=1.4
 return {'resolution':N,'valid_chart_texels':int(valid.sum()),'skin_maps':['skin_anatomical_albedo.png','skin_anatomical_normal.png','skin_anatomical_orm.png']}

def surfaces(g):
 # Each surface retains its own measured normal and roughness characteristics.
 specs=[
  ('leather','Leather026_2K-PNG',(.032,.013,.007),.51,.075,0,1.0,[g['leather']]),
  ('strap_leather','Leather026_2K-PNG',(.061,.027,.011),.45,.08,0,1.0,[g['strapmat']]),
  ('cotton','Fabric036_2K-PNG',(.36,.32,.249),.83,.045,0,1.0,[g['shirtmat']]),
  ('canvas','Fabric036_2K-PNG',(.17,.171,.104),.89,.05,0,1.0,[g['canvas']]),
  ('twill','Fabric036_2K-PNG',(.18,.14,.098),.84,.05,0,1.0,[g['pantsmat']]),
  ('wood','Wood066_2K-PNG',(.15,.071,.025),.74,.16,0,1.0,[g['wood']]),
  ('brass','Metal030_1K-PNG',(.34,.24,.10),.36,.09,.92,1.0,[g['metal']]),
  ('steel','Metal030_1K-PNG',(.40,.41,.39),.29,.08,.97,1.0,[g['steel']]),
 ]
 report=[]
 for name,src,tint,meanr,variation,metal,repeat,mats in specs:
  col=image_read(SRC/(src+'_Color.png'));r=image_read(SRC/(src+'_Roughness.png'))[:,:,0]
  # Normalize only pigmentation; retain fine albedo variation from the source.
  lum=np.mean(col,axis=2);pig=np.clip(lum/(np.median(lum)+1e-6),.45,1.65)
  pig=1+(pig-1)*(.70 if name=='wood'else .45)
  color=np.array(tint,np.float32)*pig[:,:,None]
  rough=np.clip(meanr+(r-r.mean())/(r.std()+1e-6)*variation,.16,.97)
  orm=np.stack([np.ones_like(r),rough,np.full_like(r,metal)],axis=-1)
  albedo=save(name+'_albedo',color,True);orr=save(name+'_orm',orm)
  norm=bpy.data.images.load(str(SRC/(src+'_NormalGL.png')),check_existing=True);norm.colorspace_settings.name='Non-Color'
  # Store a compact 8-bit runtime normal while retaining the source data.
  nn=save(name+'_normal',image_read(SRC/(src+'_NormalGL.png')))
  for mat in mats:
   p=node_material(mat,albedo,nn,orr,repeat,.30 if name in ['leather','strap_leather']else .25)
   if name in ['cotton','canvas','twill']:p.inputs['Sheen Weight'].default_value=.18;p.inputs['Sheen Roughness'].default_value=.7
  report.append({'surface':name,'source':src.split('_')[0],'repeat':repeat,'roughness_mean':float(rough.mean()),'roughness_stddev':float(rough.std()),'metallic':metal})
 return report

def apply(g):
 print('Building anatomical skin atlas...',flush=True);skin=skin_atlas(g)
 print('Building distinct PBR materials...',flush=True);report=surfaces(g)
 # Felt fibers remain softer than the measured leather and cotton.
 felt=g['felt'];p=felt.node_tree.nodes.get('Principled BSDF');p.inputs['Roughness'].default_value=.94;p.inputs['Sheen Weight'].default_value=.28
 # A small neutral reflection source makes material response inspectable.
 g['scene'].cycles.samples=24
 calibrate_uv(g)
 studio_response(g)
 for im in list(bpy.data.images):
  if im.users==0:bpy.data.images.remove(im)
 (OUT/'material_report.json').write_text(json.dumps({'skin':skin,'surfaces':report,'baked_lighting':False,'occlusion_channels':'uniform white','sources':['https://ambientcg.com/a/Leather026','https://ambientcg.com/a/Fabric036','https://ambientcg.com/a/Wood066','https://ambientcg.com/a/Metal030']},indent=2))
 g['rig']['material_revision']='v2: anatomical skin atlas + distinct PBR surface response'
 bpy.ops.file.pack_all()

def calibrate_uv(g):
 sizes={g['leather'].name:.20,g['strapmat'].name:.20,g['shirtmat'].name:.12,g['canvas'].name:.18,g['pantsmat'].name:.15,g['metal'].name:.08,g['steel'].name:.08}
 for ob in g['character'].objects:
  if ob.type!='MESH' or not ob.data.materials:continue
  size=sizes.get(ob.data.materials[0].name)
  if size is None or not ob.data.uv_layers:continue
  me=ob.data;orig=me.uv_layers.get('UVMap') or me.uv_layers[0]
  me.calc_loop_triangles();area=uva=0
  for tr in me.loop_triangles:
   a,b,c=[me.vertices[i].co for i in tr.vertices];area+=(b-a).cross(c-a).length*.5
   a,b,c=[orig.data[i].uv for i in tr.loops];uva+=abs((b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x))*.5
  if uva<1e-10:continue
  scale=math.sqrt(area/uva)/size
  layer=me.uv_layers.get('SurfaceUV') or me.uv_layers.new(name='SurfaceUV')
  for i,uv in enumerate(orig.data):layer.data[i].uv=uv.uv*scale
  me.uv_layers.active=layer;layer.active_render=True;ob['surface_tile_metres']=size

def studio_response(g):
 scene=g['scene'];scene.cycles.samples=24;scene.cycles.use_adaptive_sampling=True;scene.cycles.adaptive_threshold=.05
 scene.cycles.max_bounces=5;scene.cycles.diffuse_bounces=2;scene.cycles.glossy_bounces=3
 scene.world.node_tree.nodes.get('Background').inputs[0].default_value=(.06,.065,.07,1)
 for name,power,size in [('Studio key • neutral',550,1.7),('Studio fill • neutral',190,3.0),('Studio rim • neutral',450,2.4)]:
  o=bpy.data.objects.get(name)
  if o:o.data.energy=power;o.data.size=size
 # Eye surface has a clear, localized response distinct from skin.
 white=g.get('white') or bpy.data.materials.get('Eyes • sclera')
 if white:
  p=white.node_tree.nodes.get('Principled BSDF');p.inputs['Roughness'].default_value=.16;p.inputs['Coat Weight'].default_value=.28;p.inputs['Coat Roughness'].default_value=.10
