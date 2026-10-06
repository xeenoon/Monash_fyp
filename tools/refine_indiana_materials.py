"""Recalibrate the saved anatomy revision without rebuilding the rig or pose."""
import bpy,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import indiana_character_materials as m
OUT=ROOT/'assets/characters/indiana_jones'
bpy.ops.wm.open_mainfile(filepath=str(OUT/'indiana_jones.blend'))
names={'skin':'Skin • neutral warm beige','leather':'Jacket • dark brown lambskin','strapmat':'Belt and strap • brown hide','shirtmat':'Shirt • stone cotton','canvas':'Satchel • faded olive canvas','pantsmat':'Trousers • taupe wool twill','wood':'Torch • unlit wood','metal':'Hardware • aged brass','steel':'Hardware • dull nickel','felt':'Fedora • sable rabbit felt','white':'Eyes • sclera'}
g={k:bpy.data.materials[v]for k,v in names.items()};g.update({'scene':bpy.context.scene,'character':bpy.data.collections['INDIANA • character'],'body':bpy.data.objects['Body • anatomical quad mesh'],'hands':{s:bpy.data.objects['Hand • '+s+' anatomical rebuild']for s in ['l','r']},'arm':bpy.data.objects['Indiana_Rig'].data,'rig':bpy.data.objects['Indiana_Rig']})
# Regenerate compact runtime skin maps; geometry and anatomical UVs are unchanged.
for mat in [g['skin'],g['leather'],g['strapmat'],g['shirtmat'],g['canvas'],g['pantsmat'],g['wood'],g['metal'],g['steel']]:mat.node_tree.nodes.clear()
for im in list(bpy.data.images):
 if im.users==0:bpy.data.images.remove(im)
m.apply(g)
import indiana_character_finish as finish
finish.apply(g)
m.calibrate_uv(g)
bpy.context.scene.camera=bpy.data.objects['Camera • full figure']
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'indiana_jones.blend'))
print('MATERIAL CALIBRATION COMPLETE')
