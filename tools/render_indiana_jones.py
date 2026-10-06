"""Render one saved character camera without modifying the asset."""
import bpy,sys
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'assets/characters/indiana_jones'
view=sys.argv[sys.argv.index('--')+1] if '--' in sys.argv else 'full_body'
views={'full_body':('full figure',800,1000),'portrait':('face and shoulder',900,900),'grip':('torch grip',800,800),'left_hand':('left hand',800,800),'arm_chain':('arm chain',900,900),'rear':('rear',800,1000)}
name,w,h=views[view]
bpy.ops.wm.open_mainfile(filepath=str(OUT/'indiana_jones.blend'))
s=bpy.context.scene;s.camera=bpy.data.objects['Camera • '+name]
s.render.resolution_x=w;s.render.resolution_y=h;s.render.resolution_percentage=100
s.cycles.samples=24;s.cycles.adaptive_threshold=.06
s.render.filepath=str(OUT/'previews'/f'{view}.png')
bpy.ops.render.render(write_still=True)
