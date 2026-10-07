"""World-space placement, exclusion masks, slope rejection and reproducibility."""
from pathlib import Path
import sys
import tempfile
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from terrain_tiles import Tile,Key,HEIGHT_F32,_encode_tile
from scatter_vegetation import placements


def fixture(path,gradient):
    yy,xx=np.mgrid[-1:6,-1:6]
    heights=(100+xx*gradient+yy*.2).astype('<f4')
    minimum=float(heights.min())
    tile=Tile(Key(0,0,0),None,7,7,1,HEIGHT_F32,minimum,float(np.ptp(heights)),0,49,
              (0,0,4,4),(1,0,0,0,1,0,0,0,1,10000,minimum,-20000),0,
              heights.tobytes(),bytes([255]*6+[1]),b'',"test-projected","scatter-test","")
    path.write_bytes(_encode_tile(tile))


with tempfile.TemporaryDirectory() as directory:
    tile=Path(directory)/"test.trn"; fixture(tile,.1)
    a=placements(tile,"grass",12,71)
    assert a==placements(tile,"grass",12,71),"seed must reproduce placement and variants"
    assert a!=placements(tile,"grass",12,72),"different seeds must vary the scatter"
    assert a["placed_count"]==12
    expected=np.array([-.1,1,-.2]);expected/=np.linalg.norm(expected)
    for instance in a["instances"]:
        x,y,z=instance["position"]
        assert 9998<=x<=10002 and -20002<=z<=-19998
        assert abs(y-(100+.1*(x-9998)+.2*(z+20002)-.015))<1e-4,"floating/buried foliage"
        assert np.allclose(instance["terrain_normal"],expected,atol=1e-4)
    mask=Path(directory)/"exclude.png";Image.new('L',(8,8),0).save(mask)
    assert placements(tile,"grass",12,71,mask)["placed_count"]==0,"black mask must exclude plants"
    fixture(tile,100)
    assert placements(tile,"grass",12,71)["placed_count"]==0,"cliffs must reject scatter"
print("Vegetation scatter OK: deterministic, grounded, slope-limited and mask-aware")
