"""Check the delivered sculpture's geometry, texture and animation contracts."""
import json
from pathlib import Path
import re
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[1]
folder = root / "assets/characters/medusa"
doc = json.loads((folder / "medusa.gltf").read_text())
buffer = (folder / doc["buffers"][0]["uri"]).read_bytes()

def read(index):
    a = doc["accessors"][index]
    v = doc["bufferViews"][a["bufferView"]]
    width = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4}[a["type"]]
    return np.frombuffer(buffer, dtype="<u4" if a["componentType"] == 5125 else "<f4",
                         count=a["count"]*width, offset=v.get("byteOffset",0)).reshape(-1,width)

limit = int(re.search(r"DUNGEON_GUARDIAN_MAX_DRAWS (\d+)",
                     (root/"src/dungeon_guardian.h").read_text())[1])
primitives = [p for m in doc["meshes"] for p in m["primitives"]]
assert len(primitives) <= limit
vertices = triangles = 0
for p in primitives:
    pos, normal, uv = (read(p["attributes"][a]) for a in ("POSITION","NORMAL","TEXCOORD_0"))
    assert len(pos) == len(normal) == len(uv)
    assert all(np.isfinite(a).all() for a in (pos,normal,uv))
    assert np.allclose(np.linalg.norm(normal,axis=1),1,atol=.001)
    indices = read(p["indices"]).reshape(-1,3)
    assert indices.max() < len(pos)
    a,b,c = (pos[indices[:,i]] for i in range(3))
    face = np.cross(b-a,c-a)
    area = np.linalg.norm(face,axis=1)
    # Pole fans can be tiny, but the visible surface must face its normals.
    visible = area > 1e-9
    orientation = np.sum(face * normal[indices].mean(axis=1),axis=1)
    assert np.mean(orientation[visible] > 0) > .995, "inverted sweep winding"
    vertices += len(pos); triangles += len(indices)
for image in doc["images"]:
    with Image.open(folder / image["uri"]) as im:
        assert im.size == (1024,1024) and im.mode == "RGB"
clip = doc["animations"][0]
assert clip["name"] == "Serpentine" and len(clip["channels"]) == 68
jaw_open=[]
parents={child:i for i,node in enumerate(doc["nodes"]) for child in node.get("children",[])}
for channel in clip["channels"]:
    s = clip["samplers"][channel["sampler"]]
    times, values = read(s["input"]), read(s["output"])
    assert np.all(np.diff(times[:,0]) > 0)
    assert np.allclose(values[0],values[-1],atol=1e-6), "animation loop seam"
    assert np.max(np.abs(values-values[0])) > .01, "stationary crown"
    node=channel["target"]["node"]
    name=doc["nodes"][node]["name"]
    if channel["target"]["path"] == "rotation":
        assert np.allclose(np.linalg.norm(values,axis=1),1,atol=1e-6)
    else:
        assert channel["target"]["path"] == "scale"
        assert (values > 0).all(), "collapsed/inverted strike transform"
        if name.startswith("Flicking tongue"):
            assert doc["nodes"][parents[node]]["name"].startswith("Hissing jaw")
        else:
            assert name.startswith(("Crown serpent", "Head socket"))
    if name.startswith("Hissing jaw"):
        assert doc["nodes"][parents[node]]["name"].startswith("Searching head")
        opened=values[:,0] > .20
        assert .02 < opened.mean() < .15, "hiss must open, pause and recover"
        angle=2*np.arcsin(values[:,0])
        assert angle.max() > 1.25, "jaw must gape wide"
        velocity=np.diff(angle)/np.diff(times[:,0])
        assert velocity.min() < -20, "jaw closure must snap, not ease slowly"
        jaw_open.append(opened)
    if name.startswith("Searching head") and channel["target"]["path"] == "rotation":
        assert doc["nodes"][parents[node]]["name"].startswith("Head socket")
        assert np.ptp(values[:,1]) > .15, "head must look around independently"
assert len(jaw_open) == 11
assert np.max(np.sum(jaw_open,axis=0)) <= 2, "heads hiss in unison"
# Measure head movement in the root's frame: neck extension must really carry
# the head forward, rather than just making its jaw or its face larger.
channels={(c["target"]["node"],c["target"]["path"]):read(clip["samplers"][c["sampler"]]["output"])
          for c in clip["channels"]}
for head,node in enumerate(doc["nodes"]):
    if not node["name"].startswith("Head socket"):
        continue
    root=parents[head]
    extension=channels[root,"scale"]
    assert np.allclose(extension*channels[head,"scale"],1,atol=1e-6), "stretched face"
    forward=extension[:,2]*node["translation"][2]
    assert np.ptp(forward) > .40, "strike has no forward reach"
    assert np.max(np.diff(forward)/np.diff(times[:,0])) > 4, "strike is too slow"
assert doc["materials"][2]["name"] == "Amber serpent eyes"
print(f"Medusa asset OK: {len(primitives)} draws, {vertices:,} vertices, {triangles:,} triangles, 11 independent hissing heads")
