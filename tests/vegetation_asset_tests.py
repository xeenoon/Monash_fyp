"""Offline checks for the scatter pack and the renderer's opaque glTF contract."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/"assets/vegetation"


def check():
    manifest=json.loads((PACK/"manifest.json").read_text())
    assert manifest["up_axis"]=="Y" and manifest["units"]=="metres"
    assert len(manifest["assets"])==18
    assets={asset["id"]:asset for asset in manifest["assets"]}
    laurel=assets["laurel_giant"]["lods"][0]
    oak=assets["oak_tall"]["lods"][0]
    assert laurel["bounds_max"][1]>oak["bounds_max"][1]*1.4
    assert laurel["foliage_elements"]>oak["foliage_elements"]*2
    assert all(laurel["bounds_max"][axis]-laurel["bounds_min"][axis]>18 for axis in (0,2)), "laurel crown must spread widely in both directions"
    paths=set(); triangles=0
    for asset in manifest["assets"]:
        assert asset["spacing_radius_m"]>0 and 0<asset["max_slope_degrees"]<90
        assert len(asset["lods"])==3
        assert asset["lods"][0]["triangles"]>asset["lods"][1]["triangles"]>asset["lods"][2]["triangles"]
        for lod in asset["lods"]:
            path=PACK/lod["path"];assert path not in paths;paths.add(path)
            doc=json.loads(path.read_text()); binary=(path.parent/doc["buffers"][0]["uri"]).read_bytes()
            assert len(binary)==doc["buffers"][0]["byteLength"]
            def read(index):
                a=doc["accessors"][index]; v=doc["bufferViews"][a["bufferView"]]
                n={"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4}[a["type"]]
                return np.frombuffer(binary,"<u4" if a["componentType"]==5125 else "<f4",
                                     count=a["count"]*n,offset=v["byteOffset"]).reshape(-1,n)
            positions=[]; foliage=[]; count=0
            for primitive in doc["meshes"][0]["primitives"]:
                p,n,uv,t=(read(primitive["attributes"][key]) for key in ("POSITION","NORMAL","TEXCOORD_0","TANGENT"))
                assert len(p)==len(n)==len(uv)==len(t)
                assert all(np.isfinite(v).all() for v in (p,n,uv,t)),path
                assert np.allclose(np.linalg.norm(n,axis=1),1,atol=.002),path
                assert np.allclose(np.linalg.norm(t[:,:3],axis=1),1,atol=.002),path
                assert np.max(abs(np.sum(n*t[:,:3],axis=1)))<.002,path
                indices=read(primitive["indices"]).reshape(-1,3)
                assert indices.max()<len(p)
                faces=p[indices]
                cross=np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0])
                valid=np.linalg.norm(cross,axis=1)>1e-10
                winding=np.sum(cross*n[indices].mean(axis=1),axis=1)
                assert np.mean(winding[valid]>0)>.98,(path,"inside-out geometry")
                count+=len(indices);positions.append(p)
                if asset["id"]=="laurel_giant" and primitive["material"]>0:
                    foliage.append(p)
            positions=np.concatenate(positions)
            if foliage:
                leaves=np.concatenate(foliage)
                radius=np.linalg.norm(leaves[:,[0,2]],axis=1)
                # Reference silhouette: low skirt and a high centre, not an
                # upward fan with its highest foliage at the outer rim.
                assert np.quantile(leaves[:,1],.10)<lod["bounds_max"][1]*.40
                assert leaves[radius<4,1].max()>leaves[radius>8.5,1].max()+1
            assert count==lod["triangles"] and len(positions)==lod["vertices"]
            assert np.allclose(positions.min(axis=0),lod["bounds_min"])
            assert np.allclose(positions.max(axis=0),lod["bounds_max"])
            assert positions[:,1].min()>-.5,(path,"buried pivot")
            for mat in doc["materials"]:
                assert mat.get("alphaMode","OPAQUE")=="OPAQUE"
            for image in doc["images"]:
                assert (path.parent/image["uri"]).is_file(),image
            triangles+=count
    source=json.loads((PACK/"material_sources.json").read_text())
    assert source["license"]=="CC0-1.0"
    for material in source["materials"]:
        for channel,info in material["files"].items():
            path=PACK/info["path"]
            assert hashlib.md5(path.read_bytes()).hexdigest()==info["md5"],path
            with Image.open(path) as im:
                assert min(im.size)>= (2048 if channel=="height" else 4096),path
    print(f"Vegetation assets OK: {len(assets)} assets, {len(paths)} LOD models, {triangles:,} triangles; scanned maps verified")


if __name__=="__main__":
    check()
