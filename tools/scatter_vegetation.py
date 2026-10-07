#!/usr/bin/env python3
"""Create deterministic scatter placements on one existing terrain .trn tile.

Outputs double-precision world positions, terrain normals, per-asset rotations,
scale and LOD paths. This is an offline placement file, not a renderer toggle.
Trees remain vertical; ground plants may align to the sampled triangle normal.
"""
import argparse
import json
import math
from pathlib import Path
import random
import numpy as np
from PIL import Image
from terrain_tiles import read_tile

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/"assets/vegetation"


def placements(tile_path,category,count,seed,mask_path=None):
    tile=read_tile(Path(tile_path))
    g=tile.gutter
    heights=np.asarray(tile.decoded_heights()).reshape(tile.height,tile.width)[g:-g,g:-g]
    validity=np.array([tile.is_valid(i) for i in range(tile.width*tile.height)]).reshape(tile.height,tile.width)[g:-g,g:-g]
    h,w=heights.shape
    sx=tile.extent[2]-tile.extent[0];sz=tile.extent[3]-tile.extent[1]
    dx=sx/(w-1);dz=sz/(h-1)
    rotation=np.asarray(tile.transform[:9]).reshape(3,3).T
    translation=np.asarray(tile.transform[9:])
    if not np.allclose(rotation,np.eye(3),atol=1e-8):
        raise ValueError("Scatter currently supports projected Y-up terrain tiles only")
    catalog=json.loads((PACK/"manifest.json").read_text())
    allowed=[a for a in catalog["assets"] if a["category"] in category.split(",")]
    if not allowed:raise ValueError("No matching asset categories")
    mask=np.asarray(Image.open(mask_path).convert('L')) if mask_path else None
    rng=random.Random(seed);result=[]
    # Spatial buckets bound overlap checks even for a dense grass scatter.
    cell=max(a["spacing_radius_m"]*a["scale_range"][1]*2 for a in allowed)
    buckets={}
    for attempt in range(max(1000,count*80)):
        if len(result)>=count:break
        u,v=rng.random(),rng.random()
        if mask is not None and rng.random()>mask[min(int(v*mask.shape[0]),mask.shape[0]-1),min(int(u*mask.shape[1]),mask.shape[1]-1)]/255:
            continue
        x,z=u*sx-sx/2,v*sz-sz/2
        ix,iz=min(int(u*(w-1)),w-2),min(int(v*(h-1)),h-2)
        if not validity[iz:iz+2,ix:ix+2].all():continue
        a,b,c,d=heights[iz,ix],heights[iz,ix+1],heights[iz+1,ix],heights[iz+1,ix+1]
        fx,fz=u*(w-1)-ix,v*(h-1)-iz
        # Match terrain_tile.c's v00-v11 diagonal, including height and slope.
        if fx>=fz:
            height=a+(b-a)*fx+(d-b)*fz;grad_x=(b-a)/dx;grad_z=(d-b)/dz
        else:
            height=a+(d-c)*fx+(c-a)*fz;grad_x=(d-c)/dx;grad_z=(c-a)/dz
        normal=np.array([-grad_x,1.,-grad_z]);normal/=np.linalg.norm(normal)
        slope=math.degrees(math.acos(float(normal[1])))
        asset=rng.choice(allowed)
        if slope>asset["max_slope_degrees"]:continue
        scale=rng.uniform(*asset["scale_range"]);radius=asset["spacing_radius_m"]*scale
        key=(math.floor(x/cell),math.floor(z/cell))
        neighbours=(p for ox in (-1,0,1) for oz in (-1,0,1) for p in buckets.get((key[0]+ox,key[1]+oz),[]))
        if any((x-p[0])**2+(z-p[1])**2<(radius+p[2])**2 for p in neighbours):continue
        buckets.setdefault(key,[]).append((x,z,radius))
        world=translation+np.array([x,height-tile.minimum,z])
        world[1]-=asset["ground_sink_m"]
        result.append({"asset":asset["id"],"position":world.tolist(),"yaw_radians":rng.uniform(0,math.tau),
                       "scale":scale,"terrain_normal":normal.tolist(),"align_to_slope":asset["align_to_slope"]})
    return {"version":1,"seed":seed,"terrain_tile":str(Path(tile_path).resolve()),
            "asset_manifest":str(PACK/"manifest.json"),"coordinate_space":"world XYZ metres, Y up",
            "requested_count":count,"placed_count":len(result),"instances":result}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile",required=True,type=Path)
    parser.add_argument("--category",default="tree,sapling")
    parser.add_argument("--count",type=int,default=100)
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--mask",type=Path,help="Optional greyscale density mask; black excludes roads/water")
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    if args.count<0:parser.error("--count must be nonnegative")
    data=placements(args.tile,args.category,args.count,args.seed,args.mask)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,indent=2)+"\n")
    print(f"Placed {data['placed_count']}/{args.count} instances -> {args.out}")


if __name__=="__main__":main()
