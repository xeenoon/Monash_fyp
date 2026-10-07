#!/usr/bin/env python3
"""Build the dungeon's Medusa: swept 3D splines, sculpted masks and PBR maps.

No Blender or downloaded art required: python3 tools/build_medusa_guardian.py
Requires numpy and Pillow. Metres, Y up, face towards +Z. Generated assets are
checked in, so neither Python nor the generator is needed to play the game.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/characters/medusa"
TAU = math.tau


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def textures():
    """Periodic scale relief, mottling, pores and mineral veins; no baked light."""
    y, x = np.mgrid[0:1024, 0:1024] / 1024
    grain = (np.sin(TAU * (83*x + 71*y)) * np.sin(TAU * (137*x - 53*y)))
    cloud = (np.sin(TAU*(3*x+2*y)) + .5*np.sin(TAU*(7*x-5*y))
             + .25*np.cos(TAU*(19*x+11*y))) / 1.75
    row = np.floor(y*24)
    sx = ((x*16 + .5*(row % 2)) % 1 - .5)*2
    sy = (y*24) % 1
    scale = np.clip(1 - sx*sx - ((sy-.43)/.64)**2, 0, 1)
    relief = np.sqrt(scale)*.8 + .025*grain
    veins = np.exp(-np.abs(np.sin(TAU*(4*x+3*y) + 1.7*np.sin(TAU*(x-2*y))))*38)
    for name in ("scales", "marble"):
        if name == "scales":
            blend = np.clip(.45 + .35*cloud + .16*np.sin(TAU*(2*x-y)), 0, 1)
            rgb = np.array([.21, .39, .32])[None,None,:]*(1-blend[...,None])
            rgb += np.array([.36, .20, .39])[None,None,:]*blend[...,None]
            rgb *= (.55 + .65*np.sqrt(scale) + .10*grain)[...,None]
            rgb += (.10*veins)[...,None]*np.array([.45,.8,.55])
            height = relief
            rough = .30 + .27*(1-scale) + .06*cloud
            ao = .66 + .34*np.sqrt(scale)
        else:
            rgb = np.array([.64,.68,.55])[None,None,:]*(.90+.12*cloud+.025*grain)[...,None]
            rgb *= (1-.58*veins)[...,None]
            height = .025*cloud + .012*grain - .055*veins
            rough = .51 + .12*cloud - .10*veins
            ao = 1-.16*veins
        dx = (np.roll(height,-1,axis=1)-np.roll(height,1,axis=1))*5
        dy = (np.roll(height,-1,axis=0)-np.roll(height,1,axis=0))*5
        normal = unit(np.stack((-dx,-dy,np.ones_like(dx)),axis=-1))*.5+.5
        orm = np.stack((ao,rough,np.zeros_like(x)),axis=-1)
        for suffix, array in (("albedo",rgb),("normal",normal),("orm",orm)):
            Image.fromarray(np.uint8(np.clip(array,0,1)*255)).save(OUT/f"{name}_{suffix}.png")


class Geometry:
    def __init__(self):
        self.parts = {}

    def add(self, material, p, n, uv, idx):
        self.parts.setdefault(material, []).append((np.asarray(p),np.asarray(n),np.asarray(uv),np.asarray(idx)))

    def ellipsoid(self, material, center, size, segments=32, rings=20, sculpt=False, chin=None):
        # Offset poles very slightly to keep UV tangent triangles nondegenerate.
        v = np.linspace(.0001,math.pi-.0001,rings+1)
        u = np.linspace(0,TAU,segments+1)
        vv, uu = np.meshgrid(v,u,indexing="ij")
        normal = np.stack((np.sin(vv)*np.cos(uu),np.cos(vv),np.sin(vv)*np.sin(uu)),axis=-1)
        p = normal*np.array(size)
        if sculpt:
            xx, yy = p[...,0], p[...,1]
            front = np.maximum(normal[...,2],0)**5
            # Recessed eye sockets, a narrowed jaw and a drawn brow.
            sockets = sum(np.exp(-((xx-s*.112)/.072)**2-((yy-.025)/.047)**2) for s in (-1,1))
            p[...,2] -= front*.043*sockets
            cheeks = sum(np.exp(-((xx-s*.126)/.065)**2-((yy+.063)/.045)**2) for s in (-1,1))
            p[...,2] += front*.018*cheeks
            p[...,0] *= 1-.22*np.clip(-yy/.25,0,1)
            if chin is not None:
                p[...,1] = np.where(p[...,1] < chin, chin+(p[...,1]-chin)*.15, p[...,1])
        p += center
        idx = grid_indices(rings,segments,reverse=True)
        # Recompute normals on the sculpt, where the analytic ellipsoid differs.
        n = unit(normal/np.array(size))
        if sculpt:
            flat = p.reshape(-1,3)
            faces = idx.reshape(-1,3)
            fn = np.cross(flat[faces[:,1]]-flat[faces[:,0]],flat[faces[:,2]]-flat[faces[:,0]])
            nn = np.zeros_like(flat)
            for c in range(3): np.add.at(nn,faces[:,c],fn)
            n = unit(nn).reshape(p.shape)
            seam = unit(n[:,0]+n[:,-1]); n[:,0] = n[:,-1] = seam
        self.add(material,p.reshape(-1,3),n.reshape(-1,3),
                 np.stack((uu/TAU,vv/math.pi),axis=-1).reshape(-1,2),idx)

    def tube(self, material, points, radii, sides=16, samples=12, squash=1):
        """Catmull-Rom centreline -> parallel-transport frames -> swept rings.

        Radius and arc-length UVs belong to whole rings. No independently
        displaced vertices, disconnected cones, or camera-facing strips.
        """
        points = np.asarray(points,dtype=float)
        padded = np.vstack((2*points[0]-points[1],points,2*points[-1]-points[-2]))
        centers=[]; rr=[]
        for i in range(len(points)-1):
            a,b,c,d = padded[i:i+4]
            for t in np.linspace(0,1,samples,endpoint=False):
                centers.append(.5*((2*b)+(-a+c)*t+(2*a-5*b+4*c-d)*t*t+(-a+3*b-3*c+d)*t*t*t))
                rr.append(radii[i]*(1-t)+radii[i+1]*t)
        centers=np.vstack((centers,points[-1])); rr=np.r_[rr,radii[-1]]
        tangent=unit(np.gradient(centers,axis=0))
        axis=np.array([0.,0.,1.]) if abs(tangent[0,2])<.9 else np.array([1.,0.,0.])
        across=unit(np.cross(tangent[0],axis)); frames=[]
        for direction in tangent:
            across=unit(across-direction*np.dot(across,direction))
            frames.append((across.copy(),np.cross(direction,across)))
        frames=np.asarray(frames)
        theta=np.linspace(0,TAU,sides+1)
        radial=frames[:,0,None,:]*np.cos(theta)[None,:,None]+frames[:,1,None,:]*np.sin(theta)[None,:,None]*squash
        p=centers[:,None,:]+radial*rr[:,None,None]
        # Derivatives include radius taper and bending, so highlights follow it.
        along=np.gradient(p,axis=0)
        around=-frames[:,0,None,:]*np.sin(theta)[None,:,None]+frames[:,1,None,:]*np.cos(theta)[None,:,None]*squash
        n=unit(np.cross(around,along))
        length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(centers,axis=0),axis=1))]
        uv=np.stack(np.broadcast_arrays(theta[None,:]/TAU,length[:,None]/.48),axis=-1)
        idx=grid_indices(len(centers)-1,sides,reverse=True)
        self.add(material,p.reshape(-1,3),n.reshape(-1,3),uv.reshape(-1,2),idx)
        # Closed end discs; hidden root disc still makes a watertight volume.
        for row,sign in ((0,-1),(-1,1)):
            cap=np.vstack((centers[row],p[row]))
            ci=[]
            for j in range(sides):
                ci.extend((0,j+1,j+2) if sign>0 else (0,j+2,j+1))
            self.add(material,cap,np.tile(tangent[row]*sign,(len(cap),1)),np.zeros((len(cap),2)),ci)


def grid_indices(rings,sides,reverse=False):
    idx=[]
    for i in range(rings):
        for j in range(sides):
            a=i*(sides+1)+j; b=a+sides+1
            idx.extend((a,b,a+1,a+1,b,b+1) if not reverse else (a,a+1,b,a+1,b+1,b))
    return np.array(idx,dtype=np.uint32)


def mask(g, center, scale=1, serpent=False):
    """Layered sculpt: recessed sockets, lids, cheekbones, nose, lips and fangs."""
    c=np.asarray(center)
    def ball(mat,pos,size):
        g.ellipsoid(mat,c+np.array(pos)*scale,np.array(size)*scale,12 if serpent else 24,8 if serpent else 16)
    def line(mat,points,r):
        g.tube(mat,c+np.array(points)*scale,np.array(r)*scale,8 if serpent else 12,4 if serpent else 6)
    stone=0 if serpent else 1
    g.ellipsoid(stone,c,np.array([.205,.268,.163])*scale,24 if serpent else 40,16 if serpent else 28,
                sculpt=True,chin=-.11*scale if serpent else None)
    for s in (-1,1):
        ball(4,(s*.104,.026,.143),(.067,.030,.025))
        ball(2,(s*.105,.028,.163),(.044,.015,.015))
        ball(4,(s*.105,.029,.178),(.006,.015,.004))
        line(stone,[(s*.035,.041,.151),(s*.10,.053,.158),(s*.17,.064,.118)], [.012,.021,.008])
        line(stone,[(s*.044,.012,.151),(s*.11,.006,.155),(s*.169,.025,.12)], [.006,.008,.006])
        if serpent:
            line(5,[(s*.056,-.119,.14),(s*.055,-.175,.162),(s*.042,-.187,.181)],[.018,.013,.001])
    ball(stone,(0,-.004,.163),(.029,.072,.038))
    ball(stone,(0,-.052,.19),(.040,.024,.030))
    ball(4,(0,-.124,.139),(.066,.026,.022))
    line(3,[(-.065,-.12,.142),(-.026,-.102,.157),(0,-.111,.161),(.026,-.102,.157),(.065,-.12,.142)], [.006,.011,.01,.011,.006])
    if not serpent:
        line(stone,[(-.06,-.13,.139),(0,-.147,.155),(.06,-.13,.139)],[.006,.015,.006])
        ball(3,(0,.159,.137),(.045,.059,.018))
        ball(2,(0,.158,.157),(.023,.037,.012))
        ball(4,(0,.158,.168),(.006,.031,.004))


class Asset:
    def __init__(self):
        self.data=bytearray()
        self.pivots={0:np.zeros(3)}
        self.doc={"asset":{"version":"2.0","generator":"Gameport spline Medusa"},
                  "scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"name":"Medusa","children":[]}],
                  "meshes":[],"accessors":[],"bufferViews":[],"materials":[],"images":[],"textures":[],
                  "samplers":[{"magFilter":9729,"minFilter":9987,"wrapS":10497,"wrapT":10497}]}

    def accessor(self,array,kind,indices=False):
        array=np.asarray(array,dtype="<u4" if indices else "<f4")
        while len(self.data)%4:self.data.append(0)
        view=len(self.doc["bufferViews"])
        self.doc["bufferViews"].append({"buffer":0,"byteOffset":len(self.data),"byteLength":array.nbytes})
        self.data.extend(array.tobytes())
        result={"bufferView":view,"componentType":5125 if indices else 5126,"count":len(array),"type":kind}
        if kind=="VEC3":result.update(min=array.min(axis=0).tolist(),max=array.max(axis=0).tolist())
        if kind=="SCALAR":result.update(min=[float(array.min())],max=[float(array.max())])
        self.doc["accessors"].append(result)
        return len(self.doc["accessors"])-1

    def texture(self,name):
        i=len(self.doc["images"])
        self.doc["images"].append({"uri":name})
        self.doc["textures"].append({"source":i,"sampler":0})
        return {"index":i}

    def material(self,name,color,rough,texture=None,metal=0):
        pbr={"baseColorFactor":[*color,1],"roughnessFactor":rough,"metallicFactor":metal}
        mat={"name":name,"pbrMetallicRoughness":pbr}
        if texture:
            pbr["baseColorTexture"]=self.texture(texture+"_albedo.png")
            pbr["metallicRoughnessTexture"]=self.texture(texture+"_orm.png")
            mat["occlusionTexture"]=pbr["metallicRoughnessTexture"].copy()
            mat["normalTexture"]={**self.texture(texture+"_normal.png"),"scale":.7}
        self.doc["materials"].append(mat)

    def node(self,name,g,pivot=(0,0,0),parent=0):
        primitives=[]
        for material,parts in g.parts.items():
            p=[]; n=[]; uv=[]; idx=[]; count=0
            for pp,nn,tt,ii in parts:
                p.extend(pp-np.asarray(pivot)); n.extend(nn); uv.extend(tt); idx.extend(ii+count); count+=len(pp)
            primitives.append({"attributes":{"POSITION":self.accessor(p,"VEC3"),"NORMAL":self.accessor(n,"VEC3"),
                                              "TEXCOORD_0":self.accessor(uv,"VEC2")},
                               "indices":self.accessor(idx,"SCALAR",True),"material":material})
        mesh=len(self.doc["meshes"]); self.doc["meshes"].append({"name":name,"primitives":primitives})
        node=self.anchor(name,pivot,parent)
        self.doc["nodes"][node]["mesh"]=mesh
        return node

    def anchor(self,name,pivot,parent):
        node=len(self.doc["nodes"])
        self.pivots[node]=np.asarray(pivot)
        self.doc["nodes"].append({"name":name,
                                  "translation":(self.pivots[node]-self.pivots[parent]).tolist()})
        self.doc["nodes"][parent].setdefault("children",[]).append(node)
        return node

    def animate(self,rigs,tendrils):
        # Different phases and numbers of oscillations keep the heads from
        # moving in unison. Each attack is a fast double strike, with a hard
        # jaw closure at full extension and recoil before the second bite.
        # Bake at 60 Hz so the 45 ms jaw slam survives LINEAR interpolation.
        times=np.linspace(0,16,961)
        time_index=self.accessor(times,"SCALAR")
        clip={"name":"Serpentine","samplers":[],"channels":[]}
        def channel(node,path,values):
            sampler=len(clip["samplers"])
            clip["samplers"].append({"input":time_index,"output":self.accessor(values,"VEC4" if path=="rotation" else "VEC3"),"interpolation":"LINEAR"})
            clip["channels"].append({"sampler":sampler,"target":{"node":node,"path":path}})
        def rotate(node,pitch,yaw,roll):
            # Quaternion Ry * Rx * Rz, glTF xyzw; all angles in radians.
            sx,cx=np.sin(pitch/2),np.cos(pitch/2)
            sy,cy=np.sin(yaw/2),np.cos(yaw/2)
            sz,cz=np.sin(roll/2),np.cos(roll/2)
            q=np.column_stack((cy*sx*cz+sy*cx*sz,sy*cx*cz-cy*sx*sz,
                               cy*cx*sz-sy*sx*cz,cy*cx*cz+sy*sx*sz))
            channel(node,"rotation",q)
        def smooth(t):
            t=np.clip(t,0,1)
            return t*t*(3-2*t)
        def pulse(t,start,rise,hold,fall):
            return smooth((t-start)/rise)*(1-smooth((t-start-rise-hold)/fall))
        t=times*TAU/16
        for i,(root,socket,head,jaw,tongue) in enumerate(rigs):
            phase=i*2.39996
            local=(times-(.8+i*.71))%8
            strike=pulse(local,0,.10,.08,.20)+.85*pulse(local,.43,.08,.06,.28)
            recoil=pulse(local,.26,.07,.02,.14)
            reach=strike-.16*recoil
            gape=pulse(local,0,.065,.085,.045)+pulse(local,.43,.055,.075,.045)
            search=np.sin(t*(2+i%3)+phase)
            rotate(root,.065*np.sin(t*2+phase)+.22*reach,
                   .12*search,.075*np.sin(t*3+phase+.7)+.09*recoil*np.sin(i))
            # Extend the whole swept neck from its fixed root; the head stays
            # attached. Compensate its scale so faces retain their proportions.
            depth=max(.10,self.pivots[head][2]-self.pivots[root][2])
            extension=np.column_stack((np.ones_like(t),np.ones_like(t),1+.42/depth*reach))
            channel(root,"scale",extension)
            # Undo extension BEFORE the independent head rotation, avoiding
            # shear when a head turns while its neck is fully extended.
            channel(socket,"scale",1/extension)
            rotate(head,.14*np.sin(t*3+phase)-.30*strike+.15*recoil,
                   .32*search*(1-.9*strike),.12*np.sin(t*2+phase+1))
            rotate(jaw,.015+1.35*gape,np.zeros_like(t),np.zeros_like(t))
            flick=gape*(.5+.5*np.sin(local*TAU*11))
            channel(tongue,"scale",np.column_stack((np.ones_like(t),.12+.88*flick,.28+1.1*flick)))
        for i,node in enumerate(tendrils):
            phase=i*2.7
            rotate(node,.08*np.sin(t*2+phase),.10*np.sin(t*3+phase),.06*np.sin(t+phase))
        self.doc["animations"]=[clip]

    def save(self):
        self.doc["buffers"]=[{"uri":"medusa.bin","byteLength":len(self.data)}]
        (OUT/"medusa.bin").write_bytes(self.data)
        (OUT/"medusa.gltf").write_text(json.dumps(self.doc,indent=2)+"\n")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    textures()
    asset=Asset()
    asset.material("Iridescent serpent scales",(1,1,1),1,"scales")
    asset.material("Veined petrified ivory",(1,1,1),1,"marble")
    asset.material("Amber serpent eyes",(.85,.58,.12),.19)
    asset.material("Oxidised ceremonial bronze",(.29,.16,.055),.37,metal=.78)
    asset.material("Obsidian sockets and slit pupils",(.009,.014,.016),.28)
    asset.material("Old ivory fangs",(.70,.65,.43),.34)
    body=Geometry()
    # A narrowing, ribbed bust grows out of the coiled mass.
    body.tube(0,[(0,.20,-.04),(0,.52,-.03),(0,.82,0),(0,1.09,0),(0,1.28,0)],
              [.27,.28,.20,.29,.115],32,12,squash=.73)
    body.ellipsoid(1,(0,1.05,.10),(.245,.255,.14),40,28)
    body.tube(1,[(0,1.13,0),(0,1.30,.01),(0,1.43,.01)],[.12,.087,.10],24,10)
    mask(body,(0,1.61,.047))
    for s in (-1,1):
        # Clavicles and nested bronze ribs give a recognisable ancient idol silhouette.
        for i in range(5):
            y=1.15-i*.065
            body.tube(3,[(0,y-.035,.224),(s*.11,y-.005,.214),(s*(.235-i*.018),y+.025,.13)],
                      [.009,.014,.008],10,7)
        body.tube(0,[(s*.23,1.18,0),(s*.40,1.02,.02),(s*.43,.77,.12),(s*.34,.61,.25),
                     (s*.24,.73,.32),(s*.30,.87,.31)], [.105,.094,.073,.054,.035,.003],18,12)
    # Five serpentine roots, curling back towards the body rather than straight spokes.
    for i in range(5):
        angle=i*TAU/5+.3
        pts=[]
        for j in range(8):
            t=j/7; a=angle+t*3.2; r=.12+.50*math.sin(t*math.pi*.82)
            pts.append((r*math.cos(a),.10+.32*(1-t)**2,r*math.sin(a)))
        body.tube(0,pts,[.13,.15,.135,.115,.09,.061,.033,.003],20,10)
    asset.node("Petrified oracle and root coils",body)
    rigs=[]
    tendrils=[]
    # Uneven crown: tips face the player, so the many little masks read in play.
    for i in range(11):
        a=(i/10)*math.pi
        x=math.cos(a); height=math.sin(a)
        root=(x*.13,1.73+height*.06,-.035-(i%3)*.035)
        tip=(x*(.56+.075*(i%2)),1.64+height*.43+.05*math.sin(i*4),.10+.12*(i%3))
        pts=[root,(x*.31,1.83+height*.16,-.14),
             (x*(.55+.09*(i%2)),1.90+height*.37,-.16+.07*math.sin(i)),
             (tip[0]+x*.13,tip[1]+.15,tip[2]-.16),tip]
        g=Geometry()
        g.tube(0,pts,[.070,.073,.062,.048,.043],20,15)
        root_node=asset.node(f"Crown serpent {i+1:02}",g,root)
        head=Geometry(); mask(head,tip,.34,True)
        socket_node=asset.anchor(f"Head socket {i+1:02}",tip,root_node)
        head_node=asset.node(f"Searching head {i+1:02}",head,tip,socket_node)
        p=np.asarray(tip)
        jaw=Geometry()
        jaw.ellipsoid(0,p+(0,-.059,.035),(.027,.013,.021),16,10)
        jaw.tube(0,p+np.array([(-.023,-.049,.043),(0,-.052,.056),(.023,-.049,.043)]),
                 [.004,.006,.004],10,6)
        jaw_node=asset.node(f"Hissing jaw {i+1:02}",jaw,p+(0,-.037,.018),head_node)
        tongue=Geometry()
        # Long forked tongue below each small mask.
        for s in (-1,1):
            tongue.tube(3,p+np.array([(0,-.055,.05),(0,-.09,.07),(s*.017,-.11,.075)]),[.005,.004,.001],8,5)
        tongue_node=asset.node(f"Flicking tongue {i+1:02}",tongue,p+(0,-.055,.05),jaw_node)
        rigs.append((root_node,socket_node,head_node,jaw_node,tongue_node))
    # Two long tendrils frame the face; their recurved tips echo the crown.
    for s in (-1,1):
        root=(s*.16,1.72,-.02); g=Geometry()
        g.tube(0,[root,(s*.29,1.52,-.01),(s*.35,1.25,.18),(s*.49,1.32,.28),
                  (s*.48,1.48,.26),(s*.39,1.47,.24)], [.06,.062,.045,.033,.022,.002],18,14)
        tendrils.append(asset.node(f"Temple tendril {s}",g,root))
    asset.animate(rigs,tendrils)
    asset.save()
    print(f"Medusa: {len(asset.doc['nodes'])} nodes, "
          f"{sum(len(m['primitives']) for m in asset.doc['meshes'])} draws, {len(asset.data):,} geometry bytes")


if __name__ == "__main__":
    main()
