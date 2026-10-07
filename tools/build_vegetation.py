#!/usr/bin/env python3
"""Build detailed, opaque glTF vegetation for the terrain renderer.

Run download_vegetation_materials.py once, then this script (NumPy/Pillow).
All woody surfaces are swept 3D curves. Leaves, needles and grass have actual
silhouettes and two surfaces; no alpha-test support or billboard shader needed.
"""
from pathlib import Path
import argparse
import json
import math
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/vegetation"
TAU = math.tau


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def frame(direction):
    direction = unit(direction)
    side = unit(np.cross(direction, (0,1,0) if abs(direction[1]) < .9 else (1,0,0)))
    return side, unit(np.cross(side, direction))


def curve(points,t):
    """The same centreline evaluated by tube(), for attached child branches."""
    points=np.asarray(points,float)
    padded=np.vstack((2*points[0]-points[1],points,2*points[-1]-points[-2]))
    scaled=np.clip(t,0,1)*(len(points)-1)
    i=min(int(scaled),len(points)-2);u=scaled-i
    a,b,c,d=padded[i:i+4]
    return .5*(2*b+(-a+c)*u+(2*a-5*b+4*c-d)*u*u+(-a+3*b-3*c+d)*u*u*u)


def foliage_maps():
    """Authored 2K leaf/blade surfaces, with aligned veins and variable wax."""
    y, x = np.mgrid[:2048,:2048] / 2047
    u = abs(x-.5)*2
    grain = np.sin(x*1723+y*1397)*np.sin(x*4937-y*1829)
    for name, color, blade in (("leaf", (.22,.36,.075), False),
                               ("needle", (.12,.23,.10), True),
                               ("grass", (.26,.38,.085), True),
                               ("dry_grass", (.53,.42,.19), True),
                               ("fern", (.15,.30,.055), False)):
        midrib = np.exp(-(u/.027)**2)
        if blade:
            veins = (.5+.5*np.cos(x*TAU*18))**12
            height = .012*veins+.12*midrib+.018*grain
            pigment = .78+.23*np.sin(y*math.pi)-.13*u+.035*grain
        else:
            veins = np.exp(-(np.sin((y-u*.28)*TAU*9)/.07)**2)*(1-u)
            height = .08*veins+.22*midrib+.012*grain
            pigment = .82+.16*np.sin(y*math.pi)-.16*u+.045*grain
        rgb = np.asarray(color)[None,None,:]*pigment[...,None]
        rgb += (midrib*.065+veins*.035)[...,None]*np.array([1,.85,.35])
        dx=(np.roll(height,-1,1)-np.roll(height,1,1))*9
        dy=(np.roll(height,-1,0)-np.roll(height,1,0))*9
        normal=unit(np.stack((-dx,-dy,np.ones_like(x)),axis=-1))*.5+.5
        orm=np.stack((np.ones_like(x),.46+.12*u+.05*grain,np.zeros_like(x)),axis=-1)
        for channel, data in (("albedo",rgb),("normal",normal),("orm",orm)):
            Image.fromarray(np.uint8(np.clip(data,0,1)*255)).save(OUT/"textures"/f"{name}_{channel}.png")


MATERIALS = [
    ("Scanned furrowed bark", "broadleaf_bark", (1,1,1)),
    ("Scanned conifer bark", "conifer_bark", (1,1,1)),
    ("Mature broadleaf", "leaf", (1,1,1)),
    ("Sunlit young broadleaf", "leaf", (1.15,1.10,.85)),
    ("Evergreen needles", "needle", (1,1,1)),
    ("Meadow grass", "grass", (1,1,1)),
    ("Dry grass", "dry_grass", (1,1,1)),
    ("Fern fronds", "fern", (1,1,1)),
    ("Ivory flower petals", None, (.78,.75,.58)),
    ("Golden pollen", None, (.65,.35,.035)),
    ("Exposed heartwood", "broadleaf_bark", (1.3,1.13,.83)),
]


class Geometry:
    def __init__(self, lod, heights):
        self.parts={}
        self.lod=lod
        self.heights=heights
        self.branch_count=0
        self.leaf_count=0

    def add(self, material, positions, normals, uvs, tangents, indices):
        self.parts.setdefault(material, []).append((np.asarray(positions,dtype=np.float32),
            np.asarray(normals,dtype=np.float32),np.asarray(uvs,dtype=np.float32),
            np.asarray(tangents,dtype=np.float32),np.asarray(indices,dtype=np.uint32)))

    def tube(self, points, radii, material=0, sides=16, samples=6, bark=True):
        """Smooth radius-interpolated spline sweep, transported frame, metre UVs."""
        if bark and self.lod and max(radii)<(.004 if self.lod==1 else .016):
            return
        self.branch_count+=1
        points=np.asarray(points,float); radii=np.asarray(radii)
        padded=np.vstack((2*points[0]-points[1],points,2*points[-1]-points[-2]))
        centers=[]; sizes=[]
        samples=max(3,samples//(self.lod+1)); sides=max(6,sides//(self.lod+1))
        for i in range(len(points)-1):
            a,b,c,d=padded[i:i+4]
            for t in np.linspace(0,1,samples,endpoint=False):
                centers.append(.5*(2*b+(-a+c)*t+(2*a-5*b+4*c-d)*t*t+(-a+3*b-3*c+d)*t*t*t))
                sizes.append(radii[i]*(1-t)+radii[i+1]*t)
        centers=np.vstack((centers,points[-1])); sizes=np.r_[sizes,radii[-1]]
        along=unit(np.gradient(centers,axis=0))
        across=frame(along[0])[0]; frames=[]
        for direction in along:
            across=unit(across-direction*np.dot(across,direction))
            frames.append((across.copy(),np.cross(direction,across)))
        f=np.asarray(frames); theta=np.linspace(0,TAU,sides+1)
        radial=f[:,0,None,:]*np.cos(theta)[None,:,None]+f[:,1,None,:]*np.sin(theta)[None,:,None]
        length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(centers,axis=0),axis=1))]
        # Physical texel scale at the base of each branch: no 4K texture
        # stretched along a ten-metre trunk or squeezed into every tiny twig.
        circumference=max(.035,radii[0]*TAU)
        uv=np.stack(np.broadcast_arrays(theta[None,:]/TAU*circumference,length[:,None]),axis=-1)
        rr=np.broadcast_to(sizes[:,None],(len(sizes),sides+1)).copy()
        if bark and material in self.heights:
            im=self.heights[material]; h,w=im.shape
            xi=(uv[...,0]%1*(w-1)).astype(int); yi=(uv[...,1]%1*(h-1)).astype(int)
            # Scanned height actually changes the trunk silhouette, beyond
            # the normal map. Keep amplitude below the small branch radius.
            detail=(im[yi,xi]-.5)*np.minimum(sizes[:,None]*.10,.012)
            detail[:,-1]=detail[:,0]
            rr+=detail
        # Broad root fluting and longitudinal ribs survive at silhouette size.
        rr*=1+.025*np.cos(theta[None,:]*7+length[:,None]*.8)
        p=centers[:,None,:]+radial*rr[...,None]
        around=-f[:,0,None,:]*np.sin(theta)[None,:,None]+f[:,1,None,:]*np.cos(theta)[None,:,None]
        n=unit(np.cross(around,np.gradient(p,axis=0)))
        tangent=unit(around-n*np.sum(around*n,axis=-1,keepdims=True))
        tangents=np.concatenate((tangent,np.ones((*tangent.shape[:2],1))),axis=-1)
        idx=[]
        for i in range(len(centers)-1):
            for j in range(sides):
                a=i*(sides+1)+j; b=a+sides+1
                idx.extend((a,a+1,b,a+1,b+1,b))
        self.add(material,p.reshape(-1,3),n.reshape(-1,3),uv.reshape(-1,2),tangents.reshape(-1,4),idx)
        for row,sign in ((0,-1),(-1,1)):
            cap=np.vstack((centers[row],p[row])); ci=[]
            for j in range(sides): ci.extend((0,j+1,j+2) if sign>0 else (0,j+2,j+1))
            capuv=np.vstack(((.5,.5),np.column_stack((.5+.48*np.cos(theta),.5+.48*np.sin(theta)))))
            self.add(material,cap,np.tile(along[row]*sign,(len(cap),1)),capuv,
                     np.tile([*f[row,0],1],(len(cap),1)),ci)

    def leaf(self, base, direction, length, width, material, seed, style="leaf"):
        """A cambered, tapered, curled surface with an opaque underside."""
        rng=np.random.default_rng(seed)
        # Stable thinning at distance retains the same branches and leaf sites.
        if self.lod and seed%(3**self.lod) != 0: return
        self.leaf_count+=1
        length*=(1,1.18,1.55)[self.lod]
        width*=(1,1.35,1.8)[self.lod]
        direction=unit(direction); side,normal=frame(direction)
        angle=rng.uniform(-math.pi,math.pi)
        side,normal=side*math.cos(angle)+normal*math.sin(angle),normal*math.cos(angle)-side*math.sin(angle)
        segments=(7,4,3)[self.lod] if style != "needle" else (3,2,1)[self.lod]
        t=np.linspace(0,1,segments+1); u=np.array([-1.,0.,1.])
        if style in ("grass","needle"):
            profile=(1-t)**.65
        else:
            profile=np.sin(math.pi*t)**.75
            if style=="oak": profile*=.76+.24*np.cos(t*TAU*3)
        profile=np.maximum(profile,.006)
        bend=rng.uniform(.05,.35) if style != "grass" else rng.uniform(.15,.60)
        center=np.asarray(base)+t[:,None]*direction*length-normal[None,:]*(t[:,None]**2)*length*bend
        p=center[:,None,:]+side[None,None,:]*u[None,:,None]*profile[:,None,None]*width/2
        p+=normal[None,None,:]*(1-u*u)[None,:,None]*profile[:,None,None]*width*.14
        n=unit(np.cross(np.gradient(p,axis=1),np.gradient(p,axis=0)))
        tangent=unit(np.gradient(p,axis=1))
        uv=np.stack(np.broadcast_arrays((u[None,:]+1)/2,t[:,None]),axis=-1)
        idx=[]
        for i in range(segments):
            for j in range(2):
                a=i*3+j; b=a+3; idx.extend((a,a+1,b,a+1,b+1,b))
        idx=np.asarray(idx)
        for sign in (1,-1):
            # Laurel has a visibly heavier, leathery blade. Taper the offset
            # at the tips too, so the two surfaces never fold through them.
            thickness=min(.00065 if style=="laurel" else .00012,width*.003)*profile[:,None,None]
            pp=p+n*thickness*sign
            tt=np.concatenate((tangent,np.full((*tangent.shape[:2],1),sign)),axis=-1)
            indices=idx if sign==1 else idx.reshape(-1,3)[:,::-1].ravel()
            self.add(material,pp.reshape(-1,3),(n*sign).reshape(-1,3),uv.reshape(-1,2),tt.reshape(-1,4),indices)

    def needles(self,base,direction,length,seed):
        """A dense cylindrical needle shoot, authored as solid tapered needles.

        Vectorised as one mesh chunk per shoot. Tiny needle cross-sections are
        triangular with smooth normals, while trunks and branches remain round.
        """
        rng=np.random.default_rng(seed); direction=unit(direction)
        count=(36,4,1)[self.lod]
        along=np.linspace(.05,1,count)
        side,up=frame(direction)
        theta=along*TAU*13+seed
        outward=side[None,:]*np.cos(theta[:,None])+up[None,:]*np.sin(theta[:,None])
        dirs=unit(outward+direction*.4)
        centers=np.asarray(base)+direction[None,:]*along[:,None]*length
        nl=rng.uniform(.065,.12,(count,1))*(1,1.3,2)[self.lod]
        width=rng.uniform(.0018,.0028,(count,1))*(1,2.5,9)[self.lod]
        aa=unit(np.cross(dirs,np.broadcast_to(direction,dirs.shape)));bb=np.cross(dirs,aa)
        angle=np.arange(3)*TAU/3
        radial=aa[:,None,:]*np.cos(angle)[None,:,None]+bb[:,None,:]*np.sin(angle)[None,:,None]
        rings=centers[:,None,None,:]+dirs[:,None,None,:]*np.array([0,.60])[None,:,None,None]*nl[:,None,:,None]
        rings=rings+radial[:,None,:,:]*width[:,None,:,None]*np.array([.7,1])[None,:,None,None]
        tip=centers+dirs*nl
        p=np.concatenate((rings.reshape(count,6,3),tip[:,None,:]),axis=1)
        n=np.concatenate((np.tile(radial,(1,2,1)),dirs[:,None,:]),axis=1)
        uv=np.tile(np.array([[0,0],[.5,0],[1,0],[0,.6],[.5,.6],[1,.6],[.5,1]]),(count,1,1))
        tangent=unit(np.cross(dirs[:,None,:],n));tangent[:,-1]=aa
        tt=np.concatenate((tangent,np.ones((count,7,1))),axis=-1)
        template=np.array([0,1,3,1,4,3,1,2,4,2,5,4,2,0,5,0,3,5,3,4,6,4,5,6,5,3,6])
        indices=(template[None,:]+np.arange(count)[:,None]*7).ravel()
        self.add(4,p.reshape(-1,3),n.reshape(-1,3),uv.reshape(-1,2),tt.reshape(-1,4),indices)
        self.leaf_count+=count


def roots(g,rng,height,material):
    radius=height*.032
    for i in range(8):
        angle=i*TAU/8+rng.uniform(-.2,.2); d=np.array([math.cos(angle),0,math.sin(angle)])
        reach=rng.uniform(.65,1.2)*radius*3.5
        g.tube([(0,.22,0),d*reach*.4+(0,.12,0),d*reach*.8+(0,.045,0),d*reach+(0,.01,0)],
               [radius*.57,radius*.40,radius*.15,.008],material,20,7)


def broadleaf(g,seed,height=10,spread=3.1,young=False,dense=False):
    rng=np.random.default_rng(seed); wood=0
    lean=rng.uniform(-.35,.35,2)
    def trunk(t): return np.array([lean[0]*t*t,height*t,lean[1]*t*t])
    g.tube([trunk(t) for t in np.linspace(0,.84,11)],
           [height*.036*(1-t/.85)**1.2+.004 for t in np.linspace(0,.84,11)],wood,40,9)
    roots(g,rng,height,wood)
    limbs,secondaries,twigs,leaves=(20,10,9,24) if dense else (9 if young else 13,7,6,22)
    for i in range(limbs):
        t=rng.uniform(.30,.66) if not young else rng.uniform(.24,.67)
        a=i*2.39996+rng.uniform(-.25,.25)
        d=np.array([math.cos(a),0,math.sin(a)])
        reach=spread*(.85+.25*rng.random())*(1-.16*max(0,(t-.5)/.5))
        start=trunk(t); rise=height*(rng.uniform(.70,.92)-t)
        path=[start,start+d*reach*.32+(0,rise*.32,0),start+d*reach*.76+(0,rise*.67,0),start+d*reach+(0,rise,0)]
        r=height*.014*(1-t)**.6
        g.tube(path,[r*1.18,r*.78,r*.34,.009],wood,22,8)
        for j in range(secondaries):
            f=.28+j*(.63/(secondaries-1))
            base=curve(path,f)
            az=a+(-1 if j%2 else 1)*rng.uniform(.55,1.12)
            sd=np.array([math.cos(az),rng.uniform(.35,1.05),math.sin(az)])
            # Spread the laurel crown sideways without also stretching it up.
            if dense: sd[1]*=4.5/spread
            ln=reach*rng.uniform(.28,.48)*(1-.30*f)
            end=base+sd*ln
            secondary=[base,base+sd*ln*.5+(0,.06,0),end]
            g.tube(secondary,[r*.36,.015,.003],wood,12,5)
            for k in range(twigs):
                f2=.16+k*(.80/(twigs-1))
                twigbase=curve(secondary,f2)
                ta=az+(-1 if k%2 else 1)*rng.uniform(.6,1.2)
                td=np.array([math.cos(ta),rng.uniform(-.15,.65),math.sin(ta)])
                tl=rng.uniform(.32,.65)*(height/10)**.5
                if dense: tl*=1.25
                twig=[twigbase,twigbase+td*tl*.55,twigbase+td*tl+(0,.07,0)]
                g.tube(twig,[.009,.005,.0015],wood,8,3)
                for leaf in range(leaves):
                    lf=.10+.90*leaf/(leaves-1)
                    point=curve(twig,lf)
                    lateral=ta+(-1 if leaf%2 else 1)*rng.uniform(.75,1.5)
                    direction=(math.cos(lateral),rng.uniform(-.5,.7),math.sin(lateral))
                    g.leaf(point,direction,rng.uniform(.28,.40) if dense else rng.uniform(.16,.24),
                           rng.uniform(.15,.22) if dense else rng.uniform(.085,.14),
                           2+int(leaf%5==0),seed*100000+i*10000+j*1000+k*30+leaf,
                           "laurel" if dense else "oak")


def rounded_broadleaf(g,seed,height=18,spread=8):
    """Open-grown crown: low forks, rounded lobes and pendent outer sprays.

    Main endpoints fill a dome instead of all rising to the same outer rim.
    Every child still attaches to its parent's actual swept centreline.
    """
    rng=np.random.default_rng(seed)
    def trunk(t):
        return np.array([.28*math.sin(t*3),height*t,.20*t*t])
    ts=np.linspace(0,.83,12)
    g.tube([trunk(t) for t in ts],
           [height*.042*(1-t/.86)**1.3+.009 for t in ts],0,40,9)
    roots(g,rng,height,0)
    for i in range(40):
        level=(i+.5)/40
        az=i*2.39996+rng.uniform(-.2,.2)
        radial=np.array([math.cos(az),0,math.sin(az)])
        tip_y=height*(.31+.60*level)+rng.uniform(-.35,.35)
        dome=(tip_y-height*.51)/(height*.45)
        reach=spread*math.sqrt(max(.10,1-dome*dome))*rng.uniform(.90,1.06)
        start=trunk(.18+.30*level)
        end=radial*reach+np.array([0,tip_y,0])
        rise=end[1]-start[1]
        path=[start,
              start+radial*reach*.30+(0,rise*.58+.45,0),
              start+radial*reach*.72+(0,rise+.55,0),end]
        radius=height*(.019-.009*level)
        g.tube(path,[radius,radius*.72,radius*.34,.012],0,24,9)
        for j in range(9):
            f=.43+.57*j/8
            base=curve(path,f)
            angle=az+(-1 if j%2 else 1)*rng.uniform(.55,1.45)
            side=np.array([math.cos(angle),0,math.sin(angle)])
            length=rng.uniform(1.1,2.1)
            # Layer foliage above AND below the scaffold; outer shoots sag.
            lift=rng.uniform(-1.25,1.25)-.30*f
            secondary=[base,base+side*length*.5+(0,lift*.35+.20,0),
                       base+side*length+(0,lift,0)]
            g.tube(secondary,[.038,.019,.003],0,12,5)
            for k in range(7):
                twigbase=curve(secondary,.15+.85*k/6)
                ta=angle+(-1 if k%2 else 1)*rng.uniform(.65,1.65)
                direction=np.array([math.cos(ta),rng.uniform(-1.25,1.25),math.sin(ta)])
                length2=rng.uniform(.65,1.05)
                twig=[twigbase,twigbase+direction*length2*.5+(0,.10,0),
                      twigbase+direction*length2-(0,rng.uniform(.12,.36),0)]
                g.tube(twig,[.010,.005,.0015],0,8,3)
                for leaf in range(22):
                    point=curve(twig,.08+.92*leaf/21)
                    lateral=ta+(-1 if leaf%2 else 1)*rng.uniform(.65,1.55)
                    ld=(math.cos(lateral),rng.uniform(-.65,.45),math.sin(lateral))
                    g.leaf(point,ld,rng.uniform(.28,.40),rng.uniform(.15,.22),
                           2+int(leaf%5==0),seed*100000+i*10000+j*1000+k*30+leaf,"laurel")


def conifer(g,seed,height=10,spruce=False,wind=0):
    rng=np.random.default_rng(seed); wood=1
    def trunk(t): return np.array([wind*t*t,height*t,.12*math.sin(t*3)])
    g.tube([trunk(t) for t in np.linspace(0,1,13)],
           [height*.029*(1-t)**1.1+.003 for t in np.linspace(0,1,13)],wood,36,8)
    roots(g,rng,height,wood)
    tiers=13 if spruce else 9
    for i in range(tiers):
        tier=(.13 if spruce else .33)+i*((.94-(.13 if spruce else .33))/(tiers-1))
        for j in range(6):
            t=float(np.clip(tier+rng.uniform(-.037,.037),.10,.96))
            reach=height*(.32 if spruce else .36)*(1-t)**.75
            a=j*TAU/6+i*1.31+rng.uniform(-.22,.22)
            d=np.array([math.cos(a),0,math.sin(a)])
            base=trunk(t); ln=reach*rng.uniform(.75,1.15)
            droop=-ln*.26 if spruce else ln*.20
            def branch(f): return base+d*ln*f+(0,droop*math.sin(f*math.pi*.8)+.1*f*f,0)
            r=height*.008*(1-t)**.6
            branch_path=[branch(f) for f in (0,.3,.7,1)]
            g.tube(branch_path,[r,r*.8,r*.3,.002],wood,14,5)
            for k in range(7):
                f=.22+k*.12; tb=curve(branch_path,f)
                ta=a+(-1 if k%2 else 1)*rng.uniform(.55,1.10)
                td=unit([math.cos(ta),rng.uniform(-.85,-.2) if spruce else rng.uniform(.0,.8),math.sin(ta)])
                tl=ln*(.40-.20*f)
                g.tube([tb,tb+td*tl*.55,tb+td*tl],[.008,.005,.0015],wood,8,3)
                side,up=frame(td)
                for shoot in range(7):
                    sf=.18+shoot*.12
                    point=tb+td*tl*sf
                    direction=unit(td*.45+side*(-1 if shoot%2 else 1)*.65+up*rng.uniform(-.7,.6))
                    sl=max(.10,tl*(.48-.18*sf))
                    g.tube([point,point+direction*sl],[.003,.0007],wood,6,3)
                    g.needles(point,direction,sl,seed*100000+i*10000+j*1000+k*10+shoot)
    # Needle-bearing leader and its small ascending shoots close the crown.
    for i in range(14):
        t=.86+i*.01; a=i*2.39996
        d=unit([math.cos(a)*.5,1,math.sin(a)*.5])
        g.needles(trunk(t),d,height*.055*(1-(t-.86)*3),seed*1000000+i)
    # Small broken/dead lower branches retain the trunk silhouette in winter.
    for i in range(5):
        a=i*2.4; base=trunk(.1+i*.035); d=np.array([math.cos(a),.18,math.sin(a)])
        g.tube([base,base+d*.24,base+d*.48],[.035,.024,.008],wood,12,5)


def grass(g,seed,kind="meadow"):
    rng=np.random.default_rng(seed)
    count={"meadow":240,"short":170,"dry":145}[kind]
    radius=.28 if kind!="short" else .40
    for i in range(count):
        angle=rng.uniform(0,TAU); r=radius*math.sqrt(rng.random())
        point=np.array([math.cos(angle)*r,0,math.sin(angle)*r])
        direction=unit([math.cos(angle)*rng.uniform(.1,.7),1,math.sin(angle)*rng.uniform(.1,.7)])
        length=rng.uniform(.26,.65) if kind=="meadow" else rng.uniform(.08,.24) if kind=="short" else rng.uniform(.38,.85)
        g.leaf(point,direction,length,rng.uniform(.007,.014),6 if kind=="dry" else 5,seed*1000+i,"grass")
        if kind=="dry" and i%12==0:
            end=point+direction*length
            g.tube([point,end*.5+point*.5,end],[.0017,.0012,.0005],6,6,3,False)
            for j in range(9):
                p=end-direction*.13+j*.014*direction
                a=j*2.4
                g.leaf(p,[math.cos(a)*.4,.8,math.sin(a)*.4],.026,.006,6,seed*10000+i*10+j,"needle")


def fern(g,seed):
    rng=np.random.default_rng(seed)
    for i in range(11):
        a=i*2.39996; d=np.array([math.cos(a),0,math.sin(a)])
        length=rng.uniform(.55,1.0)
        def stem(t): return d*length*t+(0,length*(1.3*t-.95*t*t)+.025,0)
        g.tube([stem(t) for t in np.linspace(0,1,9)],[.008*(1-t)+.001 for t in np.linspace(0,1,9)],7,8,4,False)
        side=np.array([-d[2],.10,d[0]])
        for j in range(1,19):
            t=j/20; span=length*.27*math.sin(t*math.pi)**.7
            for sign in (-1,1):
                tip=stem(t)+side*sign*span+d*span*.2
                g.tube([stem(t),tip],[.0025,.0006],7,6,3,False)
                for k in range(6):
                    along=(k+.4)/6
                    base=stem(t)+side*sign*span*along+d*span*.2*along
                    direction=side*sign*.25+d*(-1 if k%2 else 1)*.7+(0,.10,0)
                    g.leaf(base,direction,span*.26*(1-along*.7),span*.10,7,seed*10000+i*1000+j*20+k*2+(sign+1)//2)


def shrub(g,seed):
    rng=np.random.default_rng(seed)
    for i in range(9):
        a=i*2.39996; d=np.array([math.cos(a),1.6,math.sin(a)])
        length=rng.uniform(.45,.75)
        g.tube([(0,0,0),d*length*.45,d*length],[.025,.018,.004],0,12,6)
        for j in range(7):
            base=d*length*(.25+j*.1); b=a+(-1 if j%2 else 1)*1.05
            direction=np.array([math.cos(b),.30,math.sin(b)])
            tip=base+direction*.27
            g.tube([base,(base+tip)/2,tip],[.005,.003,.0009],0,8,3)
            for k in range(14):
                f=k/13
                g.leaf(base+direction*.27*f,[math.cos(b+(-1 if k%2 else 1)),.3,math.sin(b+(-1 if k%2 else 1))],
                       .105,.049,2+k%2,seed*10000+i*1000+j*20+k)


def flowers(g,seed):
    grass(g,seed,"short")
    rng=np.random.default_rng(seed)
    for i in range(15):
        base=np.array([rng.uniform(-.25,.25),0,rng.uniform(-.25,.25)])
        tip=base+np.array([rng.uniform(-.06,.06),rng.uniform(.20,.42),rng.uniform(-.06,.06)])
        g.tube([base,(base+tip)/2,tip],[.002,.0015,.001],5,6,3,False)
        for j in range(10):
            a=j*TAU/10
            g.leaf(tip,[math.cos(a),.2,math.sin(a)],.034,.014,8,seed*1000+i*10+j)
        g.tube([tip,tip+(0,.009,0)],[.010,.008],9,12,3,False)


def deadwood(g,seed,stump=False):
    rng=np.random.default_rng(seed)
    if stump:
        roots(g,rng,7,0)
        g.tube([(0,0,0),(.02,.30,.01),(.05,.63,0)],[.28,.23,.21],0,40,10)
        for i in range(8):
            a=i*TAU/8; d=np.array([math.cos(a),0,math.sin(a)])
            base=d*.17+(0,.58,0)
            g.tube([base,base+d*.035+(0,rng.uniform(.1,.25),0)],[.06,.002],10,10,4)
    else:
        g.tube([(-1.25,.18,0),(-.5,.21,.10),(.3,.18,.04),(1.3,.15,-.07)],[.20,.21,.17,.12],0,36,10)
        for i in range(4):
            base=np.array([-.8+i*.50,.21,.06]); d=np.array([.1,.28,(-1 if i%2 else 1)*.35])
            g.tube([base,base+d*.6,base+d],[.06,.035,.012],0,14,5)


class Export:
    def __init__(self):
        self.buffer=bytearray()
        self.doc={"asset":{"version":"2.0","generator":"Gameport detailed vegetation"},
                  "scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
                  "meshes":[],"materials":[],"images":[],"textures":[],
                  "samplers":[{"magFilter":9729,"minFilter":9987,"wrapS":10497,"wrapT":10497}],
                  "accessors":[],"bufferViews":[]}

    def accessor(self,a,kind,index=False):
        a=np.asarray(a,dtype="<u4" if index else "<f4")
        while len(self.buffer)%4:self.buffer.append(0)
        v=len(self.doc["bufferViews"])
        self.doc["bufferViews"].append({"buffer":0,"byteOffset":len(self.buffer),"byteLength":a.nbytes})
        self.buffer.extend(a.tobytes())
        acc={"bufferView":v,"componentType":5125 if index else 5126,"count":len(a),"type":kind}
        if kind=="VEC3":acc.update(min=a.min(axis=0).tolist(),max=a.max(axis=0).tolist())
        self.doc["accessors"].append(acc)
        return len(self.doc["accessors"])-1

    def texture(self,path):
        i=len(self.doc["images"]); self.doc["images"].append({"uri":"../textures/"+path})
        self.doc["textures"].append({"source":i,"sampler":0}); return {"index":i}

    def write(self,g,name):
        primitives=[]; all_pos=[]; triangles=0; vertices=0
        for mat,parts in sorted(g.parts.items()):
            positions=[]; normals=[]; uvs=[]; tangents=[]; indices=[]; offset=0
            for p,n,uv,t,idx in parts:
                positions.append(p); normals.append(n); uvs.append(uv); tangents.append(t)
                indices.append(idx+offset); offset+=len(p)
            arrays=list(map(np.concatenate,(positions,normals,uvs,tangents,indices)))
            p,n,uv,t,idx=arrays
            all_pos.append(p); triangles+=len(idx)//3; vertices+=len(p)
            material_index=len(self.doc["materials"])
            label,texture,color=MATERIALS[mat]
            pbr={"baseColorFactor":[*color,1],"metallicFactor":0,"roughnessFactor":1 if texture else .7}
            material={"name":label,"pbrMetallicRoughness":pbr,"doubleSided":False}
            if texture:
                bark="bark" in texture
                pbr["baseColorTexture"]=self.texture(texture+"_albedo."+("jpg" if bark else "png"))
                pbr["metallicRoughnessTexture"]=self.texture(texture+"_orm."+("jpg" if bark else "png"))
                material["normalTexture"]={**self.texture(texture+"_normal.png"),"scale":.85 if bark else .55}
                material["occlusionTexture"]=pbr["metallicRoughnessTexture"].copy()
            self.doc["materials"].append(material)
            primitives.append({"material":material_index,"attributes":{
                "POSITION":self.accessor(p,"VEC3"),"NORMAL":self.accessor(n,"VEC3"),
                "TEXCOORD_0":self.accessor(uv,"VEC2"),"TANGENT":self.accessor(t,"VEC4")},
                "indices":self.accessor(idx,"SCALAR",True)})
        self.doc["meshes"]=[{"name":name,"primitives":primitives}]
        self.doc["nodes"][0]["name"]=name
        self.doc["buffers"]=[{"uri":name+".bin","byteLength":len(self.buffer)}]
        (OUT/"models"/(name+".bin")).write_bytes(self.buffer)
        (OUT/"models"/(name+".gltf")).write_text(json.dumps(self.doc,indent=2)+"\n")
        positions=np.concatenate(all_pos)
        return {"path":"models/"+name+".gltf","vertices":vertices,"triangles":triangles,
                "draws":len(primitives),"bounds_min":positions.min(axis=0).tolist(),
                "bounds_max":positions.max(axis=0).tolist(),"branch_sweeps":g.branch_count,
                "foliage_elements":g.leaf_count}


ASSETS=[
    ("pine_mature",conifer,{"seed":101,"height":12},"tree",[.85,1.2],32),
    ("pine_windswept",conifer,{"seed":113,"height":8.5,"wind":1.35},"tree",[.8,1.15],38),
    ("spruce_mature",conifer,{"seed":127,"height":10,"spruce":True},"tree",[.9,1.25],32),
    ("spruce_young",conifer,{"seed":139,"height":4.2,"spruce":True},"sapling",[.75,1.25],38),
    ("oak_spreading",broadleaf,{"seed":151,"height":9,"spread":3.8},"tree",[.85,1.15],26),
    ("oak_tall",broadleaf,{"seed":163,"height":12,"spread":3.0},"tree",[.9,1.2],26),
    ("laurel_giant",rounded_broadleaf,{"seed":251,"height":18,"spread":8.0},"tree",[.9,1.15],26),
    ("oak_sapling",broadleaf,{"seed":173,"height":3.8,"spread":1.1,"young":True},"sapling",[.75,1.3],32),
    ("grass_meadow_a",grass,{"seed":181},"grass",[.7,1.25],42),
    ("grass_meadow_b",grass,{"seed":193},"grass",[.7,1.3],42),
    ("grass_short",grass,{"seed":197,"kind":"short"},"grass",[.7,1.3],45),
    ("grass_dry",grass,{"seed":211,"kind":"dry"},"grass",[.8,1.2],45),
    ("fern_large",fern,{"seed":223},"undergrowth",[.8,1.2],35),
    ("fern_small",fern,{"seed":227},"undergrowth",[.5,.8],35),
    ("woodland_shrub",shrub,{"seed":229},"undergrowth",[.7,1.2],32),
    ("meadow_flowers",flowers,{"seed":233},"flowers",[.75,1.2],30),
    ("fallen_log",deadwood,{"seed":239},"deadwood",[.8,1.4],22),
    ("broken_stump",deadwood,{"seed":241,"stump":True},"deadwood",[.75,1.3],32),
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only",help="Comma-separated asset ids (for art iteration)")
    parser.add_argument("--skip-textures",action="store_true")
    args=parser.parse_args()
    (OUT/"models").mkdir(parents=True,exist_ok=True)
    (OUT/"textures").mkdir(exist_ok=True)
    heights={}
    for i,role in enumerate(("broadleaf_bark","conifer_bark")):
        path=OUT/"textures"/(role+"_height.png")
        if not path.exists(): raise SystemExit("Run tools/download_vegetation_materials.py first")
        im=np.asarray(Image.open(path).convert("F"),dtype=float)
        heights[i]=im/max(float(im.max()),1)
    if not args.skip_textures:foliage_maps()
    manifest_path=OUT/"manifest.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "version":1,"units":"metres","up_axis":"Y","pivot":"ground centre",
        "materials":"material_sources.json","assets":[]}
    for name,fn,kw,category,scale,slope in ASSETS:
        if args.only and name not in args.only.split(","):continue
        record={"id":name,"category":category,"seed":kw["seed"],"scale_range":scale,
                "yaw_degrees":[0,360],"max_slope_degrees":slope,"ground_sink_m":.015,
                "align_to_slope":category not in ("tree","sapling"),"lods":[]}
        for lod in (0,1,2):
            g=Geometry(lod,heights);fn(g,**kw)
            result=Export().write(g,f"{name}_lod{lod}")
            result["distance_m"]=(0,45,120)[lod] if category=="tree" else (0,16,45)[lod] if category=="sapling" else (0,9,25)[lod]
            record["lods"].append(result)
            print(f"{name} LOD{lod}: {result['triangles']:,} triangles; {g.branch_count} branch sweeps; {g.leaf_count} leaves/blades",flush=True)
        b=record["lods"][0]
        record["spacing_radius_m"]=round(max(abs(b["bounds_min"][0]),abs(b["bounds_max"][0]),
                                             abs(b["bounds_min"][2]),abs(b["bounds_max"][2]))*.8,3)
        record["trunk_collision_radius_m"]=round(kw.get("height",0)*.035,3)
        manifest["assets"]=[a for a in manifest["assets"] if a["id"]!=name]+[record]
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Vegetation pack ready: {OUT}")


if __name__=="__main__":
    main()
