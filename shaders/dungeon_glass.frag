#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "pbr_common.glsl"
#include "dungeon_noise.glsl"
#include "environment_lighting.glsl"

layout(location=0) in vec2 uv;
layout(location=1) in vec3 normal;
layout(location=2) in vec4 tangent;
layout(location=3) in vec3 camera_relative_position;
layout(location=4) in vec4 current_clip;
layout(location=5) in vec4 previous_clip;
layout(location=6) in vec3 local_position;
layout(location=0) out vec4 out_color;
layout(location=1) out vec2 out_motion;
layout(set=1,binding=0) uniform sampler2D scene_color;
layout(set=1,binding=1) uniform sampler2D scene_depth;
layout(push_constant) uniform DrawData {
    mat4 local_to_camera_relative;
    vec4 geometry;
    vec4 elevation_uv;
    vec4 material;
    vec4 debug;
} draw;

// Exact convex volume of the beveled mesh in dungeon_mesh.c, in metres.
const float R=.038, H=.046, B=.004, SQRT2=1.41421356237;
const vec4 planes[8]=vec4[8](
    vec4(1,0,0,R),vec4(0,1,0,R),vec4(-1,-1,0,0),
    vec4(0,0,1,H),vec4(0,0,-1,0),
    vec4(1,0,1,R+H-B),vec4(0,1,1,R+H-B),
    vec4(-1,-1,SQRT2,SQRT2*(H-B)));

float fresnel_dielectric(float cos_i, float eta_i, float eta_t) {
    cos_i=clamp(cos_i,0.0,1.0);
    float sin_t2=pow(eta_i/eta_t,2.0)*(1.0-cos_i*cos_i);
    if (sin_t2>=1.0) return 1.0;
    float cos_t=sqrt(1.0-sin_t2);
    float rs=(eta_i*cos_i-eta_t*cos_t)/max(eta_i*cos_i+eta_t*cos_t,1e-5);
    float rp=(eta_t*cos_i-eta_i*cos_t)/max(eta_t*cos_i+eta_i*cos_t,1e-5);
    return .5*(rs*rs+rp*rp);
}

bool project_scene(vec3 p, out vec2 screen, out float gap) {
    vec4 clip=frame.view_projection*vec4(p,1);
    if (clip.w<=frame.near_plane) return false;
    screen=clip.xy/clip.w*.5+.5;
    if (any(lessThan(screen,vec2(.001))) || any(greaterThan(screen,vec2(.999)))) return false;
    float depth=textureLod(scene_depth,screen,0).r;
    if (depth<=0.0) return false;
    gap=linear_view_depth(clip.z/clip.w)-linear_view_depth(depth);
    return true;
}

// Intersect the outgoing optical ray with the captured scene depth. Small
// initial steps resolve the board immediately behind the glass; longer steps
// can reach adjacent door/wall surfaces. A binary refinement avoids swimming.
bool trace_scene(vec3 origin, vec3 ray, out vec2 hit_uv) {
    float previous_t=0.0, t=.001, step_size=.0015;
    for (int step=0;step<36;++step) {
        vec2 at; float gap;
        if (!project_scene(origin+ray*t,at,gap)) return false;
        if (gap>=0.0) {
            float low=previous_t,high=t;
            for (int refine=0;refine<6;++refine) {
                float mid=(low+high)*.5;
                vec2 probe; float g;
                if (!project_scene(origin+ray*mid,probe,g)) return false;
                if (g>=0.0) high=mid; else low=mid;
            }
            if (!project_scene(origin+ray*high,hit_uv,gap)) return false;
            // Reject foreground depth discontinuities instead of dragging
            // unrelated silhouettes into the glass.
            return gap<.04;
        }
        previous_t=t;
        t+=step_size;
        step_size*=1.22;
    }
    return false;
}
vec3 background(vec2 screen, float haze) {
    vec2 pixel=1.0/vec2(textureSize(scene_color,0));
    float centre_depth=linear_view_depth(textureLod(scene_depth,screen,0).r);
    vec3 color=textureLod(scene_color,screen,0).rgb*.5;
    float weight=.5;
    const vec2 taps[4]=vec2[4](vec2(-1,0),vec2(1,0),vec2(0,-1),vec2(0,1));
    for (int i=0;i<4;++i) {
        vec2 sample_uv=screen+taps[i]*pixel*(.45+haze*1.4);
        float depth=linear_view_depth(textureLod(scene_depth,sample_uv,0).r);
        float w=abs(depth-centre_depth)<.035 ? .125 : 0.0;
        color+=textureLod(scene_color,sample_uv,0).rgb*w;
        weight+=w;
    }
    return color/weight;
}
vec3 offscreen_reflection(vec3 ray, float roughness) {
    // Indoor environment intensity is small; nearby torch glints are evaluated
    // directly below. The cube only fills rays unavailable in screen space.
    return textureLod(env_specular,ray,roughness*env.env_params.w).rgb*
           env.env_params.z*frame.point_light_options.z*.08;
}

float lens_width(float y,int kind) {
    const float radius=.12,aperture=.042,edge=.004;
    float root=sqrt(max(radius*radius-y*y,0.0));
    return edge+(kind==1 ? root-sqrt(radius*radius-aperture*aperture) : radius-root);
}
bool optical_exit(vec3 point,vec3 ray,int kind,out float nearest,out vec3 exit_normal) {
    nearest=1e5;exit_normal=vec3(0,0,1);
    if(kind==0) {
        for(int face=0;face<8;++face) {
            float denom=dot(planes[face].xyz,ray);
            if(denom<=.00001)continue;
            float t=(planes[face].w-dot(planes[face].xyz,point))/denom;
            if(t>.000001&&t<nearest){nearest=t;exit_normal=normalize(planes[face].xyz);}
        }
    } else {
        const float radius=.12,aperture=.042,edge=.004;
        float center=kind==1 ? sqrt(radius*radius-aperture*aperture)-edge : radius+edge;
        for(int side=-1;side<=1;side+=2) {
            float cx=float(side)*(kind==1 ? -center : center);
            vec2 rel=point.xy-vec2(cx,0);
            float a=dot(ray.xy,ray.xy),b=dot(rel,ray.xy);
            float disc=b*b-a*(dot(rel,rel)-radius*radius);
            if(a>.000001&&disc>=0.0)for(int root=-1;root<=1;root+=2) {
                float t=(-b+float(root)*sqrt(disc))/a;
                vec3 hit=point+ray*t;
                vec3 n=vec3((hit.xy-vec2(cx,0))/radius,0);
                if(kind==2)n=-n;
                if(t>.000001&&t<nearest&&abs(hit.y)<=aperture+.00001&&hit.x*side>=0.0&&
                   abs(abs(hit.x)-lens_width(hit.y,kind))<.0001&&hit.z>=-.00001&&hit.z<=H+.00001&&dot(n,ray)>0.0) {
                    nearest=t;exit_normal=normalize(n);
                }
            }
            if(abs(ray.y)>.000001) {
                float t=(float(side)*aperture-point.y)/ray.y;
                vec3 hit=point+ray*t;
                if(t>.000001&&t<nearest&&abs(hit.x)<=lens_width(aperture,kind)&&hit.z>=0.0&&hit.z<=H&&ray.y*side>0.0) {
                    nearest=t;exit_normal=vec3(0,side,0);
                }
            }
        }
        if(abs(ray.z)>.000001) {
            float t=((ray.z>0.0 ? H : 0.0)-point.z)/ray.z;
            vec3 hit=point+ray*t;
            if(t>.000001&&t<nearest&&abs(hit.y)<=aperture&&abs(hit.x)<=lens_width(hit.y,kind)) {
                nearest=t;exit_normal=vec3(0,0,sign(ray.z));
            }
        }
    }
    return nearest<1.0;
}

float refracted_channel(vec3 incident, vec3 entry_normal, float ior, int channel,
                        float haze, out float distance_in_glass) {
    vec3 ray=refract(incident,entry_normal,1.0/ior);
    vec3 point=local_position+ray*.00002;
    float throughput=1.0-fresnel_dielectric(dot(-incident,entry_normal),1.0,ior);
    distance_in_glass=0.0;
    mat3 rotation=mat3(draw.local_to_camera_relative);
    for (int bounce=0;bounce<6;++bounce) {
        float nearest;vec3 exit_normal;
        if(!optical_exit(point,ray,int(draw.material.w+.5),nearest,exit_normal))break;
        point+=ray*nearest;
        distance_in_glass+=nearest;
        vec3 outside=refract(ray,-exit_normal,ior);
        if (dot(outside,outside)<.00001) {
            ray=reflect(ray,exit_normal);
            point+=ray*.00002;
            continue;
        }
        throughput*=1.0-fresnel_dielectric(dot(ray,exit_normal),ior,1.0);
        vec3 world_ray=normalize(rotation*outside);
        vec3 world_point=(draw.local_to_camera_relative*vec4(point,1)).xyz;
        vec2 screen;
        vec3 color;
        if (trace_scene(world_point+world_ray*.0001,world_ray,screen)) color=background(screen,haze);
        else color=offscreen_reflection(world_ray,.18);
        // Very slight warm absorption and neutral volume scattering, both
        // proportional to actual optical path length rather than white alpha.
        float extinction=vec3(.35,.40,.55)[channel]+.8;
        return color[channel]*throughput*exp(-extinction*distance_in_glass);
    }
    return offscreen_reflection(normalize(rotation*ray),.2)[channel]*throughput;
}

void main() {
    vec3 N=normalize(normal), V=normalize(-camera_relative_position);
    // Only entry faces shade: back faces are exit interfaces handled above.
    if (dot(N,V)<=0.0) discard;
    mat3 world_to_local=transpose(mat3(draw.local_to_camera_relative));
    vec3 incident=normalize(world_to_local*(-V));
    vec3 entry_normal=normalize(world_to_local*N);
    float dust=clamp(.28+dungeon_fbm(local_position.xy*150.0)*.8,.05,.6);
    float flecks=smoothstep(.20,.38,dungeon_noise(local_position.xy*1800.0));
    float haze=dust+flecks*.15;
    float distance_r,distance_g,distance_b;
    vec3 transmitted;
    transmitted.r=refracted_channel(incident,entry_normal,1.507,0,haze,distance_r);
    transmitted.g=refracted_channel(incident,entry_normal,1.513,1,haze,distance_g);
    transmitted.b=refracted_channel(incident,entry_normal,1.522,2,haze,distance_b);

    vec3 reflected_ray=reflect(-V,N);
    vec2 reflected_uv;
    vec3 reflected=trace_scene(camera_relative_position+N*.001,reflected_ray,reflected_uv)
        ? background(reflected_uv,haze) : offscreen_reflection(reflected_ray,.16+dust*.1);
    float F=fresnel_dielectric(dot(N,V),1.0,1.513);
    vec3 glints=vec3(0), diffuse=vec3(.012);
    float roughness=.13+dust*.16;
    for (int i=0;i<clamp(int(frame.point_light_options.x+.5),0,16);++i) {
        vec3 delta=frame.point_light_position_radius[i].xyz-camera_relative_position;
        float d2=max(dot(delta,delta),.001);
        vec3 L=delta*inversesqrt(d2);
        float reach=sqrt(d2)/max(frame.point_light_position_radius[i].w,.001);
        float window=max(1.0-pow(reach,4.0),0.0);
        vec3 radiance=frame.point_light_color_intensity[i].rgb*
            frame.point_light_color_intensity[i].w*window*window/d2;
        float NoL=max(dot(N,L),0.0);
        glints+=ue_default_lit_bxdf(vec3(0),vec3(.042),roughness,N,V,L).specular*radiance*NoL;
        diffuse+=radiance*NoL/PBR_PI;
    }
    float length_glass=(distance_r+distance_g+distance_b)/3.0;
    float scatter=1.0-exp(-.8*length_glass);
    // Dust occupies only a small surface fraction. White volume scattering
    // is weak and thickness-dependent; the background stays dominant.
    float surface_dust=.018*dust+.025*flecks;
    vec3 color=transmitted+reflected*F+glints+
               diffuse*vec3(.75,.73,.68)*(scatter*.12+surface_dust);
    // Distance to the second closest hull plane measures a physical edge:
    // one plane contains this fragment; a second meets it along the rim.
    // This outlines actual facet edges, never the mesh's triangle diagonals.
    float nearest=1e3,second=1e3;
    if(draw.material.w<.5) {
        for (int face=0;face<8;++face) {
            float distance=abs(dot(planes[face].xyz,local_position)-planes[face].w)/length(planes[face].xyz);
            if (distance<nearest) { second=nearest; nearest=distance; }
            else second=min(second,distance);
        }
    } else {
        float outline=min(abs(abs(local_position.x)-lens_width(local_position.y,int(draw.material.w+.5))),
                          abs(abs(local_position.y)-.042));
        float cap=min(abs(local_position.z),abs(local_position.z-H));
        second=max(outline,cap);
    }
    float aa=max(fwidth(second),.0001);
    float edge=1.0-smoothstep(.0006-aa,.0013+aa,second);
    // Denser white at the edges, with some background still visible through it.
    vec3 edge_color=vec3(.68,.69,.67)+diffuse*.055;
    color=mix(color,edge_color,edge*.62);
    out_color=vec4(max(color,vec3(0)),1.0);
    out_motion=vec2(2.0);
}
