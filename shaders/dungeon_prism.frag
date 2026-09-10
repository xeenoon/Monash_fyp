#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "pbr_common.glsl"
#include "dungeon_noise.glsl"

layout(location=0) in vec2 uv;
layout(location=1) in vec3 normal;
layout(location=2) in vec4 tangent;
layout(location=3) in vec3 camera_relative_position;
layout(location=4) in vec4 current_clip;
layout(location=5) in vec4 previous_clip;
layout(location=6) in vec3 local_position;
layout(set=1,binding=0) uniform sampler2D wood_albedo;
layout(location=0) out vec4 out_color;
layout(location=1) out vec2 out_motion;
layout(push_constant) uniform DrawData {
    mat4 local_to_camera_relative;
    vec4 geometry; // tint
    vec4 elevation_uv; // board offset xy, clockwise mesh angle, board width
    vec4 material; // mode: board/crystal/beam/ring/key/emitter, lit, rotating, reserved
    vec4 debug;
} draw;


/* The board and the refracted view share real medieval wood, darkened with
 * grain splits, dirt and old nail heads. Coordinates are board-local metres. */
vec3 backing(vec2 p) {
    vec2 tex=vec2(p.y,p.x)*1.3+vec2(.31,.57);
    vec3 wood=texture(wood_albedo,tex).rgb*vec3(.14,.105,.075);
    float grain=dungeon_fbm(vec2(p.x*9.0,p.y*110.0));
    wood*=.8+grain*.6;
    float row=floor((p.y+.39)/.156);
    float y=mod(p.y+.39,.156)-.078;
    // Wandering lengthwise splits taper from opposing board ends.
    float crack_y=.021*sin(row*3.1)+.0025*sin(p.x*45.0+row*6.0)+
                  .0013*dungeon_noise(vec2(p.x*100.0,row));
    float width=.0008+.002*pow(abs(p.x)/.39,2.0);
    float split=1.0-smoothstep(width,width+.0008,abs(y-crack_y));
    float extent=smoothstep(.01,.16,abs(p.x)+.07*sin(row*5.0));
    wood*=1.0-split*extent*.88;
    float branch=abs(y-crack_y-.018*(p.x+.15));
    wood*=1.0-(1.0-smoothstep(.0003,.0012,branch))*extent*.35;
    float nail=length(vec2(abs(p.x)-.347,y-.005*sin(row*2.0)));
    float rim=1.0-smoothstep(.006,.009,nail);
    vec3 iron=vec3(.021,.016,.011)*(1.0+.4*grain);
    iron+=vec3(.03,.024,.016)*(1.0-smoothstep(.001,.003,abs(nail-.005)));
    wood=mix(wood,iron,rim);
    return max(wood,vec3(.002));
}
float segment_distance(vec2 p, vec2 a, vec2 b) {
    vec2 ab=b-a;
    return length(p-a-ab*clamp(dot(p-a,ab)/max(dot(ab,ab),1e-8),0.0,1.0));
}
vec3 light_surface(vec3 base, vec3 N, vec3 V, float roughness, bool crystal) {
    vec3 color = base * .65; // slight ambient fill in the unlit approach view
    for (int i=0; i<clamp(int(frame.point_light_options.x+.5),0,16); ++i) {
        vec3 delta=frame.point_light_position_radius[i].xyz-camera_relative_position;
        float distance2=max(dot(delta,delta),.001);
        vec3 L=delta*inversesqrt(distance2);
        float reach=sqrt(distance2)/max(frame.point_light_position_radius[i].w,.001);
        float window=max(1.0-pow(reach,4.0),0.0);
        vec3 radiance=frame.point_light_color_intensity[i].rgb *
            frame.point_light_color_intensity[i].w * window*window/distance2;
        UeDefaultLit bxdf=ue_default_lit_bxdf(crystal ? vec3(0) : base,vec3(.04),roughness,N,V,L);
        color+=(bxdf.diffuse+bxdf.specular)*radiance*max(dot(N,L),0.0);
    }
    return color;
}

void main() {
    int mode=int(draw.material.x+.5);
    out_motion=vec2(2.0); // same temporal-history rejection as moving lock hardware
    if (mode>=2) {
        vec2 p=uv*2.0-1.0;
        float alpha=1.0;
        vec3 color=draw.geometry.rgb;
        if (mode==2) {
            // Soft fan: width grows continuously along each traced segment.
            // No laser core; low-contrast haze keeps the light broad and dusty.
            float width=mix(draw.elevation_uv.x,draw.elevation_uv.y,uv.x);
            float cross_beam=abs(p.y)/max(width,.01);
            float haze=.92+.08*dungeon_noise(camera_relative_position.xy*95.0+frame.time*.12);
            alpha=.48*exp(-cross_beam*cross_beam*2.5)*haze;
            alpha*=1.0-smoothstep(.78,1.0,cross_beam);
        } else if (mode==3) {
            float radius=length(p);
            float wear=dungeon_noise(p*17.0)*.013;
            float ring=abs(radius-.86+wear);
            alpha=(1.0-smoothstep(.016,.038,ring))*(.72+.28*dungeon_noise(p*35.0));
            // Distinct solid browsing ring vs broken rotating ring + arrow tips.
            if (draw.material.z>.5) {
                float angle=atan(p.y,p.x);
                alpha*=smoothstep(.12,.3,abs(sin(angle*2.0)));
                float tips=min(segment_distance(p,vec2(.82,0),vec2(.63,.20)),
                               segment_distance(p,vec2(-.82,0),vec2(-.63,-.20)));
                alpha=max(alpha,1.0-smoothstep(.035,.07,tips));
            }
        } else if(mode==6) {
            float edge=max(abs(p.x),abs(p.y));
            if(edge>.97)discard;
            bool border=abs(p.x)>.95||abs(p.y)>.72;
            bool filled=uv.x<=clamp(draw.material.z,0.0,1.0);
            color=border ? vec3(.32,.24,.13) : filled ? vec3(1.0,.73,.32) : vec3(.025,.018,.010);
            alpha=.90;
        } else if (mode==4) {
            // Dark keyhole entrance, not a large touch-anywhere target.
            float head=length(p);
            float stem=segment_distance(p,vec2(0,-.08),vec2(0,-.48));
            float hole=1.0-smoothstep(.16,.19,min(head,stem));
            float rim=1.0-smoothstep(.03,.07,abs(length(p)-.72));
            color=mix(draw.geometry.rgb,vec3(.008,.006,.004),hole);
            alpha=max(hole,rim);
        } else {
            float shaft=segment_distance(p,vec2(-.7,0),vec2(.6,0));
            float arrow=min(segment_distance(p,vec2(.6,0),vec2(.18,.4)),
                            segment_distance(p,vec2(.6,0),vec2(.18,-.4)));
            alpha=1.0-smoothstep(.06,.12,min(shaft,arrow));
        }
        if (alpha<.01) discard;
        out_color=vec4(color,alpha);
        return;
    }
    vec3 V=normalize(-camera_relative_position), N=normalize(normal);
    if (dot(N,V)<0.0) N=-N;
    if (mode==0) {
        vec2 board=local_position.xy*draw.elevation_uv.w;
        vec3 base=backing(board);
        // Fine grain relief catches grazing torchlight without geometry noise.
        float relief=dungeon_noise(vec2(board.x*12.0,board.y*150.0));
        vec3 T=normalize(tangent.xyz), bitangent=normalize(cross(N,T));
        N=normalize(N+T*dFdx(relief)*.3+bitangent*dFdy(relief)*.3);
        out_color=vec4(light_surface(base,N,V,.85,false),1);
        return;
    }
    out_color=vec4(1,0,1,1); // invalid material mode
}
