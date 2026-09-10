#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
layout(location=0) in vec2 uv;
layout(location=0) out vec4 out_color;
layout(location=1) out vec2 out_motion;
layout(push_constant) uniform DrawData {
    mat4 local_to_camera_relative;vec4 geometry;vec4 elevation_uv;vec4 material;vec4 debug;
} draw;
void main(){
    float across=abs(uv.y*2.0-1.0);
    float alpha=exp(-across*across*3.0)*(1.0-smoothstep(.8,1.0,across))*.45;
    if(alpha<.005)discard;
    out_color=vec4(draw.geometry.rgb,alpha);
    out_motion=vec2(2.0);
}
