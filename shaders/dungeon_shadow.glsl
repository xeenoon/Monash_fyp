#ifndef DUNGEON_SHADOW_GLSL
#define DUNGEON_SHADOW_GLSL
#include "shader_dump.glsl"
struct PointShadowNode {
    vec3 minimum; uint end;
    vec3 maximum; uint leaf;
    vec4 a, b, c;
};
layout(std430, set=0, binding=6) readonly buffer PointShadowGeometry {
    PointShadowNode shadow_nodes[];
};
bool shadow_box(vec3 o, vec3 d, vec3 lower, vec3 upper, float limit) {
    float near_t=0.0, far_t=limit;
    for(int k=0;k<3;k++) {
        if(abs(d[k])<1e-8) {
            if(o[k]<lower[k] || o[k]>upper[k]) return false;
        } else {
            float a=(lower[k]-o[k])/d[k], b=(upper[k]-o[k])/d[k];
            near_t=max(near_t,min(a,b)); far_t=min(far_t,max(a,b));
        }
    }
    return far_t>=near_t;
}
/* Stackless traversal has no fixed geometry budget or stack overflow path. */
int dungeon_shadow_trace(vec3 origin, vec3 direction, inout float distance,
                         out vec3 hit_normal, bool any_hit) {
    int hit=-1;
    uint count=uint(frame.point_shadow_origin.w);
    for(uint i=0u;i<count;) {
        PointShadowNode n=shadow_nodes[i]; uint index=i;
        if(!shadow_box(origin,direction,n.minimum,n.maximum,distance)) { i=n.end; continue; }
        ++i;
        if(n.leaf==0u) continue;
        vec3 e1=n.b.xyz-n.a.xyz, e2=n.c.xyz-n.a.xyz;
        vec3 p=cross(direction,e2); float det=dot(e1,p);
        if(abs(det)<1e-8) continue;
        vec3 v=origin-n.a.xyz; float u=dot(v,p)/det;
        if(u<0.0 || u>1.0) continue;
        vec3 q=cross(v,e1); float w=dot(direction,q)/det;
        if(w<0.0 || u+w>1.0) continue;
        float t=dot(e2,q)/det;
        if(t<=0.0001 || t>=distance) continue;
        distance=t; hit=int(index); hit_normal=normalize(cross(e1,e2));
        if(any_hit) return hit;
    }
    return hit;
}
float point_light_visibility(vec3 surface, vec3 geometric_normal, vec3 light,
                             int light_index, vec3 unshadowed) {
    vec3 n=normalize(geometric_normal);
    vec3 delta=light-surface;
    if(dot(n,delta)<0.0) n=-n;
    vec3 origin=surface+n*0.003;
    delta=light-origin;
    float length_to_light=length(delta);
    if(length_to_light<0.006) return 1.0;
    float distance=length_to_light-0.003;
    vec3 hit_normal=vec3(0);
    int hit=dungeon_shadow_trace(origin+frame.point_shadow_origin.xyz,
        delta/length_to_light,distance,hit_normal,true);
    float visibility=hit<0 ? 1.0 : 0.0;
    /* One selected light per pixel keeps the dump within its layer budget.
       point_light_options.w is the diagnostic light index (default player). */
    if(light_index==int(frame.point_light_options.w))
        shader_dump(DUMP_SHADER_POINT_SHADOW,vec4(light_index,hit,hit<0 ? -1.0 : distance,visibility),
            vec4(unshadowed,length_to_light),vec4(surface+frame.point_shadow_origin.xyz,0),
            vec4(n,0),vec4(light+frame.point_shadow_origin.xyz,0));
    return visibility;
}
#endif
