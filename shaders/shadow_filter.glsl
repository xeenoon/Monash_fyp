#ifndef SHADOW_FILTER_GLSL
#define SHADOW_FILTER_GLSL

/* Shared receiver-side CSM filtering. Rotation is keyed to the shadow texel,
   which is stable while the camera moves inside a snapped cascade texel. */
const float SHADOW_TAU = 6.28318530718;
const vec2 shadow_vogel8[8] = vec2[8](
    vec2(.25, 0), vec2(-.31930089, .29248416), vec2(.04891348, -.55687296),
    vec2(.40238643, .52496207), vec2(-.73851585, -.13074535),
    vec2(.69968677, -.44490278), vec2(-.23419666, .87043202), vec2(-.44604915, -.85938364));

struct ShadowResult { float visibility; float receiver_bias; uint cascade; vec3 coordinate;
                      float blocker_depth; float filter_radius; };

vec2 shadow_rotate(vec2 v, vec2 cs) { return vec2(cs.x*v.x + cs.y*v.y, cs.x*v.y - cs.y*v.x); }
float shadow_hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
vec3 shadow_coordinate(uint cascade, vec3 receiver) {
    vec4 c = frame.shadow_view_projection[cascade] * vec4(receiver, 1.0);
    c.xyz /= c.w; return vec3(c.xy * .5 + .5, c.z);
}
float shadow_compare(uint cascade, vec3 c, float radius, uint taps) {
    float angle = shadow_hash(floor(c.xy * frame.shadow_pcss.y)) * SHADOW_TAU;
    vec2 cs = vec2(cos(angle), sin(angle)); vec2 texel = vec2(radius / frame.shadow_pcss.y);
    float v = 0.0;
    for (uint i=0u; i<taps; ++i) v += texture(shadow_map, vec4(c.xy + shadow_rotate(shadow_vogel8[i & 7u], cs)*texel, float(cascade), c.z));
    return v / float(taps);
}
float shadow_pcss(uint cascade, vec3 c, out float blocker_depth, out float filter_radius) {
    float radius_m = frame.shadow_radii[cascade];
    float search_texels = max(1.0, frame.shadow_quality.w * frame.shadow_pcss.y / (2.0 * radius_m));
    float angle = shadow_hash(floor(c.xy * frame.shadow_pcss.y)) * SHADOW_TAU;
    vec2 cs = vec2(cos(angle), sin(angle)); vec2 texel = vec2(search_texels / frame.shadow_pcss.y);
    float sum = 0.0; uint count = 0u;
    for (uint i=0u; i<8u; ++i) { float d = texture(shadow_map_raw, vec3(c.xy + shadow_rotate(shadow_vogel8[i], cs)*texel, float(cascade))).r; if (d < c.z) { sum += d; ++count; } }
    if (count == 0u) { blocker_depth = c.z; filter_radius = 0.5; return 1.0; }
    blocker_depth = sum / float(count);
    /* Orthographic depth span is 8*radius. Convert depth separation to metres. */
    float separation_m = max(0.0, (c.z - blocker_depth) * 8.0 * radius_m);
    float penumbra_m = separation_m * tan(frame.shadow_quality.z);
    filter_radius = clamp(max(.5, penumbra_m * frame.shadow_pcss.y / (2.0 * radius_m)), .5, frame.shadow_pcss.x);
    return shadow_compare(cascade, c, filter_radius, 16u);
}
ShadowResult shadow_evaluate(vec3 position, vec3 geometric_normal) {
    vec3 to_sun = normalize(-frame.sun_direction.xyz);
    float bias = frame.shadow_parameters.x * (1.0 - max(dot(geometric_normal, to_sun), 0.0));
    vec3 receiver = position + geometric_normal * bias;
    ShadowResult r = ShadowResult(1.0, bias, 4u, vec3(0), 1.0, 0.0); vec3 clip = vec3(0);
    for (uint i=0u;i<4u;++i) { vec4 q=frame.shadow_view_projection[i]*vec4(receiver,1); q.xyz/=q.w; vec3 c=vec3(q.xy*.5+.5,q.z); if(all(greaterThanEqual(c,vec3(0)))&&all(lessThanEqual(c,vec3(1)))) { r.cascade=i; r.coordinate=c; clip=q.xyz; break; } }
    if (r.cascade >= 4u) return r;
    if (int(frame.shadow_quality.x) == 0) r.visibility = texture(shadow_map, vec4(r.coordinate.xy,float(r.cascade),r.coordinate.z));
    else if (int(frame.shadow_quality.x) == 1) r.visibility = shadow_compare(r.cascade,r.coordinate,frame.shadow_quality.y,8u);
    else r.visibility = shadow_pcss(r.cascade,r.coordinate,r.blocker_depth,r.filter_radius);
    if (r.cascade < 3u) { vec3 e=clamp((abs(vec3(clip.xy,clip.z*2.-1.))-.8)*5.,0.,1.); float blend=max(e.x,max(e.y,e.z)); if(blend>0.) { vec3 next=shadow_coordinate(r.cascade+1u,receiver); float v=int(frame.shadow_quality.x)==2 ? shadow_pcss(r.cascade+1u,next,r.blocker_depth,r.filter_radius) : (int(frame.shadow_quality.x)==1 ? shadow_compare(r.cascade+1u,next,frame.shadow_quality.y,8u) : texture(shadow_map,vec4(next.xy,float(r.cascade+1u),next.z))); r.visibility=mix(r.visibility,v,blend); } }
    return r;
}
#endif
