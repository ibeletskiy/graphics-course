#version 450

layout(location = 0) out vec4 color;
layout(location = 1) out vec4 material;
layout(push_constant) uniform Params {
    vec2 resolution;
    float time;
} params;

void main() {
    const float trackWidth = 0.028;
    const float padRadius = 0.045;
    const float ringWidth = 0.015;
    const float pulseSpeed = 0.22;
    const float pulseWidth = 0.18;
    vec2 uv = gl_FragCoord.xy / params.resolution * 4.0;
    vec2 cell = floor(uv);
    vec2 p = fract(uv) - 0.5;
    if (mod(cell.x + cell.y, 2.0) > 0.5)
        p.x = -p.x;
    if (mod(cell.y, 2.0) > 0.5)
        p = p.yx;

    vec2 points[5] = vec2[5](vec2(-0.36, -0.28), vec2(-0.10, -0.28),
        vec2(0.10, -0.08), vec2(0.10, 0.28), vec2(0.36, 0.28));
    float dist = 10.0;
    float along = 0.0;
    float total = 0.0;
    for (int i = 0; i < 4; ++i) {
        vec2 ab = points[i + 1] - points[i];
        float len = length(ab);
        float t = clamp(dot(p - points[i], ab) / dot(ab, ab), 0.0, 1.0);
        float d = length(p - points[i] - ab * t);
        if (d < dist) {
            dist = d;
            along = total + t * len;
        }
        total += len;
    }

    float aa = 4.0 / min(params.resolution.x, params.resolution.y);
    float track = 1.0 - smoothstep(trackWidth - aa, trackWidth + aa, dist);
    float pad = min(length(p - points[0]), length(p - points[4]));
    float ring = 1.0 - smoothstep(ringWidth - aa, ringWidth + aa, abs(pad - padRadius));
    float head = mod(params.time * pulseSpeed + dot(cell, vec2(0.37, 0.61)), total + 0.6) - 0.3;
    float signal = exp(-pow((along - head) / pulseWidth, 2.0));
    float conductor = max(track, ring);
    vec3 col = mix(vec3(0.018, 0.095, 0.075), vec3(0.30, 0.39, 0.16), conductor);
    col = mix(col, vec3(0.65, 1.0, 0.80), signal * track);
    float glow = signal * (1.0 - smoothstep(trackWidth, 0.055, dist));
    color = vec4(col, glow);
    material = vec4(mix(0.80, 0.25, conductor), mix(0.0, 0.90, conductor), 0.0, 1.0);
}
