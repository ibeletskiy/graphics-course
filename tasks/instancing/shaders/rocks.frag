#version 450

layout(binding = 1) uniform sampler2D albedoMap;
layout(binding = 2) uniform sampler2D materialMap;
layout(binding = 3) uniform sampler2D normalMap;

layout(push_constant) uniform Params {
    mat4 viewProj;
    vec4 eye;
    vec4 light;
    vec4 lightColor;
} params;

layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec3 tint;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265;

vec3 surfaceNormal() {
    vec3 n = normalize(worldNormal);

    vec3 dx = dFdx(worldPos);
    vec3 dy = dFdy(worldPos);
    vec2 tx = dFdx(uv);
    vec2 ty = dFdy(uv);
    float det = tx.x * ty.y - tx.y * ty.x;

    vec3 bump = texture(normalMap, uv).xyz * 2.0 - 1.0;
    if (abs(det) > 1e-8) {
        vec3 t = dx * ty.y - dy * tx.y;
        t = normalize((t - n * dot(n, t)) * sign(det));

        vec3 rawB = (dy * tx.x - dx * ty.x) / det;
        vec3 b = cross(n, t);
        b *= sign(dot(b, rawB));

        n = normalize(mat3(t, b, n) * bump);
    }

    return n;
}

vec3 lighting(vec3 base, float roughness, float occlusion, vec3 n) {
    vec3 v = normalize(params.eye.xyz - worldPos);
    vec3 l = normalize(params.light.xyz + vec3(0, 0.00001, 0));
    vec3 h = normalize(v + l);

    float nv = max(dot(n, v), 0.001);
    float nl = max(dot(n, l), 0.0);
    float nh = max(dot(n, h), 0.0);
    float vh = max(dot(v, h), 0.0);

    float a = roughness * roughness;
    float a2 = a * a;
    float d = nh * nh * (a2 - 1.0) + 1.0;
    float distribution = a2 / (PI * d * d);

    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float geometry = nv / (nv * (1.0 - k) + k) * nl;
    geometry /= nl * (1.0 - k) + k;

    vec3 fresnel = vec3(0.04) + vec3(0.96) * pow(1.0 - vh, 5.0);
    vec3 specular = distribution * geometry * fresnel / max(4.0 * nv * nl, 0.001);
    vec3 diffuse = (1.0 - fresnel) * base / PI;

    vec3 color = (diffuse + specular) * nl * params.light.w;
    color *= params.lightColor.rgb;
    color += base * params.lightColor.w * occlusion;

    return color;
}

void main() {
    vec3 base = texture(albedoMap, uv).rgb * tint;
    vec3 arm = texture(materialMap, uv).rgb;
    float roughness = max(arm.g, 0.08);

    vec3 n = surfaceNormal();
    vec3 color = lighting(base, roughness, arm.r, n);

    color = vec3(1.0) - exp(-color * params.eye.w);
    outColor = vec4(color, 1);
}
