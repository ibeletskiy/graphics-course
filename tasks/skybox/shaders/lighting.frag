#version 450
#extension GL_GOOGLE_include_directive : require
#include "gbuffer.glsl"

layout(binding = 0) uniform sampler2D baseMap;
layout(binding = 1) uniform sampler2D normalMap;
layout(binding = 2) uniform sampler2D materialMap;
layout(binding = 3) uniform sampler2D depthMap;
layout(binding = 4) uniform samplerCube skyMap;

layout(push_constant) uniform Params {
  mat4 invViewProj;
  vec4 eye;
  vec4 light;
  vec4 settings;
  vec4 color;
} params;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265;

vec3 calcLight(vec3 position, vec3 N, vec3 base, vec3 material)
{
  float roughness = max(material.r, 0.04);
  float metalness = material.g;
  vec3 V = normalize(params.eye.xyz - position);
  vec3 L = normalize(params.light.xyz + vec3(0, 0.00001, 0));
  float NdotV = dot(N, V);
  float NdotL = max(dot(N, L), 0.0);
  if (NdotV <= 0.0 || NdotL <= 0.0) {
    return vec3(0);
  }

  vec3 H = normalize(V + L);
  NdotV = max(NdotV, 0.001);
  float NdotH = clamp(dot(N, H), 0.0, 1.0);
  float VdotH = clamp(dot(V, H), 0.0, 1.0);

  float alpha = roughness * roughness;
  float a2 = alpha * alpha;
  float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
  float D = a2 / (PI * denom * denom);

  float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
  float G = NdotV / (NdotV * (1.0 - k) + k);
  G *= NdotL / (NdotL * (1.0 - k) + k);

  vec3 f0 = mix(vec3(0.04), base, metalness);
  vec3 F = f0 + (1.0 - f0) * pow(1.0 - VdotH, 5.0);
  vec3 specular = D * G * F / max(4.0 * NdotV * NdotL, 0.001);
  vec3 diffuse = (1.0 - F) * (1.0 - metalness) * base / PI;

  vec3 lightColor = params.color.rgb;
  return (diffuse + specular) * lightColor * NdotL;
}

void main()
{
  vec2 uv = gl_FragCoord.xy / vec2(textureSize(depthMap, 0));
  float depth = texture(depthMap, uv).r;
  if (depth >= 1.0) {
    vec3 rayDir = normalize(getPosition(uv, 1.0, params.invViewProj) - params.eye.xyz);
    vec3 color = texture(skyMap, rayDir).rgb;
    outColor = vec4(vec3(1) - exp(-color * params.eye.w), 1);
    return;
  }

  vec3 base = texture(baseMap, uv).rgb;
  vec4 surface = texture(materialMap, uv);
  if (surface.a > 0.5) {
    outColor = vec4(base, 1);
    return;
  }

  vec3 position = getPosition(uv, depth, params.invViewProj);
  vec3 N = unpackNormal(texture(normalMap, uv).rg);
  vec3 material = surface.rgb;
  vec3 color = calcLight(position, N, base, material);

  vec3 ambient = mix(vec3(0.50, 0.62, 0.72), vec3(0.90, 0.94, 1.0), N.y * 0.5 + 0.5);
  color += base * params.settings.x * material.b * ambient;
  color = vec3(1) - exp(-color * params.eye.w);

  outColor = vec4(color, 1.0);
}
