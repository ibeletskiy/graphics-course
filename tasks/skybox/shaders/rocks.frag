#version 450
#extension GL_GOOGLE_include_directive : require
#include "gbuffer.glsl"

layout(early_fragment_tests) in;

layout(binding = 1) uniform sampler2D albedoMap;
layout(binding = 2) uniform sampler2D materialMap;
layout(binding = 3) uniform sampler2D normalMap;

layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec3 tint;

layout(location = 0) out vec4 outBase;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outMaterial;

vec3 surfaceNormal()
{
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
    float handedness = dot(b, rawB) < 0.0 ? -1.0 : 1.0;
    b *= handedness;

    n = normalize(mat3(t, b, n) * bump);
  }

  return n;
}

void main()
{
  vec3 base = texture(albedoMap, uv).rgb * tint;
  vec3 arm = texture(materialMap, uv).rgb;
  float roughness = max(arm.g, 0.08);
  vec3 n = surfaceNormal();

  outBase = vec4(base, 1);
  outNormal = vec4(packNormal(n), 0, 1);
  outMaterial = vec4(roughness, arm.b, arm.r, 0);
}
