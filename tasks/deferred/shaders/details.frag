#version 450
#extension GL_GOOGLE_include_directive : require
#include "gbuffer.glsl"

const int DETAIL_COUNT = 6;

layout(binding = 0) uniform sampler2D baseMap;
layout(binding = 1) uniform sampler2D normalMap;
layout(binding = 2) uniform sampler2D materialMap;
layout(binding = 3) uniform sampler2D depthMap;

layout(binding = 4) uniform Details {
  mat4 invViewProj;
  vec4 strength;
  mat4 boxes[DETAIL_COUNT];
  vec4 layers[DETAIL_COUNT];
} details;

layout(binding = 5) uniform sampler2DArray detailColor;
layout(binding = 6) uniform sampler2DArray detailSurface;

layout(location = 0) out vec4 outBase;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outMaterial;

vec3 detailNormal(mat4 worldToBox, vec3 normal)
{
  mat3 basis = inverse(mat3(worldToBox));
  basis[0] = normalize(basis[0]);
  basis[1] = normalize(basis[1]);
  basis[2] = normalize(basis[2]);
  return normalize(basis * normal);
}

void main()
{
  vec2 uv = gl_FragCoord.xy / vec2(textureSize(depthMap, 0));
  outBase = texture(baseMap, uv);
  outNormal = texture(normalMap, uv);
  outMaterial = texture(materialMap, uv);

  float depth = texture(depthMap, uv).r;
  if (depth >= 1.0) {
    return;
  }

  vec3 position = getPosition(uv, depth, details.invViewProj);
  vec3 normal = unpackNormal(outNormal.xy);
  for (int i = 0; i < DETAIL_COUNT; ++i) {
    vec3 local = (details.boxes[i] * vec4(position, 1)).xyz;
    if (any(greaterThan(abs(local), vec3(1)))) {
      continue;
    }

    vec3 coord = vec3(local.xz * 0.5 + 0.5, details.layers[i].x);
    vec4 color = textureLod(detailColor, coord, 0.0);
    vec4 surface = textureLod(detailSurface, coord, 0.0);
    float alpha = color.a * details.strength.x;

    outBase.rgb = mix(outBase.rgb, color.rgb, alpha * details.strength.y);
    vec3 detail = detailNormal(details.boxes[i], surface.xyz * 2.0 - 1.0);
    normal = normalize(mix(normal, detail, alpha * details.strength.z));
    outMaterial.r = mix(outMaterial.r, surface.a, alpha * details.strength.w);
  }

  outNormal = vec4(packNormal(normal), 0, 1);
}
