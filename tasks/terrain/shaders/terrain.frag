#version 450

layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec2 mapUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D heightmap;

layout(set = 0, binding = 1) uniform sampler2D sandAlbedo;
layout(set = 0, binding = 2) uniform sampler2D sandRoughness;
layout(set = 0, binding = 3) uniform sampler2D sandHeight;

layout(set = 0, binding = 4) uniform sampler2D grassAlbedo;
layout(set = 0, binding = 5) uniform sampler2D grassRoughness;
layout(set = 0, binding = 6) uniform sampler2D grassHeight;

layout(set = 0, binding = 7) uniform sampler2D rockAlbedo;
layout(set = 0, binding = 8) uniform sampler2D rockRoughness;
layout(set = 0, binding = 9) uniform sampler2D rockHeight;

layout(set = 0, binding = 10) uniform sampler2D snowAlbedo;
layout(set = 0, binding = 11) uniform sampler2D snowRoughness;
layout(set = 0, binding = 12) uniform sampler2D snowHeight;

layout(push_constant) uniform Params
{
  mat4 viewProj;
  vec4 eye;
  vec4 light;
  vec4 surface;
  vec4 bands;
} p;

const float PI = 3.14159265359;
const float WORLD_SIZE = 800.0;

vec3 terrainNormal(vec2 uv)
{
  vec2 texel = 1.0 / vec2(textureSize(heightmap, 0));
  float hL = texture(heightmap, uv - vec2(texel.x, 0)).r;
  float hR = texture(heightmap, uv + vec2(texel.x, 0)).r;
  float hU = texture(heightmap, uv - vec2(0, texel.y)).r;
  float hD = texture(heightmap, uv + vec2(0, texel.y)).r;

  return normalize(vec3(hL - hR, 2.0 * WORLD_SIZE * texel.x, hU - hD));
}

vec4 materialWeights(float height)
{
  float grass = smoothstep(p.bands.x - p.bands.w, p.bands.x + p.bands.w, height);
  float rock = smoothstep(p.bands.y - p.bands.w, p.bands.y + p.bands.w, height);
  float snow = smoothstep(p.bands.z - p.bands.w, p.bands.z + p.bands.w, height);

  vec4 weights;
  weights.x = 1.0 - grass;
  weights.y = grass * (1.0 - rock);
  weights.z = grass * rock * (1.0 - snow);
  weights.w = grass * rock * snow;

  return weights;
}

float detail(vec2 uv, vec4 weights)
{
  vec4 heights;
  heights.x = texture(sandHeight, uv).r;
  heights.y = texture(grassHeight, uv).r;
  heights.z = texture(rockHeight, uv).r;
  heights.w = texture(snowHeight, uv).r;

  return dot(heights, weights);
}

vec3 detailedNormal(vec3 N, vec2 uv, vec4 weights)
{
  vec2 dt = 1.0 / vec2(textureSize(sandHeight, 0));
  vec2 dh;
  dh.x = detail(uv + vec2(dt.x, 0), weights) - detail(uv - vec2(dt.x, 0), weights);
  dh.y = detail(uv + vec2(0, dt.y), weights) - detail(uv - vec2(0, dt.y), weights);
  dh *= p.surface.y * p.surface.x / (2.0 * WORLD_SIZE * dt);

  vec3 T = normalize(vec3(1, -N.x / max(N.y, 0.05), 0));
  vec3 B = normalize(cross(T, N));

  return normalize(N - T * dh.x - B * dh.y);
}

vec3 lighting(vec3 N, vec3 albedo, float roughness)
{
  vec3 V = normalize(p.eye.xyz - worldPos);
  vec3 L = normalize(p.light.xyz + vec3(0, 0.00001, 0));
  vec3 H = normalize(V + L);

  float NdotL = max(dot(N, L), 0.0);
  float NdotV = max(dot(N, V), 0.001);
  float NdotH = max(dot(N, H), 0.0);
  float VdotH = max(dot(V, H), 0.0);

  float alpha = roughness * roughness;
  float a2 = alpha * alpha;
  float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
  float D = a2 / max(PI * denom * denom, 0.00001);

  float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
  float G = NdotV / (NdotV * (1.0 - k) + k);
  G *= NdotL / max(NdotL * (1.0 - k) + k, 0.001);

  vec3 F = vec3(0.04) + vec3(0.96) * pow(1.0 - VdotH, 5.0);
  vec3 specular = D * G * F / max(4.0 * NdotV * NdotL, 0.001);
  vec3 diffuse = (vec3(1) - F) * albedo / PI;

  vec3 lightColor = vec3(1.0, 0.94, 0.82) * p.light.w;
  vec3 color = (diffuse + specular) * lightColor * NdotL;
  vec3 ambient = mix(vec3(0.50, 0.62, 0.72), vec3(0.90, 0.94, 1.0), N.y * 0.5 + 0.5);
  color += albedo * p.surface.w * ambient;

  return color;
}

void main()
{
  float height = texture(heightmap, mapUV).r;
  if (height < 0.12) {
    float shallow = smoothstep(-1.7, 0.15, height);
    vec3 ocean = mix(vec3(0.006, 0.025, 0.065), vec3(0.025, 0.13, 0.18), shallow);
    outColor = vec4(ocean, 1);
    return;
  }

  vec3 N = terrainNormal(mapUV);
  vec4 weights = materialWeights(height);
  vec2 uv = mapUV * p.surface.x;

  vec3 albedo = texture(sandAlbedo, uv).rgb * weights.x;
  albedo += texture(grassAlbedo, uv).rgb * weights.y;
  albedo += texture(rockAlbedo, uv).rgb * weights.z;
  albedo += texture(snowAlbedo, uv).rgb * weights.w;

  vec4 roughnessMaps;
  roughnessMaps.x = texture(sandRoughness, uv).r;
  roughnessMaps.y = texture(grassRoughness, uv).r;
  roughnessMaps.z = texture(rockRoughness, uv).r;
  roughnessMaps.w = texture(snowRoughness, uv).r;
  float roughness = dot(roughnessMaps, weights);
  roughness = clamp(roughness * p.surface.z, 0.08, 1.0);

  N = detailedNormal(N, uv, weights);
  vec3 color = lighting(N, albedo, roughness);

  color = vec3(1) - exp(-color * p.eye.w);
  outColor = vec4(color, 1);
}
