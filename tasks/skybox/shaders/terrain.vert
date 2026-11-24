#version 450

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec3 worldPos;
layout(location = 1) out vec2 mapUV;

layout(set = 0, binding = 0) uniform sampler2D heightmap;

layout(push_constant) uniform Params
{
  mat4 viewProj;
  vec4 eye;
  vec4 light;
  vec4 surface;
  vec4 bands;
} p;

const float WORLD_SIZE = 800.0;

void main()
{
  mapUV = inUV;
  float height = textureLod(heightmap, inUV, 0).r;
  vec2 pos = (inUV - 0.5) * WORLD_SIZE;

  worldPos = vec3(pos.x, max(height, 0.0), pos.y);
  gl_Position = p.viewProj * vec4(worldPos, 1);
}
