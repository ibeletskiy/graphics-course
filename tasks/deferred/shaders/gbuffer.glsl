vec2 signNotZero(vec2 v)
{
  float x = v.x < 0.0 ? -1.0 : 1.0;
  float y = v.y < 0.0 ? -1.0 : 1.0;
  return vec2(x, y);
}

vec2 packNormal(vec3 n)
{
  n /= abs(n.x) + abs(n.y) + abs(n.z);
  if (n.z < 0.0) {
    n.xy = (1.0 - abs(n.yx)) * signNotZero(n.xy);
  }

  return n.xy;
}

vec3 unpackNormal(vec2 p)
{
  vec3 n = vec3(p, 1.0 - abs(p.x) - abs(p.y));
  if (n.z < 0.0) {
    n.xy = (1.0 - abs(n.yx)) * signNotZero(n.xy);
  }

  return normalize(n);
}

vec3 getPosition(vec2 uv, float depth, mat4 invViewProj)
{
  vec4 point = invViewProj * vec4(uv * 2.0 - 1.0, depth, 1.0);
  return point.xyz / point.w;
}
