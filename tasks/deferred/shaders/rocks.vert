#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;

struct Instance {
    mat4 model;
    vec4 color;
};

layout(binding = 0, std430) readonly buffer Instances {
    Instance instances[];
};

layout(push_constant) uniform Params {
    mat4 viewProj;
    vec4 eye;
    vec4 light;
    vec4 lightColor;
} params;

layout(location = 0) out vec3 worldPos;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec2 uv;
layout(location = 3) out vec3 tint;

void main() {
    Instance instance = instances[gl_InstanceIndex];

    worldPos = (instance.model * vec4(position, 1)).xyz;
    worldNormal = normalize(mat3(instance.model) * normal);

    uv = texcoord;
    tint = instance.color.rgb;

    gl_Position = params.viewProj * vec4(worldPos, 1);
}
