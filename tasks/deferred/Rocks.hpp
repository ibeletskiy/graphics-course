#pragma once

#include <array>
#include <span>
#include <vector>

#include <glm/glm.hpp>
#include <etna/Buffer.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/Image.hpp>
#include <etna/Sampler.hpp>

namespace tinygltf
{
class Model;
}

class Rocks
{
public:
  struct Params
  {
    glm::mat4 viewProj;
    glm::vec4 eye;
    glm::vec4 light;
    glm::vec4 lightColor;
  };

  struct Terrain
  {
    std::span<const float> heights;
    int mapSize;
    float worldSize;
    float heightScale;
    int seed;
  };

  explicit Rocks(std::span<const vk::Format> formats);

  void setupPipelines(std::span<const vk::Format> formats);
  void update(vk::CommandBuffer cmd, const glm::vec3& eye, const glm::mat4& viewProj, const Terrain& terrain);
  void draw(vk::CommandBuffer cmd, const Params& params, bool depthOnly = false);
  void gui();

private:
  struct Vertex
  {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
  };

  struct Instance
  {
    glm::mat4 model;
    glm::vec4 color;
  };

  struct Mesh
  {
    etna::Buffer vertices;
    etna::Buffer indices;
    etna::Buffer instances;
    std::array<etna::Image, 3> textures;
    std::vector<Instance> visible;

    uint32_t indexCount = 0;
    glm::vec3 center{};
    float radius = 0;
  };

  struct View
  {
    glm::vec3 eye;
    std::array<glm::vec4, 6> planes;
  };

  void load(Mesh& mesh, const char* name);
  void loadTextures(Mesh& mesh, const tinygltf::Model& model);
  void generate(const View& view, const Terrain& terrain);
  void generateChunk(glm::ivec2 chunk, const View& view, const Terrain& terrain);
  bool isVisible(glm::vec3 center, float radius, const View& view) const;
  void upload(vk::CommandBuffer cmd);

  std::array<Mesh, 2> meshes;
  etna::GraphicsPipeline pipeline;
  etna::GraphicsPipeline depthPipeline;
  etna::Sampler sampler;

  float density = 0.45f;
  float distance = 220.0f;
  glm::vec2 sizes{1.5f, 5.0f};
};
