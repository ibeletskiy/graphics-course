#pragma once

#include <array>
#include <vector>
#include <etna/Buffer.hpp>
#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/BlockingTransferHelper.hpp>
#include "FramePacket.hpp"
#include "Rocks.hpp"

class WorldRenderer
{
public:
  WorldRenderer();

  void allocateResources(glm::uvec2 resolution);
  void setupPipelines(vk::Format format);
  void update(const FramePacket& packet);
  void drawGui();
  void renderWorld(vk::CommandBuffer cmd, vk::Image image, vk::ImageView view);

private:
  static constexpr int mapSize = 1024;
  static constexpr std::uint32_t cells = 256;
  static constexpr float worldSize = 800;

  void loadMaterials();
  void createMesh();
  void regenerate();
  float fbm(glm::vec2 p) const;
  float terrainHeight(glm::vec2 p) const;
  etna::Image loadTexture(const char* name, bool srgb);

  std::unique_ptr<etna::OneShotCmdMgr> oneShot;
  etna::BlockingTransferHelper transfer;

  etna::Image heightmap;
  std::vector<float> heights;
  std::array<etna::Image, 12> textures;
  etna::Sampler heightSampler;
  etna::Sampler materialSampler;

  etna::Buffer vertices;
  etna::Buffer indices;
  etna::Image depth;
  etna::GraphicsPipeline pipeline;
  std::unique_ptr<Rocks> rocks;

  glm::uvec2 resolution;
  Camera camera;

  struct PushConstants
  {
    glm::mat4 viewProj{1};
    glm::vec4 eye{0, 0, 0, 1.0f};
    glm::vec4 light{-0.6f, 0.7f, -0.35f, 3.0f};
    glm::vec4 surface{55.0f, 0.18f, 1.0f, 0.32f};
    glm::vec4 bands{6.0f, 43.0f, 61.0f, 4.0f};
  } params;

  int seed = 14;
  int octaves = 6;
  float frequency = 3.3f;
  float persistence = 0.48f;
  float lacunarity = 2.1f;
  float heightScale = 64;
  float coastWidth = 0.40f;
};
