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
#include "Sky.hpp"

class WorldRenderer
{
public:
  WorldRenderer();

  void allocateResources(glm::uvec2 resolution);
  void setupPipelines(vk::Format format);
  void update(const FramePacket& packet);
  void drawGui();
  void renderWorld(vk::CommandBuffer cmd, vk::Image image, vk::ImageView view);
  void readTimes();

private:
  static constexpr int mapSize = 1024;
  static constexpr std::uint32_t cells = 256;
  static constexpr float worldSize = 800;

  void loadMaterials();
  void createMesh();
  void regenerate();
  float fbm(glm::vec2 p) const;
  float terrainHeight(glm::vec2 p) const;
  float getHeight(float x, float z) const;
  etna::Image loadTexture(const char* name, bool srgb);
  void createDetails();
  void updateDetails();
  void drawTerrain(vk::CommandBuffer cmd, bool depthOnly);
  void drawGeometry(vk::CommandBuffer cmd, bool depthOnly);
  void drawDetails(vk::CommandBuffer cmd);
  void drawLighting(vk::CommandBuffer cmd, vk::Image image, vk::ImageView view);
  void drawStats();

  std::unique_ptr<etna::OneShotCmdMgr> oneShot;
  etna::BlockingTransferHelper transfer;

  etna::Image heightmap;
  std::vector<float> heights;
  std::array<etna::Image, 12> textures;
  etna::Sampler heightSampler;
  etna::Sampler materialSampler;
  etna::Sampler screenSampler;

  etna::Buffer vertices;
  etna::Buffer indices;
  etna::Image depth;
  std::array<etna::Image, 3> gbuffer;
  std::array<etna::Image, 3> detailGbuffer;
  std::array<vk::Format, 3> gformats;
  std::array<etna::Image, 2> detailTextures;
  etna::Buffer detailBuffer;

  etna::GraphicsPipeline pipeline;
  etna::GraphicsPipeline depthPipeline;
  etna::GraphicsPipeline detailPipeline;
  etna::GraphicsPipeline lightPipeline;
  std::unique_ptr<Rocks> rocks;
  std::unique_ptr<Sky> sky;
  vk::Format targetFormat{};
  bool compact = true;

  vk::UniqueQueryPool timestamps;
  bool hasTimes = false;
  bool lastCompact = true;
  std::array<std::array<double, 4>, 2> passTimes{};
  std::array<double, 2> frameTimes{};
  std::array<std::uint32_t, 2> samples{};

  struct DetailData
  {
    glm::mat4 invViewProj{1};
    glm::vec4 strength{0.9f, 1, 0.65f, 1};
    std::array<glm::mat4, 6> boxes;
    std::array<glm::vec4, 6> layers;
  } details;

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
