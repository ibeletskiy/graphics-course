#pragma once

#include <etna/ComputePipeline.hpp>
#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <glm/glm.hpp>

class Sky
{
public:
  Sky();

  bool drawGui();
  void bake();
  etna::ImageBinding getBinding() const;
  glm::vec3 sunDir() const;
  glm::vec3 sunLight() const;

private:
  etna::Image image;
  etna::Sampler sampler;
  etna::ComputePipeline pipeline;

  float sunAngle = -125.0f;
  float sunHeight = 12.0f;
  float sunPower = 6.0f;
  glm::vec3 sunColor{1.0f, 0.68f, 0.38f};

  float coverage = 0.58f;
  float density = 0.04f;
  float scale = 0.007f;
  float minHeight = 120.0f;
  float maxHeight = 220.0f;
  int steps = 32;
  int shadowSteps = 6;
  bool dirty = true;

  static constexpr uint32_t side = 2048;
  static constexpr uint32_t rowsPerBatch = 32;
};
