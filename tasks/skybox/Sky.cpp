#include "Sky.hpp"

#include <cmath>

#include <etna/Etna.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/OneShotCmdMgr.hpp>
#include <etna/PipelineManager.hpp>
#include <imgui.h>

Sky::Sky()
{
  auto& context = etna::get_context();
  image = context.createImage(etna::Image::CreateInfo{
    .extent = {side, side, 1},
    .name = "sky_cubemap",
    .format = vk::Format::eR16G16B16A16Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
    .layers = 6,
    .flags = vk::ImageCreateFlagBits::eCubeCompatible,
  });

  sampler = etna::Sampler(etna::Sampler::CreateInfo{.name = "sky_sampler"});
  etna::create_program("sky", {SKYBOX_SHADERS_ROOT "sky.comp.spv"});
  pipeline = context.getPipelineManager().createComputePipeline("sky", {});
  bake();
}

glm::vec3 Sky::sunDir() const
{
  const float angle = glm::radians(sunAngle);
  const float height = glm::radians(sunHeight);
  const float radius = std::cos(height);

  return {radius * std::cos(angle), std::sin(height), radius * std::sin(angle)};
}

glm::vec3 Sky::sunLight() const
{
  return sunColor * sunPower;
}

bool Sky::drawGui()
{
  bool changed = false;
  changed |= ImGui::SliderFloat("Sun angle", &sunAngle, -180.0f, 180.0f, "%.0f deg");
  changed |= ImGui::SliderFloat("Sun height", &sunHeight, 1.0f, 80.0f, "%.1f deg");
  changed |= ImGui::ColorEdit3("Sun color", &sunColor.x);
  changed |= ImGui::SliderFloat("Sun power", &sunPower, 0.0f, 30.0f);

  changed |= ImGui::SliderFloat("Cloud coverage", &coverage, 0.0f, 1.0f);
  changed |= ImGui::SliderFloat("Cloud density", &density, 0.005f, 0.15f, "%.3f");
  changed |= ImGui::SliderFloat("Cloud scale", &scale, 0.003f, 0.05f, "%.3f");
  changed |= ImGui::SliderFloat("Layer bottom", &minHeight, 10.0f, maxHeight - 10.0f, "%.0f");
  changed |= ImGui::SliderFloat("Layer top", &maxHeight, minHeight + 10.0f, 600.0f, "%.0f");
  changed |= ImGui::SliderInt("Ray steps", &steps, 8, 96);
  changed |= ImGui::SliderInt("Shadow steps", &shadowSteps, 1, 16);

  dirty |= changed;
  return dirty && !ImGui::IsAnyItemActive();
}

etna::ImageBinding Sky::getBinding() const
{
  const etna::Image::ViewParams view{.layerCount = 6, .type = vk::ImageViewType::eCube};
  return image.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, view);
}

void Sky::bake()
{
  auto& context = etna::get_context();
  ETNA_CHECK_VK_RESULT(context.getDevice().waitIdle());
  auto commands = context.createOneShotCmdMgr();

  struct Params
  {
    glm::vec4 sun;
    glm::vec4 color;
    glm::vec4 clouds;
    glm::vec4 layer;
    glm::ivec4 work;
  } params{
    glm::vec4(sunDir(), sunPower),
    glm::vec4(sunColor, 0.0f),
    {coverage, density, scale, 0.0f},
    {minHeight, maxHeight, 4000.0f, 0.0f},
    {0, 0, steps, shadowSteps},
  };

  const auto layout = pipeline.getVkPipelineLayout();
  const auto descriptorLayout = etna::get_shader_program("sky").getDescriptorLayoutId(0);
  const etna::Image::ViewParams view{.layerCount = 6, .type = vk::ImageViewType::e2DArray};
  const auto binding = image.genBinding({}, vk::ImageLayout::eGeneral, view);
  etna::DescriptorSet set;

  for (uint32_t face = 0; face < 6; ++face) {
    for (uint32_t row = 0; row < side; row += rowsPerBatch) {
      auto cmd = commands->start();
      ETNA_CHECK_VK_RESULT(cmd.begin(vk::CommandBufferBeginInfo{}));

      if (face == 0 && row == 0) {
        set = etna::create_descriptor_set(descriptorLayout, cmd, {{0, binding}}, BarrierBehavior::eGenerateBarriers);
        etna::flush_barriers(cmd);
      }

      params.work.x = static_cast<int>(face);
      params.work.y = static_cast<int>(row);
      cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getVkPipeline());
      cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout, 0, {set.getVkSet()}, {});
      cmd.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(params), &params);
      cmd.dispatch(side / 8, rowsPerBatch / 8, 1);

      if (face == 5 && row + rowsPerBatch == side) {
        const auto stage = vk::PipelineStageFlagBits2::eFragmentShader;
        const auto access = vk::AccessFlagBits2::eShaderSampledRead;
        const auto readLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        etna::set_state(cmd, image.get(), stage, access, readLayout, vk::ImageAspectFlagBits::eColor);
        etna::flush_barriers(cmd);
      }

      ETNA_CHECK_VK_RESULT(cmd.end());
      commands->submitAndWait(cmd);
    }
  }

  dirty = false;
}
