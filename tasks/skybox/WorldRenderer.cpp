#include "WorldRenderer.hpp"
#include "Mipmaps.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/type_precision.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/DescriptorSet.hpp>
#include <imgui.h>
#include <stb_image.h>

namespace
{
float randomValue(int x, int y, int seed)
{
  std::uint32_t h = static_cast<std::uint32_t>(x) * 374761393u;
  h += static_cast<std::uint32_t>(y) * 668265263u;
  h += static_cast<std::uint32_t>(seed) * 2246822519u;
  h = (h ^ (h >> 13)) * 1274126177u;

  return static_cast<float>(h ^ (h >> 16)) / 4294967295.0f;
}

float noise(glm::vec2 p, int seed)
{
  const auto cell = glm::ivec2(glm::floor(p));
  glm::vec2 t = glm::fract(p);
  t = t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);

  const float a = glm::mix(randomValue(cell.x, cell.y, seed), randomValue(cell.x + 1, cell.y, seed), t.x);
  const float b = glm::mix(randomValue(cell.x, cell.y + 1, seed), randomValue(cell.x + 1, cell.y + 1, seed), t.x);

  return glm::mix(a, b, t.y);
}

float shoreDistance(glm::vec2 p)
{
  static const glm::vec2 outline[]{ // totally not suspicious shape: wikipedia.org/wiki/Little_Saint_James
    {167, 50}, {177, 60}, {185, 62}, {190, 71}, {199, 84}, {198, 88},
    {181, 102}, {186, 130}, {194, 143}, {211, 158}, {224, 160}, {230, 186},
    {238, 194}, {244, 210}, {241, 217}, {234, 222}, {227, 216}, {213, 215},
    {211, 204}, {200, 195}, {195, 196}, {180, 187}, {177, 181}, {164, 175},
    {156, 174}, {147, 179}, {144, 185}, {137, 183}, {130, 194}, {120, 194},
    {106, 200}, {88, 204}, {83, 207}, {66, 200}, {65, 176}, {74, 138},
    {75, 128}, {69, 121}, {69, 114}, {74, 106}, {87, 100}, {108, 108},
    {142, 110}, {151, 107}, {159, 95}, {164, 80}, {167, 63}, {163, 57},
  };

  constexpr float outlineScale = 100;
  const glm::vec2 outlineCenter{154, 145};
  p = p * outlineScale + outlineCenter;

  float distance = glm::length(p - outline[0]);
  bool inside = false;

  for (std::size_t i = 0; i < std::size(outline); ++i) {
    const auto a = outline[i];
    const auto b = outline[(i + 1) % std::size(outline)];
    const auto edge = b - a;

    const float t = glm::clamp(glm::dot(p - a, edge) / glm::dot(edge, edge), 0.0f, 1.0f);
    distance = std::min(distance, glm::length(p - a - edge * t));

    if ((a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) * edge.x / edge.y) {
      inside = !inside;
    }
  }

  return (inside ? distance : -distance) / outlineScale;
}

float detailAlpha(glm::vec2 p, int layer)
{
  if (layer == 0) {
    const float sole = glm::length((p - glm::vec2(0, -0.27f)) / glm::vec2(0.53f, 0.63f));
    const float heel = glm::length((p - glm::vec2(0, 0.59f)) / glm::vec2(0.40f, 0.25f));
    const float shape = 1.0f - glm::smoothstep(0.91f, 1.0f, std::min(sole, heel));
    const float stripe = std::sin((p.y + std::abs(p.x) * 0.22f) * 36.0f);
    const float tread = 0.45f + 0.55f * glm::smoothstep(-0.15f, 0.15f, stripe);

    return shape * tread;
  }

  const float edge = noise(p * 4.0f + 8.0f, 72) * 0.18f;
  const float shape = 1.0f - glm::smoothstep(0.65f + edge, 0.87f + edge, glm::length(p));
  const float variation = 0.55f + 0.45f * noise(p * 9.0f + 2.0f, 33);

  return shape * variation;
}
}

WorldRenderer::WorldRenderer()
  : oneShot{etna::get_context().createOneShotCmdMgr()}
  , transfer{etna::BlockingTransferHelper::CreateInfo{.stagingSize = 16 * 1024 * 1024}}
{
  static_assert(sizeof(PushConstants) == 128);

  heightSampler = etna::Sampler(etna::Sampler::CreateInfo{.name = "terrain_height_sampler"});
  materialSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .addressMode = vk::SamplerAddressMode::eRepeat, .name = "terrain_material_sampler"});
  screenSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eNearest, .name = "gbuffer_sampler"});

  createMesh();
  regenerate();
  loadMaterials();
  createDetails();

  const auto vertexShader = SKYBOX_SHADERS_ROOT "terrain.vert.spv";
  const auto fragmentShader = SKYBOX_SHADERS_ROOT "terrain.frag.spv";
  const auto fullscreenShader = SKYBOX_SHADERS_ROOT "fullscreen.vert.spv";
  const auto detailShader = SKYBOX_SHADERS_ROOT "details.frag.spv";
  const auto lightingShader = SKYBOX_SHADERS_ROOT "lighting.frag.spv";
  etna::create_program("skybox_terrain", {vertexShader, fragmentShader});
  etna::create_program("skybox_terrain_depth", {vertexShader});
  etna::create_program("skybox_details", {fullscreenShader, detailShader});
  etna::create_program("skybox_lighting", {fullscreenShader, lightingShader});

  auto& ctx = etna::get_context();
  const auto mapped = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
  detailBuffer = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = sizeof(DetailData),
    .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
    .allocationCreate = mapped,
    .name = "detail_data",
  });

  const vk::QueryPoolCreateInfo queryInfo{.queryType = vk::QueryType::eTimestamp, .queryCount = 8};
  timestamps = etna::unwrap_vk_result(ctx.getDevice().createQueryPoolUnique(queryInfo));
  sky = std::make_unique<Sky>();
}

void WorldRenderer::loadMaterials()
{
  const std::array<const char*, 4> names{"sand", "grass", "rock", "snow"};

  for (std::size_t i = 0; i < names.size(); ++i) {
    const std::string base = std::string(names[i]);
    textures[i * 3] = loadTexture((base + "_albedo.jpg").c_str(), true);
    textures[i * 3 + 1] = loadTexture((base + "_roughness.jpg").c_str(), false);
    textures[i * 3 + 2] = loadTexture((base + "_height.jpg").c_str(), false);
  }
}

etna::Image WorldRenderer::loadTexture(const char* name, bool srgb)
{
  const std::string path = std::string(GRAPHICS_COURSE_RESOURCES_ROOT) + "/terrain/" + name;
  int width = 0;
  int height = 0;
  int channels = 0;
  unsigned char* pixels = stbi_load(path.c_str(), &width, &height, &channels, 4);
  const auto mipCount = static_cast<uint32_t>(std::log2(std::max(width, height))) + 1;
  const auto transferUsage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;

  auto image = etna::get_context().createImage(etna::Image::CreateInfo{
    .extent = {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), 1},
    .name = name,
    .format = srgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm,
    .imageUsage = transferUsage | vk::ImageUsageFlagBits::eSampled,
    .mipLevels = mipCount,
  });

  const auto data = reinterpret_cast<const std::byte*>(pixels);
  const auto size = static_cast<std::size_t>(width) * height * 4;
  transfer.uploadImage(*oneShot, image, 0, 0, {data, size});
  stbi_image_free(pixels);
  generateMips(image, *oneShot);

  return image;
}

void WorldRenderer::createMesh()
{
  std::vector<glm::vec2> verts;
  std::vector<std::uint32_t> inds;

  for (std::uint32_t z = 0; z <= cells; ++z) {
    for (std::uint32_t x = 0; x <= cells; ++x) {
      verts.emplace_back(glm::vec2(x, z) / static_cast<float>(cells));
    }
  }

  for (std::uint32_t z = 0; z < cells; ++z) {
    for (std::uint32_t x = 0; x < cells; ++x) {
      const auto a = z * (cells + 1) + x;
      const auto b = a + cells + 1;
      inds.insert(inds.end(), {a, b, a + 1, a + 1, b, b + 1});
    }
  }

  auto& ctx = etna::get_context();

  vertices = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = verts.size() * sizeof(glm::vec2),
    .bufferUsage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .name = "terrain_vertices",
  });

  indices = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = inds.size() * sizeof(std::uint32_t),
    .bufferUsage = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .name = "terrain_indices",
  });

  transfer.uploadBuffer<glm::vec2>(*oneShot, vertices, 0, verts);
  transfer.uploadBuffer<std::uint32_t>(*oneShot, indices, 0, inds);
}

float WorldRenderer::fbm(glm::vec2 p) const
{
  float value = 0;
  float weight = 1;
  float sum = 0;
  glm::vec2 q = p * frequency + glm::vec2(7.2f, 3.6f);

  for (int octave = 0; octave < octaves; ++octave) {
    value += noise(q, seed + octave * 17) * weight;
    sum += weight;
    weight *= persistence;
    q *= lacunarity;
  }

  return value / sum;
}

float WorldRenderer::terrainHeight(glm::vec2 p) const
{
  const float radialMask = glm::smoothstep(0.0f, coastWidth, 1.25f - glm::length(p));
  const float coastMask = glm::smoothstep(0.0f, coastWidth * 0.35f, shoreDistance(p));
  const float value = fbm(p);

  const glm::vec2 west = (p - glm::vec2(-0.30f, -0.05f)) / glm::vec2(0.42f, 0.38f);
  const glm::vec2 north = (p - glm::vec2(-0.04f, -0.48f)) / glm::vec2(0.18f, 0.29f);
  float hills = 0.63f * std::exp(-glm::dot(west, west));
  hills += 0.18f * std::exp(-glm::dot(north, north));

  const float height = heightScale * radialMask * coastMask * (0.11f + 0.55f * value + hills);
  return height - 1.7f;
}

void WorldRenderer::regenerate()
{
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());

  heights.resize(mapSize * mapSize);

  for (int z = 0; z < mapSize; ++z) {
    for (int x = 0; x < mapSize; ++x) {
      const glm::vec2 uv = (glm::vec2(x, z) + 0.5f) / static_cast<float>(mapSize);
      const glm::vec2 p = uv * 2.0f - 1.0f;
      heights[static_cast<std::size_t>(z) * mapSize + x] = terrainHeight(p);
    }
  }

  heightmap = etna::get_context().createImage(etna::Image::CreateInfo{
    .extent = {mapSize, mapSize, 1}, .name = "terrain_heightmap",
    .format = vk::Format::eR32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
  });

  const auto data = reinterpret_cast<const std::byte*>(heights.data());
  transfer.uploadImage(*oneShot, heightmap, 0, 0, {data, heights.size() * sizeof(float)});
}

float WorldRenderer::getHeight(float x, float z) const
{
  glm::vec2 p = (glm::vec2(x, z) / worldSize + 0.5f) * static_cast<float>(mapSize) - 0.5f;
  p = glm::clamp(p, glm::vec2(0), glm::vec2(mapSize - 1));

  const glm::ivec2 a = glm::ivec2(p);
  const glm::ivec2 b = glm::min(a + 1, glm::ivec2(mapSize - 1));
  const glm::vec2 f = glm::fract(p);
  const float top = glm::mix(heights[a.y * mapSize + a.x], heights[a.y * mapSize + b.x], f.x);
  const float bottom = glm::mix(heights[b.y * mapSize + a.x], heights[b.y * mapSize + b.x], f.x);

  return glm::mix(top, bottom, f.y);
}

void WorldRenderer::createDetails()
{
  constexpr int size = 256;
  for (auto& image : detailTextures) {
    image = etna::get_context().createImage(etna::Image::CreateInfo{
      .extent = {size, size, 1},
      .name = "terrain_details",
      .format = vk::Format::eR8G8B8A8Unorm,
      .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
      .layers = 2,
    });
  }

  for (uint32_t layer = 0; layer < 2; ++layer) {
    std::vector<float> alpha(size * size);
    std::vector<glm::u8vec4> colors(size * size);
    std::vector<glm::u8vec4> surface(size * size);
    const glm::vec3 base = layer == 0 ? glm::vec3(0.13f, 0.083f, 0.033f) : glm::vec3(0.24f, 0.145f, 0.065f);

    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        const glm::vec2 p = glm::vec2(x, y) / static_cast<float>(size - 1) * 2.0f - 1.0f;
        const auto i = static_cast<std::size_t>(y) * size + x;
        const float opacity = detailAlpha(p, layer);
        const float variation = 0.75f + 0.25f * noise(p * 18.0f + 5.0f, 13);

        alpha[i] = opacity;
        colors[i] = glm::u8vec4(glm::vec4(base * variation, opacity) * 255.0f);
      }
    }

    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        const auto i = static_cast<std::size_t>(y) * size + x;
        const float dx = alpha[y * size + std::min(x + 1, size - 1)] - alpha[y * size + std::max(x - 1, 0)];
        const float dz = alpha[std::min(y + 1, size - 1) * size + x] - alpha[std::max(y - 1, 0) * size + x];
        const auto normal = glm::normalize(glm::vec3(dx * 4.0f, 1, dz * 4.0f));

        surface[i] = glm::u8vec4(glm::vec4(normal * 0.5f + 0.5f, 0) * 255.0f);
        surface[i].a = layer == 0 ? 230 : 250;
      }
    }

    transfer.uploadImage(*oneShot, detailTextures[0], 0, layer, std::as_bytes(std::span(colors)));
    transfer.uploadImage(*oneShot, detailTextures[1], 0, layer, std::as_bytes(std::span(surface)));
  }
}

void WorldRenderer::updateDetails()
{
  const std::array<glm::vec2, 6> centers{{
    {-115, 90}, {-112.5f, 86}, {-116, 82}, {-113.5f, 78}, {-125, 87}, {-105, 76}}};
  details.invViewProj = glm::inverse(params.viewProj);

  for (std::size_t i = 0; i < centers.size(); ++i) {
    const auto center = centers[i];
    const glm::vec3 size = i < 4 ? glm::vec3(1.1f, 6, 2.1f) : glm::vec3(5, 8, 6);
    const glm::vec3 position{center.x, getHeight(center.x, center.y), center.y};
    glm::mat4 box = glm::translate(glm::mat4(1), position);
    box = glm::rotate(box, i % 2 == 0 ? -0.10f : 0.12f, glm::vec3(0, 1, 0));

    details.boxes[i] = glm::inverse(glm::scale(box, size));
    details.layers[i] = glm::vec4(i < 4 ? 0.0f : 1.0f, 0, 0, 0);
  }

  std::memcpy(detailBuffer.data(), &details, sizeof(details));
}

void WorldRenderer::allocateResources(glm::uvec2 res)
{
  resolution = res;
  auto& ctx = etna::get_context();
  depth = ctx.createImage(etna::Image::CreateInfo{
    .extent = {res.x, res.y, 1}, .name = "terrain_depth", .format = vk::Format::eD32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
  });

  gformats = {vk::Format::eR8G8B8A8Unorm, vk::Format::eR16G16Snorm, vk::Format::eR8G8B8A8Unorm};
  if (!compact) {
    gformats.fill(vk::Format::eR16G16B16A16Sfloat);
  }

  const auto usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
  for (std::size_t i = 0; i < gbuffer.size(); ++i) {
    gbuffer[i] = ctx.createImage(etna::Image::CreateInfo{
      .extent = {res.x, res.y, 1}, .name = "gbuffer", .format = gformats[i], .imageUsage = usage});
    detailGbuffer[i] = ctx.createImage(etna::Image::CreateInfo{
      .extent = {res.x, res.y, 1}, .name = "gbuffer_with_details", .format = gformats[i], .imageUsage = usage});
  }
}

void WorldRenderer::setupPipelines(vk::Format format)
{
  targetFormat = format;
  if (!rocks) {
    rocks = std::make_unique<Rocks>(gformats);
  } else {
    rocks->setupPipelines(gformats);
  }

  const etna::VertexShaderInputDescription::Binding vertexBinding{.byteStreamDescription = {
    .stride = sizeof(glm::vec2),
    .attributes = {{.format = vk::Format::eR32G32Sfloat, .offset = 0}},
  }};
  const etna::VertexShaderInputDescription input{.bindings = {vertexBinding}};
  const auto rgb = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB;
  const vk::PipelineColorBlendAttachmentState blend{.colorWriteMask = rgb | vk::ColorComponentFlagBits::eA};
  const std::vector<vk::PipelineColorBlendAttachmentState> blends(3, blend);
  const std::vector<vk::Format> formats(gformats.begin(), gformats.end());
  auto& manager = etna::get_context().getPipelineManager();

  depthPipeline = manager.createGraphicsPipeline("skybox_terrain_depth", {
    .vertexShaderInput = input,
    .blendingConfig = {.attachments = {}},
    .fragmentShaderOutput = {.depthAttachmentFormat = vk::Format::eD32Sfloat},
  });
  pipeline = manager.createGraphicsPipeline("skybox_terrain", {
    .vertexShaderInput = input,
    .blendingConfig = {.attachments = blends},
    .depthConfig = {
      .depthTestEnable = true, .depthWriteEnable = false, .depthCompareOp = vk::CompareOp::eEqual, .maxDepthBounds = 1},
    .fragmentShaderOutput = {.colorAttachmentFormats = formats, .depthAttachmentFormat = vk::Format::eD32Sfloat},
  });
  detailPipeline = manager.createGraphicsPipeline("skybox_details", {
    .blendingConfig = {.attachments = blends},
    .depthConfig = {.depthTestEnable = false, .depthWriteEnable = false},
    .fragmentShaderOutput = {.colorAttachmentFormats = formats},
  });
  lightPipeline = manager.createGraphicsPipeline("skybox_lighting", {
    .depthConfig = {.depthTestEnable = false, .depthWriteEnable = false},
    .fragmentShaderOutput = {.colorAttachmentFormats = {format}},
  });
}

void WorldRenderer::update(const FramePacket& packet)
{
  camera = packet.mainCam;
}

void WorldRenderer::drawTerrain(vk::CommandBuffer cmd, bool depthOnly)
{
  auto& selected = depthOnly ? depthPipeline : pipeline;
  const auto program = depthOnly ? "skybox_terrain_depth" : "skybox_terrain";
  const auto info = etna::get_shader_program(program);
  const auto layout = selected.getVkPipelineLayout();
  const auto imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
  std::vector<etna::Binding> bindings{{0, heightmap.genBinding(heightSampler.get(), imageLayout)}};

  if (!depthOnly) {
    for (std::uint32_t i = 0; i < textures.size(); ++i) {
      bindings.emplace_back(i + 1, textures[i].genBinding(materialSampler.get(), imageLayout));
    }
  }

  const auto set = etna::create_descriptor_set(info.getDescriptorLayoutId(0), cmd, bindings, BarrierBehavior::eSuppressBarriers);
  vk::ShaderStageFlags stages = vk::ShaderStageFlagBits::eVertex;
  if (!depthOnly) {
    stages |= vk::ShaderStageFlagBits::eFragment;
  }

  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, selected.getVkPipeline());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, {set.getVkSet()}, {});
  cmd.pushConstants<PushConstants>(layout, stages, 0, {params});
  cmd.bindVertexBuffers(0, {vertices.get()}, {0});
  cmd.bindIndexBuffer(indices.get(), 0, vk::IndexType::eUint32);
  cmd.drawIndexed(cells * cells * 6, 1, 0, 0, 0);
}

void WorldRenderer::drawGeometry(vk::CommandBuffer cmd, bool depthOnly)
{
  const vk::Rect2D rect{{0, 0}, {resolution.x, resolution.y}};
  using Target = etna::RenderTargetState::AttachmentParams;
  std::vector<Target> attachments;
  if (!depthOnly) {
    for (auto& target : gbuffer) {
      attachments.push_back({.image = target.get(), .view = target.getView({})});
    }
  }

  const auto loadOp = depthOnly ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
  const Target depthTarget{.image = depth.get(), .view = depth.getView({}), .loadOp = loadOp};
  etna::RenderTargetState targets(cmd, rect, attachments, depthTarget);

  drawTerrain(cmd, depthOnly);
  const glm::vec4 lightColor{1.0f, 0.94f, 0.82f, params.surface.w};
  const Rocks::Params rockParams{params.viewProj, params.eye, params.light, lightColor};
  rocks->draw(cmd, rockParams, depthOnly);
}

void WorldRenderer::drawDetails(vk::CommandBuffer cmd)
{
  const auto imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
  const auto depthLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
  std::vector<etna::Binding> bindings;
  for (std::uint32_t i = 0; i < gbuffer.size(); ++i) {
    bindings.emplace_back(i, gbuffer[i].genBinding(screenSampler.get(), imageLayout));
  }
  bindings.emplace_back(3, depth.genBinding(screenSampler.get(), depthLayout));
  bindings.emplace_back(4, detailBuffer.genBinding());

  const etna::Image::ViewParams layers{.layerCount = 2, .type = vk::ImageViewType::e2DArray};
  for (std::uint32_t i = 0; i < detailTextures.size(); ++i) {
    bindings.emplace_back(5 + i, detailTextures[i].genBinding(materialSampler.get(), imageLayout, layers));
  }
  const auto descriptorLayout = etna::get_shader_program("skybox_details").getDescriptorLayoutId(0);
  const auto set = etna::create_descriptor_set(descriptorLayout, cmd, bindings);

  std::vector<etna::RenderTargetState::AttachmentParams> attachments;
  for (auto& target : detailGbuffer) {
    attachments.push_back({.image = target.get(), .view = target.getView({})});
  }
  const vk::Rect2D rect{{0, 0}, {resolution.x, resolution.y}};
  etna::RenderTargetState targets(cmd, rect, attachments, {});

  const auto layout = detailPipeline.getVkPipelineLayout();
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, detailPipeline.getVkPipeline());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, {set.getVkSet()}, {});
  cmd.draw(3, 1, 0, 0);
}

void WorldRenderer::drawLighting(vk::CommandBuffer cmd, vk::Image image, vk::ImageView view)
{
  const auto imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
  const auto depthLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
  std::vector<etna::Binding> bindings;
  for (std::uint32_t i = 0; i < detailGbuffer.size(); ++i) {
    bindings.emplace_back(i, detailGbuffer[i].genBinding(screenSampler.get(), imageLayout));
  }
  bindings.emplace_back(3, depth.genBinding(screenSampler.get(), depthLayout));
  bindings.emplace_back(4, sky->getBinding());
  const auto descriptorLayout = etna::get_shader_program("skybox_lighting").getDescriptorLayoutId(0);
  const auto set = etna::create_descriptor_set(descriptorLayout, cmd, bindings);

  const glm::vec4 settings{params.surface.w, 0, 0, 0};
  const glm::vec4 color{sky->sunLight(), 0};
  struct LightData
  {
    glm::mat4 invViewProj;
    glm::vec4 eye;
    glm::vec4 light;
    glm::vec4 settings;
    glm::vec4 color;
  } lightData{details.invViewProj, params.eye, params.light, settings, color};

  const vk::Rect2D rect{{0, 0}, {resolution.x, resolution.y}};
  etna::RenderTargetState targets(cmd, rect, {{.image = image, .view = view}}, {});
  const auto layout = lightPipeline.getVkPipelineLayout();
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, lightPipeline.getVkPipeline());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, {set.getVkSet()}, {});
  cmd.pushConstants<LightData>(layout, vk::ShaderStageFlagBits::eFragment, 0, {lightData});
  cmd.draw(3, 1, 0, 0);
}

void WorldRenderer::renderWorld(vk::CommandBuffer cmd, vk::Image image, vk::ImageView view)
{
  const float aspect = static_cast<float>(resolution.x) / static_cast<float>(resolution.y);
  params.viewProj = camera.projTm(aspect) * camera.viewTm();
  params.eye = glm::vec4(camera.position, params.eye.w);
  params.light = glm::vec4(sky->sunDir(), 1);

  const Rocks::Terrain terrain{heights, mapSize, worldSize, heightScale, seed};
  rocks->update(cmd, camera.position, params.viewProj, terrain);
  updateDetails();

  const auto stage = vk::PipelineStageFlagBits2::eVertexShader | vk::PipelineStageFlagBits2::eFragmentShader;
  const auto access = vk::AccessFlagBits2::eShaderSampledRead;
  const auto layout = vk::ImageLayout::eShaderReadOnlyOptimal;
  const auto color = vk::ImageAspectFlagBits::eColor;
  etna::set_state(cmd, heightmap.get(), stage, access, layout, color);
  for (auto& texture : textures) {
    etna::set_state(cmd, texture.get(), vk::PipelineStageFlagBits2::eFragmentShader, access, layout, color);
  }

  const auto vertexStage = vk::PipelineStageFlagBits2::eVertexAttributeInput;
  const auto vertexAccess = vk::AccessFlagBits2::eVertexAttributeRead;
  etna::set_state(cmd, vertices.get(), vertexStage, vertexAccess);
  etna::set_state(cmd, indices.get(), vk::PipelineStageFlagBits2::eIndexInput, vk::AccessFlagBits2::eIndexRead);
  etna::flush_barriers(cmd);

  const auto pool = timestamps.get();
  const auto start = vk::PipelineStageFlagBits2::eTopOfPipe;
  const auto finish = vk::PipelineStageFlagBits2::eBottomOfPipe;
  cmd.resetQueryPool(pool, 0, 8);
  cmd.writeTimestamp2(start, pool, 0);
  drawGeometry(cmd, true);
  cmd.writeTimestamp2(finish, pool, 1);

  cmd.writeTimestamp2(start, pool, 2);
  drawGeometry(cmd, false);
  cmd.writeTimestamp2(finish, pool, 3);

  cmd.writeTimestamp2(start, pool, 4);
  drawDetails(cmd);
  cmd.writeTimestamp2(finish, pool, 5);

  cmd.writeTimestamp2(start, pool, 6);
  drawLighting(cmd, image, view);
  cmd.writeTimestamp2(finish, pool, 7);
  hasTimes = true;
  lastCompact = compact;
}

void WorldRenderer::readTimes()
{
  if (!hasTimes) {
    return;
  }

  std::array<std::uint64_t, 8> values;
  const auto& ctx = etna::get_context();
  const auto pool = timestamps.get();
  const auto flags = vk::QueryResultFlagBits::e64;
  const auto result = ctx.getDevice().getQueryPoolResults(pool, 0, 8, sizeof(values), values.data(), sizeof(values[0]), flags);
  if (result == vk::Result::eSuccess) {
    const std::size_t mode = lastCompact ? 0 : 1;
    const double period = ctx.getPhysicalDevice().getProperties().limits.timestampPeriod / 1000000.0;
    for (std::size_t i = 0; i < 4; ++i) {
      passTimes[mode][i] += static_cast<double>(values[i * 2 + 1] - values[i * 2]) * period;
    }

    frameTimes[mode] += static_cast<double>(values[7] - values[0]) * period;
    ++samples[mode];
  }

  hasTimes = false;
}

void WorldRenderer::drawStats()
{
  if (ImGui::Checkbox("Compact G-buffer", &compact)) {
    ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
    allocateResources(resolution);
    setupPipelines(targetFormat);
  }

  ImGui::Text("G-buffer: %u bytes / pixel", compact ? 16u : 28u);
  for (std::size_t mode = 0; mode < 2; ++mode) {
    const double count = std::max(samples[mode], 1u);
    const auto& times = passTimes[mode];
    const auto name = mode == 0 ? "Compact" : "Wide";
    ImGui::Text("%s: %.3f ms (%u samples)", name, frameTimes[mode] / count, samples[mode]);
    ImGui::Text(" Depth %.3f / G-buffer %.3f", times[0] / count, times[1] / count);
    ImGui::Text(" Details %.3f / Light %.3f", times[2] / count, times[3] / count);
  }

  if (ImGui::Button("Reset GPU averages")) {
    passTimes = {};
    frameTimes = {};
    samples = {};
  }
}

void WorldRenderer::drawGui()
{
  ImGui::SetNextWindowPos({18, 18}, ImGuiCond_Once);
  ImGui::SetNextWindowSize({340, 0}, ImGuiCond_Once);
  ImGui::Begin("Sky and landscape");

  if (ImGui::CollapsingHeader("Sun and clouds", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (sky->drawGui()) {
      sky->bake();
    }
  }

  if (ImGui::CollapsingHeader("G-buffer / GPU profiler")) {
    drawStats();
  }

  if (ImGui::CollapsingHeader("Footprints and trampled ground")) {
    ImGui::SliderFloat("Opacity", &details.strength.x, 0, 1);
    ImGui::SliderFloat("Albedo effect", &details.strength.y, 0, 1);
    ImGui::SliderFloat("Normal effect", &details.strength.z, 0, 1);
    ImGui::SliderFloat("Roughness##details", &details.strength.w, 0, 1);
  }

  if (ImGui::CollapsingHeader("Rocks")) {
    rocks->gui();
  }

  if (ImGui::CollapsingHeader("Landscape")) {
    ImGui::InputInt("Seed", &seed);
    ImGui::SliderInt("Octaves", &octaves, 1, 9);
    ImGui::SliderFloat("Frequency", &frequency, 1, 9);
    ImGui::SliderFloat("Persistence", &persistence, 0.15f, 0.8f);
    ImGui::SliderFloat("Lacunarity", &lacunarity, 1.3f, 3.0f);
    ImGui::SliderFloat("Height", &heightScale, 15, 110);
    ImGui::SliderFloat("Coast width", &coastWidth, 0.15f, 0.8f);

    if (ImGui::Button("Regenerate")) {
      regenerate();
    }
  }

  if (ImGui::CollapsingHeader("Materials")) {
    ImGui::SliderFloat("Tiling", &params.surface.x, 8, 130);
    ImGui::SliderFloat("Height detail", &params.surface.y, 0, 2);
    ImGui::SliderFloat("Roughness", &params.surface.z, 0.2f, 1.5f);

    ImGui::SliderFloat("Sand height", &params.bands.x, 1, 12);
    ImGui::SliderFloat("Rock height", &params.bands.y, 15, 75);
    ImGui::SliderFloat("Snow height", &params.bands.z, 25, 100);
    ImGui::SliderFloat("Blend width", &params.bands.w, 0.5f, 12);
  }

  if (ImGui::CollapsingHeader("Lighting")) {
    ImGui::SliderFloat("Ambient", &params.surface.w, 0.02f, 0.6f);
    ImGui::SliderFloat("Exposure", &params.eye.w, 0.25f, 3);
  }

  ImGui::Separator();
  ImGui::Text("WASD / R,F - move; Shift - faster");
  ImGui::Text("RMB - mouse look; wheel - zoom");

  ImGui::End();
}
