#include "WorldRenderer.hpp"
#include "Mipmaps.hpp"

#include <algorithm>
#include <cmath>
#include <string>

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
}

WorldRenderer::WorldRenderer()
  : oneShot{etna::get_context().createOneShotCmdMgr()}
  , transfer{etna::BlockingTransferHelper::CreateInfo{.stagingSize = 16 * 1024 * 1024}}
{
  static_assert(sizeof(PushConstants) == 128);

  heightSampler = etna::Sampler(etna::Sampler::CreateInfo{.name = "terrain_height_sampler"});
  materialSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .addressMode = vk::SamplerAddressMode::eRepeat, .name = "terrain_material_sampler"});

  createMesh();
  regenerate();
  loadMaterials();

  const auto vertexShader = INSTANCING_SHADERS_ROOT "terrain.vert.spv";
  const auto fragmentShader = INSTANCING_SHADERS_ROOT "terrain.frag.spv";
  etna::create_program("terrain_material", {vertexShader, fragmentShader});
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

void WorldRenderer::allocateResources(glm::uvec2 res)
{
  resolution = res;

  depth = etna::get_context().createImage(etna::Image::CreateInfo{
    .extent = {res.x, res.y, 1}, .name = "terrain_depth", .format = vk::Format::eD32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
  });
}

void WorldRenderer::setupPipelines(vk::Format format)
{
  rocks = std::make_unique<Rocks>(format);

  const etna::VertexShaderInputDescription::Binding vertexBinding{.byteStreamDescription = {
    .stride = sizeof(glm::vec2),
    .attributes = {{.format = vk::Format::eR32G32Sfloat, .offset = 0}},
  }};

  const etna::GraphicsPipeline::CreateInfo info{
    .vertexShaderInput = {.bindings = {vertexBinding}},
    .fragmentShaderOutput = {.colorAttachmentFormats = {format}, .depthAttachmentFormat = vk::Format::eD32Sfloat},
  };

  pipeline = etna::get_context().getPipelineManager().createGraphicsPipeline("terrain_material", info);
}

void WorldRenderer::update(const FramePacket& packet)
{
  camera = packet.mainCam;
}

void WorldRenderer::renderWorld(vk::CommandBuffer cmd, vk::Image image, vk::ImageView view)
{
  const float aspect = static_cast<float>(resolution.x) / static_cast<float>(resolution.y);
  params.viewProj = camera.projTm(aspect) * camera.viewTm();
  params.eye = glm::vec4(camera.position, params.eye.w);

  const Rocks::Terrain terrain{heights, mapSize, worldSize, heightScale, seed};
  rocks->update(cmd, camera.position, params.viewProj, terrain);

  const auto info = etna::get_shader_program("terrain_material");
  std::vector<etna::Binding> bindings;
  bindings.emplace_back(0, heightmap.genBinding(heightSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal));

  for (std::uint32_t i = 0; i < textures.size(); ++i) {
    bindings.emplace_back(i + 1, textures[i].genBinding(materialSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal));
  }

  const auto set = etna::create_descriptor_set(info.getDescriptorLayoutId(0), cmd, bindings);

  const vk::Rect2D rect{{0, 0}, {resolution.x, resolution.y}};
  using Target = etna::RenderTargetState::AttachmentParams;
  const Target target{
    .image = image, .view = view, .clearColorValue = std::array<float, 4>{0.006f, 0.025f, 0.065f, 1}};
  const Target depthTarget{.image = depth.get(), .view = depth.getView({})};
  etna::RenderTargetState targets(cmd, rect, {target}, depthTarget);

  const auto layout = pipeline.getVkPipelineLayout();
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.getVkPipeline());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, {set.getVkSet()}, {});

  const auto stages = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
  cmd.pushConstants<PushConstants>(layout, stages, 0, {params});

  cmd.bindVertexBuffers(0, {vertices.get()}, {0});
  cmd.bindIndexBuffer(indices.get(), 0, vk::IndexType::eUint32);
  cmd.drawIndexed(cells * cells * 6, 1, 0, 0, 0);

  const glm::vec4 lightColor{1.0f, 0.94f, 0.82f, params.surface.w};
  const Rocks::Params rockParams{params.viewProj, params.eye, params.light, lightColor};
  rocks->draw(cmd, rockParams);
}

void WorldRenderer::drawGui()
{
  ImGui::SetNextWindowPos({18, 18}, ImGuiCond_Once);
  ImGui::SetNextWindowSize({340, 0}, ImGuiCond_Once);
  ImGui::Begin("Island terrain");

  if (ImGui::CollapsingHeader("Rocks", ImGuiTreeNodeFlags_DefaultOpen)) {
    rocks->gui();
  }

  if (ImGui::CollapsingHeader("Landscape", ImGuiTreeNodeFlags_DefaultOpen)) {
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

  if (ImGui::CollapsingHeader("Materials", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::SliderFloat("Tiling", &params.surface.x, 8, 130);
    ImGui::SliderFloat("Height detail", &params.surface.y, 0, 2);
    ImGui::SliderFloat("Roughness", &params.surface.z, 0.2f, 1.5f);

    ImGui::SliderFloat("Sand height", &params.bands.x, 1, 12);
    ImGui::SliderFloat("Rock height", &params.bands.y, 15, 75);
    ImGui::SliderFloat("Snow height", &params.bands.z, 25, 100);
    ImGui::SliderFloat("Blend width", &params.bands.w, 0.5f, 12);
  }

  if (ImGui::CollapsingHeader("Lighting")) {
    ImGui::SliderFloat3("Sun direction", &params.light.x, -1, 1);
    ImGui::SliderFloat("Sun intensity", &params.light.w, 0, 8);
    ImGui::SliderFloat("Ambient", &params.surface.w, 0.02f, 0.6f);
    ImGui::SliderFloat("Exposure", &params.eye.w, 0.25f, 3);
  }

  ImGui::Separator();
  ImGui::Text("WASD / R,F - move; Shift - faster");
  ImGui::Text("RMB - mouse look; wheel - zoom");

  ImGui::End();
}
