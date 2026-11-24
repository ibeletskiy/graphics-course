#include "Rocks.hpp"
#include "Mipmaps.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>
#include <imgui.h>
#include <tiny_gltf.h>
#include <etna/BlockingTransferHelper.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>

namespace
{
constexpr float chunkSize = 32.0f;
constexpr int rocksPerChunk = 16;
constexpr uint32_t maxInstances = 32768;
constexpr size_t updateSize = 65536;

uint32_t hash(uint32_t x)
{
  x ^= x >> 16;
  x *= 0x7feb352d;
  x ^= x >> 15;
  x *= 0x846ca68b;

  return x ^ (x >> 16);
}

float random(uint32_t& state)
{
  state = hash(state + 0x9e3779b9);
  return static_cast<float>(state >> 8) / 16777216.0f;
}

float sampleHeight(std::span<const float> data, int size, glm::vec2 uv)
{
  glm::vec2 p = uv * static_cast<float>(size) - 0.5f;
  p = glm::clamp(p, glm::vec2(0), glm::vec2(static_cast<float>(size - 1)));

  const glm::ivec2 a = glm::ivec2(p);
  const glm::ivec2 b = glm::min(a + 1, glm::ivec2(size - 1));
  const glm::vec2 f = glm::fract(p);
  const float top = glm::mix(data[a.y * size + a.x], data[a.y * size + b.x], f.x);
  const float bottom = glm::mix(data[b.y * size + a.x], data[b.y * size + b.x], f.x);

  return glm::mix(top, bottom, f.y);
}
}

Rocks::Rocks(std::span<const vk::Format> formats)
  : sampler(etna::Sampler::CreateInfo{.addressMode = vk::SamplerAddressMode::eRepeat, .name = "rock sampler"})
{
  load(meshes[0], "boulder_01");
  load(meshes[1], "rock_09");

  const auto vertexShader = SKYBOX_SHADERS_ROOT "rocks.vert.spv";
  const auto fragmentShader = SKYBOX_SHADERS_ROOT "rocks.frag.spv";
  etna::create_program("skybox_rocks", {vertexShader, fragmentShader});
  etna::create_program("skybox_rocks_depth", {vertexShader});
  setupPipelines(formats);
}

void Rocks::setupPipelines(std::span<const vk::Format> formats)
{
  const etna::VertexByteStreamFormatDescription vertexFormat{
    .stride = sizeof(Vertex),
    .attributes = {
      {.format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, position)},
      {.format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, normal)},
      {.format = vk::Format::eR32G32Sfloat, .offset = offsetof(Vertex, uv)},
    },
  };
  const etna::VertexShaderInputDescription::Binding binding{.byteStreamDescription = vertexFormat};
  const etna::VertexShaderInputDescription input{.bindings = {binding}};
  const auto rgb = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB;
  const vk::PipelineColorBlendAttachmentState blend{
    .colorWriteMask = rgb | vk::ColorComponentFlagBits::eA,
  };

  pipeline = {};
  depthPipeline = {};
  auto& manager = etna::get_context().getPipelineManager();
  depthPipeline = manager.createGraphicsPipeline("skybox_rocks_depth", {
    .vertexShaderInput = input,
    .blendingConfig = {.attachments = {}},
    .fragmentShaderOutput = {.depthAttachmentFormat = vk::Format::eD32Sfloat},
  });
  pipeline = manager.createGraphicsPipeline("skybox_rocks", {
    .vertexShaderInput = input,
    .blendingConfig = {.attachments = std::vector<vk::PipelineColorBlendAttachmentState>(3, blend)},
    .depthConfig = {
      .depthTestEnable = true,
      .depthWriteEnable = false,
      .depthCompareOp = vk::CompareOp::eEqual,
      .maxDepthBounds = 1,
    },
    .fragmentShaderOutput = {
      .colorAttachmentFormats = {formats.begin(), formats.end()},
      .depthAttachmentFormat = vk::Format::eD32Sfloat,
    },
  });
}

void Rocks::load(Mesh& mesh, const char* name)
{
  tinygltf::TinyGLTF loader;
  tinygltf::Model model;
  std::string error, warning;
  const std::string folder = std::string(GRAPHICS_COURSE_RESOURCES_ROOT) + "/rocks/" + name;
  const std::string path = folder + "/" + name + ".gltf";
  ETNA_VERIFYF(loader.LoadASCIIFromFile(&model, &error, &warning, path), "{}", error);

  const auto& prim = model.meshes[0].primitives[0];
  const auto& position = model.accessors[prim.attributes.at("POSITION")];
  const auto& normal = model.accessors[prim.attributes.at("NORMAL")];
  const auto& uv = model.accessors[prim.attributes.at("TEXCOORD_0")];
  const auto& indices = model.accessors[prim.indices];

  auto data = [&](const tinygltf::Accessor& accessor, size_t i) {
    const auto& view = model.bufferViews[accessor.bufferView];
    const auto stride = static_cast<size_t>(accessor.ByteStride(view));
    const auto offset = view.byteOffset + accessor.byteOffset + i * stride;
    return model.buffers[view.buffer].data.data() + offset;
  };

  glm::vec3 lo, hi;
  for (int i = 0; i < 3; ++i) {
    lo[i] = static_cast<float>(position.minValues[i]);
    hi[i] = static_cast<float>(position.maxValues[i]);
  }

  const glm::vec3 extent = hi - lo;
  const float scale = 1.0f / std::max({extent.x, extent.y, extent.z});
  const glm::vec3 origin{(lo.x + hi.x) * 0.5f, lo.y, (lo.z + hi.z) * 0.5f};
  mesh.center = ((lo + hi) * 0.5f - origin) * scale;
  mesh.radius = glm::length(extent) * scale * 0.5f;

  std::vector<Vertex> vertices(position.count);
  for (size_t i = 0; i < vertices.size(); ++i) {
    auto& vertex = vertices[i];
    std::memcpy(&vertex.position, data(position, i), sizeof(vertex.position));
    std::memcpy(&vertex.normal, data(normal, i), sizeof(vertex.normal));
    std::memcpy(&vertex.uv, data(uv, i), sizeof(vertex.uv));
    vertex.position = (vertex.position - origin) * scale;
  }

  std::vector<uint32_t> elements(indices.count);
  for (size_t i = 0; i < elements.size(); ++i) {
    if (indices.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
      uint16_t index;
      std::memcpy(&index, data(indices, i), sizeof(index));
      elements[i] = index;
    } else {
      std::memcpy(&elements[i], data(indices, i), sizeof(uint32_t));
    }
  }
  mesh.indexCount = static_cast<uint32_t>(elements.size());

  auto& ctx = etna::get_context();
  mesh.vertices = ctx.createBuffer({
    .size = vertices.size() * sizeof(Vertex),
    .bufferUsage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .name = name,
  });
  mesh.indices = ctx.createBuffer({
    .size = elements.size() * sizeof(uint32_t),
    .bufferUsage = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .name = name,
  });
  mesh.instances = ctx.createBuffer({
    .size = maxInstances * sizeof(Instance),
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .name = "rock instances",
  });

  auto commands = ctx.createOneShotCmdMgr();
  etna::BlockingTransferHelper transfer({.stagingSize = 2048 * 2048 * 4});
  transfer.uploadBuffer<Vertex>(*commands, mesh.vertices, 0, vertices);
  transfer.uploadBuffer<uint32_t>(*commands, mesh.indices, 0, elements);

  loadTextures(mesh, model);
}

void Rocks::loadTextures(Mesh& mesh, const tinygltf::Model& model)
{
  const auto& material = model.materials[model.meshes[0].primitives[0].material];
  const auto& pbr = material.pbrMetallicRoughness;
  const std::array textureIds{pbr.baseColorTexture.index, pbr.metallicRoughnessTexture.index, material.normalTexture.index};
  const auto transferUsage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;

  auto& ctx = etna::get_context();
  auto commands = ctx.createOneShotCmdMgr();
  etna::BlockingTransferHelper transfer({.stagingSize = 2048 * 2048 * 4});

  for (size_t i = 0; i < textureIds.size(); ++i) {
    const auto& image = model.images[model.textures[textureIds[i]].source];
    const auto mipCount = static_cast<uint32_t>(std::log2(std::max(image.width, image.height))) + 1;
    mesh.textures[i] = ctx.createImage({
      .extent = {static_cast<uint32_t>(image.width), static_cast<uint32_t>(image.height), 1},
      .name = image.name,
      .format = i == 0 ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm,
      .imageUsage = transferUsage | vk::ImageUsageFlagBits::eSampled,
      .mipLevels = mipCount,
    });

    transfer.uploadImage(*commands, mesh.textures[i], 0, 0, std::as_bytes(std::span(image.image)));
    generateMips(mesh.textures[i], *commands);
  }
}

bool Rocks::isVisible(glm::vec3 center, float radius, const View& view) const
{
  if (glm::distance(center, view.eye) - radius > distance) {
    return false;
  }

  for (const auto& plane : view.planes) {
    if (glm::dot(plane, glm::vec4(center, 1)) < -radius) {
      return false;
    }
  }

  return true;
}

void Rocks::generateChunk(glm::ivec2 chunk, const View& view, const Terrain& terrain)
{
  uint32_t state = hash(static_cast<uint32_t>(chunk.x));
  state ^= hash(static_cast<uint32_t>(chunk.y) + 12345);
  state ^= hash(static_cast<uint32_t>(terrain.seed));

  for (int i = 0; i < rocksPerChunk; ++i) {
    const float x = random(state);
    const float z = random(state);
    const glm::vec2 p = (glm::vec2(chunk) + glm::vec2(x, z)) * chunkSize;
    const float chance = random(state);
    const float size = glm::mix(sizes.x, sizes.y, random(state));
    const float angle = random(state) * glm::two_pi<float>();
    const size_t type = random(state) < 0.5f ? 0 : 1;
    const glm::vec3 color = glm::vec3(0.85f + 0.3f * random(state));

    const glm::vec2 uv = p / terrain.worldSize + 0.5f;
    if (uv.x < 0 || uv.y < 0 || uv.x > 1 || uv.y > 1) {
      continue;
    }

    const float height = sampleHeight(terrain.heights, terrain.mapSize, uv);
    float weight = glm::smoothstep(1.0f, 8.0f, height);
    weight *= 1.0f - glm::smoothstep(terrain.heightScale * 0.65f, terrain.heightScale * 0.9f, height);
    if (chance >= density * weight) {
      continue;
    }

    const glm::vec3 position{p.x, height - size * 0.06f, p.y};
    auto& mesh = meshes[type];
    const glm::vec3 center = position + mesh.center * size;
    if (!isVisible(center, mesh.radius * size, view)) {
      continue;
    }

    glm::mat4 model = glm::translate(glm::mat4(1), position);
    model = glm::rotate(model, angle, glm::vec3(0, 1, 0));
    model = glm::scale(model, glm::vec3(size));
    mesh.visible.push_back({model, glm::vec4(color, 1)});
  }
}

void Rocks::generate(const View& view, const Terrain& terrain)
{
  for (auto& mesh : meshes) {
    mesh.visible.clear();
  }

  const glm::vec2 eye{view.eye.x, view.eye.z};
  glm::ivec2 first = glm::ivec2(glm::floor((eye - distance) / chunkSize));
  glm::ivec2 last = glm::ivec2(glm::floor((eye + distance) / chunkSize));
  const int firstChunk = static_cast<int>(std::floor(-terrain.worldSize * 0.5f / chunkSize));
  const int lastChunk = static_cast<int>(std::floor(terrain.worldSize * 0.5f / chunkSize));
  first = glm::max(first, glm::ivec2(firstChunk));
  last = glm::min(last, glm::ivec2(lastChunk));

  for (int z = first.y; z <= last.y; ++z) {
    for (int x = first.x; x <= last.x; ++x) {
      generateChunk({x, z}, view, terrain);
    }
  }
}

void Rocks::upload(vk::CommandBuffer cmd)
{
  for (auto& mesh : meshes) {
    if (mesh.visible.empty()) {
      continue;
    }
    ETNA_VERIFY(mesh.visible.size() <= maxInstances);

    const auto instanceBuffer = mesh.instances.get();
    etna::set_state(cmd, instanceBuffer, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
    etna::flush_barriers(cmd);

    const auto bytes = std::as_bytes(std::span(mesh.visible));
    for (size_t offset = 0; offset < bytes.size(); offset += updateSize) {
      const size_t size = std::min(updateSize, bytes.size() - offset);
      cmd.updateBuffer(instanceBuffer, offset, size, bytes.data() + offset);
    }

    etna::set_state(cmd, instanceBuffer, vk::PipelineStageFlagBits2::eVertexShader, vk::AccessFlagBits2::eShaderRead);
    const auto vertexStage = vk::PipelineStageFlagBits2::eVertexAttributeInput;
    const auto vertexAccess = vk::AccessFlagBits2::eVertexAttributeRead;
    etna::set_state(cmd, mesh.vertices.get(), vertexStage, vertexAccess);
    etna::set_state(cmd, mesh.indices.get(), vk::PipelineStageFlagBits2::eIndexInput, vk::AccessFlagBits2::eIndexRead);

    const auto stage = vk::PipelineStageFlagBits2::eFragmentShader;
    const auto access = vk::AccessFlagBits2::eShaderRead;
    const auto layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    for (auto& texture : mesh.textures) {
      etna::set_state(cmd, texture.get(), stage, access, layout, vk::ImageAspectFlagBits::eColor);
    }
  }

  etna::flush_barriers(cmd);
}

void Rocks::update(vk::CommandBuffer cmd, const glm::vec3& eye, const glm::mat4& viewProj, const Terrain& terrain)
{
  const glm::mat4 rows = glm::transpose(viewProj);
  View view{eye, {}};
  view.planes[0] = rows[3] + rows[0];
  view.planes[1] = rows[3] - rows[0];
  view.planes[2] = rows[3] + rows[1];
  view.planes[3] = rows[3] - rows[1];
  view.planes[4] = rows[2];
  view.planes[5] = rows[3] - rows[2];

  for (auto& plane : view.planes) {
    plane /= glm::length(glm::vec3(plane));
  }

  generate(view, terrain);
  upload(cmd);
}

void Rocks::draw(vk::CommandBuffer cmd, const Params& params, bool depthOnly)
{
  auto& selected = depthOnly ? depthPipeline : pipeline;
  const auto layout = selected.getVkPipelineLayout();
  const auto program = depthOnly ? "skybox_rocks_depth" : "skybox_rocks";
  const auto descriptorLayout = etna::get_shader_program(program).getDescriptorLayoutId(0);
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, selected.getVkPipeline());
  cmd.pushConstants<Params>(layout, vk::ShaderStageFlagBits::eVertex, 0, {params});

  for (auto& mesh : meshes) {
    if (mesh.visible.empty()) {
      continue;
    }

    const auto imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    std::vector<etna::Binding> bindings{{0, mesh.instances.genBinding()}};
    if (!depthOnly) {
      for (uint32_t i = 0; i < mesh.textures.size(); ++i) {
        bindings.emplace_back(i + 1, mesh.textures[i].genBinding(sampler.get(), imageLayout));
      }
    }
    const auto set = etna::create_descriptor_set(descriptorLayout, cmd, bindings, BarrierBehavior::eSuppressBarriers);

    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, {set.getVkSet()}, {});
    cmd.bindVertexBuffers(0, {mesh.vertices.get()}, {0});
    cmd.bindIndexBuffer(mesh.indices.get(), 0, vk::IndexType::eUint32);
    cmd.drawIndexed(mesh.indexCount, static_cast<uint32_t>(mesh.visible.size()), 0, 0, 0);
  }
}

void Rocks::gui()
{
  ImGui::SliderFloat("Rock density", &density, 0.0f, 1.0f);
  ImGui::SliderFloat("Rock distance", &distance, 40.0f, 1200.0f, "%.0f");
  ImGui::SliderFloat2("Rock sizes", &sizes.x, 0.5f, 12.0f);
  sizes.y = std::max(sizes.x, sizes.y);
}
