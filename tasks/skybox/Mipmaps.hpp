#pragma once

#include <algorithm>
#include <array>
#include <etna/Etna.hpp>
#include <etna/Image.hpp>
#include <etna/OneShotCmdMgr.hpp>

inline void generateMips(etna::Image& image, etna::OneShotCmdMgr& commands) {
  auto cmd = commands.start();
  ETNA_CHECK_VK_RESULT(cmd.begin(vk::CommandBufferBeginInfo{}));

  const auto texture = image.get();
  const auto extent = image.getExtent();
  int width = static_cast<int>(extent.width);
  int height = static_cast<int>(extent.height);
  const auto stage = vk::PipelineStageFlagBits2::eTransfer;
  const auto access = vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite;
  const auto layout = vk::ImageLayout::eGeneral;
  const auto aspect = vk::ImageAspectFlagBits::eColor;

  for (uint32_t mip = 1; mip < image.getMipLevelCount(); ++mip) {
    etna::set_state(cmd, texture, stage, access, layout, aspect, ForceSetState::eTrue);
    etna::flush_barriers(cmd);

    const int nextWidth = std::max(width / 2, 1);
    const int nextHeight = std::max(height / 2, 1);
    const vk::ImageBlit region{
      .srcSubresource = {aspect, mip - 1, 0, 1},
      .srcOffsets = std::array<vk::Offset3D, 2>{{{0, 0, 0}, {width, height, 1}}},
      .dstSubresource = {aspect, mip, 0, 1},
      .dstOffsets = std::array<vk::Offset3D, 2>{{{0, 0, 0}, {nextWidth, nextHeight, 1}}},
    };
    cmd.blitImage(texture, layout, texture, layout, {region}, vk::Filter::eLinear);

    width = nextWidth;
    height = nextHeight;
  }

  ETNA_CHECK_VK_RESULT(cmd.end());
  commands.submitAndWait(cmd);
}
