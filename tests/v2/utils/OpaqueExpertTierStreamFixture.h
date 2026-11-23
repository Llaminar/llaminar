/**
 * @file OpaqueExpertTierStreamFixture.h
 * @brief Opaque bytes and a legal expanded layout for host stream protocol tests.
 *
 * These fixtures exercise the existing wire protocol without relying on any
 * ordinary source format being convertible by the legacy GPU tier packer.
 */
#pragma once

#include "execution/moe/ExpertTierWeightStream.h"

#include <algorithm>
#include <utility>

namespace llaminar2::test
{
    /** @return Deterministic live bytes, with zero padding, under a symmetric expanded ABI. */
    inline cpu::native_quantised::CPUNativeQuantisedPackedWeights opaqueExpandedExpertTierBytes(int n, int k)
    {
        using namespace cpu::native_quantised;
        CPUNativeQuantisedPackedWeights packed;
        packed.N = n;
        packed.K = k;
        packed.N_padded = (n + 63) / 64 * 64;
        packed.blocks_per_row = k / 32;
        // A non-Q8 symmetric provenance tag disambiguates the compensated
        // expanded ABI; the bytes deliberately do not encode a source tensor.
        packed.codebook_id = native_quantised_formats::Q4_0.codebook_id;
        packed.payload_bytes = 32;
        packed.encoding = CPUNativeQuantisedEncoding::ExpandedInt8;
        packed.data_stride = preparedDataStride(packed.encoding, packed.codebook_id);
        packed.interleaved_block_stride = preparedInterleavedBlockStride(packed.encoding, false, packed.codebook_id);
        packed.native_interleaved.resize(packed.preparedBytes(), 0);
        const auto bytes = packed.native_interleaved.beginRewrite();
        for (int chunk = 0; chunk < packed.N_padded / 64; ++chunk)
            for (int kb = 0; kb < packed.blocks_per_row; ++kb)
            {
                const auto first = packed.blockOffset(chunk, kb);
                const auto fill = [&](std::size_t offset, int count) {
                    for (int i = 0; i < count; ++i)
                        bytes[first + offset + i] = static_cast<std::uint8_t>((first + offset + i) * 37u + 11u);
                };
                const int live = std::min(64, n - chunk * 64);
                for (int group = 0; group < 8; ++group)
                    fill(group * 256, live * 4);
                fill(packed.data_stride, live * 2);
                fill(packed.data_stride + 128, live * 2);
            }
        return packed;
    }

    /** @return A demotion wire fixture for the opaque expanded ABI, not a native source conversion. */
    inline ExpertTierWeightStreamManifest opaqueExpandedDemotionManifest(int n, int k, std::uint64_t epoch,
        int layer, int expert, ExpertTierWeightProjection projection, std::uint32_t units_per_chunk)
    {
        auto manifest = makeCpuToGpuExpertTierWeightStreamManifest(
            opaqueExpandedExpertTierBytes(n, k), epoch, layer, expert, projection, units_per_chunk);
        std::swap(manifest.source_packing, manifest.destination_packing);
        manifest.manifest_hash = manifest.computedHash();
        return manifest;
    }
}
