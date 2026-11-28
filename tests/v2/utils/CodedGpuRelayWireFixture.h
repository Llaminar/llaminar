/**
 * @file CodedGpuRelayWireFixture.h
 * @brief Scalar wire oracle for phased GPU roots with ragged blocks and shared code bytes.
 */
#pragma once

#include "execution/moe/ExpertTierGpuBlobTransferLane.h"

#include <algorithm>
#include <vector>

namespace llaminar2::test
{
    /** @brief Host-owned byte patterns for one live coded projection with trailing reader guards. */
    struct CodedGpuRelayWireFixture
    {
        using Projection = MoEExpertProjectionOwnership::Projection;
        using Residence = MoEProjectionResidence;
        static constexpr Projection source_root{Residence::ParticipantInputSlice, 13, 512, 0, 7, 32, 352};
        static constexpr Projection destination_root{Residence::ParticipantInputSlice, 13, 512, 4, 9, 64, 320};
        static constexpr uint32_t chunk_capacity = 128;

        GpuExpertPackedDescriptor geometry;
        std::vector<uint8_t> payload, codes, headers;

        /** @brief Generate distinguishable planes including nonzero neighboring and padding bits. */
        CodedGpuRelayWireFixture(const NativeQuantisedFormatInfo &format, const Projection &root,
                                 uint8_t shift, uint32_t seed)
        {
            namespace sb = native_quantised_superblock;
            const auto physical = canonicalDeviceVnniFormat(format);
            const auto blocks = static_cast<uint32_t>(root.columns / 32);
            const auto phase = sb::superblockPhase(root.first_column / 32);
            const size_t slots = size_t(root.rows) * blocks;
            const auto bits = slots * sb::superblockCodePlaneBits(physical.codebook_id);
            geometry = {.n = root.rows, .k = root.columns, .blocks_per_row = blocks,
                .codebook_id = physical.codebook_id, .payload_bytes_per_block = physical.payload_bytes_per_block,
                .is_asymmetric = physical.is_asymmetric, .has_emins = physical.has_emins,
                .metadata_layout = nativeQuantisedMetadataLayout(NativeQuantisedMetadataRepresentation::SuperblockCoded,
                    phase, bits ? shift : 0u),
                .vnni_bytes = slots * physical.payload_bytes_per_block,
                .scales_bytes = bits ? (bits + shift + 7u) / 8u : 0u,
                .mins_bytes = sb::planeBytes(physical.codebook_id, root.rows, blocks, phase).superblock};
            const auto fill = [&seed](auto &array, size_t bytes) {
                array.resize(bytes);
                for (auto &value : array)
                {
                    seed = seed * 1664525u + 1013904223u;
                    value = static_cast<uint8_t>(seed >> 24);
                }
            };
            fill(payload, geometry.vnni_bytes);
            fill(codes, geometry.scales_bytes ? geometry.scales_bytes + 4u : 0u);
            fill(headers, geometry.mins_bytes);
        }

        /** @return Exact descriptor for the host oracle or known GPU allocations. */
        GpuExpertPackedDescriptor descriptor(uint8_t *p, uint8_t *c, uint8_t *h) const noexcept
        {
            auto result = geometry;
            result.ptrs = {.d_vnni = p, .d_scales = geometry.scales_bytes ? c : p, .d_mins = h};
            return result;
        }
        /** @return Device-free descriptor used only by the protocol and manifest. */
        GpuExpertPackedDescriptor descriptor() noexcept
        {
            return descriptor(payload.data(), codes.data(), headers.data());
        }

        /** @brief Update only the selected bit field, including across byte boundaries. */
        static void putBits(uint8_t *data, size_t offset, unsigned width, uint32_t value)
        {
            for (unsigned i = 0; i < width; ++i)
            {
                const size_t bit = offset + i;
                const auto mask = static_cast<uint8_t>(1u << (bit % 8u));
                data[bit / 8u] = (data[bit / 8u] & ~mask) | ((value >> i) & 1u ? mask : 0u);
            }
        }

        /** @return Live wire bytes, independently transposed from block-major GPU planes. */
        std::vector<uint8_t> wire(const ExpertTierWeightDeviceLayout &layout, uint32_t first, uint32_t count) const
        {
            namespace sb = native_quantised_superblock;
            const auto planes = expertTierCodedPlanes(geometry.codebook_id);
            const unsigned bits = planes.low_bits + planes.high_bits;
            const unsigned field = sb::superblockFieldBytes(geometry.codebook_id);
            const auto shift = nativeQuantisedLayoutCodeBitShift(geometry.metadata_layout);
            std::vector<uint8_t> result(layout.chunkBytes(count, first));
            size_t offset = 0;
            for (uint32_t unit = first; unit < first + count; ++unit)
            {
                const auto kb = unit % layout.blocks_per_row;
                const auto cols = expertTierWireColumns(layout, unit);
                const auto block = layout.cpuStorage().wireGeometry(cols);
                const auto payload_width = geometry.payload_bytes_per_block;
                for (unsigned col = 0; col < cols; ++col)
                {
                    const auto row = (unit / layout.blocks_per_row) * 64u + layout.gpu_first_row + col;
                    const auto gpu = sb::blockMajorAddress(kb + layout.gpu_first_column / 32, row,
                        geometry.n, layout.gpu_superblock_phase);
                    for (unsigned byte = 0; byte < payload_width; ++byte)
                        result[offset + expertTierCodedPayloadOffset(payload_width, cols, col, byte)] =
                            payload[gpu.linear * payload_width + byte];
                    auto *metadata = result.data() + offset + payload_width * cols;
                    if (bits)
                    {
                        const auto code = sb::loadPackedBits(codes.data(), gpu.linear * bits + shift, bits);
                        putBits(metadata, col * planes.low_bits, planes.low_bits, code);
                        putBits(metadata + planes.bytes(cols, planes.low_bits), col * planes.high_bits,
                            planes.high_bits, code >> planes.low_bits);
                    }
                    if (block.closesSuperblock(kb))
                        std::copy_n(headers.data() + gpu.superblock * field, field,
                            result.data() + offset + block.block_stride + col * field);
                }
                offset += block.block_stride + (block.closesSuperblock(kb) ? field * cols : 0u);
            }
            return result;
        }

        /** @brief Apply only live source bits to expected destination storage; guards remain untouched. */
        void copyIntersectionFrom(const CodedGpuRelayWireFixture &source,
                                  const ExpertTierWeightProjectionIntersection &overlap)
        {
            namespace sb = native_quantised_superblock;
            const auto width = geometry.payload_bytes_per_block;
            const auto cb = geometry.codebook_id;
            const auto bits = sb::superblockCodePlaneBits(cb);
            const auto field = sb::superblockFieldBytes(cb);
            for (int kb = 0; kb < overlap.columns() / 32; ++kb)
                for (int row = 0; row < overlap.rows(); ++row)
                {
                    const auto src = sb::blockMajorAddress(kb + overlap.sourceColumnOrigin() / 32,
                        row + overlap.sourceRowOrigin(), source.geometry.n,
                        nativeQuantisedLayoutPhase(source.geometry.metadata_layout));
                    const auto dst = sb::blockMajorAddress(kb + overlap.destinationColumnOrigin() / 32,
                        row + overlap.destinationRowOrigin(), geometry.n,
                        nativeQuantisedLayoutPhase(geometry.metadata_layout));
                    std::copy_n(source.payload.data() + src.linear * width, width, payload.data() + dst.linear * width);
                    std::copy_n(source.headers.data() + src.superblock * field, field, headers.data() + dst.superblock * field);
                    if (bits)
                        putBits(codes.data(), dst.linear * bits + nativeQuantisedLayoutCodeBitShift(geometry.metadata_layout),
                            bits, sb::loadPackedBits(source.codes.data(),
                                src.linear * bits + nativeQuantisedLayoutCodeBitShift(source.geometry.metadata_layout), bits));
                }
        }
    };
} // namespace llaminar2::test
