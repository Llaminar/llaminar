/**
 * @file NativeQuantisedPreparedByteFixture.h
 * @brief Small device-free canonical byte fixtures for native CPU leaf tests.
 *
 * The fixture owns exactly one aligned interleaved test stream and keeps source
 * identity separate from execution encoding. It is not a tensor preparer,
 * allocator substitute or performance fixture: full-engine tests must also use
 * real prepared owners/slots and source tensors. The four-MiB guard prevents a
 * malformed Unit geometry from becoming a model-sized allocation; kernel
 * tournaments raise it explicitly to time production-sized shapes.
 */
#pragma once

#include "kernels/cpu/gemm/CPUNativeQuantisedWeightView.h"
#include "tensors/IQQuantTables.h"
#include "tensors/MXFP4Format.h"
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace llaminar2::test
{
/** @brief Retain one bounded aligned byte publication without source/decoded mirrors. */
class NativeQuantisedPreparedByteFixture
{
public:
    /** @brief Construct a bounded canonical byte fixture for one source identity.
     * @param source Original registry metadata.
     * @param n Logical output columns.
     * @param k Logical reduction elements.
     * @param max_bytes Allocation bound; the default is the Unit budget.
     * @throws std::invalid_argument For nonpositive geometry.
     * @throws std::overflow_error If the requested fixture exceeds @p max_bytes.
     */
    explicit NativeQuantisedPreparedByteFixture(const NativeQuantisedFormatInfo &source, int n = 65, int k = 65,
        std::size_t max_bytes = 4u * 1024u * 1024u)
        : descriptor{.columns = n, .reduction_width = k,
            .source = {.codebook_id = source.codebook_id, .is_superblock = source.is_superblock, .present = true},
            .encoding = cpu::native_quantised::preparedEncodingForSource(source),
            .numerical_policy = CPUProjectionNumericalPolicy::GPUAlignedExpert,
            .is_asymmetric = source.is_asymmetric}
    {
        if (n <= 0 || k <= 0)
            throw std::invalid_argument("Native CPU byte fixture requires positive geometry");
        const std::uint64_t chunks = (std::uint64_t(n) + 63) / 64;
        const std::uint64_t blocks_per_row = (std::uint64_t(k) + 31) / 32;
        const auto stride = cpu::native_quantised::preparedInterleavedBlockStride(descriptor.encoding, descriptor.is_asymmetric, descriptor.source.codebook_id);
        const auto header = cpu::native_quantised::preparedSuperblockHeaderBytes(descriptor.encoding, descriptor.source.codebook_id);
        const cpu::native_quantised::CPUNativeQuantisedBlockGeometry geometry{
            .block_stride = stride, .header_bytes = header,
            .blocks_per_row = static_cast<std::size_t>(blocks_per_row), .phase = 0};
        if (stride == 0 || chunks > max_bytes / geometry.rowBytes())
            throw std::overflow_error("Native CPU byte fixture exceeds its declared byte bound");
        const auto extent = geometry.matrixBytes(chunks);
        storage_.resize(extent + 127);
        const auto offset = (64 - reinterpret_cast<std::uintptr_t>(storage_.data()) % 64) % 64;
        bytes = std::span(storage_).subspan(offset, extent);
    }
    NativeQuantisedPreparedByteFixture(const NativeQuantisedPreparedByteFixture &) = delete;
    NativeQuantisedPreparedByteFixture &operator=(const NativeQuantisedPreparedByteFixture &) = delete;

    /** @return A checked immutable view borrowing the same fixture root. */
    [[nodiscard]] cpu::native_quantised::CPUNativeQuantisedWeightView view() const
    {
        // Fixture bytes are sealed afresh on every view: tests mutate them freely.
        return cpu::native_quantised::CPUNativeQuantisedWeightView::fromPublished(descriptor, bytes,
            cpu::native_quantised::CPUNativeQuantisedMetadataSeal::validate(descriptor, bytes));
    }

    /**
     * @brief Fill live compact/native payloads and source-factor metadata directly.
     * @details This is an adversarial byte-layout fixture, not a source-quantizer
     * implementation. Grid/sign bytes span their full decoder domains; physical
     * tail columns remain zero initialized. Tests compare with independent
     * source-order integer decoders and separately memory-rounded FP oracles.
     */
    void seedPayloadAndMetadata()
    {
        const auto layout = view();
        for (int column = 0; column < layout.N; ++column)
            for (int block = 0; block < layout.blocks_per_row; ++block)
            {
                const int local = column % 64;
                const int chunk = column / 64;
                using namespace cpu::native_quantised;
                const auto unit = layout.geometry().blockOffset(chunk, block);
                const std::uint32_t seed = 0x9e3779b9u * (column + 1u) ^ 0x85ebca6bu * (block + 3u);
                const int words = layout.usesCompactMultiScale() ? preparedCompactPayloadWords(layout.codebook_id) : layout.payload_bytes / 4;
                const int tail = layout.usesCompactMultiScale() ? preparedCompactPayloadTailBytes(layout.codebook_id) : 0;
                const bool q5k = layout.usesCompactMultiScale() && layout.codebook_id == 7;
                const int payload_bytes = layout.usesCompactMultiScale() ? preparedCompactPayloadBytes(layout.codebook_id) : layout.payload_bytes;
                for (int byte = 0; byte < (q5k ? 16 : payload_bytes); ++byte)
                {
                    const auto offset = unit + (byte < 4 * words ? (byte / 4) * 256 + local * 4 + byte % 4
                        : words * 256 + local * tail + byte - 4 * words);
                    bytes[offset] = static_cast<std::uint8_t>((seed >> ((byte % 4) * 8)) + byte * 29u);
                }
                if (q5k)
                {
                    const std::uint8_t qh[]{static_cast<std::uint8_t>(seed), static_cast<std::uint8_t>(seed >> 8),
                        static_cast<std::uint8_t>(seed >> 16), static_cast<std::uint8_t>(seed >> 24)};
                    coded_metadata::storeQ5HighBits(bytes.data() + unit + 1024, local, qh);
                }
                if (layout.usesInlineCompensation())
                {
                    int sum = 0;
                    for (int byte = 0; byte < layout.payload_bytes; ++byte)
                    {
                        const auto value = bytes[unit + (byte / 4) * 256 + local * 4 + byte % 4];
                        if (layout.usesExpandedInt8())
                            sum += value < 128 ? value : int(value) - 256;
                        else if (layout.codebook_id == 4)
                            sum += kvalues_iq4nl_i8[value & 15] + kvalues_iq4nl_i8[value >> 4];
                        else if (layout.codebook_id == kNativeQuantisedMXFP4Codebook)
                            sum += mxfp4::kValues[value & 15] + mxfp4::kValues[value >> 4];
                        else
                            sum += (value & 15) + (value >> 4) - (layout.codebook_id == 0 ? 16 : 0);
                    }
                    const auto compensation = static_cast<std::int16_t>(sum);
                    std::memcpy(bytes.data() + unit + layout.data_stride + local * 2,
                        &compensation, sizeof(compensation));
                }
                if (layout.superblockCoded())
                {
                    const auto codebook = superblockMetadataCodebook(layout.encoding, layout.codebook_id);
                    if (native_quantised_superblock::superblockCodePlaneBits(codebook))
                        coded_metadata::storeCode(bytes.data() + unit + preparedCodedPlaneOffset(layout.encoding, layout.data_stride),
                            codebook, local, seed);
                    if (block % 8 == 0)
                    {
                        const std::uint32_t d = 0x2401u + (seed & 0x1fffu);
                        const std::uint32_t word = d | (native_quantised_superblock::superblockHasDmin(codebook) ? d << 16 : 0u);
                        coded_metadata::storeSuperblockWord(bytes.data() + layout.geometry().headerOffset(chunk, block),
                            codebook, local, word);
                    }
                    continue;
                }
                const auto primary = unit + layout.data_stride + (layout.usesInlineCompensation() ? 128 : 0) + local * 2;
                if (layout.scaleEncoding() == CPUNativeQuantisedScaleEncoding::E8M0)
                {
                    bytes[unit + layout.data_stride + local] = seed % 29u == 7u ? 0u : 2u + (seed >> 8) % 199u;
                    continue;
                }
                writeWord(primary, static_cast<std::uint16_t>(0x2401u + (seed & 0x1fffu)));
                if (layout.is_asymmetric || layout.usesIQ3SExpandedSourceScale())
                    writeWord(primary + 128, layout.usesIQ3SExpandedSourceScale() ? static_cast<std::uint16_t>(seed & 15u)
                        : static_cast<std::uint16_t>(0xa301u + (seed & 0x0fffu)));
            }
    }

    cpu::native_quantised::CPUNativeQuantisedWeightDescriptor descriptor; ///< Mutable cold admission input for negative tests.
    std::span<std::uint8_t> bytes; ///< Single owned stream, writable only between synchronous kernel calls.

private:
    /** @brief Copy one test metadata word without imposing alignment or aliasing.
     * @param offset Checked byte offset inside the fixture's canonical metadata.
     * @param word Original half/coefficient bits.
     */
    void writeWord(std::size_t offset, std::uint16_t word)
    { std::memcpy(bytes.data() + offset, &word, sizeof(word)); }

    std::vector<std::uint8_t> storage_; ///< Sole test byte owner, including explicit alignment/negative-span padding.
};
} // namespace llaminar2::test
