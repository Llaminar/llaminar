/**
 * @file Test__NativeQuantisedFileDensityAllFormats.cpp
 * @brief Audit the logical prepared storage of every cataloged GGUF format.
 *
 * No weights or GPU runtime are needed: CPU unit geometry and GPU pool-plane
 * geometry are the same contracts used by the production packers. Compare
 * matrix bytes before allocator alignment, not allocation BOMs or staging.
 */
#include "kernels/cpu/gemm/CPUNativeQuantisedEncoding.h"
#include "loaders/GPUPreparedWeightPoolLayout.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include <string_view>

namespace llaminar2
{
namespace
{
    namespace cpu_nq = cpu::native_quantised;
    using namespace native_quantised_formats;

    /** @brief Match the source identity to its actual on-disk type, not a prepared byte count. */
    std::size_t ggufBytesPer256(const NativeQuantisedFormatInfo *format)
    {
        if (format == &IQ4_NL) return sizeof(IQ4_NLBlock) * (256 / IQ4_NLBlock::BLOCK_SIZE);
        if (format == &Q8_0) return sizeof(Q8_0Block) * (256 / Q8_0Block::BLOCK_SIZE);
        if (format == &Q8_1) return sizeof(Q8_1Block) * (256 / Q8_1Block::BLOCK_SIZE);
        if (format == &Q4_0) return sizeof(Q4_0Block) * (256 / Q4_0Block::BLOCK_SIZE);
        if (format == &Q4_1) return sizeof(Q4_1Block) * (256 / Q4_1Block::BLOCK_SIZE);
        if (format == &Q5_0) return sizeof(Q5_0Block) * (256 / Q5_0Block::BLOCK_SIZE);
        if (format == &Q5_1) return sizeof(Q5_1Block) * (256 / Q5_1Block::BLOCK_SIZE);
        if (format == &Q6_K) return sizeof(Q6_KBlock) * (256 / Q6_KBlock::BLOCK_SIZE);
        if (format == &Q2_K) return sizeof(Q2_KBlock) * (256 / Q2_KBlock::BLOCK_SIZE);
        if (format == &Q5_K) return sizeof(Q5_KBlock) * (256 / Q5_KBlock::BLOCK_SIZE);
        if (format == &Q3_K) return sizeof(Q3_KBlock) * (256 / Q3_KBlock::BLOCK_SIZE);
        if (format == &Q4_K) return sizeof(Q4_KBlock) * (256 / Q4_KBlock::BLOCK_SIZE);
        if (format == &Q8_K) return sizeof(Q8_KBlock) * (256 / Q8_KBlock::BLOCK_SIZE);
        if (format == &IQ4_XS) return sizeof(IQ4_XSBlock) * (256 / IQ4_XSBlock::BLOCK_SIZE);
        if (format == &IQ2_XXS) return sizeof(IQ2_XXSBlock) * (256 / IQ2_XXSBlock::BLOCK_SIZE);
        if (format == &IQ2_XS) return sizeof(IQ2_XSBlock) * (256 / IQ2_XSBlock::BLOCK_SIZE);
        if (format == &IQ3_XXS) return sizeof(IQ3_XXSBlock) * (256 / IQ3_XXSBlock::BLOCK_SIZE);
        if (format == &IQ2_S) return sizeof(IQ2_SBlock) * (256 / IQ2_SBlock::BLOCK_SIZE);
        if (format == &IQ3_S) return sizeof(IQ3_SBlock) * (256 / IQ3_SBlock::BLOCK_SIZE);
        if (format == &IQ1_S) return sizeof(IQ1_SBlock) * (256 / IQ1_SBlock::BLOCK_SIZE);
        if (format == &IQ1_M) return sizeof(IQ1_MBlock) * (256 / IQ1_MBlock::BLOCK_SIZE);
        if (format == &MXFP4) return sizeof(MXFP4Block) * (256 / MXFP4Block::BLOCK_SIZE);
        return 0;
    }

    /** @return CPU byte limit for 256 values, excluding unused source fields. */
    std::size_t cpuBytesPer256Limit(const NativeQuantisedFormatInfo *format, std::size_t gguf)
    {
        // Q8_1's source sum s is derived from the payload and unused for weights.
        if (format == &Q8_1) return 8 * (2 + 32);
        // Q8_K's bsums are unused; projection already stages d as FP16 per 32 signed bytes.
        if (format == &Q8_K) return 8 * (2 + 32);
        return gguf;
    }

    TEST(NativeQuantisedFileDensityAllFormats, LogicalPreparedPlanesDoNotExceedGGUF)
    {
        constexpr unsigned k = 256; // One complete source superblock, or eight ordinary blocks.
        constexpr unsigned blocks = k / 32;
        ASSERT_EQ(kAllSourceFormats.size(), 22u);
        std::cout << "format GGUF CPU CUDA_SourceNative CUDA_MigrationReusable (bytes per 32; excludes pad)\n";
        for (const auto &[name, format] : kAllSourceFormats)
        {
            SCOPED_TRACE(name);
            const auto gguf = ggufBytesPer256(format);
            ASSERT_GT(gguf, 0u);
            const auto footprint = cpu_nq::preparedFootprintForFormat(*format);
            ASSERT_TRUE(footprint.isValid());
            const auto geometry = cpu_nq::preparedBlockGeometry(
                footprint.encoding, footprint.is_asymmetric, format->codebook_id, blocks);

            for (const std::size_t n : {64u, 65u})
            {
                SCOPED_TRACE(n);
                const std::size_t chunks = (n + 63) / 64;
                const std::size_t padded_n = chunks * 64;
                const auto cpu = geometry.matrixBytes(chunks);
                EXPECT_LE(cpu, padded_n * cpuBytesPer256Limit(format, gguf));

                auto source = nativeQuantisedPackedRegionSizes(n, k, *format);
                std::size_t code_pad = 0;
                const auto physical = canonicalDeviceVnniCodebookId(*format);
                if (gpuPreparedSuperblockCoded(*format, GPUPreparedWeightPoolLayout::SourceNative, DeviceId::cuda(0)))
                {
                    const auto planes = NativeQuantisedMetadataPlaneBytes::coded(physical, n, blocks, 0);
                    source.scales_bytes = planes.scales;
                    source.mins_bytes = planes.secondary;
                    source.emins_bytes = planes.emins;
                    if (native_quantised_superblock::superblockCodePlaneBits(physical))
                        code_pad = native_quantised_superblock::kSuperblockCodePlanePadBytes;
                }
                EXPECT_LE(source.logicalBytes(), n * gguf + code_pad);

                const auto allocation = gpuPreparedWeightPoolAllocation(*format, GPUPreparedWeightPoolLayout::MigrationReusable);
                const bool migration_coded = gpuPreparedSuperblockCoded(
                    *format, GPUPreparedWeightPoolLayout::MigrationReusable, DeviceId::cuda(0));
                const auto migration_pad = migration_coded ? code_pad : 0u;
                const auto migration_planes = migration_coded
                    ? NativeQuantisedMetadataPlaneBytes::coded(physical, n, blocks, 0)
                    : NativeQuantisedMetadataPlaneBytes::of(n * blocks, physical, allocation.has_mins, allocation.has_emins);
                const NativeQuantisedPackedRegionSizes migration{
                    n * blocks * allocation.payload_bytes_per_block,
                    migration_planes.scales, migration_planes.secondary, migration_planes.emins};
                if (migration_coded || format == &Q5_0 || format == &Q5_1 || format == &Q8_1 || format == &Q8_K)
                    EXPECT_EQ(migration.logicalBytes(), n * cpuBytesPer256Limit(format, gguf) + migration_pad);
                else
                    EXPECT_LE(migration.logicalBytes(), n * gguf);

                if (n == 64)
                    std::cout << name << ' ' << double(gguf) / blocks << ' '
                              << double(cpu) / (padded_n * blocks) << ' '
                              << double(source.logicalBytes() - code_pad) / (n * blocks) << ' '
                              << double(migration.logicalBytes() - migration_pad) / (n * blocks) << '\n';
            }
        }
    }
}
}
