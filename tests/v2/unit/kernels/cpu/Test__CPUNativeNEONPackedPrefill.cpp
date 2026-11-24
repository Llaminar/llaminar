/**
 * @file Test__CPUNativeNEONPackedPrefill.cpp
 * @brief Byte-exact transient NativeQuantised prefill against the serial NEON leaf.
 *
 * The same worker bank is poisoned and reused across the supported dot families.
 * Canonical K ordering, guarded N tails and bounded team scratch are part of
 * the tested contract, independent of how the compact bytes were prepared.
 */
#include "kernels/cpu/gemm/CPUNativeNEONPackedPrefill.h"
#include "utils/NativeQuantisedPreparedByteFixture.h"
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cstring>
#include <span>
#include <vector>

namespace llaminar2::test
{
namespace
{
using namespace cpu::native_quantised;
namespace ms = cpu::native_quantised::multi_scale;

/** @brief Build retained Q8 test rows with full-byte extrema and exact integer sums.
 * @param blocks Logical K blocks per row.
 * @param rows Independent activation row count.
 * @return One owned contiguous quantized input stream.
 */
std::vector<Q8_1Block> activationRows(int blocks, int rows)
{
    std::vector<Q8_1Block> activation(std::size_t(blocks) * rows);
    constexpr std::array<std::uint16_t, 4> scales{0x0001, 0x3c01, 0x2aa1, 0x0401};
    for (int row = 0; row < rows; ++row)
        for (int block = 0; block < blocks; ++block)
        {
            auto &a = activation[std::size_t(row) * blocks + block];
            a.d = scales[(row + block) % scales.size()];
            int sum = 0;
            for (int k = 0; k < 32; ++k)
            {
                const int value = (row * 37 + block * 19 + k * 53) % 256 - 128;
                a.qs[k] = ((row + block + k / 8) % 7 == 0) ? 0 : static_cast<std::int8_t>(value);
                sum += a.qs[k];
            }
            a.sum_qs = static_cast<std::int16_t>(sum);
        }
    return activation;
}
} // namespace

/** @test Packed prefill overwrites dirty weight/carry storage, preserving serial bytes and N guards.
 * @details Covers odd row pairs, a ragged final source superblock, empty ordered
 * partitions and a true ordered partition that crosses the 16-block panel.
 */
TEST(CPUNativeNEONPackedPrefill, MatchesSerialAcrossPanelsAndTails)
{
    constexpr auto serial = ms::NEONByteDot::ASIMD;
    constexpr std::uint32_t dirty_bits = 0x7fc12345u;
    constexpr float dirty = std::bit_cast<float>(dirty_bits);
    struct Case { int rows, n, blocks; };
    constexpr Case cases[]{{16, 65, 9}, {17, 66, 17}, {19, 67, 257}};
    for (const auto *source : {&native_quantised_formats::IQ3_S, &native_quantised_formats::Q5_K,
         &native_quantised_formats::Q5_0})
        for (const auto &cell : cases)
            for (const auto policy : {CPUProjectionNumericalPolicy::BackendNative, CPUProjectionNumericalPolicy::GPUAlignedExpert})
            {
                const auto [rows, n, blocks] = cell;
                SCOPED_TRACE(::testing::Message() << "codebook=" << int(source->codebook_id) << " m=" << rows
                    << " n=" << n << " kblocks=" << blocks << " policy=" << int(policy));
                NativeQuantisedPreparedByteFixture fixture(*source, n, blocks * 32, 4u * 1024u * 1024u);
                fixture.descriptor.numerical_policy = policy;
                fixture.seedPayloadAndMetadata();
                const auto weights = fixture.view();
                const auto activation = activationRows(blocks, rows);
                const int stride = n + 3;
                std::vector<float> result(rows * stride, dirty);
                std::vector<float> expected(result);
                for (int row = 0; row < rows; ++row)
                    for (int chunk = 0; chunk < weights.N_padded / 64; ++chunk)
                    {
                        const int live = std::min(64, n - 64 * chunk);
                        const std::array<std::span<const Q8_1Block>, 1> a{{{activation.data() + row * blocks, std::size_t(blocks)}}};
                        const std::array<std::span<float>, 1> o{{{expected.data() + row * stride + 64 * chunk, std::size_t(live)}}};
                        CPUNativeNEONProjectionTile<1>::bind(weights, a, o, chunk).execute<serial>();
                    }
                const auto batch = CPUNativeNEONProjectionBatch::bind(weights, activation,
                    std::span(result).first(std::size_t(rows - 1) * stride + n), rows, stride, NEONProjectionRowTile::Pair);
                const int workers = omp_get_max_threads();
                const auto demand = neon_panel::workspaceFloats(weights.encoding, weights.codebook_id,
                    weights.is_asymmetric, rows, weights.K, workers, policy);
                std::vector<float> slab(demand + 32, dirty);
                const auto aligned = reinterpret_cast<std::uintptr_t>(slab.data() + 1);
                std::span<float> scratch(reinterpret_cast<float *>((aligned + 63) & ~std::uintptr_t(63)), demand);
                ASSERT_THROW((void)CPUNativeNEONPackedPrefill::bind(batch, scratch.first(demand - 1), workers),
                    std::invalid_argument);
                const auto prefill = CPUNativeNEONPackedPrefill::bind(batch, scratch, workers);
                const auto compare = [&]<ms::NEONByteDot Dot> {
                    std::fill(scratch.begin(), scratch.end(), std::bit_cast<float>(0x7fa00ee1u));
                    std::fill(result.begin(), result.end(), dirty);
                    prefill.execute<Dot>();
                    EXPECT_EQ(std::memcmp(expected.data(), result.data(), result.size() * sizeof(float)), 0);
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(scratch.data()[-1]), dirty_bits);
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(scratch.data()[demand]), dirty_bits);
                };
                compare.template operator()<serial>();
#if defined(__ARM_FEATURE_DOTPROD)
                compare.template operator()<ms::NEONByteDot::DotProd>();
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
                compare.template operator()<ms::NEONByteDot::I8MM>();
#endif
            }
}

} // namespace llaminar2::test
