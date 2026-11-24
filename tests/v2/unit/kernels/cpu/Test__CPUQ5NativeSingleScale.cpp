/**
 * @file Test__CPUQ5NativeSingleScale.cpp
 * @brief Native Q5_0/Q5_1 storage and unchanged projection arithmetic.
 *
 * The oracle decodes GGUF block fields directly. It models the former biased
 * INT8 dot as well as the signed dot, then applies the original FP program
 * under both CPU numerical policies. No alternate prepared path is needed.
 */
#include "kernels/cpu/gemm/CPUNativeQuantisedWeightPacker.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedContributionContract.h"
#include "../../../utils/TestTensorFactory.h"
#if defined(__aarch64__) && defined(__ARM_NEON)
#include "kernels/cpu/gemm/CPUNativeNEONPreparedProjection.h"
#else
#include "kernels/cpu/gemm/CPUNativeQuantisedQ5X86Kernels.h"
#endif

#include <gtest/gtest.h>
#include <cmath>
#include <random>

namespace llaminar2::test
{
namespace
{
    using namespace cpu::native_quantised;
    using Policy = CPUProjectionNumericalPolicy;

    /** @brief Reproduce the former prepared dot, without accessing the new layout. */
    template <class Block>
    float referenceContribution(const Block &w, const Q8_1Block &a, float prior, Policy policy)
    {
        constexpr bool asymmetric = std::is_same_v<Block, Q5_1Block>;
        int biased = 0, compensation = 0, signed_dot = 0;
        for (int i = 0; i < 32; ++i)
        {
            const int code = ((w.qs[i % 16] >> (i < 16 ? 0 : 4)) & 15) +
                16 * ((w.qh[i / 8] >> (i % 8)) & 1);
            const int q = code - (asymmetric ? 0 : 16);
            biased += (int(a.qs[i]) + 128) * q;
            compensation += q;
            signed_dot += int(a.qs[i]) * q;
        }
        const float dot = static_cast<float>(biased - 128 * compensation);
        EXPECT_EQ(int(dot), signed_dot);
        namespace exact = device_fp32_contract;
        const float act = nativeQuantisedFP16ScaleToFP32(a.d);
        const float scale = exact::multiply(act, fp16_to_fp32(w.d));
        if (policy == Policy::BackendNative)
        {
            float result = std::fma(dot, scale, prior);
            if constexpr (asymmetric)
                result = std::fma(exact::multiply(act, float(a.sum_qs)), fp16_to_fp32(w.m), result);
            return result;
        }
        float term = exact::multiply(scale, dot);
        if constexpr (asymmetric)
            term = exact::add(term, exact::multiply(exact::multiply(act, fp16_to_fp32(w.m)), float(a.sum_qs)));
        return exact::add(prior, term);
    }

    /** @brief Compare a native live-row tile against the old program in increasing partition order. */
    template <std::size_t Rows, class Block>
    void checkTile(const CPUNativeQuantisedPackedWeights &packed, const Block *raw,
        const std::array<std::vector<Q8_1Block>, 4> &activations, int chunk)
    {
        const int live = std::min(64, packed.N - 64 * chunk);
        const int blocks = packed.blocks_per_row;
        const auto policy = packed.numerical_policy;
        const int parts = policy == Policy::BackendNative ? 1 :
            MoEProjectionNumericalContract::orderedKPartitionsForWidth(packed.K);
        const int width = (blocks + parts - 1) / parts;
        std::array<std::array<float, 64>, Rows> actual{};
#if defined(__aarch64__) && defined(__ARM_NEON)
        std::array<std::span<const Q8_1Block>, Rows> input;
        std::array<std::span<float>, Rows> output;
        for (std::size_t row = 0; row < Rows; ++row)
        {
            input[row] = activations[row];
            output[row] = {actual[row].data(), static_cast<std::size_t>(live)};
        }
        const auto tile = CPUNativeNEONProjectionTile<Rows>::bind(packed.readView(), input, output, chunk);
#if defined(__ARM_FEATURE_MATMUL_INT8)
        if constexpr (Rows > 1)
            tile.template execute<multi_scale::NEONByteDot::I8MM>();
        else
            tile.template execute<multi_scale::NEONByteDot::DotProd>();
#elif defined(__ARM_FEATURE_DOTPROD)
        tile.template execute<multi_scale::NEONByteDot::DotProd>();
#else
        tile.execute();
#endif
#else
        std::array<const Q8_1Block *, Rows> input;
        std::array<float *, Rows> output;
        std::array<std::array<float, 64>, Rows> partial{};
        for (std::size_t row = 0; row < Rows; ++row)
        {
            input[row] = activations[row].data();
            output[row] = partial[row].data();
        }
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        constexpr int lanes = 16;
#else
        constexpr int lanes = 8;
#endif
        for (int p = 0; p < parts; ++p)
        {
            q5::dispatch<lanes>(packed, input, output, chunk, p * width, std::min(blocks, (p + 1) * width), false);
            for (std::size_t row = 0; row < Rows; ++row)
                for (int n = 0; n < live; ++n)
                    actual[row][n] = p == 0 ? partial[row][n] : device_fp32_contract::add(actual[row][n], partial[row][n]);
        }
#endif
        for (std::size_t row = 0; row < Rows; ++row)
            for (int n = 0; n < live; ++n)
            {
                const auto *source = raw + (64 * chunk + n) * blocks;
                float expected = 0.0f;
                for (int p = 0; p < parts; ++p)
                {
                    float partial = 0.0f;
                    for (int b = p * width; b < std::min(blocks, (p + 1) * width); ++b)
                        partial = referenceContribution(source[b], activations[row][b], partial, policy);
                    expected = p == 0 ? partial : device_fp32_contract::add(expected, partial);
                }
                EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[row][n]), std::bit_cast<std::uint32_t>(expected))
                    << "Rows=" << Rows << " row=" << row << " column=" << (64 * chunk + n);
            }
    }

    /** @brief Exercise exact native packing and projection on both full and tail N tiles. */
    template <class Tensor>
    void checkSource(const Tensor &source)
    {
        const int blocks = static_cast<int>(source.shape()[1] / 32);
        const auto *raw = source.blocks();
        std::mt19937 rng(221);
        std::array<std::vector<Q8_1Block>, 4> activations;
        for (auto &row : activations)
        {
            row.resize(blocks);
            for (auto &block : row)
            {
                block.d = 0x2000 + rng() % 0x1800;
                block.sum_qs = 0;
                for (auto &q : block.qs)
                {
                    q = int(rng() % 256) - 128;
                    block.sum_qs += q;
                }
            }
        }
        for (const auto policy : {Policy::BackendNative, Policy::GPUAlignedExpert})
        {
            SCOPED_TRACE(static_cast<int>(policy));
            CPUNativeQuantisedPackedWeights packed;
            packed.numerical_policy = policy;
            ASSERT_TRUE(packWeightsCPUNativeQuantised(&source, packed));
            ASSERT_EQ(packed.encoding, CPUNativeQuantisedEncoding::Q5NativeSingleScale);
            EXPECT_FALSE(packed.usesInlineCompensation());
            EXPECT_EQ(packed.preparedBytes(), packed.N_padded * blocks * sizeof(*raw));
            EXPECT_EQ(packed.native_interleaved.size(), packed.preparedBytes());
            for (int chunk = 0; chunk < packed.N_padded / 64; ++chunk)
            {
                checkTile<1>(packed, raw, activations, chunk);
                checkTile<2>(packed, raw, activations, chunk);
                checkTile<3>(packed, raw, activations, chunk);
                checkTile<4>(packed, raw, activations, chunk);
            }
        }
    }

    TEST(CPUQ5NativeSingleScale, FileDensityAndByteExactProjection)
    {
        for (const std::size_t k : {288u, 544u})
        {
            SCOPED_TRACE(k);
            {
                SCOPED_TRACE("Q5_0");
                checkSource(*TestTensorFactory::createQ5_0Random({67, k}));
            }
            {
                SCOPED_TRACE("Q5_1");
                checkSource(*TestTensorFactory::createQ5_1Random({67, k}));
            }
        }
    }

#if defined(__aarch64__) && defined(__ARM_NEON)
    TEST(CPUQ5NativeSingleScale, WidenedScaleMatchesEveryStagedHalfWord)
    {
        using Ops = multi_scale::NEONMultiScaleOps<multi_scale::NEONByteDot::ASIMD>;
        for (unsigned first = 0; first < 65536; first += 4)
        {
            std::uint16_t word[4];
            for (unsigned i = 0; i < 4; ++i)
                word[i] = static_cast<std::uint16_t>(first + i);
            const auto widened = Ops::half(word);
            for (unsigned i = 0; i < 4; ++i)
            {
                const float reference = nativeQuantisedFP16ScaleToFP32(word[i]);
                if (std::isnan(reference))
                    EXPECT_TRUE(std::isnan(widened[i]));
                else
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(widened[i]), std::bit_cast<std::uint32_t>(reference));
            }
        }
    }
#endif
}
}
