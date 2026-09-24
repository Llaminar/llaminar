/**
 * @file Test__CPUAttentionKeyQ8.cpp
 * @brief Native CPU AQ8 byte conformance, ISA dispatch and concurrency proofs.
 *
 * These model-free integration checks exercise the production SIMD codec, not
 * an alternate tensor oracle. The scalar authority independently authors every
 * expected block and decoded coordinate, including rational-bin boundaries.
 * Request/ring/prefix lifecycle remains covered by the production cache suite.
 */
#include <gtest/gtest.h>
#include "kernels/cpu/attention/CPUAttentionKeyQ8.h"
#include "../AttentionKeyQ8DeviceTestCommon.h"

#include <array>
#include <cstring>
#include <omp.h>

namespace llaminar2::test
{
namespace
{
using cpu::attention_key_q8::ISA;

/** @brief Enumerate every available explicit implementation plus runtime dispatch. */
std::vector<ISA> implementations()
{
    std::vector<ISA> result{ISA::Automatic, ISA::AVX2};
    if (cpu_supports_avx512()) result.push_back(ISA::AVX512);
    return result;
}

/** @brief Prove all head bytes and reconstruction bytes independently of scheduling. */
template <int D>
void conformance()
{
    // Adjacent anchors vary rather than assuming a zero bias. The common GPU
    // fixture still supplies zeros, negative codes and model-scale outliers.
    for (const int heads : {1, 2, 3, 15, 16, 17, 31, 64, 139})
    {
        const auto source = makeAttentionKeyQ8DeviceInput<D>(heads);
        std::vector<float> anchors(source.size()), residual(source.size());
        for (std::size_t index = 0; index < source.size(); ++index)
        {
            anchors[index] = index % 5 == 0 ? source[index] : static_cast<float>(index % 17) * 0.03125f;
            residual[index] = source[index] - anchors[index];
        }
        const auto expected = makeAttentionKeyQ8ReferenceBlocks<D>(residual);
        auto decoded = makeAttentionKeyQ8ReferenceDecoded<D>(expected);
        for (std::size_t index = 0; index < decoded.size(); ++index) decoded[index] += anchors[index];
        for (const ISA isa : implementations())
        {
            // Cover every positive team size in the CTest-owned physical-core
            // allocation, including odd teams and fewer heads than workers.
            for (int workers = 1; workers <= omp_get_max_threads(); ++workers)
            {
                SCOPED_TRACE(::testing::Message() << "D=" << D << " heads=" << heads
                             << " isa=" << static_cast<int>(isa) << " workers=" << workers);
                std::vector<AttentionKeyQ8Block<D>> blocks(heads);
                std::vector<float> actual(source.size());
#pragma omp parallel for num_threads(workers) schedule(static)
                for (int head = 0; head < heads; ++head)
                {
                    const auto anchor = std::span<const float, D>(anchors.data() + head * D, D);
                    cpu::attention_key_q8::quantize<D>(
                        std::span<const float, D>(source.data() + head * D, D), anchor, blocks[head], isa);
                    cpu::attention_key_q8::dequantize<D>(
                        blocks[head], anchor, std::span<float, D>(actual.data() + head * D, D), isa);
                }
                ASSERT_EQ(std::memcmp(blocks.data(), expected.data(), blocks.size() * sizeof(blocks[0])), 0);
                ASSERT_EQ(std::memcmp(actual.data(), decoded.data(), actual.size() * sizeof(float)), 0);
            }
        }
    }
}

/** @brief SIMD lane searches must agree at and immediately beside every bin boundary. */
template <int D>
void boundaries()
{
    std::array<float, D> anchor{}, source{}, residual{};
    for (const float scale : {0.0001f, 1.0f, 130.0f, 8192.0f})
    {
        for (int bin = 0; bin < 127; ++bin)
        {
            const float boundary = scale * static_cast<float>((2 * bin + 1) * (2 * bin + 1)) / (254.0f * 254.0f);
            for (int coordinate = 0; coordinate < D; ++coordinate)
            {
                const float magnitude = coordinate % 3 == 0 ? std::nextafter(boundary, 0.0f)
                    : coordinate % 3 == 1 ? boundary : std::nextafter(boundary, scale);
                source[coordinate] = coordinate % 2 ? -magnitude : magnitude;
            }
            source[0] = scale;
            AttentionKeyQ8Block<D> expected{};
            attentionKeyQ8QuantizeReference<D>(source, expected);
            for (const ISA isa : implementations())
            {
                AttentionKeyQ8Block<D> actual{};
                cpu::attention_key_q8::quantize<D>(source, anchor, actual, isa);
                ASSERT_EQ(std::memcmp(&expected, &actual, sizeof(actual)), 0)
                    << "D=" << D << " bin=" << bin << " scale=" << scale;
            }
        }
    }
}

TEST(CPUAttentionKeyQ8, AllWidthsISAsAndWorkerCountsMatchScalarBytes)
{
    conformance<64>(); conformance<128>(); conformance<256>();
}

TEST(CPUAttentionKeyQ8, EveryRationalBoundaryMatchesScalarBytes)
{
    boundaries<64>(); boundaries<128>(); boundaries<256>();
}

/** @brief Finite FP32 exponent extremes preserve the rational oracle's decisions. */
template <int D>
void exponentExtremes()
{
    std::array<float, D> source{}, anchor{};
    for (int exponent = -149; exponent <= 127; ++exponent)
    {
        const float scale = std::ldexp(1.0f, exponent);
        for (int coordinate = 0; coordinate < D; ++coordinate)
            source[coordinate] = scale * (static_cast<float>(coordinate % 17) / 16.0f) * (coordinate % 2 ? -1.0f : 1.0f);
        source[0] = scale;
        AttentionKeyQ8Block<D> expected{};
        attentionKeyQ8QuantizeReference<D>(source, expected);
        for (const ISA isa : implementations())
        {
            AttentionKeyQ8Block<D> actual{};
            cpu::attention_key_q8::quantize<D>(source, anchor, actual, isa);
            ASSERT_EQ(std::memcmp(&expected, &actual, sizeof(actual)), 0)
                << "D=" << D << " exponent=" << exponent << " ISA=" << static_cast<int>(isa);
        }
    }
}

TEST(CPUAttentionKeyQ8, FiniteExponentRangeMatchesScalarBytes)
{
    exponentExtremes<64>(); exponentExtremes<128>(); exponentExtremes<256>();
}

TEST(CPUAttentionKeyQ8, InvalidInputCannotPublishPartOfABlock)
{
    std::array<float, 64> source{}, anchor{};
    for (const ISA isa : implementations())
    {
        for (const float invalid : {std::numeric_limits<float>::infinity(),
                                   -std::numeric_limits<float>::infinity(),
                                   std::numeric_limits<float>::quiet_NaN()})
        {
            for (int coordinate = 0; coordinate < 64; ++coordinate)
            {
                AttentionKeyQ8Block<64> actual;
                std::memset(&actual, 0x5a, sizeof(actual));
                const auto before = actual;
                source.fill(0.0f);
                source[coordinate] = invalid;
                EXPECT_THROW(cpu::attention_key_q8::quantize<64>(source, anchor, actual, isa), std::domain_error);
                EXPECT_EQ(std::memcmp(&before, &actual, sizeof(actual)), 0);
            }
        }
    }
}

TEST(CPUAttentionKeyQ8, AutomaticDispatchHonorsTheCanonicalISAPolicy)
{
    const ISA expected = activeISALevel() == ISALevel::AVX512 ? ISA::AVX512 : ISA::AVX2;
    EXPECT_EQ(cpu::attention_key_q8::resolveISA(ISA::Automatic), expected);
    if (!cpu_supports_avx512())
        EXPECT_THROW(cpu::attention_key_q8::resolveISA(ISA::AVX512), std::invalid_argument);
}
} // namespace
} // namespace llaminar2::test
