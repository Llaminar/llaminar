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
#include "utils/CPUExplicitRoundingOracle.h"

#include <array>
#include <cstring>
#include <omp.h>

namespace llaminar2::test
{
namespace
{
using cpu::attention_key_q8::ISA;

/** @brief Enumerate compiled native implementations without advertising foreign leaves.
 * @return Every available explicit route plus canonical automatic dispatch.
 */
std::vector<ISA> implementations()
{
    std::vector<ISA> result{ISA::Automatic};
#if LLAMINAR_COMPILED_WITH_NEON
    if (cpu_supports_neon()) result.push_back(ISA::NEON);
#endif
#if defined(__AVX2__)
    if (cpu_supports_avx2()) result.push_back(ISA::AVX2);
#endif
#if defined(__AVX512F__)
    if (cpu_supports_avx512()) result.push_back(ISA::AVX512);
#endif
    return result;
}

/** @brief Prove all head bytes and reconstruction bytes independently of scheduling.
 * @tparam D Complete production physical head width.
 */
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
                int observed_workers = 0;
#pragma omp parallel num_threads(workers)
                {
#pragma omp single
                    observed_workers = omp_get_num_threads();
#pragma omp for schedule(static)
                    for (int head = 0; head < heads; ++head)
                    {
                        const auto anchor = std::span<const float, D>(anchors.data() + head * D, D);
                        cpu::attention_key_q8::quantize<D>(
                            std::span<const float, D>(source.data() + head * D, D), anchor, blocks[head], isa);
                        cpu::attention_key_q8::dequantize<D>(
                            blocks[head], anchor, std::span<float, D>(actual.data() + head * D, D), isa);
                    }
                }
                ASSERT_EQ(observed_workers, workers);
                ASSERT_EQ(std::memcmp(blocks.data(), expected.data(), blocks.size() * sizeof(blocks[0])), 0);
                ASSERT_EQ(std::memcmp(actual.data(), decoded.data(), actual.size() * sizeof(float)), 0);
            }
        }
    }
}

/** @brief SIMD lane searches must agree at and immediately beside every bin boundary.
 * @tparam D Complete production physical head width.
 */
template <int D>
void boundaries()
{
    std::array<float, D> anchor{}, source{};
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

/** @brief Finite FP32 exponent extremes preserve the rational oracle's decisions.
 * @tparam D Complete production physical head width.
 */
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

/** @brief Exercise the complete zero-residual production branch with retained nonzero/signed-zero anchors.
 * @tparam D Actual complete physical key-head width.
 */
template <int D> void zeroResiduals()
{
    for (int variant = 0; variant < 3; ++variant)
    {
        std::array<float, D> source{}, anchor{}, expected_decoded{}, actual_decoded{};
        for (int coordinate = 0; coordinate < D; ++coordinate)
        {
            anchor[coordinate] = variant == 0 ? float(coordinate % 17 - 8) * 0.03125f
                : std::copysign(0.0f, coordinate % 2 ? -1.0f : 1.0f);
            source[coordinate] = variant == 2 ? -anchor[coordinate] : anchor[coordinate];
            expected_decoded[coordinate] = memoryRoundedAdd(0.0f, anchor[coordinate]);
        }
        const auto original_source = source, original_anchor = anchor;
        const AttentionKeyQ8Block<D> expected{};
        for (const ISA isa : implementations())
        {
            AttentionKeyQ8Block<D> actual;
            std::memset(&actual, 0xa5, sizeof(actual));
            actual_decoded.fill(919.25f);
            cpu::attention_key_q8::quantize<D>(source, anchor, actual, isa);
            EXPECT_EQ(std::memcmp(&actual, &expected, sizeof(actual)), 0);
            cpu::attention_key_q8::dequantize<D>(actual, anchor, actual_decoded, isa);
            EXPECT_EQ(std::memcmp(actual_decoded.data(), expected_decoded.data(), sizeof(actual_decoded)), 0);
            EXPECT_EQ(std::memcmp(source.data(), original_source.data(), sizeof(source)), 0);
            EXPECT_EQ(std::memcmp(anchor.data(), original_anchor.data(), sizeof(anchor)), 0);
        }
    }
}

/** @test Exactly zero residuals author every byte, including positive-zero scale and all-zero codes. */
TEST(CPUAttentionKeyQ8, ZeroResidualRetainsAnchorsAndAuthorsEveryByte)
{
    zeroResiduals<64>(); zeroResiduals<128>(); zeroResiduals<256>();
}

/** @brief Validate every original coordinate before any block publication, including anchor/overflow errors.
 * @tparam D Actual complete cache-head geometry.
 */
template <int D> void anchoredFailures()
{
    std::array<float, D> source{}, anchor{};
    for (const ISA isa : implementations())
        for (int coordinate = 0; coordinate < D; ++coordinate)
            for (int invalid_case = 0; invalid_case < 4; ++invalid_case)
            {
                source.fill(0); anchor.fill(0);
                if (invalid_case == 0) anchor[coordinate] = std::numeric_limits<float>::infinity();
                if (invalid_case == 1) anchor[coordinate] = std::numeric_limits<float>::quiet_NaN();
                if (invalid_case == 2)
                { source[coordinate] = std::numeric_limits<float>::max(); anchor[coordinate] = -std::numeric_limits<float>::max(); }
                if (invalid_case == 3)
                { source[coordinate] = -std::numeric_limits<float>::max(); anchor[coordinate] = std::numeric_limits<float>::max(); }
                AttentionKeyQ8Block<D> output;
                std::memset(&output, 0x5a, sizeof(output)); const auto before = output;
                EXPECT_THROW(cpu::attention_key_q8::quantize<D>(source, anchor, output, isa), std::domain_error);
                EXPECT_EQ(std::memcmp(&before, &output, sizeof(output)), 0);
            }
}

/** @test Corrupt anchors and finite subtraction overflow cannot partially author a physical block. */
TEST(CPUAttentionKeyQ8, AnchoredFailuresRejectBeforePublicationForEveryWidth)
{
    anchoredFailures<64>(); anchoredFailures<128>(); anchoredFailures<256>();
}

/** @test Codec outputs preserve both original cache authorities before any access through aliased spans. */
TEST(CPUAttentionKeyQ8, SourceAnchorAndCompressedStorageAliasesRejectBeforeMutation)
{
    std::array<float, 65> source{}, anchor{};
    source.fill(0.125f); anchor.fill(0.25f);
    const auto original_source = source, original_anchor = anchor;
    AttentionKeyQ8Block<64> block;
    std::memset(&block, 0x5a, sizeof(block)); const auto original_block = block;
    for (const ISA isa : implementations())
    {
        const auto compressed_as_source = std::span<const float, 64>(reinterpret_cast<const float *>(&block), 64);
        const auto compressed_as_output = std::span<float, 64>(reinterpret_cast<float *>(&block), 64);
        EXPECT_THROW(cpu::attention_key_q8::quantize<64>(compressed_as_source, std::span<const float, 64>(anchor.data(), 64), block, isa), std::invalid_argument);
        EXPECT_THROW(cpu::attention_key_q8::quantize<64>(std::span<const float, 64>(source.data(), 64), compressed_as_source, block, isa), std::invalid_argument);
        EXPECT_THROW(cpu::attention_key_q8::dequantize<64>(block, std::span<const float, 64>(anchor.data(), 64), compressed_as_output, isa), std::invalid_argument);
        EXPECT_THROW(cpu::attention_key_q8::dequantize<64>(block, std::span<const float, 64>(anchor.data(), 64), std::span<float, 64>(anchor.data(), 64), isa), std::invalid_argument);
        EXPECT_THROW(cpu::attention_key_q8::dequantize<64>(block, std::span<const float, 64>(anchor.data(), 64), std::span<float, 64>(anchor.data() + 1, 64), isa), std::invalid_argument);
        EXPECT_EQ(std::memcmp(&block, &original_block, sizeof(block)), 0);
        EXPECT_EQ(std::memcmp(source.data(), original_source.data(), sizeof(source)), 0);
        EXPECT_EQ(std::memcmp(anchor.data(), original_anchor.data(), sizeof(anchor)), 0);
    }
}

/** @brief Check every legal signed code and representative physical scale without a SIMD-derived oracle.
 * @tparam D Actual physical cache-head width.
 */
template <int D> void codeDomains()
{
    std::array<float, D> anchor{}, expected{}, actual{};
    AttentionKeyQ8Block<D> block{};
    for (int exponent : {-149, -126, -16, 0, 16, 112})
        for (int start = 0; start < 255; start += D)
        {
            block.quadratic_scale = std::ldexp(1.0f, exponent);
            for (int coordinate = 0; coordinate < D; ++coordinate)
            {
                block.codes[coordinate] = static_cast<std::int8_t>((start + coordinate) % 255 - 127);
                anchor[coordinate] = float(coordinate % 13 - 6) * 0.03125f;
            }
            attentionKeyQ8DequantizeReference<D>(block, expected);
            for (int coordinate = 0; coordinate < D; ++coordinate) expected[coordinate] += anchor[coordinate];
            for (const ISA isa : implementations())
            {
                actual.fill(919.25f);
                cpu::attention_key_q8::dequantize<D>(block, anchor, actual, isa);
                EXPECT_EQ(std::memcmp(actual.data(), expected.data(), sizeof(actual)), 0);
            }
        }
}

/** @test Every legal signed byte decodes in the shared two-multiply/add program. */
TEST(CPUAttentionKeyQ8, EveryLegalCodeAndPhysicalScaleMatchesScalarBytes)
{
    codeDomains<64>(); codeDomains<128>(); codeDomains<256>();
}

TEST(CPUAttentionKeyQ8, AutomaticDispatchHonorsTheCanonicalISAPolicy)
{
    ISA expected = ISA::Automatic;
    switch (activeISALevel())
    {
    case ISALevel::AVX512: expected = ISA::AVX512; break;
    case ISALevel::AVX2: expected = ISA::AVX2; break;
    case ISALevel::NEON:
    case ISALevel::NEONDotProd: expected = ISA::NEON; break;
    default: FAIL() << "AQ8 production conformance requires a native ISA";
    }
    EXPECT_EQ(cpu::attention_key_q8::resolveISA(ISA::Automatic), expected);
    const auto available = implementations();
    for (ISA requested : {ISA::AVX2, ISA::AVX512, ISA::NEON, static_cast<ISA>(255)})
        if (std::find(available.begin(), available.end(), requested) == available.end())
            EXPECT_THROW(cpu::attention_key_q8::resolveISA(requested), std::invalid_argument);
}
} // namespace
} // namespace llaminar2::test
