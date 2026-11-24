/**
 * @file Test__CPUCodedFileDensityQ6IQ4.cpp
 * @brief Exact extent and source-factor arithmetic of coded Q6_K and IQ4_XS CPU weights.
 * The per-block reference uses the tensor's staged word publisher directly, without
 * borrowing the coded reader or installing a second prepared representation.
 */
#include <gtest/gtest.h>

#include "../../../utils/TestTensorFactory.h"
#include "execution/moe/ExpertTierWeightStream.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedMultiScaleKernels.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedWeightPacker.h"
#if defined(__aarch64__) && defined(__ARM_NEON)
#include "kernels/cpu/gemm/CPUNativeNEONPreparedProjection.h"
#endif

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace llaminar2::test
{
namespace
{
using namespace cpu::native_quantised;
namespace sb = native_quantised_superblock;
constexpr auto kIQ4XS = kNativeQuantisedIQ4XSSourceScaleCodebook;
constexpr int kN = 67;

/** @return Q6_K or IQ4_XS source with padded storage for a ragged final superblock. */
template <std::uint8_t Codebook> auto source(int k)
{
    if constexpr (Codebook == 8)
        return TestTensorFactory::createQ6_KRandom({kN, static_cast<std::size_t>(k)}, 3u);
    else
        return TestTensorFactory::createIQ4_XSRandom({kN, static_cast<std::size_t>(k)}, 7u);
}

/** @brief Count stored bytes independently of CPU geometry and check the final header position. */
template <std::uint8_t Codebook> void verifyExtent()
{
    constexpr std::size_t padded_columns = 128;
    constexpr std::size_t payload_and_codes = Codebook == 8 ? 26 * 64 : 16 * 64 + 48;
    constexpr std::size_t header = 2 * 64;
    for (int k : {32, 256, 288, 544, 1024})
    {
        SCOPED_TRACE(k);
        const auto tensor = source<Codebook>(k);
        CPUNativeQuantisedPackedWeights packed;
        ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), packed));
        const std::size_t blocks = k / 32;
        const std::size_t superblocks = (k + 255) / 256;
        const std::size_t row_bytes = payload_and_codes * blocks + header * superblocks;
        EXPECT_EQ(packed.interleaved_block_stride, payload_and_codes);
        EXPECT_EQ(packed.native_interleaved.size(), padded_columns / 64 * row_bytes);
        EXPECT_FALSE(packed.usesInlineCompensation());
        const auto view = packed.readView();
        EXPECT_EQ(view.superblockHeader(1, blocks - 1) - view.bytes().data(), 2 * row_bytes - header);
        if (k % 256 == 0)
        {
            constexpr auto file_superblock = Codebook == 8 ? 210u : 136u;
            EXPECT_EQ(view.bytes().size(),
                padded_columns * superblocks * file_superblock);
        }
    }
}

/** @return FP32 bits, treating different NaN payloads as equivalent. */
std::uint32_t comparableBits(float value)
{
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return (bits & 0x7fffffffu) > 0x7f800000u ? 0x7fc00000u : bits;
}

/** @brief Exercise every code against both signs, all half exponents and boundary mantissas. */
template <std::uint8_t Codebook, int Lanes> void verifySIMDFactors()
{
    using Ops = multi_scale::SIMD<Lanes>;
    using L = coded_metadata::Lanes<Ops>;
    constexpr auto encoding = Codebook == 8 ? CPUNativeQuantisedEncoding::Q6KNativeDualScale
                                           : CPUNativeQuantisedEncoding::NibbleLUTSourceScale;
    constexpr int codes = 1 << sb::superblockCodeBits(Codebook);
    for (int sign = 0; sign <= 1; ++sign)
        for (int exponent = 0; exponent < 32; ++exponent)
            for (int mantissa : {0, 1, 341, 1023})
            {
                const auto word = static_cast<std::uint16_t>((sign << 15) | (exponent << 10) | mantissa);
                typename Ops::I code{};
                const auto d = L::half(typename Ops::I{} + static_cast<int>(word));
                for (int first = 0; first < codes; first += Lanes)
                {
                    for (int lane = 0; lane < Lanes; ++lane)
                        code[lane] = first + lane;
                    const auto factors = coded_metadata::sourceScaleFactors<Codebook, Ops>(code, d);
                    for (int lane = 0; lane < Lanes; ++lane)
                    {
                        const auto staged = sb::decodeSuperblockMetadata<Codebook>(first + lane, word);
                        const auto expected0 = comparableBits(nativeQuantisedPrimaryScale(encoding, Codebook, staged.primary, staged.secondary));
                        const auto expected1 = comparableBits(nativeQuantisedSecondaryScale(encoding, Codebook, staged.primary, staged.secondary));
                        if (comparableBits(factors.primary[lane]) != expected0 || comparableBits(factors.secondary[lane]) != expected1)
                        {
                            ASSERT_EQ(comparableBits(factors.primary[lane]), expected0) << "half=" << word << " code=" << first + lane;
                            ASSERT_EQ(comparableBits(factors.secondary[lane]), expected1) << "half=" << word << " code=" << first + lane;
                        }
                    }
                }
            }
}

#if defined(__aarch64__) && defined(__ARM_NEON)
/** @brief Compare the production tile with per-block words and untouched source-order payload. */
template <std::uint8_t Codebook, std::size_t Rows>
void verifyProjection(const CPUNativeQuantisedPackedWeights &packed, const IINT8Unpackable &tensor)
{
    constexpr int k = 544, blocks = k / 32;
    constexpr int payload_bytes = Codebook == 8 ? 24 : 16;
    const auto view = packed.readView();
    const bool expert = view.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert;
    SCOPED_TRACE(::testing::Message() << "codebook=" << int(Codebook) << " rows=" << Rows << " expert=" << expert);
    std::array<std::array<Q8_1Block, blocks>, Rows> activation{};
    std::array<std::array<float, kN>, Rows> actual{};
    std::array<std::span<const Q8_1Block>, Rows> inputs;
    for (std::size_t row = 0; row < Rows; ++row)
    {
        for (int b = 0; b < blocks; ++b)
        {
            auto &a = activation[row][b];
            a.d = static_cast<std::uint16_t>(0x245bu + b * 17 + row * 43);
            for (int i = 0; i < 32; ++i)
            {
                a.qs[i] = static_cast<std::int8_t>((row * 31 + b * 13 + i * 49) % 256 - 128);
                a.sum_qs += a.qs[i];
            }
        }
        inputs[row] = activation[row];
    }
    for (int chunk = 0; chunk < 2; ++chunk)
    {
        std::array<std::span<float>, Rows> outputs;
        for (std::size_t row = 0; row < Rows; ++row)
            outputs[row] = std::span(actual[row]).subspan(chunk * 64, chunk == 0 ? 64 : kN - 64);
        CPUNativeNEONProjectionTile<Rows>::bind(view, inputs, outputs, chunk)
            .template execute<multi_scale::NEONByteDot::ASIMD>();
    }

    using namespace device_fp32_contract;
    const int partitions = expert ? MoEProjectionNumericalContract::orderedKPartitionsForWidth(k) : 1;
    const int span = (blocks + partitions - 1) / partitions;
    for (int column = 0; column < kN; ++column)
    {
        std::array<std::uint8_t, blocks * payload_bytes> payload{};
        std::array<std::uint16_t, blocks> primary{}, secondary{};
        const VnniPackContext staged{.N = 1, .K = k, .blocks_per_row = blocks, .payload_bytes = payload_bytes,
            .payload_array = payload.data(), .scales_array = primary.data(), .mins_array = secondary.data()};
        for (int b = 0; b < blocks; ++b)
        {
            tensor.packVnniBlock(staged, column, 0, b);
            const auto coded = view.codedBlockMetadata(column / 64, b, column % 64);
            ASSERT_EQ(coded.primary, primary[b]);
            ASSERT_EQ(coded.secondary, secondary[b]);
        }
        for (std::size_t row = 0; row < Rows; ++row)
        {
            float expected = 0;
            for (int partition = 0; partition < partitions; ++partition)
            {
                float partial = 0;
                for (int b = partition * span; b < std::min((partition + 1) * span, blocks); ++b)
                {
                    int low = 0, high = 0;
                    const auto *p = payload.data() + b * payload_bytes;
                    const auto &a = activation[row][b];
                    for (int i = 0; i < 32; ++i)
                    {
                        const int nibble = (p[i % 16] >> (i >= 16 ? 4 : 0)) & 15;
                        int weight = kvalues_iq4nl_i8[nibble];
                        if constexpr (Codebook == 8)
                            weight = nibble + 16 * ((p[16 + i / 4] >> (2 * (i % 4))) & 3) - 32;
                        (i < 16 ? low : high) += weight * a.qs[i];
                    }
                    const float d = nativeQuantisedFP16ScaleToFP32(a.d);
                    const float s = nativeQuantisedPrimaryScale(view.encoding, view.codebook_id, primary[b], secondary[b]);
                    if constexpr (Codebook == 8)
                    {
                        const float second = nativeQuantisedSecondaryScale(view.encoding, view.codebook_id, primary[b], secondary[b]);
                        partial = add(partial, multiply(d, add(multiply(s, float(low)), multiply(second, float(high)))));
                    }
                    else
                    {
                        const float combined = multiply(d, s);
                        partial = expert ? add(partial, multiply(combined, float(low + high)))
                                         : std::fma(combined, float(low + high), partial);
                    }
                }
                expected = partition == 0 ? partial : add(expected, partial);
            }
            ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[row][column]), std::bit_cast<std::uint32_t>(expected))
                << "row=" << row << " column=" << column;
        }
    }
}

/** @brief Run bounded grouped tiles against the staged per-block oracle on both K trees. */
template <std::uint8_t Codebook> void verifyProjectionPolicies()
{
    const auto tensor = source<Codebook>(544);
    CPUNativeQuantisedPackedWeights packed;
    ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), packed));
    for (auto policy : {CPUProjectionNumericalPolicy::BackendNative, CPUProjectionNumericalPolicy::GPUAlignedExpert})
    {
        packed.numerical_policy = policy;
        packed.sealPublication();
        verifyProjection<Codebook, 1>(packed, *tensor);
        verifyProjection<Codebook, 2>(packed, *tensor);
        verifyProjection<Codebook, 3>(packed, *tensor);
        verifyProjection<Codebook, 4>(packed, *tensor);
    }
}
#endif
} // namespace

TEST(CPUCodedFileDensityQ6IQ4, Q6KPreparedBytesEqualFileDensity)
{
    verifyExtent<8>();
}

TEST(CPUCodedFileDensityQ6IQ4, IQ4XSPreparedBytesEqualFileDensityPlusCompensation)
{
    verifyExtent<kIQ4XS>();
}

TEST(CPUCodedFileDensityQ6IQ4, SimdSourceFactorsEqualStagedWordsExhaustively)
{
#if defined(__aarch64__) && defined(__ARM_NEON)
    verifySIMDFactors<8, 4>();
    verifySIMDFactors<kIQ4XS, 4>();
#else
    verifySIMDFactors<8, 8>();
    verifySIMDFactors<kIQ4XS, 8>();
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    verifySIMDFactors<8, 16>();
    verifySIMDFactors<kIQ4XS, 16>();
#endif
#endif
}

#if defined(__aarch64__) && defined(__ARM_NEON)
TEST(CPUCodedFileDensityQ6IQ4, NEONProjectionEqualsPerBlockSourceOracle)
{
    verifyProjectionPolicies<8>();
    verifyProjectionPolicies<kIQ4XS>();
}
#endif
/**
 * @test Coded Q6_K and IQ4_XS units migrate in their coded form in both directions.
 *
 * Both manifest builders admit the coded CPU layout (the coded GPU migration
 * slots now exist), carry the coded CPU encoding unchanged and size the wire
 * from the coded geometry, so the stream never reintroduces per-block planes.
 */
TEST(CPUCodedFileDensityQ6IQ4, CodedSourceScalesMigrateAsCodedTierStreams)
{
    const auto verify = []<std::uint8_t Codebook>(const char *name) {
        SCOPED_TRACE(name);
        auto tensor = source<Codebook>(256);
        CPUNativeQuantisedPackedWeights packed;
        ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), packed));
        ASSERT_TRUE(packed.superblockCoded());
        const auto to_gpu = makeCpuToGpuExpertTierWeightStreamManifest(packed, 1, 0, 0,
            ExpertTierWeightProjection::Down, 1);
        const auto to_cpu = makeGpuToCpuExpertTierWeightStreamManifest(*tensor->vnniFormatInfo(), packed.N, packed.K,
            1, 0, 0, ExpertTierWeightProjection::Down, 1);
        std::string error;
        EXPECT_TRUE(to_gpu.valid(&error)) << error;
        EXPECT_TRUE(to_cpu.valid(&error)) << error;
        EXPECT_EQ(to_gpu.cpu_encoding, packed.encoding);
        EXPECT_EQ(to_cpu.cpu_encoding, packed.encoding);
        EXPECT_EQ(to_gpu.cpu_block_stride, static_cast<std::uint32_t>(packed.interleaved_block_stride));
        EXPECT_EQ(to_cpu.cpu_block_stride, static_cast<std::uint32_t>(packed.interleaved_block_stride));
        EXPECT_EQ(to_gpu.total_stream_bytes, to_cpu.total_stream_bytes);
    };
    verify.template operator()<8>("Q6_K");
    verify.template operator()<kIQ4XS>("IQ4_XS");
}
} // namespace llaminar2::test
