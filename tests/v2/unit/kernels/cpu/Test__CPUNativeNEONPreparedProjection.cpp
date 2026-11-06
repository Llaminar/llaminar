/**
 * @file Test__CPUNativeNEONPreparedProjection.cpp
 * @brief Native prepared projection leaves against independent serial/source-order oracles.
 *
 * Tests execute the actual native interleaved decoders and rounded K traversal,
 * including the shared compact program, not row replay. Four-column ASIMD and
 * explicitly targeted DotProd implementations
 * share payload decode across up to four rows. Logical N tails must neither
 * load nor publish outside the supplied output extent. Full factory/model and
 * worker-scaling certification remains a separate production integration gate.
 */
#include "kernels/cpu/gemm/CPUNativeNEONPreparedProjection.h"
#include "utils/ARMCPUFeatures.h"
#include "utils/CPUExplicitRoundingOracle.h"
#include "utils/NativeVNNIPreparedByteFixture.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <span>
#include <vector>

namespace llaminar2::test
{
namespace
{
using namespace cpu::native_vnni;
namespace ms = cpu::native_vnni::multi_scale;

/** @brief Admit the whole compiler envelope before native numerical work. */
class CPUNativeNEONPreparedProjection : public ::testing::Test
{
protected:
    /** @brief Reject unavailable native instructions rather than skipping/relabeling them. */
    static void SetUpTestSuite()
    {
        cpu::ARMCPUCapabilities::native().require(cpu::ARMCPUFeature::ASIMD);
#if defined(__ARM_FEATURE_DOTPROD)
        cpu::ARMCPUCapabilities::native().require(cpu::ARMCPUFeature::DotProd);
#endif
    }
};

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

/**
 * @brief Independent source-order integer decode for a non-compact resident weight.
 * @param packed Canonical read view.
 * @param column Original logical output column.
 * @param block Logical K block.
 * @param k Element in the 32-value block.
 * @return Exact signed source integer, without scale/minimum application.
 */
int sourceInteger(const CPUNativeVNNIWeightView &packed, int column, int block, int k)
{
    const int local = column % 64;
    const auto *unit = packed.interleavedBase() +
        (std::size_t(column / 64) * packed.blocks_per_row + block) * packed.interleaved_block_stride;
    if (packed.usesExpandedInt8())
    {
        const int raw = unit[(k / 4) * 256 + local * 4 + k % 4];
        return raw < 128 ? raw : raw - 256;
    }
    const int raw = unit[((k % 16) / 4) * 256 + local * 4 + k % 4];
    const int nibble = k < 16 ? raw & 15 : raw >> 4;
    if (packed.usesQ6KNativeDualScale())
    {
        const auto *planes = unit + 1024 + (k / 4) * 64 + (local / 16) * 16;
        std::uint64_t first = 0, second = 0;
        std::memcpy(&first, planes, sizeof(first));
        std::memcpy(&second, planes + 8, sizeof(second));
        const int bit = (local % 16) * 4 + k % 4;
        return nibble + int((first >> bit) & 1) * 16 + int((second >> bit) & 1) * 32 - 32;
    }
    // MXFP4 maps through its own E2M1 table, never IQ4_NL's grid.
    if (packed.codebook_id == kNativeVnniMXFP4Codebook)
    {
        static constexpr int kE2M1Doubled[8] = {0, 1, 2, 3, 4, 6, 8, 12};
        return (nibble & 8) ? -kE2M1Doubled[nibble & 7] : kE2M1Doubled[nibble & 7];
    }
    return packed.codebook_id == 4 ? kvalues_iq4nl_i8[nibble]
        : packed.codebook_id == 0 ? nibble - 8 : nibble;
}

/**
 * @brief Append one independently decoded/memory-rounded source block to the reference.
 * @param packed Canonical source/encoding/arithmetic identity.
 * @param activation Original persistent Q8 block.
 * @param column Logical output column.
 * @param block Logical K block.
 * @param accumulator Reference's preceding FP32 state.
 * @return Exact next state, retaining native FMA versus expert separate rounding.
 */
float appendPreparedReference(const CPUNativeVNNIWeightView &packed, const Q8_1Block &activation,
    int column, int block, float accumulator)
{
    if (packed.usesCompactMultiScale())
        return memoryRoundedAdd(accumulator, ms::scalarReferenceBlock(
            packed, activation, column / 64, block, column % 64));
    int low = 0, high = 0;
    for (int k = 0; k < 32; ++k)
    {
        const int product = sourceInteger(packed, column, block, k) * int(activation.qs[k]);
        if (k < 16) low += product;
        else high += product;
    }
    const float scale = packed.blockScale(column / 64, block, column % 64);
    const float activation_scale = nativeVNNIFP16ScaleToFP32(activation.d);
    if (packed.usesQ6KNativeDualScale())
    {
        const float second = packed.blockMin(column / 64, block, column % 64);
        const float pair = memoryRoundedAdd(memoryRoundedMultiply(scale, float(low)),
            memoryRoundedMultiply(second, float(high)));
        return memoryRoundedAdd(accumulator, memoryRoundedMultiply(activation_scale, pair));
    }
    const float combined = memoryRoundedMultiply(activation_scale, scale);
    const float dot = static_cast<float>(low + high);
    if (packed.numerical_policy == CPUProjectionNumericalPolicy::BackendNative)
    {
        float result = std::fma(combined, dot, accumulator);
        if (packed.is_asymmetric)
            result = std::fma(memoryRoundedMultiply(activation_scale, float(activation.sum_qs)),
                packed.blockMin(column / 64, block, column % 64), result);
        return result;
    }
    float contribution = memoryRoundedMultiply(combined, dot);
    if (packed.is_asymmetric)
    {
        const float correction = memoryRoundedMultiply(memoryRoundedMultiply(activation_scale,
            packed.blockMin(column / 64, block, column % 64)), float(activation.sum_qs));
        contribution = memoryRoundedAdd(contribution, correction);
    }
    return memoryRoundedAdd(accumulator, contribution);
}

/** @brief Compare grouped production bytes with serial M=1 and source-order contributions.
 * @tparam Rows Bounded physical shared-weight row tile.
 * @tparam Ops Explicitly admitted native instruction family.
 * @tparam Tree Contiguous or canonical ordered-expert traversal.
 * @param source Original registered native quantized source identity.
 * @param n Logical output width, including non-vector and non-chunk tails.
 * @param blocks Logical K block count spanning partition boundaries.
 */
template <std::size_t Rows, typename Ops, ms::Reduction Tree>
void verifyPrepared(const NativeVnniFormatInfo &source, int n, int blocks, bool rotated = false)
{
    SCOPED_TRACE(::testing::Message() << "codebook=" << int(source.codebook_id)
        << " rows=" << Rows << " n=" << n << " blocks=" << blocks << " tree=" << int(Tree)
        << " rotated=" << rotated);
    // Rotation publishes the symmetric ExpandedInt8 stream for any original
    // source; the Q8_0 fixture owns that geometry, provenance stays original.
    NativeVNNIPreparedByteFixture fixture(rotated ? native_vnni_formats::Q8_0 : source, n, blocks * 32);
    if (rotated)
        fixture.descriptor.source = {
            .codebook_id = source.codebook_id, .is_superblock = source.is_superblock, .present = true};
    if constexpr (Tree == ms::Reduction::Contiguous)
        fixture.descriptor.numerical_policy = CPUProjectionNumericalPolicy::BackendNative;
    fixture.seedPayloadAndMetadata();
    const auto packed = fixture.view();
    const auto activation = activationRows(blocks, Rows);
    std::array<const Q8_1Block *, Rows> inputs;
    std::array<std::span<const Q8_1Block>, Rows> input_spans;
    std::array<std::span<float>, Rows> output_spans;
    const int stride = n + 3;
    const float dirty = std::bit_cast<float>(0x7fc12345u);
    std::vector<float> grouped(Rows * stride, dirty);
    for (std::size_t row = 0; row < Rows; ++row)
    {
        inputs[row] = activation.data() + row * blocks;
        input_spans[row] = {inputs[row], std::size_t(blocks)};
    }
    const int partitions = MoEProjectionNumericalContract::orderedKPartitionsForWidth(blocks * 32);
    const int span = (blocks + partitions - 1) / partitions;
    for (int chunk = 0; chunk < packed.N_padded / 64; ++chunk)
    {
        const int columns = std::min(64, n - chunk * 64);
        for (std::size_t row = 0; row < Rows; ++row)
            output_spans[row] = {grouped.data() + row * stride + chunk * 64, std::size_t(columns)};
        CPUNativeNEONProjectionTile<Rows>::bind(packed, input_spans, output_spans, chunk)
            .template execute<Ops::instruction>();
    }
    for (std::size_t row = 0; row < Rows; ++row)
    {
        std::vector<float> serial(n, dirty);
        const std::array<std::span<const Q8_1Block>, 1> input{input_spans[row]};
        for (int chunk = 0; chunk < packed.N_padded / 64; ++chunk)
        {
            const int columns = std::min(64, n - chunk * 64);
            const std::array<std::span<float>, 1> output{
                std::span(serial.data() + chunk * 64, std::size_t(columns))};
            CPUNativeNEONProjectionTile<1>::bind(packed, input, output, chunk)
                .template execute<Ops::instruction>();
        }
        ASSERT_EQ(std::memcmp(grouped.data() + row * stride, serial.data(), n * sizeof(float)), 0);
        for (int column = 0; column < n; ++column)
        {
            float expected = 0.0f;
            const int count = Tree == ms::Reduction::OrderedPartitions ? partitions : 1;
            for (int partition = 0; partition < count; ++partition)
            {
                const int first = Tree == ms::Reduction::OrderedPartitions ? partition * span : 0;
                const int last = Tree == ms::Reduction::OrderedPartitions ? std::min(first + span, blocks) : blocks;
                float partial = 0.0f;
                for (int block = first; block < last; ++block)
                    partial = appendPreparedReference(packed, inputs[row][block], column, block, partial);
                expected = partition == 0 ? partial : memoryRoundedAdd(expected, partial);
            }
            ASSERT_EQ(std::bit_cast<std::uint32_t>(grouped[row * stride + column]),
                std::bit_cast<std::uint32_t>(expected)) << "row=" << row << " column=" << column;
        }
        for (int column = n; column < stride; ++column)
            EXPECT_EQ(std::bit_cast<std::uint32_t>(grouped[row * stride + column]), 0x7fc12345u);
    }
}

/** @brief Sweep each compact format across native row, N-tail and K-partition geometry. */
template <typename Ops> void verifyAllCompact()
{
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
    {
        if (preparedEncodingForSource(*source.metadata) != CPUNativeVNNIEncoding::CompactMultiScale)
            continue;
        for (const int n : {1, 3, 4, 5, 63, 64, 65})
            for (const int blocks : {1, 2, 7, 16, 17, 33})
            {
                verifyPrepared<1, Ops, ms::Reduction::Contiguous>(*source.metadata, n, blocks);
                verifyPrepared<2, Ops, ms::Reduction::OrderedPartitions>(*source.metadata, n, blocks);
                verifyPrepared<3, Ops, ms::Reduction::OrderedPartitions>(*source.metadata, n, blocks);
                verifyPrepared<4, Ops, ms::Reduction::OrderedPartitions>(*source.metadata, n, blocks);
            }
    }
}
/** @brief Cover all registered sources and all six physical encodings on both arithmetic policies. */
template <typename Ops> void verifyAllPrepared()
{
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
        for (const int n : {3, 65})
            for (const int blocks : {1, 17})
            {
                verifyPrepared<2, Ops, ms::Reduction::Contiguous>(*source.metadata, n, blocks);
                verifyPrepared<3, Ops, ms::Reduction::Contiguous>(*source.metadata, n, blocks);
                verifyPrepared<4, Ops, ms::Reduction::Contiguous>(*source.metadata, n, blocks);
                verifyPrepared<2, Ops, ms::Reduction::OrderedPartitions>(*source.metadata, n, blocks);
                verifyPrepared<3, Ops, ms::Reduction::OrderedPartitions>(*source.metadata, n, blocks);
                verifyPrepared<4, Ops, ms::Reduction::OrderedPartitions>(*source.metadata, n, blocks);
            }
}
} // namespace

/** @test Every source's ASIMD projection obeys serial bytes and its independent numerical program. */
TEST_F(CPUNativeNEONPreparedProjection, ASIMDEverySourceAndEncodingMatchesSerialAndSourceOrder)
{ verifyAllPrepared<ms::NEONMultiScaleOps<ms::NEONByteDot::ASIMD>>(); }

#if defined(__ARM_FEATURE_DOTPROD)
/** @test Every source's explicitly admitted SDOT projection retains the same complete program. */
TEST_F(CPUNativeNEONPreparedProjection, DotProdEverySourceAndEncodingMatchesSerialAndSourceOrder)
{ verifyAllPrepared<ms::NEONMultiScaleOps<ms::NEONByteDot::DotProd>>(); }
#endif

/** @test The baseline widening-dot leaf is exact on every compact source and physical tile. */
TEST_F(CPUNativeNEONPreparedProjection, ASIMDCompactMatchesSerialAndSourceOrder)
{ verifyAllCompact<ms::NEONMultiScaleOps<ms::NEONByteDot::ASIMD>>(); }

#if defined(__ARM_FEATURE_DOTPROD)
/** @test Explicitly admitted SDOT uses the same source factors and serial bytes. */
TEST_F(CPUNativeNEONPreparedProjection, DotProdCompactMatchesSerialAndSourceOrder)
{ verifyAllCompact<ms::NEONMultiScaleOps<ms::NEONByteDot::DotProd>>(); }
#endif

/** @test Invalid binding geometry/aliases fail before any projection writes. */
TEST_F(CPUNativeNEONPreparedProjection, RejectsIncompleteAliasedAndUnalignedExpertBindings)
{
    NativeVNNIPreparedByteFixture fixture(native_vnni_formats::Q4_0, 3, 32);
    const auto weights = fixture.view();
    const auto activation = activationRows(1, 2);
    std::array<std::span<const Q8_1Block>, 2> input{
        std::span(activation.data(), 1), std::span(activation.data() + 1, 1)};
    std::array<float, 6> storage;
    storage.fill(123.0f);
    std::array<std::span<float>, 2> output{
        std::span(storage.data(), 3), std::span(storage.data() + 3, 3)};
    EXPECT_THROW((void)CPUNativeNEONProjectionTile<2>::bind(weights, input, output, -1), std::invalid_argument);
    EXPECT_THROW((void)CPUNativeNEONProjectionTile<2>::bind(weights, input, output, 1), std::invalid_argument);
    auto missing = input;
    missing[1] = {};
    EXPECT_THROW((void)CPUNativeNEONProjectionTile<2>::bind(weights, missing, output, 0), std::invalid_argument);
    auto short_output = output;
    short_output[0] = short_output[0].first(2);
    EXPECT_THROW((void)CPUNativeNEONProjectionTile<2>::bind(weights, input, short_output, 0), std::invalid_argument);
    auto aliased = output;
    aliased[1] = aliased[0];
    EXPECT_THROW((void)CPUNativeNEONProjectionTile<2>::bind(weights, input, aliased, 0), std::invalid_argument);
    aliased[0] = {reinterpret_cast<float *>(fixture.bytes.data()), 3};
    EXPECT_THROW((void)CPUNativeNEONProjectionTile<2>::bind(weights, input, aliased, 0), std::invalid_argument);
    for (const auto value : storage) EXPECT_EQ(value, 123.0f);

    NativeVNNIPreparedByteFixture tail(native_vnni_formats::Q4_0, 3, 33);
    const auto tail_activation = activationRows(2, 1);
    const std::array<std::span<const Q8_1Block>, 1> tail_input{std::span(tail_activation)};
    const std::array<std::span<float>, 1> tail_output{std::span(storage.data(), 3)};
    EXPECT_THROW((void)CPUNativeNEONProjectionTile<1>::bind(tail.view(), tail_input, tail_output, 0), std::invalid_argument);
}

/** @test Opaque source words cannot hide reserved bits by masking them into a usable coefficient. */
TEST_F(CPUNativeNEONPreparedProjection, RejectsReservedSourceCoefficientBits)
{
    for (const auto *source : {&native_vnni_formats::IQ2_S, &native_vnni_formats::IQ3_S, &native_vnni_formats::IQ4_XS})
    {
        SCOPED_TRACE(int(source->codebook_id));
        NativeVNNIPreparedByteFixture fixture(*source, 8, 32);
        fixture.seedPayloadAndMetadata();
        const auto weights = fixture.view();
        const auto secondary = weights.data_stride + (weights.usesInlineCompensation() ? 256 : 128);
        const auto activation = activationRows(1, 1);
        const std::array<std::span<const Q8_1Block>, 1> input{std::span(activation)};
        std::array<float, 8> result;
        const std::array<std::span<float>, 1> output{std::span(result)};
        const auto tile = CPUNativeNEONProjectionTile<1>::bind(weights, input, output, 0);
        tile.execute();
        const auto reference = result;
        for (int bad_column = 0; bad_column < 8; ++bad_column)
        {
            SCOPED_TRACE(bad_column);
            std::uint16_t original = 0;
            auto *word = fixture.bytes.data() + secondary + bad_column * 2;
            std::memcpy(&original, word, sizeof(original));
            const std::uint16_t invalid = 0xffff;
            std::memcpy(word, &invalid, sizeof(invalid));
            result.fill(123.0f);
            EXPECT_THROW(tile.execute(), std::invalid_argument);
            // Rejection is fatal, not transactional rollback. Earlier vectors
            // can be authored; the outer stage must never publish failed work.
            const int failed_vector = bad_column / 4 * 4;
            for (int column = 0; column < 8; ++column)
                if (column < failed_vector)
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(result[column]), std::bit_cast<std::uint32_t>(reference[column]));
                else
                    EXPECT_EQ(result[column], 123.0f);
            std::memcpy(word, &original, sizeof(original));
        }
    }
}

/** @test Native metadata widening and half conversion cover every possible persisted word. */
TEST_F(CPUNativeNEONPreparedProjection, EveryHalfAndOpaqueWordHasExactBits)
{
    using Ops = ms::NEONMultiScaleOps<ms::NEONByteDot::ASIMD>;
    for (int first = 0; first < 65536; first += 4)
    {
        const std::array<std::uint16_t, 4> words{
            static_cast<std::uint16_t>(first), static_cast<std::uint16_t>(first + 1),
            static_cast<std::uint16_t>(first + 2), static_cast<std::uint16_t>(first + 3)};
        const auto half = std::bit_cast<std::array<float, 4>>(Ops::half(words.data()));
        const auto widened = Ops::words(words.data());
        const auto raw = std::bit_cast<std::array<int, 4>>(widened);
        std::array<std::uint32_t, 4> pairs;
        for (int lane = 0; lane < 4; ++lane)
            pairs[lane] = (std::uint32_t(words[lane] ^ 0xa5b6u) << 16) | words[lane];
        const auto packed = std::bit_cast<Ops::I>(pairs);
        const auto low = std::bit_cast<std::array<float, 4>>(Ops::halfWords(packed));
        const auto high = std::bit_cast<std::array<float, 4>>(Ops::halfWords(packed >> 16));
        for (const std::uint32_t mask : {15u, 63u, 255u, 65535u})
            EXPECT_EQ(Ops::metadataFits(widened, mask), std::uint32_t(first + 3) <= mask);
        for (int lane = 0; lane < 4; ++lane)
        {
            EXPECT_EQ(std::bit_cast<std::uint32_t>(half[lane]),
                std::bit_cast<std::uint32_t>(fp16_to_fp32(words[lane])));
            EXPECT_EQ(raw[lane], words[lane]);
            EXPECT_EQ(std::bit_cast<std::uint32_t>(low[lane]), std::bit_cast<std::uint32_t>(fp16_to_fp32(words[lane])));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(high[lane]),
                std::bit_cast<std::uint32_t>(fp16_to_fp32(static_cast<std::uint16_t>(words[lane] ^ 0xa5b6u))));
        }
    }
}

/** @test Native MXFP4 BF16 power-of-two words widen exactly; FP16 interpretation is never used. */
TEST_F(CPUNativeNEONPreparedProjection, EveryAdmittedPowerOfTwoBF16WordWidensExactly)
{
    using Ops = ms::NEONMultiScaleOps<ms::NEONByteDot::ASIMD>;
    std::vector<std::uint16_t> words{0};
    for (int e = 2; e <= 252; ++e)
        words.push_back(mxfp4::powerOfTwoBF16Word(static_cast<std::uint8_t>(e)));
    while (words.size() % 4 != 0) words.push_back(0);
    for (std::size_t first = 0; first < words.size(); first += 4)
    {
        const auto widened = std::bit_cast<std::array<std::uint32_t, 4>>(Ops::bf16(words.data() + first));
        for (int lane = 0; lane < 4; ++lane)
        {
            const auto word = words[first + lane];
            std::uint8_t e = 0;
            ASSERT_TRUE(mxfp4::e8m0FromPowerOfTwoBF16Word(word, e));
            EXPECT_EQ(widened[lane], e == 0 ? 0u : mxfp4::normalScaleFP32Bits(e)) << "word=" << word;
        }
    }
}

/** @test Rotated MXFP4 (source 25, ExpandedInt8) dispatches the expanded FP16 program, not the BF16 nibble one. */
TEST_F(CPUNativeNEONPreparedProjection, RotatedMXFP4ExecutesExpandedInt8WithFP16Scales)
{
    using Ops = ms::NEONMultiScaleOps<ms::NEONByteDot::ASIMD>;
    for (const int n : {3, 65})
        for (const int blocks : {1, 17})
        {
            verifyPrepared<1, Ops, ms::Reduction::Contiguous>(native_vnni_formats::MXFP4, n, blocks, true);
            verifyPrepared<4, Ops, ms::Reduction::Contiguous>(native_vnni_formats::MXFP4, n, blocks, true);
            verifyPrepared<2, Ops, ms::Reduction::OrderedPartitions>(native_vnni_formats::MXFP4, n, blocks, true);
            verifyPrepared<3, Ops, ms::Reduction::OrderedPartitions>(native_vnni_formats::MXFP4, n, blocks, true);
        }
}

/** @test Native MXFP4 and its rotated form keep distinct scale-word interpretations. */
TEST_F(CPUNativeNEONPreparedProjection, MXFP4ScaleWordsFollowThePreparedIdentity)
{
    NativeVNNIPreparedByteFixture native(native_vnni_formats::MXFP4, 65, 64);
    ASSERT_EQ(native.descriptor.encoding, CPUNativeVNNIEncoding::NibbleLUT);
    EXPECT_EQ(native.view().scaleEncoding(), CPUNativeVNNIScaleEncoding::PowerOfTwoBF16);
    native.seedPayloadAndMetadata();
    const auto view = native.view();
    for (int column = 0; column < 65; ++column)
        for (int block = 0; block < 2; ++block)
        {
            const auto word = view.chunkScales(column / 64, block)[column % 64];
            std::uint8_t e = 0;
            ASSERT_TRUE(mxfp4::e8m0FromPowerOfTwoBF16Word(word, e)) << "seeded an inadmissible word";
            EXPECT_EQ(std::bit_cast<std::uint32_t>(view.blockScale(column / 64, block, column % 64)),
                e == 0 ? 0u : mxfp4::normalScaleFP32Bits(e));
        }
    EXPECT_EQ(preparedScaleEncoding(kNativeVnniMXFP4Codebook, CPUNativeVNNIEncoding::ExpandedInt8),
        CPUNativeVNNIScaleEncoding::FP16);
    EXPECT_EQ(preparedScaleEncoding(4, CPUNativeVNNIEncoding::NibbleLUT), CPUNativeVNNIScaleEncoding::FP16);
}

/** @test Widening reductions retain every signed-byte extremum and exact quarter order. */
TEST_F(CPUNativeNEONPreparedProjection, ActivationSumsIncludeSignedMinimum)
{
    for (int seed = 0; seed < 256; ++seed)
    {
        std::array<std::int8_t, 32> values;
        std::array<int, 4> expected{};
        for (int k = 0; k < 32; ++k)
        {
            values[k] = static_cast<std::int8_t>((seed + k * 29) % 256 - 128);
            expected[k / 8] += values[k];
        }
        EXPECT_EQ(activationQuarterSums(values.data()), expected);
        EXPECT_EQ(activationHalfSum(values.data()), expected[0] + expected[1]);
    }
}
} // namespace llaminar2::test
