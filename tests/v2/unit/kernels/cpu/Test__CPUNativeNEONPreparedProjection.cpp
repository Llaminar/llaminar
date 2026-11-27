/**
 * @file Test__CPUNativeNEONPreparedProjection.cpp
 * @brief Native prepared projection leaves against independent serial/source-order oracles.
 *
 * Tests execute the actual native interleaved decoders and rounded K traversal,
 * including the shared compact program, not row replay. Four-column ASIMD and
 * explicitly targeted DotProd implementations
 * share payload decode across up to four rows; the I8MM family consumes those
 * rows in SMMLA pairs and must reproduce the SDOT bytes and operand grouping. Every entry
 * admits the calling thread's FPCR profile and fails fatally under any other one. Logical N tails must neither
 * load nor publish outside the supplied output extent. Full factory/model and
 * worker-scaling certification remains a separate production integration gate.
 */
#include "kernels/cpu/gemm/CPUNativeNEONPreparedProjection.h"
#include "kernels/cpu/gemm/CPUNativeNEONProjectionBatch.h"
#include "tensors/NativeQuantisedSuperblockMetadata.h"
#include "utils/ARMCPUFeatures.h"
#include "utils/CPUExplicitRoundingOracle.h"
#include "utils/NativeQuantisedPreparedByteFixture.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cmath>
#include <initializer_list>
#include <cstring>
#include <cstdio>
#include <span>
#include <vector>

namespace llaminar2::cpu::native_quantised
{
/** @brief Test-only access to retired wide programs; never linked into the production probe. */
struct CPUNativeNEONProjectionWideOracle
{
    /** @brief Execute the original physical tile on the same admitted row spans and K tree. */
    template <typename Ops, CPUProjectionNumericalPolicy Policy, std::size_t Rows>
    static void execute(const CPUNativeNEONProjectionTile<Rows> &tile)
    {
        using Tile = CPUNativeNEONProjectionTile<Rows>;
        using E = CPUNativeQuantisedEncoding;
        using M = typename Tile::NibbleMap;
        switch (tile.weights_.encoding)
        {
        case E::CompactMultiScale:
            if (tile.weights_.codebook_id == 7)
                return tile.template runCoded<Ops, Policy, 7, 0>();
            return tile.template runCoded<Ops, Policy, 11, 0>();
        case E::NibbleLUTSourceScale:
            return tile.template run<Ops, Policy, E::NibbleLUTSourceScale, M::NonLinear, false, 0>();
        case E::Q4KSuperblockCoded:
            return tile.template run<Ops, Policy, E::Q4KSuperblockCoded, M::Unsigned, true, 0>();
        case E::Q5NativeSingleScale:
            if (tile.weights_.is_asymmetric)
                return tile.template run<Ops, Policy, E::Q5NativeSingleScale, M::Unsigned, true, 0>();
            return tile.template run<Ops, Policy, E::Q5NativeSingleScale, M::Centered, false, 0>();
        default: throw std::invalid_argument("wide oracle requires a row-decomposition source");
        }
    }
};
}

namespace llaminar2::test
{
namespace
{
using namespace cpu::native_quantised;
namespace ms = cpu::native_quantised::multi_scale;

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
#if defined(__ARM_FEATURE_MATMUL_INT8)
        cpu::ARMCPUCapabilities::native().require(cpu::ARMCPUFeature::IntegerMatMul);
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
int sourceInteger(const CPUNativeQuantisedWeightView &packed, int column, int block, int k)
{
    const int local = column % 64;
    const auto *unit = packed.blockUnit(column / 64, block);
    if (packed.usesSignedInt8SingleScale())
    {
        const int raw = unit[(k / 4) * 256 + local * 4 + k % 4];
        return raw < 128 ? raw : raw - 256;
    }
    if (packed.usesCompactMultiScale() && packed.codebook_id == 11)
    {
        const int group = k / 4;
        const auto qh = unit[512 + local * 4];
        const auto qs = unit[(group / 4) * 256 + local * 4 + group % 4];
        const auto signs = group < 6 ? unit[512 + local * 4 + 1 + group / 2] : unit[768 + local];
        const auto grid = iq3s_grid[qs | (((qh >> group) & 1) << 8)];
        const int value = (grid >> (8 * (k % 4))) & 255;
        return signs & (1 << (k % 8)) ? -value : value;
    }
    const int raw = unit[((k % 16) / 4) * 256 + local * 4 + k % 4];
    const int nibble = k < 16 ? raw & 15 : raw >> 4;
    if (packed.usesCompactMultiScale() && packed.codebook_id == 7)
    {
        const auto qh = unit[1024 + 32 * (k / 4) + local / 2];
        return nibble + 16 * ((qh >> (4 * (local % 2) + k % 4)) & 1);
    }
    if (packed.encoding == CPUNativeQuantisedEncoding::Q5NativeSingleScale)
    {
        const auto fifth = unit[1024 + local * 4 + (k % 16) / 4];
        const int bit = k % 4 + (k < 16 ? 0 : 4);
        return nibble + 16 * ((fifth >> bit) & 1) - (packed.is_asymmetric ? 0 : 16);
    }
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
    if (packed.codebook_id == kNativeQuantisedMXFP4Codebook)
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
float appendPreparedReference(const CPUNativeQuantisedWeightView &packed, const Q8_1Block &activation,
    int column, int block, float accumulator)
{
    if (packed.usesCompactMultiScale() && packed.codebook_id != 7 && packed.codebook_id != 11)
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
    const float activation_scale = nativeQuantisedFP16ScaleToFP32(activation.d);
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
void verifyPrepared(const NativeQuantisedFormatInfo &source, int n, int blocks)
{
    SCOPED_TRACE(::testing::Message() << "codebook=" << int(source.codebook_id)
        << " rows=" << Rows << " n=" << n << " blocks=" << blocks << " tree=" << int(Tree));
    NativeQuantisedPreparedByteFixture fixture(source, n, blocks * 32);
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
    for (const auto &source : native_quantised_formats::kAllSourceFormats)
    {
        if (preparedEncodingForSource(*source.metadata) != CPUNativeQuantisedEncoding::CompactMultiScale)
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
/**
 * @brief Exercise compact ordered trees independently of the numerical policy.
 * @tparam Codebook Exact compact source identity.
 * @tparam Rows Physical rows sharing the K walk.
 * @tparam Ops Explicitly admitted integer instruction family.
 * @tparam Tree Contiguous or ordered-partition compact K walk.
 * @param policy Select ordinary additions or the GPU-aligned ordered-NaN program.
 * @details BackendNative plus OrderedPartitions is intentionally called directly:
 * the ordinary projection tile selects a contiguous tree for BackendNative and
 * therefore does not certify this independently supported compact entry.
 */
template <int Codebook, std::size_t Rows, typename Ops, ms::Reduction Tree = ms::Reduction::OrderedPartitions>
void verifyCompactOrderedPolicy(CPUProjectionNumericalPolicy policy)
{
    for (const auto &source : native_quantised_formats::kAllSourceFormats)
    {
        if (source.metadata->codebook_id != Codebook)
            continue;
        for (const int n : {1, 3, 4, 5, 63, 64, 65})
            for (const int blocks : {1, 2, 7, 16, 17, 33})
            {
                SCOPED_TRACE(::testing::Message() << "codebook=" << Codebook << " rows=" << Rows
                    << " n=" << n << " blocks=" << blocks << " policy=" << int(policy));
                NativeQuantisedPreparedByteFixture fixture(*source.metadata, n, blocks * 32);
                fixture.descriptor.numerical_policy = policy;
                fixture.seedPayloadAndMetadata();
                const auto packed = fixture.view();
                const auto activation = activationRows(blocks, Rows);
                const int stride = n + 3;
                const float guard = -123.0f;
                std::vector<float> grouped(Rows * stride, guard), serial(Rows * stride, guard);
                std::array<const Q8_1Block *, Rows> inputs{};
                for (std::size_t row = 0; row < Rows; ++row)
                    inputs[row] = activation.data() + row * blocks;
                const int partitions = Tree == ms::Reduction::OrderedPartitions
                    ? MoEProjectionNumericalContract::orderedKPartitionsForWidth(blocks * 32) : 1;
                const int span = (blocks + partitions - 1) / partitions;
                for (int chunk = 0; chunk < packed.N_padded / 64; ++chunk)
                {
                    const int columns = std::min(64, n - chunk * 64);
                    std::array<float *, Rows> outputs{};
                    for (std::size_t row = 0; row < Rows; ++row)
                        outputs[row] = grouped.data() + row * stride + chunk * 64;
                    if constexpr (Tree == ms::Reduction::OrderedPartitions)
                        ms::dispatchOrdered<4, Rows, Ops>(packed, inputs, outputs, chunk, blocks, partitions, span, columns);
                    else
                        ms::dispatch<4, Rows, Ops>(packed, inputs, outputs, chunk, 0, blocks, false, columns);
                    for (std::size_t row = 0; row < Rows; ++row)
                    {
                        const std::array<const Q8_1Block *, 1> input{inputs[row]};
                        const std::array<float *, 1> output{serial.data() + row * stride + chunk * 64};
                        if constexpr (Tree == ms::Reduction::OrderedPartitions)
                            ms::dispatchOrdered<4, 1, Ops>(packed, input, output, chunk, blocks, partitions, span, columns);
                        else
                            ms::dispatch<4, 1, Ops>(packed, input, output, chunk, 0, blocks, false, columns);
                    }
                }
                for (std::size_t row = 0; row < Rows; ++row)
                {
                    for (int column = 0; column < n; ++column)
                    {
                        float expected = 0.0f;
                        for (int partition = 0; partition < partitions; ++partition)
                        {
                            float partial = 0.0f;
                            for (int block = partition * span; block < std::min((partition + 1) * span, blocks); ++block)
                                partial = appendPreparedReference(packed, inputs[row][block], column, block, partial);
                            expected = partition == 0 ? partial : memoryRoundedAdd(expected, partial);
                        }
                        ASSERT_EQ(std::bit_cast<std::uint32_t>(grouped[row * stride + column]),
                            std::bit_cast<std::uint32_t>(serial[row * stride + column]));
                        ASSERT_EQ(std::bit_cast<std::uint32_t>(grouped[row * stride + column]),
                            std::bit_cast<std::uint32_t>(expected));
                    }
                    for (int column = n; column < stride; ++column)
                    {
                        ASSERT_EQ(grouped[row * stride + column], guard);
                        ASSERT_EQ(serial[row * stride + column], guard);
                    }
                }
            }
    }
}

/** @brief Cover the independently selected compact K tree on both numerical policies. */
template <typename Ops> void verifyCompactOrderedPolicies()
{
    verifyCompactOrderedPolicy<12, 4, Ops>(CPUProjectionNumericalPolicy::BackendNative);
    verifyCompactOrderedPolicy<17, 2, Ops>(CPUProjectionNumericalPolicy::BackendNative);
    verifyCompactOrderedPolicy<17, 2, Ops>(CPUProjectionNumericalPolicy::GPUAlignedExpert);
    verifyCompactOrderedPolicy<17, 2, Ops, ms::Reduction::Contiguous>(CPUProjectionNumericalPolicy::BackendNative);
    verifyCompactOrderedPolicy<17, 2, Ops, ms::Reduction::Contiguous>(CPUProjectionNumericalPolicy::GPUAlignedExpert);
}

/** @brief Cover all registered sources and physical encodings on both arithmetic policies. */
template <typename Ops> void verifyAllPrepared()
{
    for (const auto &source : native_quantised_formats::kAllSourceFormats)
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
/**
 * @brief Prove the padding contract at a partial final chunk for one source/geometry/family.
 * @tparam Ops Admitted family (ASIMD, DotProd, or I8MM row pairs).
 * @details Two complementary proofs on separate fixtures. Poison rejection: for
 * coefficient-ruled encodings every padding word a vector of the last strip or
 * its neighbour loads is poisoned in turn, and admission must reject it before
 * any tile exists. Positive execution: with the producer's benign padding,
 * every row tile 1..4 on both policies must equal serial bytes, the
 * independent numerical program and the output guards (verifyPrepared).
 */
template <typename Ops> void verifyTailPadding(const NativeQuantisedFormatInfo &source, int n, int blocks)
{
    SCOPED_TRACE(::testing::Message() << "tail codebook=" << int(source.codebook_id) << " n=" << n << " blocks=" << blocks);
    {
        NativeQuantisedPreparedByteFixture fixture(source, n, blocks * 32);
        fixture.seedPayloadAndMetadata();
        const auto view = fixture.view();
        if (view.superblockCoded())
        {
            // Every bit pattern fits the coded planes, but no header bytes may be missing.
            EXPECT_THROW((void)CPUNativeQuantisedMetadataSeal::validate(fixture.descriptor,
                fixture.bytes.first(fixture.bytes.size() - 1)), std::invalid_argument);
        }
    }
    // Row tiles 1..4 under both policies (contiguous backend-native and
    // ordered expert partitions), on a fixture separate from the poison checks.
    verifyPrepared<1, Ops, ms::Reduction::Contiguous>(source, n, blocks);
    verifyPrepared<2, Ops, ms::Reduction::Contiguous>(source, n, blocks);
    verifyPrepared<3, Ops, ms::Reduction::Contiguous>(source, n, blocks);
    verifyPrepared<4, Ops, ms::Reduction::Contiguous>(source, n, blocks);
    verifyPrepared<1, Ops, ms::Reduction::OrderedPartitions>(source, n, blocks);
    verifyPrepared<2, Ops, ms::Reduction::OrderedPartitions>(source, n, blocks);
    verifyPrepared<3, Ops, ms::Reduction::OrderedPartitions>(source, n, blocks);
    verifyPrepared<4, Ops, ms::Reduction::OrderedPartitions>(source, n, blocks);
}

/** @brief Every source family at N mod 4 = 1/2/3, inside and across chunk tails. */
template <typename Ops> void verifyAllTailPadding()
{
    for (const auto &source : native_quantised_formats::kAllSourceFormats)
        for (const int n : {1, 2, 3, 5, 6, 7, 61, 62, 63, 65, 66, 67, 127})
            verifyTailPadding<Ops>(*source.metadata, n, 3);
}
} // namespace

/** @test The padding contract holds at every N%4 tail for ASIMD. */
TEST_F(CPUNativeNEONPreparedProjection, ASIMDTailPaddingContract)
{ verifyAllTailPadding<ms::NEONMultiScaleOps<ms::NEONByteDot::ASIMD>>(); }

#if defined(__ARM_FEATURE_DOTPROD)
/** @test The padding contract holds at every N%4 tail for SDOT. */
TEST_F(CPUNativeNEONPreparedProjection, DotProdTailPaddingContract)
{ verifyAllTailPadding<ms::NEONMultiScaleOps<ms::NEONByteDot::DotProd>>(); }
#endif

#if defined(__ARM_FEATURE_MATMUL_INT8)
/** @test The padding contract holds at every N%4 tail for SMMLA row pairs. */
TEST_F(CPUNativeNEONPreparedProjection, I8MMTailPaddingContract)
{ verifyAllTailPadding<ms::NEONMultiScaleOps<ms::NEONByteDot::I8MM>>(); }
#endif

namespace
{
/**
 * @brief Prove that every panel width writes the strip program's bytes, and nothing past N.
 * @tparam Rows Physical row tile. @tparam Ops Admitted family.
 * @details Strips=1 is the per-strip program; 2 and 4 interleave strips and
 * cover the whole 64-column chunk, so N tails exercise partial strips and
 * strips wholly in the sealed padding columns, which must never be stored.
 */
template <std::size_t Rows, typename Ops> void verifyPanelWidths(const NativeQuantisedFormatInfo &source, int n, int blocks,
    CPUProjectionNumericalPolicy policy)
{
    SCOPED_TRACE(::testing::Message() << "codebook=" << int(source.codebook_id) << " rows=" << Rows << " n=" << n
        << " blocks=" << blocks << " policy=" << int(policy));
    NativeQuantisedPreparedByteFixture fixture(source, n, blocks * 32);
    fixture.descriptor.numerical_policy = policy;
    fixture.seedPayloadAndMetadata();
    const auto packed = fixture.view();
    const auto activation = activationRows(blocks, Rows);
    std::array<std::span<const Q8_1Block>, Rows> input;
    for (std::size_t row = 0; row < Rows; ++row) input[row] = {activation.data() + row * blocks, std::size_t(blocks)};
    const float dirty = std::bit_cast<float>(0x7fc12345u);
    const int stride = packed.N_padded + 4;
    const auto run = [&]<int Strips>() {
        std::vector<float> out(Rows * stride, dirty);
        for (int chunk = 0; chunk < packed.N_padded / 64; ++chunk)
        {
            const int columns = std::min(64, n - chunk * 64);
            std::array<std::span<float>, Rows> output;
            for (std::size_t row = 0; row < Rows; ++row)
                output[row] = {out.data() + row * stride + chunk * 64, std::size_t(columns)};
            CPUNativeNEONProjectionTile<Rows>::bind(packed, input, output, chunk).template executeWith<Ops, Strips>();
        }
        return out;
    };
    const auto reference = run.template operator()<1>();
    for (std::size_t row = 0; row < Rows; ++row)
        for (int column = n; column < stride; ++column)
            ASSERT_EQ(std::bit_cast<std::uint32_t>(reference[row * stride + column]), 0x7fc12345u);
    const auto two = run.template operator()<2>();
    const auto four = run.template operator()<4>();
    EXPECT_EQ(std::memcmp(two.data(), reference.data(), reference.size() * sizeof(float)), 0) << "strips=2";
    EXPECT_EQ(std::memcmp(four.data(), reference.data(), reference.size() * sizeof(float)), 0) << "strips=4";
}

/** @brief Sweep every non-compact source over panel widths, N tails, rows 1..4 and both policies. */
template <typename Ops> void verifyAllPanelWidths()
{
    for (const auto &source : native_quantised_formats::kAllSourceFormats)
    {
        // Compact multi-scale tiles keep their own column walk (no panel width).
        if (preparedEncodingForSource(*source.metadata) == CPUNativeQuantisedEncoding::CompactMultiScale)
            continue;
        for (const int n : {1, 5, 8, 12, 15, 16, 17, 20, 31, 33, 47, 63, 64, 65, 100})
            for (const int blocks : {1, 3, 17})
                for (const auto policy : {CPUProjectionNumericalPolicy::BackendNative,
                                          CPUProjectionNumericalPolicy::GPUAlignedExpert})
                {
                    verifyPanelWidths<1, Ops>(*source.metadata, n, blocks, policy);
                    verifyPanelWidths<2, Ops>(*source.metadata, n, blocks, policy);
                    verifyPanelWidths<3, Ops>(*source.metadata, n, blocks, policy);
                    verifyPanelWidths<4, Ops>(*source.metadata, n, blocks, policy);
                }
    }
}
} // namespace

/** @test Every decode panel width reproduces the per-strip program's bytes (ASIMD). */
TEST_F(CPUNativeNEONPreparedProjection, ASIMDPanelWidthsMatchTheStripProgram)
{ verifyAllPanelWidths<ms::NEONMultiScaleOps<ms::NEONByteDot::ASIMD>>(); }

#if defined(__ARM_FEATURE_DOTPROD)
/** @test Every decode panel width reproduces the per-strip program's bytes (DotProd). */
TEST_F(CPUNativeNEONPreparedProjection, DotProdPanelWidthsMatchTheStripProgram)
{ verifyAllPanelWidths<ms::NEONMultiScaleOps<ms::NEONByteDot::DotProd>>(); }
#endif

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

#if defined(__ARM_FEATURE_MATMUL_INT8)
namespace
{
using I8MMOps = ms::NEONMultiScaleOps<ms::NEONByteDot::I8MM>;
using SDOTOps = ms::NEONMultiScaleOps<ms::NEONByteDot::DotProd>;

// A lone I8MM row compiles the very SDOT instantiation, so M=1 decode and
// odd-row tiles cannot emit SMMLA or change their code.
static_assert(CPUNativeNEONProjectionTile<1>::leafInstruction<ms::NEONByteDot::I8MM>() == ms::NEONByteDot::DotProd);
static_assert(CPUNativeNEONProjectionTile<2>::leafInstruction<ms::NEONByteDot::I8MM>() == ms::NEONByteDot::I8MM);
static_assert(CPUNativeNEONProjectionTile<3>::leafInstruction<ms::NEONByteDot::I8MM>() == ms::NEONByteDot::I8MM);
static_assert(CPUNativeNEONProjectionTile<4>::leafInstruction<ms::NEONByteDot::I8MM>() == ms::NEONByteDot::I8MM);
static_assert(CPUNativeNEONProjectionTile<1>::leafInstruction<ms::NEONByteDot::DotProd>() == ms::NEONByteDot::DotProd);
static_assert(ms::usesRowPairs<I8MMOps> && !ms::usesRowPairs<SDOTOps> &&
              !ms::usesRowPairs<ms::NEONMultiScaleOps<ms::NEONByteDot::ASIMD>>);

/** @brief Run one complete tile family over every chunk into a canary-padded bank.
 * @tparam Rows Physical shared rows.
 * @tparam Instruction Native instruction family under test.
 * @param packed Prepared weight view.
 * @param activation Rows*blocks persistent Q8 blocks.
 * @param n Logical output width.
 * @param stride Output row stride, larger than n so canaries follow every row.
 * @return The complete bank, including untouched canary words.
 */
template <std::size_t Rows, ms::NEONByteDot Instruction>
std::vector<float> runTile(const CPUNativeQuantisedWeightView &packed, const std::vector<Q8_1Block> &activation,
    int n, int stride)
{
    const int blocks = packed.blocks_per_row;
    std::vector<float> bank(Rows * std::size_t(stride), std::bit_cast<float>(0x7fc12345u));
    std::array<std::span<const Q8_1Block>, Rows> input;
    std::array<std::span<float>, Rows> output;
    for (std::size_t row = 0; row < Rows; ++row)
        input[row] = {activation.data() + row * blocks, std::size_t(blocks)};
    for (int chunk = 0; chunk < packed.N_padded / 64; ++chunk)
    {
        const int columns = std::min(64, n - chunk * 64);
        for (std::size_t row = 0; row < Rows; ++row)
            output[row] = {bank.data() + row * stride + chunk * 64, std::size_t(columns)};
        CPUNativeNEONProjectionTile<Rows>::bind(packed, input, output, chunk).template execute<Instruction>();
    }
    return bank;
}

/** @brief Require SMMLA row-pair bytes, including canaries past N, to equal SDOT's.
 * @tparam Rows Physical shared rows; odd counts keep one SDOT row.
 * @param source Registered source format.
 * @param n Logical output width.
 * @param blocks Logical K blocks.
 * @param expert Whether the ordered expert tree, rather than one K chain, is used.
 */
template <std::size_t Rows>
void verifyI8MMAgainstSDOT(const NativeQuantisedFormatInfo &source, int n, int blocks, bool expert)
{
    SCOPED_TRACE(::testing::Message() << "codebook=" << int(source.codebook_id) << " rows=" << Rows
        << " n=" << n << " blocks=" << blocks << " expert=" << expert);
    NativeQuantisedPreparedByteFixture fixture(source, n, blocks * 32);
    if (!expert)
        fixture.descriptor.numerical_policy = CPUProjectionNumericalPolicy::BackendNative;
    fixture.seedPayloadAndMetadata();
    const auto packed = fixture.view();
    const auto activation = activationRows(blocks, Rows);
    const int stride = n + 5;
    const auto pairs = runTile<Rows, ms::NEONByteDot::I8MM>(packed, activation, n, stride);
    const auto sdot = runTile<Rows, ms::NEONByteDot::DotProd>(packed, activation, n, stride);
    ASSERT_EQ(std::memcmp(pairs.data(), sdot.data(), pairs.size() * sizeof(float)), 0);
    for (std::size_t row = 0; row < Rows; ++row)
        for (int column = n; column < stride; ++column)
            ASSERT_EQ(std::bit_cast<std::uint32_t>(pairs[row * stride + column]), 0x7fc12345u);
}

/**
 * @brief Integer-leaf observer for the operand-grouping oracle.
 * @tparam Base Production operation set whose arithmetic still executes.
 * @details Activation words encode (row, block, group) identities instead of
 * numerical values. Each SDOT call and each SMMLA row-pair call is expanded
 * into its (row, block, group, weight vector) product terms, so the two
 * walks can be compared term for term without trusting the kernel output.
 */
/// One (row, block, group) activation identity consuming one decoded weight vector.
struct GroupingTerm
{
    int row, block, group;
    std::array<std::uint8_t, 16> weights;
    auto operator<=>(const GroupingTerm &) const = default;
};

template <typename Base> struct GroupingObserver : Base
{
    using I = typename Base::I;
    using Term = GroupingTerm;
    static inline std::vector<Term> terms; ///< Single-threaded tile execution log.

    /** @brief Decode the identity word written by identityRows(). */
    static Term term(std::uint32_t word, I weights)
    {
        return {int(word & 0xff), int((word >> 8) & 0xff), int((word >> 16) & 0xff),
                std::bit_cast<std::array<std::uint8_t, 16>>(weights)};
    }
    /** @brief Record and execute one single-row SDOT/ASIMD leaf call. */
    static I dot(I acc, I weights, I activation)
    {
        terms.push_back(term(std::bit_cast<std::array<std::uint32_t, 4>>(activation)[0], weights));
        return Base::dot(acc, weights, activation);
    }
    /** @brief Record and execute one single-row lane-indexed leaf call (decode panels). */
    template <int Group> static I dotLane(I acc, I weights, const typename Base::ActivationBlock &block)
    {
        const auto words = std::bit_cast<std::array<std::uint32_t, 4>>(Group < 4 ? block.low : block.high);
        terms.push_back(term(words[Group % 4], weights));
        return Base::template dotLane<Group>(acc, weights, block);
    }
    /** @brief Record and execute one two-row SMMLA call as four product terms. */
    static void pairDot(I &columns01, I &columns23, I activation, I first, I second)
        requires Base::row_pairs
    {
        const auto words = std::bit_cast<std::array<std::uint32_t, 4>>(activation);
        terms.push_back(term(words[0], first));
        terms.push_back(term(words[1], second));
        terms.push_back(term(words[2], first));
        terms.push_back(term(words[3], second));
        Base::pairDot(columns01, columns23, activation, first, second);
    }
};

/** @brief Build rows whose every four-byte activation word names (row, block, group).
 * @param blocks Logical K blocks.
 * @param rows Physical rows.
 * @return Persistent blocks; the numerical result of this input is irrelevant.
 */
std::vector<Q8_1Block> identityRows(int blocks, int rows)
{
    std::vector<Q8_1Block> activation(std::size_t(blocks) * rows);
    for (int row = 0; row < rows; ++row)
        for (int block = 0; block < blocks; ++block)
        {
            auto &a = activation[std::size_t(row) * blocks + block];
            a.d = 0x3c00;
            a.sum_qs = 0;
            for (int group = 0; group < 8; ++group)
            {
                const std::uint32_t word = std::uint32_t(row) | std::uint32_t(block) << 8 |
                    std::uint32_t(group) << 16 | 0x5au << 24;
                std::memcpy(a.qs + group * 4, &word, sizeof(word));
            }
        }
    return activation;
}

/** @brief Compare the SMMLA walk's product terms with the serial SDOT walk's.
 * @tparam Rows Physical shared rows.
 * @param source Registered source format.
 * @param expert Whether the ordered expert tree is used.
 * @details The oracle is the serial strip walk (Strips = 1), which computes
 * live strips only. The installed single-row decode panel also computes the
 * strips its final panel covers in the chunk's sealed padding columns (never
 * stored); its terms must be exactly the oracle's plus those padding strips.
 * Compact multi-scale encodings keep their own column walk (no panels).
 */
template <std::size_t Rows> void verifyI8MMGrouping(const NativeQuantisedFormatInfo &source, bool expert)
{
    SCOPED_TRACE(::testing::Message() << "codebook=" << int(source.codebook_id) << " rows=" << Rows
        << " expert=" << expert);
    constexpr int blocks = 3, n = 67;
    NativeQuantisedPreparedByteFixture fixture(source, n, blocks * 32);
    if (!expert)
        fixture.descriptor.numerical_policy = CPUProjectionNumericalPolicy::BackendNative;
    fixture.seedPayloadAndMetadata();
    const auto packed = fixture.view();
    const auto activation = identityRows(blocks, Rows);
    std::vector<float> bank(Rows * std::size_t(n));
    const auto observe = [&]<typename Ops, int Strips>(std::size_t rows_per_tile) {
        Ops::terms.clear();
        for (int chunk = 0; chunk < packed.N_padded / 64; ++chunk)
        {
            const int columns = std::min(64, n - chunk * 64);
            const auto bind = [&]<std::size_t R>(std::size_t first) {
                std::array<std::span<const Q8_1Block>, R> input;
                std::array<std::span<float>, R> output;
                for (std::size_t row = 0; row < R; ++row)
                {
                    input[row] = {activation.data() + (first + row) * blocks, std::size_t(blocks)};
                    output[row] = {bank.data() + (first + row) * n + chunk * 64, std::size_t(columns)};
                }
                CPUNativeNEONProjectionTile<R>::bind(packed, input, output, chunk).template executeWith<Ops, Strips>();
            };
            if (rows_per_tile == 1)
                for (std::size_t row = 0; row < Rows; ++row) bind.template operator()<1>(row);
            else
                bind.template operator()<Rows>(0);
        }
        auto terms = Ops::terms;
        std::sort(terms.begin(), terms.end());
        return terms;
    };
    const auto serial = observe.template operator()<GroupingObserver<SDOTOps>, 1>(1);
    const auto pairs = observe.template operator()<GroupingObserver<I8MMOps>, 0>(Rows);
    ASSERT_FALSE(serial.empty());
    ASSERT_EQ(serial.size(), pairs.size());
    ASSERT_TRUE(serial == pairs);
    // Installed decode panels: every live product, plus whole padding strips of
    // the final chunk's last panel (n = 67 leaves one live strip in chunk 1).
    // Compact multi-scale tiles have their own column walk and no panels.
    const auto panels = observe.template operator()<GroupingObserver<SDOTOps>, 0>(1);
    const int width = [&] {
        using Tile = CPUNativeNEONProjectionTile<1>;
        switch (packed.encoding)
        {
        case CPUNativeQuantisedEncoding::SignedInt8SingleScale:
            return Tile::template panelStrips<SDOTOps, CPUNativeQuantisedEncoding::SignedInt8SingleScale>();
        case CPUNativeQuantisedEncoding::Q6KNativeDualScale:
            return Tile::template panelStrips<SDOTOps, CPUNativeQuantisedEncoding::Q6KNativeDualScale>();
        case CPUNativeQuantisedEncoding::NibbleLUTSourceScale:
            return Tile::template panelStrips<SDOTOps, CPUNativeQuantisedEncoding::NibbleLUTSourceScale>();
        case CPUNativeQuantisedEncoding::NibbleLUT:
            return Tile::template panelStrips<SDOTOps, CPUNativeQuantisedEncoding::NibbleLUT>();
        case CPUNativeQuantisedEncoding::Q5NativeSingleScale:
            return packed.is_asymmetric
                ? Tile::template panelStrips<SDOTOps, CPUNativeQuantisedEncoding::Q5NativeSingleScale, true>()
                : Tile::template panelStrips<SDOTOps, CPUNativeQuantisedEncoding::Q5NativeSingleScale, false>();
        case CPUNativeQuantisedEncoding::Q4KSuperblockCoded:
            return Tile::template panelStrips<SDOTOps, CPUNativeQuantisedEncoding::Q4KSuperblockCoded, true>();
        case CPUNativeQuantisedEncoding::CompactMultiScale:
            return 1; // Own four-column strip walks; no panels.
        }
        return 0;
    }();
    ASSERT_GT(width, 0);
    const int live_strips = ((n % 64) + 3) / 4;
    const int padding_strips = (width - live_strips % width) % width;
    ASSERT_EQ(panels.size(), serial.size() + std::size_t(Rows) * blocks * 8 * padding_strips);
    ASSERT_TRUE(std::includes(panels.begin(), panels.end(), serial.begin(), serial.end()));
}
} // namespace
#endif

/** @brief Compare installed decomposition with the original wide arithmetic on exact live spans. */
template <typename Ops, std::size_t Rows, CPUProjectionNumericalPolicy Policy>
void verifyRowDecomposition()
{
    using namespace cpu::native_quantised;
    for (const auto &source : native_quantised_formats::kAllSourceFormats)
    {
        const auto encoding = preparedEncodingForSource(*source.metadata);
        if (!(encoding == CPUNativeQuantisedEncoding::Q5NativeSingleScale ||
              encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale ||
              encoding == CPUNativeQuantisedEncoding::Q4KSuperblockCoded ||
              (encoding == CPUNativeQuantisedEncoding::CompactMultiScale &&
               (source.metadata->codebook_id == 7 || source.metadata->codebook_id == 11))))
            continue;
        for (const int n : {1, 3, 4, 63, 64, 65})
            for (const int blocks : {1, 2, 7, 16, 17, 33})
            {
                NativeQuantisedPreparedByteFixture fixture(*source.metadata, n, blocks * 32);
                fixture.descriptor.numerical_policy = Policy;
                fixture.seedPayloadAndMetadata();
                const auto packed = fixture.view();
                const auto activation = activationRows(blocks, Rows);
                const int stride = n + 3;
                const float dirty = std::bit_cast<float>(0x7fc12345u);
                std::vector<float> installed(Rows * stride, dirty), wide(Rows * stride, dirty);
                std::array<std::span<const Q8_1Block>, Rows> input;
                std::array<std::span<float>, Rows> output, reference;
                for (int repeat = 0; repeat < 2; ++repeat)
                {
                    std::fill(installed.begin(), installed.end(), dirty);
                    std::fill(wide.begin(), wide.end(), dirty);
                    for (int chunk = 0; chunk < packed.N_padded / 64; ++chunk)
                    {
                        const int columns = std::min(64, n - chunk * 64);
                        for (std::size_t row = 0; row < Rows; ++row)
                        {
                            input[row] = {activation.data() + row * blocks, std::size_t(blocks)};
                            output[row] = {installed.data() + row * stride + chunk * 64, std::size_t(columns)};
                            reference[row] = {wide.data() + row * stride + chunk * 64, std::size_t(columns)};
                        }
                        CPUNativeNEONProjectionTile<Rows>::bind(packed, input, output, chunk)
                            .template execute<Ops::instruction>();
                        CPUNativeNEONProjectionWideOracle::execute<Ops, Policy>(
                            CPUNativeNEONProjectionTile<Rows>::bind(packed, input, reference, chunk));
                    }
                    ASSERT_EQ(std::memcmp(installed.data(), wide.data(), installed.size() * sizeof(float)), 0)
                        << "rows=" << Rows << " n=" << n << " blocks=" << blocks;
                }
            }
    }
}

/** @brief Exercise both numerical policies and every decomposed physical width. */
template <typename Ops> void verifyRowDecompositionPolicies()
{
    using P = CPUProjectionNumericalPolicy;
    if constexpr (Ops::instruction != ms::NEONByteDot::I8MM)
    {
        verifyRowDecomposition<Ops, 1, P::GPUAlignedExpert>();
        verifyRowDecomposition<Ops, 1, P::BackendNative>();
    }
    verifyRowDecomposition<Ops, 2, P::GPUAlignedExpert>();
    verifyRowDecomposition<Ops, 2, P::BackendNative>();
    verifyRowDecomposition<Ops, 3, P::GPUAlignedExpert>();
    verifyRowDecomposition<Ops, 4, P::GPUAlignedExpert>();
    verifyRowDecomposition<Ops, 3, P::BackendNative>();
    verifyRowDecomposition<Ops, 4, P::BackendNative>();
}

/** @test Export the actual compiled routing authority for the checker-inventory contract. */
TEST_F(CPUNativeNEONPreparedProjection, DumpRowDecompositionKeys)
{
    for (const auto &key : kNEONRowDecompositions)
        std::printf("NEON_ROW_ROUTE %d:%d:%d:%d:%d:%d %d %d\n", key.rows, int(key.family),
            int(key.policy), int(key.encoding), key.map, int(key.minimum), int(key.pair_family), int(key.tail_family));
}

/** @test Original wide and installed row-pair programs have identical output and guard bytes. */
TEST_F(CPUNativeNEONPreparedProjection, RowDecompositionMatchesOriginalWideProgram)
{
    verifyRowDecompositionPolicies<ms::NEONMultiScaleOps<ms::NEONByteDot::ASIMD>>();
#if defined(__ARM_FEATURE_DOTPROD)
    verifyRowDecompositionPolicies<ms::NEONMultiScaleOps<ms::NEONByteDot::DotProd>>();
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    verifyRowDecompositionPolicies<ms::NEONMultiScaleOps<ms::NEONByteDot::I8MM>>();
#endif
}

/** @test ASIMD directly certifies compact ordered trees independently of numerical policy. */
TEST_F(CPUNativeNEONPreparedProjection, ASIMDCompactOrderedTreesRespectNumericalPolicy)
{ verifyCompactOrderedPolicies<ms::NEONMultiScaleOps<ms::NEONByteDot::ASIMD>>(); }

#if defined(__ARM_FEATURE_DOTPROD)
/** @test SDOT certifies both IQ1_M policy helpers and the IQ3_XXS ordinary-add ordered tree. */
TEST_F(CPUNativeNEONPreparedProjection, DotProdCompactOrderedTreesRespectNumericalPolicy)
{ verifyCompactOrderedPolicies<ms::NEONMultiScaleOps<ms::NEONByteDot::DotProd>>(); }
#endif

#if defined(__ARM_FEATURE_MATMUL_INT8)
/** @test I8MM certifies IQ3_XXS four-row ordinary-add ordered trees against serial and source bytes. */
TEST_F(CPUNativeNEONPreparedProjection, I8MMCompactOrderedTreesRespectNumericalPolicy)
{ verifyCompactOrderedPolicies<I8MMOps>(); }

/** @test Every source's row-pair SMMLA projection retains the complete serial program. */
TEST_F(CPUNativeNEONPreparedProjection, I8MMEverySourceAndEncodingMatchesSerialAndSourceOrder)
{ verifyAllPrepared<I8MMOps>(); }

/** @test Row-pair SMMLA is exact on every compact source and physical tile. */
TEST_F(CPUNativeNEONPreparedProjection, I8MMCompactMatchesSerialAndSourceOrder)
{ verifyAllCompact<I8MMOps>(); }

/** @test SMMLA bytes equal SDOT bytes for every source over N%4 tails and odd rows, untouched past N. */
TEST_F(CPUNativeNEONPreparedProjection, I8MMRowPairsMatchSDOTAcrossNTailsAndOddRows)
{
    for (const auto &source : native_quantised_formats::kAllSourceFormats)
        for (const int n : {1, 2, 3, 4, 5, 6, 7, 62, 63, 65, 66, 67, 130})
            for (const int blocks : {1, 2, 17})
                for (const bool expert : {false, true})
                {
                    verifyI8MMAgainstSDOT<2>(*source.metadata, n, blocks, expert);
                    verifyI8MMAgainstSDOT<3>(*source.metadata, n, blocks, expert);
                    verifyI8MMAgainstSDOT<4>(*source.metadata, n, blocks, expert);
                }
}

/** @test Every encoding's SMMLA operands are exactly the serial SDOT walk's product terms. */
TEST_F(CPUNativeNEONPreparedProjection, I8MMOperandGroupingMatchesSerialSDOTWalk)
{
    for (const auto &source : native_quantised_formats::kAllSourceFormats)
        for (const bool expert : {false, true})
        {
            verifyI8MMGrouping<2>(*source.metadata, expert);
            verifyI8MMGrouping<3>(*source.metadata, expert);
            verifyI8MMGrouping<4>(*source.metadata, expert);
        }
}

/** @test The SMMLA zip/split places each (row, column, K) product in the SDOT walk's lane. */
TEST_F(CPUNativeNEONPreparedProjection, I8MMRowPairPrimitivePlacesEveryProduct)
{
    for (int row = 0; row < 2; ++row)
        for (int column = 0; column < 4; ++column)
            for (int k = 0; k < 8; ++k)
            {
                std::array<std::int8_t, 16> rows{}, first{}, second{};
                rows[row * 8 + k] = 3;
                (k < 4 ? first : second)[column * 4 + k % 4] = -5;
                I8MMOps::I columns01{}, columns23{}, row0, row1;
                I8MMOps::pairDot(columns01, columns23, I8MMOps::activationPair(rows.data(), rows.data() + 8),
                    std::bit_cast<I8MMOps::I>(first), std::bit_cast<I8MMOps::I>(second));
                I8MMOps::splitPair(columns01, columns23, row0, row1);
                // The serial walk: group k/4 of this row, SDOT lane `column`.
                std::array<int, 4> expected0{}, expected1{};
                (row == 0 ? expected0 : expected1)[column] = -15;
                const auto observed0 = std::bit_cast<std::array<int, 4>>(row0);
                const auto observed1 = std::bit_cast<std::array<int, 4>>(row1);
                EXPECT_EQ(observed0, expected0) << row << column << k;
                EXPECT_EQ(observed1, expected1) << row << column << k;
            }
}
#endif

/** @test Invalid binding geometry/aliases fail before any projection writes. */
TEST_F(CPUNativeNEONPreparedProjection, RejectsIncompleteAliasedAndUnalignedExpertBindings)
{
    NativeQuantisedPreparedByteFixture fixture(native_quantised_formats::Q4_0, 3, 32);
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

    NativeQuantisedPreparedByteFixture tail(native_quantised_formats::Q4_0, 3, 33);
    const auto tail_activation = activationRows(2, 1);
    const std::array<std::span<const Q8_1Block>, 1> tail_input{std::span(tail_activation)};
    const std::array<std::span<float>, 1> tail_output{std::span(storage.data(), 3)};
    EXPECT_THROW((void)CPUNativeNEONProjectionTile<1>::bind(tail.view(), tail_input, tail_output, 0), std::invalid_argument);
}

/**
 * @test Publication rejects reserved bits in inline coefficients or a truncated
 * coded header before a tile can write output.
 */
TEST_F(CPUNativeNEONPreparedProjection, RejectsInvalidSourceMetadataBeforeAnyWrite)
{
    // IQ2_S now has an exact-width 8-bit plane, with no reserved storage bits.
    for (const auto *source : {&native_quantised_formats::IQ3_S, &native_quantised_formats::IQ4_XS})
    {
        SCOPED_TRACE(int(source->codebook_id));
        NativeQuantisedPreparedByteFixture fixture(*source, 8, 32);
        fixture.seedPayloadAndMetadata();
        const auto weights = fixture.view();
        const auto activation = activationRows(1, 1);
        const std::array<std::span<const Q8_1Block>, 1> input{std::span(activation)};
        std::array<float, 8> result;
        const std::array<std::span<float>, 1> output{std::span(result)};
        CPUNativeNEONProjectionTile<1>::bind(weights, input, output, 0).execute();
        const auto reference = result;
        if (weights.superblockCoded())
        {
            result.fill(123.0f);
            EXPECT_THROW((void)CPUNativeQuantisedMetadataSeal::validate(fixture.descriptor,
                fixture.bytes.first(fixture.bytes.size() - 1)), std::invalid_argument);
            for (const float value : result) EXPECT_EQ(value, 123.0f);
        }
        result.fill(123.0f);
        CPUNativeNEONProjectionTile<1>::bind(fixture.view(), input, output, 0).execute();
        for (int column = 0; column < 8; ++column)
            EXPECT_EQ(std::bit_cast<std::uint32_t>(result[column]), std::bit_cast<std::uint32_t>(reference[column]));
    }
}

/** @test Native metadata widening and half conversion cover every possible persisted word. */
TEST_F(CPUNativeNEONPreparedProjection, EveryHalfAndOpaqueWordHasExactBits)
{
    ASSERT_TRUE(cpu::ARMFloatingPointProfile::admitted(cpu::ARMFloatingPointProfile::current()));
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
            EXPECT_EQ(std::bit_cast<std::uint32_t>(nativeQuantisedFP16ScaleToFP32(words[lane])),
                std::bit_cast<std::uint32_t>(fp16_to_fp32(words[lane])));
            EXPECT_EQ(raw[lane], words[lane]);
            EXPECT_EQ(std::bit_cast<std::uint32_t>(low[lane]), std::bit_cast<std::uint32_t>(fp16_to_fp32(words[lane])));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(high[lane]),
                std::bit_cast<std::uint32_t>(fp16_to_fp32(static_cast<std::uint16_t>(words[lane] ^ 0xa5b6u))));
        }
    }
}

/** @test Native MXFP4 E8M0 bytes widen exactly; FP16 interpretation is never used. */
TEST_F(CPUNativeNEONPreparedProjection, EveryAdmittedE8M0ByteWidensExactly)
{
    using Ops = ms::NEONMultiScaleOps<ms::NEONByteDot::ASIMD>;
    std::vector<std::uint8_t> words{0};
    for (int e = 2; e <= 252; ++e)
        words.push_back(static_cast<std::uint8_t>(e));
    while (words.size() % 4 != 0) words.push_back(0);
    for (std::size_t first = 0; first < words.size(); first += 4)
    {
        const auto widened = std::bit_cast<std::array<std::uint32_t, 4>>(Ops::e8m0(words.data() + first));
        for (int lane = 0; lane < 4; ++lane)
        {
            const auto e = words[first + lane];
            EXPECT_EQ(widened[lane], std::uint32_t{mxfp4::powerOfTwoBF16Word(e)} << 16) << "exponent=" << int(e);
        }
    }
}

/** @test Native MXFP4 scale bytes remain distinct from FP16 words. */
TEST_F(CPUNativeNEONPreparedProjection, MXFP4ScaleFieldsFollowThePreparedIdentity)
{
    NativeQuantisedPreparedByteFixture native(native_quantised_formats::MXFP4, 65, 64);
    ASSERT_EQ(native.descriptor.encoding, CPUNativeQuantisedEncoding::NibbleLUT);
    EXPECT_EQ(native.view().scaleEncoding(), CPUNativeQuantisedScaleEncoding::E8M0);
    native.seedPayloadAndMetadata();
    const auto view = native.view();
    for (int column = 0; column < 65; ++column)
        for (int block = 0; block < 2; ++block)
        {
            const auto e = view.admittedE8M0Scales(column / 64, block)[column % 64];
            ASSERT_TRUE(e == 0 || mxfp4::exponentFitsNativeE8M0(e)) << "seeded an inadmissible exponent";
            EXPECT_EQ(std::bit_cast<std::uint32_t>(view.blockScale(column / 64, block, column % 64)),
                e == 0 ? 0u : mxfp4::normalScaleFP32Bits(e));
        }
    EXPECT_THROW((void)view.chunkScales(0, 0), std::logic_error);
    for (const auto invalid : {1, 253, 254, 255})
    {
        native.bytes[view.data_stride] = static_cast<std::uint8_t>(invalid);
        EXPECT_THROW((void)native.view(), std::invalid_argument);
    }
    EXPECT_EQ(preparedScaleEncoding(native_quantised_formats::Q8_0.codebook_id, CPUNativeQuantisedEncoding::SignedInt8SingleScale),
        CPUNativeQuantisedScaleEncoding::FP16);
    EXPECT_EQ(preparedScaleEncoding(4, CPUNativeQuantisedEncoding::NibbleLUT), CPUNativeQuantisedScaleEncoding::FP16);
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

namespace
{
/**
 * @brief Prove dotLane<G> over a loaded block equals dot() over group G's broadcast word.
 * @tparam Ops Admitted family.
 * @details Seeds include the extreme byte pairs (-128 x -128, -128 x 127) on
 * every lane, so widening, sign and accumulation are all exercised.
 */
template <typename Ops> void verifyDotLanes()
{
    using I = typename Ops::I;
    for (int seed = 0; seed < 64; ++seed)
    {
        std::array<std::int8_t, 32> activation;
        std::array<std::int8_t, 16> weights;
        for (int k = 0; k < 32; ++k)
            activation[k] = static_cast<std::int8_t>(seed < 2 ? -128 : (seed * 37 + k * 101) % 256 - 128);
        for (int k = 0; k < 16; ++k)
            weights[k] = static_cast<std::int8_t>(seed == 0 ? -128 : seed == 1 ? 127 : (seed * 53 + k * 71) % 256 - 128);
        const auto block = Ops::activationBlock(activation.data());
        const auto weight_vector = std::bit_cast<I>(weights);
        const I acc{seed, -seed, 1 << 20, -(1 << 20)};
        [&]<int... G>(std::integer_sequence<int, G...>) {
            ([&] {
                std::int32_t word;
                std::memcpy(&word, activation.data() + G * 4, sizeof(word));
                const I expected = Ops::dot(acc, weight_vector, I{} + word);
                const I actual = Ops::template dotLane<G>(acc, weight_vector, block);
                EXPECT_EQ(std::memcmp(&expected, &actual, sizeof(I)), 0) << "seed=" << seed << " group=" << G;
            }(), ...);
        }(std::make_integer_sequence<int, 8>{});
    }
}
} // namespace

/** @test Lane-indexed activation dots equal broadcast-word dots (ASIMD). */
TEST(CPUNativeNEONMultiScaleOps, ASIMDDotLaneEqualsBroadcastDot)
{ verifyDotLanes<ms::NEONMultiScaleOps<ms::NEONByteDot::ASIMD>>(); }

#if defined(__ARM_FEATURE_DOTPROD)
/** @test Lane-indexed activation dots equal broadcast-word dots (DotProd). */
TEST(CPUNativeNEONMultiScaleOps, DotProdDotLaneEqualsBroadcastDot)
{
    cpu::ARMCPUCapabilities::native().requireCompiledEnvelope(); // The envelope is admitted, never skipped.
    verifyDotLanes<ms::NEONMultiScaleOps<ms::NEONByteDot::DotProd>>();
}
#endif

namespace
{
/** @brief Test-only FPCR access; production code never writes FPCR. */
std::uint64_t readFpcr() { return cpu::ARMFloatingPointProfile::current(); }
void writeFpcr(std::uint64_t value) { asm volatile("msr fpcr, %0" :: "r"(value) : "memory"); }

/** @brief Run @p body with only the rounding/flush/half-format FPCR controls replaced, then restore. */
template <typename Body> void underFpcrProfile(std::uint64_t profile, Body &&body)
{
    using P = cpu::ARMFloatingPointProfile;
    const auto saved = readFpcr();
    writeFpcr((saved & ~P::kControls) | profile);
    body();
    writeFpcr(saved);
}

constexpr std::uint64_t kRModeMinusInfinity = 2ull << 22;

/** @brief One coded Q4_K publication and an M = 2 workload bound to the tile and batch entries. */
struct Q4KEntryWorkload
{
    NativeQuantisedPreparedByteFixture fixture{native_quantised_formats::Q4_K, 67, 288};
    std::vector<Q8_1Block> activation = activationRows(9, 2);
    std::vector<float> output = std::vector<float>(2 * 67);

    Q4KEntryWorkload() { fixture.seedPayloadAndMetadata(); }

    /** @brief Execute through the public tile entry (both rows of chunk 0). */
    void tile()
    {
        const auto view = fixture.view();
        std::array<std::span<const Q8_1Block>, 2> input{std::span<const Q8_1Block>(activation).first(9),
                                                        std::span<const Q8_1Block>(activation).subspan(9, 9)};
        std::array<std::span<float>, 2> out{std::span<float>(output).first(64), std::span<float>(output).subspan(67, 64)};
        CPUNativeNEONProjectionTile<2>::bind(view, input, out, 0).execute();
    }

    /**
     * @brief Execute every task of the batch entry on this thread.
     * @details runTask() is the unit each worker of execute()'s workshare
     * runs, and it admits that worker's FPCR. Driving it directly keeps this
     * core-free test independent of an OpenMP runtime.
     */
    void batch()
    {
        const auto view = fixture.view();
        const auto bound = CPUNativeNEONProjectionBatch::bind(view, activation, output, 2, 67, NEONProjectionRowTile::Pair);
        for (std::int64_t task = 0; task < bound.taskCount(); ++task)
            bound.runTask(task);
    }
};
} // namespace

/**
 * @test The shared profile predicate admits exactly RNE with FZ, FZ16, AHP, DN and the FEAT_AFP controls
 * (FIZ, AH, NEP) clear. AH changes NaN propagation of FADD and FMLA, which the GPU-aligned earlier-NaN
 * contract depends on; CPUs without FEAT_AFP read these bits as zero, so the predicate is the witness there.
 */
TEST(CPUNativeNEONFloatingPointProfile, AdmitsOnlyTheContractProfile)
{
    using P = cpu::ARMFloatingPointProfile;
    EXPECT_TRUE(P::admitted(0));
    EXPECT_TRUE(P::admitted(1ull << 21)); // FPCR.IDE, an exception-trap enable outside the numeric controls.
    for (const std::uint64_t foreign :
         std::initializer_list<std::uint64_t>{1ull << 22, 2ull << 22, 3ull << 22, P::kFZ, P::kFZ16, P::kAHP, P::kDN,
             P::kFIZ, P::kAH, P::kNEP})
        EXPECT_FALSE(P::admitted(foreign)) << std::hex << foreign;
    EXPECT_EQ(P::kControls, (3ull << 22) | (1ull << 24) | (1ull << 19) | (1ull << 26) | (1ull << 25) | 0x7ull);
    EXPECT_TRUE(P::admitted(readFpcr())) << "the test process must start in the contract profile";
}

/**
 * @test The ambient-FPCR hazard the admission guards against is real on this
 * machine: binary16 products flush a live subnormal scale under FZ16, and the
 * hardware narrowing rounds toward the FPCR mode, and default-NaN mode
 * replaces the sign and payload of a widened NaN half (the reader's -0
 * canonicalization alone could not fix any of these). Under DN the coded Q4_K
 * reader itself loses the codec's negative NaN minimum of `-(inf * 0)`.
 */
TEST(CPUNativeNEONFloatingPointProfile, ForeignProfilesChangeNativeArithmetic)
{
    volatile float third = -1.0f / 3.0f; // 0xB555 to nearest, 0xB556 toward -inf.
    volatile std::uint16_t narrowed_nearest = 0, narrowed_down = 0;
    underFpcrProfile(0, [&] { narrowed_nearest = vget_lane_u16(vreinterpret_u16_f16(vcvt_f16_f32(vdupq_n_f32(third))), 0); });
    underFpcrProfile(kRModeMinusInfinity,
        [&] { narrowed_down = vget_lane_u16(vreinterpret_u16_f16(vcvt_f16_f32(vdupq_n_f32(third))), 0); });
    EXPECT_EQ(narrowed_nearest, 0xB555);
    EXPECT_EQ(narrowed_down, 0xB556);
    // FCVT half -> single of a negative NaN with payload: propagated, or the default NaN.
    volatile std::uint16_t nan_half = 0xFE01;
    volatile std::uint32_t widened_propagated = 0, widened_default = 0;
    const auto widen = [&] {
        return vgetq_lane_u32(vreinterpretq_u32_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(nan_half)))), 0);
    };
    underFpcrProfile(0, [&] { widened_propagated = widen(); });
    underFpcrProfile(cpu::ARMFloatingPointProfile::kDN, [&] { widened_default = widen(); });
    EXPECT_EQ(widened_propagated, 0xFFC02000u);
    EXPECT_EQ(widened_default, 0x7FC00000u);
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    {
        // The fused Q4_K reader on d = 1.0, dmin = +inf, code 0: the codec's
        // minimum is the negated product -(inf * 0) = 0xFFC00000.
        alignas(16) std::array<std::uint8_t, 256> header{};
        alignas(16) std::array<std::uint8_t, 96> codes{};
        for (int column = 0; column < 64; ++column)
            coded_metadata::storeSuperblockWord(header.data(), 5, column,
                native_quantised_superblock::packSuperblockWord(0x3C00, 0x7C00));
        const auto reader_minimum = [&] {
            float32x4_t scale, minimum;
            coded_metadata::neon_half::q4KFactors(codes.data(), 0,
                coded_metadata::neon_half::q4KHalfBase(header.data(), 0), scale, minimum);
            return vgetq_lane_u32(vreinterpretq_u32_f32(minimum), 0);
        };
        volatile std::uint32_t minimum_contract = 0, minimum_default_nan = 0;
        underFpcrProfile(0, [&] { minimum_contract = reader_minimum(); });
        underFpcrProfile(cpu::ARMFloatingPointProfile::kDN, [&] { minimum_default_nan = reader_minimum(); });
        const auto staged = native_quantised_superblock::decodeSuperblockMetadata<5>(0,
            native_quantised_superblock::packSuperblockWord(0x3C00, 0x7C00));
        EXPECT_EQ(staged.secondary, 0xFE00);
        EXPECT_EQ(minimum_contract, 0xFFC00000u);
        EXPECT_EQ(minimum_default_nan, 0x7FC00000u);
    }
    volatile std::uint16_t tiny = 0x0001, two = 0x4000, product_default = 0, product_flushed = 0;
    const auto multiply = [&] {
        return vget_lane_u16(vreinterpret_u16_f16(vmul_f16(vreinterpret_f16_u16(vdup_n_u16(tiny)),
                                                           vreinterpret_f16_u16(vdup_n_u16(two)))), 0);
    };
    underFpcrProfile(0, [&] { product_default = multiply(); });
    underFpcrProfile(cpu::ARMFloatingPointProfile::kFZ16, [&] { product_flushed = multiply(); });
    EXPECT_EQ(product_default, 0x0002);
    EXPECT_EQ(product_flushed, 0x0000);
#endif
}

/**
 * @test The coded Q4_K projection runs only under the contract profile: its
 * tile and batch entries fail fatally, with the profile diagnostic, under
 * FPCR.RMode = toward -inf, FPCR.FZ16 = 1 and FPCR.DN = 1, and execute normally (and
 * byte-identically) after the caller's profile is restored.
 */
TEST(CPUNativeNEONFloatingPointProfile, Q4KEntriesRejectForeignFpcrProfiles)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    Q4KEntryWorkload workload;
    workload.batch();
    const auto reference = workload.output;
    for (const std::uint64_t foreign : std::initializer_list<std::uint64_t>{
             kRModeMinusInfinity, cpu::ARMFloatingPointProfile::kFZ16, cpu::ARMFloatingPointProfile::kDN})
    {
        SCOPED_TRACE(::testing::Message() << "FPCR controls 0x" << std::hex << foreign);
        EXPECT_DEATH(underFpcrProfile(foreign, [&] { workload.tile(); }),
            "Native NEON quantized projection requires the process FP profile");
        EXPECT_DEATH(underFpcrProfile(foreign, [&] { workload.batch(); }),
            "Native NEON quantized projection batch requires the process FP profile");
    }
    std::fill(workload.output.begin(), workload.output.end(), -1.0f);
    workload.batch();
    EXPECT_EQ(std::memcmp(workload.output.data(), reference.data(), reference.size() * sizeof(float)), 0);
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
/** @test The I8MM family's single-row lane leaf equals its broadcast SDOT leaf. */
TEST(CPUNativeNEONMultiScaleOps, I8MMDotLaneEqualsBroadcastDot)
{
    cpu::ARMCPUCapabilities::native().requireCompiledEnvelope(); // The envelope is admitted, never skipped.
    verifyDotLanes<ms::NEONMultiScaleOps<ms::NEONByteDot::I8MM>>();
}
#endif
} // namespace llaminar2::test
