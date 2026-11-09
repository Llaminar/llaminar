/**
 * @file Test__CPUNativeNEONPreparedProjection.cpp
 * @brief Native prepared projection leaves against independent serial/source-order oracles.
 *
 * Tests execute the actual native interleaved decoders and rounded K traversal,
 * including the shared compact program, not row replay. Four-column ASIMD and
 * explicitly targeted DotProd implementations
 * share payload decode across up to four rows; the I8MM family consumes those
 * rows in SMMLA pairs and must reproduce the SDOT bytes and operand grouping. Logical N tails must neither
 * load nor publish outside the supplied output extent. Full factory/model and
 * worker-scaling certification remains a separate production integration gate.
 */
#include "kernels/cpu/gemm/CPUNativeNEONPreparedProjection.h"
#include "utils/ARMCPUFeatures.h"
#include "utils/CPUExplicitRoundingOracle.h"
#include "utils/NativeQuantisedPreparedByteFixture.h"

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
/** @brief Cover all registered sources and all six physical encodings on both arithmetic policies. */
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
    const auto observe = [&]<typename Ops>(std::size_t rows_per_tile) {
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
                CPUNativeNEONProjectionTile<R>::bind(packed, input, output, chunk).template executeWith<Ops>();
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
    const auto serial = observe.template operator()<GroupingObserver<SDOTOps>>(1);
    const auto pairs = observe.template operator()<GroupingObserver<I8MMOps>>(Rows);
    ASSERT_FALSE(serial.empty());
    ASSERT_EQ(serial.size(), pairs.size());
    ASSERT_TRUE(serial == pairs);
}
} // namespace
#endif

#if defined(__ARM_FEATURE_MATMUL_INT8)
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
 * @test Opaque source words cannot hide reserved bits by masking them into a
 * usable coefficient. Publication admission rejects them before any tile or
 * worker team exists, so no output is ever authored.
 */
TEST_F(CPUNativeNEONPreparedProjection, RejectsReservedSourceCoefficientBitsBeforeAnyWrite)
{
    for (const auto *source : {&native_quantised_formats::IQ2_S, &native_quantised_formats::IQ3_S, &native_quantised_formats::IQ4_XS})
    {
        SCOPED_TRACE(int(source->codebook_id));
        NativeQuantisedPreparedByteFixture fixture(*source, 8, 32);
        fixture.seedPayloadAndMetadata();
        const auto weights = fixture.view();
        const auto secondary = weights.data_stride + (weights.usesInlineCompensation() ? 256 : 128);
        const auto activation = activationRows(1, 1);
        const std::array<std::span<const Q8_1Block>, 1> input{std::span(activation)};
        std::array<float, 8> result;
        const std::array<std::span<float>, 1> output{std::span(result)};
        CPUNativeNEONProjectionTile<1>::bind(weights, input, output, 0).execute();
        const auto reference = result;
        // Every lane of the unit, including the 56 padding columns past N=8.
        for (int bad_column = 0; bad_column < 64; ++bad_column)
        {
            SCOPED_TRACE(bad_column);
            std::uint16_t original = 0;
            auto *word = fixture.bytes.data() + secondary + bad_column * 2;
            std::memcpy(&original, word, sizeof(original));
            const std::uint16_t invalid = 0xffff;
            std::memcpy(word, &invalid, sizeof(invalid));
            result.fill(123.0f);
            EXPECT_THROW((void)fixture.view(), std::invalid_argument);
            for (const float value : result) EXPECT_EQ(value, 123.0f);
            std::memcpy(word, &original, sizeof(original));
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
