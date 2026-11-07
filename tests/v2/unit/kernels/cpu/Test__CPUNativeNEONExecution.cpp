/**
 * @file Test__CPUNativeNEONExecution.cpp
 * @brief Production ARM projection entrypoints over actual retained prepared owners.
 *
 * Small canonical byte fixtures are copied once into the real prepared-owner
 * backing before invocation. No tensor preparer is substituted in production.
 * Tests cover the owner/entrypoint/workspace ABI, not full graph, slot/migration,
 * model-serving or performance certification. Serial M1 calls are test oracles.
 */
#include "kernels/cpu/gemm/CPUNativeVNNIGemv.h"
#include "kernels/cpu/gemm/CPUQ8ResidualCorrection.h"
#include "utils/NativeVNNIPreparedByteFixture.h"
#include "utils/CPUExplicitRoundingOracle.h"
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <cstring>
#include <bit>
#include <vector>

namespace llaminar2::test
{
namespace
{
using namespace cpu::native_vnni;

/** @brief Restore caller thread policy after exact-budget native workshare tests. */
class Budget final
{
public:
    /** @brief Save the caller's budget and disable runtime team shrinking. */
    Budget() : old_(omp_get_max_threads()), dynamic_(omp_get_dynamic())
    { omp_set_dynamic(0); omp_set_num_threads(1); }
    /** @brief Restore the caller's unchanged policy on every exit. */
    ~Budget() { omp_set_num_threads(old_); omp_set_dynamic(dynamic_); }
private:
    int old_, dynamic_;
};

/** @brief Admit the entire translation unit's native instruction envelope. */
class CPUNativeNEONExecution : public ::testing::Test
{
protected:
    /** @brief Missing optional instruction families are fatal, never skipped. */
    static void SetUpTestSuite()
    {
        cpu::ARMCPUCapabilities::native().require(cpu::ARMCPUFeature::ASIMD);
#if defined(__ARM_FEATURE_DOTPROD)
        cpu::ARMCPUCapabilities::native().require(cpu::ARMCPUFeature::DotProd);
#endif
    }
};

/** @brief Materialize a bounded real prepared owner before any timed/executing operation.
 * @param source Original source catalog identity.
 * @param n Live output columns.
 * @param k Logical reduction width.
 * @param policy Complete arithmetic identity.
 * @return Actual aligned resident owner with no archive/decoded-weight mirror.
 */
CPUNativeVNNIPackedWeights owner(const NativeVnniFormatInfo &source, int n, int k,
    CPUProjectionNumericalPolicy policy = CPUProjectionNumericalPolicy::BackendNative)
{
    NativeVNNIPreparedByteFixture fixture(source, n, k);
    fixture.descriptor.numerical_policy = policy;
    fixture.seedPayloadAndMetadata();
    const auto view = fixture.view();
    CPUNativeVNNIPackedWeights result;
    result.N = view.N; result.K = view.K; result.N_padded = view.N_padded;
    result.blocks_per_row = view.blocks_per_row;
    result.payload_bytes = view.source_payload_bytes;
    result.codebook_id = view.codebook_id; result.is_superblock = view.is_superblock;
    result.is_asymmetric = view.is_asymmetric; result.encoding = view.encoding;
    result.numerical_policy = policy; result.data_stride = view.data_stride;
    result.interleaved_block_stride = view.interleaved_block_stride;
    result.native_interleaved.resize(fixture.bytes.size());
    std::copy(fixture.bytes.begin(), fixture.bytes.end(), result.native_interleaved.begin());
    return result;
}

/** @brief Populate original-order Q8 rows independently of any row scheduler.
 * @param rows Original physical capacity.
 * @param blocks Complete K blocks per physical row.
 * @param salt Distinct input identity for multi-input bundles.
 * @return Persistent complete Q8 bank with signed extrema and exact sums.
 */
std::vector<Q8_1Block> q8Rows(int rows, int blocks, int salt = 0)
{
    std::vector<Q8_1Block> bank(std::size_t(rows) * blocks);
    for (int row = 0; row < rows; ++row)
        for (int block = 0; block < blocks; ++block)
        {
            auto &q = bank[std::size_t(row) * blocks + block];
            q.d = std::uint16_t(0x3401 + (row * 17 + block * 33 + salt) % 512);
            int sum = 0;
            for (int i = 0; i < 32; ++i)
            { q.qs[i] = std::int8_t((row * 37 + block * 13 + i * 73 + salt) % 256 - 128); sum += q.qs[i]; }
            q.sum_qs = std::int16_t(sum);
        }
    return bank;
}

/** @brief Compare complete physical banks including untouched canaries.
 * @tparam T Trivial published element type.
 * @param actual New native publication.
 * @param expected Independent serial publication with identical original holes.
 */
template <typename T> void bytes(const std::vector<T> &actual, const std::vector<T> &expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)), 0);
}

/** @brief Author a reference through the canonical native M1 entrypoint, only in tests.
 * @param packed Actual retained prepared owner.
 * @param input Complete original Q8 bank.
 * @param output Original physical output bank with prefilled inactive/padded bytes.
 * @param rows Original physical M.
 * @param stride Original output row stride.
 * @param live Original useful prefix length.
 */
void serial(const CPUNativeVNNIPackedWeights &packed, const std::vector<Q8_1Block> &input,
    std::vector<float> &output, int rows, int stride, int live)
{
    omp_set_num_threads(1);
    for (int row = 0; row < rows; ++row)
        if (row < live) gemv_native_vnni_preq(packed, input.data() + std::size_t(row) * packed.blocks_per_row,
            output.data() + std::size_t(row) * stride, {});
}
/** @brief Independent memory-rounded /4 SiLU reference for complete native channel vectors.
 * @param input One original finite FP32 channel.
 * @return The established CPU range-reduction/FMA/division program's result.
 */
float quarterSilu(float input)
{
    const float value = memoryRoundedMultiply(input, 0.25f);
    const float argument = std::clamp(-value, -88.0f, 88.0f);
    const float scaled = memoryRoundedMultiply(argument, 1.4426950408889634f);
    const float integer = std::nearbyint(scaled);
    const float fraction = memoryRoundedAdd(scaled, -integer);
    float polynomial = std::fma(0.001333355814642f, fraction, 0.009618129107629f);
    polynomial = std::fma(polynomial, fraction, 0.055504108664822f);
    polynomial = std::fma(polynomial, fraction, 0.240226506959101f);
    polynomial = std::fma(polynomial, fraction, 0.693147180559945f);
    polynomial = std::fma(polynomial, fraction, 1.0f);
    const auto exponent = std::uint32_t(int(integer) + 127) << 23;
    const float exponential = memoryRoundedMultiply(polynomial, std::bit_cast<float>(exponent));
    const float denominator = memoryRoundedAdd(1.0f, exponential);
    volatile float inverse = 1.0f / denominator;
    return memoryRoundedMultiply(value, inverse);
}
} // namespace

/** @test Real prepared owners retain native M1 bytes for every source, policy and grouped boundary. */
TEST_F(CPUNativeNEONExecution, AllOwnersSerialGroupedAndPrefillRetainBytes)
{
    const Budget budget;
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
        for (const auto policy : {CPUProjectionNumericalPolicy::BackendNative, CPUProjectionNumericalPolicy::GPUAlignedExpert})
        {
            SCOPED_TRACE(::testing::Message() << "source=" << int(source.metadata->codebook_id) << " super=" << source.metadata->is_superblock << " policy=" << int(policy));
            const auto packed = owner(*source.metadata, 65, 17 * 32, policy);
            ASSERT_TRUE(packed.hasInterleavedData());
            EXPECT_TRUE(packed.payload.empty()); EXPECT_TRUE(packed.int8_flat.empty());
            for (const int rows : {1, 2, 3, 4, 5, 8, 15, 16, 31, 33})
            {
                const auto input = q8Rows(rows, packed.blocks_per_row);
                constexpr int stride = 68;
                const auto size = std::size_t(rows - 1) * stride + packed.N;
                std::vector<float> expected(size + 4, 123456.0f);
                serial(packed, input, expected, rows, stride, rows);
                EXPECT_NE(expected[0], 123456.0f); // A no-op facade cannot certify itself.
                for (const int workers : {1, 3})
                {
                    omp_set_num_threads(workers);
                    std::vector<float> actual(size + 4, 123456.0f);
                    gemm_native_vnni_preq_decode_equivalent_rows(packed, input.data(), actual.data(), {}, rows, stride);
                    bytes(actual, expected);
                    std::fill(actual.begin(), actual.end(), 123456.0f);
                    gemm_native_vnni_preq(packed, input.data(), actual.data(), {}, rows, stride);
                    bytes(actual, expected);
                }
                std::int32_t live = rows / 2;
                const auto work = ProjectionRowWork::fromPrefix(DeviceRowRange::deviceCounted(rows, &live));
                std::fill(expected.begin(), expected.end(), 123456.0f);
                serial(packed, input, expected, rows, stride, live);
                std::vector<float> actual(size + 4, 123456.0f);
                omp_set_num_threads(3);
#pragma omp parallel
                {
                    gemm_native_vnni_preq_decode_equivalent_rows(packed, input.data(), actual.data(), {}, rows, stride,
                        ISAPath::AUTO, VerifierRowsPolicy::Auto, true, 0, &work);
                }
                bytes(actual, expected);
            }
        }
}

/** @test Every explicit native N owner and admitted instruction family retains the serial program. */
TEST_F(CPUNativeNEONExecution, NativeInstructionAndNChunkCandidatesStayExact)
{
    const Budget budget;
    const auto packed = owner(native_vnni_formats::Q4_K, 193, 256);
    const auto input = q8Rows(5, packed.blocks_per_row);
    std::vector<float> expected(5 * 193 + 4, 123456.0f);
    serial(packed, input, expected, 5, 193, 5);
    for (const int chunks : {1, 2, 4, 8, 16})
    {
        std::vector<float> actual(expected.size(), 123456.0f);
        omp_set_num_threads(3);
        gemm_native_vnni_preq_decode_equivalent_rows(packed, input.data(), actual.data(), {}, 5, 193,
            ISAPath::NEON, VerifierRowsPolicy::WideRows, true, chunks);
        bytes(actual, expected);
#if defined(__ARM_FEATURE_DOTPROD)
        std::fill(actual.begin(), actual.end(), 123456.0f);
        gemm_native_vnni_preq_decode_equivalent_rows(packed, input.data(), actual.data(), {}, 5, 193,
            ISAPath::NEON_DOTPROD, VerifierRowsPolicy::Pairwise, true, chunks);
        bytes(actual, expected);
#endif
    }
    std::array<float, 193> output{};
    for (const auto foreign : {ISAPath::AVX2, ISAPath::AVX512, ISAPath::SCALAR})
        EXPECT_THROW(gemv_native_vnni_preq(packed, input.data(), output.data(), {}, foreign), std::invalid_argument);
}

/** @test Native fused descriptors retain per-member rows/strides/bias and reject sibling aliases. */
TEST_F(CPUNativeNEONExecution, FusedActualOwnersRetainMemberGeometryAndInputs)
{
    const Budget budget;
    constexpr int rows = 7;
    std::array<CPUNativeVNNIPackedWeights, 3> packed{
        owner(native_vnni_formats::Q4_0, 5, 256), owner(native_vnni_formats::Q6_K, 65, 256), owner(native_vnni_formats::IQ2_S, 129, 256)};
    std::array<std::vector<Q8_1Block>, 3> input;
    std::array<std::vector<float>, 3> actual, expected, bias;
    std::array<FusedVerifierRowsDesc, 3> descriptors;
    for (int p = 0; p < 3; ++p)
    {
        const int m = rows - p * 2, stride = packed[p].N + 3;
        input[p] = q8Rows(m, packed[p].blocks_per_row, p * 17);
        const auto size = std::size_t(m - 1) * stride + packed[p].N;
        actual[p].assign(size + 4, 123456.0f); expected[p] = actual[p];
        bias[p].resize(packed[p].N);
        for (int n = 0; n < packed[p].N; ++n) bias[p][n] = float(n - 7) * 0.017891f;
        serial(packed[p], input[p], expected[p], m, stride, m);
        for (int row = 0; row < m; ++row)
            for (int n = 0; n < packed[p].N; ++n)
                expected[p][std::size_t(row) * stride + n] = memoryRoundedAdd(expected[p][std::size_t(row) * stride + n], bias[p][n]);
        descriptors[p] = {.packed = &packed[p], .output = actual[p].data(), .bias = bias[p].data(), .N = packed[p].N,
            .ldc = stride, .rows = m, .input = input[p].data()};
    }
    auto metadata = descriptors;
    for (auto &d : metadata) { d.input = nullptr; d.output = nullptr; d.bias = nullptr; }
    EXPECT_EQ(planNativeVNNIFusedRows(metadata, rows, 3).sharedPartialFloats(), 0u);
    omp_set_num_threads(3);
    EXPECT_TRUE(gemm_native_vnni_fused_verifier_rows_preq(nullptr, descriptors.data(), {}, 3, rows, 8));
    for (int p = 0; p < 3; ++p) bytes(actual[p], expected[p]);
    auto aliases = descriptors;
    aliases[1].output = aliases[0].output;
    for (auto &bank : actual) std::fill(bank.begin(), bank.end(), 123456.0f);
    EXPECT_THROW((void)gemm_native_vnni_fused_verifier_rows_preq(nullptr, aliases.data(), {}, 3, rows, 8), std::invalid_argument);
    for (const auto &bank : actual) for (const auto value : bank) EXPECT_EQ(value, 123456.0f);
}

/** @test FP32 publication/BLAS and paired-Q8 operation use real owner bindings for every format. */
TEST_F(CPUNativeNEONExecution, AllSourcesFP32EpilogueAndTwoTermRetainOriginalRows)
{
    const Budget budget;
    constexpr int rows = 5, n = 5, k = 256, blocks = k / 32;
    std::array<float, rows * k> source;
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = float(int(i % 255) - 127) * 0.013457f;
    std::int32_t live = 3;
    const auto range = DeviceRowRange::deviceCounted(rows, &live);
    const auto work = ProjectionRowWork::fromPrefix(range);
    for (const auto &format : native_vnni_formats::kAllSourceFormats)
    {
        SCOPED_TRACE(::testing::Message() << "source=" << int(format.metadata->codebook_id) << " super=" << format.metadata->is_superblock);
        const auto packed = owner(*format.metadata, n, k);
        std::vector<Q8_1Block> primary(rows * blocks), residual(rows * blocks), reference(rows * blocks), reference_residual(rows * blocks);
        for (int row = 0; row < rows; ++row)
            for (int block = 0; block < blocks; ++block)
            {
                const auto offset = std::size_t(row) * blocks + block;
                const auto *input = source.data() + std::size_t(row) * k + block * 32;
                quantizeActivationBlock(input, reference[offset], 32, packed.numerical_policy);
                std::array<float, 32> difference;
                const float scale = fp16_to_fp32(reference[offset].d);
                for (int i = 0; i < 32; ++i)
                    difference[i] = memoryRoundedAdd(input[i], -memoryRoundedMultiply(scale, float(reference[offset].qs[i])));
                quantizeActivationBlock(difference.data(), reference_residual[offset], 32, packed.numerical_policy);
            }
        std::vector<float> product(rows * n + 4, 123456.0f), actual(product.size(), 123456.0f), expected = actual;
        serial(packed, reference, expected, rows, n, live);
        for (int i = 0; i < live * n; ++i)
            // Canonical BLAS epilogue: round(beta*prior), then one fused
            // alpha*product + scaled_prior (the same program on every ISA/GPU).
            expected[i] = std::fma(0.75f, expected[i], memoryRoundedMultiply(123456.0f, -0.125f));
        omp_set_num_threads(3);
        multiply_native_vnni_with_epilogue(packed, source.data(), actual.data(), rows, n, 0.75f, -0.125f,
            primary, product, {}, &work);
        bytes(actual, expected);
        std::fill(actual.begin(), actual.end(), 123456.0f);
        expected = actual;
        std::vector<float> second = actual;
        serial(packed, reference, expected, rows, n, live);
        serial(packed, reference_residual, second, rows, n, live);
        for (int i = 0; i < live * n; ++i) expected[i] = memoryRoundedAdd(expected[i], second[i]);
        omp_set_num_threads(3);
        cpuQ8ResidualCorrectedProjection(packed, source.data(), actual.data(), range, primary, residual);
        bytes(actual, expected);
        EXPECT_EQ(std::memcmp(primary.data(), reference.data(), std::size_t(live) * blocks * sizeof(Q8_1Block)), 0);
        EXPECT_EQ(std::memcmp(residual.data(), reference_residual.data(), std::size_t(live) * blocks * sizeof(Q8_1Block)), 0);
    }
}

/** @test Native HC polynomial/FMA/division lanes preserve independent source-order bytes. */
TEST_F(CPUNativeNEONExecution, NativeHCChannelsMatchIndependentPolynomial)
{
    std::array<float, 260> source, actual{}, expected{};
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = float(int(i) - 130) * 3.123457f;
    for (int i = 0; i < 260; ++i) expected[i] = quarterSilu(source[i]);
    qwen4exp::cpu_detail::residualDownChannels(source.data(), actual.data(), 0, 260, 4);
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), sizeof(actual)), 0);
}

/** @test Native HC publication retains original FP32 activation and canonical two-term output for every source. */
TEST_F(CPUNativeNEONExecution, AllSourcesHCInputAndTwoTermPublicationMatchIndependentBytes)
{
    const Budget budget;
    constexpr int rows = 5, live = 3, k = 256, n = 5, blocks = k / 32;
    std::array<float, rows * k> raw;
    for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = float(int(i % 255) - 127) * 0.137891f;
    for (const auto &format : native_vnni_formats::kAllSourceFormats)
    {
        const auto packed = owner(*format.metadata, n, k);
        std::vector<float> activated(rows * k, 123456.0f), expected_activation = activated;
        std::vector<Q8_1Block> primary(rows * blocks), residual(rows * blocks), reference(rows * blocks), secondary(rows * blocks);
        for (int row = 0; row < live; ++row)
            for (int block = 0; block < blocks; ++block)
            {
                const auto offset = std::size_t(row) * blocks + block, base = std::size_t(row) * k + block * 32;
                for (int i = 0; i < 32; ++i) expected_activation[base + i] = quarterSilu(raw[base + i]);
                quantizeActivationBlock(expected_activation.data() + base, reference[offset], 32, packed.numerical_policy);
                std::array<float, 32> difference;
                const float scale = fp16_to_fp32(reference[offset].d);
                for (int i = 0; i < 32; ++i)
                    difference[i] = memoryRoundedAdd(expected_activation[base + i], -memoryRoundedMultiply(scale, float(reference[offset].qs[i])));
                quantizeActivationBlock(difference.data(), secondary[offset], 32, packed.numerical_policy);
            }
        std::vector<float> actual(rows * n + 4, 123456.0f), expected = actual, second = actual;
        serial(packed, reference, expected, rows, n, live);
        serial(packed, secondary, second, rows, n, live);
        for (int i = 0; i < live * n; ++i) expected[i] = memoryRoundedAdd(expected[i], second[i]);
        std::int32_t active = live;
        omp_set_num_threads(3);
        cpuHCResidualDownQ8Projection(packed, raw.data(), activated.data(), actual.data(),
            DeviceRowRange::deviceCounted(rows, &active), primary, residual);
        bytes(activated, expected_activation);
        bytes(actual, expected);
    }
}

/** @test Both input-publication writes and final beta writes preserve every retained authority. */
TEST_F(CPUNativeNEONExecution, RejectsInputPublicationAndBetaWorkAliasesBeforeWrites)
{
    const Budget budget;
    constexpr int rows = 5, k = 256, blocks = k / 32;
    auto packed = owner(native_vnni_formats::Q4_0, 5, k);
    std::vector<float> source(rows * k, 0.125f), output(rows * 5, 123456.0f), product(rows * 5, 123456.0f);
    std::vector<Q8_1Block> activation(rows * blocks);
    std::memset(activation.data(), 0x71, activation.size() * sizeof(Q8_1Block));
    const auto before_activation = activation;
    const auto before_source = source;
    const std::vector<std::uint8_t> before_weights(packed.native_interleaved.begin(), packed.native_interleaved.end());
    const auto reject = [&](std::span<Q8_1Block> bank, const ProjectionRowWork *work) {
        EXPECT_THROW(multiply_native_vnni_with_epilogue(packed, source.data(), output.data(), rows, 5,
            1.0f, 0.0f, bank, {}, {}, work), std::invalid_argument);
    };
    // An interior cross-row alias corrupts unread FP32 values unless the first
    // (writing) phase is proved separately from the later projection input.
    reject({reinterpret_cast<Q8_1Block *>(source.data() + 32), activation.size()}, nullptr);
    reject({reinterpret_cast<Q8_1Block *>(packed.native_interleaved.data()), activation.size()}, nullptr);
    std::vector<std::int32_t> words(activation.size() * sizeof(Q8_1Block) / sizeof(std::int32_t), 0);
    words[32] = 3;
    const auto work = ProjectionRowWork::fromPrefix(DeviceRowRange::deviceCounted(rows, &words[32]));
    const auto before_words = words;
    reject({reinterpret_cast<Q8_1Block *>(words.data()), activation.size()}, &work);
    EXPECT_EQ(words, before_words);
    std::vector<std::int32_t> output_words(rows * 5, 0);
    output_words[0] = 3;
    const auto beta_work = ProjectionRowWork::fromPrefix(DeviceRowRange::deviceCounted(rows, output_words.data()));
    const auto before_output_words = output_words;
    EXPECT_THROW(multiply_native_vnni_with_epilogue(packed, source.data(), reinterpret_cast<float *>(output_words.data()),
        rows, 5, 0.75f, -0.125f, activation, product, {}, &beta_work), std::invalid_argument);
    EXPECT_EQ(output_words, before_output_words);
    bytes(activation, before_activation);
    bytes(source, before_source);
    EXPECT_EQ(std::memcmp(packed.native_interleaved.data(), before_weights.data(), before_weights.size()), 0);
    for (const auto value : output) EXPECT_EQ(value, 123456.0f);
    for (const auto value : product) EXPECT_EQ(value, 123456.0f);
}

/** @test Two-term completion releases the primary bank only after every validation read. */
TEST_F(CPUNativeNEONExecution, RetainedTwoTermCompletesBeforePrimaryReuse)
{
    const Budget budget;
    constexpr int rows = 7, k = 1056;
    const auto packed = owner(native_vnni_formats::Q8_0, 5, k);
    std::vector<float> source(rows * k), output(rows * 5);
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = float(int(i % 251) - 125) * 0.013457f;
    std::vector<Q8_1Block> primary(std::size_t(rows) * packed.blocks_per_row), residual(primary.size());
    const auto range = DeviceRowRange::fullyActive(rows);
    omp_set_num_threads(3);
    for (int iteration = 0; iteration < 20; ++iteration)
    {
#pragma omp parallel
        {
            cpuQ8ResidualCorrectedProjection(packed, source.data(), output.data(), range, primary, residual);
            // This is a legitimate next workshare after completion, not an
            // input alias: a released bank can immediately publish new data.
#pragma omp for schedule(static)
            for (std::size_t block = 0; block < primary.size(); ++block) primary[block].d = 0x7c00;
#pragma omp barrier
        }
    }
    for (const auto &block : primary) EXPECT_EQ(block.d, 0x7c00);
}

/** @test Single-factor actual owners do not read an absent secondary vector. */
TEST_F(CPUNativeNEONExecution, ResidentSingleScaleOwnerMinIsExactZero)
{
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
    {
        const auto packed = owner(*source.metadata, 1, 32);
        const auto view = packed.readView();
        if (!view.is_asymmetric && !view.usesQ6KNativeDualScale() && !view.usesCompactMultiScale())
            EXPECT_EQ(std::bit_cast<std::uint32_t>(packed.blockMin(0, 0, 0)), 0u);
    }
}
/** @test Every source consumes fused native SwiGLU/Q8 publication through the real retained down-projection APIs. */
TEST_F(CPUNativeNEONExecution, AllSourcesFusedSwiGLUQ8DownRetainSerialRows)
{
    const Budget budget;
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
        for (const auto policy : {CPUProjectionNumericalPolicy::BackendNative, CPUProjectionNumericalPolicy::GPUAlignedExpert})
            for (const int k : {32, 64, 96, 1056})
            {
                const auto packed = owner(*source.metadata, 5, k, policy);
                for (const int rows : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 31, 33})
                {
                    SCOPED_TRACE(::testing::Message() << int(source.metadata->codebook_id) << ':' << int(policy) << ':' << k << ':' << rows);
                    std::vector<float> gate(std::size_t(rows) * k), up(gate.size()), activation(gate.size());
                    for (std::size_t i = 0; i < gate.size(); ++i)
                    { gate[i] = float(int(i % 131) - 65) * 0.25f; up[i] = float(int(i % 97) - 48) * 0.5f; }
                    std::vector<Q8_1Block> expected_q8(std::size_t(rows) * packed.blocks_per_row + 2);
                    std::memset(expected_q8.data(), 0x61, expected_q8.size() * sizeof(Q8_1Block));
                    auto actual_q8 = expected_q8;
                    for (int row = 0; row < rows; ++row)
                    {
                        const auto offset = std::size_t(row) * k;
                        if (policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                            primitives::compute_swiglu_gpu_aligned_expert_serial(gate.data() + offset, up.data() + offset, activation.data() + offset, k);
                        else
                            primitives::compute_swiglu_serial(gate.data() + offset, up.data() + offset, activation.data() + offset, k);
                        for (int block = 0; block < packed.blocks_per_row; ++block)
                            quantizeActivationBlock(activation.data() + offset + std::size_t(block) * 32,
                                expected_q8[std::size_t(row) * packed.blocks_per_row + block], 32, policy);
                    }
                    constexpr int stride = 8;
                    std::vector<float> expected(std::size_t(rows - 1) * stride + packed.N + 4, 123456.0f), actual = expected;
                    serial(packed, expected_q8, expected, rows, stride, rows);
                    omp_set_num_threads(3);
#pragma omp parallel
                    {
                        swiglu_quantize_activations_to_q8_1(gate.data(), up.data(), actual_q8.data(), rows, k, packed.blocks_per_row, policy);
                        gemm_native_vnni_preq_decode_equivalent_rows(packed, actual_q8.data(), actual.data(), {}, rows, stride);
                    }
                    bytes(actual_q8, expected_q8);
                    bytes(actual, expected);
                    ASSERT_NE(actual[0], 123456.0f);
                }
            }
}

} // namespace llaminar2::test
