/**
 * @file Test__IQ2SOriginalEquation.cpp
 * @brief Independent original-IQ2_S metadata and actual CPU Q8 arithmetic proofs.
 *
 * Literal source blocks cover every scale byte, signed zero, half subnormals,
 * normal rounding and derived factors beyond the finite half range. Payload
 * decoding is independent of the prepared decoder. Windowed preparation and
 * row views exercise absolute source coordinates; scalar and both SIMD widths
 * retain their existing two dots, contribution order and grouped partition tree.
 */
#include <gtest/gtest.h>
#include <omp.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>
#include "tensors/TensorClasses.h"
#include "tensors/FP16Utils.h"
#include "tensors/IQQuantTables.h"
#include "tensors/VnniPackContext.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemv.h"
#include "kernels/cpu/gemm/CPUNativeVNNIMultiScaleKernels.h"
#include "kernels/common/DeviceFP32NumericalContract.h"
#include "kernels/common/DeviceQ8ActivationNumericalContract.h"

namespace llaminar2::test
{
namespace
{
using namespace cpu::native_vnni;
constexpr int kFullN = 73, kStart = 3, kN = 67, kK = 288;
constexpr int kBlocks = kK / 32, kSuperblocks = (kK + 255) / 256;
constexpr std::array<uint16_t, 20> kBases = {
    0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400,
    0x1401, 0x9401, 0x3555, 0xb555, 0x3bff, 0xbbff, 0x3c00, 0xbc00,
    0x3c01, 0xbc01, 0x7bff, 0xfbff};

/**
 * @brief Fill all source grid/sign bitplanes without quantizing floating data.
 * @param block Original 82-byte source block to initialize.
 * @param ordinal Deterministic pattern index independent of retained model data.
 */
void fillSource(IQ2_SBlock &block, size_t ordinal)
{
    block.d = kBases[ordinal % kBases.size()];
    for (int i = 0; i < 64; ++i) block.qs[i] = uint8_t(ordinal * 37 + i * 53);
    for (int i = 0; i < 8; ++i)
    {
        block.qh[i] = uint8_t(ordinal * 41 + i * 71);
        block.scales[i] = uint8_t(ordinal * 8 + i);
    }
}

/**
 * @brief Recover one signed integer directly from original source grid/signs.
 * @param block Original GGUF block, never a prepared payload.
 * @param position Element index zero through 255.
 * @return Signed source grid value without a scale.
 */
int sourceInteger(const IQ2_SBlock &block, int position)
{
    const int sub = position / 32, quarter = (position % 32) / 8, lane = position % 8;
    const int grid_index = block.qs[sub * 4 + quarter] |
        (((unsigned(block.qh[sub]) >> (quarter * 2)) & 3u) << 8);
    const int magnitude = int((iq2s_grid[grid_index] >> (lane * 8)) & 255u);
    return (block.qs[32 + sub * 4 + quarter] & (1u << lane)) ? -magnitude : magnitude;
}

/**
 * @brief Independently evaluate the source dyadic factor without derived half conversion.
 * @param base Original half bits, including either signed zero.
 * @param nibble Original unsigned coefficient zero through fifteen.
 * @return Exact FP32 base times odd coefficient divided by eight.
 */
float sourceFactor(uint16_t base, int nibble)
{
    // This algebraic oracle uses an integer odd multiplier and power-of-two
    // scaling, rather than either production factor resolver's expression.
    return std::ldexp(fp16_to_fp32(base) * float(2 * nibble + 1), -3);
}

/**
 * @brief Original-block Q8 contribution with separate defining FP32 boundaries.
 * @param block Original source block.
 * @param sub Logical 32-value block within the source superblock.
 * @param a Unchanged prequantized activation block.
 * @return Ordered two-dot block contribution.
 */
float contribution(const IQ2_SBlock &block, int sub, const Q8_1Block &a)
{
    int dots[2]{};
    for (int lane = 0; lane < 32; ++lane)
        dots[lane / 16] += sourceInteger(block, sub * 32 + lane) * int(a.qs[lane]);
    const float low = sourceFactor(block.d, block.scales[sub] & 15);
    const float high = sourceFactor(block.d, block.scales[sub] >> 4);
    return device_fp32_contract::multiply(fp16_to_fp32(a.d), device_fp32_contract::add(
        device_fp32_contract::multiply(low, float(dots[0])),
        device_fp32_contract::multiply(high, float(dots[1]))));
}

/** @brief Bound tiny source preparation without changing the caller's worker policy. */
class IQ2SOriginalEquation : public ::testing::Test
{
protected:
    /** @brief Retain the caller's worker count and use two tiny independent chunks. */
    void SetUp() override { previous_ = omp_get_max_threads(); omp_set_num_threads(2); }
    /** @brief Restore worker policy on success and assertion unwinding. */
    void TearDown() override { omp_set_num_threads(previous_); }
private:
    int previous_ = 1;
};

/**
 * @brief Test one actual SIMD grouped kernel against independent source arithmetic.
 * @tparam Lanes Actual AVX2 eight-column or AVX512 sixteen-column implementation.
 * @tparam Rows Shared-weight row count one through four.
 * @param packed Native prepared matrix including N-tail padding.
 * @param source Immutable original source blocks in row-major superblocks.
 * @param activations Unchanged Q8 rows.
 * @param chunk Destination 64-column chunk.
 */
template<int Lanes, size_t Rows>
void checkSIMD(const CPUNativeVNNIPackedWeights &packed,
    const std::vector<IQ2_SBlock> &source,
    const std::array<std::array<Q8_1Block, kBlocks>, 4> &activations, int chunk)
{
    std::array<const Q8_1Block *, Rows> a{};
    std::array<std::array<float, 64>, Rows> actual{};
    std::array<float *, Rows> output{};
    for (size_t row = 0; row < Rows; ++row) { a[row] = activations[row].data(); output[row] = actual[row].data(); }
    for (bool ordered : {false, true})
    {
        for (auto &row : actual) row.fill(123.0f);
        if (ordered) multi_scale::dispatchOrdered<Lanes, Rows>(packed, a, output, chunk, kBlocks, 3, 3);
        else multi_scale::dispatch<Lanes, Rows>(packed, a, output, chunk, 0, kBlocks, false);
        for (int n = 0; n < 64 && chunk * 64 + n < kN; ++n)
        for (size_t row = 0; row < Rows; ++row)
        {
            float expected = 0.0f;
            for (int part = 0; part < (ordered ? 3 : 1); ++part)
            {
                float partial = 0.0f;
                for (int kb = ordered ? part * 3 : 0; kb < (ordered ? (part + 1) * 3 : kBlocks); ++kb)
                {
                    const auto &block = source[(kStart + chunk * 64 + n) * kSuperblocks + kb / 8];
                    partial = device_fp32_contract::add(partial, contribution(block, kb % 8, activations[row][kb]));
                }
                expected = part == 0 ? partial : device_fp32_contract::add(expected, partial);
            }
            EXPECT_EQ(std::bit_cast<uint32_t>(actual[row][n]), std::bit_cast<uint32_t>(expected))
                << "lanes=" << Lanes << " rows=" << Rows << " chunk=" << chunk << " column=" << n << " ordered=" << ordered;
        }
    }
}
}

/**
 * @brief Ordinary scale re-rounding changes an exact source projection with fixed Q8.
 *
 * Grid0 contains eight8s. Original d=1+2^-10 and nibble1 give3075/8192.
 * Low/high integer dots8 and1016 sum to1024 after the same separately rounded
 * FP32 dot factors. The original source result is384.375 exactly. Today's
 * derived-half scale769/2048 instead gives384.5 exactly. No Q8 error, variable
 * reduction, activation underflow, transcendental or cancellation is involved.
 */
TEST_F(IQ2SOriginalEquation, OrdinaryFiniteScaleChangesOriginalEquationWithFixedQ8)
{
    IQ2_SBlock original{};
    original.d = 0x3c01;
    original.scales[0] = 0x11;
    ASSERT_EQ(iq2s_grid[0], uint64_t{0x0808080808080808});
    std::vector<uint8_t> raw(sizeof(original));
    std::memcpy(raw.data(), &original, raw.size());
    IQ2_STensor source({1, 32}, raw);
    CPUNativeVNNIPackedWeights packed;
    ASSERT_TRUE(packWeightsCPUNativeVNNI(&source, packed));
    ASSERT_EQ(packed.codebook_id, 13);
    Q8_1Block activation{};
    activation.d = 0x3c00; activation.qs[0] = 1; activation.qs[31] = 127; activation.sum_qs = 128;
    const Q8_1Block before = activation;
    ASSERT_EQ(device_q8_activation_contract::scale(127.0f), 1.0f);
    for (int lane = 0; lane < 32; ++lane)
        ASSERT_EQ(device_q8_activation_contract::quantize(float(activation.qs[lane]), 1.0f), activation.qs[lane]);
    ASSERT_EQ(contribution(original, 0, activation), 384.375f);
    float actual = -123.0f;
    gemv_native_vnni_scalar(packed, &activation, &actual, 1, 1);
    EXPECT_EQ(std::bit_cast<uint32_t>(actual), std::bit_cast<uint32_t>(384.375f));
    EXPECT_EQ(std::memcmp(&activation, &before, sizeof(activation)), 0);
    EXPECT_EQ(std::memcmp(source.raw_data(), raw.data(), raw.size()), 0);
}

TEST_F(IQ2SOriginalEquation, EverySourceScaleByteAndFiniteBaseSurvivesLiterally)
{
    constexpr int N = 256 * kBases.size();
    std::vector<IQ2_SBlock> blocks(N);
    for (int n = 0; n < N; ++n)
    {
        fillSource(blocks[n], n);
        blocks[n].d = kBases[n / 256];
        blocks[n].scales[0] = uint8_t(n % 256);
    }
    std::vector<uint8_t> bytes(blocks.size() * sizeof(IQ2_SBlock));
    std::memcpy(bytes.data(), blocks.data(), bytes.size());
    IQ2_STensor tensor({N, 32}, bytes);
    std::vector<uint8_t> payload(size_t(N) * 9);
    std::vector<uint16_t> bases(N), coefficients(N);
    VnniPackContext context{.raw_bytes = nullptr, .N = N, .K = 32, .blocks_per_row = 1,
        .payload_bytes = 9, .payload_array = payload.data(), .scales_array = bases.data(),
        .mins_array = coefficients.data(), .emins_array = nullptr};
    bool observed_old_underflow = false, observed_old_overflow = false, observed_old_rounding = false;
    for (int n = 0; n < N; ++n)
    {
        tensor.packVnniBlock(context, n, n, 0);
        ASSERT_EQ(bases[n], blocks[n].d) << "source row=" << n;
        ASSERT_EQ(coefficients[n], uint16_t(blocks[n].scales[0])) << "source row=" << n;
        ASSERT_EQ(coefficients[n] & 0xff00u, 0);
        for (int j = 0; j < 9; ++j)
        {
            const uint8_t expected = j < 4 ? blocks[n].qs[j] : j == 4 ? blocks[n].qh[0] : blocks[n].qs[32 + j - 5];
            ASSERT_EQ(payload[n * 9 + j], expected);
        }
        for (int nibble : {n % 16, (n % 256) >> 4})
        {
            const float expected = sourceFactor(blocks[n].d, nibble);
            const float old_rounded = fp16_to_fp32(fp32_to_fp16(expected));
            observed_old_underflow |= expected != 0.0f && old_rounded == 0.0f;
            observed_old_overflow |= std::isfinite(expected) && !std::isfinite(old_rounded);
            observed_old_rounding |= std::isfinite(old_rounded) && old_rounded != expected;
        }
    }
    EXPECT_TRUE(observed_old_underflow);
    EXPECT_TRUE(observed_old_overflow);
    EXPECT_TRUE(observed_old_rounding);
}

TEST_F(IQ2SOriginalEquation, SourceViewAndWindowKeepAbsoluteCoordinates)
{
    std::vector<IQ2_SBlock> blocks(kFullN * kSuperblocks);
    for (size_t b = 0; b < blocks.size(); ++b) fillSource(blocks[b], b);
    std::vector<uint8_t> bytes(blocks.size() * sizeof(IQ2_SBlock));
    std::memcpy(bytes.data(), blocks.data(), bytes.size());
    auto parent = std::make_shared<IQ2_STensor>(std::vector<size_t>{kFullN, kK}, bytes);
    auto view_base = parent->create_view({kN, kK}, kStart * kK);
    auto *view = dynamic_cast<IQ2_STensor *>(view_base.get());
    ASSERT_NE(view, nullptr);
    constexpr int origin = 6, count = 3;
    std::vector<uint8_t> payload(count * kN * 9, 0xa5);
    std::vector<uint16_t> bases(count * kN, 0xa5a5), coefficients(bases.size(), 0x5a5a);
    VnniPackContext context{.raw_bytes = nullptr, .N = kN, .K = kK, .blocks_per_row = kBlocks,
        .payload_bytes = 9, .payload_array = payload.data(), .scales_array = bases.data(),
        .mins_array = coefficients.data(), .emins_array = nullptr,
        .destination_block_origin = origin, .destination_block_count = count};
    for (int kb = origin; kb < origin + count; ++kb)
    for (int n = 0; n < kN; ++n)
    {
        view->packVnniBlock(context, n, n, kb);
        const auto &block = blocks[(n + kStart) * kSuperblocks + kb / 8];
        const size_t index = size_t(kb - origin) * kN + n;
        ASSERT_EQ(bases[index], block.d);
        ASSERT_EQ(coefficients[index], uint16_t(block.scales[kb % 8]));
        for (int j = 0; j < 9; ++j)
        {
            const int sub = kb % 8;
            const uint8_t expected = j < 4 ? block.qs[sub * 4 + j] : j == 4 ? block.qh[sub] : block.qs[32 + sub * 4 + j - 5];
            ASSERT_EQ(payload[index * 9 + j], expected);
        }
    }
}

TEST_F(IQ2SOriginalEquation, ScalarAndAllSIMDRowsUseOriginalFactors)
{
    std::vector<IQ2_SBlock> source(kFullN * kSuperblocks);
    for (size_t b = 0; b < source.size(); ++b) fillSource(source[b], b);
    std::vector<uint8_t> bytes(source.size() * sizeof(IQ2_SBlock));
    std::memcpy(bytes.data(), source.data(), bytes.size());
    IQ2_STensor tensor({kFullN, kK}, bytes);
    CPUNativeVNNIPackedWeights packed;
    ASSERT_TRUE(packWeightsCPUNativeVNNI(&tensor, packed, kStart, kStart + kN));
    ASSERT_TRUE(packed.usesCompactMultiScale());
    ASSERT_EQ(packed.codebook_id, 13);
    EXPECT_EQ(packed.interleaved_block_stride, 1536);
    EXPECT_EQ(packed.native_interleaved.size(), size_t(2 * kBlocks * 1536));
    std::array<std::array<Q8_1Block, kBlocks>, 4> activations{};
    for (int row = 0; row < 4; ++row)
    for (int kb = 0; kb < kBlocks; ++kb)
    {
        auto &a = activations[row][kb];
        a.d = 0x0400;
        for (int lane = 0; lane < 32; ++lane)
        { a.qs[lane] = int8_t((row * 31 + kb * 43 + lane * 53) % 255 - 127); a.sum_qs += a.qs[lane]; }
        a.sum_qs += 127 - a.qs[31];
        a.qs[31] = 127;
        const float published_scale = fp16_to_fp32(a.d);
        ASSERT_EQ(device_q8_activation_contract::scale(127.0f * published_scale), published_scale);
        for (int lane = 0; lane < 32; ++lane)
            ASSERT_EQ(device_q8_activation_contract::quantize(float(a.qs[lane]) * published_scale, published_scale), a.qs[lane]);
    }
    const auto unchanged_activations = activations;
    std::array<float, kN> scalar{};
    for (int row = 0; row < 4; ++row)
    {
        gemv_native_vnni_scalar(packed, activations[row].data(), scalar.data(), kN, kBlocks);
        for (int n = 0; n < kN; ++n)
        {
            float expected = 0.0f;
            for (int kb = 0; kb < kBlocks; ++kb)
            {
                const auto &block = source[(n + kStart) * kSuperblocks + kb / 8];
                const float low = sourceFactor(block.d, block.scales[kb % 8] & 15);
                const float high = sourceFactor(block.d, block.scales[kb % 8] >> 4);
                EXPECT_EQ(std::bit_cast<uint32_t>(packed.blockScale(n / 64, kb, n % 64)), std::bit_cast<uint32_t>(low));
                EXPECT_EQ(std::bit_cast<uint32_t>(packed.blockMin(n / 64, kb, n % 64)), std::bit_cast<uint32_t>(high));
                expected = device_fp32_contract::add(expected, contribution(block, kb % 8, activations[row][kb]));
            }
            EXPECT_EQ(std::bit_cast<uint32_t>(scalar[n]), std::bit_cast<uint32_t>(expected));
        }
    }
    for (int chunk = 0; chunk < 2; ++chunk)
    {
        checkSIMD<8, 1>(packed, source, activations, chunk);
        checkSIMD<8, 2>(packed, source, activations, chunk);
        checkSIMD<8, 3>(packed, source, activations, chunk);
        checkSIMD<8, 4>(packed, source, activations, chunk);
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        checkSIMD<16, 1>(packed, source, activations, chunk);
        checkSIMD<16, 2>(packed, source, activations, chunk);
        checkSIMD<16, 3>(packed, source, activations, chunk);
        checkSIMD<16, 4>(packed, source, activations, chunk);
#endif
    }
    EXPECT_EQ(std::memcmp(activations.data(), unchanged_activations.data(), sizeof(activations)), 0);
}
} // namespace llaminar2::test
