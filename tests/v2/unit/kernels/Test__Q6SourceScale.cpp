/**
 * @file Test__Q6SourceScale.cpp
 * @brief Original-source Q6_K factor, layout, and Q8 arithmetic regressions.
 *
 * Independent source decoding covers signed coefficients, half subnormals,
 * products that cannot be stored as half, TP row slicing and K/N tails. Scalar,
 * AVX2 and AVX512 consumers must reproduce the same ordered Q8 dot arithmetic.
 * Tiny owned tensors keep this gate device-free and independent of real models.
 */
#include <gtest/gtest.h>
#include <omp.h>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <vector>
#include "tensors/TensorClasses.h"
#include "tensors/FP16Utils.h"
#include "tensors/VnniPackContext.h"
#include "tensors/NativeVnniQ6Scale.h"
#include "kernels/cpu/gemm/CPUNativeVNNIQ6Kernels.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemv.h"
#include "kernels/common/DeviceFP32NumericalContract.h"

namespace llaminar2::test
{
namespace
{
using namespace cpu::native_vnni;
constexpr int kFullN = 73;
constexpr int kStart = 3;
constexpr int kN = 67;
constexpr int kK = 288;
constexpr int kBlocks = kK / 32;
constexpr int kSuperblocks = (kK + 255) / 256;
constexpr std::array<uint16_t, 16> kBases = {
    0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400,
    0x1401, 0x9401, 0x3555, 0xb555, 0x3c01, 0xbc01, 0x7bff, 0xfbff};
constexpr std::array<int8_t, 8> kCoefficients = {-128, 127, -1, 1, 0, 3, -7, 63};

/** @brief Adversarial original blocks with every bitplane and scale sign active. */
std::vector<Q6_KBlock> sourceBlocks()
{
    std::vector<Q6_KBlock> blocks(kFullN * kSuperblocks);
    for (size_t b = 0; b < blocks.size(); ++b)
    {
        auto &block = blocks[b];
        block.d = kBases[b % kBases.size()];
        for (int i = 0; i < 16; ++i)
            block.scales[i] = kCoefficients[(b + i) % kCoefficients.size()];
        for (int i = 0; i < 128; ++i)
            block.ql[i] = static_cast<uint8_t>(b * 37 + i * 53);
        for (int i = 0; i < 64; ++i)
            block.qh[i] = static_cast<uint8_t>(b * 41 + i * 71);
    }
    return blocks;
}

/** @brief Create owned source storage without quantizing any existing weight. */
Q6_KTensor sourceTensor(const std::vector<Q6_KBlock> &blocks)
{
    std::vector<uint8_t> bytes(blocks.size() * sizeof(Q6_KBlock));
    std::memcpy(bytes.data(), blocks.data(), bytes.size());
    return Q6_KTensor({kFullN, kK}, bytes);
}

/**
 * @brief Independently decode the signed six-bit integer at an original offset.
 * @param block Original GGUF superblock.
 * @param position Element offset zero through 255.
 * @return Centered source integer, with no scale applied.
 */
int sourceInteger(const Q6_KBlock &block, int position)
{
    const int half = position / 128;
    const int group = (position % 128) / 32;
    const int lane = position % 32;
    const uint8_t low = block.ql[half * 64 + (group % 2) * 32 + lane];
    const uint8_t high = block.qh[half * 32 + lane];
    return ((low >> (group / 2 * 4)) & 15) +
           (((high >> (group * 2)) & 3) * 16) - 32;
}

/**
 * @brief Original-source block oracle with the established separate FP32 edges.
 * @param source Original weight superblock.
 * @param subblock Logical 32-element block in that superblock.
 * @param activation Already quantized Q8 block, unchanged by this fix.
 * @return Source-exact contribution with the production arithmetic order.
 */
float contribution(const Q6_KBlock &source, int subblock, const Q8_1Block &activation)
{
    int low = 0, high = 0;
    for (int lane = 0; lane < 16; ++lane)
    {
        low += sourceInteger(source, subblock * 32 + lane) * activation.qs[lane];
        high += sourceInteger(source, subblock * 32 + lane + 16) * activation.qs[lane + 16];
    }
    const float base = fp16_to_fp32(source.d);
    const float low_scale = device_fp32_contract::multiply(base, float(source.scales[subblock * 2]));
    const float high_scale = device_fp32_contract::multiply(base, float(source.scales[subblock * 2 + 1]));
    return device_fp32_contract::multiply(fp16_to_fp32(activation.d),
        device_fp32_contract::add(device_fp32_contract::multiply(low_scale, float(low)),
                                 device_fp32_contract::multiply(high_scale, float(high))));
}

/** @brief Scope the small packing fixture's workers without leaking test policy. */
class Q6SourceScale : public ::testing::Test
{
protected:
    /** @brief Keep tiny tensor preparation bounded to its two independent chunks. */
    void SetUp() override { previous_ = omp_get_max_threads(); omp_set_num_threads(2); }
    /** @brief Restore the caller's worker policy even when an assertion fails. */
    void TearDown() override { omp_set_num_threads(previous_); }
private:
    int previous_ = 1;
};
}

TEST_F(Q6SourceScale, EverySignedCoefficientPairRoundTrips)
{
    for (int low = -128; low <= 127; ++low)
    for (int high = -128; high <= 127; ++high)
    {
        const uint16_t word = native_vnni_q6_scale::pack(int8_t(low), int8_t(high));
        ASSERT_EQ(native_vnni_q6_scale::low(word), low);
        ASSERT_EQ(native_vnni_q6_scale::high(word), high);
    }
}

TEST_F(Q6SourceScale, OriginalMetadataAndPayloadSurviveWindowedPreparation)
{
    const auto blocks = sourceBlocks();
    auto tensor = sourceTensor(blocks);
    // A bounded window crossing a superblock boundary must retain absolute
    // source indexing independently of destination row and K origins.
    constexpr int origin = 6, count = 3;
    std::vector<uint8_t> payload(count * kN * 24, 0xa5);
    std::vector<uint16_t> bases(count * kN, 0xa5a5), coefficients(bases.size(), 0x5a5a);
    VnniPackContext context{.raw_bytes = nullptr, .N = kN, .K = kK,
        .blocks_per_row = kBlocks, .payload_bytes = 24,
        .payload_array = payload.data(), .scales_array = bases.data(),
        .mins_array = coefficients.data(), .emins_array = nullptr,
        .destination_block_origin = origin, .destination_block_count = count};
    for (int kb = origin; kb < origin + count; ++kb)
    for (int n = 0; n < kN; ++n)
    {
        tensor.packVnniBlock(context, n + kStart, n, kb);
        const auto &source = blocks[(n + kStart) * kSuperblocks + kb / 8];
        const size_t index = (kb - origin) * kN + n;
        ASSERT_EQ(bases[index], source.d);
        const uint16_t expected = uint8_t(source.scales[(kb % 8) * 2]) |
            (uint16_t(uint8_t(source.scales[(kb % 8) * 2 + 1])) << 8);
        ASSERT_EQ(coefficients[index], expected);
        for (int lane = 0; lane < 32; ++lane)
        {
            const uint8_t *bytes = payload.data() + index * 24;
            const int raw = ((bytes[lane % 16] >> (lane / 16 * 4)) & 15) |
                (((bytes[16 + lane / 4] >> (lane % 4 * 2)) & 3) << 4);
            ASSERT_EQ(raw - 32, sourceInteger(source, (kb % 8) * 32 + lane));
        }
    }
}

TEST_F(Q6SourceScale, ScalarAndSIMDUseOriginalFactorsWithQ8Activations)
{
    const auto blocks = sourceBlocks();
    auto tensor = sourceTensor(blocks);
    CPUNativeVNNIPackedWeights packed;
    ASSERT_TRUE(packWeightsCPUNativeVNNI(&tensor, packed, kStart, kStart + kN));
    ASSERT_TRUE(packed.usesQ6KNativeDualScale());
    EXPECT_EQ(packed.interleaved_block_stride, 1792);
    EXPECT_EQ(packed.native_interleaved.size(), size_t(2 * kBlocks * 1792));
    std::array<std::array<Q8_1Block, kBlocks>, 4> activations{};
    for (int row = 0; row < 4; ++row)
    for (int kb = 0; kb < kBlocks; ++kb)
    {
        auto &a = activations[row][kb];
        a.d = 0x2400;
        for (int lane = 0; lane < 32; ++lane)
        {
            a.qs[lane] = int8_t((row * 31 + kb * 43 + lane * 53) % 256 - 128);
            a.sum_qs += a.qs[lane];
        }
    }
    // Execute the actual scalar consumer, not only its metadata accessors.
    std::array<std::array<float, kN>, 4> scalar_actual{};
    for (int row = 0; row < 4; ++row)
        gemv_native_vnni_scalar(packed, activations[row].data(),
            scalar_actual[row].data(), kN, kBlocks);
    for (int chunk = 0; chunk < 2; ++chunk)
    {
        std::array<std::array<float, 64>, 4> expected{}, actual{};
        for (int n = 0; n < 64 && chunk * 64 + n < kN; ++n)
        for (int kb = 0; kb < kBlocks; ++kb)
        {
            const auto &source = blocks[(kStart + chunk * 64 + n) * kSuperblocks + kb / 8];
            const float base = fp16_to_fp32(source.d);
            EXPECT_EQ(std::bit_cast<uint32_t>(packed.blockScale(chunk, kb, n)),
                std::bit_cast<uint32_t>(device_fp32_contract::multiply(base, float(source.scales[kb % 8 * 2]))));
            EXPECT_EQ(std::bit_cast<uint32_t>(packed.blockMin(chunk, kb, n)),
                std::bit_cast<uint32_t>(device_fp32_contract::multiply(base, float(source.scales[kb % 8 * 2 + 1]))));
            for (int row = 0; row < 4; ++row)
                expected[row][n] = device_fp32_contract::add(expected[row][n],
                    contribution(source, kb % 8, activations[row][kb]));
        }
        for (int row = 0; row < 4; ++row)
            for (int n = 0; n < 64 && chunk * 64 + n < kN; ++n)
                EXPECT_EQ(std::bit_cast<uint32_t>(scalar_actual[row][chunk * 64 + n]),
                    std::bit_cast<uint32_t>(expected[row][n]))
                    << "scalar chunk=" << chunk << " row=" << row << " column=" << n;
        const auto compare = [&](int rows) {
            for (int row = 0; row < rows; ++row)
                EXPECT_EQ(std::memcmp(actual[row].data(), expected[row].data(), 64 * sizeof(float)), 0)
                    << "chunk=" << chunk << " row=" << row;
        };
        for (int row = 0; row < 4; ++row)
            gemvQ6KNativeAVX2Chunk(packed, activations[row].data(), actual[row].data(), chunk, 0, kBlocks);
        compare(4);
        actual = {};
        gemmQ6KNativeTwoRowsAVX2Chunk(packed, activations[0].data(), activations[1].data(),
            actual[0].data(), actual[1].data(), chunk, 0, kBlocks, false);
        compare(2);
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        actual = {};
        for (int row = 0; row < 4; ++row)
            gemvQ6KNativeAVX512Chunk(packed, activations[row].data(), actual[row].data(), chunk, 0, kBlocks);
        compare(4);
        actual = {};
        gemmQ6KNativeTwoRowsAVX512Chunk(packed, activations[0].data(), activations[1].data(),
            actual[0].data(), actual[1].data(), chunk, 0, kBlocks, false);
        compare(2);
        actual = {};
        gemmQ6KNativeThreeRowsAVX512Chunk(packed, activations[0].data(), activations[1].data(), activations[2].data(),
            actual[0].data(), actual[1].data(), actual[2].data(), chunk, 0, kBlocks, false);
        compare(3);
        actual = {};
        gemmQ6KNativeFourRowsAVX512Chunk(packed, activations[0].data(), activations[1].data(), activations[2].data(), activations[3].data(),
            actual[0].data(), actual[1].data(), actual[2].data(), actual[3].data(), chunk, 0, kBlocks, false);
        compare(4);
#endif
    }
}
} // namespace llaminar2::test
