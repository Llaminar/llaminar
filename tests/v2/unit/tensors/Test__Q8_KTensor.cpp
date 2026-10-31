/**
 * @file Test__Q8_KTensor.cpp
 * @brief Scalar/SIMD source-value checks for292-byte FP32-scaled Q8_K blocks.
 * Legacy unit-scale fixtures explicitly initialize d=1. Focused regressions
 * below independently encode raw GGUF bytes and exercise nonunit source d.
 *
 * The custom entry owns one lexical native GPU lifetime without probing a
 * device. Native workers/caches retire before process exit and MPI finalization.
 */

#include "backends/GPUDeviceProcessLifetime.h"
#include "tensors/Tensors.h"
#include "tensors/SIMDHelpers.h"
#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <random>
#include <cmath>

using namespace llaminar2;

// Tolerance: Very strict since this is just int8→float conversion (exact)
constexpr float TOLERANCE = 1e-6f;

class Q8_KSIMDTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Nothing to set up - tests create blocks on the fly
    }

    void TearDown() override {}

    /**
     * @brief Create a Q8_KBlock with specified int8 values
     * @param values 256 int8_t values for qs[] array
     * @return Initialized Q8_KBlock
     */
    Q8_KBlock createBlock(const std::vector<int8_t> &values)
    {
        Q8_KBlock block;
        std::memset(&block, 0, sizeof(Q8_KBlock));
        block.d = 1.0f;

        // Copy values into qs[] (must be exactly 256 elements)
        assert(values.size() == 256 && "Q8_K block requires 256 elements");
        std::memcpy(block.qs, values.data(), 256);

        // bsums[] not used in decode - leave as zeros
        return block;
    }

    /**
     * @brief Compare two float arrays with tolerance
     * @param expected Expected float values
     * @param actual Actual float values
     * @param count Number of elements to compare
     * @param tolerance Maximum allowed absolute difference
     * @return true if all elements match within tolerance
     */
    bool compareArrays(const float *expected, const float *actual, size_t count, float tolerance)
    {
        for (size_t i = 0; i < count; ++i)
        {
            float diff = std::abs(expected[i] - actual[i]);
            if (diff > tolerance)
            {
                std::cerr << "Mismatch at index " << i << ": expected " << expected[i]
                          << ", got " << actual[i] << " (diff=" << diff << ")" << std::endl;
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Create a random Q8_KTensor for testing
     */
    std::unique_ptr<Q8_KTensor> createRandomTensor(size_t rows, size_t cols)
    {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(-128, 127);

        std::vector<size_t> shape = {rows, cols};
        size_t blocks_per_row = (cols + 255) / 256;
        size_t total_blocks = rows * blocks_per_row;
        std::vector<uint8_t> raw_data(total_blocks * sizeof(Q8_KBlock));

        for (size_t b = 0; b < total_blocks; ++b)
        {
            Q8_KBlock *block = reinterpret_cast<Q8_KBlock *>(raw_data.data() + b * sizeof(Q8_KBlock));
            block->d = 1.0f;
            for (size_t i = 0; i < 256; ++i)
            {
                block->qs[i] = static_cast<int8_t>(dist(rng));
            }
            std::memset(block->bsums, 0, sizeof(block->bsums));
        }

        return std::make_unique<Q8_KTensor>(shape, raw_data);
    }
};

// ===== SIMD Equivalency Tests =====

#ifdef __AVX2__
TEST_F(Q8_KSIMDTest, ScalarVsAVX2Equivalency)
{
    // Create test block with varying int8 values
    std::vector<int8_t> values(256);
    for (size_t i = 0; i < 256; ++i)
    {
        values[i] = static_cast<int8_t>((i % 256) - 128); // Range: -128 to 127
    }
    Q8_KBlock block = createBlock(values);

    // Decode with scalar and AVX2
    std::vector<float> scalar_output(256);
    std::vector<float> avx2_output(256);

    Q8_KTensor::decodeBlockScalar(block, scalar_output.data());
    Q8_KTensor::decodeBlockAVX2(block, avx2_output.data());

    // Compare results (should be identical - exact conversion)
    EXPECT_TRUE(compareArrays(scalar_output.data(), avx2_output.data(), 256, TOLERANCE))
        << "Scalar vs AVX2 mismatch";
}
#endif // __AVX2__

#ifdef __AVX512F__
TEST_F(Q8_KSIMDTest, ScalarVsAVX512Equivalency)
{
    // Create test block with varying int8 values
    std::vector<int8_t> values(256);
    for (size_t i = 0; i < 256; ++i)
    {
        values[i] = static_cast<int8_t>((i % 256) - 128); // Range: -128 to 127
    }
    Q8_KBlock block = createBlock(values);

    // Decode with scalar and AVX512
    std::vector<float> scalar_output(256);
    std::vector<float> avx512_output(256);

    Q8_KTensor::decodeBlockScalar(block, scalar_output.data());
    Q8_KTensor::decodeBlockAVX512(block, avx512_output.data());

    // Compare results (should be identical - exact conversion)
    EXPECT_TRUE(compareArrays(scalar_output.data(), avx512_output.data(), 256, TOLERANCE))
        << "Scalar vs AVX512 mismatch";
}
#endif // __AVX512F__

#if defined(__AVX2__) && defined(__AVX512F__)
TEST_F(Q8_KSIMDTest, AVX2VsAVX512Equivalency)
{
    // Create test block with varying int8 values
    std::vector<int8_t> values(256);
    for (size_t i = 0; i < 256; ++i)
    {
        values[i] = static_cast<int8_t>((i % 256) - 128); // Range: -128 to 127
    }
    Q8_KBlock block = createBlock(values);

    // Decode with AVX2 and AVX512
    std::vector<float> avx2_output(256);
    std::vector<float> avx512_output(256);

    Q8_KTensor::decodeBlockAVX2(block, avx2_output.data());
    Q8_KTensor::decodeBlockAVX512(block, avx512_output.data());

    // Compare results (should be identical - exact conversion)
    EXPECT_TRUE(compareArrays(avx2_output.data(), avx512_output.data(), 256, TOLERANCE))
        << "AVX2 vs AVX512 mismatch";
}
#endif // __AVX2__ && __AVX512F__

// ===== Edge Case Tests =====

TEST_F(Q8_KSIMDTest, EdgeCase_AllZeros)
{
    // All int8 values = 0 → all float outputs = 0.0f
    std::vector<int8_t> values(256, 0);
    Q8_KBlock block = createBlock(values);

    std::vector<float> output(256);
    Q8_KTensor::decodeBlock(block, output.data());

    // All outputs should be exactly 0.0f
    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_FLOAT_EQ(output[i], 0.0f) << "Index " << i << " should be 0.0f";
    }
}

TEST_F(Q8_KSIMDTest, EdgeCase_AllPositiveMax)
{
    // All int8 values = 127 (max positive) → all float outputs = 127.0f
    std::vector<int8_t> values(256, 127);
    Q8_KBlock block = createBlock(values);

    std::vector<float> output(256);
    Q8_KTensor::decodeBlock(block, output.data());

    // All outputs should be exactly 127.0f
    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_FLOAT_EQ(output[i], 127.0f) << "Index " << i << " should be 127.0f";
    }
}

TEST_F(Q8_KSIMDTest, EdgeCase_AllNegativeMin)
{
    // All int8 values = -128 (min negative) → all float outputs = -128.0f
    std::vector<int8_t> values(256, -128);
    Q8_KBlock block = createBlock(values);

    std::vector<float> output(256);
    Q8_KTensor::decodeBlock(block, output.data());

    // All outputs should be exactly -128.0f
    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_FLOAT_EQ(output[i], -128.0f) << "Index " << i << " should be -128.0f";
    }
}

TEST_F(Q8_KSIMDTest, EdgeCase_AlternatingSign)
{
    // Alternating positive/negative pattern: -128, 127, -128, 127, ...
    std::vector<int8_t> values(256);
    for (size_t i = 0; i < 256; ++i)
    {
        values[i] = (i % 2 == 0) ? -128 : 127;
    }
    Q8_KBlock block = createBlock(values);

    std::vector<float> output(256);
    Q8_KTensor::decodeBlock(block, output.data());

    // Check alternating pattern
    for (size_t i = 0; i < 256; ++i)
    {
        float expected = (i % 2 == 0) ? -128.0f : 127.0f;
        EXPECT_FLOAT_EQ(output[i], expected) << "Index " << i << " should be " << expected;
    }
}

TEST_F(Q8_KSIMDTest, EdgeCase_ChunkBoundaries)
{
    // Test AVX2 (8-element) and AVX512 (16-element) boundaries
    // Pattern: First 16 elements = -128, next 16 = 127, repeat
    std::vector<int8_t> values(256);
    for (size_t i = 0; i < 256; ++i)
    {
        values[i] = ((i / 16) % 2 == 0) ? -128 : 127;
    }
    Q8_KBlock block = createBlock(values);

    std::vector<float> output(256);
    Q8_KTensor::decodeBlock(block, output.data());

    // Verify pattern
    for (size_t i = 0; i < 256; ++i)
    {
        float expected = ((i / 16) % 2 == 0) ? -128.0f : 127.0f;
        EXPECT_FLOAT_EQ(output[i], expected) << "Index " << i << " should be " << expected;
    }
}

TEST_F(Q8_KSIMDTest, EdgeCase_RandomValues)
{
    // Random int8 values across full range [-128, 127]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-128, 127);

    std::vector<int8_t> values(256);
    for (size_t i = 0; i < 256; ++i)
    {
        values[i] = static_cast<int8_t>(dist(gen));
    }
    Q8_KBlock block = createBlock(values);

    std::vector<float> output(256);
    Q8_KTensor::decodeBlock(block, output.data());

    // Verify each output matches input (exact int→float conversion)
    for (size_t i = 0; i < 256; ++i)
    {
        float expected = static_cast<float>(values[i]);
        EXPECT_FLOAT_EQ(output[i], expected) << "Index " << i << " should be " << expected;
    }
}

// =============================================================================
// to<T>() Template Method Tests
// =============================================================================

TEST_F(Q8_KSIMDTest, ToFloat_TemplateMethod)
{
    // Create a random tensor (1x256 = 1 block)
    auto tensor = createRandomTensor(1, 256);
    const size_t total = 256;

    // Test template method
    std::vector<float> template_output(total);
    tensor->to<float>(template_output.data());

    // Test legacy method
    std::vector<float> legacy_output(total);
    tensor->to_fp32(legacy_output.data());

    // Should produce identical results
    for (size_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(template_output[i], legacy_output[i])
            << "Mismatch at index " << i;
    }
}

TEST_F(Q8_KSIMDTest, ToBF16_TemplateMethod)
{
    // Create random tensor
    auto tensor = createRandomTensor(1, 256);
    const size_t total = 256;

    // Test template method
    std::vector<uint16_t> template_output(total);
    tensor->to<uint16_t>(template_output.data(), TensorType::BF16);

    // Test legacy method
    std::vector<uint16_t> legacy_output(total);
    tensor->to_bf16(legacy_output.data());

    // Should produce identical results
    for (size_t i = 0; i < total; ++i)
    {
        EXPECT_EQ(template_output[i], legacy_output[i])
            << "Mismatch at index " << i;
    }
}

TEST_F(Q8_KSIMDTest, ToFP16_TemplateMethod)
{
    // Create random tensor
    auto tensor = createRandomTensor(1, 256);
    const size_t total = 256;

    // Test template method
    std::vector<uint16_t> template_output(total);
    tensor->to<uint16_t>(template_output.data(), TensorType::FP16);

    // Test legacy method
    std::vector<uint16_t> legacy_output(total);
    tensor->to_fp16(legacy_output.data());

    // Should produce identical results
    for (size_t i = 0; i < total; ++i)
    {
        EXPECT_EQ(template_output[i], legacy_output[i])
            << "Mismatch at index " << i;
    }
}

TEST_F(Q8_KSIMDTest, ToINT8_TemplateMethod)
{
    // Create tensor
    auto tensor = createRandomTensor(1, 256);
    const size_t total = 256;

    // Convert to INT8
    std::vector<int8_t> int8_output(total);
    tensor->to<int8_t>(int8_output.data());

    // Verify INT8 range
    for (size_t i = 0; i < total; ++i)
    {
        EXPECT_GE(int8_output[i], -127);
        EXPECT_LE(int8_output[i], 127);
    }
}

TEST_F(Q8_KSIMDTest, ToINT32_TemplateMethod)
{
    // Create tensor
    auto tensor = createRandomTensor(1, 256);
    const size_t total = 256;

    // Convert to INT32
    std::vector<int32_t> int32_output(total);
    tensor->to<int32_t>(int32_output.data());

    // Just verify no crash
    EXPECT_NE(int32_output.data(), nullptr);
}

TEST_F(Q8_KSIMDTest, RoundTrip)
{
    // Create tensor
    auto tensor = createRandomTensor(1, 256);
    const size_t total = 256;

    // Round trip: Q8_K -> FP32 -> BF16 -> FP32
    std::vector<float> fp32_1(total);
    tensor->to<float>(fp32_1.data());

    // Create BF16 tensor from FP32 data
    auto fp32_temp = std::make_shared<FP32Tensor>(std::vector<size_t>{1, 256});
    std::memcpy(fp32_temp->mutable_data(), fp32_1.data(), total * sizeof(float));

    std::vector<uint16_t> bf16_data(total);
    fp32_temp->to<uint16_t>(bf16_data.data(), TensorType::BF16);

    auto bf16_tensor = std::make_shared<BF16Tensor>(std::vector<size_t>{1, 256}, bf16_data);

    // Convert back to FP32
    std::vector<float> fp32_2(total);
    bf16_tensor->to<float>(fp32_2.data());

    // Verify accuracy (BF16 precision ~3 decimal places)
    size_t mismatches = 0;
    for (size_t i = 0; i < total; ++i)
    {
        float diff = std::abs(fp32_1[i] - fp32_2[i]);
        float rel_error = (fp32_1[i] != 0.0f) ? diff / std::abs(fp32_1[i]) : diff;
        if (rel_error > 0.05f)
        { // 5% tolerance for BF16
            ++mismatches;
        }
    }
    EXPECT_LT(mismatches, 13) << "Too many mismatches in round-trip conversion";
}

// ===== Main =====
int main(int argc, char **argv)
{
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

/**
 * @brief Independent GGUF byte fixtures for original Q8_K FP32-scale values.
 *
 * These tests intentionally encode292-byte source blocks without using the
 * production block struct. Existing FP16 prepared metadata is tested as an
 * explicit precision limitation, never used as the original-value oracle.
 */
#include "loaders/ModelLoader.h"
#include "tensors/NativeVnniFormatInfo.h"
#include <array>
#include <bit>
#include <limits>
#include <sys/mman.h>
#include <unistd.h>

namespace
{
/** @brief Serialize source blocks independently; @return Original GGUF bytes. */
std::vector<uint8_t> q8kSourceBytes()
{
    constexpr size_t source_bytes = 292;
    const std::array<float, 8> scales{
        0.1f, -2.25f, 0x1p-25f, std::bit_cast<float>(uint32_t{0x100}),
        -0.1f, 1.5f, -0x1p-25f, -std::bit_cast<float>(uint32_t{0x100})};
    std::vector<uint8_t> bytes(source_bytes * scales.size());
    for (size_t block = 0; block < scales.size(); ++block)
    {
        std::memcpy(bytes.data() + block * source_bytes, &scales[block], 4);
        for (size_t partial = 0; partial < 16; ++partial)
        {
            int16_t sum = 0;
            for (size_t lane = 0; lane < 16; ++lane)
            {
                const size_t i = partial * 16 + lane;
                const auto q = static_cast<int8_t>(int((block * 23 + i) % 256) - 128);
                std::memcpy(bytes.data() + block * source_bytes + 4 + i, &q, 1);
                sum = static_cast<int16_t>(sum + q);
            }
            std::memcpy(bytes.data() + block * source_bytes + 260 + partial * 2, &sum, 2);
        }
    }
    return bytes;
}

/** @brief Read represented values from independent serialized coordinates. */
std::vector<float> q8kSourceOracle(const std::vector<uint8_t>& bytes)
{
    std::vector<float> result(bytes.size() / 292 * 256);
    for (size_t block = 0; block < bytes.size() / 292; ++block)
    {
        float scale;
        std::memcpy(&scale, bytes.data() + block * 292, 4);
        for (size_t i = 0; i < 256; ++i)
        {
            int8_t q;
            std::memcpy(&q, bytes.data() + block * 292 + 4 + i, 1);
            result[block * 256 + i] = scale * static_cast<float>(q);
        }
    }
    return result;
}
} // namespace

TEST(Q8KSourceScale, Original292ByteDescriptorAndAllDecodeISAs)
{
    static_assert(std::endian::native == std::endian::little);
    EXPECT_EQ(sizeof(Q8_KBlock), 292u);
    EXPECT_EQ(offsetof(Q8_KBlock, d), 0u);
    EXPECT_EQ(offsetof(Q8_KBlock, qs), 4u);
    EXPECT_EQ(offsetof(Q8_KBlock, bsums), 260u);
    GGUFTensorInfo info{};
    info.type = GGUFTensorType::Q8_K;
    EXPECT_EQ(info.getTypeSize(), 292u);
    const auto bytes = q8kSourceBytes();
    const auto expected = q8kSourceOracle(bytes);
    const std::vector<uint8_t> legacy_288_bytes(8 * 288);
    EXPECT_THROW((Q8_KTensor(std::vector<size_t>{4, 512}, legacy_288_bytes)),
        std::invalid_argument);
    auto tensor = std::make_shared<Q8_KTensor>(std::vector<size_t>{4, 512}, bytes);
    std::vector<float> actual(expected.size());
    tensor->to_fp32(actual.data());
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * 4), 0);
    for (size_t b = 0; b < 8; ++b)
    {
        Q8_KBlock block{};
        std::memcpy(&block, bytes.data() + b * 292, 292);
        std::array<float, 256> decoded{};
        Q8_KTensor::decodeBlockScalar(block, decoded.data());
        EXPECT_EQ(std::memcmp(decoded.data(), expected.data() + b * 256, 256 * 4), 0);
#if defined(__AVX2__)
        Q8_KTensor::decodeBlockAVX2(block, decoded.data());
        EXPECT_EQ(std::memcmp(decoded.data(), expected.data() + b * 256, 256 * 4), 0);
#endif
#if defined(__AVX512F__)
        Q8_KTensor::decodeBlockAVX512(block, decoded.data());
        EXPECT_EQ(std::memcmp(decoded.data(), expected.data() + b * 256, 256 * 4), 0);
#endif
    }
}

TEST(Q8KSourceScale, MmapNestedSlicesKeepValuesAndOwner)
{
    const auto bytes = q8kSourceBytes();
    const auto expected = q8kSourceOracle(bytes);
    char name[] = "/tmp/llaminar-q8k-source-XXXXXX";
    const int fd = ::mkstemp(name);
    ASSERT_GE(fd, 0);
    size_t written = 0;
    while (written < bytes.size())
    {
        const auto count = ::write(fd, bytes.data() + written, bytes.size() - written);
        if (count <= 0) break;
        written += static_cast<size_t>(count);
    }
    void* mapping = written == bytes.size()
        ? ::mmap(nullptr, bytes.size(), PROT_READ, MAP_PRIVATE, fd, 0) : MAP_FAILED;
    ::close(fd);
    ::unlink(name);
    ASSERT_NE(mapping, MAP_FAILED);
    bool retired = false;
    auto owner = std::shared_ptr<void>(mapping, [&](void* p) {
        ::munmap(p, bytes.size());
        retired = true;
    });
    std::weak_ptr<void> lifetime = owner;
    auto tensor = std::make_shared<Q8_KTensor>(std::vector<size_t>{4, 512},
        static_cast<const uint8_t*>(mapping), bytes.size(), owner);
    owner.reset();
    auto first = tensor->create_view({3, 512}, 512);
    auto nested = first->create_view({1, 512}, 512);
    tensor.reset();first.reset();
    ASSERT_FALSE(lifetime.expired());
    std::array<float, 512> decoded{};
    nested->to_fp32(decoded.data());
    EXPECT_EQ(std::memcmp(decoded.data(), expected.data() + 2 * 512, 512 * 4), 0);
    EXPECT_EQ(nested->size_bytes(), 2 * 292u);
    nested.reset();
    EXPECT_TRUE(lifetime.expired());
    EXPECT_TRUE(retired);
}

TEST(Q8KSourceScale, NativeInt8PublicationRetainsOriginalFP32Scale)
{
    const auto bytes = q8kSourceBytes();
    auto tensor = std::make_shared<Q8_KTensor>(std::vector<size_t>{4, 512}, bytes);
    for (size_t row = 0; row < 4; ++row)
    {
        for (size_t super = 0; super < 2; ++super)
        {
            const size_t b = row * 2 + super;
            float original;
            std::memcpy(&original, bytes.data() + b * 292, 4);
            std::array<int8_t, 256> codes{};
            std::array<float, 8> scales{};
            tensor->unpack_superblock_to_int8(row, super, codes.data(), scales.data());
            EXPECT_EQ(std::memcmp(codes.data(), bytes.data() + b * 292 + 4, 256), 0);
            for (size_t sub = 0; sub < 8; ++sub)
            {
                EXPECT_EQ(std::bit_cast<uint32_t>(scales[sub]), std::bit_cast<uint32_t>(original));
                EXPECT_EQ(std::bit_cast<uint32_t>(tensor->get_block_scale(row, super * 8 + sub)),
                    std::bit_cast<uint32_t>(original));
            }
        }
    }
}

TEST(Q8KSourceScale, ExistingPreparedFP16ScaleGapIsExplicit)
{
    const auto bytes = q8kSourceBytes();
    auto tensor = std::make_shared<Q8_KTensor>(std::vector<size_t>{4, 512}, bytes);
    std::array<uint8_t, 32> payload{};
    std::array<uint16_t, 1> scales{};
    VnniPackContext context{};
    context.N = 1;
    context.K = 512;
    context.blocks_per_row = 16;
    context.payload_bytes = 32;
    context.payload_array = payload.data();
    context.scales_array = scales.data();
    tensor->packVnniBlock(context, 0, 0, 0);
    EXPECT_EQ(std::memcmp(payload.data(), bytes.data() + 4, 32), 0);
    const float original = tensor->get_block_scale(0, 0);
    EXPECT_EQ(scales[0], fp32_to_fp16(original));
    EXPECT_NE(fp16_to_fp32(scales[0]), original);
    EXPECT_EQ(std::bit_cast<uint32_t>(original), std::bit_cast<uint32_t>(0.1f));
    // FP32 source scale2^-25 is nonzero but underflows the existing prepared
    // binary16 scale; the original tensor/embedding value remains nonzero.
    EXPECT_EQ(tensor->get_block_scale(1, 0), 0x1p-25f);
    EXPECT_EQ(fp16_to_fp32(fp32_to_fp16(tensor->get_block_scale(1, 0))), 0.0f);
}
