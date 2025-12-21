/**
 * @file TestTensorFactory.h
 * @brief Test-friendly tensor factory with convenience methods
 * @author David Sanftenberg
 *
 * This header provides a simplified TensorFactory for unit tests that:
 * - Does NOT require MPIContext (uses CPU device -1)
 * - Provides random data filling
 * - Supports creating quantized tensors with test patterns
 * - Is header-only for easy inclusion
 *
 * Usage:
 *   #include "utils/TestTensorFactory.h"
 *
 *   auto input = TestTensorFactory::createFP32({32, 896});
 *   TestTensorFactory::fillRandom(input.get());
 *
 *   auto weights = TestTensorFactory::createQ8_0Random({1024, 896});
 */

#pragma once

#include "tensors/Tensors.h"
#include <memory>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>

namespace llaminar2::test
{

    /**
     * @brief Test-friendly tensor factory without MPI dependencies
     *
     * All tensors are created on CPU (device_idx = -1).
     * Provides random filling and pattern generation for testing.
     */
    class TestTensorFactory
    {
    public:
        // =========================================================================
        // Activation Tensor Creation (FP32, FP16, BF16, Q8_1)
        // =========================================================================

        /**
         * @brief Create FP32 tensor
         * @param shape Tensor dimensions
         * @return Uninitialized FP32 tensor
         */
        static std::unique_ptr<FP32Tensor> createFP32(const std::vector<size_t> &shape)
        {
            return std::make_unique<FP32Tensor>(shape, /*device_idx=*/-1);
        }

        /**
         * @brief Create FP32 tensor filled with random values
         * @param shape Tensor dimensions
         * @param min Minimum value (default -1.0)
         * @param max Maximum value (default 1.0)
         * @param seed Random seed (default 42 for reproducibility)
         */
        static std::unique_ptr<FP32Tensor> createFP32Random(
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42)
        {
            auto tensor = createFP32(shape);
            fillRandom(tensor.get(), min, max, seed);
            return tensor;
        }

        /**
         * @brief Create FP32 tensor filled with zeros
         */
        static std::unique_ptr<FP32Tensor> createFP32Zeros(const std::vector<size_t> &shape)
        {
            auto tensor = createFP32(shape);
            fillZeros(tensor.get());
            return tensor;
        }

        /**
         * @brief Create FP32 tensor filled with ones
         */
        static std::unique_ptr<FP32Tensor> createFP32Ones(const std::vector<size_t> &shape)
        {
            auto tensor = createFP32(shape);
            fillValue(tensor.get(), 1.0f);
            return tensor;
        }

        /**
         * @brief Create FP16 tensor
         */
        static std::unique_ptr<FP16Tensor> createFP16(const std::vector<size_t> &shape)
        {
            return std::make_unique<FP16Tensor>(shape, /*device_idx=*/-1);
        }

        /**
         * @brief Create FP16 tensor filled with random values
         */
        static std::unique_ptr<FP16Tensor> createFP16Random(
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42)
        {
            auto tensor = createFP16(shape);
            fillRandomFP16(tensor.get(), min, max, seed);
            return tensor;
        }

        /**
         * @brief Create BF16 tensor
         */
        static std::unique_ptr<BF16Tensor> createBF16(const std::vector<size_t> &shape)
        {
            return std::make_unique<BF16Tensor>(shape, /*device_idx=*/-1);
        }

        /**
         * @brief Create BF16 tensor filled with random values
         */
        static std::unique_ptr<BF16Tensor> createBF16Random(
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42)
        {
            auto tensor = createBF16(shape);
            fillRandomBF16(tensor.get(), min, max, seed);
            return tensor;
        }

        /**
         * @brief Create INT32 tensor
         */
        static std::unique_ptr<INT32Tensor> createINT32(const std::vector<size_t> &shape)
        {
            return std::make_unique<INT32Tensor>(shape, /*device_idx=*/-1);
        }

        /**
         * @brief Create Q8_1 tensor (activation quantization format)
         */
        static std::unique_ptr<Q8_1Tensor> createQ8_1(const std::vector<size_t> &shape)
        {
            return std::make_unique<Q8_1Tensor>(shape, /*device_idx=*/-1);
        }

        /**
         * @brief Create Q8_1 tensor with random quantized data
         *
         * Generates random FP32 values, then quantizes to Q8_1 format
         * with proper scale (d) and sum fields per block.
         */
        static std::unique_ptr<Q8_1Tensor> createQ8_1Random(
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42)
        {
            constexpr size_t BLOCK_SIZE = 32; // Q8_1 uses 32-element blocks

            size_t rows = shape[0];
            size_t cols = shape[1];
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t total_blocks = rows * blocks_per_row;

            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> dist(min, max);

            // Q8_1 block: 32 int8 values + fp16 scale (d) + fp16 sum (s)
            std::vector<uint8_t> raw_data(total_blocks * sizeof(block_q8_1));
            auto *blocks = reinterpret_cast<block_q8_1 *>(raw_data.data());

            for (size_t i = 0; i < total_blocks; ++i)
            {
                float values[BLOCK_SIZE];
                float max_abs = 0.0f;
                float sum = 0.0f;

                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    values[j] = dist(rng);
                    max_abs = std::max(max_abs, std::abs(values[j]));
                    sum += values[j];
                }

                float scale = max_abs / 127.0f;
                blocks[i].d = fp32_to_fp16(scale);
                blocks[i].s = fp32_to_fp16(sum); // Q8_1 stores sum for efficient dot products

                float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    int32_t q = static_cast<int32_t>(std::round(values[j] * inv_scale));
                    q = std::clamp(q, -128, 127);
                    blocks[i].qs[j] = static_cast<int8_t>(q);
                }
            }

            return std::make_unique<Q8_1Tensor>(shape, raw_data.data(), raw_data.size(), /*device_idx=*/-1);
        }

        // =========================================================================
        // Quantized Weight Tensor Creation (Q8_0, Q4_0, IQ4_NL, etc.)
        // =========================================================================

        /**
         * @brief Create Q8_0 tensor with random quantized data
         *
         * Creates realistic quantized weights by:
         * 1. Generating random FP32 values
         * 2. Quantizing them to Q8_0 format
         *
         * @param shape Tensor dimensions [rows, cols]
         * @param seed Random seed
         */
        static std::unique_ptr<Q8_0Tensor> createQ8_0Random(
            const std::vector<size_t> &shape,
            uint32_t seed = 42)
        {
            // Q8_0 block: 32 int8 values + 1 fp16 scale = 34 bytes
            constexpr size_t BLOCK_SIZE = 32;

            size_t rows = shape[0];
            size_t cols = shape[1];
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t total_blocks = rows * blocks_per_row;

            // Generate random FP32 data first
            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(0.0f, 0.1f); // Normal distribution typical for weights

            std::vector<uint8_t> raw_data(total_blocks * sizeof(block_q8_0));
            auto *blocks = reinterpret_cast<block_q8_0 *>(raw_data.data());

            for (size_t i = 0; i < total_blocks; ++i)
            {
                // Generate random values for this block
                float max_abs = 0.0f;
                float values[BLOCK_SIZE];
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    values[j] = dist(rng);
                    max_abs = std::max(max_abs, std::abs(values[j]));
                }

                // Compute scale
                float scale = max_abs / 127.0f;
                blocks[i].d = fp32_to_fp16(scale);

                // Quantize
                float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    int32_t q = static_cast<int32_t>(std::round(values[j] * inv_scale));
                    q = std::clamp(q, -128, 127);
                    blocks[i].qs[j] = static_cast<int8_t>(q);
                }
            }

            return std::make_unique<Q8_0Tensor>(shape, raw_data.data(), raw_data.size(), /*device_idx=*/-1);
        }

        /**
         * @brief Create Q4_0 tensor with random quantized data
         */
        static std::unique_ptr<Q4_0Tensor> createQ4_0Random(
            const std::vector<size_t> &shape,
            uint32_t seed = 42)
        {
            constexpr size_t BLOCK_SIZE = 32;

            size_t rows = shape[0];
            size_t cols = shape[1];
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t total_blocks = rows * blocks_per_row;

            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(0.0f, 0.1f);

            std::vector<uint8_t> raw_data(total_blocks * sizeof(block_q4_0));
            auto *blocks = reinterpret_cast<block_q4_0 *>(raw_data.data());

            for (size_t i = 0; i < total_blocks; ++i)
            {
                float max_abs = 0.0f;
                float values[BLOCK_SIZE];
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    values[j] = dist(rng);
                    max_abs = std::max(max_abs, std::abs(values[j]));
                }

                float scale = max_abs / 7.0f; // Q4 range is [-8, 7]
                blocks[i].d = fp32_to_fp16(scale);

                float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
                for (size_t j = 0; j < BLOCK_SIZE / 2; ++j)
                {
                    int32_t q0 = static_cast<int32_t>(std::round(values[2 * j] * inv_scale)) + 8;
                    int32_t q1 = static_cast<int32_t>(std::round(values[2 * j + 1] * inv_scale)) + 8;
                    q0 = std::clamp(q0, 0, 15);
                    q1 = std::clamp(q1, 0, 15);
                    blocks[i].qs[j] = static_cast<uint8_t>((q1 << 4) | q0);
                }
            }

            return std::make_unique<Q4_0Tensor>(shape, raw_data.data(), raw_data.size(), /*device_idx=*/-1);
        }

        // =========================================================================
        // Fill Methods (operate on existing tensors)
        // =========================================================================

        /**
         * @brief Fill FP32 tensor with random values
         * @param tensor Tensor to fill
         * @param min Minimum value
         * @param max Maximum value
         * @param seed Random seed
         */
        static void fillRandom(FP32Tensor *tensor, float min = -1.0f, float max = 1.0f, uint32_t seed = 42)
        {
            if (!tensor)
                return;
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> dist(min, max);
            float *data = tensor->mutable_data();
            for (size_t i = 0; i < tensor->numel(); ++i)
            {
                data[i] = dist(rng);
            }
        }

        /**
         * @brief Fill FP32 tensor with normal distribution (typical for weights)
         */
        static void fillNormal(FP32Tensor *tensor, float mean = 0.0f, float stddev = 0.1f, uint32_t seed = 42)
        {
            if (!tensor)
                return;
            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(mean, stddev);
            float *data = tensor->mutable_data();
            for (size_t i = 0; i < tensor->numel(); ++i)
            {
                data[i] = dist(rng);
            }
        }

        /**
         * @brief Fill FP32 tensor with zeros
         */
        static void fillZeros(FP32Tensor *tensor)
        {
            if (!tensor)
                return;
            std::memset(tensor->mutable_data(), 0, tensor->numel() * sizeof(float));
        }

        /**
         * @brief Fill FP32 tensor with constant value
         */
        static void fillValue(FP32Tensor *tensor, float value)
        {
            if (!tensor)
                return;
            float *data = tensor->mutable_data();
            for (size_t i = 0; i < tensor->numel(); ++i)
            {
                data[i] = value;
            }
        }

        /**
         * @brief Fill FP32 tensor with sequential values (for debugging)
         * @param start Starting value
         * @param step Step between values
         */
        static void fillSequential(FP32Tensor *tensor, float start = 0.0f, float step = 1.0f)
        {
            if (!tensor)
                return;
            float *data = tensor->mutable_data();
            float val = start;
            for (size_t i = 0; i < tensor->numel(); ++i)
            {
                data[i] = val;
                val += step;
            }
        }

        /**
         * @brief Fill FP32 tensor with repeating pattern (for pattern detection)
         * Values are (index % 256) / 256.0 - 0.5
         */
        static void fillPattern(FP32Tensor *tensor)
        {
            if (!tensor)
                return;
            float *data = tensor->mutable_data();
            for (size_t i = 0; i < tensor->numel(); ++i)
            {
                data[i] = static_cast<float>(i % 256) / 256.0f - 0.5f;
            }
        }

        /**
         * @brief Fill FP16 tensor with random values
         */
        static void fillRandomFP16(FP16Tensor *tensor, float min = -1.0f, float max = 1.0f, uint32_t seed = 42)
        {
            if (!tensor)
                return;
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> dist(min, max);
            uint16_t *data = tensor->mutable_data();
            for (size_t i = 0; i < tensor->numel(); ++i)
            {
                data[i] = fp32_to_fp16(dist(rng));
            }
        }

        /**
         * @brief Fill BF16 tensor with random values
         */
        static void fillRandomBF16(BF16Tensor *tensor, float min = -1.0f, float max = 1.0f, uint32_t seed = 42)
        {
            if (!tensor)
                return;
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> dist(min, max);
            uint16_t *data = tensor->mutable_data();
            for (size_t i = 0; i < tensor->numel(); ++i)
            {
                data[i] = fp32_to_bf16(dist(rng));
            }
        }

        // =========================================================================
        // Comparison Utilities
        // =========================================================================

        /**
         * @brief Check if two values are approximately equal
         */
        static bool approxEqual(float a, float b, float rtol = 1e-4f, float atol = 1e-6f)
        {
            return std::abs(a - b) <= atol + rtol * std::abs(b);
        }

        /**
         * @brief Compute mean squared error between two arrays
         */
        static float computeMSE(const float *a, const float *b, size_t count)
        {
            float sum = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                float diff = a[i] - b[i];
                sum += diff * diff;
            }
            return sum / static_cast<float>(count);
        }

        /**
         * @brief Compute max absolute difference between two arrays
         */
        static float computeMaxAbsDiff(const float *a, const float *b, size_t count)
        {
            float max_diff = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
            }
            return max_diff;
        }

        /**
         * @brief Compute cosine similarity between two arrays
         */
        static float computeCosineSimilarity(const float *a, const float *b, size_t count)
        {
            float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                dot += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }
            float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
            return (denom > 0.0f) ? dot / denom : 0.0f;
        }

        /**
         * @brief Check if tensor contains any NaN or Inf values
         */
        static bool hasNaNOrInf(const FP32Tensor *tensor)
        {
            if (!tensor)
                return false;
            const float *data = tensor->data();
            for (size_t i = 0; i < tensor->numel(); ++i)
            {
                if (std::isnan(data[i]) || std::isinf(data[i]))
                {
                    return true;
                }
            }
            return false;
        }

    private:
        // FP32 to FP16 conversion helper
        static uint16_t fp32_to_fp16(float value)
        {
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(float));

            uint32_t sign = (bits >> 16) & 0x8000;
            int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
            uint32_t frac = (bits >> 13) & 0x3FF;

            if (exp <= 0)
            {
                return static_cast<uint16_t>(sign); // Underflow to zero
            }
            else if (exp >= 31)
            {
                return static_cast<uint16_t>(sign | 0x7C00); // Overflow to infinity
            }

            return static_cast<uint16_t>(sign | (exp << 10) | frac);
        }

        // FP32 to BF16 conversion helper
        // BF16 is simply the upper 16 bits of FP32 (truncation)
        static uint16_t fp32_to_bf16(float value)
        {
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(float));
            // BF16 = upper 16 bits of FP32 with rounding
            uint32_t rounding = 0x7FFF + ((bits >> 16) & 1);
            bits += rounding;
            return static_cast<uint16_t>(bits >> 16);
        }
    };

} // namespace llaminar2::test
