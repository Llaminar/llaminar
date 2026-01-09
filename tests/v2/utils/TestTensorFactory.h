/**
 * @file TestTensorFactory.h
 * @brief Test-friendly tensor factory with convenience methods
 * @author David Sanftenberg
 *
 * This header provides a simplified TensorFactory for unit tests that:
 * - Does NOT require MPIContext (uses CPU device -1)
 * - Provides random data filling
 * - Supports creating quantized tensors with test patterns
 * - Supports VNNI-packed weight creation for Q16 attention tests
 * - Is header-only for easy inclusion
 *
 * Usage:
 *   #include "utils/TestTensorFactory.h"
 *
 *   auto input = TestTensorFactory::createFP32({32, 896});
 *   TestTensorFactory::fillRandom(input.get());
 *
 *   auto weights = TestTensorFactory::createQ8_0Random({1024, 896});
 *
 *   // For Q16 attention tests requiring VNNI-packed Wo weights:
 *   auto wo_q8 = TestTensorFactory::createQ8_1FromFP32Random({d_model, d_model});
 *   auto wo_packed = TestTensorFactory::packToVNNI(wo_q8.get());
 */

#pragma once

#include "backends/DeviceId.h"
#include "tensors/Tensors.h"
#include "kernels/KernelFactory.h"
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
            return std::make_unique<FP32Tensor>(shape);
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
            return std::make_unique<FP16Tensor>(shape);
        }

        /**
         * @brief Create FP16 tensor filled with random values
         */
        static std::unique_ptr<FP16Tensor> createFP16Random(
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42)
        {
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> dist(min, max);

            size_t numel = 1;
            for (auto s : shape)
                numel *= s;

            std::vector<uint16_t> data(numel);
            for (size_t i = 0; i < numel; ++i)
            {
                data[i] = fp32_to_fp16(dist(rng));
            }

            return std::make_unique<FP16Tensor>(shape, data);
        }

        /**
         * @brief Create BF16 tensor
         */
        static std::unique_ptr<BF16Tensor> createBF16(const std::vector<size_t> &shape)
        {
            return std::make_unique<BF16Tensor>(shape);
        }

        /**
         * @brief Create BF16 tensor filled with random values
         */
        static std::unique_ptr<BF16Tensor> createBF16Random(
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42)
        {
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> dist(min, max);

            size_t numel = 1;
            for (auto s : shape)
                numel *= s;

            std::vector<uint16_t> data(numel);
            for (size_t i = 0; i < numel; ++i)
            {
                data[i] = fp32_to_bf16(dist(rng));
            }

            return std::make_unique<BF16Tensor>(shape, data);
        }

        /**
         * @brief Create INT32 tensor
         */
        static std::unique_ptr<INT32Tensor> createINT32(const std::vector<size_t> &shape)
        {
            return std::make_unique<INT32Tensor>(shape);
        }

        /**
         * @brief Create Q8_1 tensor (activation quantization format)
         */
        static std::unique_ptr<Q8_1Tensor> createQ8_1(const std::vector<size_t> &shape)
        {
            return std::make_unique<Q8_1Tensor>(shape, DeviceId::cpu());
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
            std::vector<uint8_t> raw_data(total_blocks * sizeof(Q8_1Block));
            auto *blocks = reinterpret_cast<Q8_1Block *>(raw_data.data());

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

                float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
                int16_t sum_qs = 0;
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    int32_t q = static_cast<int32_t>(std::round(values[j] * inv_scale));
                    q = std::clamp(q, -128, 127);
                    blocks[i].qs[j] = static_cast<int8_t>(q);
                    sum_qs += q;
                }
                blocks[i].sum_qs = sum_qs; // Q8_1 stores raw integer sum
            }

            return std::make_unique<Q8_1Tensor>(shape, reinterpret_cast<const Q8_1Block *>(raw_data.data()), total_blocks, DeviceId::cpu());
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

            std::vector<uint8_t> raw_data(total_blocks * sizeof(Q8_0Block));
            auto *blocks = reinterpret_cast<Q8_0Block *>(raw_data.data());

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

            return std::make_unique<Q8_0Tensor>(shape, raw_data);
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

            std::vector<uint8_t> raw_data(total_blocks * sizeof(Q4_0Block));
            auto *blocks = reinterpret_cast<Q4_0Block *>(raw_data.data());

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

            return std::make_unique<Q4_0Tensor>(shape, raw_data);
        }

        /**
         * @brief Create IQ4_NL tensor with random quantized data
         *
         * Creates realistic IQ4_NL quantized weights by:
         * 1. Generating random FP32 values from normal distribution
         * 2. Quantizing to IQ4_NL format using non-linear codebook
         *
         * IQ4_NL uses kvalues_iq4nl[16] lookup table for non-linear quantization.
         *
         * @param shape Tensor dimensions [rows, cols]
         * @param seed Random seed
         */
        static std::unique_ptr<IQ4_NLTensor> createIQ4_NLRandom(
            const std::vector<size_t> &shape,
            uint32_t seed = 42)
        {
            constexpr size_t BLOCK_SIZE = 32;

            size_t rows = shape[0];
            size_t cols = shape[1];
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t total_blocks = rows * blocks_per_row;

            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(0.0f, 0.1f); // Normal distribution typical for weights

            // IQ4_NL block: 2 bytes FP16 scale + 16 bytes packed indices = 18 bytes
            std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ4_NLBlock));
            auto *blocks = reinterpret_cast<IQ4_NLBlock *>(raw_data.data());

            // Non-linear codebook from IQQuantTables.h
            static constexpr float kvalues[16] = {
                -127.0f, -104.0f, -83.0f, -65.0f,
                -49.0f, -35.0f, -22.0f, -10.0f,
                1.0f, 13.0f, 25.0f, 38.0f,
                53.0f, 69.0f, 89.0f, 113.0f};

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

                // Compute scale (map max_abs to ~113, the max positive in codebook)
                float scale = max_abs / 113.0f;
                if (scale < 1e-10f)
                    scale = 1e-10f;
                blocks[i].d = fp32_to_fp16(scale);

                float inv_scale = 1.0f / scale;

                // Quantize each value to nearest codebook index
                for (size_t j = 0; j < BLOCK_SIZE / 2; ++j)
                {
                    // First element (stored in low nibble)
                    float v0 = values[j] * inv_scale;
                    int best_idx0 = 8; // Default to ~1.0
                    float best_dist0 = std::abs(v0 - kvalues[8]);
                    for (int k = 0; k < 16; ++k)
                    {
                        float d = std::abs(v0 - kvalues[k]);
                        if (d < best_dist0)
                        {
                            best_dist0 = d;
                            best_idx0 = k;
                        }
                    }

                    // Second element (stored in high nibble, offset by 16 in block)
                    float v1 = values[j + 16] * inv_scale;
                    int best_idx1 = 8;
                    float best_dist1 = std::abs(v1 - kvalues[8]);
                    for (int k = 0; k < 16; ++k)
                    {
                        float d = std::abs(v1 - kvalues[k]);
                        if (d < best_dist1)
                        {
                            best_dist1 = d;
                            best_idx1 = k;
                        }
                    }

                    // Pack: low nibble = first, high nibble = second
                    blocks[i].qs[j] = static_cast<uint8_t>((best_idx1 << 4) | best_idx0);
                }
            }

            return std::make_unique<IQ4_NLTensor>(shape, raw_data);
        }

        // =========================================================================
        // Q4_1 Tensor Creation (4-bit with min value)
        // =========================================================================

        /**
         * @brief Create Q4_1 tensor with random quantized data
         */
        static std::unique_ptr<Q4_1Tensor> createQ4_1Random(
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

            std::vector<uint8_t> raw_data(total_blocks * sizeof(Q4_1Block));
            auto *blocks = reinterpret_cast<Q4_1Block *>(raw_data.data());

            for (size_t i = 0; i < total_blocks; ++i)
            {
                float min_val = std::numeric_limits<float>::max();
                float max_val = std::numeric_limits<float>::lowest();
                float values[BLOCK_SIZE];
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    values[j] = dist(rng);
                    min_val = std::min(min_val, values[j]);
                    max_val = std::max(max_val, values[j]);
                }

                float scale = (max_val - min_val) / 15.0f;
                blocks[i].d = fp32_to_fp16(scale);
                blocks[i].m = fp32_to_fp16(min_val);

                float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
                for (size_t j = 0; j < BLOCK_SIZE / 2; ++j)
                {
                    int32_t q0 = static_cast<int32_t>(std::round((values[2 * j] - min_val) * inv_scale));
                    int32_t q1 = static_cast<int32_t>(std::round((values[2 * j + 1] - min_val) * inv_scale));
                    q0 = std::clamp(q0, 0, 15);
                    q1 = std::clamp(q1, 0, 15);
                    blocks[i].qs[j] = static_cast<uint8_t>((q1 << 4) | q0);
                }
            }

            return std::make_unique<Q4_1Tensor>(shape, raw_data);
        }

        // =========================================================================
        // Q5_0 Tensor Creation (5-bit quantization)
        // =========================================================================

        /**
         * @brief Create Q5_0 tensor with random quantized data
         */
        static std::unique_ptr<Q5_0Tensor> createQ5_0Random(
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

            std::vector<uint8_t> raw_data(total_blocks * sizeof(Q5_0Block));
            auto *blocks = reinterpret_cast<Q5_0Block *>(raw_data.data());

            for (size_t i = 0; i < total_blocks; ++i)
            {
                float max_abs = 0.0f;
                float values[BLOCK_SIZE];
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    values[j] = dist(rng);
                    max_abs = std::max(max_abs, std::abs(values[j]));
                }

                float scale = max_abs / 15.0f; // Q5 symmetric range is [-16, 15]
                blocks[i].d = fp32_to_fp16(scale);
                memset(blocks[i].qh, 0, sizeof(blocks[i].qh));

                float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
                for (size_t j = 0; j < BLOCK_SIZE / 2; ++j)
                {
                    int32_t q0 = static_cast<int32_t>(std::round(values[2 * j] * inv_scale)) + 16;
                    int32_t q1 = static_cast<int32_t>(std::round(values[2 * j + 1] * inv_scale)) + 16;
                    q0 = std::clamp(q0, 0, 31);
                    q1 = std::clamp(q1, 0, 31);
                    // Store lower 4 bits in qs, 5th bit in qh
                    blocks[i].qs[j] = static_cast<uint8_t>(((q1 & 0x0F) << 4) | (q0 & 0x0F));
                    // High bits stored in qh (4 bytes = 32 bits)
                    if (q0 & 0x10)
                        blocks[i].qh[j / 8] |= (1 << (j % 8));
                    if (q1 & 0x10)
                        blocks[i].qh[(j + 16) / 8] |= (1 << ((j + 16) % 8));
                }
            }

            return std::make_unique<Q5_0Tensor>(shape, raw_data);
        }

        // =========================================================================
        // Q5_1 Tensor Creation (5-bit with min value)
        // =========================================================================

        /**
         * @brief Create Q5_1 tensor with random quantized data
         */
        static std::unique_ptr<Q5_1Tensor> createQ5_1Random(
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

            std::vector<uint8_t> raw_data(total_blocks * sizeof(Q5_1Block));
            auto *blocks = reinterpret_cast<Q5_1Block *>(raw_data.data());

            for (size_t i = 0; i < total_blocks; ++i)
            {
                float min_val = std::numeric_limits<float>::max();
                float max_val = std::numeric_limits<float>::lowest();
                float values[BLOCK_SIZE];
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    values[j] = dist(rng);
                    min_val = std::min(min_val, values[j]);
                    max_val = std::max(max_val, values[j]);
                }

                float scale = (max_val - min_val) / 31.0f;
                blocks[i].d = fp32_to_fp16(scale);
                blocks[i].m = fp32_to_fp16(min_val);
                memset(blocks[i].qh, 0, sizeof(blocks[i].qh));

                float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
                for (size_t j = 0; j < BLOCK_SIZE / 2; ++j)
                {
                    int32_t q0 = static_cast<int32_t>(std::round((values[2 * j] - min_val) * inv_scale));
                    int32_t q1 = static_cast<int32_t>(std::round((values[2 * j + 1] - min_val) * inv_scale));
                    q0 = std::clamp(q0, 0, 31);
                    q1 = std::clamp(q1, 0, 31);
                    blocks[i].qs[j] = static_cast<uint8_t>(((q1 & 0x0F) << 4) | (q0 & 0x0F));
                    if (q0 & 0x10)
                        blocks[i].qh[j / 8] |= (1 << (j % 8));
                    if (q1 & 0x10)
                        blocks[i].qh[(j + 16) / 8] |= (1 << ((j + 16) % 8));
                }
            }

            return std::make_unique<Q5_1Tensor>(shape, raw_data);
        }

        // =========================================================================
        // K-Quant Tensor Factory Methods (256-element super-blocks)
        // =========================================================================

        /**
         * @brief Create Q6_K tensor with random quantized data
         * K-quant format with 256 elements per super-block
         */
        static std::unique_ptr<Q6_KTensor> createQ6_KRandom(
            const std::vector<size_t> &shape,
            uint32_t seed = 42)
        {
            constexpr size_t BLOCK_SIZE = 256;

            size_t rows = shape[0];
            size_t cols = shape[1];
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t total_blocks = rows * blocks_per_row;

            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(0.0f, 0.1f);

            std::vector<uint8_t> raw_data(total_blocks * sizeof(Q6_KBlock));
            auto *blocks = reinterpret_cast<Q6_KBlock *>(raw_data.data());

            for (size_t i = 0; i < total_blocks; ++i)
            {
                float max_abs = 0.0f;
                float values[BLOCK_SIZE];
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    values[j] = dist(rng);
                    max_abs = std::max(max_abs, std::abs(values[j]));
                }

                float scale = max_abs / 31.0f; // 6-bit has range [-32, 31]
                blocks[i].d = fp32_to_fp16(scale);

                // Initialize per-block scales to 1
                for (int s = 0; s < 16; ++s)
                {
                    blocks[i].scales[s] = 32; // Neutral scale (centered)
                }

                float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
                // Simplified quantization - just store values directly
                for (size_t j = 0; j < 128; ++j)
                {
                    int32_t q = static_cast<int32_t>(std::round(values[j * 2] * inv_scale)) + 32;
                    q = std::clamp(q, 0, 63);
                    blocks[i].ql[j] = static_cast<uint8_t>(q & 0x0F);
                    blocks[i].qh[j / 2] |= static_cast<uint8_t>(((q >> 4) & 0x03) << ((j % 2) * 2));
                }
            }

            return std::make_unique<Q6_KTensor>(shape, raw_data);
        }

        /**
         * @brief Create Q2_K tensor with random quantized data
         */
        static std::unique_ptr<Q2_KTensor> createQ2_KRandom(
            const std::vector<size_t> &shape,
            uint32_t seed = 42)
        {
            constexpr size_t BLOCK_SIZE = 256;

            size_t rows = shape[0];
            size_t cols = shape[1];
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t total_blocks = rows * blocks_per_row;

            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(0.0f, 0.1f);

            std::vector<uint8_t> raw_data(total_blocks * sizeof(Q2_KBlock));
            auto *blocks = reinterpret_cast<Q2_KBlock *>(raw_data.data());

            for (size_t i = 0; i < total_blocks; ++i)
            {
                float max_abs = 0.0f;
                float values[BLOCK_SIZE];
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    values[j] = dist(rng);
                    max_abs = std::max(max_abs, std::abs(values[j]));
                }

                float scale = max_abs / 1.5f; // Q2 range is small
                blocks[i].d = fp32_to_fp16(scale);
                blocks[i].dmin = fp32_to_fp16(0.0f);

                memset(blocks[i].scales, 0x44, sizeof(blocks[i].scales)); // Neutral scales

                float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
                for (size_t j = 0; j < 64; ++j)
                {
                    uint8_t packed = 0;
                    for (size_t k = 0; k < 4; ++k)
                    {
                        int32_t q = static_cast<int32_t>(std::round(values[j * 4 + k] * inv_scale)) + 2;
                        q = std::clamp(q, 0, 3);
                        packed |= static_cast<uint8_t>(q << (k * 2));
                    }
                    blocks[i].qs[j] = packed;
                }
            }

            return std::make_unique<Q2_KTensor>(shape, raw_data);
        }

        /**
         * @brief Create Q3_K tensor with random quantized data
         */
        static std::unique_ptr<Q3_KTensor> createQ3_KRandom(
            const std::vector<size_t> &shape,
            uint32_t seed = 42)
        {
            constexpr size_t BLOCK_SIZE = 256;

            size_t rows = shape[0];
            size_t cols = shape[1];
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t total_blocks = rows * blocks_per_row;

            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(0.0f, 0.1f);

            std::vector<uint8_t> raw_data(total_blocks * sizeof(Q3_KBlock));
            auto *blocks = reinterpret_cast<Q3_KBlock *>(raw_data.data());

            for (size_t i = 0; i < total_blocks; ++i)
            {
                float max_abs = 0.0f;
                float values[BLOCK_SIZE];
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    values[j] = dist(rng);
                    max_abs = std::max(max_abs, std::abs(values[j]));
                }

                float scale = max_abs / 3.5f; // Q3 range
                blocks[i].d = fp32_to_fp16(scale);

                memset(blocks[i].hmask, 0, sizeof(blocks[i].hmask));
                memset(blocks[i].scales, 0x44, sizeof(blocks[i].scales)); // Neutral scales

                float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
                for (size_t j = 0; j < 64; ++j)
                {
                    uint8_t packed = 0;
                    for (size_t k = 0; k < 4; ++k)
                    {
                        int32_t q = static_cast<int32_t>(std::round(values[j * 4 + k] * inv_scale)) + 4;
                        q = std::clamp(q, 0, 7);
                        packed |= static_cast<uint8_t>((q & 0x03) << (k * 2));
                        if (q & 0x04)
                            blocks[i].hmask[j * 4 / 8 + k / 8] |= (1 << ((j * 4 + k) % 8));
                    }
                    blocks[i].qs[j] = packed;
                }
            }

            return std::make_unique<Q3_KTensor>(shape, raw_data);
        }

        /**
         * @brief Create Q4_K tensor with random quantized data
         */
        static std::unique_ptr<Q4_KTensor> createQ4_KRandom(
            const std::vector<size_t> &shape,
            uint32_t seed = 42)
        {
            constexpr size_t BLOCK_SIZE = 256;

            size_t rows = shape[0];
            size_t cols = shape[1];
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t total_blocks = rows * blocks_per_row;

            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(0.0f, 0.1f);

            std::vector<uint8_t> raw_data(total_blocks * sizeof(Q4_KBlock));
            auto *blocks = reinterpret_cast<Q4_KBlock *>(raw_data.data());

            for (size_t i = 0; i < total_blocks; ++i)
            {
                float max_abs = 0.0f;
                float values[BLOCK_SIZE];
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    values[j] = dist(rng);
                    max_abs = std::max(max_abs, std::abs(values[j]));
                }

                float scale = max_abs / 7.0f;
                blocks[i].d = fp32_to_fp16(scale);
                blocks[i].dmin = fp32_to_fp16(0.0f);

                memset(blocks[i].scales, 0x44, sizeof(blocks[i].scales));

                float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
                for (size_t j = 0; j < 128; ++j)
                {
                    int32_t q0 = static_cast<int32_t>(std::round(values[j * 2] * inv_scale)) + 8;
                    int32_t q1 = static_cast<int32_t>(std::round(values[j * 2 + 1] * inv_scale)) + 8;
                    q0 = std::clamp(q0, 0, 15);
                    q1 = std::clamp(q1, 0, 15);
                    blocks[i].qs[j] = static_cast<uint8_t>((q1 << 4) | q0);
                }
            }

            return std::make_unique<Q4_KTensor>(shape, raw_data);
        }

        /**
         * @brief Create Q5_K tensor with random quantized data
         */
        static std::unique_ptr<Q5_KTensor> createQ5_KRandom(
            const std::vector<size_t> &shape,
            uint32_t seed = 42)
        {
            constexpr size_t BLOCK_SIZE = 256;

            size_t rows = shape[0];
            size_t cols = shape[1];
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t total_blocks = rows * blocks_per_row;

            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(0.0f, 0.1f);

            std::vector<uint8_t> raw_data(total_blocks * sizeof(Q5_KBlock));
            auto *blocks = reinterpret_cast<Q5_KBlock *>(raw_data.data());

            for (size_t i = 0; i < total_blocks; ++i)
            {
                float max_abs = 0.0f;
                float values[BLOCK_SIZE];
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    values[j] = dist(rng);
                    max_abs = std::max(max_abs, std::abs(values[j]));
                }

                float scale = max_abs / 15.0f;
                blocks[i].d = fp32_to_fp16(scale);
                blocks[i].dmin = fp32_to_fp16(0.0f);

                memset(blocks[i].scales, 0x44, sizeof(blocks[i].scales));
                memset(blocks[i].qh, 0, sizeof(blocks[i].qh));

                float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
                for (size_t j = 0; j < 128; ++j)
                {
                    int32_t q0 = static_cast<int32_t>(std::round(values[j * 2] * inv_scale)) + 16;
                    int32_t q1 = static_cast<int32_t>(std::round(values[j * 2 + 1] * inv_scale)) + 16;
                    q0 = std::clamp(q0, 0, 31);
                    q1 = std::clamp(q1, 0, 31);
                    blocks[i].qs[j] = static_cast<uint8_t>(((q1 & 0x0F) << 4) | (q0 & 0x0F));
                    if (q0 & 0x10)
                        blocks[i].qh[j / 4] |= (1 << ((j % 4) * 2));
                    if (q1 & 0x10)
                        blocks[i].qh[j / 4] |= (1 << ((j % 4) * 2 + 1));
                }
            }

            return std::make_unique<Q5_KTensor>(shape, raw_data);
        }

        // =========================================================================
        // IQ (Importance Quantization) Tensor Factory Methods
        // =========================================================================

        /**
         * @brief Create IQ4_XS tensor with random quantized data
         * 4-bit importance quantization with extra-small overhead
         */
        static std::unique_ptr<IQ4_XSTensor> createIQ4_XSRandom(
            const std::vector<size_t> &shape,
            uint32_t seed = 42)
        {
            constexpr size_t BLOCK_SIZE = 256;

            size_t rows = shape[0];
            size_t cols = shape[1];
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t total_blocks = rows * blocks_per_row;

            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(0.0f, 0.1f);

            std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ4_XSBlock));
            auto *blocks = reinterpret_cast<IQ4_XSBlock *>(raw_data.data());

            for (size_t i = 0; i < total_blocks; ++i)
            {
                float max_abs = 0.0f;
                float values[BLOCK_SIZE];
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    values[j] = dist(rng);
                    max_abs = std::max(max_abs, std::abs(values[j]));
                }

                float scale = max_abs / 7.0f;
                blocks[i].d = fp32_to_fp16(scale);
                blocks[i].scales_h = 0;
                memset(blocks[i].scales_l, 0x44, sizeof(blocks[i].scales_l));

                float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
                for (size_t j = 0; j < 128; ++j)
                {
                    int32_t q0 = static_cast<int32_t>(std::round(values[j * 2] * inv_scale)) + 8;
                    int32_t q1 = static_cast<int32_t>(std::round(values[j * 2 + 1] * inv_scale)) + 8;
                    q0 = std::clamp(q0, 0, 15);
                    q1 = std::clamp(q1, 0, 15);
                    blocks[i].qs[j] = static_cast<uint8_t>((q1 << 4) | q0);
                }
            }

            return std::make_unique<IQ4_XSTensor>(shape, raw_data);
        }

        /**
         * @brief Create IQ2_XXS tensor with random quantized data
         * 2-bit extra-extra-small importance quantization
         */
        static std::unique_ptr<IQ2_XXSTensor> createIQ2_XXSRandom(
            const std::vector<size_t> &shape,
            uint32_t seed = 42)
        {
            constexpr size_t BLOCK_SIZE = 256;

            size_t rows = shape[0];
            size_t cols = shape[1];
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t total_blocks = rows * blocks_per_row;

            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(0.0f, 0.1f);

            std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ2_XXSBlock));
            auto *blocks = reinterpret_cast<IQ2_XXSBlock *>(raw_data.data());

            for (size_t i = 0; i < total_blocks; ++i)
            {
                float max_abs = 0.0f;
                float values[BLOCK_SIZE];
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    values[j] = dist(rng);
                    max_abs = std::max(max_abs, std::abs(values[j]));
                }

                float scale = max_abs / 1.5f;
                blocks[i].d = fp32_to_fp16(scale);

                // Simplified: just fill with default grid indices
                for (size_t j = 0; j < 32; ++j)
                {
                    blocks[i].qs[j] = static_cast<uint16_t>(rng() & 0xFFFF);
                }
            }

            return std::make_unique<IQ2_XXSTensor>(shape, raw_data);
        }

        /**
         * @brief Create IQ2_XS tensor with random quantized data
         */
        static std::unique_ptr<IQ2_XSTensor> createIQ2_XSRandom(
            const std::vector<size_t> &shape,
            uint32_t seed = 42)
        {
            constexpr size_t BLOCK_SIZE = 256;

            size_t rows = shape[0];
            size_t cols = shape[1];
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t total_blocks = rows * blocks_per_row;

            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(0.0f, 0.1f);

            std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ2_XSBlock));
            auto *blocks = reinterpret_cast<IQ2_XSBlock *>(raw_data.data());

            for (size_t i = 0; i < total_blocks; ++i)
            {
                float max_abs = 0.0f;
                float values[BLOCK_SIZE];
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    values[j] = dist(rng);
                    max_abs = std::max(max_abs, std::abs(values[j]));
                }

                float scale = max_abs / 1.5f;
                blocks[i].d = fp32_to_fp16(scale);
                memset(blocks[i].scales, 0x44, sizeof(blocks[i].scales));

                for (size_t j = 0; j < 32; ++j)
                {
                    blocks[i].qs[j] = static_cast<uint16_t>(rng() & 0xFFFF);
                }
            }

            return std::make_unique<IQ2_XSTensor>(shape, raw_data);
        }

        /**
         * @brief Create IQ2_S tensor with random quantized data
         */
        static std::unique_ptr<IQ2_STensor> createIQ2_SRandom(
            const std::vector<size_t> &shape,
            uint32_t seed = 42)
        {
            constexpr size_t BLOCK_SIZE = 256;

            size_t rows = shape[0];
            size_t cols = shape[1];
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t total_blocks = rows * blocks_per_row;

            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(0.0f, 0.1f);

            std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ2_SBlock));
            auto *blocks = reinterpret_cast<IQ2_SBlock *>(raw_data.data());

            for (size_t i = 0; i < total_blocks; ++i)
            {
                float max_abs = 0.0f;
                float values[BLOCK_SIZE];
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    values[j] = dist(rng);
                    max_abs = std::max(max_abs, std::abs(values[j]));
                }

                float scale = max_abs / 1.5f;
                blocks[i].d = fp32_to_fp16(scale);
                memset(blocks[i].scales, 0x44, sizeof(blocks[i].scales));
                memset(blocks[i].qh, 0, sizeof(blocks[i].qh));

                for (size_t j = 0; j < 64; ++j)
                {
                    blocks[i].qs[j] = static_cast<uint8_t>(rng() & 0xFF);
                }
            }

            return std::make_unique<IQ2_STensor>(shape, raw_data);
        }

        /**
         * @brief Create IQ3_XXS tensor with random quantized data
         */
        static std::unique_ptr<IQ3_XXSTensor> createIQ3_XXSRandom(
            const std::vector<size_t> &shape,
            uint32_t seed = 42)
        {
            constexpr size_t BLOCK_SIZE = 256;

            size_t rows = shape[0];
            size_t cols = shape[1];
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t total_blocks = rows * blocks_per_row;

            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(0.0f, 0.1f);

            std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ3_XXSBlock));
            auto *blocks = reinterpret_cast<IQ3_XXSBlock *>(raw_data.data());

            for (size_t i = 0; i < total_blocks; ++i)
            {
                float max_abs = 0.0f;
                float values[BLOCK_SIZE];
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    values[j] = dist(rng);
                    max_abs = std::max(max_abs, std::abs(values[j]));
                }

                float scale = max_abs / 3.5f;
                blocks[i].d = fp32_to_fp16(scale);

                for (size_t j = 0; j < 96; ++j)
                {
                    blocks[i].qs[j] = static_cast<uint8_t>(rng() & 0xFF);
                }
            }

            return std::make_unique<IQ3_XXSTensor>(shape, raw_data);
        }

        /**
         * @brief Create IQ3_S tensor with random quantized data
         */
        static std::unique_ptr<IQ3_STensor> createIQ3_SRandom(
            const std::vector<size_t> &shape,
            uint32_t seed = 42)
        {
            constexpr size_t BLOCK_SIZE = 256;

            size_t rows = shape[0];
            size_t cols = shape[1];
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t total_blocks = rows * blocks_per_row;

            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(0.0f, 0.1f);

            std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ3_SBlock));
            auto *blocks = reinterpret_cast<IQ3_SBlock *>(raw_data.data());

            for (size_t i = 0; i < total_blocks; ++i)
            {
                float max_abs = 0.0f;
                float values[BLOCK_SIZE];
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    values[j] = dist(rng);
                    max_abs = std::max(max_abs, std::abs(values[j]));
                }

                float scale = max_abs / 3.5f;
                blocks[i].d = fp32_to_fp16(scale);
                memset(blocks[i].qh, 0, sizeof(blocks[i].qh));
                memset(blocks[i].signs, 0, sizeof(blocks[i].signs));
                memset(blocks[i].scales, 0x44, sizeof(blocks[i].scales));

                for (size_t j = 0; j < 64; ++j)
                {
                    blocks[i].qs[j] = static_cast<uint8_t>(rng() & 0xFF);
                }
            }

            return std::make_unique<IQ3_STensor>(shape, raw_data);
        }

        /**
         * @brief Create IQ1_S tensor with random quantized data
         */
        static std::unique_ptr<IQ1_STensor> createIQ1_SRandom(
            const std::vector<size_t> &shape,
            uint32_t seed = 42)
        {
            constexpr size_t BLOCK_SIZE = 256;

            size_t rows = shape[0];
            size_t cols = shape[1];
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t total_blocks = rows * blocks_per_row;

            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(0.0f, 0.1f);

            std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ1_SBlock));
            auto *blocks = reinterpret_cast<IQ1_SBlock *>(raw_data.data());

            for (size_t i = 0; i < total_blocks; ++i)
            {
                float max_abs = 0.0f;
                float values[BLOCK_SIZE];
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    values[j] = dist(rng);
                    max_abs = std::max(max_abs, std::abs(values[j]));
                }

                float scale = max_abs / 0.5f;
                blocks[i].d = fp32_to_fp16(scale);

                for (size_t j = 0; j < 32; ++j)
                {
                    blocks[i].qs[j] = static_cast<uint8_t>(rng() & 0xFF);
                }
                for (size_t j = 0; j < 8; ++j)
                {
                    blocks[i].qh[j] = static_cast<uint16_t>(rng() & 0xFFFF);
                }
            }

            return std::make_unique<IQ1_STensor>(shape, raw_data);
        }

        /**
         * @brief Create IQ1_M tensor with random quantized data
         */
        static std::unique_ptr<IQ1_MTensor> createIQ1_MRandom(
            const std::vector<size_t> &shape,
            uint32_t seed = 42)
        {
            constexpr size_t BLOCK_SIZE = 256;

            size_t rows = shape[0];
            size_t cols = shape[1];
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t total_blocks = rows * blocks_per_row;

            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(0.0f, 0.1f);

            std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ1_MBlock));
            auto *blocks = reinterpret_cast<IQ1_MBlock *>(raw_data.data());

            for (size_t i = 0; i < total_blocks; ++i)
            {
                float values[BLOCK_SIZE];
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    values[j] = dist(rng);
                }

                for (size_t j = 0; j < 32; ++j)
                {
                    blocks[i].qs[j] = static_cast<uint8_t>(rng() & 0xFF);
                }
                for (size_t j = 0; j < 16; ++j)
                {
                    blocks[i].qh[j] = static_cast<uint8_t>(rng() & 0xFF);
                }
                for (size_t j = 0; j < 8; ++j)
                {
                    blocks[i].scales[j] = static_cast<uint8_t>(rng() & 0xFF);
                }
            }

            return std::make_unique<IQ1_MTensor>(shape, raw_data);
        }

        // =========================================================================
        // Q16_1 Tensor Creation (for Q16 integer-domain attention)
        // =========================================================================

        /**
         * @brief Create empty Q16_1 tensor
         * @param shape Tensor dimensions [rows, cols]
         * @return Uninitialized Q16_1 tensor
         */
        static std::unique_ptr<Q16_1Tensor> createQ16_1(const std::vector<size_t> &shape)
        {
            return std::make_unique<Q16_1Tensor>(shape, DeviceId::cpu());
        }

        /**
         * @brief Create Q16_1 tensor with random quantized data
         *
         * Generates random FP32 values, then quantizes to Q16_1 format
         * with proper scale and sum fields per block.
         *
         * @param shape Tensor dimensions [rows, cols]
         * @param min Minimum value for random generation
         * @param max Maximum value for random generation
         * @param seed Random seed
         */
        static std::unique_ptr<Q16_1Tensor> createQ16_1Random(
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42)
        {
            // Generate FP32 data first
            size_t numel = 1;
            for (auto s : shape)
                numel *= s;

            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> dist(min, max);

            std::vector<float> fp32_data(numel);
            for (size_t i = 0; i < numel; ++i)
            {
                fp32_data[i] = dist(rng);
            }

            // Use built-in quantization
            auto tensor = Q16_1Tensor::quantize_from_fp32(fp32_data.data(), shape);

            // Convert shared_ptr to unique_ptr by creating a new tensor
            // This is safe because we own the only reference
            auto result = std::make_unique<Q16_1Tensor>(*tensor);
            return result;
        }

        // =========================================================================
        // VNNI-Packed Weight Creation (for Q16 attention Wo projection)
        // =========================================================================

        /**
         * @brief Create Q8_1 tensor by quantizing FP32 random data
         *
         * This is useful for creating weight tensors that will be packed
         * to VNNI format. Uses the tensor's built-in quantization.
         *
         * @param shape Tensor dimensions [rows, cols]
         * @param mean Mean for normal distribution (default 0.0)
         * @param stddev Standard deviation (default 0.1, typical for weights)
         * @param seed Random seed
         */
        static std::shared_ptr<Q8_1Tensor> createQ8_1FromFP32Random(
            const std::vector<size_t> &shape,
            float mean = 0.0f, float stddev = 0.1f,
            uint32_t seed = 42)
        {
            size_t numel = 1;
            for (auto s : shape)
                numel *= s;

            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(mean, stddev);

            std::vector<float> fp32_data(numel);
            for (size_t i = 0; i < numel; ++i)
            {
                fp32_data[i] = dist(rng);
            }

            return Q8_1Tensor::quantize_from_fp32(fp32_data.data(), shape);
        }

        /**
         * @brief Pack a quantized tensor to VNNI format for Q16 attention
         *
         * Takes a Q8_1 (or other INT8-based) tensor and returns VNNI-packed
         * weights suitable for use with Q16FusedAttention kernels.
         *
         * The packed weights are cached in the tensor, so this is safe to
         * call multiple times - subsequent calls return the cached version.
         *
         * @param tensor Q8_1 tensor to pack (must implement IINT8Unpackable)
         * @return Pointer to VNNI-packed weights (lifetime tied to tensor)
         * @throws std::runtime_error if packing fails
         *
         * Example usage:
         *   auto wo_q8 = TestTensorFactory::createQ8_1FromFP32Random({d_model, d_model});
         *   auto wo_packed = TestTensorFactory::packToVNNI(wo_q8.get());
         *   params.Wo_packed = wo_packed;
         */
        static const llaminar2::gemm_v4::QuantisedPackedWeights *packToVNNI(
            const TensorBase *tensor)
        {
            return llaminar::v2::kernels::KernelFactory::ensurePackedWeightsInTensorCache(tensor);
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
