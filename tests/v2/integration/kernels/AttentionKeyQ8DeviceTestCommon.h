/**
 * @file AttentionKeyQ8DeviceTestCommon.h
 * @brief Shared deterministic fixtures for CUDA/ROCm attention-key Q8 tests.
 *
 * Device tests use this header to construct identical finite inputs and scalar
 * reference blocks without depending on standard-library random distributions.
 * The final block is all zero so every backend also covers the zero-head path.
 */

#pragma once

#include "kernels/kvcache/AttentionKeyQ8Reference.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace llaminar2::test
{
    /** @brief Deterministic xorshift generator shared by device conformance tests. */
    class AttentionKeyQ8XorShift32
    {
    public:
        /** @brief Construct from a non-zero seed. */
        explicit AttentionKeyQ8XorShift32(uint32_t seed) : state_(seed) {}

        /** @brief Return the next exactly specified FP32 value in [-1, 1). */
        float symmetricUnit()
        {
            state_ ^= state_ << 13U;
            state_ ^= state_ >> 17U;
            state_ ^= state_ << 5U;
            return static_cast<float>(state_ >> 8U) * (1.0f / 8388608.0f) - 1.0f;
        }

    private:
        uint32_t state_; ///< Sole fixture-generator state.
    };

    /**
     * @brief Build model-scale attention keys for one physical head width.
     *
     * @tparam D Attention-head width.
     * @param block_count Number of complete heads to generate; must be positive.
     */
    template <int D>
    std::vector<float> makeAttentionKeyQ8DeviceInput(int block_count)
    {
        std::vector<float> input(static_cast<size_t>(block_count) * D);
        AttentionKeyQ8XorShift32 generator(0x51f15eU + static_cast<uint32_t>(D));
        for (float &value : input)
        {
            value = 4.0f * generator.symmetricUnit();
        }

        for (int block = 0; block < block_count - 1; ++block)
        {
            const size_t base = static_cast<size_t>(block) * D;
            const float outlier = 128.0f + 0.25f * static_cast<float>(block);
            input[base] = outlier;
            input[base + D / 2] = -0.8f * outlier;
        }

        // Exercise the dedicated all-zero encoding branch in the same launch.
        const size_t zero_base = static_cast<size_t>(block_count - 1) * D;
        std::fill(input.begin() + static_cast<std::ptrdiff_t>(zero_base), input.end(), 0.0f);
        return input;
    }

    /** @brief Quantize every fixture head with the authoritative scalar oracle. */
    template <int D>
    std::vector<AttentionKeyQ8Block<D>> makeAttentionKeyQ8ReferenceBlocks(
        const std::vector<float> &input)
    {
        const size_t block_count = input.size() / D;
        std::vector<AttentionKeyQ8Block<D>> blocks(block_count);
        for (size_t block = 0; block < block_count; ++block)
        {
            attentionKeyQ8QuantizeReference<D>(
                std::span<const float, D>(input.data() + block * D, D),
                blocks[block]);
        }
        return blocks;
    }

    /** @brief Decode every scalar reference block using the fixed oracle order. */
    template <int D>
    std::vector<float> makeAttentionKeyQ8ReferenceDecoded(
        const std::vector<AttentionKeyQ8Block<D>> &blocks)
    {
        std::vector<float> decoded(blocks.size() * D);
        for (size_t block = 0; block < blocks.size(); ++block)
        {
            attentionKeyQ8DequantizeReference<D>(
                blocks[block],
                std::span<float, D>(decoded.data() + block * D, D));
        }
        return decoded;
    }
} // namespace llaminar2::test
