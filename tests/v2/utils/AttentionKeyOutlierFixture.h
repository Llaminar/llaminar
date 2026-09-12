/**
 * @file AttentionKeyOutlierFixture.h
 * @brief Shared Qwen-scale key-outlier workload for codec and live-cache tests.
 *
 * Large coordinates shared by all keys cancel in softmax. A key codec must
 * still preserve the smaller coordinates that distinguish tokens. Sharing
 * this input between the scalar oracle and production append/attention tests
 * prevents certifying a codec without certifying that the backend uses it.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llaminar2::test
{
/** @brief Fully specified, model-free attention inputs with key scale skew. */
struct AttentionKeyOutlierFixture
{
    static constexpr int head_dim = 64;
    static constexpr int tokens = 9;
    static constexpr int kv_heads = 2;
    static constexpr int query_heads = 14;

    std::vector<float> keys = std::vector<float>(tokens * kv_heads * head_dim);
    std::vector<float> queries = std::vector<float>(tokens * query_heads * head_dim);
    std::vector<float> values = std::vector<float>(tokens * kv_heads * head_dim);

    /** @brief Position-major coordinate offset, independent of any backend. */
    static constexpr std::size_t offset(int token, int head, int coordinate, int heads)
    {
        return (static_cast<std::size_t>(token) * heads + head) * head_dim + coordinate;
    }

    /** @brief Construct exactly the historical scalar key-quality regression. */
    AttentionKeyOutlierFixture()
    {
        // Specify the RNG arithmetic rather than relying on library-dependent
        // random distributions. Distinct seeds prevent query/value correlation.
        const auto fill = [](std::vector<float> &target, uint32_t state, float scale)
        {
            for (float &value : target)
            {
                state ^= state << 13U;
                state ^= state >> 17U;
                state ^= state << 5U;
                value = scale * (static_cast<float>(state >> 8U) *
                                     (1.0f / 8388608.0f) - 1.0f);
            }
        };
        fill(keys, 2U, 1.0f);
        fill(queries, 2U ^ 0x9e3779b9U, 8.0f);
        fill(values, 2U ^ 0x243f6a88U, 1.0f);
        for (int token = 0; token < tokens; ++token)
        {
            for (int head = 0; head < kv_heads; ++head)
            {
                keys[offset(token, head, 0, kv_heads)] = 128.0f;
                keys[offset(token, head, 32, kv_heads)] = -102.4f;
            }
            for (int head = 0; head < query_heads; ++head)
            {
                queries[offset(token, head, 0, query_heads)] = 0.0f;
                queries[offset(token, head, 32, query_heads)] = 0.0f;
            }
        }
    }
};
} // namespace llaminar2::test
