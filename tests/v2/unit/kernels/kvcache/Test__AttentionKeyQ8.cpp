/**
 * @file Test__AttentionKeyQ8.cpp
 * @brief Device-free numerical contract tests for attention-key Q8.
 *
 * These tests gate the physical layout, byte determinism, invalid-input
 * handling, and downstream causal-attention quality. The attention regression
 * uses deterministic Qwen-scale geometry and outliers; it is intentionally a
 * context-output comparison rather than a vector-only round-trip test.
 */

#include <gtest/gtest.h>

#include "kernels/kvcache/AttentionKeyQ8Reference.h"
#include "../../../utils/AttentionKeyOutlierFixture.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace llaminar2
{
    namespace
    {
        using QualityFixture = test::AttentionKeyOutlierFixture;
        constexpr int kHeadDim = QualityFixture::head_dim;
        constexpr int kTokens = QualityFixture::tokens;
        constexpr int kKVHeads = QualityFixture::kv_heads;
        constexpr int kQueryHeads = QualityFixture::query_heads;
        constexpr int kQueriesPerKVHead = kQueryHeads / kKVHeads;

        /** @brief Return the flat offset for one token/head/coordinate tuple. */
        constexpr size_t offset(int token, int head, int coordinate, int heads)
        {
            return (static_cast<size_t>(token) * heads + head) * kHeadDim + coordinate;
        }

        /** @brief Quantize one 32-coordinate block with the former linear Q8 policy. */
        void quantizeLinearQ8Block(const float *input, float *output)
        {
            constexpr int kBlock = 32;
            float max_abs = 0.0f;
            for (int coordinate = 0; coordinate < kBlock; ++coordinate)
            {
                max_abs = std::max(max_abs, std::abs(input[coordinate]));
            }
            if (max_abs == 0.0f)
            {
                std::fill(output, output + kBlock, 0.0f);
                return;
            }

            const float scale = max_abs / 127.0f;
            for (int coordinate = 0; coordinate < kBlock; ++coordinate)
            {
                const int code = std::clamp(
                    static_cast<int>(std::round(input[coordinate] / scale)), -127, 127);
                output[coordinate] = scale * static_cast<float>(code);
            }
        }

        /** @brief Stable softmax attention for one query and its causal prefix. */
        std::array<float, kHeadDim> attend(
            const float *query,
            const std::vector<float> &keys,
            const std::vector<float> &values,
            int query_token,
            int kv_head)
        {
            std::array<float, kTokens> scores{};
            float max_score = -std::numeric_limits<float>::infinity();
            for (int key_token = 0; key_token <= query_token; ++key_token)
            {
                float dot = 0.0f;
                const float *key = keys.data() + offset(key_token, kv_head, 0, kKVHeads);
                for (int coordinate = 0; coordinate < kHeadDim; ++coordinate)
                {
                    dot += query[coordinate] * key[coordinate];
                }
                scores[key_token] = dot / std::sqrt(static_cast<float>(kHeadDim));
                max_score = std::max(max_score, scores[key_token]);
            }

            float denominator = 0.0f;
            for (int key_token = 0; key_token <= query_token; ++key_token)
            {
                scores[key_token] = std::exp(scores[key_token] - max_score);
                denominator += scores[key_token];
            }

            std::array<float, kHeadDim> context{};
            for (int key_token = 0; key_token <= query_token; ++key_token)
            {
                const float probability = scores[key_token] / denominator;
                const float *value = values.data() + offset(key_token, kv_head, 0, kKVHeads);
                for (int coordinate = 0; coordinate < kHeadDim; ++coordinate)
                {
                    context[coordinate] += probability * value[coordinate];
                }
            }
            return context;
        }

        /** @brief Compute aggregate cosine similarity for equally sized vectors. */
        double cosine(const std::vector<float> &expected, const std::vector<float> &actual)
        {
            EXPECT_EQ(expected.size(), actual.size());
            double dot = 0.0;
            double expected_norm = 0.0;
            double actual_norm = 0.0;
            for (size_t index = 0; index < expected.size(); ++index)
            {
                dot += static_cast<double>(expected[index]) * actual[index];
                expected_norm += static_cast<double>(expected[index]) * expected[index];
                actual_norm += static_cast<double>(actual[index]) * actual[index];
            }
            return dot / std::sqrt(expected_norm * actual_norm);
        }

        /** @brief Compute relative L2 error for equally sized vectors. */
        double relativeL2(const std::vector<float> &expected, const std::vector<float> &actual)
        {
            EXPECT_EQ(expected.size(), actual.size());
            double error = 0.0;
            double expected_norm = 0.0;
            for (size_t index = 0; index < expected.size(); ++index)
            {
                const double difference =
                    static_cast<double>(actual[index]) - expected[index];
                error += difference * difference;
                expected_norm += static_cast<double>(expected[index]) * expected[index];
            }
            return std::sqrt(error / expected_norm);
        }

        /** @brief Run every causal query head and concatenate its context output. */
        std::vector<float> allContexts(
            const std::vector<float> &queries,
            const std::vector<float> &keys,
            const std::vector<float> &values)
        {
            std::vector<float> contexts;
            contexts.reserve(kTokens * kQueryHeads * kHeadDim);
            for (int token = 0; token < kTokens; ++token)
            {
                for (int query_head = 0; query_head < kQueryHeads; ++query_head)
                {
                    const int kv_head = query_head / kQueriesPerKVHead;
                    const float *query =
                        queries.data() + offset(token, query_head, 0, kQueryHeads);
                    const auto context = attend(query, keys, values, token, kv_head);
                    contexts.insert(contexts.end(), context.begin(), context.end());
                }
            }
            return contexts;
        }
    } // namespace

    TEST(Test__AttentionKeyQ8, PhysicalLayoutIsBlobTransferable)
    {
        EXPECT_EQ(sizeof(AttentionKeyQ8Block_64), 68U);
        EXPECT_EQ(sizeof(AttentionKeyQ8Block_128), 132U);
        EXPECT_EQ(sizeof(AttentionKeyQ8Block_256), 260U);
        EXPECT_TRUE(std::is_trivially_copyable_v<AttentionKeyQ8Block_64>);
        EXPECT_TRUE(std::is_trivially_copyable_v<AttentionKeyQ8Block_128>);
        EXPECT_TRUE(std::is_trivially_copyable_v<AttentionKeyQ8Block_256>);
    }

    TEST(Test__AttentionKeyQ8, ZeroAndExtremaRoundTripDeterministically)
    {
        std::array<float, kHeadDim> input{};
        AttentionKeyQ8Block_64 first{};
        AttentionKeyQ8Block_64 second{};
        std::array<float, kHeadDim> output{};

        attentionKeyQ8QuantizeReference<kHeadDim>(input, first);
        attentionKeyQ8QuantizeReference<kHeadDim>(input, second);
        attentionKeyQ8DequantizeReference<kHeadDim>(first, output);
        EXPECT_EQ(std::memcmp(&first, &second, sizeof(first)), 0);
        EXPECT_EQ(first.quadratic_scale, 0.0f);
        EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](float value)
                                { return value == 0.0f; }));

        input[0] = 128.0f;
        input[1] = -128.0f;
        attentionKeyQ8QuantizeReference<kHeadDim>(input, first);
        attentionKeyQ8DequantizeReference<kHeadDim>(first, output);
        EXPECT_EQ(first.codes[0], 127);
        EXPECT_EQ(first.codes[1], -127);
        EXPECT_FLOAT_EQ(output[0], 128.0f);
        EXPECT_FLOAT_EQ(output[1], -128.0f);
    }

    TEST(Test__AttentionKeyQ8, RationalThresholdSearchMatchesQuadraticRounding)
    {
        constexpr float kMaxAbs = 128.0f;
        // Probe a dense, deterministic grid plus every exact representable
        // source code. Points exactly on a decision boundary are intentionally
        // covered separately by the physical-byte backend tests.
        for (int sample = 0; sample <= 10000; ++sample)
        {
            const float magnitude =
                kMaxAbs * (static_cast<float>(sample) / 10000.0f);
            const int direct = static_cast<int>(std::round(
                std::sqrt(magnitude / kMaxAbs) * 127.0f));
            const int rational = static_cast<int>(
                attentionKeyQ8MagnitudeCodeReference(magnitude, kMaxAbs));
            EXPECT_EQ(rational, direct) << "sample=" << sample;
        }

        for (int code = 0; code <= 127; ++code)
        {
            const float normalized = static_cast<float>(code) / 127.0f;
            const float magnitude = kMaxAbs * normalized * normalized;
            EXPECT_EQ(attentionKeyQ8MagnitudeCodeReference(magnitude, kMaxAbs), code)
                << "code=" << code;
        }
    }

    TEST(Test__AttentionKeyQ8, RejectsNonFiniteInputAndCorruptPhysicalState)
    {
        std::array<float, kHeadDim> input{};
        input[7] = std::numeric_limits<float>::quiet_NaN();
        AttentionKeyQ8Block_64 block{};
        EXPECT_THROW(attentionKeyQ8QuantizeReference<kHeadDim>(input, block),
                     std::domain_error);

        std::array<float, kHeadDim> output{};
        block.quadratic_scale = 1.0f;
        block.codes[3] = -128;
        EXPECT_THROW(attentionKeyQ8DequantizeReference<kHeadDim>(block, output),
                     std::domain_error);
    }

    TEST(Test__AttentionKeyQ8, AnchoredQwenScaleCausalAttentionPreservesContext)
    {
        const test::AttentionKeyOutlierFixture fixture;
        const auto &keys = fixture.keys;
        const auto &queries = fixture.queries;
        const auto &values = fixture.values;

        std::vector<float> linear_keys(keys.size());
        std::vector<float> attention_q8_keys(keys.size());
        std::array<float, kKVHeads * kHeadDim> anchors{};
        for (int head = 0; head < kKVHeads; ++head)
        {
            for (int coordinate = 0; coordinate < kHeadDim; ++coordinate)
            {
                float sum = 0.0f;
                for (int token = 0; token < kTokens; ++token)
                {
                    sum += keys[offset(token, head, coordinate, kKVHeads)];
                }
                anchors[static_cast<size_t>(head) * kHeadDim + coordinate] =
                    sum / static_cast<float>(kTokens);
            }
        }

        for (int token = 0; token < kTokens; ++token)
        {
            for (int head = 0; head < kKVHeads; ++head)
            {
                const float *source = keys.data() + offset(token, head, 0, kKVHeads);
                float *linear =
                    linear_keys.data() + offset(token, head, 0, kKVHeads);
                quantizeLinearQ8Block(source, linear);
                quantizeLinearQ8Block(source + 32, linear + 32);

                std::array<float, kHeadDim> residual{};
                for (int coordinate = 0; coordinate < kHeadDim; ++coordinate)
                {
                    residual[coordinate] =
                        source[coordinate] -
                        anchors[static_cast<size_t>(head) * kHeadDim + coordinate];
                }
                AttentionKeyQ8Block_64 block{};
                attentionKeyQ8QuantizeReference<kHeadDim>(
                    residual, block);
                float *decoded =
                    attention_q8_keys.data() + offset(token, head, 0, kKVHeads);
                attentionKeyQ8DequantizeReference<kHeadDim>(
                    block, std::span<float, kHeadDim>(decoded, kHeadDim));
                for (int coordinate = 0; coordinate < kHeadDim; ++coordinate)
                {
                    decoded[coordinate] +=
                        anchors[static_cast<size_t>(head) * kHeadDim + coordinate];
                }
            }
        }

        const std::vector<float> reference_contexts = allContexts(queries, keys, values);
        const std::vector<float> linear_contexts =
            allContexts(queries, linear_keys, values);
        const std::vector<float> attention_q8_contexts =
            allContexts(queries, attention_q8_keys, values);

        const double linear_cosine = cosine(reference_contexts, linear_contexts);
        const double attention_q8_cosine =
            cosine(reference_contexts, attention_q8_contexts);
        const double attention_q8_relative_l2 =
            relativeL2(reference_contexts, attention_q8_contexts);

        // This guard keeps the fixture diagnostic: if somebody accidentally
        // makes it benign, it must be replaced with a scale-stress case that
        // still reproduces the historical Q8 key failure.
        EXPECT_LT(linear_cosine, 0.94);
        EXPECT_GT(attention_q8_cosine, 0.995);
        EXPECT_LT(attention_q8_relative_l2, 0.06);
    }
} // namespace llaminar2
