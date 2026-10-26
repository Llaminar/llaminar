/**
 * @file Test__CPUGDNRepeatedHeadPreprocessing.cpp
 * @brief Byte authentication of shared mapped Q/K normalization and output writes.
 *
 * Merged QKV repeats source heads while value heads own independent gates and
 * recurrence. Explicitly expanded separate inputs provide the oracle for offset
 * mapping, reduction rounding, full output publication, and checkpoint history.
 */
#include "kernels/cpu/gdn/CPUGatedDeltaNet.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <vector>

TEST(CPUGDNRepeatedHeadPreprocessing, MappedHeadsMatchExpandedInputsAndContinuation)
{
    constexpr int rows = 9, heads = 5, key_heads = 2, dv = 9;
    for (const int dk : {1, 7, 17, 65, 128, 257})
        for (const int offset : {-5, -1, 0, 1, 4})
            for (const bool normalize : {false, true})
            {
                SCOPED_TRACE(::testing::Message() << "dk=" << dk << " offset=" << offset
                    << " normalize=" << normalize);
                const int stride = 2 * key_heads * dk + heads * dv + 3;
                std::vector<float> merged(rows * stride), q(rows * heads * dk), k(q.size()),
                    v(rows * heads * dv), alpha(rows * heads), beta(alpha.size()),
                    a(heads), bias(heads), actual(v.size(), -1234.0f), expected(v.size(), 9876.0f);
                const llaminar2::CPUGDNStateLayout layout(heads, dk, dv);
                std::vector<float> state(layout.floats(), 0.0f), reference_state = state;
                for (int i = 0; i < rows * stride; ++i)
                    merged[i] = static_cast<float>((i * 37) % 101 - 50) / 128.0f;
                for (int t = 0; t < rows; ++t)
                    for (int h = 0; h < heads; ++h)
                    {
                        int source = (h + offset) % key_heads;
                        if (source < 0) source += key_heads;
                        const float *raw = merged.data() + t * stride;
                        std::memcpy(q.data() + (t * heads + h) * dk, raw + source * dk, dk * sizeof(float));
                        std::memcpy(k.data() + (t * heads + h) * dk, raw + (key_heads + source) * dk, dk * sizeof(float));
                        std::memcpy(v.data() + (t * heads + h) * dv, raw + 2 * key_heads * dk + h * dv, dv * sizeof(float));
                        alpha[t * heads + h] = static_cast<float>((t * 7 + h * 3) % 19 - 9) / 16.0f;
                        beta[t * heads + h] = static_cast<float>((t * 3 + h * 7) % 17 - 8) / 32.0f;
                    }
                for (int h = 0; h < heads; ++h)
                {
                    a[h] = -0.2f - h * 0.07f;
                    bias[h] = 0.01f * h;
                }
                llaminar2::CPUGatedDeltaNet compact, expanded;
                // A second call exercises pending history through another epoch,
                // while different poison values detect incomplete output writes.
                for (int invocation = 0; invocation < 2; ++invocation)
                {
                    std::fill(actual.begin(), actual.end(), -1234.0f);
                    std::fill(expected.begin(), expected.end(), 9876.0f);
                    ASSERT_TRUE(compact.chunkForwardMergedQKV(merged.data(), stride,
                        alpha.data(), beta.data(), a.data(), bias.data(), actual.data(), state.data(),
                        rows, key_heads, heads, dk, dv, offset, 8, normalize));
                    ASSERT_TRUE(expanded.chunk_forward(q.data(), k.data(), v.data(),
                        alpha.data(), beta.data(), a.data(), bias.data(), expected.data(), reference_state.data(),
                        rows, heads, dk, dv, 8, normalize));
                    ASSERT_EQ(0, std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)));
                    ASSERT_EQ(0, std::memcmp(state.data(), reference_state.data(), state.size() * sizeof(float)));
                }
            }
}
