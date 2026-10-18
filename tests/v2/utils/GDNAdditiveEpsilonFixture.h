/**
 * @file GDNAdditiveEpsilonFixture.h
 * @brief Independent closed-form low-norm operands for the Qwen/FLA GDN equation.
 *
 * Only row zero has nonzero Q/K. With zero initial state, beta_raw=0 and A=0,
 * the exact first update is S[0,0]=0.5*a/sqrt(a*a+epsilon). Later rows leave it
 * unchanged and emit zero, independently of the native recurrence schedule.
 * Four heads distinguish zero, tiny, epsilon-scale and ordinary operands.
 */
#pragma once

#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <vector>

namespace llaminar2::test
{
    /** @brief Own original operands and an equation-derived state/output oracle. */
    struct GDNAdditiveEpsilonFixture
    {
        static constexpr int heads = 4, dv = 128;
        static constexpr std::array<float, heads> amplitudes{0.f, 1e-8f, 1e-3f, 1.f};
        int rows, dk;
        std::vector<float> q, k, v, alpha, beta, a_log, dt_bias, zero_state;

        /**
         * @brief Build sparse original operands with zero initial recurrent state.
         * @param m Live rows.
         * @param width Key width.
         */
        GDNAdditiveEpsilonFixture(int m, int width)
            : rows(m), dk(width), q(m * heads * width, 0.f), k(q.size(), 0.f),
              v(m * heads * dv, 0.f), alpha(m * heads, 0.f), beta(alpha.size(), 0.f),
              a_log(heads, 0.f), dt_bias(heads, 0.f), zero_state(heads * width * dv, 0.f)
        {
            for (int h = 0; h < heads; ++h)
            {
                q[h * dk] = k[h * dk] = amplitudes[h];
                v[h * dv] = 1.f;
            }
        }

        /**
         * @brief Check every output/state element against the independent closed form.
         * @param output Actual live rows.
         * @param state Actual terminal recurrence state.
         */
        void expect(const std::vector<float> &output, const std::vector<float> &state) const
        {
            ASSERT_EQ(output.size(), static_cast<size_t>(rows * heads * dv));
            ASSERT_EQ(state.size(), zero_state.size());
            for (int h = 0; h < heads; ++h)
            {
                const double a = amplitudes[h];
                const double normalized = a / std::sqrt(a * a + static_cast<double>(1e-6f));
                const double expected_state = 0.5 * normalized;
                const double expected_output = 0.5 * normalized * normalized / std::sqrt(double(dk));
                for (int row = 0; row < rows; ++row)
                {
                    for (int col = 0; col < dv; ++col)
                    {
                        const float actual = output[(row * heads + h) * dv + col];
                        const double wanted = (row == 0 && col == 0) ? expected_output : 0.;
                        EXPECT_TRUE(std::isfinite(actual));
                        EXPECT_NEAR(actual, wanted, 2e-7 + std::abs(wanted) * 2e-5)
                            << "head=" << h << " row=" << row << " col=" << col;
                    }
                }
                for (int key = 0; key < dk; ++key)
                {
                    for (int col = 0; col < dv; ++col)
                    {
                        const float actual = state[(h * dk + key) * dv + col];
                        const double wanted = (key == 0 && col == 0) ? expected_state : 0.;
                        EXPECT_TRUE(std::isfinite(actual));
                        EXPECT_NEAR(actual, wanted, 2e-7 + std::abs(wanted) * 2e-5)
                            << "head=" << h << " key=" << key << " col=" << col;
                    }
                }
            }
        }
    };
} // namespace llaminar2::test
