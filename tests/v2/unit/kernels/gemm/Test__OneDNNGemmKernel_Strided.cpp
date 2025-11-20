#include "gtest/gtest.h"
#include "kernels/cpu/gemm_v4/OneDNNGemmKernel.h"
#include <vector>
#include <cmath>
#include <limits>

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

namespace
{
    class OneDNNGemmKernelStridedTest : public ::testing::Test
    {
    protected:
        OneDNNGemmKernel kernel;

        void SetUp() override
        {
            // No setup needed
        }
    };

    TEST_F(OneDNNGemmKernelStridedTest, FusedSoftmaxStrided_FP32_Causal)
    {
        // Test case: 2 heads, seq_len=4, head_dim=4
        // We simulate Q @ K^T -> Scores
        // Q: [2, 4, 4] (n_heads, seq_len, head_dim)
        // K: [2, 4, 4]
        // Scores: [2, 4, 4] (n_heads, seq_len, seq_len)

        int n_heads = 2;
        int seq_len = 4;
        int head_dim = 4;
        float scale = 1.0f;

        // Initialize Q and K with identity-like patterns to make result predictable
        std::vector<float> Q(n_heads * seq_len * head_dim);
        std::vector<float> K(n_heads * seq_len * head_dim);
        std::vector<float> scores(n_heads * seq_len * seq_len);

        for (int i = 0; i < Q.size(); ++i)
            Q[i] = 1.0f;
        for (int i = 0; i < K.size(); ++i)
            K[i] = 1.0f;

        // Strides
        int lda = n_heads * head_dim; // Stride between rows of Q (if Q is [seq_len, n_heads, head_dim])
        // Wait, CpuAttentionKernelT uses:
        // Q_h = Q + h * head_dim
        // lda = n_heads * head_dim
        // This implies Q is [seq_len, n_heads, head_dim] layout.
        // Let's match that.

        // Fill Q and K such that Q[s, h, :] and K[s, h, :] are distinct
        for (int s = 0; s < seq_len; ++s)
        {
            for (int h = 0; h < n_heads; ++h)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    Q[(s * n_heads + h) * head_dim + d] = (s == d) ? 1.0f : 0.0f;
                    K[(s * n_heads + h) * head_dim + d] = (s == d) ? 1.0f : 0.0f;
                }
            }
        }

        // Run for each head
        for (int h = 0; h < n_heads; ++h)
        {
            const float *Q_h = Q.data() + h * head_dim;
            const float *K_h = K.data() + h * head_dim;
            float *scores_h = scores.data() + h * seq_len * seq_len;

            bool success = kernel.multiply_with_softmax_strided_typed<float, float>(
                Q_h, K_h, scores_h,
                seq_len, seq_len, head_dim,
                lda, lda, seq_len,
                scale,
                true,    // transpose_B
                1,       // softmax_axis
                nullptr, // mask
                true,    // is_causal
                nullptr, -1, ActivationFormat::FP32);
            ASSERT_TRUE(success);
        }

        // Verify results
        // Q @ K^T should be identity matrix (since Q=I, K=I)
        // Causal mask should mask upper triangle (j > i)
        // Softmax should be applied row-wise

        for (int h = 0; h < n_heads; ++h)
        {
            float *scores_h = scores.data() + h * seq_len * seq_len;
            for (int i = 0; i < seq_len; ++i)
            {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; ++j)
                {
                    float val = scores_h[i * seq_len + j];
                    sum += val;
                    if (j > i)
                    {
                        EXPECT_EQ(val, 0.0f) << "Head " << h << " pos " << i << "," << j << " should be masked (0 after softmax)";
                    }
                }
                EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Head " << h << " row " << i << " sum should be 1.0";
            }
        }
    }
}
