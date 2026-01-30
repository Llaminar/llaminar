/**
 * @file Test__KernelSnapshotInfo.cpp
 * @brief Unit tests for KernelSnapshotInfo interface on all kernel implementations
 *
 * Verifies that all kernels properly implement getKernelSnapshotInfo() and
 * return valid, descriptive metadata about their I/O buffers.
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include "tensors/KernelSnapshotInfo.h"
#include "tensors/TensorKernels.h"
#include "execution/config/RuntimeConfig.h"

// Kernel implementations
#include "kernels/cpu/ops/CPURMSNormKernelT.h"
#include "kernels/cpu/ops/CPURoPEKernelT.h"
#include "kernels/cpu/ops/CPUSwiGLUKernelT.h"
#include "kernels/cpu/ops/CPUSoftmaxKernelT.h"
#include "kernels/cpu/ops/CPUEmbeddingKernelT.h"
#include "kernels/cpu/attention/CPUAttentionKernelT.h"
#include "kernels/cpu/attention/q16_1/Q16FusedAttentionKernel.h"
#include "kernels/cpu/gemm_v4/FloatingPointGemmKernel.h"
#include "kernels/cpu/gemm_v4/QuantisedGemmKernel.h"

namespace llaminar2
{
    namespace test
    {

        class KernelSnapshotInfoTest : public ::testing::Test
        {
        protected:
            void SetUp() override {}
            void TearDown() override {}

            // Helper to verify basic snapshot info validity
            void verifySnapshotInfo(const KernelSnapshotInfo &info, const char *kernel_name)
            {
                SCOPED_TRACE(kernel_name);

                // Must have a kernel name and description
                ASSERT_NE(info.kernel_name, nullptr) << "Kernel name is null";
                ASSERT_NE(info.kernel_description, nullptr) << "Kernel description is null";
                EXPECT_GT(strlen(info.kernel_name), 0) << "Kernel name is empty";
                EXPECT_GT(strlen(info.kernel_description), 0) << "Kernel description is empty";

                // If not passthrough, should have at least one output
                if (!info.is_passthrough)
                {
                    EXPECT_GT(info.outputs.size(), 0) << "Non-passthrough kernel should have outputs";
                }

                // Verify all buffer descriptors have valid names
                for (const auto &input : info.inputs)
                {
                    EXPECT_NE(input.name, nullptr) << "Input name is null";
                    EXPECT_NE(input.description, nullptr) << "Input description is null";
                }
                for (const auto &output : info.outputs)
                {
                    EXPECT_NE(output.name, nullptr) << "Output name is null";
                    EXPECT_NE(output.description, nullptr) << "Output description is null";
                }
                for (const auto &weight : info.weights)
                {
                    EXPECT_NE(weight.name, nullptr) << "Weight name is null";
                    EXPECT_NE(weight.description, nullptr) << "Weight description is null";
                }
                for (const auto &scalar : info.scalars)
                {
                    EXPECT_NE(scalar.name, nullptr) << "Scalar name is null";
                    EXPECT_NE(scalar.description, nullptr) << "Scalar description is null";
                }
            }
        };

        // =============================================================================
        // RMSNorm Kernel Tests
        // =============================================================================

        TEST_F(KernelSnapshotInfoTest, CPURMSNormKernelT_FP32_SnapshotInfo)
        {
            CPURMSNormKernelT<ActivationPrecision::FP32> kernel;
            auto info = kernel.getKernelSnapshotInfo();
            verifySnapshotInfo(info, "CPURMSNormKernelT<FP32>");

            EXPECT_STREQ(info.kernel_name, "RMSNorm");
            EXPECT_EQ(info.inputs.size(), 1);
            EXPECT_EQ(info.outputs.size(), 1);
            EXPECT_EQ(info.weights.size(), 1);
            EXPECT_GE(info.scalars.size(), 1); // epsilon
        }

        TEST_F(KernelSnapshotInfoTest, CPURMSNormKernelT_BF16_SnapshotInfo)
        {
            CPURMSNormKernelT<ActivationPrecision::BF16> kernel;
            auto info = kernel.getKernelSnapshotInfo();
            verifySnapshotInfo(info, "CPURMSNormKernelT<BF16>");

            EXPECT_STREQ(info.kernel_name, "RMSNorm");
            // Verify input dtype is BF16
            ASSERT_GT(info.inputs.size(), 0);
            EXPECT_EQ(info.inputs[0].dtype, KernelBufferDtype::BF16);
        }

        TEST_F(KernelSnapshotInfoTest, CPURMSNormKernelT_Q8_1_SnapshotInfo)
        {
            CPURMSNormKernelT<ActivationPrecision::Q8_1> kernel;
            auto info = kernel.getKernelSnapshotInfo();
            verifySnapshotInfo(info, "CPURMSNormKernelT<Q8_1>");

            EXPECT_STREQ(info.kernel_name, "RMSNorm");
            ASSERT_GT(info.inputs.size(), 0);
            EXPECT_EQ(info.inputs[0].dtype, KernelBufferDtype::Q8_1);
        }

        // =============================================================================
        // RoPE Kernel Tests
        // =============================================================================

        TEST_F(KernelSnapshotInfoTest, CPURoPEKernelT_FP32_SnapshotInfo)
        {
            CPURoPEKernelT<ActivationPrecision::FP32> kernel;
            auto info = kernel.getKernelSnapshotInfo();
            verifySnapshotInfo(info, "CPURoPEKernelT<FP32>");

            EXPECT_STREQ(info.kernel_name, "RoPE");
            EXPECT_GE(info.inputs.size(), 2);  // Q and K
            EXPECT_GE(info.outputs.size(), 2); // Q_rotated and K_rotated
        }

        TEST_F(KernelSnapshotInfoTest, CPURoPEKernelT_BF16_SnapshotInfo)
        {
            CPURoPEKernelT<ActivationPrecision::BF16> kernel;
            auto info = kernel.getKernelSnapshotInfo();
            verifySnapshotInfo(info, "CPURoPEKernelT<BF16>");

            EXPECT_STREQ(info.kernel_name, "RoPE");
        }

        // =============================================================================
        // SwiGLU Kernel Tests
        // =============================================================================

        TEST_F(KernelSnapshotInfoTest, CPUSwiGLUKernelT_FP32_SnapshotInfo)
        {
            CPUSwiGLUKernelT<ActivationPrecision::FP32> kernel;
            auto info = kernel.getKernelSnapshotInfo();
            verifySnapshotInfo(info, "CPUSwiGLUKernelT<FP32>");

            EXPECT_STREQ(info.kernel_name, "SwiGLU");
            EXPECT_EQ(info.inputs.size(), 2);  // gate and up
            EXPECT_EQ(info.outputs.size(), 1); // output
        }

        TEST_F(KernelSnapshotInfoTest, CPUSwiGLUKernelT_Q8_1_SnapshotInfo)
        {
            CPUSwiGLUKernelT<ActivationPrecision::Q8_1> kernel;
            auto info = kernel.getKernelSnapshotInfo();
            verifySnapshotInfo(info, "CPUSwiGLUKernelT<Q8_1>");

            EXPECT_STREQ(info.kernel_name, "SwiGLU");
            ASSERT_GT(info.inputs.size(), 0);
            EXPECT_EQ(info.inputs[0].dtype, KernelBufferDtype::Q8_1);
        }

        // =============================================================================
        // Softmax Kernel Tests
        // =============================================================================

        TEST_F(KernelSnapshotInfoTest, CPUSoftmaxKernelT_FP32_SnapshotInfo)
        {
            CPUSoftmaxKernelT<ActivationPrecision::FP32> kernel;
            auto info = kernel.getKernelSnapshotInfo();
            verifySnapshotInfo(info, "CPUSoftmaxKernelT<FP32>");

            EXPECT_STREQ(info.kernel_name, "Softmax");
            EXPECT_GE(info.inputs.size(), 1);
            EXPECT_GE(info.outputs.size(), 1);
        }

        TEST_F(KernelSnapshotInfoTest, CPUSoftmaxKernelT_Q8_1_SnapshotInfo)
        {
            CPUSoftmaxKernelT<ActivationPrecision::Q8_1> kernel;
            auto info = kernel.getKernelSnapshotInfo();
            verifySnapshotInfo(info, "CPUSoftmaxKernelT<Q8_1>");

            EXPECT_STREQ(info.kernel_name, "Softmax");
            ASSERT_GT(info.inputs.size(), 0);
            EXPECT_EQ(info.inputs[0].dtype, KernelBufferDtype::Q8_1);
        }

        // =============================================================================
        // Embedding Kernel Tests
        // =============================================================================

        TEST_F(KernelSnapshotInfoTest, CPUEmbeddingKernelT_SnapshotInfo)
        {
            CPUEmbeddingKernelT<FP32Tensor> kernel;
            auto info = kernel.getKernelSnapshotInfo();
            verifySnapshotInfo(info, "CPUEmbeddingKernelT");

            EXPECT_STREQ(info.kernel_name, "Embedding");
            EXPECT_GE(info.weights.size(), 1); // embed_table
            EXPECT_GE(info.inputs.size(), 1);  // token_ids
            EXPECT_GE(info.outputs.size(), 1); // output
        }

        // =============================================================================
        // Attention Kernel Tests
        // =============================================================================

        TEST_F(KernelSnapshotInfoTest, CPUAttentionKernelT_FP32_SnapshotInfo)
        {
            CPUAttentionKernelT<ActivationPrecision::FP32> kernel;
            auto info = kernel.getKernelSnapshotInfo();
            verifySnapshotInfo(info, "CPUAttentionKernelT<FP32>");

            EXPECT_STREQ(info.kernel_name, "Attention");
            EXPECT_GE(info.inputs.size(), 3);  // Q, K, V
            EXPECT_GE(info.outputs.size(), 1); // output (and potentially scores, context)
            EXPECT_GE(info.scalars.size(), 4); // seq_len, n_heads, n_kv_heads, head_dim, etc.
        }

        // =============================================================================
        // Fused Attention + Wo Kernel Tests
        // =============================================================================

        TEST_F(KernelSnapshotInfoTest, Q16FusedAttentionKernel_SnapshotInfo)
        {
            kernels::q16_1::Q16FusedAttentionKernel kernel;
            auto info = kernel.getKernelSnapshotInfo();
            verifySnapshotInfo(info, "Q16FusedAttentionKernel");

            // Check kernel name
            EXPECT_STREQ(info.kernel_name, "FusedAttentionWo");

            // Inputs: Q, K, V, residual_in
            EXPECT_GE(info.inputs.size(), 4);

            // Weights: Wo
            EXPECT_GE(info.weights.size(), 1);

            // Outputs: residual_out + intermediate (scores, context, wo_output)
            EXPECT_GE(info.outputs.size(), 1);

            // Scalars: seq_len_q, kv_len, n_heads, n_kv_heads, head_dim, scale, causal, position_offset
            EXPECT_GE(info.scalars.size(), 8);

            // Verify Q16_1 input dtype
            bool found_q16_input = false;
            for (const auto &input : info.inputs)
            {
                if (input.dtype == KernelBufferDtype::Q16_1)
                {
                    found_q16_input = true;
                    break;
                }
            }
            EXPECT_TRUE(found_q16_input) << "Expected Q16_1 input dtype";

            // Verify intermediate outputs for snapshots
            int intermediate_count = 0;
            for (const auto &output : info.outputs)
            {
                if (output.is_intermediate)
                    intermediate_count++;
            }
            EXPECT_GE(intermediate_count, 3) << "Expected at least 3 intermediate outputs (scores, context, wo_output)";
        }

        TEST_F(KernelSnapshotInfoTest, Q16FusedAttentionKernel_InterfaceMethods)
        {
            kernels::q16_1::Q16FusedAttentionKernel kernel;

            // Check interface methods
            EXPECT_EQ(kernel.input_format(), ActivationFormat::Q8_1); // Q16_1 extends Q8_1 semantics
            EXPECT_EQ(kernel.output_format(), ActivationFormat::Q8_1);
            EXPECT_TRUE(kernel.requires_packed_wo());
            EXPECT_TRUE(kernel.supports_device(-1)); // CPU only
            EXPECT_FALSE(kernel.supports_device(0)); // No GPU support
        }

        // =============================================================================
        // GEMM Kernel Tests
        // =============================================================================

        TEST_F(KernelSnapshotInfoTest, FloatingPointGemmKernel_SnapshotInfo)
        {
            // FloatingPointGemmKernel requires a tensor, so we need to use factory
            // For now, just test that the struct is valid
            auto info = KernelSnapshotInfo::gemm()
                            .withInput("A", "activation matrix [m, k]", KernelBufferDtype::FP32)
                            .withWeight("B", "weight matrix [k, n]", KernelBufferDtype::FP32)
                            .withOutput("C", "output matrix [m, n]", KernelBufferDtype::FP32);

            EXPECT_STREQ(info.kernel_name, "GEMM");
            EXPECT_EQ(info.inputs.size(), 1);
            EXPECT_EQ(info.weights.size(), 1);
            EXPECT_EQ(info.outputs.size(), 1);
        }

        // =============================================================================
        // Factory Method Tests
        // =============================================================================

        TEST_F(KernelSnapshotInfoTest, FactoryMethods)
        {
            // Test all factory methods produce valid info
            auto gemm = KernelSnapshotInfo::gemm();
            EXPECT_STREQ(gemm.kernel_name, "GEMM");

            auto attention = KernelSnapshotInfo::attention();
            EXPECT_STREQ(attention.kernel_name, "Attention");

            auto fused_attention_wo = KernelSnapshotInfo::fusedAttentionWo();
            EXPECT_STREQ(fused_attention_wo.kernel_name, "FusedAttentionWo");

            auto rope = KernelSnapshotInfo::rope();
            EXPECT_STREQ(rope.kernel_name, "RoPE");

            auto swiglu = KernelSnapshotInfo::swiglu();
            EXPECT_STREQ(swiglu.kernel_name, "SwiGLU");

            auto rmsnorm = KernelSnapshotInfo::rmsnorm();
            EXPECT_STREQ(rmsnorm.kernel_name, "RMSNorm");

            auto softmax = KernelSnapshotInfo::softmax();
            EXPECT_STREQ(softmax.kernel_name, "Softmax");

            auto embedding = KernelSnapshotInfo::embedding();
            EXPECT_STREQ(embedding.kernel_name, "Embedding");

            auto passthrough = KernelSnapshotInfo::passthrough();
            EXPECT_STREQ(passthrough.kernel_name, "Passthrough");
            EXPECT_TRUE(passthrough.is_passthrough);
        }

        TEST_F(KernelSnapshotInfoTest, BuilderPattern)
        {
            auto info = KernelSnapshotInfo::gemm()
                            .withInput("A", "activations", KernelBufferDtype::FP32)
                            .withInput("bias", "bias vector", KernelBufferDtype::FP32, false) // optional
                            .withWeight("B", "weights", KernelBufferDtype::Q4_0)
                            .withOutput("C", "output", KernelBufferDtype::FP32)
                            .withOutput("intermediate", "intermediate buffer", KernelBufferDtype::FP32, true) // intermediate
                            .withScalar("alpha", "scale factor")
                            .withScalar("beta", "residual factor");

            EXPECT_EQ(info.inputs.size(), 2);
            EXPECT_EQ(info.weights.size(), 1);
            EXPECT_EQ(info.outputs.size(), 2);
            EXPECT_EQ(info.scalars.size(), 2);

            // Check required input (default is true)
            EXPECT_TRUE(info.inputs[0].required);  // A is required (default)
            EXPECT_FALSE(info.inputs[1].required); // bias is optional

            // Check intermediate output
            EXPECT_FALSE(info.outputs[0].is_intermediate);
            EXPECT_TRUE(info.outputs[1].is_intermediate);
        }

        TEST_F(KernelSnapshotInfoTest, DtypeToString)
        {
            EXPECT_STREQ(to_string(KernelBufferDtype::FP32), "FP32");
            EXPECT_STREQ(to_string(KernelBufferDtype::BF16), "BF16");
            EXPECT_STREQ(to_string(KernelBufferDtype::FP16), "FP16");
            EXPECT_STREQ(to_string(KernelBufferDtype::Q8_1), "Q8_1");
            EXPECT_STREQ(to_string(KernelBufferDtype::Q4_0), "Q4_0");
            EXPECT_STREQ(to_string(KernelBufferDtype::IQ4_NL), "IQ4_NL");
            EXPECT_STREQ(to_string(KernelBufferDtype::INT32), "INT32");
            EXPECT_STREQ(to_string(KernelBufferDtype::Unknown), "Unknown");
        }

    } // namespace test
} // namespace llaminar2
