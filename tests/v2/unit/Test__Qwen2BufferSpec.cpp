/**
 * @file Test__Qwen2BufferSpec.cpp
 * @brief Unit tests for Qwen2BufferSpec
 * @author David Sanftenberg
 * @date January 2025
 */

#include <gtest/gtest.h>
#include "../../src/v2/pipelines/qwen/Qwen2BufferSpec.h"

namespace llaminar2
{
    namespace test
    {

        // =========================================================================
        // Test Fixture
        // =========================================================================

        class Qwen2BufferSpecTest : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                // Qwen2.5-0.5B-like config
                builder_ = std::make_unique<Qwen2BufferSpecBuilder>(
                    896,   // d_model
                    14,    // n_heads
                    2,     // n_kv_heads
                    64,    // head_dim
                    4864,  // d_ff
                    151936 // vocab_size
                );
            }

            std::unique_ptr<Qwen2BufferSpecBuilder> builder_;
        };

        // =========================================================================
        // Construction Tests
        // =========================================================================

        TEST_F(Qwen2BufferSpecTest, ConstructWithValidConfig)
        {
            EXPECT_NE(builder_, nullptr);
        }

        TEST_F(Qwen2BufferSpecTest, ConstructWithInvalidDModel)
        {
            EXPECT_THROW(
                Qwen2BufferSpecBuilder(0, 14, 2, 64, 4864, 151936),
                std::invalid_argument);
        }

        TEST_F(Qwen2BufferSpecTest, ConstructWithInvalidHeads)
        {
            EXPECT_THROW(
                Qwen2BufferSpecBuilder(896, 0, 2, 64, 4864, 151936),
                std::invalid_argument);
        }

        // =========================================================================
        // Layer Spec Tests
        // =========================================================================

        TEST_F(Qwen2BufferSpecTest, BuildLayerSpecsCreatesAllBuffers)
        {
            auto specs = builder_->buildLayerSpecs(128);

            // Count by role
            size_t scratch_count = 0;
            size_t inout_count = 0;

            for (const auto &spec : specs)
            {
                if (spec.role == BufferRole::SCRATCH)
                    ++scratch_count;
                else if (spec.role == BufferRole::INOUT)
                    ++inout_count;
            }

            // Should have INOUT buffers: residual, normalized
            EXPECT_GE(inout_count, 2u);

            // Should have SCRATCH buffers: Q, K, V, attn_output, attn_proj,
            // workspace_scores, workspace_context, workspace_mask,
            // gate, up, ffn_output
            EXPECT_GE(scratch_count, 11u);
        }

        TEST_F(Qwen2BufferSpecTest, BuildLayerSpecsHasCorrectShapes)
        {
            int seq_len = 128;
            auto specs = builder_->buildLayerSpecs(seq_len);

            // Find Q buffer
            const Qwen2BufferSpec *q_spec = nullptr;
            for (const auto &spec : specs)
            {
                if (spec.name == BufferNames::Q)
                {
                    q_spec = &spec;
                    break;
                }
            }

            ASSERT_NE(q_spec, nullptr);
            EXPECT_EQ(q_spec->shape.size(), 2u);
            EXPECT_EQ(q_spec->shape[0], static_cast<size_t>(seq_len)); // seq_len
            EXPECT_EQ(q_spec->shape[1], 14u * 64u);                    // n_heads * head_dim = 896
        }

        TEST_F(Qwen2BufferSpecTest, BuildLayerSpecsFFNShapes)
        {
            int seq_len = 64;
            auto specs = builder_->buildLayerSpecs(seq_len);

            // Find gate buffer
            const Qwen2BufferSpec *gate_spec = nullptr;
            for (const auto &spec : specs)
            {
                if (spec.name == BufferNames::GATE)
                {
                    gate_spec = &spec;
                    break;
                }
            }

            ASSERT_NE(gate_spec, nullptr);
            EXPECT_EQ(gate_spec->shape.size(), 2u);
            EXPECT_EQ(gate_spec->shape[0], static_cast<size_t>(seq_len));
            EXPECT_EQ(gate_spec->shape[1], 4864u); // d_ff
        }

        // =========================================================================
        // Model Spec Tests
        // =========================================================================

        TEST_F(Qwen2BufferSpecTest, BuildModelSpecsHasLogits)
        {
            auto specs = builder_->buildModelSpecs(1, 128);

            const Qwen2BufferSpec *logits_spec = nullptr;
            for (const auto &spec : specs)
            {
                if (spec.name == BufferNames::LOGITS)
                {
                    logits_spec = &spec;
                    break;
                }
            }

            ASSERT_NE(logits_spec, nullptr);
            EXPECT_EQ(logits_spec->role, BufferRole::OUTPUT);
            EXPECT_EQ(logits_spec->shape[0], 128u);    // batch * seq_len
            EXPECT_EQ(logits_spec->shape[1], 151936u); // vocab_size
        }

        TEST_F(Qwen2BufferSpecTest, BuildModelSpecsBatchedShape)
        {
            auto specs = builder_->buildModelSpecs(4, 32);

            const Qwen2BufferSpec *hidden_spec = nullptr;
            for (const auto &spec : specs)
            {
                if (spec.name == BufferNames::CURRENT_HIDDEN)
                {
                    hidden_spec = &spec;
                    break;
                }
            }

            ASSERT_NE(hidden_spec, nullptr);
            EXPECT_EQ(hidden_spec->shape[0], 4u * 32u); // batch * seq_len
            EXPECT_EQ(hidden_spec->shape[1], 896u);     // d_model
        }

        // =========================================================================
        // Attention Spec Tests
        // =========================================================================

        TEST_F(Qwen2BufferSpecTest, BuildAttentionSpecsKVHeads)
        {
            auto specs = builder_->buildAttentionSpecs(64);

            // K and V should use n_kv_heads, not n_heads (GQA)
            const Qwen2BufferSpec *k_spec = nullptr;
            const Qwen2BufferSpec *v_spec = nullptr;

            for (const auto &spec : specs)
            {
                if (spec.name == BufferNames::K)
                    k_spec = &spec;
                if (spec.name == BufferNames::V)
                    v_spec = &spec;
            }

            ASSERT_NE(k_spec, nullptr);
            ASSERT_NE(v_spec, nullptr);

            // n_kv_heads * head_dim = 2 * 64 = 128
            EXPECT_EQ(k_spec->shape[1], 2u * 64u);
            EXPECT_EQ(v_spec->shape[1], 2u * 64u);
        }

        // =========================================================================
        // FFN Spec Tests
        // =========================================================================

        TEST_F(Qwen2BufferSpecTest, BuildFFNSpecsAllScratch)
        {
            auto specs = builder_->buildFFNSpecs(32);

            for (const auto &spec : specs)
            {
                EXPECT_EQ(spec.role, BufferRole::SCRATCH);
            }

            EXPECT_EQ(specs.size(), 3u); // gate, up, ffn_output
        }

        // =========================================================================
        // Aliasing Tests
        // =========================================================================

        TEST_F(Qwen2BufferSpecTest, GetAliasingGroupsReturnsGroups)
        {
            auto groups = builder_->getAliasingGroups();

            // Should have at least 3 aliasing groups
            EXPECT_GE(groups.size(), 3u);

            // Each group should have at least 2 buffers
            for (const auto &group : groups)
            {
                EXPECT_GE(group.size(), 2u);
            }
        }

        TEST_F(Qwen2BufferSpecTest, EstimateMemorySavingsPositive)
        {
            auto [original, optimized] = builder_->estimateMemorySavings(128);

            // Optimized should be less than original
            EXPECT_LT(optimized, original);

            // Savings should be positive (at least some reduction)
            // Note: Actual savings depends on buffer size ratios
            // FFN buffers (d_ff=4864) are larger than attention buffers (d_model=896)
            double savings = 100.0 * (original - optimized) / original;
            EXPECT_GT(savings, 0.0);
        }

        // =========================================================================
        // Conversion Tests
        // =========================================================================

        TEST_F(Qwen2BufferSpecTest, ToBufferRequirementsConverts)
        {
            auto specs = builder_->buildFFNSpecs(32);
            auto reqs = toBufferRequirements(specs);

            EXPECT_EQ(reqs.buffers.size(), specs.size());

            for (size_t i = 0; i < specs.size(); ++i)
            {
                EXPECT_EQ(reqs.buffers[i].name, specs[i].name);
                EXPECT_EQ(reqs.buffers[i].role, specs[i].role);
                EXPECT_EQ(reqs.buffers[i].shape, specs[i].shape);
            }
        }

        // =========================================================================
        // Tensor Type Tests
        // =========================================================================

        TEST_F(Qwen2BufferSpecTest, CustomTensorType)
        {
            Qwen2BufferSpecBuilder bf16_builder(
                896, 14, 2, 64, 4864, 151936,
                BufferTensorType::BF16);

            auto specs = bf16_builder.buildLayerSpecs(32);

            for (const auto &spec : specs)
            {
                EXPECT_EQ(spec.tensor_type, BufferTensorType::BF16);
            }
        }

        TEST_F(Qwen2BufferSpecTest, CustomDeviceIndex)
        {
            Qwen2BufferSpecBuilder gpu_builder(
                896, 14, 2, 64, 4864, 151936,
                BufferTensorType::FP32,
                0 // GPU device 0
            );

            auto specs = gpu_builder.buildLayerSpecs(32);

            for (const auto &spec : specs)
            {
                EXPECT_EQ(spec.device_idx, 0);
            }
        }

        // =========================================================================
        // Phase 3: Local Head Tests (Column-Parallel QKV)
        // =========================================================================

        TEST_F(Qwen2BufferSpecTest, LocalHeadsQBufferShape)
        {
            // Simulate 2-rank tensor parallelism: local_n_heads = 7 (out of 14)
            int local_n_heads = 7;
            int local_n_kv_heads = 1;
            Qwen2BufferSpecBuilder local_builder(
                896,              // d_model
                local_n_heads,    // n_heads (local)
                local_n_kv_heads, // n_kv_heads (local)
                64,               // head_dim
                4864,             // d_ff
                151936            // vocab_size
            );

            int seq_len = 32;
            auto specs = local_builder.buildLayerSpecs(seq_len);

            // Find Q buffer
            const Qwen2BufferSpec *q_spec = nullptr;
            for (const auto &spec : specs)
            {
                if (spec.name == BufferNames::Q)
                {
                    q_spec = &spec;
                    break;
                }
            }

            ASSERT_NE(q_spec, nullptr);
            EXPECT_EQ(q_spec->shape.size(), 2u);
            EXPECT_EQ(q_spec->shape[0], static_cast<size_t>(seq_len));
            // Q should be [seq_len, local_n_heads * head_dim] = [32, 7*64] = [32, 448]
            EXPECT_EQ(q_spec->shape[1], static_cast<size_t>(local_n_heads * 64));
        }

        TEST_F(Qwen2BufferSpecTest, LocalHeadsKVBufferShapes)
        {
            // Simulate 2-rank tensor parallelism: local_n_kv_heads = 1 (out of 2)
            int local_n_heads = 7;
            int local_n_kv_heads = 1;
            Qwen2BufferSpecBuilder local_builder(
                896, local_n_heads, local_n_kv_heads, 64, 4864, 151936);

            int seq_len = 32;
            auto specs = local_builder.buildLayerSpecs(seq_len);

            // Find K and V buffers
            const Qwen2BufferSpec *k_spec = nullptr;
            const Qwen2BufferSpec *v_spec = nullptr;
            for (const auto &spec : specs)
            {
                if (spec.name == BufferNames::K)
                    k_spec = &spec;
                if (spec.name == BufferNames::V)
                    v_spec = &spec;
            }

            ASSERT_NE(k_spec, nullptr);
            ASSERT_NE(v_spec, nullptr);

            // K/V should be [seq_len, local_n_kv_heads * head_dim] = [32, 1*64] = [32, 64]
            EXPECT_EQ(k_spec->shape[0], static_cast<size_t>(seq_len));
            EXPECT_EQ(k_spec->shape[1], static_cast<size_t>(local_n_kv_heads * 64));
            EXPECT_EQ(v_spec->shape[0], static_cast<size_t>(seq_len));
            EXPECT_EQ(v_spec->shape[1], static_cast<size_t>(local_n_kv_heads * 64));
        }

        TEST_F(Qwen2BufferSpecTest, LocalHeadsAttentionOutputShape)
        {
            // With local heads, attention output should also be local
            int local_n_heads = 7;
            int local_n_kv_heads = 1;
            Qwen2BufferSpecBuilder local_builder(
                896, local_n_heads, local_n_kv_heads, 64, 4864, 151936);

            int seq_len = 32;
            auto specs = local_builder.buildAttentionSpecs(seq_len);

            // Find attention output buffer
            const Qwen2BufferSpec *attn_output_spec = nullptr;
            for (const auto &spec : specs)
            {
                if (spec.name == BufferNames::ATTN_OUTPUT)
                {
                    attn_output_spec = &spec;
                    break;
                }
            }

            ASSERT_NE(attn_output_spec, nullptr);
            // ATTN_OUTPUT should be [seq_len, local_n_heads * head_dim] = [32, 448]
            EXPECT_EQ(attn_output_spec->shape[0], static_cast<size_t>(seq_len));
            EXPECT_EQ(attn_output_spec->shape[1], static_cast<size_t>(local_n_heads * 64));
        }

        TEST_F(Qwen2BufferSpecTest, LocalHeadsWorkspaceScoresShape)
        {
            // Workspace scores shape depends on local heads for attention computation
            int local_n_heads = 7;
            int local_n_kv_heads = 1;
            Qwen2BufferSpecBuilder local_builder(
                896, local_n_heads, local_n_kv_heads, 64, 4864, 151936);

            int seq_len = 32;
            auto specs = local_builder.buildAttentionSpecs(seq_len);

            // Find workspace scores buffer
            const Qwen2BufferSpec *scores_spec = nullptr;
            for (const auto &spec : specs)
            {
                if (spec.name == BufferNames::WORKSPACE_SCORES)
                {
                    scores_spec = &spec;
                    break;
                }
            }

            ASSERT_NE(scores_spec, nullptr);
            // WORKSPACE_SCORES should be [local_n_heads, seq_len, seq_len] = [7, 32, 32]
            EXPECT_EQ(scores_spec->shape.size(), 3u);
            EXPECT_EQ(scores_spec->shape[0], static_cast<size_t>(local_n_heads));
            EXPECT_EQ(scores_spec->shape[1], static_cast<size_t>(seq_len));
            EXPECT_EQ(scores_spec->shape[2], static_cast<size_t>(seq_len));
        }

        TEST_F(Qwen2BufferSpecTest, LocalHeadsFFNUnchanged)
        {
            // FFN buffers should NOT depend on local heads (d_ff is independent)
            int local_n_heads = 7;
            int local_n_kv_heads = 1;
            Qwen2BufferSpecBuilder local_builder(
                896, local_n_heads, local_n_kv_heads, 64, 4864, 151936);

            int seq_len = 32;
            auto specs = local_builder.buildFFNSpecs(seq_len);

            const Qwen2BufferSpec *gate_spec = nullptr;
            for (const auto &spec : specs)
            {
                if (spec.name == BufferNames::GATE)
                {
                    gate_spec = &spec;
                    break;
                }
            }

            ASSERT_NE(gate_spec, nullptr);
            // FFN gate should still be [seq_len, d_ff] = [32, 4864]
            EXPECT_EQ(gate_spec->shape[0], static_cast<size_t>(seq_len));
            EXPECT_EQ(gate_spec->shape[1], 4864u);
        }

        TEST_F(Qwen2BufferSpecTest, LocalHeadsResidualUnchanged)
        {
            // Residual/normalized buffers should NOT depend on local heads
            int local_n_heads = 7;
            int local_n_kv_heads = 1;
            Qwen2BufferSpecBuilder local_builder(
                896, local_n_heads, local_n_kv_heads, 64, 4864, 151936);

            int seq_len = 32;
            auto specs = local_builder.buildLayerSpecs(seq_len);

            const Qwen2BufferSpec *residual_spec = nullptr;
            const Qwen2BufferSpec *normalized_spec = nullptr;
            for (const auto &spec : specs)
            {
                if (spec.name == BufferNames::RESIDUAL)
                    residual_spec = &spec;
                if (spec.name == BufferNames::NORMALIZED)
                    normalized_spec = &spec;
            }

            ASSERT_NE(residual_spec, nullptr);
            ASSERT_NE(normalized_spec, nullptr);
            // Both should be [seq_len, d_model] = [32, 896]
            EXPECT_EQ(residual_spec->shape[1], 896u);
            EXPECT_EQ(normalized_spec->shape[1], 896u);
        }

    } // namespace test
} // namespace llaminar2
