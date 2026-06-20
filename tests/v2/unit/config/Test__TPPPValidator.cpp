/**
 * @file Test__TPPPValidator.cpp
 * @brief Unit tests for TP/PP validation against model architecture
 *
 * Tests that the validator correctly identifies incompatible configurations.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "config/TPPPValidator.h"
#include <unordered_set>

namespace llaminar2::test
{

    /**
     * @brief Mock model context for testing validation
     */
    class MockModelContext : public IModelContext
    {
    public:
        int n_heads_ = 14;
        int n_kv_heads_ = 2;
        int n_layers_ = 24;
        int vocab_size_ = 151936;
        int ffn_hidden_ = 4864;
        int embedding_dim_ = 896;
        std::unordered_set<std::string> tensors_;

        // IModelContext interface
        const std::string &path() const override
        {
            static std::string p = "test.gguf";
            return p;
        }
        const std::string &architecture() const override
        {
            static std::string a = "qwen2";
            return a;
        }
        std::shared_ptr<IModelLoader> loader() override { return nullptr; }
        std::shared_ptr<TensorBase> getWeightForDevice(const std::string &, DeviceId) override { return nullptr; }
        bool hasTensor(const std::string &name) const override { return tensors_.find(name) != tensors_.end(); }
        std::shared_ptr<IWeightManager> weightManager() override { return nullptr; }

        int blockCount() const override { return n_layers_; }
        int embeddingLength() const override { return embedding_dim_; }
        int headCount() const override { return n_heads_; }
        int headCountKV() const override { return n_kv_heads_; }
        int vocabSize() const override { return vocab_size_; }
        int contextLength() const override { return 32768; }
        int feedForwardLength() const override { return ffn_hidden_; }
        int keyLength() const override { return 0; }
    };

    // =========================================================================
    // Fixture
    // =========================================================================

    class Test__TPPPValidator : public ::testing::Test
    {
    protected:
        MockModelContext model_;
        OrchestrationConfig config_;

        void SetUp() override
        {
            // Default Qwen2.5-0.5B parameters
            model_.n_heads_ = 14;
            model_.n_kv_heads_ = 2;
            model_.n_layers_ = 24;
            model_.vocab_size_ = 151936;
            model_.ffn_hidden_ = 4864;
            model_.embedding_dim_ = 896;

            // Default config (no parallelism)
            config_ = OrchestrationConfig::defaults();
        }
    };

    // =========================================================================
    // No Parallelism (should always pass)
    // =========================================================================

    TEST_F(Test__TPPPValidator, NoParallelism_AlwaysValid)
    {
        config_.tp_degree = 1;
        config_.pp_degree = 1;

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_TRUE(result.valid) << result.toString();
        EXPECT_TRUE(result.errors.empty());
    }

    // =========================================================================
    // TP Head Divisibility
    // =========================================================================

    TEST_F(Test__TPPPValidator, TP2_WithQwen05B_Valid)
    {
        // Qwen2.5-0.5B has 2 KV heads - TP=2 is valid
        config_.tp_degree = 2;

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_TRUE(result.valid) << result.toString();
    }

    TEST_F(Test__TPPPValidator, TP3_WithQwen05B_InvalidKVHeads)
    {
        // Qwen2.5-0.5B has 2 KV heads - TP=3 is invalid (2 % 3 != 0)
        config_.tp_degree = 3;

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_FALSE(result.valid);
        EXPECT_GE(result.errors.size(), 1);

        // Should mention KV heads not divisible
        bool found_kv_error = false;
        for (const auto &e : result.errors)
        {
            if (e.find("n_kv_heads") != std::string::npos && e.find("not divisible") != std::string::npos)
            {
                found_kv_error = true;
                break;
            }
        }
        EXPECT_TRUE(found_kv_error) << "Expected error about KV heads divisibility";
    }

    TEST_F(Test__TPPPValidator, TP4_WithQwen05B_InvalidKVHeads)
    {
        // Qwen2.5-0.5B has 2 KV heads - TP=4 is invalid (2 % 4 != 0)
        config_.tp_degree = 4;

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_FALSE(result.valid);
    }

    TEST_F(Test__TPPPValidator, TP7_WithQwen05B_InvalidQueryHeads)
    {
        // Qwen2.5-0.5B has 14 Q heads - TP=7 is valid for Q but invalid for KV (2 % 7 != 0)
        config_.tp_degree = 7;

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_FALSE(result.valid);
    }

    TEST_F(Test__TPPPValidator, TP2_With28HeadModel_Valid)
    {
        // Model with 28 heads and 4 KV heads - TP=2 is valid
        model_.n_heads_ = 28;
        model_.n_kv_heads_ = 4;
        config_.tp_degree = 2;

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_TRUE(result.valid) << result.toString();
    }

    TEST_F(Test__TPPPValidator, TP4_With28HeadModel_Valid)
    {
        // Model with 28 heads and 4 KV heads - TP=4 is valid
        model_.n_heads_ = 28;
        model_.n_kv_heads_ = 4;
        config_.tp_degree = 4;

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_TRUE(result.valid) << result.toString();
    }

    // =========================================================================
    // TP Dimension Divisibility
    // =========================================================================

    TEST_F(Test__TPPPValidator, TP_InvalidFFNDim)
    {
        // FFN hidden = 4864 is not divisible by 3
        model_.n_kv_heads_ = 6; // Make KV heads valid for TP=3
        model_.n_heads_ = 12;   // Make Q heads valid for TP=3
        config_.tp_degree = 3;

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_FALSE(result.valid);

        bool found_ffn_error = false;
        for (const auto &e : result.errors)
        {
            if (e.find("ffn_hidden") != std::string::npos)
            {
                found_ffn_error = true;
                break;
            }
        }
        EXPECT_TRUE(found_ffn_error) << "Expected error about FFN dimension";
    }

    TEST_F(Test__TPPPValidator, TP_InvalidVocabSize)
    {
        // Vocab = 151936 = 2^11 * 74 = divisible by 2 but not by 3
        model_.n_kv_heads_ = 6;
        model_.n_heads_ = 12;
        model_.ffn_hidden_ = 4860; // Divisible by 3
        config_.tp_degree = 3;

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_FALSE(result.valid);

        bool found_vocab_error = false;
        for (const auto &e : result.errors)
        {
            if (e.find("vocab_size") != std::string::npos)
            {
                found_vocab_error = true;
                break;
            }
        }
        EXPECT_TRUE(found_vocab_error) << "Expected error about vocab size";
    }

    // =========================================================================
    // PP Layer Divisibility
    // =========================================================================

    TEST_F(Test__TPPPValidator, PP2_With24Layers_Valid)
    {
        // 24 layers / 2 = 12 layers per stage
        config_.pp_degree = 2;

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_TRUE(result.valid) << result.toString();
    }

    TEST_F(Test__TPPPValidator, PPManual_Qwen36NextNSidecarExcludedFromLayerCoverage)
    {
        model_.n_layers_ = 65;
        model_.tensors_.insert("blk.64.nextn.eh_proj.weight");

        config_.pp_degree = 2;
        config_.pp_split = PPSplitMode::MANUAL;
        config_.pp_stage_definitions = {
            PPStageDefinition::parse("0=stage0:0-31"),
            PPStageDefinition::parse("1=stage1:32-63"),
        };

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_TRUE(result.valid) << result.toString();
    }

    TEST_F(Test__TPPPValidator, PP4_With24Layers_Valid)
    {
        // 24 layers / 4 = 6 layers per stage
        config_.pp_degree = 4;

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_TRUE(result.valid) << result.toString();
    }

    TEST_F(Test__TPPPValidator, PP5_With24Layers_Invalid)
    {
        // 24 layers / 5 = 4.8 layers per stage (invalid with EQUAL split)
        config_.pp_degree = 5;
        config_.pp_split = PPSplitMode::EQUAL;

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_FALSE(result.valid);

        bool found_layer_error = false;
        for (const auto &e : result.errors)
        {
            if (e.find("n_layers") != std::string::npos && e.find("not divisible") != std::string::npos)
            {
                found_layer_error = true;
                break;
            }
        }
        EXPECT_TRUE(found_layer_error) << "Expected error about layer count divisibility";
    }

    TEST_F(Test__TPPPValidator, PP_TooManyStages)
    {
        // More PP stages than layers
        config_.pp_degree = 30;
        config_.pp_split = PPSplitMode::EQUAL;

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_FALSE(result.valid);
    }

    TEST_F(Test__TPPPValidator, CPULayers_ExceedsTotal)
    {
        config_.cpu_layers = 100; // More than 24 layers

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_FALSE(result.valid);

        bool found_cpu_error = false;
        for (const auto &e : result.errors)
        {
            if (e.find("cpu-layers") != std::string::npos && e.find("exceeds") != std::string::npos)
            {
                found_cpu_error = true;
                break;
            }
        }
        EXPECT_TRUE(found_cpu_error) << "Expected error about CPU layers exceeding total";
    }

    // =========================================================================
    // Proportional Weights
    // =========================================================================

    TEST_F(Test__TPPPValidator, ProportionalTP_ValidWeights)
    {
        // 14 heads with weights [0.5, 0.5] = 7 heads each (valid)
        model_.n_kv_heads_ = 2;
        config_.tp_degree = 2;
        config_.tp_weights = {0.5f, 0.5f};

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_TRUE(result.valid) << result.toString();
    }

    TEST_F(Test__TPPPValidator, ProportionalTP_FractionalHeads)
    {
        // 14 heads with weights [0.73, 0.27] = 10.22 and 3.78 heads (invalid)
        config_.tp_degree = 2;
        config_.tp_weights = {0.73f, 0.27f};

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_FALSE(result.valid);

        bool found_fractional_error = false;
        for (const auto &e : result.errors)
        {
            if (e.find("fractional") != std::string::npos)
            {
                found_fractional_error = true;
                break;
            }
        }
        EXPECT_TRUE(found_fractional_error) << "Expected error about fractional head count";
    }

    // =========================================================================
    // Combined TP + PP
    // =========================================================================

    TEST_F(Test__TPPPValidator, Combined_TP2_PP2_Valid)
    {
        // TP=2 (valid for 2 KV heads), PP=2 (valid for 24 layers)
        config_.tp_degree = 2;
        config_.pp_degree = 2;

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_TRUE(result.valid) << result.toString();
    }

    TEST_F(Test__TPPPValidator, Combined_TP2_PP5_InvalidPP)
    {
        // TP=2 valid, PP=5 invalid (24 % 5 != 0)
        config_.tp_degree = 2;
        config_.pp_degree = 5;
        config_.pp_split = PPSplitMode::EQUAL;

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_FALSE(result.valid);
    }

    // =========================================================================
    // Error Message Quality
    // =========================================================================

    TEST_F(Test__TPPPValidator, ErrorMessage_SuggestsValidTPDegrees)
    {
        // TP=3 invalid - error should suggest valid degrees (1, 2)
        config_.tp_degree = 3;

        auto result = TPPPValidator::validate(config_, model_);

        EXPECT_FALSE(result.valid);

        // Check that error message suggests valid alternatives
        bool suggests_valid = false;
        for (const auto &e : result.errors)
        {
            if (e.find("Valid TP degrees") != std::string::npos &&
                (e.find("1, 2") != std::string::npos || e.find("1,2") != std::string::npos))
            {
                suggests_valid = true;
                break;
            }
        }
        EXPECT_TRUE(suggests_valid) << "Expected error to suggest valid TP degrees (1, 2)";
    }

} // namespace llaminar2::test
