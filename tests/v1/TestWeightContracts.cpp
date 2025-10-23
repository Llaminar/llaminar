/**
 * @file TestWeightContracts.cpp
 * @brief Unit tests for the weight contract validation system
 *
 * Tests cover:
 * - Symbolic expression evaluation
 * - Shape validation (success and failure cases)
 * - Error message formatting
 * - Global and per-layer validation
 * - Qwen canonical format compliance
 * - Multi-architecture support preparation
 *
 * @author David Sanftenberg
 */

#include "gtest/gtest.h"
#include "WeightContracts.h"
#include "tensors/TensorFactory.h"
#include "TransformerConfig.h"
#include <memory>
#include <stdexcept>

using namespace llaminar;

namespace
{

    /**
     * @brief Create test configuration with typical Qwen 0.5B dimensions
     */
    TransformerLayerConfig createTestConfig()
    {
        TransformerLayerConfig cfg;
        cfg.n_head = 14;
        cfg.n_head_kv = 2; // GQA
        cfg.head_dim = 64;
        cfg.d_model = 896; // 14 * 64
        cfg.d_ff = 4864;   // ~5.4 * d_model
        cfg.vocab_size = 151669;
        cfg.max_seq_len = 32768;
        cfg.n_layers = 24;
        cfg.eps = 1e-6f;
        return cfg;
    }

} // anonymous namespace

// ============================================================================
// Symbolic Expression Evaluation Tests
// ============================================================================

TEST(WeightContracts, EvaluateSimpleLiteral)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("test", {"896"}, "Simple literal");

    auto shape = contract.evaluate(cfg);
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 896);
}

TEST(WeightContracts, EvaluateSimpleVariable)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("test", {"d_model"}, "Simple variable");

    auto shape = contract.evaluate(cfg);
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 896);
}

TEST(WeightContracts, EvaluateMultiplication)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("test", {"n_head*head_dim"}, "Multiplication");

    auto shape = contract.evaluate(cfg);
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 14 * 64); // 896
}

TEST(WeightContracts, EvaluateGQADimension)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("test", {"n_head_kv*head_dim"}, "GQA dimension");

    auto shape = contract.evaluate(cfg);
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 2 * 64); // 128
}

TEST(WeightContracts, EvaluateMultipleDimensions)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("test",
                                 {"n_head*head_dim", "d_model", "vocab_size"},
                                 "Multiple dimensions");

    auto shape = contract.evaluate(cfg);
    ASSERT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 896);
    EXPECT_EQ(shape[1], 896);
    EXPECT_EQ(shape[2], 151669);
}

TEST(WeightContracts, EvaluateDivision)
{
    auto cfg = createTestConfig();
    cfg.d_ff = 4864;
    WeightShapeContract contract("test", {"d_ff/4"}, "Division");

    auto shape = contract.evaluate(cfg);
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 1216); // 4864 / 4
}

TEST(WeightContracts, EvaluateAllVariables)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("test",
                                 {"d_model", "n_head", "head_dim", "n_head_kv", "d_ff", "vocab_size"},
                                 "All variables");

    auto shape = contract.evaluate(cfg);
    ASSERT_EQ(shape.size(), 6);
    EXPECT_EQ(shape[0], 896);
    EXPECT_EQ(shape[1], 14);
    EXPECT_EQ(shape[2], 64);
    EXPECT_EQ(shape[3], 2);
    EXPECT_EQ(shape[4], 4864);
    EXPECT_EQ(shape[5], 151669);
}

TEST(WeightContracts, EvaluateInvalidExpression)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("test", {"unknown_variable"}, "Invalid");

    EXPECT_THROW({ contract.evaluate(cfg); }, std::runtime_error);
}

TEST(WeightContracts, EvaluateDivisionByZero)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("test", {"d_model/0"}, "Division by zero");

    EXPECT_THROW({ contract.evaluate(cfg); }, std::runtime_error);
}

// ============================================================================
// Shape Validation Tests - Success Cases
// ============================================================================

TEST(WeightContracts, ValidateCorrectShape)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("wq", {"n_head*head_dim", "d_model"}, "Q projection");

    auto tensor = TensorFactory::create_simple({896, 896});
    EXPECT_NO_THROW({
        contract.validate(tensor, cfg);
    });
}

TEST(WeightContracts, ValidateGQAKeyProjection)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("wk", {"n_head_kv*head_dim", "d_model"}, "K projection");

    auto tensor = TensorFactory::create_simple({128, 896}); // GQA: 2*64 = 128
    EXPECT_NO_THROW({
        contract.validate(tensor, cfg);
    });
}

TEST(WeightContracts, ValidateFFNGate)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("w_gate", {"d_ff", "d_model"}, "FFN gate");

    auto tensor = TensorFactory::create_simple({4864, 896});
    EXPECT_NO_THROW({
        contract.validate(tensor, cfg);
    });
}

TEST(WeightContracts, ValidateFFNDown)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("w_down", {"d_model", "d_ff"}, "FFN down");

    auto tensor = TensorFactory::create_simple({896, 4864});
    EXPECT_NO_THROW({
        contract.validate(tensor, cfg);
    });
}

TEST(WeightContracts, ValidateLMHead)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("lm_head", {"vocab_size", "d_model"}, "LM head");

    auto tensor = TensorFactory::create_simple({151669, 896});
    EXPECT_NO_THROW({
        contract.validate(tensor, cfg);
    });
}

TEST(WeightContracts, Validate1DNorm)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("attn_norm", {"d_model"}, "RMSNorm");

    auto tensor = TensorFactory::create_simple({896});
    EXPECT_NO_THROW({
        contract.validate(tensor, cfg);
    });
}

// ============================================================================
// Shape Validation Tests - Failure Cases
// ============================================================================

TEST(WeightContracts, ValidateWrongDimension)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("wq", {"n_head*head_dim", "d_model"}, "Q projection");

    // Wrong: [d_model, n_head*head_dim] instead of [n_head*head_dim, d_model]
    auto tensor = TensorFactory::create_simple({896, 896}); // Transpose would be [896, 896] too!
    // Use different dimensions to actually test mismatch
    auto wrong_tensor = TensorFactory::create_simple({128, 896}); // Wrong first dim

    EXPECT_THROW({ contract.validate(wrong_tensor, cfg); }, std::runtime_error);
}

TEST(WeightContracts, ValidateWrongRank)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("wq", {"n_head*head_dim", "d_model"}, "Q projection");

    // Wrong: 1D instead of 2D
    auto tensor = TensorFactory::create_simple({896});

    EXPECT_THROW({ contract.validate(tensor, cfg); }, std::runtime_error);
}

TEST(WeightContracts, ValidateNullTensor)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("wq", {"n_head*head_dim", "d_model"}, "Q projection");

    std::shared_ptr<TensorBase> null_tensor;

    EXPECT_THROW({ contract.validate(null_tensor, cfg); }, std::runtime_error);
}

TEST(WeightContracts, ValidateTransposedAttentionWeight)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("wk", {"n_head_kv*head_dim", "d_model"}, "K projection");

    // Common error: transposed (old format)
    auto tensor = TensorFactory::create_simple({896, 128}); // Should be [128, 896]

    try
    {
        contract.validate(tensor, cfg);
        FAIL() << "Expected validation to throw for transposed weight";
    }
    catch (const std::runtime_error &e)
    {
        std::string error_msg = e.what();
        // Check that error message contains useful info
        EXPECT_NE(error_msg.find("wk"), std::string::npos);
        EXPECT_NE(error_msg.find("128"), std::string::npos);
        EXPECT_NE(error_msg.find("896"), std::string::npos);
        EXPECT_NE(error_msg.find("mismatch"), std::string::npos);
    }
}

TEST(WeightContracts, ValidateWrongFFNDimension)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("w_gate", {"d_ff", "d_model"}, "FFN gate");

    // Wrong: Used d_model for first dimension
    auto tensor = TensorFactory::create_simple({896, 896});

    try
    {
        contract.validate(tensor, cfg);
        FAIL() << "Expected validation to throw";
    }
    catch (const std::runtime_error &e)
    {
        std::string error_msg = e.what();
        EXPECT_NE(error_msg.find("w_gate"), std::string::npos);
        EXPECT_NE(error_msg.find("4864"), std::string::npos); // Expected d_ff
        EXPECT_NE(error_msg.find("896"), std::string::npos);  // Actual
    }
}

// ============================================================================
// Error Message Quality Tests
// ============================================================================

TEST(WeightContracts, ErrorMessageIncludesWeightName)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("attn_k.weight", {"128", "896"}, "Key projection");

    auto tensor = TensorFactory::create_simple({896, 128}); // Wrong order

    try
    {
        contract.validate(tensor, cfg);
        FAIL();
    }
    catch (const std::runtime_error &e)
    {
        std::string msg = e.what();
        EXPECT_NE(msg.find("attn_k.weight"), std::string::npos);
    }
}

TEST(WeightContracts, ErrorMessageIncludesLayerIndex)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("wq", {"896", "896"}, "Q projection");

    auto tensor = TensorFactory::create_simple({128, 896});

    try
    {
        contract.validate(tensor, cfg, 5); // Layer 5
        FAIL();
    }
    catch (const std::runtime_error &e)
    {
        std::string msg = e.what();
        EXPECT_NE(msg.find("layer 5"), std::string::npos);
    }
}

TEST(WeightContracts, ErrorMessageIncludesDescription)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("wv", {"128", "896"},
                                 "Value projection for GQA attention");

    auto tensor = TensorFactory::create_simple({896, 128});

    try
    {
        contract.validate(tensor, cfg);
        FAIL();
    }
    catch (const std::runtime_error &e)
    {
        std::string msg = e.what();
        EXPECT_NE(msg.find("Value projection for GQA attention"), std::string::npos);
    }
}

TEST(WeightContracts, ErrorMessageIncludesSymbolicExpression)
{
    auto cfg = createTestConfig();
    WeightShapeContract contract("wk", {"n_head_kv*head_dim", "d_model"},
                                 "Key projection");

    auto tensor = TensorFactory::create_simple({896, 128});

    try
    {
        contract.validate(tensor, cfg);
        FAIL();
    }
    catch (const std::runtime_error &e)
    {
        std::string msg = e.what();
        // Should show both symbolic and evaluated
        EXPECT_NE(msg.find("n_head_kv*head_dim"), std::string::npos);
        EXPECT_NE(msg.find("d_model"), std::string::npos);
        EXPECT_NE(msg.find("128"), std::string::npos);
        EXPECT_NE(msg.find("896"), std::string::npos);
    }
}

// ============================================================================
// Qwen Contract Tests
// ============================================================================

TEST(WeightContracts, QwenContractsExist)
{
    auto contracts = getQwenWeightContracts();

    // Should have global weights
    EXPECT_FALSE(contracts.global_weights.empty());

    // Should have layer weights
    EXPECT_FALSE(contracts.layer_weights.empty());
}

TEST(WeightContracts, QwenGlobalWeightsCount)
{
    auto contracts = getQwenWeightContracts();

    // Qwen has: token_embedding, output_norm, lm_head
    EXPECT_EQ(contracts.global_weights.size(), 3);
}

TEST(WeightContracts, QwenLayerWeightsCount)
{
    auto contracts = getQwenWeightContracts();

    // Qwen layer has: attn_norm, wq, wk, wv, wo, ffn_norm, w_gate, w_up, w_down
    EXPECT_EQ(contracts.layer_weights.size(), 9);
}

TEST(WeightContracts, QwenAttentionWeightFormats)
{
    auto contracts = getQwenWeightContracts();
    auto cfg = createTestConfig();

    // Find attention contracts
    for (const auto &contract : contracts.layer_weights)
    {
        if (contract.weight_name.find("attn_q") != std::string::npos)
        {
            auto shape = contract.evaluate(cfg);
            EXPECT_EQ(shape[0], 896); // n_head * head_dim
            EXPECT_EQ(shape[1], 896); // d_model
        }
        else if (contract.weight_name.find("attn_k") != std::string::npos ||
                 contract.weight_name.find("attn_v") != std::string::npos)
        {
            auto shape = contract.evaluate(cfg);
            EXPECT_EQ(shape[0], 128); // n_head_kv * head_dim (GQA)
            EXPECT_EQ(shape[1], 896); // d_model
        }
        else if (contract.weight_name.find("attn_output") != std::string::npos)
        {
            auto shape = contract.evaluate(cfg);
            EXPECT_EQ(shape[0], 896); // d_model
            EXPECT_EQ(shape[1], 896); // n_head * head_dim
        }
    }
}

TEST(WeightContracts, QwenFFNWeightFormats)
{
    auto contracts = getQwenWeightContracts();
    auto cfg = createTestConfig();

    for (const auto &contract : contracts.layer_weights)
    {
        if (contract.weight_name.find("gate") != std::string::npos ||
            contract.weight_name.find("up") != std::string::npos)
        {
            auto shape = contract.evaluate(cfg);
            EXPECT_EQ(shape[0], 4864); // d_ff
            EXPECT_EQ(shape[1], 896);  // d_model
        }
        else if (contract.weight_name.find("down") != std::string::npos)
        {
            auto shape = contract.evaluate(cfg);
            EXPECT_EQ(shape[0], 896);  // d_model
            EXPECT_EQ(shape[1], 4864); // d_ff
        }
    }
}

// ============================================================================
// Global Validation Tests
// ============================================================================

TEST(WeightContracts, ValidateGlobalWeightsSuccess)
{
    auto cfg = createTestConfig();
    auto contracts = getQwenWeightContracts();

    auto token_embedding = TensorFactory::create_simple({151669, 896});
    auto output_norm = TensorFactory::create_simple({896});
    auto lm_head = TensorFactory::create_simple({151669, 896});

    EXPECT_NO_THROW({
        contracts.validate_global(token_embedding, output_norm, lm_head, cfg);
    });
}

TEST(WeightContracts, ValidateGlobalWeightsFailure)
{
    auto cfg = createTestConfig();
    auto contracts = getQwenWeightContracts();

    auto token_embedding = TensorFactory::create_simple({151669, 896});
    auto output_norm = TensorFactory::create_simple({896});
    auto lm_head = TensorFactory::create_simple({896, 151669}); // WRONG: transposed

    EXPECT_THROW({ contracts.validate_global(token_embedding, output_norm, lm_head, cfg); }, std::runtime_error);
}

// ============================================================================
// Layer Validation Tests
// ============================================================================

TEST(WeightContracts, ValidateLayerWeightsSuccess)
{
    auto cfg = createTestConfig();
    auto contracts = getQwenWeightContracts();

    // Create all layer weights in correct format
    auto attn_norm = TensorFactory::create_simple({896});
    auto wq = TensorFactory::create_simple({896, 896});
    auto wk = TensorFactory::create_simple({128, 896});
    auto wv = TensorFactory::create_simple({128, 896});
    auto wo = TensorFactory::create_simple({896, 896});
    auto ffn_norm = TensorFactory::create_simple({896});
    auto w_gate = TensorFactory::create_simple({4864, 896});
    auto w_up = TensorFactory::create_simple({4864, 896});
    auto w_down = TensorFactory::create_simple({896, 4864});

    EXPECT_NO_THROW({
        contracts.validate_layer(0, attn_norm, wq, wk, wv, wo,
                                 ffn_norm, w_gate, w_up, w_down, cfg);
    });
}

TEST(WeightContracts, ValidateLayerWeightsWrongQShape)
{
    auto cfg = createTestConfig();
    auto contracts = getQwenWeightContracts();

    auto attn_norm = TensorFactory::create_simple({896});
    auto wq = TensorFactory::create_simple({896, 128}); // WRONG dimension
    auto wk = TensorFactory::create_simple({128, 896});
    auto wv = TensorFactory::create_simple({128, 896});
    auto wo = TensorFactory::create_simple({896, 896});
    auto ffn_norm = TensorFactory::create_simple({896});
    auto w_gate = TensorFactory::create_simple({4864, 896});
    auto w_up = TensorFactory::create_simple({4864, 896});
    auto w_down = TensorFactory::create_simple({896, 4864});

    EXPECT_THROW({ contracts.validate_layer(0, attn_norm, wq, wk, wv, wo,
                                            ffn_norm, w_gate, w_up, w_down, cfg); }, std::runtime_error);
}

TEST(WeightContracts, ValidateLayerWeightsWrongFFNShape)
{
    auto cfg = createTestConfig();
    auto contracts = getQwenWeightContracts();

    auto attn_norm = TensorFactory::create_simple({896});
    auto wq = TensorFactory::create_simple({896, 896});
    auto wk = TensorFactory::create_simple({128, 896});
    auto wv = TensorFactory::create_simple({128, 896});
    auto wo = TensorFactory::create_simple({896, 896});
    auto ffn_norm = TensorFactory::create_simple({896});
    auto w_gate = TensorFactory::create_simple({896, 4864}); // WRONG: transposed
    auto w_up = TensorFactory::create_simple({4864, 896});
    auto w_down = TensorFactory::create_simple({896, 4864});

    EXPECT_THROW({ contracts.validate_layer(0, attn_norm, wq, wk, wv, wo,
                                            ffn_norm, w_gate, w_up, w_down, cfg); }, std::runtime_error);
}

// ============================================================================
// Edge Cases and Robustness Tests
// ============================================================================

TEST(WeightContracts, ValidateVerySmallModel)
{
    TransformerLayerConfig cfg;
    cfg.n_head = 2;
    cfg.n_head_kv = 1;
    cfg.head_dim = 32;
    cfg.d_model = 64; // 2 * 32
    cfg.d_ff = 256;
    cfg.vocab_size = 1000;

    WeightShapeContract contract("wq", {"n_head*head_dim", "d_model"}, "Q");
    auto tensor = TensorFactory::create_simple({64, 64});

    EXPECT_NO_THROW({
        contract.validate(tensor, cfg);
    });
}

TEST(WeightContracts, ValidateVeryLargeModel)
{
    TransformerLayerConfig cfg;
    cfg.n_head = 64;
    cfg.n_head_kv = 8;
    cfg.head_dim = 128;
    cfg.d_model = 8192; // 64 * 128
    cfg.d_ff = 32768;
    cfg.vocab_size = 250000;

    WeightShapeContract contract("lm_head", {"vocab_size", "d_model"}, "LM head");
    auto tensor = TensorFactory::create_simple({250000, 8192});

    EXPECT_NO_THROW({
        contract.validate(tensor, cfg);
    });
}

TEST(WeightContracts, ValidateGQARatio4)
{
    TransformerLayerConfig cfg;
    cfg.n_head = 32;
    cfg.n_head_kv = 8; // 4:1 GQA ratio
    cfg.head_dim = 64;
    cfg.d_model = 2048;
    cfg.d_ff = 8192;
    cfg.vocab_size = 50000;

    WeightShapeContract contract("wk", {"n_head_kv*head_dim", "d_model"}, "K");
    auto tensor = TensorFactory::create_simple({512, 2048}); // 8 * 64 = 512

    EXPECT_NO_THROW({
        contract.validate(tensor, cfg);
    });
}

TEST(WeightContracts, ValidateNoGQA)
{
    TransformerLayerConfig cfg;
    cfg.n_head = 16;
    cfg.n_head_kv = 16; // No GQA (MHA)
    cfg.head_dim = 64;
    cfg.d_model = 1024;
    cfg.d_ff = 4096;
    cfg.vocab_size = 50000;

    WeightShapeContract contract("wk", {"n_head_kv*head_dim", "d_model"}, "K");
    auto tensor = TensorFactory::create_simple({1024, 1024}); // Same as wq

    EXPECT_NO_THROW({
        contract.validate(tensor, cfg);
    });
}

// ============================================================================
// Integration Test - Full Model Validation
// ============================================================================

TEST(WeightContracts, ValidateFullModelCorrect)
{
    auto cfg = createTestConfig();
    auto contracts = getQwenWeightContracts();

    // Global weights
    auto token_embedding = TensorFactory::create_simple({151669, 896});
    auto output_norm = TensorFactory::create_simple({896});
    auto lm_head = TensorFactory::create_simple({151669, 896});

    EXPECT_NO_THROW({
        contracts.validate_global(token_embedding, output_norm, lm_head, cfg);
    });

    // Validate multiple layers
    for (int layer = 0; layer < 3; ++layer)
    {
        auto attn_norm = TensorFactory::create_simple({896});
        auto wq = TensorFactory::create_simple({896, 896});
        auto wk = TensorFactory::create_simple({128, 896});
        auto wv = TensorFactory::create_simple({128, 896});
        auto wo = TensorFactory::create_simple({896, 896});
        auto ffn_norm = TensorFactory::create_simple({896});
        auto w_gate = TensorFactory::create_simple({4864, 896});
        auto w_up = TensorFactory::create_simple({4864, 896});
        auto w_down = TensorFactory::create_simple({896, 4864});

        EXPECT_NO_THROW({
            contracts.validate_layer(layer, attn_norm, wq, wk, wv, wo,
                                     ffn_norm, w_gate, w_up, w_down, cfg);
        });
    }
}

TEST(WeightContracts, ValidateFullModelFirstLayerError)
{
    auto cfg = createTestConfig();
    auto contracts = getQwenWeightContracts();

    // Layer 0 has error, layer 1 is correct
    auto attn_norm = TensorFactory::create_simple({896});
    auto wq = TensorFactory::create_simple({128, 896}); // WRONG
    auto wk = TensorFactory::create_simple({128, 896});
    auto wv = TensorFactory::create_simple({128, 896});
    auto wo = TensorFactory::create_simple({896, 896});
    auto ffn_norm = TensorFactory::create_simple({896});
    auto w_gate = TensorFactory::create_simple({4864, 896});
    auto w_up = TensorFactory::create_simple({4864, 896});
    auto w_down = TensorFactory::create_simple({896, 4864});

    try
    {
        contracts.validate_layer(0, attn_norm, wq, wk, wv, wo,
                                 ffn_norm, w_gate, w_up, w_down, cfg);
        FAIL() << "Should have thrown for layer 0";
    }
    catch (const std::runtime_error &e)
    {
        std::string msg = e.what();
        EXPECT_NE(msg.find("layer 0"), std::string::npos);
    }
}

// ============================================================================
// Documentation Test
// ============================================================================

TEST(WeightContracts, ContractsHaveDescriptions)
{
    auto contracts = getQwenWeightContracts();

    // All contracts should have descriptions
    for (const auto &contract : contracts.global_weights)
    {
        EXPECT_FALSE(contract.description.empty())
            << "Global weight " << contract.weight_name << " missing description";
    }

    for (const auto &contract : contracts.layer_weights)
    {
        EXPECT_FALSE(contract.description.empty())
            << "Layer weight " << contract.weight_name << " missing description";
    }
}

TEST(WeightContracts, ContractsHaveValidNames)
{
    auto contracts = getQwenWeightContracts();

    // Weight names should not be empty
    for (const auto &contract : contracts.global_weights)
    {
        EXPECT_FALSE(contract.weight_name.empty());
    }

    for (const auto &contract : contracts.layer_weights)
    {
        EXPECT_FALSE(contract.weight_name.empty());
    }
}

TEST(WeightContracts, ContractsHaveValidDimensions)
{
    auto contracts = getQwenWeightContracts();

    // All contracts should have at least one dimension
    for (const auto &contract : contracts.global_weights)
    {
        EXPECT_FALSE(contract.dim_expressions.empty())
            << "Global weight " << contract.weight_name << " has no dimensions";
    }

    for (const auto &contract : contracts.layer_weights)
    {
        EXPECT_FALSE(contract.dim_expressions.empty())
            << "Layer weight " << contract.weight_name << " has no dimensions";
    }
}
