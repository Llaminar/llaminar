/**
 * @file test_weight_validation_integration.cpp
 * @brief Integration test for weight contract validation with real GGUF models
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "weight_contracts.h"
#include "transformer_config.h"
#include "tensors/tensor_factory.h"
#include <filesystem>
#include <stdexcept>

using namespace llaminar;

/**
 * @brief Test Qwen weight contracts exist and are valid
 */
TEST(WeightValidationIntegration, QwenContractsExist)
{
    auto contracts = getQwenWeightContracts();

    // Should have 3 global weights
    EXPECT_EQ(contracts.global_weights.size(), 3);

    // Should have 9 layer weights
    EXPECT_EQ(contracts.layer_weights.size(), 9);

    // Verify global weights exist
    bool has_token_embd = false;
    bool has_output_norm = false;
    bool has_lm_head = false;

    for (const auto &w : contracts.global_weights)
    {
        if (w.weight_name == "token_embedding")
            has_token_embd = true;
        if (w.weight_name == "output_norm")
            has_output_norm = true;
        if (w.weight_name == "lm_head")
            has_lm_head = true;
    }

    EXPECT_TRUE(has_token_embd) << "Missing token_embedding contract";
    EXPECT_TRUE(has_output_norm) << "Missing output_norm contract";
    EXPECT_TRUE(has_lm_head) << "Missing lm_head contract";
}

/**
 * @brief Test that contracts validate correct shapes
 */
TEST(WeightValidationIntegration, ValidateCorrectShapes)
{
    auto contracts = getQwenWeightContracts();

    // Create a typical Qwen 0.5B config
    TransformerLayerConfig cfg;
    cfg.d_model = 896;
    cfg.n_head = 14;
    cfg.n_head_kv = 2; // GQA
    cfg.d_ff = 4864;
    cfg.vocab_size = 151936;

    // Create tensors with correct shapes (matching GGUF format)
    // All weights are [out_features, in_features]

    // Q projection: [n_head*head_dim, d_model] = [896, 896]
    auto wq = TensorFactory::create_simple({896, 896});

    // K projection: [n_head_kv*head_dim, d_model] = [128, 896]
    auto wk = TensorFactory::create_simple({128, 896});

    // V projection: [n_head_kv*head_dim, d_model] = [128, 896]
    auto wv = TensorFactory::create_simple({128, 896});

    // Attention output: [d_model, n_head*head_dim] = [896, 896]
    auto wo = TensorFactory::create_simple({896, 896});

    // FFN gate: [d_ff, d_model] = [4864, 896]
    auto gate = TensorFactory::create_simple({4864, 896});

    // FFN up: [d_ff, d_model] = [4864, 896]
    auto up = TensorFactory::create_simple({4864, 896});

    // FFN down: [d_model, d_ff] = [896, 4864]
    auto down = TensorFactory::create_simple({896, 4864});

    // LM head: [vocab_size, d_model] = [151936, 896]
    auto lm_head = TensorFactory::create_simple({151936, 896});

    // Validate each contract
    for (const auto &contract : contracts.layer_weights)
    {
        std::shared_ptr<TensorBase> tensor;

        if (contract.weight_name == "wq")
            tensor = wq;
        else if (contract.weight_name == "wk")
            tensor = wk;
        else if (contract.weight_name == "wv")
            tensor = wv;
        else if (contract.weight_name == "attn_output")
            tensor = wo;
        else if (contract.weight_name == "ffn_gate")
            tensor = gate;
        else if (contract.weight_name == "ffn_up")
            tensor = up;
        else if (contract.weight_name == "ffn_down")
            tensor = down;
        else
            continue; // Skip norms

        EXPECT_NO_THROW({
            contract.validate(tensor, cfg, 0);
        }) << "Contract validation failed for "
           << contract.weight_name;
    }

    // Validate LM head
    for (const auto &contract : contracts.global_weights)
    {
        if (contract.weight_name == "lm_head")
        {
            EXPECT_NO_THROW({
                contract.validate(lm_head, cfg);
            }) << "LM head validation failed";
            break;
        }
    }
}

/**
 * @brief Test that contracts catch transposed weights
 */
TEST(WeightValidationIntegration, DetectTransposedWeights)
{
    auto contracts = getQwenWeightContracts();

    TransformerLayerConfig cfg;
    cfg.d_model = 896;
    cfg.n_head = 14;
    cfg.n_head_kv = 2;
    cfg.d_ff = 4864;
    cfg.vocab_size = 151936;

    // Create TRANSPOSED Q weight: [d_model, n_head*head_dim] instead of [n_head*head_dim, d_model]
    auto wq_transposed = TensorFactory::create_simple({896, 896}); // Accidentally same shape!

    // Create TRANSPOSED K weight: [d_model, n_head_kv*head_dim] = [896, 128]
    // Expected: [n_head_kv*head_dim, d_model] = [128, 896]
    auto wk_transposed = TensorFactory::create_simple({896, 128});

    // Find K contract and verify it catches the error
    for (const auto &contract : contracts.layer_weights)
    {
        if (contract.weight_name == "wk")
        {
            EXPECT_THROW({ contract.validate(wk_transposed, cfg, 0); }, std::runtime_error) << "Should catch transposed K weight";
            break;
        }
    }

    // Create transposed FFN gate: [d_model, d_ff] = [896, 4864]
    // Expected: [d_ff, d_model] = [4864, 896]
    auto gate_transposed = TensorFactory::create_simple({896, 4864});

    for (const auto &contract : contracts.layer_weights)
    {
        if (contract.weight_name == "ffn_gate")
        {
            EXPECT_THROW({ contract.validate(gate_transposed, cfg, 0); }, std::runtime_error) << "Should catch transposed FFN gate";
            break;
        }
    }
}

/**
 * @brief Integration test demonstrating the contract system workflow
 */
TEST(WeightValidationIntegration, FullValidationWorkflow)
{
    // This test demonstrates the complete validation workflow:
    // 1. Get contracts for architecture
    auto contracts = getQwenWeightContracts();

    // 2. Set up model configuration
    TransformerLayerConfig cfg;
    cfg.d_model = 896;
    cfg.n_head = 14;
    cfg.n_head_kv = 2;
    cfg.d_ff = 4864;
    cfg.vocab_size = 151936;

    // 3. Create mock weights matching GGUF format
    std::map<std::string, std::shared_ptr<TensorBase>> weights;
    weights["token_embedding"] = TensorFactory::create_simple({151936, 896});
    weights["output_norm"] = TensorFactory::create_simple({896});
    weights["lm_head"] = TensorFactory::create_simple({151936, 896});

    // 4. Validate global weights
    for (const auto &contract : contracts.global_weights)
    {
        auto it = weights.find(contract.weight_name);
        if (it != weights.end())
        {
            EXPECT_NO_THROW({
                contract.validate(it->second, cfg);
            }) << "Global validation failed for "
               << contract.weight_name;
        }
    }

    // 5. Create layer weights
    std::map<std::string, std::shared_ptr<TensorBase>> layer_weights;
    layer_weights["wq"] = TensorFactory::create_simple({896, 896});
    layer_weights["wk"] = TensorFactory::create_simple({128, 896});
    layer_weights["wv"] = TensorFactory::create_simple({128, 896});
    layer_weights["attn_output"] = TensorFactory::create_simple({896, 896});
    layer_weights["attn_norm"] = TensorFactory::create_simple({896});
    layer_weights["ffn_gate"] = TensorFactory::create_simple({4864, 896});
    layer_weights["ffn_up"] = TensorFactory::create_simple({4864, 896});
    layer_weights["ffn_down"] = TensorFactory::create_simple({896, 4864});
    layer_weights["ffn_norm"] = TensorFactory::create_simple({896});

    // 6. Validate layer weights
    for (const auto &contract : contracts.layer_weights)
    {
        auto it = layer_weights.find(contract.weight_name);
        if (it != layer_weights.end())
        {
            EXPECT_NO_THROW({
                contract.validate(it->second, cfg, 0);
            }) << "Layer validation failed for "
               << contract.weight_name;
        }
    }

    // This demonstrates the complete workflow that happens in QwenModelWeights::validate()
}
