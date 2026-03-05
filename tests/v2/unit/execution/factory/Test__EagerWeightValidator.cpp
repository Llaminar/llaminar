/**
 * @file Test__EagerWeightValidator.cpp
 * @brief Unit tests for EagerWeightValidator schema-aware weight validation
 *
 * Tests that:
 * 1. All required weights present → validation succeeds
 * 2. Optional weights missing → validation succeeds, missing_optional populated
 * 3. Required weight missing → validation fails with clear error
 * 4. PP layer range validation only checks the given range
 * 5. Works correctly with both Qwen2 and Qwen3 schemas
 */

#include <gtest/gtest.h>

#include "execution/factory/EagerWeightValidator.h"
#include "models/qwen/Qwen2Schema.h"
#include "models/qwen3/Qwen3Schema.h"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

using namespace llaminar2;

namespace
{

    // =========================================================================
    // Helpers
    // =========================================================================

    /// Build the full set of weight names for a given layer count, including
    /// all possible weights (required + optional for any schema).
    std::unordered_set<std::string> buildFullWeightSet(int n_layers)
    {
        std::unordered_set<std::string> weights;
        for (int i = 0; i < n_layers; ++i)
        {
            std::string prefix = "blk." + std::to_string(i) + ".";
            // Core attention
            weights.insert(prefix + "attn_q.weight");
            weights.insert(prefix + "attn_k.weight");
            weights.insert(prefix + "attn_v.weight");
            weights.insert(prefix + "attn_output.weight");
            weights.insert(prefix + "attn_norm.weight");
            // Biases (present in some Qwen2 variants, absent in Qwen3)
            weights.insert(prefix + "attn_q.bias");
            weights.insert(prefix + "attn_k.bias");
            weights.insert(prefix + "attn_v.bias");
            // QK norm (present in Qwen3, absent in Qwen2)
            weights.insert(prefix + "attn_q_norm.weight");
            weights.insert(prefix + "attn_k_norm.weight");
            // FFN
            weights.insert(prefix + "ffn_gate.weight");
            weights.insert(prefix + "ffn_up.weight");
            weights.insert(prefix + "ffn_down.weight");
            weights.insert(prefix + "ffn_norm.weight");
        }
        return weights;
    }

    /// Predicate wrapper around an unordered_set
    auto makePredicate(const std::unordered_set<std::string> &set)
    {
        return [&set](const std::string &name) -> bool
        {
            return set.count(name) > 0;
        };
    }

    /// Check if a string is in a vector
    bool contains(const std::vector<std::string> &vec, const std::string &val)
    {
        return std::find(vec.begin(), vec.end(), val) != vec.end();
    }

    /// Check if a weight name is in the weights_to_load list
    bool inLoadList(const std::vector<std::pair<std::string, bool>> &list, const std::string &name)
    {
        return std::any_of(list.begin(), list.end(),
                           [&name](const auto &p)
                           { return p.first == name; });
    }

} // anonymous namespace

// =============================================================================
// Test: All weights present, Qwen2 schema
// =============================================================================
TEST(Test__EagerWeightValidator, AllWeightsPresent_Qwen2)
{
    Qwen2SchemaFactory schema;
    constexpr int n_layers = 2;
    auto weights = buildFullWeightSet(n_layers);

    auto result = validateLayerWeights(schema, n_layers, makePredicate(weights));

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.missing_required.empty());
    EXPECT_TRUE(result.missing_optional.empty());
    EXPECT_TRUE(result.error_message().empty());
    // All 14 weights × 2 layers = 28
    EXPECT_EQ(result.weights_to_load.size(), 28u);
}

// =============================================================================
// Test: All weights present, Qwen3 schema
// =============================================================================
TEST(Test__EagerWeightValidator, AllWeightsPresent_Qwen3)
{
    Qwen3SchemaFactory schema;
    constexpr int n_layers = 2;
    auto weights = buildFullWeightSet(n_layers);

    auto result = validateLayerWeights(schema, n_layers, makePredicate(weights));

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.missing_required.empty());
    EXPECT_TRUE(result.missing_optional.empty());
    EXPECT_EQ(result.weights_to_load.size(), 28u);
}

// =============================================================================
// Test: Qwen3 model with no biases and no QK norm → success, optional missing
// This is the actual Qwen3 scenario that was producing noisy ERROR logs.
// =============================================================================
TEST(Test__EagerWeightValidator, Qwen3_NoBiases_NoQKNorm)
{
    Qwen3SchemaFactory schema;
    constexpr int n_layers = 2;
    auto weights = buildFullWeightSet(n_layers);

    // Remove all biases (Qwen3 never has them)
    for (int i = 0; i < n_layers; ++i)
    {
        std::string prefix = "blk." + std::to_string(i) + ".";
        weights.erase(prefix + "attn_q.bias");
        weights.erase(prefix + "attn_k.bias");
        weights.erase(prefix + "attn_v.bias");
    }

    auto result = validateLayerWeights(schema, n_layers, makePredicate(weights));

    EXPECT_TRUE(result.success) << "Missing optional weights should not cause failure";
    EXPECT_TRUE(result.missing_required.empty());
    // 3 biases × 2 layers = 6 optional missing
    EXPECT_EQ(result.missing_optional.size(), 6u);
    // 11 weights per layer × 2 layers = 22
    EXPECT_EQ(result.weights_to_load.size(), 22u);

    // Verify missing optionals are the bias weights
    EXPECT_TRUE(contains(result.missing_optional, "blk.0.attn_q.bias"));
    EXPECT_TRUE(contains(result.missing_optional, "blk.0.attn_k.bias"));
    EXPECT_TRUE(contains(result.missing_optional, "blk.0.attn_v.bias"));
    EXPECT_TRUE(contains(result.missing_optional, "blk.1.attn_q.bias"));

    // And biases should NOT be in the load list
    EXPECT_FALSE(inLoadList(result.weights_to_load, "blk.0.attn_q.bias"));
    EXPECT_FALSE(inLoadList(result.weights_to_load, "blk.1.attn_v.bias"));
}

// =============================================================================
// Test: Qwen3 model with QK norm present but no biases → success
// This is the actual Qwen3-0.6B scenario: has QK norm, no biases.
// =============================================================================
TEST(Test__EagerWeightValidator, Qwen3_WithQKNorm_NoBiases)
{
    Qwen3SchemaFactory schema;
    constexpr int n_layers = 2;
    auto weights = buildFullWeightSet(n_layers);

    // Remove only biases (QK norm stays, as in real Qwen3)
    for (int i = 0; i < n_layers; ++i)
    {
        std::string prefix = "blk." + std::to_string(i) + ".";
        weights.erase(prefix + "attn_q.bias");
        weights.erase(prefix + "attn_k.bias");
        weights.erase(prefix + "attn_v.bias");
    }

    auto result = validateLayerWeights(schema, n_layers, makePredicate(weights));

    EXPECT_TRUE(result.success);
    // 3 biases × 2 layers = 6 missing optional
    EXPECT_EQ(result.missing_optional.size(), 6u);
    // QK norm weights should be in load list
    EXPECT_TRUE(inLoadList(result.weights_to_load, "blk.0.attn_q_norm.weight"));
    EXPECT_TRUE(inLoadList(result.weights_to_load, "blk.0.attn_k_norm.weight"));
}

// =============================================================================
// Test: Qwen2 model with no biases and no QK norm → success
// Qwen2 marks both biases and QK norm as optional (some variants have biases)
// =============================================================================
TEST(Test__EagerWeightValidator, Qwen2_NoBiases_NoQKNorm)
{
    Qwen2SchemaFactory schema;
    constexpr int n_layers = 2;
    auto weights = buildFullWeightSet(n_layers);

    // Remove biases + QK norm (mimics standard Qwen2-0.5B)
    for (int i = 0; i < n_layers; ++i)
    {
        std::string prefix = "blk." + std::to_string(i) + ".";
        weights.erase(prefix + "attn_q.bias");
        weights.erase(prefix + "attn_k.bias");
        weights.erase(prefix + "attn_v.bias");
        weights.erase(prefix + "attn_q_norm.weight");
        weights.erase(prefix + "attn_k_norm.weight");
    }

    auto result = validateLayerWeights(schema, n_layers, makePredicate(weights));

    EXPECT_TRUE(result.success);
    // 5 optional missing per layer × 2 = 10
    EXPECT_EQ(result.missing_optional.size(), 10u);
    // 9 required weights per layer × 2 = 18
    EXPECT_EQ(result.weights_to_load.size(), 18u);
}

// =============================================================================
// Test: Missing REQUIRED weight → fatal failure
// If attn_q.weight is missing, that's a broken model file.
// =============================================================================
TEST(Test__EagerWeightValidator, MissingRequiredWeight_SingleLayer)
{
    Qwen3SchemaFactory schema;
    constexpr int n_layers = 1;
    auto weights = buildFullWeightSet(n_layers);

    // Remove a required weight
    weights.erase("blk.0.attn_q.weight");

    auto result = validateLayerWeights(schema, n_layers, makePredicate(weights));

    EXPECT_FALSE(result.success) << "Missing required weight must cause failure";
    EXPECT_EQ(result.missing_required.size(), 1u);
    EXPECT_TRUE(contains(result.missing_required, "blk.0.attn_q.weight"));
    EXPECT_FALSE(result.error_message().empty());
    EXPECT_NE(result.error_message().find("blk.0.attn_q.weight"), std::string::npos);
}

// =============================================================================
// Test: Multiple missing required weights → all reported
// =============================================================================
TEST(Test__EagerWeightValidator, MissingMultipleRequiredWeights)
{
    Qwen2SchemaFactory schema;
    constexpr int n_layers = 2;
    auto weights = buildFullWeightSet(n_layers);

    // Remove several required weights across different layers
    weights.erase("blk.0.attn_output.weight");
    weights.erase("blk.0.ffn_gate.weight");
    weights.erase("blk.1.attn_norm.weight");

    auto result = validateLayerWeights(schema, n_layers, makePredicate(weights));

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.missing_required.size(), 3u);
    EXPECT_TRUE(contains(result.missing_required, "blk.0.attn_output.weight"));
    EXPECT_TRUE(contains(result.missing_required, "blk.0.ffn_gate.weight"));
    EXPECT_TRUE(contains(result.missing_required, "blk.1.attn_norm.weight"));

    // Error message should list all missing
    auto msg = result.error_message();
    EXPECT_NE(msg.find("blk.0.attn_output.weight"), std::string::npos);
    EXPECT_NE(msg.find("blk.0.ffn_gate.weight"), std::string::npos);
    EXPECT_NE(msg.find("blk.1.attn_norm.weight"), std::string::npos);
}

// =============================================================================
// Test: Missing ALL FFN weights for a layer → fatal (all are required)
// =============================================================================
TEST(Test__EagerWeightValidator, MissingAllFFNWeights)
{
    Qwen3SchemaFactory schema;
    constexpr int n_layers = 1;
    auto weights = buildFullWeightSet(n_layers);

    weights.erase("blk.0.ffn_gate.weight");
    weights.erase("blk.0.ffn_up.weight");
    weights.erase("blk.0.ffn_down.weight");
    weights.erase("blk.0.ffn_norm.weight");

    auto result = validateLayerWeights(schema, n_layers, makePredicate(weights));

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.missing_required.size(), 4u);
}

// =============================================================================
// Test: PP layer range — only validates layers within [first_layer, last_layer)
// =============================================================================
TEST(Test__EagerWeightValidator, PPLayerRange_OnlyChecksRange)
{
    Qwen3SchemaFactory schema;
    constexpr int n_layers = 4;

    // Only provide weights for layers 1 and 2 (PP stage covers layers 1-2)
    std::unordered_set<std::string> weights;
    for (int i = 1; i <= 2; ++i)
    {
        std::string prefix = "blk." + std::to_string(i) + ".";
        weights.insert(prefix + "attn_q.weight");
        weights.insert(prefix + "attn_k.weight");
        weights.insert(prefix + "attn_v.weight");
        weights.insert(prefix + "attn_output.weight");
        weights.insert(prefix + "attn_norm.weight");
        weights.insert(prefix + "attn_q_norm.weight");
        weights.insert(prefix + "attn_k_norm.weight");
        weights.insert(prefix + "ffn_gate.weight");
        weights.insert(prefix + "ffn_up.weight");
        weights.insert(prefix + "ffn_down.weight");
        weights.insert(prefix + "ffn_norm.weight");
    }

    // Validate only layers 1-2 (not 0 or 3)
    auto result = validateLayerWeights(schema, n_layers, makePredicate(weights),
                                       /*first_layer=*/1, /*last_layer=*/3);

    EXPECT_TRUE(result.success);
    // 3 biases × 2 layers = 6 optional missing (biases not provided)
    EXPECT_EQ(result.missing_optional.size(), 6u);
    // 11 weights × 2 layers = 22
    EXPECT_EQ(result.weights_to_load.size(), 22u);
}

// =============================================================================
// Test: PP layer range — missing required weight in range → fatal
// =============================================================================
TEST(Test__EagerWeightValidator, PPLayerRange_MissingRequiredInRange)
{
    Qwen3SchemaFactory schema;
    constexpr int n_layers = 4;

    // Provide all weights for layer 1 but incomplete for layer 2
    std::unordered_set<std::string> weights;
    for (int i = 1; i <= 2; ++i)
    {
        std::string prefix = "blk." + std::to_string(i) + ".";
        weights.insert(prefix + "attn_q.weight");
        weights.insert(prefix + "attn_k.weight");
        weights.insert(prefix + "attn_v.weight");
        weights.insert(prefix + "attn_output.weight");
        weights.insert(prefix + "attn_norm.weight");
        weights.insert(prefix + "attn_q_norm.weight");
        weights.insert(prefix + "attn_k_norm.weight");
        weights.insert(prefix + "ffn_gate.weight");
        weights.insert(prefix + "ffn_up.weight");
        weights.insert(prefix + "ffn_down.weight");
        weights.insert(prefix + "ffn_norm.weight");
    }
    // Remove a required weight from layer 2 (still in range)
    weights.erase("blk.2.ffn_down.weight");

    auto result = validateLayerWeights(schema, n_layers, makePredicate(weights),
                                       /*first_layer=*/1, /*last_layer=*/3);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.missing_required.size(), 1u);
    EXPECT_TRUE(contains(result.missing_required, "blk.2.ffn_down.weight"));
}

// =============================================================================
// Test: PP layer range — missing weight OUTSIDE range is ignored
// =============================================================================
TEST(Test__EagerWeightValidator, PPLayerRange_MissingOutsideRange_Ignored)
{
    Qwen2SchemaFactory schema;
    constexpr int n_layers = 4;

    // Only provide weights for layer 0 (the one we'll check)
    auto all_weights = buildFullWeightSet(n_layers);
    // Remove required weights from layers 1-3 (outside range [0,1))
    for (int i = 1; i < n_layers; ++i)
    {
        std::string prefix = "blk." + std::to_string(i) + ".";
        all_weights.erase(prefix + "attn_q.weight");
    }

    // Validate only layer 0
    auto result = validateLayerWeights(schema, n_layers, makePredicate(all_weights),
                                       /*first_layer=*/0, /*last_layer=*/1);

    EXPECT_TRUE(result.success) << "Weights outside the PP range should not affect validation";
}

// =============================================================================
// Test: Default last_layer (-1) means n_layers
// =============================================================================
TEST(Test__EagerWeightValidator, DefaultLastLayer_CoversAll)
{
    Qwen3SchemaFactory schema;
    constexpr int n_layers = 3;
    auto weights = buildFullWeightSet(n_layers);

    // Remove biases (optional for Qwen3)
    for (int i = 0; i < n_layers; ++i)
    {
        std::string prefix = "blk." + std::to_string(i) + ".";
        weights.erase(prefix + "attn_q.bias");
        weights.erase(prefix + "attn_k.bias");
        weights.erase(prefix + "attn_v.bias");
    }

    // Default last_layer = -1 → should cover all 3 layers
    auto result = validateLayerWeights(schema, n_layers, makePredicate(weights));

    EXPECT_TRUE(result.success);
    // 3 biases × 3 layers = 9 optional missing
    EXPECT_EQ(result.missing_optional.size(), 9u);
    // 11 weights × 3 layers = 33
    EXPECT_EQ(result.weights_to_load.size(), 33u);
}

// =============================================================================
// Test: weights_to_load tracks optionality correctly
// =============================================================================
TEST(Test__EagerWeightValidator, WeightsToLoad_TracksOptionality)
{
    Qwen3SchemaFactory schema;
    constexpr int n_layers = 1;
    auto weights = buildFullWeightSet(n_layers);

    auto result = validateLayerWeights(schema, n_layers, makePredicate(weights));

    EXPECT_TRUE(result.success);

    // Find attn_q.weight (required) and attn_q.bias (optional) in load list
    bool found_required = false;
    bool found_optional = false;
    for (const auto &[name, is_optional] : result.weights_to_load)
    {
        if (name == "blk.0.attn_q.weight")
        {
            EXPECT_FALSE(is_optional) << "attn_q.weight should be required";
            found_required = true;
        }
        if (name == "blk.0.attn_q.bias")
        {
            EXPECT_TRUE(is_optional) << "attn_q.bias should be optional for Qwen3";
            found_optional = true;
        }
    }
    EXPECT_TRUE(found_required);
    EXPECT_TRUE(found_optional) << "When bias IS present, it should appear in load list as optional";
}

// =============================================================================
// Test: Empty model (no weights at all) → all required missing
// =============================================================================
TEST(Test__EagerWeightValidator, EmptyModel_AllRequiredMissing)
{
    Qwen3SchemaFactory schema;
    constexpr int n_layers = 1;
    std::unordered_set<std::string> empty;

    auto result = validateLayerWeights(schema, n_layers, makePredicate(empty));

    EXPECT_FALSE(result.success);
    // 9 required weights for Qwen3 (5 attn + 4 FFN, no biases/norms are optional)
    EXPECT_EQ(result.missing_required.size(), 9u);
    // 5 optional weights for Qwen3 (3 biases + 2 QK norms)
    EXPECT_EQ(result.missing_optional.size(), 5u);
    EXPECT_TRUE(result.weights_to_load.empty());
}

// =============================================================================
// Test: Zero layers → trivially succeeds
// =============================================================================
TEST(Test__EagerWeightValidator, ZeroLayers_TriviallySucceeds)
{
    Qwen3SchemaFactory schema;
    std::unordered_set<std::string> empty;

    auto result = validateLayerWeights(schema, 0, makePredicate(empty));

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.missing_required.empty());
    EXPECT_TRUE(result.missing_optional.empty());
    EXPECT_TRUE(result.weights_to_load.empty());
}
