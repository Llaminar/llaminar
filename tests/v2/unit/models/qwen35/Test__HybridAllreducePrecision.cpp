/**
 * @file Test__HybridAllreducePrecision.cpp
 * @brief Unit tests for hybrid allreduce precision in Qwen3.5 architectures
 *
 * Validates that:
 * - GraphConfig::populateAllreducePrecision() with forced layers correctly
 *   assigns FP32 to all forced (FA) layers regardless of position
 * - GDN layers follow the standard count-based fp32_count policy
 * - Qwen35Schema exposes the correct default precision and fp32 count
 * - The GraphSchema tp_allreduce_fp32_forced_layers field is wired
 * - FusedAddAllreduceStage::Params includes a precision field
 */

#include <gtest/gtest.h>
#include <set>
#include <string>

#include "models/GraphTypes.h"
#include "models/qwen35/Qwen35Schema.h"
#include "models/qwen35moe/Qwen35MoESchema.h"
#include "execution/local_execution/graph/GraphSchema.h"
#include "execution/compute_stages/stages/FusedAddAllreduceStage.h"
#include "execution/compute_stages/stages/TPAllreduceStage.h"

using namespace llaminar2;

// ============================================================================
// populateAllreducePrecision — basic count-based overload (existing)
// ============================================================================

TEST(Test__HybridAllreducePrecision, CountBasedPopulate_AllFP32WhenCountEqualsLayers)
{
    GraphConfig config;
    config.n_layers = 8;
    config.populateAllreducePrecision("fp16", 8);

    for (int i = 0; i < 8; ++i)
    {
        EXPECT_EQ(config.getAllreducePrecisionForLayer(i), "fp32")
            << "Layer " << i << " should be fp32 when fp32_count == n_layers";
    }
}

TEST(Test__HybridAllreducePrecision, CountBasedPopulate_SplitPrecision)
{
    GraphConfig config;
    config.n_layers = 10;
    config.populateAllreducePrecision("fp16", 3);

    EXPECT_EQ(config.getAllreducePrecisionForLayer(0), "fp32");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(1), "fp32");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(2), "fp32");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(3), "fp16");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(9), "fp16");
}

TEST(Test__HybridAllreducePrecision, CountBasedPopulate_ZeroCount)
{
    GraphConfig config;
    config.n_layers = 4;
    config.populateAllreducePrecision("fp16", 0);

    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(config.getAllreducePrecisionForLayer(i), "fp16")
            << "Layer " << i << " should be fp16 when fp32_count == 0";
    }
}

// ============================================================================
// populateAllreducePrecision — forced-layer overload (new)
// ============================================================================

TEST(Test__HybridAllreducePrecision, ForcedLayersPopulate_FALayersAlwaysFP32)
{
    // Simulate Qwen3.5: 12 layers, FA every 4th (layers 3, 7, 11)
    GraphConfig config;
    config.n_layers = 12;
    std::set<int> fa_layers = {3, 7, 11};

    config.populateAllreducePrecision("fp16", 2, fa_layers);

    // FA layers: always fp32 regardless of position
    EXPECT_EQ(config.getAllreducePrecisionForLayer(3), "fp32");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(7), "fp32");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(11), "fp32");
}

TEST(Test__HybridAllreducePrecision, ForcedLayersPopulate_GDNLayersFollowCountPolicy)
{
    // 12 layers, FA at 3,7,11; fp32_count=2 applies to GDN layers only
    GraphConfig config;
    config.n_layers = 12;
    std::set<int> fa_layers = {3, 7, 11};

    config.populateAllreducePrecision("fp16", 2, fa_layers);

    // GDN layers: first 2 non-forced layers get fp32, rest fp16
    // GDN ordering: 0, 1, 2, 4, 5, 6, 8, 9, 10
    // non_forced_idx: 0  1  2  3  4  5  6  7   8
    EXPECT_EQ(config.getAllreducePrecisionForLayer(0), "fp32") << "GDN layer 0: first non-forced → fp32";
    EXPECT_EQ(config.getAllreducePrecisionForLayer(1), "fp32") << "GDN layer 1: second non-forced → fp32";
    EXPECT_EQ(config.getAllreducePrecisionForLayer(2), "fp16") << "GDN layer 2: third non-forced → fp16";
    EXPECT_EQ(config.getAllreducePrecisionForLayer(4), "fp16") << "GDN layer 4: beyond fp32_count → fp16";
    EXPECT_EQ(config.getAllreducePrecisionForLayer(5), "fp16");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(6), "fp16");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(8), "fp16");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(9), "fp16");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(10), "fp16");
}

TEST(Test__HybridAllreducePrecision, ForcedLayersPopulate_EmptyForcedSet_EquivalentToCountBased)
{
    GraphConfig config_forced;
    config_forced.n_layers = 8;
    config_forced.populateAllreducePrecision("fp16", 3, std::set<int>{});

    GraphConfig config_count;
    config_count.n_layers = 8;
    config_count.populateAllreducePrecision("fp16", 3);

    for (int i = 0; i < 8; ++i)
    {
        EXPECT_EQ(config_forced.getAllreducePrecisionForLayer(i),
                  config_count.getAllreducePrecisionForLayer(i))
            << "Layer " << i << ": forced with empty set should equal count-based";
    }
}

TEST(Test__HybridAllreducePrecision, ForcedLayersPopulate_AllLayersForced)
{
    GraphConfig config;
    config.n_layers = 4;
    std::set<int> all_forced = {0, 1, 2, 3};

    config.populateAllreducePrecision("fp16", 0, all_forced);

    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(config.getAllreducePrecisionForLayer(i), "fp32")
            << "Layer " << i << " should be fp32 when all layers are forced";
    }
}

TEST(Test__HybridAllreducePrecision, ForcedLayersPopulate_ZeroFP32CountWithForced)
{
    // fp32_count=0 means NO GDN layers get FP32, only forced layers do
    GraphConfig config;
    config.n_layers = 8;
    std::set<int> fa_layers = {3, 7};

    config.populateAllreducePrecision("fp16", 0, fa_layers);

    // All GDN layers should be fp16
    EXPECT_EQ(config.getAllreducePrecisionForLayer(0), "fp16");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(1), "fp16");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(2), "fp16");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(4), "fp16");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(5), "fp16");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(6), "fp16");

    // FA layers should be fp32
    EXPECT_EQ(config.getAllreducePrecisionForLayer(3), "fp32");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(7), "fp32");
}

// ============================================================================
// Qwen3.5-like topology: 40 layers, FA every 4th
// ============================================================================

TEST(Test__HybridAllreducePrecision, Qwen35MoE_40Layers_FAEvery4th)
{
    // Mirrors the real Qwen3.5 MoE 35B: 40 layers, FA at 3,7,11,...,39
    GraphConfig config;
    config.n_layers = 40;

    std::set<int> fa_layers;
    config.layer_types.resize(40);
    for (int i = 0; i < 40; ++i)
    {
        if ((i + 1) % 4 == 0)
        {
            fa_layers.insert(i);
            config.layer_types[i] = "full_attention";
        }
        else
        {
            config.layer_types[i] = "gdn";
        }
    }

    EXPECT_EQ(fa_layers.size(), 10u);

    config.populateAllreducePrecision("fp16", 6, fa_layers);

    // Verify all 10 FA layers are FP32
    for (int fa : fa_layers)
    {
        EXPECT_EQ(config.getAllreducePrecisionForLayer(fa), "fp32")
            << "FA layer " << fa << " must be fp32";
    }

    // Verify GDN layer distribution:
    // GDN order: 0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22, ...
    // First 6 GDN layers (indices 0,1,2,4,5,6) → fp32
    // Rest (8,9,10,12,...) → fp16
    int gdn_fp32_count = 0;
    int gdn_fp16_count = 0;
    for (int i = 0; i < 40; ++i)
    {
        if (fa_layers.count(i))
            continue;

        std::string prec = config.getAllreducePrecisionForLayer(i);
        if (prec == "fp32")
            ++gdn_fp32_count;
        else if (prec == "fp16")
            ++gdn_fp16_count;
    }

    EXPECT_EQ(gdn_fp32_count, 6) << "First 6 GDN layers should be fp32";
    EXPECT_EQ(gdn_fp16_count, 24) << "Remaining 24 GDN layers should be fp16";
}

TEST(Test__HybridAllreducePrecision, Qwen35Dense_24Layers_FAEvery4th)
{
    // Qwen3.5 0.8B dense: 24 layers, FA at 3,7,11,15,19,23
    GraphConfig config;
    config.n_layers = 24;

    std::set<int> fa_layers;
    for (int i = 0; i < 24; ++i)
    {
        if ((i + 1) % 4 == 0)
            fa_layers.insert(i);
    }

    EXPECT_EQ(fa_layers.size(), 6u);

    config.populateAllreducePrecision("fp16", 6, fa_layers);

    // All FA layers: FP32
    for (int fa : fa_layers)
    {
        EXPECT_EQ(config.getAllreducePrecisionForLayer(fa), "fp32");
    }

    // First 6 GDN layers (0,1,2, 4,5,6): FP32
    // Remaining GDN layers (8,9,10, 12,...): FP16
    EXPECT_EQ(config.getAllreducePrecisionForLayer(0), "fp32");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(6), "fp32"); // 6th GDN (indices: 0,1,2,4,5,6)
    EXPECT_EQ(config.getAllreducePrecisionForLayer(8), "fp16"); // 7th GDN → beyond count
}

// ============================================================================
// getAllreducePrecisionForLayer — fallback behavior
// ============================================================================

TEST(Test__HybridAllreducePrecision, EmptyMap_ReturnsEmptyString)
{
    GraphConfig config;
    config.n_layers = 8;
    // Don't call populateAllreducePrecision — map stays empty

    EXPECT_EQ(config.getAllreducePrecisionForLayer(0), "");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(7), "");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(100), "");
}

TEST(Test__HybridAllreducePrecision, PopulateClearsExistingEntries)
{
    GraphConfig config;
    config.n_layers = 4;

    // First populate
    config.populateAllreducePrecision("fp16", 2);
    EXPECT_EQ(config.getAllreducePrecisionForLayer(0), "fp32");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(3), "fp16");

    // Re-populate with different settings
    config.populateAllreducePrecision("bf16", 0);
    EXPECT_EQ(config.getAllreducePrecisionForLayer(0), "bf16");
    EXPECT_EQ(config.getAllreducePrecisionForLayer(3), "bf16");
}

// ============================================================================
// Schema: verify Qwen35 and Qwen35MoE schema settings
// ============================================================================

TEST(Test__HybridAllreducePrecision, Qwen35SchemaDefaults)
{
    Qwen35SchemaFactory factory;
    auto schema = factory.createSchema();

    EXPECT_EQ(schema.tp_allreduce_default_precision, "fp16");
    EXPECT_EQ(schema.tp_allreduce_fp32_layer_count, 6);
}

TEST(Test__HybridAllreducePrecision, Qwen35MoESchemaInheritsDefaults)
{
    Qwen35MoESchemaFactory factory;
    auto schema = factory.createSchema();

    // MoE schema inherits from Qwen35, same allreduce precision policy
    EXPECT_EQ(schema.tp_allreduce_default_precision, "fp16");
    EXPECT_EQ(schema.tp_allreduce_fp32_layer_count, 6);
}

TEST(Test__HybridAllreducePrecision, GraphSchemaForcedLayersField)
{
    GraphSchema schema;

    // Default: empty forced layers
    EXPECT_TRUE(schema.tp_allreduce_fp32_forced_layers.empty());

    // Can be populated
    schema.tp_allreduce_fp32_forced_layers = {3, 7, 11};
    EXPECT_EQ(schema.tp_allreduce_fp32_forced_layers.size(), 3u);
    EXPECT_EQ(schema.tp_allreduce_fp32_forced_layers[0], 3);
    EXPECT_EQ(schema.tp_allreduce_fp32_forced_layers[1], 7);
    EXPECT_EQ(schema.tp_allreduce_fp32_forced_layers[2], 11);
}

// ============================================================================
// FusedAddAllreduceStage: precision field wiring
// ============================================================================

TEST(Test__HybridAllreducePrecision, FusedAddAllreduceParamsHasPrecision)
{
    FusedAddAllreduceStage::Params params;

    // Default: empty (defers to global)
    EXPECT_TRUE(params.precision.empty());

    // Can be set per-layer
    params.precision = "fp32";
    EXPECT_EQ(params.precision, "fp32");

    params.precision = "fp16";
    EXPECT_EQ(params.precision, "fp16");
}

TEST(Test__HybridAllreducePrecision, TPAllreduceParamsHasPrecision)
{
    TPAllreduceParams params;

    // Baseline: TPAllreduceStage already had precision
    EXPECT_TRUE(params.precision.empty());

    params.precision = "fp32";
    EXPECT_EQ(params.precision, "fp32");
}

// ============================================================================
// Integration: end-to-end precision map for realistic hybrid topology
// ============================================================================

TEST(Test__HybridAllreducePrecision, EndToEnd_PrecisionMapMatchesLayerTypes)
{
    // Build a config mimicking Qwen35GraphConfigBuilder output
    GraphConfig config;
    config.n_layers = 16;
    config.layer_types.resize(16);

    // FA at layers 3, 7, 11, 15 (interval=4)
    std::set<int> fa_layers;
    for (int i = 0; i < 16; ++i)
    {
        if ((i + 1) % 4 == 0)
        {
            config.layer_types[i] = "full_attention";
            fa_layers.insert(i);
        }
        else
        {
            config.layer_types[i] = "gdn";
        }
    }

    // Simulate what Qwen35Graph constructor does:
    // Read schema defaults, build forced set from layer_types, populate
    Qwen35SchemaFactory factory;
    auto schema = factory.createSchema();

    config.populateAllreducePrecision(
        schema.tp_allreduce_default_precision,
        schema.tp_allreduce_fp32_layer_count,
        fa_layers);

    // Verify every layer has an explicit precision (no empty strings)
    for (int i = 0; i < 16; ++i)
    {
        std::string prec = config.getAllreducePrecisionForLayer(i);
        EXPECT_FALSE(prec.empty()) << "Layer " << i << " should have explicit precision";
        EXPECT_TRUE(prec == "fp32" || prec == "fp16")
            << "Layer " << i << " has unexpected precision: " << prec;
    }

    // Verify FA layers all FP32
    for (int fa : fa_layers)
    {
        EXPECT_EQ(config.getAllreducePrecisionForLayer(fa), "fp32")
            << "FA layer " << fa << " must be fp32";
    }

    // Verify at least some GDN layers are fp16 (otherwise what's the point)
    bool has_fp16_gdn = false;
    for (int i = 0; i < 16; ++i)
    {
        if (!fa_layers.count(i) && config.getAllreducePrecisionForLayer(i) == "fp16")
        {
            has_fp16_gdn = true;
            break;
        }
    }
    EXPECT_TRUE(has_fp16_gdn) << "At least some GDN layers should use fp16";
}
