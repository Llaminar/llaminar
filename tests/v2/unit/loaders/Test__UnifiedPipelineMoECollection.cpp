/**
 * @file Test__UnifiedPipelineMoECollection.cpp
 * @brief Unit tests for MoE expert weight collection in the unified GPU pipeline
 *
 * Verifies:
 * - parseMoELayerIndex logic (replicated from WeightManager.cpp static helper)
 * - quantizedViewRawBytes calculation for 2D expert views
 * - 3D→2D view creation for expert slicing
 * - Layer filter is respected for MoE weights
 * - Incomplete layer handling (missing gate/up/down)
 */

#include <gtest/gtest.h>
#include "tensors/TensorClasses.h"
#include "tensors/BlockStructures.h"
#include "tensors/TensorType.h"
#include "loaders/ExpertGemmRegistry.h"
#include "utils/TestTensorFactory.h"

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <cstring>

using namespace llaminar2;
using namespace llaminar2::test;

using namespace llaminar2;

// =========================================================================
// Replicate static helpers from WeightManager.cpp for testability
// =========================================================================

namespace
{

static int parseMoELayerIndex(const std::string &name)
{
    if (name.compare(0, 4, "blk.") != 0)
        return -1;
    auto dot2 = name.find('.', 4);
    if (dot2 == std::string::npos)
        return -1;
    try
    {
        return std::stoi(name.substr(4, dot2 - 4));
    }
    catch (...)
    {
        return -1;
    }
}

static size_t quantizedViewRawBytes(const TensorBase &tensor)
{
    const size_t reported = tensor.size_bytes();
    if (reported > 0)
        return reported;

    const auto &shape = tensor.shape();
    if (shape.size() != 2)
        return 0;

    const size_t rows = shape[0];
    const size_t cols = shape[1];
    auto bytes_for = [rows, cols](size_t block_size, size_t block_bytes) -> size_t
    {
        const size_t blocks_per_row = (cols + block_size - 1) / block_size;
        return rows * blocks_per_row * block_bytes;
    };

    switch (tensor.native_type())
    {
    case TensorType::IQ4_NL: return bytes_for(IQ4_NLBlock::BLOCK_SIZE, sizeof(IQ4_NLBlock));
    case TensorType::IQ4_XS: return bytes_for(IQ4_XSBlock::BLOCK_SIZE, sizeof(IQ4_XSBlock));
    case TensorType::Q8_0: return bytes_for(Q8_0Block::BLOCK_SIZE, sizeof(Q8_0Block));
    case TensorType::Q4_0: return bytes_for(Q4_0Block::BLOCK_SIZE, sizeof(Q4_0Block));
    case TensorType::Q4_1: return bytes_for(Q4_1Block::BLOCK_SIZE, sizeof(Q4_1Block));
    case TensorType::Q5_0: return bytes_for(Q5_0Block::BLOCK_SIZE, sizeof(Q5_0Block));
    case TensorType::Q5_1: return bytes_for(Q5_1Block::BLOCK_SIZE, sizeof(Q5_1Block));
    case TensorType::Q2_K: return bytes_for(Q2_KBlock::BLOCK_SIZE, sizeof(Q2_KBlock));
    case TensorType::Q3_K: return bytes_for(Q3_KBlock::BLOCK_SIZE, sizeof(Q3_KBlock));
    case TensorType::Q4_K: return bytes_for(Q4_KBlock::BLOCK_SIZE, sizeof(Q4_KBlock));
    case TensorType::Q5_K: return bytes_for(Q5_KBlock::BLOCK_SIZE, sizeof(Q5_KBlock));
    case TensorType::Q6_K: return bytes_for(Q6_KBlock::BLOCK_SIZE, sizeof(Q6_KBlock));
    case TensorType::Q8_K: return bytes_for(Q8_KBlock::BLOCK_SIZE, sizeof(Q8_KBlock));
    case TensorType::IQ2_XXS: return bytes_for(IQ2_XXSBlock::BLOCK_SIZE, sizeof(IQ2_XXSBlock));
    case TensorType::IQ2_XS: return bytes_for(IQ2_XSBlock::BLOCK_SIZE, sizeof(IQ2_XSBlock));
    case TensorType::IQ2_S: return bytes_for(IQ2_SBlock::BLOCK_SIZE, sizeof(IQ2_SBlock));
    case TensorType::IQ3_XXS: return bytes_for(IQ3_XXSBlock::BLOCK_SIZE, sizeof(IQ3_XXSBlock));
    case TensorType::IQ3_S: return bytes_for(IQ3_SBlock::BLOCK_SIZE, sizeof(IQ3_SBlock));
    case TensorType::IQ1_S: return bytes_for(IQ1_SBlock::BLOCK_SIZE, sizeof(IQ1_SBlock));
    case TensorType::IQ1_M: return bytes_for(IQ1_MBlock::BLOCK_SIZE, sizeof(IQ1_MBlock));
    default: return 0;
    }
}

} // anonymous namespace

// =========================================================================
// Helper: Create Q4_0 tensor with random data for arbitrary shape (2D or 3D)
// =========================================================================
namespace
{

std::shared_ptr<Q4_0Tensor> createQ4_0WithData(const std::vector<size_t> &shape, uint32_t seed = 42)
{
    // For Q4_0: block_size=32, sizeof(Q4_0Block)=18
    // Total elements = product of all dims
    // Blocks are packed row-major. For 3D: [d0, d1, d2] → total_rows = d1*d2 (if 3D) or d0 (if 2D)
    // K (cols) = d0 (3D GGUF) or d1 (2D)

    size_t total_elements = 1;
    for (auto d : shape)
        total_elements *= d;

    // Cols is always last contiguous dim for block quantization
    // For GGUF 3D: shape[0]=cols, total_rows = shape[1]*shape[2]
    // For 2D: shape[0]=rows, shape[1]=cols
    size_t cols = (shape.size() == 3) ? shape[0] : shape.back();
    size_t total_rows = total_elements / cols;
    size_t blocks_per_row = (cols + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;
    size_t total_blocks = total_rows * blocks_per_row;

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    std::vector<uint8_t> raw_data(total_blocks * sizeof(Q4_0Block));
    auto *blocks = reinterpret_cast<Q4_0Block *>(raw_data.data());

    for (size_t i = 0; i < total_blocks; ++i)
    {
        float max_abs = 0.0f;
        float values[Q4_0Block::BLOCK_SIZE];
        for (size_t j = 0; j < Q4_0Block::BLOCK_SIZE; ++j)
        {
            values[j] = dist(rng);
            max_abs = std::max(max_abs, std::abs(values[j]));
        }
        float scale = max_abs / 7.0f;
        // Simple FP32→FP16 conversion
        uint16_t fp16_scale;
        {
            uint32_t x;
            std::memcpy(&x, &scale, sizeof(x));
            uint32_t sign = (x >> 31) & 0x1;
            int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = (x >> 13) & 0x3FF;
            if (exp <= 0)
                fp16_scale = static_cast<uint16_t>(sign << 15);
            else if (exp >= 31)
                fp16_scale = static_cast<uint16_t>((sign << 15) | 0x7C00);
            else
                fp16_scale = static_cast<uint16_t>((sign << 15) | (exp << 10) | mant);
        }
        blocks[i].d = fp16_scale;
        float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
        for (size_t j = 0; j < Q4_0Block::BLOCK_SIZE / 2; ++j)
        {
            int32_t q0 = static_cast<int32_t>(std::round(values[2 * j] * inv_scale)) + 8;
            int32_t q1 = static_cast<int32_t>(std::round(values[2 * j + 1] * inv_scale)) + 8;
            q0 = std::clamp(q0, 0, 15);
            q1 = std::clamp(q1, 0, 15);
            blocks[i].qs[j] = static_cast<uint8_t>((q1 << 4) | q0);
        }
    }

    return std::make_shared<Q4_0Tensor>(shape, raw_data);
}

} // anonymous namespace

// =========================================================================
// parseMoELayerIndex tests
// =========================================================================

TEST(Test__UnifiedPipelineMoECollection, ParseLayerIndex_GateExps)
{
    EXPECT_EQ(parseMoELayerIndex("blk.0.ffn_gate_exps.weight"), 0);
    EXPECT_EQ(parseMoELayerIndex("blk.5.ffn_gate_exps.weight"), 5);
    EXPECT_EQ(parseMoELayerIndex("blk.23.ffn_gate_exps.weight"), 23);
}

TEST(Test__UnifiedPipelineMoECollection, ParseLayerIndex_UpExps)
{
    EXPECT_EQ(parseMoELayerIndex("blk.0.ffn_up_exps.weight"), 0);
    EXPECT_EQ(parseMoELayerIndex("blk.12.ffn_up_exps.weight"), 12);
}

TEST(Test__UnifiedPipelineMoECollection, ParseLayerIndex_DownExps)
{
    EXPECT_EQ(parseMoELayerIndex("blk.7.ffn_down_exps.weight"), 7);
}

TEST(Test__UnifiedPipelineMoECollection, ParseLayerIndex_InvalidNames)
{
    EXPECT_EQ(parseMoELayerIndex("output.weight"), -1);
    EXPECT_EQ(parseMoELayerIndex("blk"), -1);
    EXPECT_EQ(parseMoELayerIndex("blk."), -1);
    EXPECT_EQ(parseMoELayerIndex("blk.abc.ffn_gate_exps.weight"), -1);
    EXPECT_EQ(parseMoELayerIndex(""), -1);
    EXPECT_EQ(parseMoELayerIndex("foo.0.bar"), -1);
}

// =========================================================================
// quantizedViewRawBytes tests
// =========================================================================

TEST(Test__UnifiedPipelineMoECollection, QuantizedViewRawBytes_Q4_0)
{
    // Q4_0: block_size=32, sizeof(Q4_0Block)=18 bytes (FP16 scale + 16 nibble bytes)
    // 128 rows, 1024 cols → 32 blocks/row → 128*32*18 = 73728
    auto tensor = createQ4_0WithData({128, 1024});
    ASSERT_NE(tensor, nullptr);

    // Full tensor should report nonzero size_bytes
    const size_t full_bytes = tensor->size_bytes();
    EXPECT_GT(full_bytes, 0u);

    // Create a 2D view — views may report 0 size_bytes
    auto view = tensor->create_view({64, 1024}, 0);
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape().size(), 2u);
    EXPECT_EQ(view->shape()[0], 64u);
    EXPECT_EQ(view->shape()[1], 1024u);

    // quantizedViewRawBytes should compute the correct size
    const size_t view_bytes = quantizedViewRawBytes(*view);
    // 64 rows × (1024/32)=32 blocks/row × 18 bytes/block = 36864
    const size_t expected = 64 * 32 * sizeof(Q4_0Block);
    EXPECT_EQ(view_bytes, expected);
}

TEST(Test__UnifiedPipelineMoECollection, QuantizedViewRawBytes_IQ4_NL)
{
    // create_view() uses shared_from_this(), so tensor must be in a shared_ptr
    std::shared_ptr<IQ4_NLTensor> tensor = TestTensorFactory::createIQ4_NLRandom({256, 512});
    ASSERT_NE(tensor, nullptr);

    auto view = tensor->create_view({32, 512}, 0);
    ASSERT_NE(view, nullptr);

    const size_t view_bytes = quantizedViewRawBytes(*view);
    // 32 rows × (512/32)=16 blocks/row × sizeof(IQ4_NLBlock)
    const size_t expected = 32 * 16 * sizeof(IQ4_NLBlock);
    EXPECT_EQ(view_bytes, expected);
}

TEST(Test__UnifiedPipelineMoECollection, QuantizedViewRawBytes_FP32_ReturnsReported)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    const size_t bytes = quantizedViewRawBytes(*tensor);
    // FP32 reports nonzero size_bytes, so quantizedViewRawBytes returns it
    EXPECT_GT(bytes, 0u);
}

// =========================================================================
// Expert view creation from 3D tensors
// =========================================================================

TEST(Test__UnifiedPipelineMoECollection, ExpertViewCreation_Q4_0_3D)
{
    // Simulate a 3D expert tensor: [cols=128, rows_per_expert=64, num_experts=4]
    const size_t cols = 128;
    const size_t rows_per_expert = 64;
    const size_t num_experts = 4;
    auto tensor_3d = createQ4_0WithData({cols, rows_per_expert, num_experts});
    ASSERT_NE(tensor_3d, nullptr);

    const size_t elements_per_expert = rows_per_expert * cols;

    // Create views for each expert
    for (size_t e = 0; e < num_experts; ++e)
    {
        const size_t element_offset = e * elements_per_expert;
        auto view = tensor_3d->create_view({rows_per_expert, cols}, element_offset);
        ASSERT_NE(view, nullptr) << "Failed to create view for expert " << e;
        EXPECT_EQ(view->shape().size(), 2u);
        EXPECT_EQ(view->shape()[0], rows_per_expert);
        EXPECT_EQ(view->shape()[1], cols);

        // raw_data() should be non-null (pointing into parent's data)
        EXPECT_NE(view->raw_data(), nullptr);
    }
}

// =========================================================================
// MoE weight grouping logic
// =========================================================================

TEST(Test__UnifiedPipelineMoECollection, WeightGrouping_CompleteLayer)
{
    // Simulate cache entries
    std::unordered_map<std::string, std::shared_ptr<TensorBase>> cache;

    const size_t cols = 128, rows = 64, n_experts = 4;
    cache["blk.0.ffn_gate_exps.weight"] = createQ4_0WithData({cols, rows, n_experts}, 1);
    cache["blk.0.ffn_up_exps.weight"] = createQ4_0WithData({cols, rows, n_experts}, 2);
    cache["blk.0.ffn_down_exps.weight"] = createQ4_0WithData({cols, rows, n_experts}, 3);

    // Group by layer
    struct MoELayerTensors
    {
        std::shared_ptr<TensorBase> gate, up, down;
    };
    std::unordered_map<int, MoELayerTensors> moe_layers;

    for (const auto &[name, tensor] : cache)
    {
        if (name.find("_exps.weight") == std::string::npos)
            continue;
        int layer_idx = parseMoELayerIndex(name);
        if (layer_idx < 0)
            continue;

        if (name.find("ffn_gate_exps.weight") != std::string::npos)
            moe_layers[layer_idx].gate = tensor;
        else if (name.find("ffn_up_exps.weight") != std::string::npos)
            moe_layers[layer_idx].up = tensor;
        else if (name.find("ffn_down_exps.weight") != std::string::npos)
            moe_layers[layer_idx].down = tensor;
    }

    ASSERT_EQ(moe_layers.size(), 1u);
    ASSERT_TRUE(moe_layers.count(0));
    EXPECT_NE(moe_layers[0].gate, nullptr);
    EXPECT_NE(moe_layers[0].up, nullptr);
    EXPECT_NE(moe_layers[0].down, nullptr);
}

TEST(Test__UnifiedPipelineMoECollection, WeightGrouping_MultipleLayers)
{
    std::unordered_map<std::string, std::shared_ptr<TensorBase>> cache;

    const size_t cols = 128, rows = 64, n_experts = 8;
    for (int layer : {2, 5, 10})
    {
        std::string prefix = "blk." + std::to_string(layer) + ".";
        cache[prefix + "ffn_gate_exps.weight"] =
            createQ4_0WithData({cols, rows, n_experts}, static_cast<uint32_t>(layer * 3 + 1));
        cache[prefix + "ffn_up_exps.weight"] =
            createQ4_0WithData({cols, rows, n_experts}, static_cast<uint32_t>(layer * 3 + 2));
        cache[prefix + "ffn_down_exps.weight"] =
            createQ4_0WithData({cols, rows, n_experts}, static_cast<uint32_t>(layer * 3 + 3));
    }

    struct MoELayerTensors
    {
        std::shared_ptr<TensorBase> gate, up, down;
    };
    std::unordered_map<int, MoELayerTensors> moe_layers;

    for (const auto &[name, tensor] : cache)
    {
        if (name.find("_exps.weight") == std::string::npos)
            continue;
        int layer_idx = parseMoELayerIndex(name);
        if (layer_idx < 0)
            continue;

        if (name.find("ffn_gate_exps.weight") != std::string::npos)
            moe_layers[layer_idx].gate = tensor;
        else if (name.find("ffn_up_exps.weight") != std::string::npos)
            moe_layers[layer_idx].up = tensor;
        else if (name.find("ffn_down_exps.weight") != std::string::npos)
            moe_layers[layer_idx].down = tensor;
    }

    EXPECT_EQ(moe_layers.size(), 3u);
    EXPECT_TRUE(moe_layers.count(2));
    EXPECT_TRUE(moe_layers.count(5));
    EXPECT_TRUE(moe_layers.count(10));
}

TEST(Test__UnifiedPipelineMoECollection, WeightGrouping_IncompleteLayer_MissingDown)
{
    std::unordered_map<std::string, std::shared_ptr<TensorBase>> cache;

    const size_t cols = 128, rows = 64, n_experts = 4;
    cache["blk.3.ffn_gate_exps.weight"] = createQ4_0WithData({cols, rows, n_experts}, 10);
    cache["blk.3.ffn_up_exps.weight"] = createQ4_0WithData({cols, rows, n_experts}, 11);
    // Missing: blk.3.ffn_down_exps.weight

    struct MoELayerTensors
    {
        std::shared_ptr<TensorBase> gate, up, down;
    };
    std::unordered_map<int, MoELayerTensors> moe_layers;

    for (const auto &[name, tensor] : cache)
    {
        if (name.find("_exps.weight") == std::string::npos)
            continue;
        int layer_idx = parseMoELayerIndex(name);
        if (layer_idx < 0)
            continue;

        if (name.find("ffn_gate_exps.weight") != std::string::npos)
            moe_layers[layer_idx].gate = tensor;
        else if (name.find("ffn_up_exps.weight") != std::string::npos)
            moe_layers[layer_idx].up = tensor;
        else if (name.find("ffn_down_exps.weight") != std::string::npos)
            moe_layers[layer_idx].down = tensor;
    }

    ASSERT_EQ(moe_layers.size(), 1u);
    auto &layer = moe_layers[3];
    EXPECT_NE(layer.gate, nullptr);
    EXPECT_NE(layer.up, nullptr);
    EXPECT_EQ(layer.down, nullptr); // incomplete — should be skipped in pipeline
}

// =========================================================================
// Layer filter tests
// =========================================================================

TEST(Test__UnifiedPipelineMoECollection, LayerFilter_RespectsFilter)
{
    std::unordered_map<std::string, std::shared_ptr<TensorBase>> cache;

    const size_t cols = 128, rows = 64, n_experts = 4;
    for (int layer : {0, 1, 2})
    {
        std::string prefix = "blk." + std::to_string(layer) + ".";
        cache[prefix + "ffn_gate_exps.weight"] =
            createQ4_0WithData({cols, rows, n_experts}, static_cast<uint32_t>(layer * 3 + 1));
        cache[prefix + "ffn_up_exps.weight"] =
            createQ4_0WithData({cols, rows, n_experts}, static_cast<uint32_t>(layer * 3 + 2));
        cache[prefix + "ffn_down_exps.weight"] =
            createQ4_0WithData({cols, rows, n_experts}, static_cast<uint32_t>(layer * 3 + 3));
    }

    // Filter: only accept layer 1
    auto layer_filter = [](const std::string &name) -> bool
    {
        return name.find("blk.1.") != std::string::npos;
    };

    struct MoELayerTensors
    {
        std::shared_ptr<TensorBase> gate, up, down;
    };
    std::unordered_map<int, MoELayerTensors> moe_layers;

    for (const auto &[name, tensor] : cache)
    {
        if (name.find("_exps.weight") == std::string::npos)
            continue;
        if (!layer_filter(name))
            continue;
        int layer_idx = parseMoELayerIndex(name);
        if (layer_idx < 0)
            continue;

        if (name.find("ffn_gate_exps.weight") != std::string::npos)
            moe_layers[layer_idx].gate = tensor;
        else if (name.find("ffn_up_exps.weight") != std::string::npos)
            moe_layers[layer_idx].up = tensor;
        else if (name.find("ffn_down_exps.weight") != std::string::npos)
            moe_layers[layer_idx].down = tensor;
    }

    // Only layer 1 should be collected
    EXPECT_EQ(moe_layers.size(), 1u);
    EXPECT_TRUE(moe_layers.count(1));
    EXPECT_FALSE(moe_layers.count(0));
    EXPECT_FALSE(moe_layers.count(2));
}

// =========================================================================
// Slot name generation
// =========================================================================

TEST(Test__UnifiedPipelineMoECollection, SlotNameGeneration)
{
    // Verify the naming convention for MoE expert slots
    using Role = ExpertGemmRegistry::WeightRole;

    struct MoEExpertJob
    {
        int layer_idx;
        int expert_idx;
        Role role;
        std::string slot_name;
    };

    auto make_slot = [](int layer, const char *tag, int expert) -> std::string
    {
        return "moe_L" + std::to_string(layer) + "_" + tag + "_e" + std::to_string(expert);
    };

    EXPECT_EQ(make_slot(0, "gate", 5), "moe_L0_gate_e5");
    EXPECT_EQ(make_slot(12, "up", 0), "moe_L12_up_e0");
    EXPECT_EQ(make_slot(3, "down", 63), "moe_L3_down_e63");
}

// =========================================================================
// Full expert job generation (end-to-end collection + view creation)
// =========================================================================

TEST(Test__UnifiedPipelineMoECollection, FullJobGeneration_4Experts)
{
    using Role = ExpertGemmRegistry::WeightRole;

    struct MoEExpertJob
    {
        int layer_idx;
        int expert_idx;
        Role role;
        std::string slot_name;
        std::shared_ptr<TensorBase> view;
    };

    std::unordered_map<std::string, std::shared_ptr<TensorBase>> cache;
    const size_t cols = 128, rows = 64, n_experts = 4;

    cache["blk.0.ffn_gate_exps.weight"] = createQ4_0WithData({cols, rows, n_experts}, 20);
    cache["blk.0.ffn_up_exps.weight"] = createQ4_0WithData({cols, rows, n_experts}, 21);
    cache["blk.0.ffn_down_exps.weight"] = createQ4_0WithData({cols, rows, n_experts}, 22);

    std::vector<MoEExpertJob> moe_jobs;

    // Replicate collection logic
    struct MoELayerTensors
    {
        std::shared_ptr<TensorBase> gate, up, down;
    };
    std::unordered_map<int, MoELayerTensors> moe_layers;

    for (const auto &[name, tensor] : cache)
    {
        if (name.find("_exps.weight") == std::string::npos)
            continue;
        int layer_idx = parseMoELayerIndex(name);
        if (layer_idx < 0)
            continue;
        if (name.find("ffn_gate_exps.weight") != std::string::npos)
            moe_layers[layer_idx].gate = tensor;
        else if (name.find("ffn_up_exps.weight") != std::string::npos)
            moe_layers[layer_idx].up = tensor;
        else if (name.find("ffn_down_exps.weight") != std::string::npos)
            moe_layers[layer_idx].down = tensor;
    }

    for (auto &[layer_idx, tensors] : moe_layers)
    {
        ASSERT_TRUE(tensors.gate && tensors.up && tensors.down);
        const auto &gate_shape = tensors.gate->shape();
        ASSERT_EQ(gate_shape.size(), 3u);

        const int ne = static_cast<int>(gate_shape[2]);

        struct RoleTensor
        {
            Role role;
            const char *tag;
            TensorBase *t;
        };
        RoleTensor roles[] = {
            {Role::GATE, "gate", tensors.gate.get()},
            {Role::UP, "up", tensors.up.get()},
            {Role::DOWN, "down", tensors.down.get()},
        };

        for (const auto &rt : roles)
        {
            // Each role uses its OWN shape (down may differ from gate/up)
            const auto &role_shape = rt.t->shape();
            const size_t role_cols = role_shape[0];
            const size_t role_rows = role_shape[1];
            const size_t role_elements_per_expert = role_rows * role_cols;

            for (int e = 0; e < ne; ++e)
            {
                auto view = rt.t->create_view({role_rows, role_cols}, e * role_elements_per_expert);
                ASSERT_NE(view, nullptr);
                std::string slot = "moe_L" + std::to_string(layer_idx) + "_"
                                   + rt.tag + "_e" + std::to_string(e);
                moe_jobs.push_back({layer_idx, e, rt.role, std::move(slot), std::move(view)});
            }
        }
    }

    // 4 experts × 3 roles = 12 jobs
    EXPECT_EQ(moe_jobs.size(), 12u);

    // Verify all views have correct shape
    for (const auto &job : moe_jobs)
    {
        EXPECT_EQ(job.view->shape().size(), 2u);
        EXPECT_EQ(job.view->shape()[0], rows);
        EXPECT_EQ(job.view->shape()[1], cols);
        EXPECT_EQ(job.layer_idx, 0);
        EXPECT_GE(job.expert_idx, 0);
        EXPECT_LT(job.expert_idx, static_cast<int>(n_experts));
    }
}

// =========================================================================
// Asymmetric dimension test: down tensor has swapped shape vs gate/up
// This is the real MoE layout (Qwen3.5-35B-A3B):
//   gate/up: [d_model, intermediate, num_experts]
//   down:    [intermediate, d_model, num_experts]
// The pipeline must use per-role dimensions, NOT assume gate's shape for all.
// =========================================================================

TEST(Test__UnifiedPipelineMoECollection, PerRoleDimensions_AsymmetricDownShape)
{
    using Role = ExpertGemmRegistry::WeightRole;

    // Real MoE dimensions (scaled down for unit test):
    // gate/up: shape[0]=d_model(128), shape[1]=intermediate(64), shape[2]=4 experts
    // down:    shape[0]=intermediate(64), shape[1]=d_model(128), shape[2]=4 experts
    const size_t d_model = 128;
    const size_t intermediate = 64;
    const size_t n_experts = 4;

    auto gate_tensor = createQ4_0WithData({d_model, intermediate, n_experts}, 30);
    auto up_tensor = createQ4_0WithData({d_model, intermediate, n_experts}, 31);
    auto down_tensor = createQ4_0WithData({intermediate, d_model, n_experts}, 32); // SWAPPED

    ASSERT_NE(gate_tensor, nullptr);
    ASSERT_NE(up_tensor, nullptr);
    ASSERT_NE(down_tensor, nullptr);

    // Verify shapes are indeed different
    ASSERT_EQ(gate_tensor->shape()[0], d_model);
    ASSERT_EQ(gate_tensor->shape()[1], intermediate);
    ASSERT_EQ(down_tensor->shape()[0], intermediate);
    ASSERT_EQ(down_tensor->shape()[1], d_model);

    struct RoleTensor
    {
        Role role;
        const char *tag;
        TensorBase *tensor_3d;
    };
    RoleTensor roles[] = {
        {Role::GATE, "gate", gate_tensor.get()},
        {Role::UP, "up", up_tensor.get()},
        {Role::DOWN, "down", down_tensor.get()},
    };

    struct MoEExpertJob
    {
        int expert_idx;
        Role role;
        std::shared_ptr<TensorBase> view;
    };
    std::vector<MoEExpertJob> jobs;

    // Use per-role dimensions (the FIXED code path)
    for (const auto &rt : roles)
    {
        const auto &role_shape = rt.tensor_3d->shape();
        const size_t role_cols = role_shape[0];
        const size_t role_rows_per_expert = role_shape[1];
        const size_t role_elements_per_expert = role_rows_per_expert * role_cols;

        for (size_t e = 0; e < n_experts; ++e)
        {
            const size_t element_offset = e * role_elements_per_expert;
            std::vector<size_t> view_shape = {role_rows_per_expert, role_cols};
            auto view = rt.tensor_3d->create_view(view_shape, element_offset);
            ASSERT_NE(view, nullptr)
                << "Failed to create view for " << rt.tag << " expert " << e;

            jobs.push_back({static_cast<int>(e), rt.role, std::move(view)});
        }
    }

    // 4 experts × 3 roles = 12 jobs
    ASSERT_EQ(jobs.size(), 12u);

    // Verify GATE views: shape should be [intermediate(64), d_model(128)]
    for (const auto &job : jobs)
    {
        if (job.role == Role::GATE || job.role == Role::UP)
        {
            EXPECT_EQ(job.view->shape()[0], intermediate)
                << "Gate/Up view rows should be intermediate=" << intermediate;
            EXPECT_EQ(job.view->shape()[1], d_model)
                << "Gate/Up view cols should be d_model=" << d_model;
        }
        else if (job.role == Role::DOWN)
        {
            EXPECT_EQ(job.view->shape()[0], d_model)
                << "Down view rows should be d_model=" << d_model;
            EXPECT_EQ(job.view->shape()[1], intermediate)
                << "Down view cols should be intermediate=" << intermediate;
        }
    }

    // Verify raw_data() is non-null for all views
    for (const auto &job : jobs)
    {
        EXPECT_NE(job.view->raw_data(), nullptr)
            << "View raw_data should be non-null (aliasing parent)";
    }

    // Verify different experts point to different memory offsets
    // (expert 0 and expert 1 of the same role should have different data pointers)
    for (const auto &rt : roles)
    {
        std::vector<const void *> ptrs;
        for (const auto &job : jobs)
        {
            if (job.role == rt.role)
                ptrs.push_back(job.view->raw_data());
        }
        ASSERT_EQ(ptrs.size(), n_experts);
        for (size_t i = 1; i < ptrs.size(); ++i)
        {
            EXPECT_NE(ptrs[i], ptrs[i - 1])
                << rt.tag << " expert " << i << " should have different data ptr than expert " << (i - 1);
        }
    }
}

// =========================================================================
// Regression: using gate shape for ALL roles would fail for asymmetric down
// This test verifies that create_view with wrong dimensions produces either
// a null view or a view with wrong shape, catching the original bug.
// =========================================================================

TEST(Test__UnifiedPipelineMoECollection, Regression_GateShapeForDownFails)
{
    // down: [intermediate=64, d_model=128, 4 experts]
    const size_t intermediate = 64;
    const size_t d_model = 128;
    const size_t n_experts = 4;

    auto down_tensor = createQ4_0WithData({intermediate, d_model, n_experts}, 40);
    ASSERT_NE(down_tensor, nullptr);

    // BUG scenario: using gate's dimensions (d_model=128, intermediate=64) on down tensor
    // Gate shape would be [d_model=128, intermediate=64, 4]
    // Trying to create view with gate's shape[1]=64 rows, shape[0]=128 cols
    // But down's parent has cols=64 (shape[0]=intermediate).
    // create_view enforces K (cols) must match parent → throws!
    const size_t gate_cols = d_model;              // 128 (gate's shape[0])
    const size_t gate_rows_per_expert = intermediate; // 64 (gate's shape[1])

    // Using gate dims on a down tensor: view K=128 doesn't match parent K=64 → exception
    EXPECT_THROW(
        down_tensor->create_view({gate_rows_per_expert, gate_cols}, 0),
        std::invalid_argument)
        << "create_view should reject mismatched K dimension (the original bug)";

    // Correct view using down's own dimensions works fine
    const size_t down_cols = intermediate;  // 64 (down's shape[0])
    const size_t down_rows = d_model;       // 128 (down's shape[1])
    auto correct_view = down_tensor->create_view({down_rows, down_cols}, 0);
    ASSERT_NE(correct_view, nullptr);
    EXPECT_EQ(correct_view->shape()[0], d_model);       // 128 ← correct
    EXPECT_EQ(correct_view->shape()[1], intermediate);  // 64 ← correct
}
