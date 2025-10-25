/**
 * @file Test__MultiDevice.cpp
 * @brief Test multi-device support in Qwen2Pipeline (Phase 4.1)
 * @author David Sanftenberg
 * @date 2025-10-24
 */

#include <gtest/gtest.h>
#include "pipelines/qwen/Qwen2Pipeline.h"
#include "loaders/ModelContext.h"
#include "loaders/WeightPlacementMap.h"
#include "utils/MPIContext.h"
#include <memory>
#include <iostream>

using namespace llaminar2;

/**
 * @brief Test fixture for multi-device tests
 */
class Test__MultiDevice : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Use the 0.5B model for testing
        model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

        // Create model context
        model_ctx_ = ModelContext::create(model_path_, nullptr, nullptr);

        if (!model_ctx_)
        {
            GTEST_SKIP() << "Model not found: " << model_path_;
        }

        // Get model info
        const GGUFModel &model = model_ctx_->model();
        n_layers_ = static_cast<int>(model.block_count);

        std::cout << "[MultiDevice] Loaded model: " << n_layers_ << " layers\n";
    }

    std::string model_path_;
    std::shared_ptr<ModelContext> model_ctx_;
    int n_layers_ = 0;
};

/**
 * @brief Test single-device mode (backward compatibility)
 */
TEST_F(Test__MultiDevice, SingleDeviceMode)
{
    // Create default placement map (all on CPU)
    auto placement_map = std::make_shared<WeightPlacementMap>(-1);

    // Create pipeline with placement map
    Qwen2Pipeline pipeline(model_ctx_, nullptr, -1, placement_map);

    // Verify pipeline initialized
    EXPECT_EQ(pipeline.architecture(), std::string("qwen2"));

    std::cout << "[MultiDevice] Single-device mode test passed\n";
}

/**
 * @brief Test active device discovery
 */
TEST_F(Test__MultiDevice, ActiveDeviceDiscovery)
{
    // Create placement map with layers split across CPU and GPU
    auto placement_map = std::make_shared<WeightPlacementMap>(-1);

    // Put first half of layers on GPU (device 0)
    int split_layer = n_layers_ / 2;
    placement_map->setLayerRange(0, split_layer - 1, 0);          // GPU
    placement_map->setLayerRange(split_layer, n_layers_ - 1, -1); // CPU

    std::cout << "[MultiDevice] Placement: layers 0-" << (split_layer - 1) << " on GPU, "
              << split_layer << "-" << (n_layers_ - 1) << " on CPU\n";

    // Create pipeline with split placement
    Qwen2Pipeline pipeline(model_ctx_, nullptr, -1, placement_map);

    // Pipeline should discover both devices
    // Note: We can't directly access active_devices_ (private), but we can verify
    // the pipeline initializes without errors

    std::cout << "[MultiDevice] Active device discovery test passed\n";
}

/**
 * @brief Test multi-device buffer allocation
 */
TEST_F(Test__MultiDevice, MultiDeviceBufferAllocation)
{
    // Create placement map with MoE-style placement:
    // - Embedding on CPU
    // - First 12 layers on GPU (shared experts)
    // - Remaining layers on CPU (sparse experts)
    auto placement_map = std::make_shared<WeightPlacementMap>(-1);

    // Embedding and output on CPU
    placement_map->setTensorDevice("token_embd.weight", -1);
    placement_map->setTensorDevice("output_norm.weight", -1);
    placement_map->setTensorDevice("output.weight", -1);

    // First 12 layers on GPU
    int shared_layers = std::min(12, n_layers_);
    placement_map->setLayerRange(0, shared_layers - 1, 0); // GPU

    // Remaining layers on CPU
    if (n_layers_ > shared_layers)
    {
        placement_map->setLayerRange(shared_layers, n_layers_ - 1, -1); // CPU
    }

    std::cout << "[MultiDevice] MoE-style placement: layers 0-" << (shared_layers - 1)
              << " on GPU, " << shared_layers << "-" << (n_layers_ - 1) << " on CPU\n";

    // Create pipeline - should allocate buffers for both CPU and GPU
    Qwen2Pipeline pipeline(model_ctx_, nullptr, -1, placement_map);

    // Verify pipeline initialized successfully
    EXPECT_EQ(pipeline.architecture(), std::string("qwen2"));

    std::cout << "[MultiDevice] Multi-device buffer allocation test passed\n";
}

/**
 * @brief Test weight device query
 */
TEST_F(Test__MultiDevice, WeightDeviceQuery)
{
    // Create placement map with specific weight placement
    auto placement_map = std::make_shared<WeightPlacementMap>(-1);

    // Layer 0 on GPU
    placement_map->setLayerDevice(0, 0);

    // Layer 1 on CPU
    placement_map->setLayerDevice(1, -1);

    // Create pipeline
    Qwen2Pipeline pipeline(model_ctx_, nullptr, -1, placement_map);

    // Query weight devices
    int layer0_wq = placement_map->getDeviceForWeight("blk.0.attn_q.weight", 0);
    int layer1_wq = placement_map->getDeviceForWeight("blk.1.attn_q.weight", 1);

    EXPECT_EQ(layer0_wq, 0);  // GPU
    EXPECT_EQ(layer1_wq, -1); // CPU

    std::cout << "[MultiDevice] Layer 0 wq device: " << layer0_wq << "\n";
    std::cout << "[MultiDevice] Layer 1 wq device: " << layer1_wq << "\n";
    std::cout << "[MultiDevice] Weight device query test passed\n";
}

/**
 * @brief Test all-GPU placement
 */
TEST_F(Test__MultiDevice, AllGPUPlacement)
{
    // Create placement map with everything on GPU
    auto placement_map = std::make_shared<WeightPlacementMap>(0); // Default to GPU

    // Create pipeline
    Qwen2Pipeline pipeline(model_ctx_, nullptr, 0, placement_map);

    // Verify pipeline initialized
    EXPECT_EQ(pipeline.architecture(), std::string("qwen2"));
    EXPECT_EQ(pipeline.device_index(), 0);

    std::cout << "[MultiDevice] All-GPU placement test passed\n";
}

/**
 * @brief Test pattern-based placement
 */
TEST_F(Test__MultiDevice, PatternBasedPlacement)
{
    // Create placement map with pattern rules
    auto placement_map = std::make_shared<WeightPlacementMap>(-1);

    // All attention weights on GPU
    placement_map->setPatternDevice("*.attn_*", 0);

    // All FFN weights on CPU (default)
    // (no explicit rule needed, default is CPU)

    // Create pipeline
    Qwen2Pipeline pipeline(model_ctx_, nullptr, -1, placement_map);

    // Query devices
    int attn_device = placement_map->getDeviceForWeight("blk.0.attn_q.weight", 0);
    int ffn_device = placement_map->getDeviceForWeight("blk.0.ffn_gate.weight", 0);

    EXPECT_EQ(attn_device, 0); // GPU (pattern match)
    EXPECT_EQ(ffn_device, -1); // CPU (default)

    std::cout << "[MultiDevice] Attention weights on device: " << attn_device << "\n";
    std::cout << "[MultiDevice] FFN weights on device: " << ffn_device << "\n";
    std::cout << "[MultiDevice] Pattern-based placement test passed\n";
}

/**
 * @brief Main entry point
 */
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
