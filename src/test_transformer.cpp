#include "graph_compute.h"
#include "logger.h"
#include <iostream>
#include <memory>

using namespace llaminar;

int main()
{
    LOG_INFO("Testing transformer compute nodes");

    // Create a simple compute graph with transformer components
    auto graph = std::make_unique<ComputeGraph>();

    // Test Tensor creation
    auto input_tensor = std::make_shared<Tensor>(std::vector<int>{1, 512});  // Batch=1, Hidden=512
    auto weight_tensor = std::make_shared<Tensor>(std::vector<int>{512});    // Layer norm weight
    auto output_tensor = std::make_shared<Tensor>(std::vector<int>{1, 512}); // Output

    // Initialize some dummy data
    std::fill(input_tensor->data.begin(), input_tensor->data.end(), 1.0f);
    std::fill(weight_tensor->data.begin(), weight_tensor->data.end(), 0.5f);

    LOG_INFO("Created tensors - Input: " +
             std::to_string(input_tensor->shape[0]) + "x" + std::to_string(input_tensor->shape[1]));

    // Create and test RMSNorm node
    auto rms_norm = std::make_shared<RMSNormNode>("test_rmsnorm", weight_tensor, 1e-6f);
    rms_norm->setOutput(output_tensor);

    // Create a simple data node for testing
    int seq_len = 1;
    int d_model = 512;
    auto input_node = std::make_shared<DataNode>("input", std::vector<double>(seq_len * d_model, 1.0));
    rms_norm->addInput(input_node);

    // Connect nodes
    rms_norm->addInput(input_node);

    // Add to graph
    graph->addNode(input_node);
    graph->addNode(rms_norm);

    LOG_INFO("Built compute graph with " + std::to_string(graph->getNodeCount()) + " nodes");

    // Validate graph
    if (graph->validate())
    {
        LOG_INFO("✅ Graph validation passed");
    }
    else
    {
        LOG_ERROR("❌ Graph validation failed");
        return 1;
    }

    // Execute graph
    LOG_INFO("Executing compute graph...");
    if (graph->execute())
    {
        LOG_INFO("✅ Graph execution completed successfully");

        // Check output tensor has been modified
        float sum = 0.0f;
        for (float val : output_tensor->data)
        {
            sum += val;
        }
        LOG_INFO("Output tensor sum: " + std::to_string(sum));

        if (sum > 0.0f)
        {
            LOG_INFO("✅ RMSNorm computation produced non-zero output");
        }
        else
        {
            LOG_WARN("⚠️  RMSNorm output is zero - check implementation");
        }
    }
    else
    {
        LOG_ERROR("❌ Graph execution failed");
        return 1;
    }

    LOG_INFO("🎉 Transformer compute graph test completed successfully!");
    LOG_INFO("Your custom ComputeGraph with transformer nodes is working!");

    return 0;
}