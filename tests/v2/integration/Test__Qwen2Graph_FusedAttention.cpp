#include <gtest/gtest.h>
#include "../../../src/v2/models/qwen/Qwen2Graph.h"
#include "../../../src/v2/execution/local_execution/graph/GraphExecutor.h"
#include "../../../src/v2/backends/ComputeBackend.h"
#include "../../../src/v2/utils/DebugEnv.h"
#include "../../../src/v2/utils/MPIContext.h"
#include "../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

class Test__Qwen2Graph_FusedAttention : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Force trace logging
        Logger::getInstance().setLogLevel(LogLevel::TRACE);

        // Create dummy contexts
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1);

        // Configure graph
        config_.n_layers = 1;
        config_.d_model = 128;
        config_.n_heads = 4;
        config_.n_kv_heads = 4;
        config_.head_dim = 32;
        config_.d_ff = 256;
        config_.vocab_size = 1000;
        config_.max_seq_len = 128;
        config_.activation_precision = ActivationPrecision::Q8_1;

        // Create dummy weights
        createDummyWeights();
        createDummyBuffers();

        // Initialize graph builder
        graph_builder_ = std::make_unique<Qwen2Graph>(config_, mpi_ctx_);
    }

    void createDummyWeights()
    {
        // Create dummy tensors for one layer
        auto wq = TestTensorFactory::createFP32({128, 128});
        auto wk = TestTensorFactory::createFP32({128, 128});
        auto wv = TestTensorFactory::createFP32({128, 128});
        auto wo = TestTensorFactory::createFP32({128, 128});
        auto attn_norm = TestTensorFactory::createFP32({128});

        auto gate_proj = TestTensorFactory::createFP32({256, 128});
        auto up_proj = TestTensorFactory::createFP32({256, 128});
        auto down_proj = TestTensorFactory::createFP32({128, 256});
        auto ffn_norm = TestTensorFactory::createFP32({128});

        // Store them in a way that they persist
        tensor_storage_.push_back(std::move(wq));
        tensor_storage_.push_back(std::move(wk));
        tensor_storage_.push_back(std::move(wv));
        tensor_storage_.push_back(std::move(wo));
        tensor_storage_.push_back(std::move(attn_norm));
        tensor_storage_.push_back(std::move(gate_proj));
        tensor_storage_.push_back(std::move(up_proj));
        tensor_storage_.push_back(std::move(down_proj));
        tensor_storage_.push_back(std::move(ffn_norm));

        // Populate Qwen2LayerWeights
        Qwen2LayerWeights layer_weights;
        layer_weights.wq = tensor_storage_[0].get();
        layer_weights.wk = tensor_storage_[1].get();
        layer_weights.wv = tensor_storage_[2].get();
        layer_weights.wo = tensor_storage_[3].get();
        layer_weights.attn_norm = tensor_storage_[4].get();
        layer_weights.gate_proj = tensor_storage_[5].get();
        layer_weights.up_proj = tensor_storage_[6].get();
        layer_weights.down_proj = tensor_storage_[7].get();
        layer_weights.ffn_norm = tensor_storage_[8].get();

        layer_weights_storage_.push_back(layer_weights);

        // Model weights
        auto lm_head = TestTensorFactory::createFP32({1000, 128});
        auto final_norm = TestTensorFactory::createFP32({128});
        auto embedding = TestTensorFactory::createFP32({1000, 128});

        tensor_storage_.push_back(std::move(lm_head));
        tensor_storage_.push_back(std::move(final_norm));
        tensor_storage_.push_back(std::move(embedding));

        weights_.lm_head = tensor_storage_[9].get();
        weights_.final_norm = tensor_storage_[10].get();
        weights_.embedding_table = tensor_storage_[11].get();

        // Set up the accessor
        weights_.get_layer_weights = [this](int i)
        {
            if (i >= 0 && i < layer_weights_storage_.size())
            {
                return layer_weights_storage_[i];
            }
            throw std::runtime_error("Layer index out of bounds in test");
        };
    }

    void createDummyBuffers()
    {
        // Create dummy buffers
        auto hidden = TestTensorFactory::createFP32({10, 128}); // batch=1, seq=10
        auto logits = TestTensorFactory::createFP32({10, 1000});

        tensor_storage_.push_back(std::move(hidden));
        tensor_storage_.push_back(std::move(logits));

        buffers_.current_hidden = tensor_storage_.back().get(); // logits is last? No, hidden then logits
        buffers_.logits = tensor_storage_.back().get();
        buffers_.current_hidden = tensor_storage_[tensor_storage_.size() - 2].get();

        // Layer buffers
        auto residual = TestTensorFactory::createFP32({10, 128});
        auto normalized = TestTensorFactory::createFP32({10, 128});
        auto Q = TestTensorFactory::createFP32({10, 128});
        auto K = TestTensorFactory::createFP32({10, 128});
        auto V = TestTensorFactory::createFP32({10, 128});
        auto attn_output = TestTensorFactory::createFP32({10, 128});
        auto gate = TestTensorFactory::createFP32({10, 256});
        auto up = TestTensorFactory::createFP32({10, 256});
        auto ffn_output = TestTensorFactory::createFP32({10, 256});
        auto attn_proj = TestTensorFactory::createFP32({10, 128});

        tensor_storage_.push_back(std::move(residual));
        tensor_storage_.push_back(std::move(normalized));
        tensor_storage_.push_back(std::move(Q));
        tensor_storage_.push_back(std::move(K));
        tensor_storage_.push_back(std::move(V));
        tensor_storage_.push_back(std::move(attn_output));
        tensor_storage_.push_back(std::move(gate));
        tensor_storage_.push_back(std::move(up));
        tensor_storage_.push_back(std::move(ffn_output));
        tensor_storage_.push_back(std::move(attn_proj));

        // Assign pointers (using back-offsets is risky if I change order, but okay for now)
        size_t base = tensor_storage_.size() - 10;
        buffers_.layer_buffers.residual = tensor_storage_[base].get();
        buffers_.layer_buffers.normalized = tensor_storage_[base + 1].get();
        buffers_.layer_buffers.Q = tensor_storage_[base + 2].get();
        buffers_.layer_buffers.K = tensor_storage_[base + 3].get();
        buffers_.layer_buffers.V = tensor_storage_[base + 4].get();
        buffers_.layer_buffers.attn_output = tensor_storage_[base + 5].get();
        buffers_.layer_buffers.gate = tensor_storage_[base + 6].get();
        buffers_.layer_buffers.up = tensor_storage_[base + 7].get();
        buffers_.layer_buffers.ffn_output = tensor_storage_[base + 8].get();
        buffers_.layer_buffers.attn_proj = tensor_storage_[base + 9].get();
        buffers_.layer_buffers.current_hidden = buffers_.current_hidden;
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
    Qwen2GraphConfig config_;
    Qwen2ModelWeights weights_;
    Qwen2ModelBuffers buffers_;
    std::unique_ptr<Qwen2Graph> graph_builder_;

    // Storage to keep tensors alive
    std::vector<std::unique_ptr<TensorBase>> tensor_storage_;
    std::vector<Qwen2LayerWeights> layer_weights_storage_;
};

TEST_F(Test__Qwen2Graph_FusedAttention, FusedAttentionEnabled)
{
    // Enable fused attention
    mutableDebugEnv().attention.fused_wo = true;
    // config_.activation_precision = ActivationPrecision::Q8_1; // Already set in SetUp

    graph_builder_->setWeights(weights_);
    graph_builder_->setBuffers(buffers_); // Set buffers

    // Build graph
    Qwen2ForwardInput input;
    input.batch_size = 1;
    input.seq_len = 10;

    Qwen2ForwardOutput output;
    auto graph = graph_builder_->buildFullForwardGraph(input, output);

    // Verify topology
    bool found_fused_stage = false;
    bool found_decomposed_stages = false;

    auto order = graph.getExecutionOrder();
    for (const auto &node_name : order)
    {
        const auto *node = graph.getNode(node_name);
        if (node && node->stage)
        {
            if (node->stage->type() == ComputeStageType::FUSED_ATTENTION_WO)
            {
                found_fused_stage = true;
            }
            if (node->stage->type() == ComputeStageType::ATTENTION)
            {
                found_decomposed_stages = true;
            }
        }
    }

    EXPECT_TRUE(found_fused_stage) << "Should find FusedAttentionWo stage when enabled";
    EXPECT_FALSE(found_decomposed_stages) << "Should NOT find AttentionCompute stage when fused is enabled";
}

TEST_F(Test__Qwen2Graph_FusedAttention, FusedAttentionDisabled)
{
    // Disable fused attention
    mutableDebugEnv().attention.fused_wo = false;

    createDummyBuffers(); // Create buffers

    auto graph_builder = std::make_unique<Qwen2Graph>(config_, mpi_ctx_);
    graph_builder->setWeights(weights_);
    graph_builder->setBuffers(buffers_); // Set buffers

    // Build graph
    Qwen2ForwardInput input;
    input.batch_size = 1;
    input.seq_len = 10;

    Qwen2ForwardOutput output;
    auto graph = graph_builder->buildFullForwardGraph(input, output);

    // Verify topology
    bool found_fused_stage = false;
    bool found_decomposed_stages = false;

    auto order = graph.getExecutionOrder();
    for (const auto &node_name : order)
    {
        const auto *node = graph.getNode(node_name);
        if (node && node->stage)
        {
            if (node->stage->type() == ComputeStageType::FUSED_ATTENTION_WO)
            {
                found_fused_stage = true;
            }
            if (node->stage->type() == ComputeStageType::ATTENTION)
            {
                found_decomposed_stages = true;
            }
        }
    }

    EXPECT_FALSE(found_fused_stage) << "Should NOT find FusedAttentionWo stage when disabled";
    EXPECT_TRUE(found_decomposed_stages) << "Should find AttentionCompute stage when fused is disabled";
}
