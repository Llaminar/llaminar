/**
 * @file Test__UnifiedPP_Execution.cpp
 * @brief Integration tests for unified PP graph execution
 *
 * Tests the execution of unified PP graphs via DeviceGraphExecutor::executeMultiDevice()
 * with:
 * 1. Multiple device contexts (one per PP stage domain)
 * 2. LocalPPTransferStage nodes executing transfers between stages
 * 3. Correct data flow through the pipeline
 *
 * Uses MockLocalPPContext for functional testing without real multi-GPU hardware.
 * Tests with real model weights but minimal sequence length for efficiency.
 *
 * @see docs/v2/UNIFIED_PP_GRAPH_ARCHITECTURE_PLAN.md
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <filesystem>

#include "models/qwen/Qwen2Graph.h"
#include "config/PipelineConfig.h"
#include "config/TPDomainConfig.h"
#include "config/PPStageConfig.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/compute_stages/stages/LocalPPTransferStage.h"
#include "loaders/ModelContext.h"
#include "loaders/ModelContextConfig.h"
#include "kernels/cpu/CPUKVCache.h"
#include "tensors/TensorClasses.h"
#include "tensors/TensorFactory.h"
#include "mocks/MockLocalPPContext.h"

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{

    // =========================================================================
    // Test Fixture
    // =========================================================================

    /**
     * @brief Test fixture for unified PP execution integration tests
     *
     * Sets up:
     * - Real model loading (Qwen2.5-0.5B-Instruct)
     * - TensorFactory for buffer allocation
     * - Mock PP contexts for transfer testing
     * - CPU device contexts for execution
     */
    class Test__UnifiedPP_Execution : public ::testing::Test
    {
    protected:
        static constexpr const char *MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

        void SetUp() override
        {
            // Skip if model not available
            if (!std::filesystem::exists(MODEL_PATH))
            {
                GTEST_SKIP() << "Model not found: " << MODEL_PATH;
            }

            // Create MPI context (single-rank for integration tests)
            mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);

            // Create tensor factory
            tensor_factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);

            // Load model context (this loads the model and provides weight access)
            model_ctx_ = ModelContext::create(MODEL_PATH);
            if (!model_ctx_)
            {
                GTEST_SKIP() << "Failed to load model: " << MODEL_PATH;
            }

            // Extract model config from IModelContext interface
            n_layers_ = model_ctx_->blockCount();
            d_model_ = model_ctx_->embeddingLength();
            n_heads_ = model_ctx_->headCount();
            n_kv_heads_ = model_ctx_->headCountKV();
            head_dim_ = d_model_ / n_heads_;
            d_ff_ = model_ctx_->feedForwardLength();
            vocab_size_ = model_ctx_->vocabSize();

            // Create CPU device contexts
            cpu_device_ = DeviceId::cpu();
            cpu_context_ = std::make_unique<CPUDeviceContext>(cpu_device_, 4);
        }

        void TearDown() override
        {
            // Cleanup
            mock_pp_contexts_.clear();
        }

        // =========================================================================
        // Helper: Create 2-stage CPU PP config (layers split at midpoint)
        // =========================================================================
        std::shared_ptr<PipelineConfig> createTwoStageCPUConfig()
        {
            auto config = std::make_shared<PipelineConfig>();
            config->total_layers = n_layers_;

            int mid_layer = n_layers_ / 2;

            // Domain 0: CPU (stage 0)
            TPDomainConfig domain0;
            domain0.name = "stage0_domain";
            domain0.devices = {cpu_device_};
            domain0.tp_backend = CollectiveBackendType::HOST;
            config->tp_domains.push_back(domain0);

            // Domain 1: CPU (stage 1) - logically separate
            TPDomainConfig domain1;
            domain1.name = "stage1_domain";
            domain1.devices = {cpu_device_};
            domain1.tp_backend = CollectiveBackendType::HOST;
            config->tp_domains.push_back(domain1);

            // Stage 0: first half of layers, has embedding
            PPStageConfig stage0;
            stage0.stage_id = 0;
            stage0.domain_name = "stage0_domain";
            stage0.first_layer = 0;
            stage0.last_layer = mid_layer;
            stage0.has_embedding = true;
            stage0.has_lm_head = false;
            config->pp_stages.push_back(stage0);

            // Stage 1: second half of layers, has LM head
            PPStageConfig stage1;
            stage1.stage_id = 1;
            stage1.domain_name = "stage1_domain";
            stage1.first_layer = mid_layer;
            stage1.last_layer = n_layers_;
            stage1.has_embedding = false;
            stage1.has_lm_head = true;
            config->pp_stages.push_back(stage1);

            // Transfer backend
            config->pp_transfer_backends[{0, 1}] = CollectiveBackendType::HOST;

            return config;
        }

        // =========================================================================
        // Helper: Create mock PP context that does memcpy transfer
        // =========================================================================
        void createMockPPContexts(PipelineConfig *config)
        {
            int num_stages = config->numStages();
            for (int from = 0; from < num_stages - 1; ++from)
            {
                int to = from + 1;
                MockLocalPPContext::Config mock_config;
                // Use different mock device addresses to ensure transfer is NOT a no-op.
                // The LocalPPTransferStage checks if src_device == dst_device and skips
                // the transfer if they're the same. By using different fake CUDA devices,
                // we ensure the mock's transfer() method is actually called.
                // NOTE: The actual execution is CPU-based; these are just identity addresses.
                mock_config.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};

                // Calculate layer boundaries based on pipeline config
                int layers_per_stage = n_layers_ / num_stages;
                std::vector<int> boundaries;
                for (int s = 0; s <= num_stages; ++s)
                {
                    boundaries.push_back(s * layers_per_stage);
                }
                // Last boundary should be total layers
                boundaries.back() = n_layers_;
                mock_config.layer_boundaries = boundaries;

                mock_config.default_backend = CollectiveBackendType::HOST;
                auto pp_ctx = std::make_unique<MockLocalPPContext>(mock_config);
                graph_config_.pp_contexts[{from, to}] = pp_ctx.get();
                mock_pp_contexts_.push_back(std::move(pp_ctx));
            }
        }

        // =========================================================================
        // Helper: Create graph configuration
        // =========================================================================
        GraphConfig createGraphConfig()
        {
            GraphConfig config;

            config.n_layers = n_layers_;
            config.d_model = d_model_;
            config.n_heads = n_heads_;
            config.n_kv_heads = n_kv_heads_;
            config.head_dim = head_dim_;
            config.d_ff = d_ff_;
            config.vocab_size = vocab_size_;
            config.d_ff_local = d_ff_;  // No TP sharding
            config.vocab_local = vocab_size_;
            config.local_n_heads = n_heads_;
            config.local_n_kv_heads = n_kv_heads_;

            config.rms_norm_eps = 1e-6f;
            config.rope_theta = 10000.0f;
            config.default_device = cpu_device_;
            config.max_seq_len = 32;  // Small for testing
            config.activation_precision = ActivationPrecision::FP32;

            return config;
        }

        // =========================================================================
        // Helper: Create model weights accessor
        // =========================================================================
        ModelWeights createModelWeights()
        {
            ModelWeights weights;

            // Load global weights
            embedding_table_ = model_ctx_->getWeightForDevice("token_embd.weight", cpu_device_);
            final_norm_ = model_ctx_->getWeightForDevice("output_norm.weight", cpu_device_);
            lm_head_ = model_ctx_->getWeightForDevice("output.weight", cpu_device_);

            weights.embedding_table = embedding_table_.get();
            weights.final_norm = final_norm_.get();
            weights.lm_head = lm_head_.get();

            // Create layer weights accessor
            weights.get_layer_weights = [this](int layer_idx) -> LayerWeights
            {
                return loadLayerWeights(layer_idx);
            };

            return weights;
        }

        // =========================================================================
        // Helper: Load weights for a single layer
        // =========================================================================
        LayerWeights loadLayerWeights(int layer_idx)
        {
            // Check cache
            if (layer_weights_cache_.count(layer_idx))
            {
                return layer_weights_cache_[layer_idx].weights;
            }

            std::string prefix = "blk." + std::to_string(layer_idx) + ".";

            LayerWeightsStorage storage;
            storage.wq = model_ctx_->getWeightForDevice(prefix + "attn_q.weight", cpu_device_);
            storage.wk = model_ctx_->getWeightForDevice(prefix + "attn_k.weight", cpu_device_);
            storage.wv = model_ctx_->getWeightForDevice(prefix + "attn_v.weight", cpu_device_);
            storage.wo = model_ctx_->getWeightForDevice(prefix + "attn_output.weight", cpu_device_);
            storage.attn_norm = model_ctx_->getWeightForDevice(prefix + "attn_norm.weight", cpu_device_);
            storage.gate_proj = model_ctx_->getWeightForDevice(prefix + "ffn_gate.weight", cpu_device_);
            storage.up_proj = model_ctx_->getWeightForDevice(prefix + "ffn_up.weight", cpu_device_);
            storage.down_proj = model_ctx_->getWeightForDevice(prefix + "ffn_down.weight", cpu_device_);
            storage.ffn_norm = model_ctx_->getWeightForDevice(prefix + "ffn_norm.weight", cpu_device_);

            // Optional biases
            if (model_ctx_->hasTensor(prefix + "attn_q.bias"))
            {
                storage.q_bias = model_ctx_->getWeightForDevice(prefix + "attn_q.bias", cpu_device_);
            }
            if (model_ctx_->hasTensor(prefix + "attn_k.bias"))
            {
                storage.k_bias = model_ctx_->getWeightForDevice(prefix + "attn_k.bias", cpu_device_);
            }
            if (model_ctx_->hasTensor(prefix + "attn_v.bias"))
            {
                storage.v_bias = model_ctx_->getWeightForDevice(prefix + "attn_v.bias", cpu_device_);
            }

            // Build weights struct
            storage.weights.wq = storage.wq.get();
            storage.weights.wk = storage.wk.get();
            storage.weights.wv = storage.wv.get();
            storage.weights.wo = storage.wo.get();
            storage.weights.attn_norm = storage.attn_norm.get();
            storage.weights.gate_proj = storage.gate_proj.get();
            storage.weights.up_proj = storage.up_proj.get();
            storage.weights.down_proj = storage.down_proj.get();
            storage.weights.ffn_norm = storage.ffn_norm.get();
            storage.weights.q_bias = storage.q_bias ? storage.q_bias.get() : nullptr;
            storage.weights.k_bias = storage.k_bias ? storage.k_bias.get() : nullptr;
            storage.weights.v_bias = storage.v_bias ? storage.v_bias.get() : nullptr;

            layer_weights_cache_[layer_idx] = std::move(storage);
            return layer_weights_cache_[layer_idx].weights;
        }

        // =========================================================================
        // Helper: Create activation buffers
        // =========================================================================
        ModelBuffers createBuffers(int seq_len)
        {
            // Helper to avoid narrowing conversions (int -> size_t)
            auto sz = [](int x) { return static_cast<size_t>(x); };

            ModelBuffers buffers;

            // Current hidden [seq_len, d_model]
            // NOTE: Both buffers.current_hidden (model-level) and buffers.layer_buffers.current_hidden
            // (layer-level) must point to the same tensor. The embedding writes to the model-level
            // buffer, and RMSNorm reads from the layer-level buffer.
            current_hidden_ = tensor_factory_->createFP32({sz(seq_len), sz(d_model_)});
            buffers.current_hidden = current_hidden_.get();
            buffers.layer_buffers.current_hidden = current_hidden_.get();  // CRITICAL: layer-level alias

            // Logits [seq_len, vocab_size]
            logits_ = tensor_factory_->createFP32({sz(seq_len), sz(vocab_size_)});
            buffers.logits = logits_.get();

            // Layer buffers
            residual_ = tensor_factory_->createFP32({sz(seq_len), sz(d_model_)});
            buffers.layer_buffers.residual = residual_.get();

            normalized_ = tensor_factory_->createFP32({sz(seq_len), sz(d_model_)});
            buffers.layer_buffers.normalized = normalized_.get();

            Q_ = tensor_factory_->createFP32({sz(seq_len), sz(d_model_)});
            buffers.layer_buffers.Q = Q_.get();

            K_ = tensor_factory_->createFP32({sz(seq_len), sz(n_kv_heads_ * head_dim_)});
            buffers.layer_buffers.K = K_.get();

            V_ = tensor_factory_->createFP32({sz(seq_len), sz(n_kv_heads_ * head_dim_)});
            buffers.layer_buffers.V = V_.get();

            attn_output_ = tensor_factory_->createFP32({sz(seq_len), sz(d_model_)});
            buffers.layer_buffers.attn_output = attn_output_.get();

            attn_proj_ = tensor_factory_->createFP32({sz(seq_len), sz(d_model_)});
            buffers.layer_buffers.attn_proj = attn_proj_.get();

            gate_ = tensor_factory_->createFP32({sz(seq_len), sz(d_ff_)});
            buffers.layer_buffers.gate = gate_.get();

            up_ = tensor_factory_->createFP32({sz(seq_len), sz(d_ff_)});
            buffers.layer_buffers.up = up_.get();

            ffn_output_ = tensor_factory_->createFP32({sz(seq_len), sz(d_ff_)});
            buffers.layer_buffers.ffn_output = ffn_output_.get();

            return buffers;
        }

        // =========================================================================
        // Helper: Create forward input
        // =========================================================================
        Qwen2ForwardInput createForwardInput(int batch_size = 1, int seq_len = 4)
        {
            // Token IDs (valid tokens from model vocabulary)
            token_ids_.resize(batch_size * seq_len);
            for (int i = 0; i < batch_size * seq_len; ++i)
            {
                token_ids_[i] = i % 1000; // Use small token IDs within vocab range
            }

            // Position IDs
            position_ids_.resize(batch_size * seq_len);
            for (int b = 0; b < batch_size; ++b)
            {
                for (int p = 0; p < seq_len; ++p)
                {
                    position_ids_[b * seq_len + p] = p;
                }
            }

            Qwen2ForwardInput input;
            input.token_ids = token_ids_.data();
            input.position_ids = position_ids_.data();
            input.batch_size = batch_size;
            input.seq_len = seq_len;
            input.position_offset = 0;
            input.device = cpu_device_;
            input.kv_cache = nullptr;
            return input;
        }

        // =========================================================================
        // Helper: Count stages by type in graph
        // =========================================================================
        int countStagesByType(const ComputeGraph &graph, ComputeStageType type)
        {
            int count = 0;
            auto order = graph.getExecutionOrder();
            for (const auto &name : order)
            {
                const ComputeNode *node = graph.getNode(name);
                if (node && node->stage && node->stage->type() == type)
                {
                    ++count;
                }
            }
            return count;
        }

        // =========================================================================
        // Helper: Check if tensor has NaN or Inf values
        // =========================================================================
        bool hasNaNOrInf(const TensorBase *tensor)
        {
            if (!tensor)
                return true;

            const float *data = tensor->data();
            size_t numel = tensor->numel();
            for (size_t i = 0; i < numel; ++i)
            {
                if (std::isnan(data[i]) || std::isinf(data[i]))
                {
                    return true;
                }
            }
            return false;
        }

        // =========================================================================
        // Member Variables
        // =========================================================================
        std::shared_ptr<MPIContext> mpi_ctx_;
        std::unique_ptr<TensorFactory> tensor_factory_;
        std::shared_ptr<ModelContext> model_ctx_;

        // Model config (Qwen2.5-0.5B hardcoded)
        int n_layers_ = 24;
        int d_model_ = 896;
        int n_heads_ = 14;
        int n_kv_heads_ = 2;
        int head_dim_ = 64;
        int d_ff_ = 4864;
        int vocab_size_ = 151936;

        // Device contexts
        DeviceId cpu_device_;
        std::unique_ptr<CPUDeviceContext> cpu_context_;

        // Graph config
        GraphConfig graph_config_;

        // Mock PP contexts
        std::vector<std::unique_ptr<MockLocalPPContext>> mock_pp_contexts_;

        // Weights storage
        std::shared_ptr<TensorBase> embedding_table_;
        std::shared_ptr<TensorBase> final_norm_;
        std::shared_ptr<TensorBase> lm_head_;

        // Layer weights cache
        struct LayerWeightsStorage
        {
            std::shared_ptr<TensorBase> wq, wk, wv, wo, attn_norm;
            std::shared_ptr<TensorBase> q_bias, k_bias, v_bias;
            std::shared_ptr<TensorBase> gate_proj, up_proj, down_proj, ffn_norm;
            LayerWeights weights;
        };
        std::unordered_map<int, LayerWeightsStorage> layer_weights_cache_;

        // Buffer storage
        std::shared_ptr<TensorBase> current_hidden_;
        std::shared_ptr<TensorBase> logits_;
        std::shared_ptr<TensorBase> residual_;
        std::shared_ptr<TensorBase> normalized_;
        std::shared_ptr<TensorBase> Q_, K_, V_;
        std::shared_ptr<TensorBase> attn_output_, attn_proj_;
        std::shared_ptr<TensorBase> gate_, up_;
        std::shared_ptr<TensorBase> ffn_output_;

        // Forward input data
        std::vector<int> token_ids_;
        std::vector<int> position_ids_;
    };

    // =========================================================================
    // 1. Basic Execution Tests
    // =========================================================================

    /**
     * @test Execute 2-stage PP forward pass with mock transfers
     *
     * Verifies that a unified PP graph can be built and executed
     * via DeviceGraphExecutor::executeMultiDevice().
     */
    TEST_F(Test__UnifiedPP_Execution, ExecuteTwoStagePPForward)
    {
        // Create pipeline config
        auto pipeline_config = createTwoStageCPUConfig();

        // Create graph config
        graph_config_ = createGraphConfig();
        graph_config_.pipeline_config = pipeline_config;

        // Create mock PP contexts
        createMockPPContexts(pipeline_config.get());

        // Create graph builder
        Qwen2Graph graph_builder(graph_config_, mpi_ctx_);

        // Set weights
        auto weights = createModelWeights();
        graph_builder.setWeights(weights);

        // Create buffers
        int seq_len = 2;
        auto buffers = createBuffers(seq_len);
        graph_builder.setBuffers(buffers);

        // Create forward input
        auto input = createForwardInput(1, seq_len);

        // Build unified PP graph
        Qwen2ForwardOutput output;
        ComputeGraph compute_graph = graph_builder.buildUnifiedPipelineGraph(input, output);

        // Verify graph was built
        EXPECT_GT(compute_graph.size(), 0) << "Graph should have nodes";

        // Create device contexts map for multi-device execution
        std::unordered_map<DeviceId, IDeviceContext *> contexts;
        contexts[cpu_device_] = cpu_context_.get();

        // Create executor
        GraphExecutorConfig exec_config;
        exec_config.enable_validation = false;  // Disable validation for performance
        exec_config.enable_profiling = false;
        DeviceGraphExecutor executor(exec_config);

        // Execute graph
        bool success = executor.executeMultiDevice(compute_graph, contexts);
        EXPECT_TRUE(success) << "Graph execution should succeed";
    }

    /**
     * @test Output logits should not contain NaN or Inf values
     */
    TEST_F(Test__UnifiedPP_Execution, ProducesValidLogits)
    {
        // Create pipeline config
        auto pipeline_config = createTwoStageCPUConfig();

        // Create graph config
        graph_config_ = createGraphConfig();
        graph_config_.pipeline_config = pipeline_config;

        // Create mock PP contexts
        createMockPPContexts(pipeline_config.get());

        // Create graph builder
        Qwen2Graph graph_builder(graph_config_, mpi_ctx_);

        // Set weights
        auto weights = createModelWeights();
        graph_builder.setWeights(weights);

        // Create buffers
        int seq_len = 2;
        auto buffers = createBuffers(seq_len);
        graph_builder.setBuffers(buffers);

        // Create forward input
        auto input = createForwardInput(1, seq_len);

        // Build and execute
        Qwen2ForwardOutput output;
        ComputeGraph compute_graph = graph_builder.buildUnifiedPipelineGraph(input, output);

        std::unordered_map<DeviceId, IDeviceContext *> contexts;
        contexts[cpu_device_] = cpu_context_.get();

        DeviceGraphExecutor executor;
        bool success = executor.executeMultiDevice(compute_graph, contexts);
        ASSERT_TRUE(success);

        // Verify output
        ASSERT_NE(output.logits, nullptr) << "Output logits should be set";
        EXPECT_FALSE(hasNaNOrInf(output.logits)) << "Logits should not contain NaN or Inf";
    }

    /**
     * @test Output logits should have correct shape [seq_len, vocab_size]
     */
    TEST_F(Test__UnifiedPP_Execution, CorrectOutputShape)
    {
        // Create pipeline config
        auto pipeline_config = createTwoStageCPUConfig();

        // Create graph config
        graph_config_ = createGraphConfig();
        graph_config_.pipeline_config = pipeline_config;

        // Create mock PP contexts
        createMockPPContexts(pipeline_config.get());

        // Create graph builder
        Qwen2Graph graph_builder(graph_config_, mpi_ctx_);

        // Set weights
        auto weights = createModelWeights();
        graph_builder.setWeights(weights);

        // Create buffers
        int seq_len = 4;
        auto buffers = createBuffers(seq_len);
        graph_builder.setBuffers(buffers);

        // Create forward input
        auto input = createForwardInput(1, seq_len);

        // Build and execute
        Qwen2ForwardOutput output;
        ComputeGraph compute_graph = graph_builder.buildUnifiedPipelineGraph(input, output);

        std::unordered_map<DeviceId, IDeviceContext *> contexts;
        contexts[cpu_device_] = cpu_context_.get();

        DeviceGraphExecutor executor;
        bool success = executor.executeMultiDevice(compute_graph, contexts);
        ASSERT_TRUE(success);

        // Verify shape
        ASSERT_NE(output.logits, nullptr);
        EXPECT_EQ(output.logits->rows(), static_cast<size_t>(seq_len));
        EXPECT_EQ(output.logits->cols(), static_cast<size_t>(vocab_size_));
    }

    // =========================================================================
    // 2. Transfer Verification Tests
    // =========================================================================

    /**
     * @test MockPPContext.transfer() should be called during PP graph execution
     */
    TEST_F(Test__UnifiedPP_Execution, PPTransferStageCalled)
    {
        // Create pipeline config
        auto pipeline_config = createTwoStageCPUConfig();

        // Create graph config
        graph_config_ = createGraphConfig();
        graph_config_.pipeline_config = pipeline_config;

        // Create mock PP contexts
        createMockPPContexts(pipeline_config.get());

        // Get the mock context for transfer verification
        ASSERT_EQ(mock_pp_contexts_.size(), 1u) << "Should have one PP context for 2-stage";
        MockLocalPPContext *mock_ctx = mock_pp_contexts_[0].get();

        // Create graph builder
        Qwen2Graph graph_builder(graph_config_, mpi_ctx_);

        // Set weights
        auto weights = createModelWeights();
        graph_builder.setWeights(weights);

        // Create buffers
        int seq_len = 2;
        auto buffers = createBuffers(seq_len);
        graph_builder.setBuffers(buffers);

        // Create forward input
        auto input = createForwardInput(1, seq_len);

        // Build and execute
        Qwen2ForwardOutput output;
        ComputeGraph compute_graph = graph_builder.buildUnifiedPipelineGraph(input, output);

        std::unordered_map<DeviceId, IDeviceContext *> contexts;
        contexts[cpu_device_] = cpu_context_.get();

        DeviceGraphExecutor executor;
        bool success = executor.executeMultiDevice(compute_graph, contexts);
        ASSERT_TRUE(success);

        // Verify transfer was called
        EXPECT_GE(mock_ctx->transferCallCount(), 1)
            << "PP transfer should have been called at least once";
    }

    /**
     * @test Transfer stage should receive correct stage indices
     */
    TEST_F(Test__UnifiedPP_Execution, TransferStageIndicesCorrect)
    {
        // Create pipeline config
        auto pipeline_config = createTwoStageCPUConfig();

        // Create graph config
        graph_config_ = createGraphConfig();
        graph_config_.pipeline_config = pipeline_config;

        // Create mock PP contexts
        createMockPPContexts(pipeline_config.get());

        // Get the mock context
        ASSERT_EQ(mock_pp_contexts_.size(), 1u);
        MockLocalPPContext *mock_ctx = mock_pp_contexts_[0].get();

        // Create graph builder
        Qwen2Graph graph_builder(graph_config_, mpi_ctx_);

        // Set weights
        auto weights = createModelWeights();
        graph_builder.setWeights(weights);

        // Create buffers
        int seq_len = 2;
        auto buffers = createBuffers(seq_len);
        graph_builder.setBuffers(buffers);

        // Create forward input
        auto input = createForwardInput(1, seq_len);

        // Build and execute
        Qwen2ForwardOutput output;
        ComputeGraph compute_graph = graph_builder.buildUnifiedPipelineGraph(input, output);

        std::unordered_map<DeviceId, IDeviceContext *> contexts;
        contexts[cpu_device_] = cpu_context_.get();

        DeviceGraphExecutor executor;
        bool success = executor.executeMultiDevice(compute_graph, contexts);
        ASSERT_TRUE(success);

        // Verify transfer indices
        ASSERT_GE(mock_ctx->transferCallCount(), 1);
        EXPECT_TRUE(mock_ctx->hasTransfer(0, 1))
            << "Should have transfer from stage 0 to stage 1";
    }

    /**
     * @test Transfer should receive a tensor (hidden state)
     */
    TEST_F(Test__UnifiedPP_Execution, TransferReceivesCorrectTensor)
    {
        // Create pipeline config
        auto pipeline_config = createTwoStageCPUConfig();

        // Create graph config
        graph_config_ = createGraphConfig();
        graph_config_.pipeline_config = pipeline_config;

        // Create mock PP contexts
        createMockPPContexts(pipeline_config.get());

        // Get the mock context
        ASSERT_EQ(mock_pp_contexts_.size(), 1u);
        MockLocalPPContext *mock_ctx = mock_pp_contexts_[0].get();

        // Create graph builder
        Qwen2Graph graph_builder(graph_config_, mpi_ctx_);

        // Set weights
        auto weights = createModelWeights();
        graph_builder.setWeights(weights);

        // Create buffers
        int seq_len = 2;
        auto buffers = createBuffers(seq_len);
        graph_builder.setBuffers(buffers);

        // Create forward input
        auto input = createForwardInput(1, seq_len);

        // Build and execute
        Qwen2ForwardOutput output;
        ComputeGraph compute_graph = graph_builder.buildUnifiedPipelineGraph(input, output);

        std::unordered_map<DeviceId, IDeviceContext *> contexts;
        contexts[cpu_device_] = cpu_context_.get();

        DeviceGraphExecutor executor;
        bool success = executor.executeMultiDevice(compute_graph, contexts);
        ASSERT_TRUE(success);

        // Verify tensor was passed to transfer
        ASSERT_GE(mock_ctx->transferCallCount(), 1);
        auto last_call = mock_ctx->lastTransferCall();
        EXPECT_NE(last_call.activations, nullptr)
            << "Transfer should receive a tensor";
    }

    // =========================================================================
    // 3. Multi-Device Context Tests
    // =========================================================================

    /**
     * @test All stages in graph should be executed
     */
    TEST_F(Test__UnifiedPP_Execution, AllStagesExecute)
    {
        // Create pipeline config
        auto pipeline_config = createTwoStageCPUConfig();

        // Create graph config
        graph_config_ = createGraphConfig();
        graph_config_.pipeline_config = pipeline_config;

        // Create mock PP contexts
        createMockPPContexts(pipeline_config.get());

        // Create graph builder
        Qwen2Graph graph_builder(graph_config_, mpi_ctx_);

        // Set weights
        auto weights = createModelWeights();
        graph_builder.setWeights(weights);

        // Create buffers
        int seq_len = 2;
        auto buffers = createBuffers(seq_len);
        graph_builder.setBuffers(buffers);

        // Create forward input
        auto input = createForwardInput(1, seq_len);

        // Build graph
        Qwen2ForwardOutput output;
        ComputeGraph compute_graph = graph_builder.buildUnifiedPipelineGraph(input, output);

        // Count stages before execution
        size_t total_stages = compute_graph.size();
        EXPECT_GT(total_stages, 0u) << "Graph should have stages";

        // Execute
        std::unordered_map<DeviceId, IDeviceContext *> contexts;
        contexts[cpu_device_] = cpu_context_.get();

        DeviceGraphExecutor executor;
        bool success = executor.executeMultiDevice(compute_graph, contexts);
        ASSERT_TRUE(success);

        // Verify all stages completed
        EXPECT_TRUE(compute_graph.allCompleted())
            << "All graph stages should be marked completed";
    }

    /**
     * @test Graph should have correct structure with PP transfer stage
     */
    TEST_F(Test__UnifiedPP_Execution, GraphHasCorrectStructure)
    {
        // Create pipeline config
        auto pipeline_config = createTwoStageCPUConfig();

        // Create graph config
        graph_config_ = createGraphConfig();
        graph_config_.pipeline_config = pipeline_config;

        // Create mock PP contexts
        createMockPPContexts(pipeline_config.get());

        // Create graph builder
        Qwen2Graph graph_builder(graph_config_, mpi_ctx_);

        // Set weights
        auto weights = createModelWeights();
        graph_builder.setWeights(weights);

        // Create buffers
        int seq_len = 2;
        auto buffers = createBuffers(seq_len);
        graph_builder.setBuffers(buffers);

        // Create forward input
        auto input = createForwardInput(1, seq_len);

        // Build graph
        Qwen2ForwardOutput output;
        ComputeGraph compute_graph = graph_builder.buildUnifiedPipelineGraph(input, output);

        // Verify embedding stage exists
        int embedding_count = countStagesByType(compute_graph, ComputeStageType::EMBEDDING);
        EXPECT_EQ(embedding_count, 1) << "Should have one embedding stage";

        // Verify PP transfer stage exists
        int transfer_count = countStagesByType(compute_graph, ComputeStageType::LOCAL_PP_TRANSFER);
        EXPECT_EQ(transfer_count, 1) << "Should have one PP transfer stage for 2-stage PP";

        // Verify LM head stage exists
        int lm_head_count = countStagesByType(compute_graph, ComputeStageType::LM_HEAD);
        EXPECT_EQ(lm_head_count, 1) << "Should have one LM head stage";
    }

    /**
     * @test Verify execution order respects PP stage boundaries
     */
    TEST_F(Test__UnifiedPP_Execution, ExecutionOrderRespectsStages)
    {
        // Create pipeline config
        auto pipeline_config = createTwoStageCPUConfig();

        // Create graph config
        graph_config_ = createGraphConfig();
        graph_config_.pipeline_config = pipeline_config;

        // Create mock PP contexts
        createMockPPContexts(pipeline_config.get());

        // Create graph builder
        Qwen2Graph graph_builder(graph_config_, mpi_ctx_);

        // Set weights
        auto weights = createModelWeights();
        graph_builder.setWeights(weights);

        // Create buffers
        int seq_len = 2;
        auto buffers = createBuffers(seq_len);
        graph_builder.setBuffers(buffers);

        // Create forward input
        auto input = createForwardInput(1, seq_len);

        // Build graph
        Qwen2ForwardOutput output;
        ComputeGraph compute_graph = graph_builder.buildUnifiedPipelineGraph(input, output);

        // Get execution order
        auto execution_order = compute_graph.getExecutionOrder();

        // Find positions of key stages
        int embedding_pos = -1;
        int transfer_pos = -1;
        int lm_head_pos = -1;

        for (size_t i = 0; i < execution_order.size(); ++i)
        {
            const ComputeNode *node = compute_graph.getNode(execution_order[i]);
            if (node && node->stage)
            {
                if (node->stage->type() == ComputeStageType::EMBEDDING)
                {
                    embedding_pos = static_cast<int>(i);
                }
                else if (node->stage->type() == ComputeStageType::LOCAL_PP_TRANSFER)
                {
                    transfer_pos = static_cast<int>(i);
                }
                else if (node->stage->type() == ComputeStageType::LM_HEAD)
                {
                    lm_head_pos = static_cast<int>(i);
                }
            }
        }

        // Verify ordering
        EXPECT_NE(embedding_pos, -1) << "Embedding stage should exist";
        EXPECT_NE(transfer_pos, -1) << "Transfer stage should exist";
        EXPECT_NE(lm_head_pos, -1) << "LM head stage should exist";

        // Embedding should come before transfer
        EXPECT_LT(embedding_pos, transfer_pos)
            << "Embedding should execute before PP transfer";

        // Transfer should come before LM head
        EXPECT_LT(transfer_pos, lm_head_pos)
            << "PP transfer should execute before LM head";
    }

} // namespace
