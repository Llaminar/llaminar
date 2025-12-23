/**
 * @file Test__LayerExecutor_Q8_1_vs_FP32_Parity.cpp
 * @brief Integration test comparing Q8_1 vs FP32 activation precision for LayerExecutor
 *
 * This test validates that the LayerExecutor produces functionally equivalent
 * results when using Q8_1 vs FP32 activation precision with real model weights.
 *
 * Uses the Qwen2.5-0.5B Q8_0 model to test with real quantized weights, which
 * properly exercises the QuantizedGemmKernel with Q8_1 activation support.
 *
 * Test scenarios:
 *   - Single layer execution with both activation precisions
 *   - FFN-only execution to isolate FFN precision effects
 *   - Full layer (Attention + FFN) execution for complete parity
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <memory>
#include <fstream>

#include "v2/models/qwen/Qwen2Graph.h"
#include "v2/execution/GraphOrchestrator.h"
#include "v2/tensors/Tensors.h"
#include "v2/tensors/TensorFactory.h"
#include "v2/tensors/UnifiedKVCache.h"
#include "v2/tensors/SIMDHelpers.h"
#include "v2/backends/ComputeBackend.h"
#include "v2/loaders/ModelLoader.h"
#include "v2/utils/MPIContext.h"
#include "v2/utils/Logger.h"

using namespace llaminar2;

namespace
{
    // Model path - uses Q8_0 model which has quantized weights for proper Q8_1 activation testing
    constexpr const char *MODEL_PATH = "models/qwen2.5-0.5b-instruct-q8_0.gguf";

    /**
     * @brief Compute cosine similarity between two float arrays
     */
    double cosine_similarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        if (norm_a < 1e-12 || norm_b < 1e-12)
            return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    /**
     * @brief Compute max absolute difference between two float arrays
     */
    float max_abs_diff(const float *a, const float *b, size_t n)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            float diff = std::abs(a[i] - b[i]);
            if (diff > max_diff)
                max_diff = diff;
        }
        return max_diff;
    }

    /**
     * @brief Compute mean absolute difference between two float arrays
     */
    float mean_abs_diff(const float *a, const float *b, size_t n)
    {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            sum += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        }
        return static_cast<float>(sum / n);
    }

} // namespace

/**
 * @brief Test fixture for LayerExecutor Q8_1 vs FP32 parity using real model
 *
 * Loads the Qwen2.5-0.5B Q8_0 model and tests a single layer with both
 * FP32 and Q8_1 activation precision to verify parity.
 */
class Test__LayerExecutor_Q8_1_vs_FP32_Parity : public ::testing::Test
{
protected:
    // MPI context (single rank for unit testing)
    std::shared_ptr<MPIContext> mpi_ctx_;
    int device_idx_ = 0;

    // Model loader and weights
    std::unique_ptr<ModelLoader> loader_;
    bool model_loaded_ = false;

    // Model dimensions (from loaded model metadata)
    int d_model_ = 0;
    int n_heads_ = 0;
    int n_kv_heads_ = 0;
    int head_dim_ = 0;
    int d_ff_ = 0;
    float rms_eps_ = 1e-6f;
    float rope_theta_ = 1000000.0f;

    // Parity thresholds - account for Q8_1 quantization noise
    // Q8_1 introduces quantization error but should maintain high cosine similarity
    static constexpr double MIN_COSINE_SIMILARITY = 0.95;
    static constexpr float MAX_MEAN_ABS_DIFF = 0.5f;

    // Tensor factory for buffer creation
    std::unique_ptr<TensorFactory> factory_;

    // Owned weight tensors (loaded from model)
    std::vector<std::shared_ptr<TensorBase>> owned_weights_;

    void SetUp() override
    {
        // Initialize MPI context
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);

        // Initialize DeviceManager
        DeviceManager::instance().initialize(-1);

        const auto &devices = DeviceManager::instance().devices();
        if (devices.empty())
        {
            GTEST_SKIP() << "No compute devices available";
        }
        device_idx_ = 0;

        // Create tensor factory
        factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);

        // Load model if file exists
        if (!std::ifstream(MODEL_PATH).good())
        {
            GTEST_SKIP() << "Model file not found: " << MODEL_PATH;
        }

        loader_ = std::make_unique<ModelLoader>(factory_.get());
        model_loaded_ = loader_->loadModel(MODEL_PATH);

        if (!model_loaded_)
        {
            GTEST_SKIP() << "Failed to load model: " << MODEL_PATH;
        }

        // Extract model dimensions from metadata
        const auto &model = loader_->getModel();
        d_model_ = static_cast<int>(model.embedding_length);
        n_heads_ = static_cast<int>(model.head_count);
        n_kv_heads_ = static_cast<int>(model.head_count_kv);
        head_dim_ = d_model_ / n_heads_;
        rms_eps_ = model.rms_norm_eps;
        rope_theta_ = model.rope_theta;

        // d_ff is in metadata (qwen2.feed_forward_length)
        if (model.metadata.count("qwen2.feed_forward_length"))
        {
            d_ff_ = static_cast<int>(model.metadata.at("qwen2.feed_forward_length").asUInt32());
        }
        else
        {
            // Fallback: typical FFN expansion ratio
            d_ff_ = d_model_ * 4;
            LOG_WARN("Warning: feed_forward_length not in metadata, using " << d_ff_);
        }

        LOG_INFO("[Test] Model dimensions: d_model=" << d_model_
                                                     << " n_heads=" << n_heads_
                                                     << " n_kv_heads=" << n_kv_heads_
                                                     << " head_dim=" << head_dim_
                                                     << " d_ff=" << d_ff_);
    }

    void TearDown() override
    {
        owned_weights_.clear();
        loader_.reset();
    }

    /**
     * @brief Create executor with specified activation precision
     *
     * Returns a GraphOrchestrator wrapping a Qwen2Graph, as the Qwen2Graph
     * execute methods are deprecated.
     */
    std::unique_ptr<GraphOrchestrator> createExecutor(ActivationPrecision precision)
    {
        Qwen2GraphConfig config;
        config.d_model = d_model_;
        config.n_heads = n_heads_;
        config.n_kv_heads = n_kv_heads_;
        config.head_dim = head_dim_;
        config.d_ff = d_ff_;
        config.rms_norm_eps = rms_eps_;
        config.rope_theta = rope_theta_;
        config.default_device = device_idx_;
        config.activation_precision = precision;
        // NOTE: Decomposed attention is now always used (Phase 7 cleanup)

        auto graph = std::make_shared<Qwen2Graph>(config, mpi_ctx_);
        return std::make_unique<GraphOrchestrator>(graph, mpi_ctx_);
    }

    /**
     * @brief Load weights for specified layer from the model
     */
    Qwen2LayerWeights loadLayerWeights(int layer_idx)
    {
        Qwen2LayerWeights weights;
        std::string prefix = "blk." + std::to_string(layer_idx) + ".";

        // Load attention weights
        auto wq = loader_->loadTensor(prefix + "attn_q.weight", device_idx_);
        auto wk = loader_->loadTensor(prefix + "attn_k.weight", device_idx_);
        auto wv = loader_->loadTensor(prefix + "attn_v.weight", device_idx_);
        auto wo = loader_->loadTensor(prefix + "attn_output.weight", device_idx_);
        auto attn_norm = loader_->loadTensor(prefix + "attn_norm.weight", device_idx_);

        // Load attention biases (optional - may be nullptr for some models)
        auto q_bias = loader_->loadTensor(prefix + "attn_q.bias", device_idx_);
        auto k_bias = loader_->loadTensor(prefix + "attn_k.bias", device_idx_);
        auto v_bias = loader_->loadTensor(prefix + "attn_v.bias", device_idx_);

        // Load FFN weights
        auto gate_proj = loader_->loadTensor(prefix + "ffn_gate.weight", device_idx_);
        auto up_proj = loader_->loadTensor(prefix + "ffn_up.weight", device_idx_);
        auto down_proj = loader_->loadTensor(prefix + "ffn_down.weight", device_idx_);
        auto ffn_norm = loader_->loadTensor(prefix + "ffn_norm.weight", device_idx_);

        // Verify all required weights loaded (biases are optional)
        if (!wq || !wk || !wv || !wo || !attn_norm ||
            !gate_proj || !up_proj || !down_proj || !ffn_norm)
        {
            LOG_ERROR("[Test] Failed to load some weights for layer " << layer_idx);
            return weights;
        }

        // Store ownership
        owned_weights_.push_back(wq);
        owned_weights_.push_back(wk);
        owned_weights_.push_back(wv);
        owned_weights_.push_back(wo);
        owned_weights_.push_back(attn_norm);
        if (q_bias)
            owned_weights_.push_back(q_bias);
        if (k_bias)
            owned_weights_.push_back(k_bias);
        if (v_bias)
            owned_weights_.push_back(v_bias);
        owned_weights_.push_back(gate_proj);
        owned_weights_.push_back(up_proj);
        owned_weights_.push_back(down_proj);
        owned_weights_.push_back(ffn_norm);

        // Set pointers
        weights.wq = wq.get();
        weights.wk = wk.get();
        weights.wv = wv.get();
        weights.wo = wo.get();
        weights.attn_norm = attn_norm.get();
        weights.q_bias = q_bias ? q_bias.get() : nullptr;
        weights.k_bias = k_bias ? k_bias.get() : nullptr;
        weights.v_bias = v_bias ? v_bias.get() : nullptr;
        weights.gate_proj = gate_proj.get();
        weights.up_proj = up_proj.get();
        weights.down_proj = down_proj.get();
        weights.ffn_norm = ffn_norm.get();

        LOG_INFO("[Test] Loaded layer " << layer_idx << " weights:"
                                        << " wq=" << wq->dtype_name() << "[" << wq->shape()[0] << "x" << wq->shape()[1] << "]"
                                        << " gate=" << gate_proj->dtype_name() << "[" << gate_proj->shape()[0] << "x" << gate_proj->shape()[1] << "]");

        return weights;
    }

    /**
     * @brief Struct to hold owned activation buffer tensors
     */
    struct OwnedActivationBuffers
    {
        std::unique_ptr<TensorBase> residual;
        std::unique_ptr<TensorBase> normalized;
        std::unique_ptr<TensorBase> Q;
        std::unique_ptr<TensorBase> K;
        std::unique_ptr<TensorBase> V;
        std::unique_ptr<TensorBase> attn_output;
        std::unique_ptr<TensorBase> attn_proj;
        std::unique_ptr<TensorBase> gate;
        std::unique_ptr<TensorBase> up;
        std::unique_ptr<TensorBase> ffn_output;
        std::unique_ptr<TensorBase> current_hidden;
        std::unique_ptr<TensorBase> workspace_scores;
        std::unique_ptr<TensorBase> workspace_context;
    };

    OwnedActivationBuffers fp32_owned_buffers_;
    OwnedActivationBuffers q8_1_owned_buffers_;

    /**
     * @brief Create activation buffers for specified precision
     */
    Qwen2ActivationBuffers createBuffers(int seq_len, ActivationPrecision precision)
    {
        Qwen2ActivationBuffers buffers;

        OwnedActivationBuffers &owned = (precision == ActivationPrecision::Q8_1)
                                            ? q8_1_owned_buffers_
                                            : fp32_owned_buffers_;

        // All activation buffers use the requested precision
        owned.residual = factory_->createActivation({static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)}, precision, device_idx_);
        owned.normalized = factory_->createActivation({static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)}, precision, device_idx_);
        owned.Q = factory_->createActivation({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads_ * head_dim_)}, precision, device_idx_);
        owned.K = factory_->createActivation({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads_ * head_dim_)}, precision, device_idx_);
        owned.V = factory_->createActivation({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads_ * head_dim_)}, precision, device_idx_);
        owned.attn_output = factory_->createActivation({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads_ * head_dim_)}, precision, device_idx_);
        owned.attn_proj = factory_->createActivation({static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)}, precision, device_idx_);
        owned.gate = factory_->createActivation({static_cast<size_t>(seq_len), static_cast<size_t>(d_ff_)}, precision, device_idx_);
        owned.up = factory_->createActivation({static_cast<size_t>(seq_len), static_cast<size_t>(d_ff_)}, precision, device_idx_);
        owned.ffn_output = factory_->createActivation({static_cast<size_t>(seq_len), static_cast<size_t>(d_ff_)}, precision, device_idx_);
        owned.current_hidden = factory_->createActivation({static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)}, precision, device_idx_);

        // Workspace buffers (always FP32 - used for attention scores)
        owned.workspace_scores = factory_->createFP32({static_cast<size_t>(n_heads_ * seq_len), static_cast<size_t>(seq_len)}, device_idx_);
        owned.workspace_context = factory_->createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(head_dim_)}, device_idx_);

        buffers.residual = owned.residual.get();
        buffers.normalized = owned.normalized.get();
        buffers.Q = owned.Q.get();
        buffers.K = owned.K.get();
        buffers.V = owned.V.get();
        buffers.attn_output = owned.attn_output.get();
        buffers.attn_proj = owned.attn_proj.get();
        buffers.gate = owned.gate.get();
        buffers.up = owned.up.get();
        buffers.ffn_output = owned.ffn_output.get();
        buffers.current_hidden = owned.current_hidden.get();
        buffers.workspace_scores = owned.workspace_scores.get();
        buffers.workspace_context = owned.workspace_context.get();
        buffers.workspace_mask = nullptr;

        return buffers;
    }

    /**
     * @brief Initialize input tensor with pseudo-random test data
     */
    void initInput(TensorBase *tensor)
    {
        if (auto *fp32 = dynamic_cast<FP32Tensor *>(tensor))
        {
            float *data = fp32->mutable_data();
            for (size_t i = 0; i < fp32->numel(); ++i)
            {
                // Pseudo-random input values in [-0.1, 0.1] range
                // Note: Cast to int to avoid unsigned arithmetic issues
                int val = static_cast<int>((i * 17 + 31) % 97) - 48;
                data[i] = 0.1f * val / 48.0f;
            }
        }
        else if (auto *q8_1 = dynamic_cast<Q8_1Tensor *>(tensor))
        {
            // For Q8_1, quantize from FP32 values
            std::vector<float> fp32_values(q8_1->numel());
            for (size_t i = 0; i < fp32_values.size(); ++i)
            {
                int val = static_cast<int>((i * 17 + 31) % 97) - 48;
                fp32_values[i] = 0.1f * val / 48.0f;
            }
            simd::quantize_fp32_to_q8_1_blocks(fp32_values.data(),
                                               q8_1->mutable_q8_1_blocks(),
                                               fp32_values.size());
        }
    }

    /**
     * @brief Compare results between FP32 and Q8_1 execution
     */
    bool compareResults(TensorBase *fp32_output, TensorBase *q8_1_output, const std::string &label)
    {
        size_t n = fp32_output->numel();
        if (n != q8_1_output->numel())
        {
            LOG_ERROR("Size mismatch: FP32=" << fp32_output->numel()
                                             << " Q8_1=" << q8_1_output->numel());
            return false;
        }

        // Use fp32_data() to get comparable FP32 values from both tensors
        const float *fp32_data = fp32_output->fp32_data();
        const float *q8_1_data = q8_1_output->fp32_data();

        double cosine = cosine_similarity(fp32_data, q8_1_data, n);
        float max_diff = max_abs_diff(fp32_data, q8_1_data, n);
        float mean_diff = mean_abs_diff(fp32_data, q8_1_data, n);

        LOG_INFO("");
        LOG_INFO("=== " << label << " ===");
        LOG_INFO("  Cosine similarity: " << std::fixed << std::setprecision(6) << cosine);
        LOG_INFO("  Max abs diff: " << std::fixed << std::setprecision(6) << max_diff);
        LOG_INFO("  Mean abs diff: " << std::fixed << std::setprecision(6) << mean_diff);

        // Sample values
        LOG_INFO("  FP32 sample [0:4]: " << std::setprecision(6)
                                         << fp32_data[0] << ", " << fp32_data[1] << ", "
                                         << fp32_data[2] << ", " << fp32_data[3]);
        LOG_INFO("  Q8_1 sample [0:4]: " << std::setprecision(6)
                                         << q8_1_data[0] << ", " << q8_1_data[1] << ", "
                                         << q8_1_data[2] << ", " << q8_1_data[3]);

        bool pass = true;

        if (cosine < MIN_COSINE_SIMILARITY)
        {
            LOG_ERROR("  ✗ Cosine similarity " << cosine << " < " << MIN_COSINE_SIMILARITY);
            pass = false;
        }
        else
        {
            LOG_INFO("  ✓ Cosine similarity OK");
        }

        if (mean_diff > MAX_MEAN_ABS_DIFF)
        {
            LOG_ERROR("  ✗ Mean abs diff " << mean_diff << " > " << MAX_MEAN_ABS_DIFF);
            pass = false;
        }
        else
        {
            LOG_INFO("  ✓ Mean abs diff OK");
        }

        return pass;
    }
};

/**
 * @brief Test FFN execution parity between Q8_1 and FP32 using real model weights
 *
 * Runs FFN block only with real Q8_0 weights to test Q8_1 activation path.
 */
TEST_F(Test__LayerExecutor_Q8_1_vs_FP32_Parity, FFNParity)
{
    constexpr int SEQ_LEN = 4;
    constexpr int LAYER_IDX = 0;

    // Load real weights from model
    auto weights = loadLayerWeights(LAYER_IDX);
    ASSERT_NE(weights.gate_proj, nullptr) << "Failed to load layer weights";

    // Create FP32 executor and buffers
    auto fp32_executor = createExecutor(ActivationPrecision::FP32);
    auto fp32_buffers = createBuffers(SEQ_LEN, ActivationPrecision::FP32);

    // Initialize FP32 input with test data
    initInput(fp32_buffers.current_hidden);

    // Create Q8_1 executor and buffers
    auto q8_1_executor = createExecutor(ActivationPrecision::Q8_1);
    auto q8_1_buffers = createBuffers(SEQ_LEN, ActivationPrecision::Q8_1);

    // Initialize Q8_1 input with same test data
    initInput(q8_1_buffers.current_hidden);

    // Execute FFN on FP32
    LOG_INFO("[FFNParity] Running FP32 FFN with real Q8_0 weights...");
    bool fp32_ok = fp32_executor->executeFFN(weights, fp32_buffers, LAYER_IDX, SEQ_LEN, device_idx_);
    ASSERT_TRUE(fp32_ok) << "FP32 FFN execution failed";

    // Execute FFN on Q8_1
    LOG_INFO("[FFNParity] Running Q8_1 FFN with real Q8_0 weights...");
    bool q8_1_ok = q8_1_executor->executeFFN(weights, q8_1_buffers, LAYER_IDX, SEQ_LEN, device_idx_);
    ASSERT_TRUE(q8_1_ok) << "Q8_1 FFN execution failed";

    // Compare results
    bool parity_ok = compareResults(fp32_buffers.current_hidden, q8_1_buffers.current_hidden,
                                    "FFN Output (current_hidden)");
    EXPECT_TRUE(parity_ok) << "FFN output parity check failed";
}

/**
 * @brief Test full layer (Attention + FFN) parity between Q8_1 and FP32
 *
 * This test requires KV cache setup which is more complex.
 * Skipped for now - use E2E tests for full layer parity.
 */
TEST_F(Test__LayerExecutor_Q8_1_vs_FP32_Parity, FullLayerParity)
{
    // Full layer requires KV cache which adds complexity.
    // The FFN test above validates the core Q8_1 activation path.
    GTEST_SKIP() << "Full layer test requires KV cache setup - use E2E tests";
}

/**
 * @brief Stress test: Multiple sequence lengths for FFN parity
 *
 * Tests various sequence lengths to ensure Q8_1 parity holds across different
 * input sizes, catching any edge cases with block alignment etc.
 */
TEST_F(Test__LayerExecutor_Q8_1_vs_FP32_Parity, FFNParity_MultipleSeqLens)
{
    constexpr int LAYER_IDX = 0;
    const std::vector<int> seq_lens = {1, 4, 16, 32, 64};

    auto weights = loadLayerWeights(LAYER_IDX);
    ASSERT_NE(weights.gate_proj, nullptr) << "Failed to load layer weights";

    for (int seq_len : seq_lens)
    {
        LOG_INFO("[MultiSeqLen] Testing seq_len=" << seq_len);

        // Reset owned buffers for each iteration
        fp32_owned_buffers_ = OwnedActivationBuffers{};
        q8_1_owned_buffers_ = OwnedActivationBuffers{};

        auto fp32_executor = createExecutor(ActivationPrecision::FP32);
        auto fp32_buffers = createBuffers(seq_len, ActivationPrecision::FP32);
        initInput(fp32_buffers.current_hidden);

        auto q8_1_executor = createExecutor(ActivationPrecision::Q8_1);
        auto q8_1_buffers = createBuffers(seq_len, ActivationPrecision::Q8_1);
        initInput(q8_1_buffers.current_hidden);

        bool fp32_ok = fp32_executor->executeFFN(weights, fp32_buffers, LAYER_IDX, seq_len, device_idx_);
        ASSERT_TRUE(fp32_ok) << "FP32 FFN failed for seq_len=" << seq_len;

        bool q8_1_ok = q8_1_executor->executeFFN(weights, q8_1_buffers, LAYER_IDX, seq_len, device_idx_);
        ASSERT_TRUE(q8_1_ok) << "Q8_1 FFN failed for seq_len=" << seq_len;

        std::string label = "FFN Output (seq_len=" + std::to_string(seq_len) + ")";
        bool parity_ok = compareResults(fp32_buffers.current_hidden, q8_1_buffers.current_hidden, label);
        EXPECT_TRUE(parity_ok) << "Parity failed for seq_len=" << seq_len;
    }
}
