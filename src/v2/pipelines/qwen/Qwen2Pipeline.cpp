/**
 * @file Qwen2Pipeline.cpp
 * @brief Qwen 2.x transformer pipeline implementation
 *
 * Greenfield V2 implementation with:
 * - Direct kernel orchestration (no operator layer)
 * - Streaming dequant in kernels (no slab cache)
 * - Per-tensor device placement
 * - Selective BF16 for bandwidth-bound ops
 *
 * @author David Sanftenberg
 */

#include "Qwen2Pipeline.h"
#include "../PipelineFactory.h"
#include "../../loaders/ModelLoader.h"
#include "../../tensors/TensorFactory.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <stdexcept>

namespace llaminar2
{

    // =============================================================================
    // Factory Registration
    // =============================================================================

    /**
     * @brief Creator function for Qwen2Pipeline
     */
    static std::unique_ptr<PipelineBase> createQwen2(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        int device_idx)
    {
        return std::make_unique<Qwen2Pipeline>(model_ctx, mpi_ctx, device_idx);
    }

    /**
     * @brief Register Qwen2Pipeline with factory
     * 
     * Made public so tests can force registration if needed
     */
    void ensureQwen2Registration()
    {
        static bool registered = false;
        if (!registered) {
            PipelineFactory::instance().registerCreator("qwen2", &createQwen2);
            registered = true;
        }
    }

    /**
     * @brief Automatic registration at startup
     */
    __attribute__((constructor)) static void initQwen2()
    {
        ensureQwen2Registration();
    }

    // =============================================================================
    // Pipeline Implementation
    // =============================================================================

    Qwen2Pipeline::Qwen2Pipeline(std::shared_ptr<ModelContext> model_ctx,
                                 std::shared_ptr<MPIContext> mpi_ctx,
                                 int device_idx)
        : PipelineBase(model_ctx, mpi_ctx, device_idx)
    {
        std::cout << "[Qwen2Pipeline] Initializing Qwen 2.x pipeline\n";

        // Read architecture from GGUF metadata
        const GGUFModel &model = model_ctx_->model();
        n_layers_ = static_cast<int>(model.block_count);
        d_model_ = static_cast<int>(model.embedding_length);
        vocab_size_ = static_cast<int>(model.vocab_size);
        n_heads_ = static_cast<int>(model.head_count);
        n_kv_heads_ = static_cast<int>(model.head_count_kv);
        
        // Calculate head_dim from d_model and n_heads
        head_dim_ = d_model_ / n_heads_;
        
        // Read FFN intermediate size from metadata
        if (model.hasMetadata("qwen2.feed_forward_length"))
        {
            d_ff_ = static_cast<int>(model.metadata.at("qwen2.feed_forward_length").asUInt32());
        }
        else
        {
            // Fallback: typical ratio for Qwen models
            d_ff_ = d_model_ * 4;
            std::cout << "[Qwen2Pipeline] Warning: feed_forward_length not in metadata, using " << d_ff_ << "\n";
        }

        std::cout << "[Qwen2Pipeline] Architecture: " << n_layers_ << " layers, "
                  << d_model_ << " d_model, " << vocab_size_ << " vocab\n";
        std::cout << "[Qwen2Pipeline] Attention: " << n_heads_ << " heads, "
                  << n_kv_heads_ << " KV heads (GQA), " << head_dim_ << " head_dim\n";
        std::cout << "[Qwen2Pipeline] FFN: " << d_ff_ << " intermediate_size (SwiGLU)\n";

        // Load model weights
        if (!load_weights(model_path_))
        {
            throw std::runtime_error("Failed to load model weights from: " + model_path_);
        }

        std::cout << "[Qwen2Pipeline] Pipeline initialized successfully\n";
    }

    bool Qwen2Pipeline::load_weights(const std::string &model_path)
    {
        std::cout << "[Qwen2Pipeline] Loading weights from: " << model_path << "\n";

        // Create TensorFactory for NUMA-aware allocation (if MPI context available)
        std::unique_ptr<TensorFactory> factory;
        if (mpi_ctx_)
        {
            factory = std::make_unique<TensorFactory>(*mpi_ctx_);
            std::cout << "[Qwen2Pipeline] Using TensorFactory with NUMA node "
                      << factory->getNumaNode() << " for rank " << mpi_ctx_->rank() << "\n";
        }
        else
        {
            std::cout << "[Qwen2Pipeline] No MPI context, using default tensor allocation\n";
        }

        // Load GGUF model file
        ModelLoader loader(factory.get());
        if (!loader.loadModel(model_path))
        {
            std::cerr << "[Qwen2Pipeline] Failed to load GGUF model: " << model_path << std::endl;
            return false;
        }

        const GGUFModel &model = loader.getModel();

        // Validate architecture
        if (model.architecture != "qwen2")
        {
            std::cerr << "[Qwen2Pipeline] Architecture mismatch: expected qwen2, got "
                      << model.architecture << std::endl;
            return false;
        }

        // Validate hyperparameters match hardcoded values
        // TODO: Use GGUF metadata to initialize architecture params instead of hardcoding
        if (model.block_count != static_cast<uint64_t>(n_layers_))
        {
            std::cerr << "[Qwen2Pipeline] Layer count mismatch: expected " << n_layers_
                      << ", got " << model.block_count << std::endl;
            return false;
        }

        std::cout << "[Qwen2Pipeline] GGUF validation passed\n";
        std::cout << "  Architecture: " << model.architecture << "\n";
        std::cout << "  Layers: " << model.block_count << "\n";
        std::cout << "  Hidden size: " << model.embedding_length << "\n";
        std::cout << "  Vocab size: " << model.vocab_size << "\n";

        // Load embedding table
        std::cout << "[Qwen2Pipeline] Loading embedding table...\n";
        embedding_table_ = loader.loadTensor("token_embd.weight", device_idx_);
        if (!embedding_table_)
        {
            std::cerr << "[Qwen2Pipeline] Failed to load embedding table" << std::endl;
            return false;
        }
        std::cout << "  Embedding shape: " << embedding_table_->shape()[0]
                  << " x " << embedding_table_->shape()[1] << "\n";

        // Load layer weights
        layers_.resize(n_layers_);
        for (int i = 0; i < n_layers_; ++i)
        {
            std::cout << "[Qwen2Pipeline] Loading layer " << i << " weights...\n";
            auto &layer = layers_[i];
            std::string prefix = "blk." + std::to_string(i) + ".";

            // Attention weights
            layer.wq = loader.loadTensor(prefix + "attn_q.weight", device_idx_);
            layer.wk = loader.loadTensor(prefix + "attn_k.weight", device_idx_);
            layer.wv = loader.loadTensor(prefix + "attn_v.weight", device_idx_);
            layer.wo = loader.loadTensor(prefix + "attn_output.weight", device_idx_);
            layer.attn_norm = loader.loadTensor(prefix + "attn_norm.weight", device_idx_);

            // FFN weights
            layer.gate_proj = loader.loadTensor(prefix + "ffn_gate.weight", device_idx_);
            layer.up_proj = loader.loadTensor(prefix + "ffn_up.weight", device_idx_);
            layer.down_proj = loader.loadTensor(prefix + "ffn_down.weight", device_idx_);
            layer.ffn_norm = loader.loadTensor(prefix + "ffn_norm.weight", device_idx_);

            // Validate all tensors loaded
            if (!layer.wq || !layer.wk || !layer.wv || !layer.wo || !layer.attn_norm ||
                !layer.gate_proj || !layer.up_proj || !layer.down_proj || !layer.ffn_norm)
            {
                std::cerr << "[Qwen2Pipeline] Failed to load some weights for layer " << i << std::endl;
                return false;
            }

            std::cout << "  Layer " << i << " loaded (Q: " << layer.wq->shape()[0] << "x" << layer.wq->shape()[1]
                      << ", K: " << layer.wk->shape()[0] << "x" << layer.wk->shape()[1]
                      << ", V: " << layer.wv->shape()[0] << "x" << layer.wv->shape()[1] << ")\n";
        }

        // Final norm and LM head
        std::cout << "[Qwen2Pipeline] Loading final norm and LM head...\n";
        final_norm_ = loader.loadTensor("output_norm.weight", device_idx_);
        lm_head_ = loader.loadTensor("output.weight", device_idx_);

        if (!final_norm_ || !lm_head_)
        {
            std::cerr << "[Qwen2Pipeline] Failed to load final norm or LM head" << std::endl;
            return false;
        }

        std::cout << "[Qwen2Pipeline] Weight loading complete\n";
        std::cout << "  Total layers: " << n_layers_ << "\n";
        std::cout << "  Final norm shape: " << final_norm_->shape()[0] << "\n";
        std::cout << "  LM head shape: " << lm_head_->shape()[0] << "x" << lm_head_->shape()[1] << "\n";

        return true;
    }

    bool Qwen2Pipeline::forward(const int *tokens, int seq_len)
    {
        std::cout << "[Qwen2Pipeline] Forward pass with seq_len=" << seq_len << "\n";

        // Allocate activation tensors if needed
        if (!current_hidden_ || static_cast<int>(current_hidden_->shape()[0]) != seq_len)
        {
            current_hidden_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)});
            std::cout << "[Qwen2Pipeline] Allocated hidden states: "
                      << seq_len << " x " << d_model_ << "\n";
        }

        // TODO: Implement embedding lookup
        // For now, just zero the hidden states
        std::memset(current_hidden_->mutable_data(), 0,
                    seq_len * d_model_ * sizeof(float));
        std::cout << "[Qwen2Pipeline] TODO: Implement embedding lookup\n";

        // Process all transformer layers
        for (int i = 0; i < n_layers_; ++i)
        {
            if (!transformer_layer(i, seq_len))
            {
                std::cerr << "[Qwen2Pipeline] Layer " << i << " failed\n";
                return false;
            }
        }

        // Final normalization
        std::cout << "[Qwen2Pipeline] TODO: Implement final RMSNorm\n";

        // LM head projection
        if (!logits_ || static_cast<int>(logits_->shape()[0]) != seq_len)
        {
            logits_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(vocab_size_)});
            std::cout << "[Qwen2Pipeline] Allocated logits: "
                      << seq_len << " x " << vocab_size_ << "\n";
        }

        std::cout << "[Qwen2Pipeline] TODO: Implement LM head projection\n";
        std::memset(logits_->mutable_data(), 0,
                    seq_len * vocab_size_ * sizeof(float));

        return true;
    }

    bool Qwen2Pipeline::transformer_layer(int layer_idx, int seq_len)
    {
        std::cout << "[Qwen2Pipeline] Processing layer " << layer_idx << "\n";

        const auto &layer = layers_[layer_idx];

        // Attention block
        if (!attention_block(layer, seq_len))
        {
            std::cerr << "[Qwen2Pipeline] Attention block failed in layer "
                      << layer_idx << "\n";
            return false;
        }

        // FFN block
        if (!ffn_block(layer, seq_len))
        {
            std::cerr << "[Qwen2Pipeline] FFN block failed in layer "
                      << layer_idx << "\n";
            return false;
        }

        return true;
    }

    bool Qwen2Pipeline::attention_block(const LayerWeights &layer, int seq_len)
    {
        std::cout << "[Qwen2Pipeline] TODO: Implement attention block\n";

        // TODO: Implement Qwen2 attention mechanism:
        // 1. RMSNorm (attn_norm)
        // 2. Q/K/V projections with GQA (n_heads vs n_kv_heads)
        // 3. RoPE application (theta=10000.0 for Qwen2)
        // 4. Attention computation (Q·K^T / sqrt(d_head), softmax, ·V)
        // 5. Output projection
        // 6. Residual connection

        return true;
    }

    bool Qwen2Pipeline::ffn_block(const LayerWeights &layer, int seq_len)
    {
        std::cout << "[Qwen2Pipeline] TODO: Implement FFN block\n";

        // TODO: Implement Qwen2 FFN mechanism (SwiGLU):
        // 1. RMSNorm (ffn_norm)
        // 2. Gate and up projections
        // 3. SwiGLU activation: gate_out * silu(up_out)
        //    where silu(x) = x * sigmoid(x)
        // 4. Down projection
        // 5. Residual connection

        return true;
    }

    const float *Qwen2Pipeline::logits() const
    {
        if (!logits_)
        {
            std::cerr << "[Qwen2Pipeline] logits() called before forward()\n";
            return nullptr;
        }
        return logits_->data();
    }

    std::shared_ptr<TensorBase> Qwen2Pipeline::get_layer_weight(
        int layer_idx, const std::string &weight_name)
    {
        if (layer_idx < 0 || layer_idx >= n_layers_)
        {
            std::cerr << "[Qwen2Pipeline] Invalid layer index: " << layer_idx << "\n";
            return nullptr;
        }

        const auto &layer = layers_[layer_idx];

        if (weight_name == "wq")
            return layer.wq;
        if (weight_name == "wk")
            return layer.wk;
        if (weight_name == "wv")
            return layer.wv;
        if (weight_name == "wo")
            return layer.wo;
        if (weight_name == "gate")
            return layer.gate_proj;
        if (weight_name == "up")
            return layer.up_proj;
        if (weight_name == "down")
            return layer.down_proj;
        if (weight_name == "attn_norm")
            return layer.attn_norm;
        if (weight_name == "ffn_norm")
            return layer.ffn_norm;

        std::cerr << "[Qwen2Pipeline] Unknown weight name: " << weight_name << "\n";
        return nullptr;
    }

} // namespace llaminar2
