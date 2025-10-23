#pragma once

#include "MPIContext.h"
#include "ComputeBackend.h"
#include "tensors/TensorBaseInterface.h"
#include <memory>
#include <vector>

namespace llaminar
{

/**
 * @brief Qwen transformer pipeline with MPI + heterogeneous compute
 * 
 * Design principles:
 * - Direct kernel orchestration (no operator layer)
 * - Supports CPU, CUDA, ROCm, Vulkan backends
 * - MPI-aware for multi-node inference
 * - Minimal abstraction overhead
 */
class QwenPipeline
{
public:
    /**
     * @brief Construct Qwen pipeline
     * 
     * @param model_path Path to GGUF model file
     * @param mpi_ctx MPI context (nullptr = single node)
     * @param device_idx Device index for execution (-1 = CPU, ≥0 = GPU device from DeviceManager)
     */
    QwenPipeline(
        const std::string& model_path,
        std::shared_ptr<MPIContext> mpi_ctx = nullptr,
        int device_idx = -1);

    /**
     * @brief Forward pass (prefill or decode)
     * 
     * @param tokens Token IDs [seq_len]
     * @param seq_len Number of tokens
     * @return true on success, false on error
     */
    bool forward(const int* tokens, int seq_len);

    /**
     * @brief Get output logits (FP32)
     * 
     * @return Logits tensor [seq_len, vocab_size]
     */
    const float* logits() const { return logits_->data(); }

private:
    // Context management
    std::shared_ptr<MPIContext> mpi_ctx_;
    int device_idx_;  // -1 = CPU, ≥0 = GPU device index from DeviceManager

    // Model architecture
    int n_layers_ = 24;
    int n_heads_ = 14;
    int n_kv_heads_ = 2;
    int head_dim_ = 64;
    int d_model_ = 896;
    int d_ff_ = 4864;
    int vocab_size_ = 151936;

    // Weights (quantized, stay on host for CPU, uploaded to GPU for GPU backends)
    std::shared_ptr<TensorBase> embedding_table_;  // [vocab_size, d_model] FP32
    std::vector<std::shared_ptr<TensorBase>> wq_;  // [n_heads*head_dim, d_model] IQ4_NL
    std::vector<std::shared_ptr<TensorBase>> wk_;  // [n_kv_heads*head_dim, d_model] IQ4_NL
    std::vector<std::shared_ptr<TensorBase>> wv_;  // [n_kv_heads*head_dim, d_model] IQ4_NL
    std::vector<std::shared_ptr<TensorBase>> wo_;  // [d_model, n_heads*head_dim] IQ4_NL
    std::vector<std::shared_ptr<TensorBase>> gate_proj_;  // [d_ff, d_model] IQ4_NL
    std::vector<std::shared_ptr<TensorBase>> up_proj_;    // [d_ff, d_model] IQ4_NL
    std::vector<std::shared_ptr<TensorBase>> down_proj_;  // [d_model, d_ff] IQ4_NL
    std::shared_ptr<TensorBase> lm_head_;  // [vocab_size, d_model] IQ4_NL

    // Activations (FP32, on host or device depending on compute backend)
    std::shared_ptr<SimpleTensor> embeddings_;  // [seq_len, d_model]
    std::shared_ptr<SimpleTensor> logits_;      // [seq_len, vocab_size]

    // Reusable kernels (created once, reused across layers)
    std::unique_ptr<ITensorRoPE> rope_kernel_;
    std::unique_ptr<ITensorSoftmax> softmax_kernel_;
    std::unique_ptr<ITensorRMSNorm> rmsnorm_kernel_;
    std::unique_ptr<ITensorSwiGLU> swiglu_kernel_;

    // Helper methods
    bool load_weights(const std::string& model_path);
    void allocate_activations(int max_seq_len);
    
    /**
     * @brief Upload weights to GPU (if using GPU backend)
     */
    void upload_weights_to_device();
    
    /**
     * @brief Single transformer layer
     */
    bool transformer_layer(int layer_idx, int seq_len);
};

// Example implementation snippet
inline bool QwenPipeline::transformer_layer(int layer_idx, int seq_len)
{
    // ===== ATTENTION =====
    // Q/K/V projections (fused streaming dequant in kernel)
    auto wq_gemm = wq_[layer_idx]->createGemm();
    auto wk_gemm = wk_[layer_idx]->createGemm();
    auto wv_gemm = wv_[layer_idx]->createGemm();

    std::shared_ptr<SimpleTensor> Q = std::make_shared<SimpleTensor>(
        std::vector<size_t>{(size_t)seq_len, (size_t)(n_heads_ * head_dim_)});
    Q->set_device(device_idx_);
    std::shared_ptr<SimpleTensor> K = std::make_shared<SimpleTensor>(
        std::vector<size_t>{(size_t)seq_len, (size_t)(n_kv_heads_ * head_dim_)});
    K->set_device(device_idx_);
    std::shared_ptr<SimpleTensor> V = std::make_shared<SimpleTensor>(
        std::vector<size_t>{(size_t)seq_len, (size_t)(n_kv_heads_ * head_dim_)});
    V->set_device(device_idx_);

    // Execute GEMMs (automatic device dispatch based on device_idx)
    wq_gemm->multiply(embeddings_->data(), Q->mutable_data(), 
                      seq_len, n_heads_ * head_dim_, d_model_,
                      true, 1.0f, 0.0f, mpi_ctx_.get(), device_idx_);
    
    wk_gemm->multiply(embeddings_->data(), K->mutable_data(),
                      seq_len, n_kv_heads_ * head_dim_, d_model_,
                      true, 1.0f, 0.0f, mpi_ctx_.get(), device_idx_);
    
    wv_gemm->multiply(embeddings_->data(), V->mutable_data(),
                      seq_len, n_kv_heads_ * head_dim_, d_model_,
                      true, 1.0f, 0.0f, mpi_ctx_.get(), device_idx_);

    // RoPE (supports BF16 internally for 2× bandwidth reduction)
    rope_kernel_->apply(Q->mutable_data(), K->mutable_data(), 
                       nullptr,  // position_ids (nullptr = use default 0..seq_len-1)
                       seq_len, n_heads_, n_kv_heads_, head_dim_,
                       true,  // use_bf16
                       mpi_ctx_.get(), device_idx_);

    // Attention scores (always FP32 for numerical stability)
    // ... standard attention implementation ...
    
    // ===== FFN =====
    auto gate_gemm = gate_proj_[layer_idx]->createGemm();
    auto up_gemm = up_proj_[layer_idx]->createGemm();
    
    std::shared_ptr<SimpleTensor> gate = std::make_shared<SimpleTensor>(
        std::vector<size_t>{(size_t)seq_len, (size_t)d_ff_});
    gate->set_device(device_idx_);
    std::shared_ptr<SimpleTensor> up = std::make_shared<SimpleTensor>(
        std::vector<size_t>{(size_t)seq_len, (size_t)d_ff_});
    up->set_device(device_idx_);
    
    gate_gemm->multiply(embeddings_->data(), gate->mutable_data(),
                       seq_len, d_ff_, d_model_,
                       true, 1.0f, 0.0f, mpi_ctx_.get(), device_idx_);
    
    up_gemm->multiply(embeddings_->data(), up->mutable_data(),
                     seq_len, d_ff_, d_model_,
                     true, 1.0f, 0.0f, mpi_ctx_.get(), device_idx_);
    
    // SwiGLU (element-wise, can use BF16)
    swiglu_kernel_->apply(gate->data(), up->data(), gate->mutable_data(),
                         seq_len, d_ff_,
                         true,  // use_bf16
                         mpi_ctx_.get(), device_idx_);
    
    // ... rest of FFN ...
    
    return true;
}

} // namespace llaminar
