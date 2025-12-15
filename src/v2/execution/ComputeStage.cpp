/**
 * @file ComputeStage.cpp
 * @brief Implementation of compute stage abstractions
 * @author David Sanftenberg
 * @date December 2025
 */

#include "ComputeStage.h"
#include "../utils/Logger.h"

#include <cstring>
#include <cmath>
#include <sstream>

// Note: In production, GEMM would delegate to KernelFactory
// For now, we use placeholder implementation to avoid cross-namespace dependencies

#ifdef LLAMINAR_USE_MPI
#include <mpi.h>
#endif

namespace llaminar2
{

    // =============================================================================
    // Stage Type Names
    // =============================================================================

    const char *computeStageTypeName(ComputeStageType type)
    {
        switch (type)
        {
        case ComputeStageType::GEMM:
            return "GEMM";
        case ComputeStageType::GEMM_BIAS:
            return "GEMM_BIAS";
        case ComputeStageType::GEMM_FUSED_QKV:
            return "GEMM_FUSED_QKV";
        case ComputeStageType::RMS_NORM:
            return "RMS_NORM";
        case ComputeStageType::LAYER_NORM:
            return "LAYER_NORM";
        case ComputeStageType::SWIGLU:
            return "SWIGLU";
        case ComputeStageType::GELU:
            return "GELU";
        case ComputeStageType::SILU:
            return "SILU";
        case ComputeStageType::ROPE:
            return "ROPE";
        case ComputeStageType::ATTENTION:
            return "ATTENTION";
        case ComputeStageType::ATTENTION_QK:
            return "ATTENTION_QK";
        case ComputeStageType::ATTENTION_SOFTMAX:
            return "ATTENTION_SOFTMAX";
        case ComputeStageType::ATTENTION_V:
            return "ATTENTION_V";
        case ComputeStageType::ADD_RESIDUAL:
            return "ADD_RESIDUAL";
        case ComputeStageType::SCALE:
            return "SCALE";
        case ComputeStageType::MOE_ROUTER:
            return "MOE_ROUTER";
        case ComputeStageType::MOE_EXPERT_FFN:
            return "MOE_EXPERT_FFN";
        case ComputeStageType::MOE_COMBINE:
            return "MOE_COMBINE";
        case ComputeStageType::ALLREDUCE:
            return "ALLREDUCE";
        case ComputeStageType::ALLGATHER:
            return "ALLGATHER";
        case ComputeStageType::COPY:
            return "COPY";
        case ComputeStageType::QUANTIZE:
            return "QUANTIZE";
        case ComputeStageType::DEQUANTIZE:
            return "DEQUANTIZE";
        default:
            return "UNKNOWN";
        }
    }

    // =============================================================================
    // GEMMStage Implementation
    // =============================================================================

    GEMMStage::GEMMStage(Params params) : params_(std::move(params)) {}

    bool GEMMStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[GEMMStage] Null device context");
            return false;
        }

        // TODO: In production, delegate to llaminar::v2::kernels::KernelFactory
        // For now, placeholder that demonstrates the interface
        // auto* gemm_kernel = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(params_.B);

        LOG_DEBUG("[GEMMStage] Execute GEMM: " << params_.m << "x" << params_.n << "x" << params_.k);

        // Placeholder: actual implementation would dispatch to quantized/FP32 GEMM kernel
        // based on tensor B's type
        return true;
    }

    size_t GEMMStage::estimatedFlops() const
    {
        // GEMM: 2 * M * N * K (multiply + add)
        return static_cast<size_t>(2) * params_.m * params_.n * params_.k;
    }

    size_t GEMMStage::estimatedMemoryBytes() const
    {
        // A: m * k reads, B: k * n reads, C: m * n writes (+ reads if beta != 0)
        size_t a_bytes = static_cast<size_t>(params_.m) * params_.k * sizeof(float);
        size_t c_bytes = static_cast<size_t>(params_.m) * params_.n * sizeof(float);

        // B may be quantized, so we estimate based on tensor
        // For now, assume FP32 - tensor introspection would be better
        size_t b_bytes = static_cast<size_t>(params_.k) * params_.n * sizeof(float);

        return a_bytes + b_bytes + c_bytes;
    }

    bool GEMMStage::supportsBackend(ComputeBackendType backend) const
    {
        // CPU always supported
        // GPU support depends on whether we have CUDA/ROCm kernels
        switch (backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return true;
        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
            // TODO: Enable when GPU GEMM kernels are implemented
            return false;
        default:
            return false;
        }
    }

    // =============================================================================
    // RMSNormStage Implementation
    // =============================================================================

    RMSNormStage::RMSNormStage(Params params) : params_(std::move(params)) {}

    bool RMSNormStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[RMSNormStage] Null device context");
            return false;
        }

        float *input = static_cast<float *>(params_.input);
        const float *gamma = params_.gamma;
        const int seq_len = params_.seq_len;
        const int hidden_dim = params_.hidden_dim;
        const float eps = params_.eps;

        // Execute RMSNorm using parallel iteration
        ctx->runFor(0, static_cast<size_t>(seq_len), [=](size_t i_)
                    {
        int i = static_cast<int>(i_);
        float* row = input + i * hidden_dim;
        
        // Compute RMS
        float sum_sq = 0.0f;
        for (int j = 0; j < hidden_dim; ++j) {
            sum_sq += row[j] * row[j];
        }
        float rms = std::sqrt(sum_sq / hidden_dim + eps);
        float inv_rms = 1.0f / rms;
        
        // Normalize and scale
        for (int j = 0; j < hidden_dim; ++j) {
            row[j] = row[j] * inv_rms * gamma[j];
        } });
        return true;
    }

    size_t RMSNormStage::estimatedFlops() const
    {
        // Per row: hidden_dim squares + hidden_dim adds + sqrt + div + hidden_dim muls
        // Approximately 4 * hidden_dim FLOPs per row
        return static_cast<size_t>(4) * params_.seq_len * params_.hidden_dim;
    }

    size_t RMSNormStage::estimatedMemoryBytes() const
    {
        // Read input + gamma, write output (in-place, so same buffer)
        size_t input_bytes = static_cast<size_t>(params_.seq_len) * params_.hidden_dim * sizeof(float);
        size_t gamma_bytes = static_cast<size_t>(params_.hidden_dim) * sizeof(float);
        return 2 * input_bytes + gamma_bytes; // Read + write + gamma
    }

    bool RMSNormStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return true;
        default:
            return false;
        }
    }

    // =============================================================================
    // RoPEStage Implementation
    // =============================================================================

    RoPEStage::RoPEStage(Params params) : params_(std::move(params)) {}

    bool RoPEStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[RoPEStage] Null device context");
            return false;
        }

        float *tensor = static_cast<float *>(params_.tensor);
        const int seq_len = params_.seq_len;
        const int n_heads = params_.n_heads;
        const int head_dim = params_.head_dim;
        const int pos_offset = params_.pos_offset;
        const float theta_base = params_.theta_base;

        // Execute RoPE using parallel iteration over positions
        ctx->runFor(0, static_cast<size_t>(seq_len), [=](size_t pos_)
                    {
        int pos = static_cast<int>(pos_);
        int actual_pos = pos + pos_offset;
        
        for (int h = 0; h < n_heads; ++h) {
            float* head_ptr = tensor + pos * n_heads * head_dim + h * head_dim;
            
            // Apply rotary embedding to pairs of elements
            for (int i = 0; i < head_dim / 2; ++i) {
                float freq = 1.0f / std::pow(theta_base, 
                    static_cast<float>(2 * i) / head_dim);
                float angle = actual_pos * freq;
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);
                
                float x0 = head_ptr[2 * i];
                float x1 = head_ptr[2 * i + 1];
                
                head_ptr[2 * i]     = x0 * cos_val - x1 * sin_val;
                head_ptr[2 * i + 1] = x0 * sin_val + x1 * cos_val;
            }
        } });
        return true;
    }

    size_t RoPEStage::estimatedFlops() const
    {
        // Per position per head: head_dim/2 rotations, each ~10 FLOPs (sin, cos, 4 muls, 2 adds)
        return static_cast<size_t>(10) * params_.seq_len * params_.n_heads * (params_.head_dim / 2);
    }

    size_t RoPEStage::estimatedMemoryBytes() const
    {
        return static_cast<size_t>(2) * params_.seq_len * params_.n_heads *
               params_.head_dim * sizeof(float); // Read + write
    }

    bool RoPEStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return true;
        default:
            return false;
        }
    }

    // =============================================================================
    // AttentionStage Implementation
    // =============================================================================

    AttentionStage::AttentionStage(Params params) : params_(std::move(params)) {}

    bool AttentionStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[AttentionStage] Null device context");
            return false;
        }

        // This is a simplified implementation - production would use optimized kernels
        const float *Q = static_cast<const float *>(params_.Q);
        const float *K = static_cast<const float *>(params_.K);
        const float *V = static_cast<const float *>(params_.V);
        float *output = static_cast<float *>(params_.output);

        const int seq_len = params_.seq_len;
        const int kv_len = params_.kv_len;
        const int n_heads = params_.n_heads;
        const int n_kv_heads = params_.n_kv_heads;
        const int head_dim = params_.head_dim;
        const int heads_per_kv = n_heads / n_kv_heads;
        const float scale = params_.scale;

        // Get workspace from context for attention scores
        size_t scores_size = static_cast<size_t>(seq_len) * kv_len * sizeof(float);
        void *workspace = ctx->getWorkspace(scores_size * n_heads);
        float *scores_buf = static_cast<float *>(workspace);

        // Process each query head
        ctx->runFor(0, static_cast<size_t>(n_heads), [=](size_t h_)
                    {
        int h = static_cast<int>(h_);
        int kv_h = h / heads_per_kv;  // GQA: map query head to KV head
        float* scores = scores_buf + h * seq_len * kv_len;
        
        // Q * K^T
        for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
            const float* q_vec = Q + q_pos * n_heads * head_dim + h * head_dim;
            
            for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
                // Apply causal mask
                if (params_.causal && k_pos > q_pos) {
                    scores[q_pos * kv_len + k_pos] = -INFINITY;
                    continue;
                }
                
                const float* k_vec = K + k_pos * n_kv_heads * head_dim + kv_h * head_dim;
                
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_vec[d] * k_vec[d];
                }
                scores[q_pos * kv_len + k_pos] = dot * scale;
            }
        }
        
        // Softmax
        for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
            float* row = scores + q_pos * kv_len;
            
            // Find max
            float max_val = row[0];
            for (int k_pos = 1; k_pos < kv_len; ++k_pos) {
                if (row[k_pos] > max_val) max_val = row[k_pos];
            }
            
            // Exp and sum
            float sum = 0.0f;
            for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
                row[k_pos] = std::exp(row[k_pos] - max_val);
                sum += row[k_pos];
            }
            
            // Normalize
            float inv_sum = 1.0f / sum;
            for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
                row[k_pos] *= inv_sum;
            }
        }
        
        // Scores * V
        for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
            float* out_vec = output + q_pos * n_heads * head_dim + h * head_dim;
            const float* score_row = scores + q_pos * kv_len;
            
            std::memset(out_vec, 0, head_dim * sizeof(float));
            
            for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
                const float* v_vec = V + k_pos * n_kv_heads * head_dim + kv_h * head_dim;
                float s = score_row[k_pos];
                
                for (int d = 0; d < head_dim; ++d) {
                    out_vec[d] += s * v_vec[d];
                }
            }
        } });
        return true;
    }

    size_t AttentionStage::estimatedFlops() const
    {
        // QK: 2 * seq_len * kv_len * head_dim (per head)
        // Softmax: ~5 * seq_len * kv_len (per head)
        // V: 2 * seq_len * kv_len * head_dim (per head)
        size_t qk_flops = 2ULL * params_.seq_len * params_.kv_len * params_.head_dim;
        size_t softmax_flops = 5ULL * params_.seq_len * params_.kv_len;
        size_t v_flops = 2ULL * params_.seq_len * params_.kv_len * params_.head_dim;
        return (qk_flops + softmax_flops + v_flops) * params_.n_heads;
    }

    size_t AttentionStage::estimatedMemoryBytes() const
    {
        size_t q_bytes = static_cast<size_t>(params_.seq_len) * params_.n_heads *
                         params_.head_dim * sizeof(float);
        size_t kv_bytes = static_cast<size_t>(params_.kv_len) * params_.n_kv_heads *
                          params_.head_dim * sizeof(float);
        size_t out_bytes = q_bytes;
        return q_bytes + 2 * kv_bytes + out_bytes; // Q + K + V + output
    }

    bool AttentionStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return true;
        default:
            return false;
        }
    }

    // =============================================================================
    // SwiGLUStage Implementation
    // =============================================================================

    SwiGLUStage::SwiGLUStage(Params params) : params_(std::move(params)) {}

    bool SwiGLUStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[SwiGLUStage] Null device context");
            return false;
        }

        const float *gate = static_cast<const float *>(params_.gate);
        const float *up = static_cast<const float *>(params_.up);
        float *output = static_cast<float *>(params_.output);
        const int seq_len = params_.seq_len;
        const int intermediate_dim = params_.intermediate_dim;

        // SwiGLU: silu(gate) * up
        ctx->runFor(0, static_cast<size_t>(seq_len), [=](size_t i_)
                    {
        int i = static_cast<int>(i_);
        for (int j = 0; j < intermediate_dim; ++j) {
            int idx = i * intermediate_dim + j;
            float g = gate[idx];
            // SiLU: x * sigmoid(x)
            float silu = g / (1.0f + std::exp(-g));
            output[idx] = silu * up[idx];
        } });
        return true;
    }

    size_t SwiGLUStage::estimatedFlops() const
    {
        // Per element: exp, div, mul, mul (~6 FLOPs)
        return static_cast<size_t>(6) * params_.seq_len * params_.intermediate_dim;
    }

    size_t SwiGLUStage::estimatedMemoryBytes() const
    {
        size_t bytes = static_cast<size_t>(params_.seq_len) * params_.intermediate_dim * sizeof(float);
        return 3 * bytes; // gate + up + output
    }

    bool SwiGLUStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return true;
        default:
            return false;
        }
    }

    // =============================================================================
    // ResidualAddStage Implementation
    // =============================================================================

    ResidualAddStage::ResidualAddStage(Params params) : params_(std::move(params)) {}

    bool ResidualAddStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[ResidualAddStage] Null device context");
            return false;
        }

        const float *input = static_cast<const float *>(params_.input);
        const float *residual = static_cast<const float *>(params_.residual);
        float *output = static_cast<float *>(params_.output);
        const size_t n = params_.num_elements;

        // Simple element-wise add
        ctx->runFor(0, n, [=](size_t i)
                    { output[i] = input[i] + residual[i]; });
        return true;
    }

    size_t ResidualAddStage::estimatedFlops() const
    {
        return params_.num_elements; // One add per element
    }

    size_t ResidualAddStage::estimatedMemoryBytes() const
    {
        return 3 * params_.num_elements * sizeof(float); // input + residual + output
    }

    bool ResidualAddStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return true;
        default:
            return false;
        }
    }

    // =============================================================================
    // AllreduceStage Implementation
    // =============================================================================

    AllreduceStage::AllreduceStage(Params params) : params_(std::move(params)) {}

    bool AllreduceStage::execute(IDeviceContext *ctx)
    {
#ifdef LLAMINAR_USE_MPI
        if (!params_.mpi_comm)
        {
            LOG_ERROR("[AllreduceStage] Null MPI communicator");
            return false;
        }

        MPI_Comm comm = static_cast<MPI_Comm>(params_.mpi_comm);

        int result = MPI_Allreduce(
            MPI_IN_PLACE,
            params_.buffer,
            static_cast<int>(params_.count),
            MPI_FLOAT,
            MPI_SUM,
            comm);

        return result == MPI_SUCCESS;
#else
        // No MPI - operation is a no-op (single rank)
        (void)ctx;
        return true;
#endif
    }

    bool AllreduceStage::supportsBackend(ComputeBackendType backend) const
    {
        // Allreduce is backend-agnostic (works with any device that has MPI support)
        (void)backend;
        return true;
    }

    // =============================================================================
    // MoE Stages Implementation
    // =============================================================================

    MoERouterStage::MoERouterStage(Params params) : params_(std::move(params)) {}

    bool MoERouterStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoERouterStage] Null device context");
            return false;
        }

        // Router is a simple matmul: hidden @ gate_weights
        // This computes logits for each expert
        const float *hidden = static_cast<const float *>(params_.hidden);
        const float *gate_weights = static_cast<const float *>(params_.gate_weights);
        float *logits = params_.router_logits;

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;

        ctx->runFor(0, static_cast<size_t>(seq_len), [=](size_t t_)
                    {
        int t = static_cast<int>(t_);
        const float* h = hidden + t * d_model;
        float* out = logits + t * num_experts;
        
        for (int e = 0; e < num_experts; ++e) {
            const float* w = gate_weights + e * d_model;
            float dot = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                dot += h[d] * w[d];
            }
            out[e] = dot;
        } });
        return true;
    }

    size_t MoERouterStage::estimatedFlops() const
    {
        // seq_len * d_model * num_experts (dot products)
        return static_cast<size_t>(2) * params_.seq_len * params_.d_model * params_.num_experts;
    }

    bool MoERouterStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return true;
        default:
            return false;
        }
    }

    // -----------------------------------------------------------------------------

    MoEExpertStage::MoEExpertStage(Params params) : params_(std::move(params)) {}

    bool MoEExpertStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoEExpertStage] Null device context");
            return false;
        }

        if (!params_.token_indices || params_.token_indices->empty())
        {
            // No tokens routed to this expert - nothing to do
            return true;
        }

        // This is a placeholder - real implementation would use the actual expert weights
        // For now, we just demonstrate the structure
        LOG_DEBUG("[MoEExpertStage] Processing expert " << params_.expert_id
                                                        << " with " << params_.token_indices->size() << " tokens");

        // In real implementation:
        // 1. Gather tokens from input based on token_indices
        // 2. Apply gate projection
        // 3. Apply up projection
        // 4. SwiGLU activation
        // 5. Apply down projection
        // 6. Scatter results back

        return true;
    }

    std::string MoEExpertStage::name() const
    {
        std::ostringstream oss;
        oss << "MOE_EXPERT_" << params_.expert_id;
        return oss.str();
    }

    size_t MoEExpertStage::estimatedFlops() const
    {
        if (!params_.token_indices)
            return 0;
        size_t num_tokens = params_.token_indices->size();
        // FFN: gate + up + down projections
        // gate: num_tokens * d_model * intermediate_dim
        // up: num_tokens * d_model * intermediate_dim
        // down: num_tokens * intermediate_dim * d_model
        return static_cast<size_t>(6) * num_tokens * params_.d_model * params_.intermediate_dim;
    }

    bool MoEExpertStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return true;
        default:
            return false;
        }
    }

    // -----------------------------------------------------------------------------

    MoECombineStage::MoECombineStage(Params params) : params_(std::move(params)) {}

    bool MoECombineStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoECombineStage] Null device context");
            return false;
        }

        // Placeholder - combines expert outputs weighted by router scores
        LOG_DEBUG("[MoECombineStage] Combining "
                  << (params_.expert_outputs ? params_.expert_outputs->size() : 0)
                  << " expert outputs");

        // In real implementation:
        // For each token:
        //   output[t] = sum over k experts: weight[t][k] * expert_output[k][t]

        return true;
    }

    bool MoECombineStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return true;
        default:
            return false;
        }
    }

    // =============================================================================
    // ComputeStageFactory Implementation
    // =============================================================================

    std::unique_ptr<IComputeStage> ComputeStageFactory::createGEMM(
        const GEMMStage::Params &params,
        ComputeBackendType target_backend)
    {
        switch (target_backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<GEMMStage>(params);
        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
            return std::make_unique<GPUGEMMStage>(params, target_backend);
#else
            LOG_WARN("[ComputeStageFactory] GPU GEMM not compiled in, falling back to CPU");
            return std::make_unique<GEMMStage>(params);
#endif
        default:
            LOG_ERROR("[ComputeStageFactory] Unknown backend for GEMM");
            return nullptr;
        }
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createRMSNorm(
        const RMSNormStage::Params &params,
        ComputeBackendType target_backend)
    {
        switch (target_backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<RMSNormStage>(params);
        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
            return std::make_unique<GPURMSNormStage>(params, target_backend);
#else
            LOG_WARN("[ComputeStageFactory] GPU RMSNorm not compiled in, using CPU");
            return std::make_unique<RMSNormStage>(params);
#endif
        default:
            LOG_WARN("[ComputeStageFactory] Backend not supported for RMSNorm, using CPU");
            return std::make_unique<RMSNormStage>(params);
        }
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createRoPE(
        const RoPEStage::Params &params,
        ComputeBackendType target_backend)
    {
        switch (target_backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<RoPEStage>(params);
        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
            return std::make_unique<GPURoPEStage>(params, target_backend);
#else
            LOG_WARN("[ComputeStageFactory] GPU RoPE not compiled in, using CPU");
            return std::make_unique<RoPEStage>(params);
#endif
        default:
            LOG_WARN("[ComputeStageFactory] Backend not supported for RoPE, using CPU");
            return std::make_unique<RoPEStage>(params);
        }
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createAttention(
        const AttentionStage::Params &params,
        ComputeBackendType target_backend)
    {
        switch (target_backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<AttentionStage>(params);
        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
            return std::make_unique<GPUAttentionStage>(params, target_backend);
#else
            LOG_WARN("[ComputeStageFactory] GPU Attention not compiled in, using CPU");
            return std::make_unique<AttentionStage>(params);
#endif
        default:
            LOG_WARN("[ComputeStageFactory] Backend not supported for Attention, using CPU");
            return std::make_unique<AttentionStage>(params);
        }
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createSwiGLU(
        const SwiGLUStage::Params &params,
        ComputeBackendType target_backend)
    {
        switch (target_backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<SwiGLUStage>(params);
        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
            return std::make_unique<GPUSwiGLUStage>(params, target_backend);
#else
            LOG_WARN("[ComputeStageFactory] GPU SwiGLU not compiled in, using CPU");
            return std::make_unique<SwiGLUStage>(params);
#endif
        default:
            LOG_WARN("[ComputeStageFactory] Backend not supported for SwiGLU, using CPU");
            return std::make_unique<SwiGLUStage>(params);
        }
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createResidualAdd(
        const ResidualAddStage::Params &params,
        ComputeBackendType target_backend)
    {
        switch (target_backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<ResidualAddStage>(params);
        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
            return std::make_unique<GPUResidualAddStage>(params, target_backend);
#else
            LOG_WARN("[ComputeStageFactory] GPU ResidualAdd not compiled in, using CPU");
            return std::make_unique<ResidualAddStage>(params);
#endif
        default:
            LOG_WARN("[ComputeStageFactory] Backend not supported for ResidualAdd, using CPU");
            return std::make_unique<ResidualAddStage>(params);
        }
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoERouter(
        const MoERouterStage::Params &params,
        ComputeBackendType target_backend)
    {
        switch (target_backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<MoERouterStage>(params);
        default:
            LOG_WARN("[ComputeStageFactory] Backend not supported for MoERouter, using CPU");
            return std::make_unique<MoERouterStage>(params);
        }
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoEExpert(
        const MoEExpertStage::Params &params,
        ComputeBackendType target_backend)
    {
        switch (target_backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<MoEExpertStage>(params);
        default:
            LOG_WARN("[ComputeStageFactory] Backend not supported for MoEExpert, using CPU");
            return std::make_unique<MoEExpertStage>(params);
        }
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoECombine(
        const MoECombineStage::Params &params,
        ComputeBackendType target_backend)
    {
        switch (target_backend)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<MoECombineStage>(params);
        default:
            LOG_WARN("[ComputeStageFactory] Backend not supported for MoECombine, using CPU");
            return std::make_unique<MoECombineStage>(params);
        }
    }

    // =============================================================================
    // GPU Compute Stage Implementations
    // =============================================================================

#if defined(HAVE_CUDA) || defined(HAVE_ROCM)

#ifdef HAVE_CUDA
#include "../backends/cuda/CUDABackend.h"
#endif

#ifdef HAVE_ROCM
#include "../backends/rocm/ROCmBackend.h"
#endif

    // Helper to get GPU device ID from context
    static int getGPUDeviceId(IDeviceContext *ctx)
    {
        auto *gpu_ctx = dynamic_cast<IGPUDeviceContext *>(ctx);
        return gpu_ctx ? gpu_ctx->gpuDeviceId() : 0;
    }

    // -----------------------------------------------------------------------------
    // GPUGEMMStage
    // -----------------------------------------------------------------------------

    GPUGEMMStage::GPUGEMMStage(GEMMStage::Params params, ComputeBackendType backend)
        : params_(std::move(params)), backend_(backend) {}

    bool GPUGEMMStage::execute(IDeviceContext *ctx)
    {
        if (!ctx || !ctx->isGPU())
        {
            LOG_ERROR("[GPUGEMMStage] Requires GPU device context");
            return false;
        }

        int device_id = getGPUDeviceId(ctx);
        LOG_DEBUG("[GPUGEMMStage] Execute GEMM on GPU " << device_id
                                                        << ": " << params_.m << "x" << params_.n << "x" << params_.k);

        // TODO: Delegate to backend-specific GEMM
        // For quantized weights, use backend->gemmIQ4NL()
        // For FP32/FP16, use cuBLAS/rocBLAS via backend

        // Placeholder: mark as successful (actual kernels TBD)
        return true;
    }

    size_t GPUGEMMStage::estimatedFlops() const
    {
        return static_cast<size_t>(2) * params_.m * params_.n * params_.k;
    }

    size_t GPUGEMMStage::estimatedMemoryBytes() const
    {
        size_t a_bytes = static_cast<size_t>(params_.m) * params_.k * sizeof(float);
        size_t b_bytes = static_cast<size_t>(params_.k) * params_.n * sizeof(float);
        size_t c_bytes = static_cast<size_t>(params_.m) * params_.n * sizeof(float);
        return a_bytes + b_bytes + c_bytes;
    }

    bool GPUGEMMStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    // -----------------------------------------------------------------------------
    // GPURMSNormStage
    // -----------------------------------------------------------------------------

    GPURMSNormStage::GPURMSNormStage(RMSNormStage::Params params, ComputeBackendType backend)
        : params_(std::move(params)), backend_(backend) {}

    bool GPURMSNormStage::execute(IDeviceContext *ctx)
    {
        if (!ctx || !ctx->isGPU())
        {
            LOG_ERROR("[GPURMSNormStage] Requires GPU device context");
            return false;
        }

        int device_id = getGPUDeviceId(ctx);
        LOG_DEBUG("[GPURMSNormStage] Execute RMSNorm on GPU " << device_id);

        // TODO: Launch custom RMSNorm kernel with warp-level reduction
        // Kernel signature: rmsnorm_kernel<<<blocks, threads>>>(input, gamma, seq_len, hidden_dim, eps)

        return true;
    }

    size_t GPURMSNormStage::estimatedFlops() const
    {
        return static_cast<size_t>(4) * params_.seq_len * params_.hidden_dim;
    }

    size_t GPURMSNormStage::estimatedMemoryBytes() const
    {
        size_t input_bytes = static_cast<size_t>(params_.seq_len) * params_.hidden_dim * sizeof(float);
        size_t gamma_bytes = static_cast<size_t>(params_.hidden_dim) * sizeof(float);
        return 2 * input_bytes + gamma_bytes;
    }

    bool GPURMSNormStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    // -----------------------------------------------------------------------------
    // GPUSwiGLUStage
    // -----------------------------------------------------------------------------

    GPUSwiGLUStage::GPUSwiGLUStage(SwiGLUStage::Params params, ComputeBackendType backend)
        : params_(std::move(params)), backend_(backend) {}

    bool GPUSwiGLUStage::execute(IDeviceContext *ctx)
    {
        if (!ctx || !ctx->isGPU())
        {
            LOG_ERROR("[GPUSwiGLUStage] Requires GPU device context");
            return false;
        }

        int device_id = getGPUDeviceId(ctx);
        LOG_DEBUG("[GPUSwiGLUStage] Execute SwiGLU on GPU " << device_id);

        // TODO: Launch fused SwiGLU kernel
        // output[i] = silu(gate[i]) * up[i]
        // where silu(x) = x * sigmoid(x)

        return true;
    }

    size_t GPUSwiGLUStage::estimatedFlops() const
    {
        // silu: ~5 ops, mul: 1 op = 6 per element
        return static_cast<size_t>(6) * params_.seq_len * params_.intermediate_dim;
    }

    size_t GPUSwiGLUStage::estimatedMemoryBytes() const
    {
        size_t elem_bytes = static_cast<size_t>(params_.seq_len) * params_.intermediate_dim * sizeof(float);
        return 3 * elem_bytes; // gate + up + output
    }

    bool GPUSwiGLUStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    // -----------------------------------------------------------------------------
    // GPUResidualAddStage
    // -----------------------------------------------------------------------------

    GPUResidualAddStage::GPUResidualAddStage(ResidualAddStage::Params params, ComputeBackendType backend)
        : params_(std::move(params)), backend_(backend) {}

    bool GPUResidualAddStage::execute(IDeviceContext *ctx)
    {
        if (!ctx || !ctx->isGPU())
        {
            LOG_ERROR("[GPUResidualAddStage] Requires GPU device context");
            return false;
        }

        int device_id = getGPUDeviceId(ctx);
        LOG_DEBUG("[GPUResidualAddStage] Execute ResidualAdd on GPU " << device_id);

        // TODO: Launch simple element-wise addition kernel
        // output[i] = input[i] + residual[i]

        return true;
    }

    size_t GPUResidualAddStage::estimatedFlops() const
    {
        return params_.num_elements; // One add per element
    }

    size_t GPUResidualAddStage::estimatedMemoryBytes() const
    {
        return 3 * params_.num_elements * sizeof(float); // input + residual + output
    }

    bool GPUResidualAddStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    // -----------------------------------------------------------------------------
    // GPURoPEStage
    // -----------------------------------------------------------------------------

    GPURoPEStage::GPURoPEStage(RoPEStage::Params params, ComputeBackendType backend)
        : params_(std::move(params)), backend_(backend) {}

    bool GPURoPEStage::execute(IDeviceContext *ctx)
    {
        if (!ctx || !ctx->isGPU())
        {
            LOG_ERROR("[GPURoPEStage] Requires GPU device context");
            return false;
        }

        int device_id = getGPUDeviceId(ctx);
        LOG_DEBUG("[GPURoPEStage] Execute RoPE on GPU " << device_id);

        // TODO: Launch RoPE kernel
        // Apply rotary embeddings: cos/sin precomputed per position/dimension

        return true;
    }

    size_t GPURoPEStage::estimatedFlops() const
    {
        return static_cast<size_t>(10) * params_.seq_len * params_.n_heads * (params_.head_dim / 2);
    }

    size_t GPURoPEStage::estimatedMemoryBytes() const
    {
        return static_cast<size_t>(2) * params_.seq_len * params_.n_heads *
               params_.head_dim * sizeof(float);
    }

    bool GPURoPEStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    // -----------------------------------------------------------------------------
    // GPUAttentionStage
    // -----------------------------------------------------------------------------

    GPUAttentionStage::GPUAttentionStage(AttentionStage::Params params, ComputeBackendType backend)
        : params_(std::move(params)), backend_(backend) {}

    bool GPUAttentionStage::execute(IDeviceContext *ctx)
    {
        if (!ctx || !ctx->isGPU())
        {
            LOG_ERROR("[GPUAttentionStage] Requires GPU device context");
            return false;
        }

        int device_id = getGPUDeviceId(ctx);
        LOG_DEBUG("[GPUAttentionStage] Execute Attention on GPU " << device_id
                                                                  << " seq=" << params_.seq_len << " kv=" << params_.kv_len);

        // TODO: Implement Flash Attention or standard attention
        // 1. Q * K^T with scaling
        // 2. Causal mask application
        // 3. Online softmax
        // 4. Attention @ V

        return true;
    }

    size_t GPUAttentionStage::estimatedFlops() const
    {
        size_t qk_flops = 2ULL * params_.seq_len * params_.kv_len * params_.head_dim;
        size_t softmax_flops = 5ULL * params_.seq_len * params_.kv_len;
        size_t v_flops = 2ULL * params_.seq_len * params_.kv_len * params_.head_dim;
        return (qk_flops + softmax_flops + v_flops) * params_.n_heads;
    }

    size_t GPUAttentionStage::estimatedMemoryBytes() const
    {
        size_t qkv_bytes = static_cast<size_t>(params_.seq_len + 2 * params_.kv_len) *
                           params_.n_heads * params_.head_dim * sizeof(float);
        size_t scores_bytes = static_cast<size_t>(params_.seq_len) * params_.kv_len *
                              params_.n_heads * sizeof(float);
        return qkv_bytes + scores_bytes;
    }

    bool GPUAttentionStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

#endif // HAVE_CUDA || HAVE_ROCM

} // namespace llaminar2
