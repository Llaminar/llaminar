/**
 * @file ROCmMoEKernel.cpp
 * @brief ROCm MoE kernel implementation — calls extern "C" HIP bridges
 *
 * Separating .hip and .cpp allows hipcc to compile only the HIP code
 * without encountering issues with MPI or other complex C++ headers.
 */

#include "ROCmMoEKernel.h"
#include "../gemm/HipBLASGemmKernel.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../backends/DeviceId.h"
#include "../../../tensors/ITensor.h"
#include "../../../utils/Logger.h"
#include "../../../utils/ROCmKernelProfiler.h"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <stdexcept>

namespace
{
    bool setMoEDevice(int device_ordinal, const char *context)
    {
        hipError_t err = hipSetDevice(device_ordinal);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel] hipSetDevice(" << device_ordinal
                                                      << ") failed in " << context << ": " << hipGetErrorString(err));
            return false;
        }
        return true;
    }
}

// Forward-declare extern "C" bridge functions (defined in ROCmMoEKernels.hip)
extern "C"
{
    bool hipMoE_softmax_topk(
        float *logits,
        int *expert_indices, float *expert_weights,
        int seq_len, int num_experts, int top_k,
        bool normalize_weights,
        int device_idx, void *stream);

    bool hipMoE_gather_tokens(
        const float *hidden, float *batch_buffer,
        const int *token_indices,
        int num_tokens, int d_model,
        int device_idx, void *stream);

    bool hipMoE_scatter_add(
        float *output, const float *expert_output,
        const int *token_indices, const float *weights,
        int num_tokens, int d_model,
        int device_idx, void *stream);

    bool hipMoE_shared_expert_gate(
        const float *input, const float *gate_inp,
        float *shared_output, float *gate_scratch,
        int seq_len, int d_model,
        int device_idx, void *stream);

    bool hipMoE_swiglu(
        float *gate, const float *up,
        int count,
        int device_idx, void *stream);

    bool hipMoE_weighted_add(
        float *output, const float *input,
        float weight, int count,
        int device_idx, void *stream);

    // Phase 2: Histogram + Expert Mask bridges
    bool hipMoE_histogram_record(
        const int *routing_indices, unsigned long long *histogram,
        int seq_len, int num_experts, int top_k, int layer_idx,
        int device_idx, void *stream);

    bool hipMoE_apply_expert_mask(
        float *routing_weights, const int *routing_indices,
        const bool *expert_mask,
        int seq_len, int top_k,
        int device_idx, void *stream);

    bool hipMoE_histogram_reset(
        unsigned long long *histogram, int layer_idx, int num_experts,
        int device_idx, void *stream);

    // Phase 3: Token grouping bridges
    bool hipMoE_count_per_expert(
        const int *routing_indices, int *expert_counts,
        int total_slots, int num_experts,
        int device_idx, void *stream);

    bool hipMoE_exclusive_scan(
        int *expert_counts, int *expert_offsets,
        int num_experts,
        int device_idx, void *stream);

    bool hipMoE_scatter_tokens(
        const int *routing_indices, const float *routing_weights,
        const int *expert_offsets, int *write_heads,
        int *grouped_token_indices, float *grouped_weights,
        int total_slots, int num_experts, int top_k,
        int device_idx, void *stream);

    // Phase 4: Tensor-aware utility bridges
    bool hipMoE_int_to_float(
        const int *d_input, float *d_output, int count,
        int device_idx, void *stream);

    bool hipMoE_float_to_int(
        const float *d_input, int *d_output, int count,
        int device_idx, void *stream);
}

namespace llaminar2
{

    ROCmMoEKernel::ROCmMoEKernel(int device_ordinal)
        : device_ordinal_(device_ordinal)
    {
        auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_ordinal);
        ROCmKernelBase::setDeviceContext(&ctx);
        ROCmKernelBase::setGPUStream(ctx.defaultStream());

        // Create hipBLAS GEMM kernel using device context (shares hipBLAS handle)
        blas_gemm_ = std::make_unique<rocm::HipBLASGemmKernel>(&ctx);

        // Ensure hipBLAS runs on the same stream as our HIP kernels
        syncBlasStream();

        LOG_DEBUG("[ROCmMoEKernel] Created for ROCm device " << device_ordinal
                                                             << " stream=" << ROCmKernelBase::getStream());
    }

    ROCmMoEKernel::~ROCmMoEKernel()
    {
        (void)setMoEDevice(device_ordinal_, "destructor");
        if (d_histogram_)
        {
            hipFree(d_histogram_);
            d_histogram_ = nullptr;
        }
        if (d_expert_mask_)
        {
            hipFree(d_expert_mask_);
            d_expert_mask_ = nullptr;
        }
        if (d_write_heads_)
        {
            hipFree(d_write_heads_);
            d_write_heads_ = nullptr;
        }
        if (d_staging_indices_)
        {
            hipFree(d_staging_indices_);
            d_staging_indices_ = nullptr;
        }
        if (d_staging_weights_)
        {
            hipFree(d_staging_weights_);
            d_staging_weights_ = nullptr;
        }
    }

    void ROCmMoEKernel::syncBlasStream()
    {
        if (blas_gemm_)
            blas_gemm_->setStream(ROCmKernelBase::getStream());
    }

    // =========================================================================
    // routeCore() — Shared GPU routing logic: gate GEMM + softmax + top-k.
    // Returns device buffers; caller is responsible for D2H and hipFree.
    // =========================================================================

    bool ROCmMoEKernel::routeCore(
        const float *hidden, const float *gate_weights,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights, DeviceRouteBuffers &bufs)
    {
        bufs.logits_count = static_cast<size_t>(seq_len) * num_experts;
        bufs.topk_count = static_cast<size_t>(seq_len) * top_k;

        hipError_t err;
        err = hipMalloc(&bufs.d_logits, bufs.logits_count * sizeof(float));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::routeCore] hipMalloc d_logits failed: " << hipGetErrorString(err));
            return false;
        }

        err = hipMalloc(&bufs.d_indices, bufs.topk_count * sizeof(int));
        if (err != hipSuccess)
        {
            hipFree(bufs.d_logits);
            bufs.d_logits = nullptr;
            LOG_ERROR("[ROCmMoEKernel::routeCore] hipMalloc d_indices failed: " << hipGetErrorString(err));
            return false;
        }

        err = hipMalloc(&bufs.d_weights, bufs.topk_count * sizeof(float));
        if (err != hipSuccess)
        {
            hipFree(bufs.d_logits);
            hipFree(bufs.d_indices);
            bufs.d_logits = nullptr;
            bufs.d_indices = nullptr;
            LOG_ERROR("[ROCmMoEKernel::routeCore] hipMalloc d_weights failed: " << hipGetErrorString(err));
            return false;
        }

        // Gate logits via hipBLAS GEMM
        if (!blas_gemm_->execute(hidden, gate_weights, bufs.d_logits,
                                 seq_len, num_experts, d_model,
                                 /*transA=*/false, /*transB=*/true))
        {
            LOG_ERROR("[ROCmMoEKernel::routeCore] hipBLAS gate logits GEMM failed");
            hipFree(bufs.d_logits);
            hipFree(bufs.d_indices);
            hipFree(bufs.d_weights);
            bufs = {};
            return false;
        }

        // Softmax + top-k selection
        if (!hipMoE_softmax_topk(bufs.d_logits, bufs.d_indices, bufs.d_weights,
                                 seq_len, num_experts, top_k,
                                 normalize_weights,
                                 device_ordinal_, getStream()))
        {
            hipFree(bufs.d_logits);
            hipFree(bufs.d_indices);
            hipFree(bufs.d_weights);
            bufs = {};
            return false;
        }

        return true;
    }

    // =========================================================================
    // route() — Gate logits + softmax + top-k on GPU, results back to host
    // =========================================================================

    bool ROCmMoEKernel::route(
        const float *hidden,
        const float *gate_weights,
        int seq_len, int d_model,
        int num_experts, int top_k,
        bool normalize_weights,
        MoERoutingResult &result)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        DeviceRouteBuffers bufs;
        if (!routeCore(hidden, gate_weights, seq_len, d_model,
                       num_experts, top_k, normalize_weights, bufs))
            return false;

        // D2H results
        result.expert_indices.resize(bufs.topk_count);
        result.expert_weights.resize(bufs.topk_count);
        result.router_logits.resize(bufs.logits_count);

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err;

        err = hipMemcpyAsync(result.router_logits.data(), bufs.d_logits,
                             bufs.logits_count * sizeof(float),
                             hipMemcpyDeviceToHost, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::route] D2H logits failed: " << hipGetErrorString(err));
            hipFree(bufs.d_logits);
            hipFree(bufs.d_indices);
            hipFree(bufs.d_weights);
            return false;
        }

        err = hipMemcpyAsync(result.expert_indices.data(), bufs.d_indices,
                             bufs.topk_count * sizeof(int),
                             hipMemcpyDeviceToHost, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::route] D2H indices failed: " << hipGetErrorString(err));
            hipFree(bufs.d_logits);
            hipFree(bufs.d_indices);
            hipFree(bufs.d_weights);
            return false;
        }

        err = hipMemcpyAsync(result.expert_weights.data(), bufs.d_weights,
                             bufs.topk_count * sizeof(float),
                             hipMemcpyDeviceToHost, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::route] D2H weights failed: " << hipGetErrorString(err));
            hipFree(bufs.d_logits);
            hipFree(bufs.d_indices);
            hipFree(bufs.d_weights);
            return false;
        }

        hipStreamSynchronize(stream);
        hipFree(bufs.d_logits);
        hipFree(bufs.d_indices);
        hipFree(bufs.d_weights);
        return true;
    }

    // =========================================================================
    // gatherTokenBatch() — All pointers are device pointers
    // =========================================================================

    void ROCmMoEKernel::gatherTokenBatch(
        const float *hidden,
        float *batch_buffer,
        const int *token_indices,
        int num_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_GATHER, static_cast<hipStream_t>(getStream()));

        if (num_tokens <= 0)
            return;

        if (!hipMoE_gather_tokens(hidden, batch_buffer, token_indices,
                                  num_tokens, d_model,
                                  device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::gatherTokenBatch] kernel launch failed");
        }
    }

    // =========================================================================
    // scatterAddWeighted() — All pointers are device pointers
    // =========================================================================

    void ROCmMoEKernel::scatterAddWeighted(
        float *output,
        const float *expert_output,
        const int *token_indices,
        const float *weights,
        int num_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SCATTER, static_cast<hipStream_t>(getStream()));

        if (num_tokens <= 0)
            return;

        if (!hipMoE_scatter_add(output, expert_output, token_indices, weights,
                                num_tokens, d_model,
                                device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::scatterAddWeighted] kernel launch failed");
        }
    }

    // =========================================================================
    // sharedExpertGate() — All pointers are device pointers
    //
    // Needs a small scratch buffer for gate values [seq_len floats].
    // Allocates on demand (small — at most a few KB).
    // =========================================================================

    void ROCmMoEKernel::sharedExpertGate(
        const float *input,
        const float *gate_inp,
        float *shared_output,
        int seq_len, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SHARED_GATE, static_cast<hipStream_t>(getStream()));

        if (seq_len <= 0)
            return;

        // Small scratch for per-token gate values
        float *d_gate_scratch = nullptr;
        hipError_t err = hipMalloc(&d_gate_scratch, seq_len * sizeof(float));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGate] hipMalloc scratch failed: "
                      << hipGetErrorString(err));
            return;
        }

        if (!hipMoE_shared_expert_gate(input, gate_inp, shared_output, d_gate_scratch,
                                       seq_len, d_model,
                                       device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGate] kernel launch failed");
        }

        hipFree(d_gate_scratch);
    }

    // =========================================================================
    // swiGLU() — All pointers are device pointers
    // =========================================================================

    void ROCmMoEKernel::swiGLU(float *gate, const float *up, int count)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SWIGLU, static_cast<hipStream_t>(getStream()));

        if (count <= 0)
            return;

        if (!hipMoE_swiglu(gate, up, count, device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::swiGLU] kernel launch failed");
        }
    }

    void ROCmMoEKernel::weightedAdd(float *output, const float *input,
                                    float weight, int count)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SCATTER, static_cast<hipStream_t>(getStream()));

        if (count <= 0)
            return;

        if (!hipMoE_weighted_add(output, input, weight, count, device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::weightedAdd] kernel launch failed");
        }
    }

    // =========================================================================
    // Phase 2: Device-resident histogram + expert mask
    // =========================================================================

    void ROCmMoEKernel::allocateHistogramBuffers(int num_layers, int num_experts)
    {
        // Already allocated with sufficient dimensions?
        if (d_histogram_ && max_layers_ >= num_layers && max_experts_ >= num_experts)
            return;

        // Free old if dimensions grew
        if (d_histogram_)
        {
            hipFree(d_histogram_);
            d_histogram_ = nullptr;
        }

        max_layers_ = num_layers;
        max_experts_ = num_experts;

        const size_t total = static_cast<size_t>(max_layers_) * max_experts_;
        hipError_t err = hipMalloc(&d_histogram_, total * sizeof(uint64_t));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::allocateHistogramBuffers] hipMalloc histogram failed: "
                      << hipGetErrorString(err));
            d_histogram_ = nullptr;
            return;
        }

        err = hipMemset(d_histogram_, 0, total * sizeof(uint64_t));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::allocateHistogramBuffers] hipMemset failed: "
                      << hipGetErrorString(err));
        }

        LOG_DEBUG("[ROCmMoEKernel] Allocated histogram buffer: "
                  << max_layers_ << " layers × " << max_experts_ << " experts");
    }

    void ROCmMoEKernel::recordHistogramDevice(
        const int *d_routing_indices, int seq_len, int top_k, int layer_idx)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (seq_len <= 0 || top_k <= 0)
            return;

        // Lazy allocate — assume at least layer_idx+1 layers, 256 experts as initial guess
        const int min_experts = 256;
        if (!d_histogram_ || layer_idx >= max_layers_)
        {
            allocateHistogramBuffers(layer_idx + 1, (max_experts_ > 0) ? max_experts_ : min_experts);
        }
        if (!d_histogram_)
            return;

        if (!hipMoE_histogram_record(
                d_routing_indices,
                reinterpret_cast<unsigned long long *>(d_histogram_),
                seq_len, max_experts_, top_k, layer_idx,
                device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::recordHistogramDevice] kernel launch failed");
        }
    }

    void ROCmMoEKernel::syncHistogramToHost(
        uint64_t *host_counts, int layer_idx, int num_experts)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (!d_histogram_)
        {
            LOG_WARN("[ROCmMoEKernel::syncHistogramToHost] No histogram allocated");
            return;
        }
        if (layer_idx >= max_layers_ || num_experts > max_experts_)
        {
            LOG_ERROR("[ROCmMoEKernel::syncHistogramToHost] layer_idx=" << layer_idx
                                                                        << " or num_experts=" << num_experts << " out of range");
            return;
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        const size_t offset = static_cast<size_t>(layer_idx) * max_experts_;

        hipError_t err = hipMemcpyAsync(
            host_counts,
            d_histogram_ + offset,
            num_experts * sizeof(uint64_t),
            hipMemcpyDeviceToHost, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::syncHistogramToHost] D2H copy failed: "
                      << hipGetErrorString(err));
            return;
        }

        hipStreamSynchronize(stream);
    }

    void ROCmMoEKernel::resetHistogramDevice(int layer_idx, int num_experts)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (!d_histogram_)
        {
            LOG_WARN("[ROCmMoEKernel::resetHistogramDevice] No histogram allocated");
            return;
        }
        if (layer_idx >= max_layers_)
        {
            LOG_ERROR("[ROCmMoEKernel::resetHistogramDevice] layer_idx=" << layer_idx << " out of range");
            return;
        }

        if (!hipMoE_histogram_reset(
                reinterpret_cast<unsigned long long *>(d_histogram_),
                layer_idx, num_experts,
                device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::resetHistogramDevice] kernel launch failed");
        }
    }

    void ROCmMoEKernel::updateExpertMaskDevice(const bool *mask, int num_experts)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (num_experts <= 0)
            return;

        // Allocate or reallocate if needed
        if (!d_expert_mask_ || num_experts > max_experts_)
        {
            if (d_expert_mask_)
                hipFree(d_expert_mask_);

            hipError_t err = hipMalloc(&d_expert_mask_, num_experts * sizeof(bool));
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmMoEKernel::updateExpertMaskDevice] hipMalloc mask failed: "
                          << hipGetErrorString(err));
                d_expert_mask_ = nullptr;
                return;
            }
            // Update max_experts_ if the mask required more
            if (num_experts > max_experts_)
                max_experts_ = num_experts;
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(
            d_expert_mask_, mask,
            num_experts * sizeof(bool),
            hipMemcpyHostToDevice, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::updateExpertMaskDevice] H2D copy failed: "
                      << hipGetErrorString(err));
        }
    }

    void ROCmMoEKernel::applyExpertMaskDevice(
        float *d_routing_weights, const int *d_routing_indices,
        int seq_len, int top_k)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (seq_len <= 0 || top_k <= 0)
            return;

        if (!d_expert_mask_)
        {
            LOG_WARN("[ROCmMoEKernel::applyExpertMaskDevice] No expert mask uploaded");
            return;
        }

        if (!hipMoE_apply_expert_mask(
                d_routing_weights, d_routing_indices,
                d_expert_mask_,
                seq_len, top_k,
                device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::applyExpertMaskDevice] kernel launch failed");
        }
    }

    // =========================================================================
    // Phase 3: Device-side token grouping
    // =========================================================================

    bool ROCmMoEKernel::groupTokensByExpertDevice(
        const int *d_routing_indices,
        const float *d_routing_weights,
        int seq_len, int num_experts, int top_k,
        int *d_expert_offsets,
        int *d_expert_counts,
        int *d_grouped_token_indices,
        float *d_grouped_weights)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (seq_len <= 0 || num_experts <= 0 || top_k <= 0)
            return false;

        const int total_slots = seq_len * top_k;
        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err;

        // Step 1: Zero expert_counts
        err = hipMemsetAsync(d_expert_counts, 0, num_experts * sizeof(int), stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] hipMemsetAsync expert_counts failed: "
                      << hipGetErrorString(err));
            return false;
        }

        // Step 2: Count per expert
        if (!hipMoE_count_per_expert(d_routing_indices, d_expert_counts,
                                     total_slots, num_experts,
                                     device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] count_per_expert failed");
            return false;
        }

        // Step 3: Exclusive scan (expert_counts → expert_offsets)
        if (!hipMoE_exclusive_scan(d_expert_counts, d_expert_offsets,
                                   num_experts,
                                   device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] exclusive_scan failed");
            return false;
        }

        // Step 4: Lazy-allocate write_heads scratch buffer
        if (!d_write_heads_ || max_write_heads_experts_ < num_experts)
        {
            if (d_write_heads_)
                hipFree(d_write_heads_);

            err = hipMalloc(&d_write_heads_, num_experts * sizeof(int));
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] hipMalloc write_heads failed: "
                          << hipGetErrorString(err));
                d_write_heads_ = nullptr;
                max_write_heads_experts_ = 0;
                return false;
            }
            max_write_heads_experts_ = num_experts;
        }

        // Zero write_heads
        err = hipMemsetAsync(d_write_heads_, 0, num_experts * sizeof(int), stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] hipMemsetAsync write_heads failed: "
                      << hipGetErrorString(err));
            return false;
        }

        // Step 5: Scatter tokens into grouped arrays
        if (!hipMoE_scatter_tokens(d_routing_indices, d_routing_weights,
                                   d_expert_offsets, d_write_heads_,
                                   d_grouped_token_indices, d_grouped_weights,
                                   total_slots, num_experts, top_k,
                                   device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] scatter_tokens failed");
            return false;
        }

        return true;
    }

    // =========================================================================
    // Tensor-aware API overrides — GPU implementations
    //
    // These use gpu_data_ptr() for device-resident data and keep
    // all computation on device.  Small host-side arrays (token indices,
    // routing weights) are uploaded via cached staging buffers.
    // =========================================================================

    void ROCmMoEKernel::ensureStagingCapacity(int count)
    {
        if (count <= staging_capacity_)
            return;

        if (!setMoEDevice(device_ordinal_, "ensureStagingCapacity"))
            return;

        if (d_staging_indices_)
            (void)hipFree(d_staging_indices_);
        if (d_staging_weights_)
            (void)hipFree(d_staging_weights_);

        hipError_t err = hipMalloc(&d_staging_indices_, count * sizeof(int));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::ensureStagingCapacity] hipMalloc indices failed: "
                      << hipGetErrorString(err));
            d_staging_indices_ = nullptr;
            d_staging_weights_ = nullptr;
            staging_capacity_ = 0;
            return;
        }

        err = hipMalloc(&d_staging_weights_, count * sizeof(float));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::ensureStagingCapacity] hipMalloc weights failed: "
                      << hipGetErrorString(err));
            (void)hipFree(d_staging_indices_);
            d_staging_indices_ = nullptr;
            d_staging_weights_ = nullptr;
            staging_capacity_ = 0;
            return;
        }
        staging_capacity_ = count;
    }

    bool ROCmMoEKernel::routeWithTensors(
        ITensor *hidden, ITensor *gate_weights,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        MoERoutingResult &host_result)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        const float *h = static_cast<const float *>(hidden->gpu_data_ptr());
        const float *g = static_cast<const float *>(gate_weights->gpu_data_ptr());
        if (!h || !g)
        {
            LOG_ERROR("[ROCmMoEKernel::routeWithTensors] null device pointer "
                      "(hidden="
                      << (const void *)h << " gate=" << (const void *)g << ")");
            return false;
        }

        DeviceRouteBuffers bufs;
        if (!routeCore(h, g, seq_len, d_model, num_experts, top_k,
                       normalize_weights, bufs))
            return false;

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err;

        // D2D: write routing results to output tensors on device.
        // Indices need int→float conversion; weights are a D2D copy.
        float *d_idx = static_cast<float *>(output_indices->gpu_data_ptr());
        float *d_wt = static_cast<float *>(output_weights->gpu_data_ptr());
        if (!d_idx || !d_wt)
        {
            LOG_ERROR("[ROCmMoEKernel::routeWithTensors] output tensors have no device allocation");
            hipFree(bufs.d_logits);
            hipFree(bufs.d_indices);
            hipFree(bufs.d_weights);
            return false;
        }

        // int→float conversion kernel (indices are int on device, tensor stores float)
        hipMoE_int_to_float(bufs.d_indices, d_idx,
                            static_cast<int>(bufs.topk_count),
                            device_ordinal_, getStream());

        err = hipMemcpyAsync(d_wt, bufs.d_weights,
                             bufs.topk_count * sizeof(float),
                             hipMemcpyDeviceToDevice, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::routeWithTensors] D2D weights failed: " << hipGetErrorString(err));
            hipFree(bufs.d_logits);
            hipFree(bufs.d_indices);
            hipFree(bufs.d_weights);
            return false;
        }

        output_indices->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        output_weights->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        // D2H for host result (needed by CPU-side expert dispatch loop)
        host_result.expert_indices.resize(bufs.topk_count);
        host_result.expert_weights.resize(bufs.topk_count);
        host_result.router_logits.resize(bufs.logits_count);

        hipMemcpyAsync(host_result.router_logits.data(), bufs.d_logits,
                       bufs.logits_count * sizeof(float), hipMemcpyDeviceToHost, stream);
        hipMemcpyAsync(host_result.expert_indices.data(), bufs.d_indices,
                       bufs.topk_count * sizeof(int), hipMemcpyDeviceToHost, stream);
        hipMemcpyAsync(host_result.expert_weights.data(), bufs.d_weights,
                       bufs.topk_count * sizeof(float), hipMemcpyDeviceToHost, stream);

        hipStreamSynchronize(stream);

        hipFree(bufs.d_logits);
        hipFree(bufs.d_indices);
        hipFree(bufs.d_weights);
        return true;
    }

    void ROCmMoEKernel::zeroBuffer(ITensor *tensor, size_t bytes)
    {
        void *ptr = tensor->gpu_data_ptr();
        if (!ptr)
        {
            LOG_ERROR("[ROCmMoEKernel::zeroBuffer] tensor has no device allocation");
            return;
        }
        hipMemset(ptr, 0, bytes);
        tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    }

    void ROCmMoEKernel::gatherTokenBatchFromTensors(
        ITensor *hidden, ITensor *batch_buffer,
        const int *host_token_indices, int num_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_GATHER, static_cast<hipStream_t>(getStream()));

        if (num_tokens <= 0)
            return;

        const float *h = static_cast<const float *>(hidden->gpu_data_ptr());
        float *b = static_cast<float *>(batch_buffer->gpu_data_ptr());

        if (!h || !b)
        {
            LOG_ERROR("[ROCmMoEKernel::gatherTokenBatchFromTensors] null device pointer");
            return;
        }

        if (!setMoEDevice(device_ordinal_, "gatherTokenBatchFromTensors"))
            return;

        // Upload host token indices to device staging buffer
        ensureStagingCapacity(num_tokens);
        if (!d_staging_indices_)
            return;

        hipError_t err = hipMemcpyAsync(
            d_staging_indices_, host_token_indices,
            num_tokens * sizeof(int), hipMemcpyHostToDevice,
            static_cast<hipStream_t>(getStream()));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::gatherTokenBatchFromTensors] H2D token indices failed: "
                      << hipGetErrorString(err));
            return;
        }

        gatherTokenBatch(h, b, d_staging_indices_, num_tokens, d_model);
        batch_buffer->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    }

    void ROCmMoEKernel::scatterAddWeightedFromTensors(
        ITensor *output, ITensor *expert_output,
        const int *host_token_indices, const float *host_weights,
        int num_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SCATTER, static_cast<hipStream_t>(getStream()));

        if (num_tokens <= 0)
            return;

        float *o = static_cast<float *>(output->gpu_data_ptr());
        const float *e = static_cast<const float *>(expert_output->gpu_data_ptr());

        if (!o || !e)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterAddWeightedFromTensors] null device pointer");
            return;
        }

        if (!setMoEDevice(device_ordinal_, "scatterAddWeightedFromTensors"))
            return;

        // Upload host indices + weights to device staging
        ensureStagingCapacity(num_tokens);
        if (!d_staging_indices_ || !d_staging_weights_)
            return;

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(
            d_staging_indices_, host_token_indices,
            num_tokens * sizeof(int), hipMemcpyHostToDevice,
            stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterAddWeightedFromTensors] H2D token indices failed: "
                      << hipGetErrorString(err));
            return;
        }
        err = hipMemcpyAsync(
            d_staging_weights_, host_weights,
            num_tokens * sizeof(float), hipMemcpyHostToDevice,
            stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterAddWeightedFromTensors] H2D weights failed: "
                      << hipGetErrorString(err));
            return;
        }

        scatterAddWeighted(o, e, d_staging_indices_, d_staging_weights_,
                           num_tokens, d_model);
    }

    void ROCmMoEKernel::sharedExpertGateFromTensors(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        int seq_len, int d_model)
    {
        const float *in = static_cast<const float *>(input->gpu_data_ptr());
        const float *gi = static_cast<const float *>(gate_inp->gpu_data_ptr());
        float *so = static_cast<float *>(shared_output->gpu_data_ptr());

        if (!in || !gi || !so)
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateFromTensors] null device pointer");
            return;
        }

        sharedExpertGate(in, gi, so, seq_len, d_model);
        shared_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    }

    void ROCmMoEKernel::swiGLUFromTensors(ITensor *gate, ITensor *up, int count)
    {
        float *g = static_cast<float *>(gate->gpu_data_ptr());
        float *u = static_cast<float *>(up->gpu_data_ptr());

        if (!g || !u)
        {
            LOG_ERROR("[ROCmMoEKernel::swiGLUFromTensors] null device pointer");
            return;
        }

        swiGLU(g, u, count);
        gate->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    }

    void ROCmMoEKernel::weightedAddFromTensors(
        ITensor *output, ITensor *input, float weight, int count)
    {
        float *o = static_cast<float *>(output->gpu_data_ptr());
        const float *in = static_cast<const float *>(input->gpu_data_ptr());

        if (!o || !in)
        {
            LOG_ERROR("[ROCmMoEKernel::weightedAddFromTensors] null device pointer");
            return;
        }

        weightedAdd(o, in, weight, count);
    }

    // =========================================================================
    // Phase 4: GPU-side expert dispatch for prefill
    // =========================================================================

    bool ROCmMoEKernel::prepareExpertGroups(
        ITensor *routing_indices, ITensor *routing_weights,
        int seq_len, int num_experts, int top_k)
    {
        if (seq_len <= 0 || num_experts <= 0 || top_k <= 0)
            return false;

        const int total_slots = seq_len * top_k;
        hipStream_t stream = static_cast<hipStream_t>(getStream());

        // 1. Ensure routing tensors are on device
        routing_indices->ensureOnDevice(DeviceId::rocm(device_ordinal_));
        routing_weights->ensureOnDevice(DeviceId::rocm(device_ordinal_));

        const float *d_float_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_float_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_float_indices || !d_float_weights)
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroups] null device pointers");
            return false;
        }

        // 2. Lazy-allocate grouping buffers
        if (total_slots > group_slots_cap_)
        {
            if (d_group_int_indices_)
                hipFree(d_group_int_indices_);
            if (d_group_token_indices_)
                hipFree(d_group_token_indices_);
            if (d_group_weights_)
                hipFree(d_group_weights_);
            hipMalloc(&d_group_int_indices_, total_slots * sizeof(int));
            hipMalloc(&d_group_token_indices_, total_slots * sizeof(int));
            hipMalloc(&d_group_weights_, total_slots * sizeof(float));
            group_slots_cap_ = total_slots;
        }
        if (num_experts > group_experts_cap_)
        {
            if (d_group_offsets_)
                hipFree(d_group_offsets_);
            if (d_group_counts_)
                hipFree(d_group_counts_);
            hipMalloc(&d_group_offsets_, num_experts * sizeof(int));
            hipMalloc(&d_group_counts_, num_experts * sizeof(int));
            group_experts_cap_ = num_experts;
        }

        // 3. Convert float indices → int on device
        if (!hipMoE_float_to_int(d_float_indices, d_group_int_indices_,
                                 total_slots, device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroups] float_to_int failed");
            return false;
        }

        // 4. Group tokens by expert (counts, offsets, scatter)
        if (!groupTokensByExpertDevice(
                d_group_int_indices_, d_float_weights,
                seq_len, num_experts, top_k,
                d_group_offsets_, d_group_counts_,
                d_group_token_indices_, d_group_weights_))
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroups] groupTokensByExpertDevice failed");
            return false;
        }

        // 5. D2H expert counts and offsets (small — num_experts ints each)
        host_expert_counts_.resize(num_experts);
        host_expert_offsets_.resize(num_experts);
        hipMemcpyAsync(host_expert_counts_.data(), d_group_counts_,
                       num_experts * sizeof(int), hipMemcpyDeviceToHost, stream);
        hipMemcpyAsync(host_expert_offsets_.data(), d_group_offsets_,
                       num_experts * sizeof(int), hipMemcpyDeviceToHost, stream);
        hipStreamSynchronize(stream);

        prepared_num_experts_ = num_experts;
        return true;
    }

    int ROCmMoEKernel::getExpertTokenCount(int expert_id) const
    {
        if (expert_id < 0 || expert_id >= prepared_num_experts_)
            return 0;
        return host_expert_counts_[expert_id];
    }

    void ROCmMoEKernel::gatherExpertBatch(
        ITensor *hidden, ITensor *batch_buffer,
        int expert_id, int d_model)
    {
        int count = getExpertTokenCount(expert_id);
        if (count <= 0)
            return;

        const float *h = static_cast<const float *>(hidden->gpu_data_ptr());
        float *b = static_cast<float *>(batch_buffer->gpu_data_ptr());
        int offset = host_expert_offsets_[expert_id];

        gatherTokenBatch(h, b, d_group_token_indices_ + offset, count, d_model);
    }

    void ROCmMoEKernel::scatterExpertResults(
        ITensor *output, ITensor *expert_results,
        int expert_id, int d_model)
    {
        int count = getExpertTokenCount(expert_id);
        if (count <= 0)
            return;

        float *o = static_cast<float *>(output->gpu_data_ptr());
        const float *r = static_cast<const float *>(expert_results->gpu_data_ptr());
        int offset = host_expert_offsets_[expert_id];

        scatterAddWeighted(o, r, d_group_token_indices_ + offset,
                           d_group_weights_ + offset, count, d_model);
    }

} // namespace llaminar2
