/**
 * @file AttentionComputeStage.cpp
 * @brief Implementation of AttentionComputeStage
 */

#include "AttentionComputeStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/TQ8Tensor.h"
#include "../../../tensors/TQ4Tensor.h"
#include "../../../kernels/cpu/CPUKVCache.h"
#include "../../../kernels/cpu/turboquant/TurboQuantContext.h"
#include "../../../kernels/cpu/rotation/ActivationRotation.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/KernelFactory.h"
#include <algorithm>
#include <limits>
#include <fstream>
#include <filesystem>

#if defined(HAVE_CUDA)
#include "../../../kernels/cuda/kvcache/CUDARingKVCacheTQ.h"
#include "../../../kernels/cuda/attention/CUDAFlashAttentionKernelT.h"
#endif

namespace llaminar2
{

    // =============================================================================
    // AttentionComputeStage Implementation
    // =============================================================================

    AttentionComputeStage::AttentionComputeStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    // =============================================================================
    // Kernel Caching (for Workspace Binding)
    // =============================================================================

    ITensorAttention *AttentionComputeStage::getOrCreateKernel()
    {
        // Need Q tensor to determine kernel type
        if (!params_.Q)
        {
            LOG_ERROR("[AttentionComputeStage::getOrCreateKernel] Q tensor not set");
            return nullptr;
        }

        auto *kernel = getOrRefreshKernelByTensorType(
            cached_kernel_,
            cached_kernel_tensor_type_,
            params_.Q,
            [&]()
            {
                return llaminar::v2::kernels::KernelFactory::getOrCreateAttention(params_.Q, params_.device_id);
            });
        if (!kernel)
        {
            LOG_ERROR("[AttentionComputeStage::getOrCreateKernel] Failed to create attention kernel for "
                      << params_.Q->dtype_name());
            return nullptr;
        }

        LOG_DEBUG("[AttentionComputeStage::getOrCreateKernel] Created and cached attention kernel for "
                  << params_.Q->dtype_name() << " on " << params_.device_id.to_string());

        return kernel;
    }

    IWorkspaceConsumer *AttentionComputeStage::getKernelAsWorkspaceConsumer()
    {
        auto *kernel = getOrCreateKernel();
        if (!kernel)
        {
            return nullptr;
        }
        return dynamic_cast<IWorkspaceConsumer *>(kernel);
    }

    WorkspaceRequirements AttentionComputeStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        auto *self = const_cast<AttentionComputeStage *>(this);
        auto *consumer = self->getKernelAsWorkspaceConsumer();
        if (!consumer)
        {
            LOG_TRACE("[AttentionComputeStage] No attention kernel available for workspace requirements");
            return WorkspaceRequirements{};
        }

        // The generic workspace allocator only has model-level sizing hints.
        // Qwen3.5 MoE uses local/tensor-parallel attention dimensions that can
        // differ from those hints.  Size attention scratch from the stage's
        // actual runtime shape so split-decode partial buffers cannot be
        // undersized and overlap PARTIAL_M/PARTIAL_L.
        const int workspace_batch = std::max({1, m, params_.batch_size});
        const int workspace_heads = std::max(n, params_.n_heads);
        const int workspace_head_dim = std::max(k, params_.head_dim);

        auto reqs = consumer->getWorkspaceRequirements(
            workspace_batch,
            workspace_heads,
            workspace_head_dim);

        const int workspace_kv_heads = std::max(1, params_.n_kv_heads);
        const size_t kv_convert_bytes = static_cast<size_t>(workspace_batch) *
                                        4096ULL *
                                        static_cast<size_t>(workspace_kv_heads) *
                                        static_cast<size_t>(workspace_head_dim) *
                                        sizeof(float);
        for (auto &buffer : reqs.buffers)
        {
            if (buffer.name == "attn_k_tmp_fp32" ||
                buffer.name == "attn_v_tmp_fp32")
            {
                buffer.size_bytes = kv_convert_bytes;
            }
        }

        return reqs;
    }

    bool AttentionComputeStage::execute(IDeviceContext *ctx)
    {
        // Dynamic kv_len: query from KV cache at execution time if available
        // This enables declarative graph construction where the stage runs after
        // KVCacheAppendStage has already appended tokens
        int effective_kv_len = params_.kv_len;
        if (params_.kv_cache && params_.layer_idx >= 0)
        {
            effective_kv_len = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);
            if (effective_kv_len == 0)
            {
                effective_kv_len = params_.seq_len; // Prefill case
            }
            LOG_TRACE("[AttentionComputeStage] Dynamic kv_len from cache: " << effective_kv_len
                                                                            << " (static was: " << params_.kv_len << ")");
        }

        // Read K/V from cache at execution time when requested.
        // This allows GPU prefill to use the FP16 tensors in the KV cache
        // (populated by KVCacheAppendStage) instead of the Q8_1 projection
        // buffers, eliminating the Q8_1→FP32→FP16 triple conversion.
        ITensor *effective_K = params_.K;
        ITensor *effective_V = params_.V;
        if (params_.kv_cache && params_.layer_idx >= 0)
        {
            // Always override K/V from KV cache when:
            // 1. read_kv_from_cache is set (GPU optimization for type conversion), OR
            // 2. We're in decode mode (effective_kv_len > seq_len) - the graph may have
            //    been built during prefill (cached_tokens=0) with K/V wired to activation
            //    scratch buffers. During decode, those buffers only hold the current token's
            //    projection. The full KV history lives in the cache (populated by
            //    KVCacheAppendStage which runs before this stage).
            // 3. apply_rope_to_k is set (rope_on_read mode) - K is stored pre-RoPE
            //    in the cache, so we must read through get_kv_converted() which
            //    fuses RoPE into the read path. This applies to BOTH prefill and decode.
            if (params_.read_kv_from_cache || effective_kv_len > params_.seq_len || params_.apply_rope_to_k)
            {
                ITensor *cache_k = nullptr;
                ITensor *cache_v = nullptr;

                // For CPU, use get_kv_converted<FP32>() which handles:
                // - Incremental dequant for all cache precisions (FP16, BF16, Q8_1, TQ4, split TQ)
                // - Optional fused RoPE-on-read (when apply_rope_to_k is set)
                // - Lazy shadow buffer management inside the cache
                // This covers both decode (full KV history) and prefill with rope_on_read
                // (K stored pre-RoPE, needs RoPE applied during read).
                const bool is_cpu_path = (gpuStream() == nullptr);
                if (is_cpu_path && (effective_kv_len > params_.seq_len || params_.apply_rope_to_k))
                {
                    // =====================================================================
                    // Fused TQ attention path: pass raw TQ8/TQ4 tensors directly to the
                    // attention kernel, eliminating FP32 shadow buffers entirely.
                    //
                    // Conditions for fused path:
                    //   - Decode mode (effective_kv_len > seq_len)
                    //   - TQ cache format (TQ8-K / TQ4-V)
                    //   - NOT rope_on_read (RoPE already applied at write time)
                    //   - TurboQuantContext available (for rotation matrices)
                    //
                    // When active, the kernel exploits rotation orthogonality:
                    //   dot(Q, dequant(K)) = (norm/D) · dot(Π·Q, centroids(K))
                    // reducing per-position cost from O(D²) to O(D).
                    // =====================================================================
                    const bool is_tq_decode = (effective_kv_len > params_.seq_len &&
                                               !params_.apply_rope_to_k &&
                                               params_.turboquant_ctx);
                    if (is_tq_decode)
                    {
                        ITensor *raw_k = params_.kv_cache->get_k(params_.layer_idx, 0);
                        ITensor *raw_v = params_.kv_cache->get_v(params_.layer_idx, 0);

                        if (raw_k && raw_v &&
                            raw_k->native_type() == TensorType::TQ8 &&
                            raw_v->native_type() == TensorType::TQ4)
                        {
                            // Set per-layer TQ context on tensors (needed by kernel for rotation)
                            const auto &layer_ctx = params_.turboquant_ctx->for_layer(params_.layer_idx);
                            auto *k_tq8 = dynamic_cast<TQ8Tensor *>(raw_k);
                            auto *v_tq4 = dynamic_cast<TQ4Tensor *>(raw_v);
                            if (k_tq8 && v_tq4)
                            {
                                k_tq8->set_turboquant_context(&layer_ctx);
                                v_tq4->set_turboquant_context(&layer_ctx);
                                effective_K = raw_k;
                                effective_V = raw_v;
                                LOG_TRACE("[AttentionComputeStage] Fused TQ path: passing raw TQ8/TQ4 tensors for layer "
                                          << params_.layer_idx << " kv_len=" << effective_kv_len);
                            }
                        }
                    }

                    // =====================================================================
                    // Fused Q16_1/Q8_1 attention path: pass raw quantized tensors directly
                    // to the attention kernel, eliminating FP32 shadow buffers.
                    //
                    // Q16_1: VNNI int16 QK dot product (VPDPWSSD) + int16 V accumulation
                    // Q8_1: int8→float inline dequant in attention's inner loop
                    //
                    // Conditions: decode mode, NOT rope_on_read, matching cache format
                    // =====================================================================
                    if (effective_K == params_.K && // not overridden by fused TQ path
                        effective_kv_len > params_.seq_len &&
                        !params_.apply_rope_to_k)
                    {
                        const auto kp = params_.kv_cache->k_precision();
                        const auto vp = params_.kv_cache->v_precision();
                        if ((kp == ActivationPrecision::Q16_1 && vp == ActivationPrecision::Q16_1) ||
                            (kp == ActivationPrecision::Q8_1 && vp == ActivationPrecision::Q8_1))
                        {
                            ITensor *raw_k = params_.kv_cache->get_k(params_.layer_idx, 0);
                            ITensor *raw_v = params_.kv_cache->get_v(params_.layer_idx, 0);
                            if (raw_k && raw_v)
                            {
                                effective_K = raw_k;
                                effective_V = raw_v;
                                LOG_TRACE("[AttentionComputeStage] Fused " << activationPrecisionToString(kp)
                                                                           << " path: passing raw tensors for layer "
                                                                           << params_.layer_idx << " kv_len=" << effective_kv_len);
                            }
                        }
                    }

                    // Fallback: dequant via get_kv_converted for unsupported formats or rope_on_read
                    if (effective_K == params_.K) // not overridden by any fused path above
                    {
                        IKVCache::KVReadParams read_params;
                        if (params_.apply_rope_to_k)
                        {
                            read_params.rope_theta = params_.rope_theta;
                            read_params.position_start = 0; // Cache rows are stored in position order
                            read_params.rope_dim = static_cast<int>(
                                params_.partial_rotary_factor * params_.head_dim);
                        }
                        read_params.n_kv_heads = params_.n_kv_heads;
                        read_params.head_dim = params_.head_dim;
                        read_params.turboquant_ctx = params_.turboquant_ctx;
                        read_params.gpu_stream = gpuStream();

                        int kv_len_out = 0;
                        if (params_.kv_cache->get_kv_converted(
                                params_.layer_idx, 0,
                                ActivationPrecision::FP32,
                                &cache_k, &cache_v, &kv_len_out,
                                &read_params))
                        {
                            effective_K = cache_k;
                            effective_V = cache_v;
                            LOG_TRACE("[AttentionComputeStage] Using cache get_kv<FP32> ("
                                      << cache_k->dtype_name() << ") for layer " << params_.layer_idx
                                      << " kv_len=" << kv_len_out);
                        }
                        else
                        {
                            LOG_WARN("[AttentionComputeStage] get_kv_converted failed, falling back to raw cache");
                            cache_k = params_.kv_cache->get_k(params_.layer_idx, 0);
                            cache_v = params_.kv_cache->get_v(params_.layer_idx, 0);
                            if (cache_k && cache_v)
                            {
                                effective_K = cache_k;
                                effective_V = cache_v;
                            }
                        }
                    } // end fallback get_kv_converted
                }
                else
                {
                    // GPU path: use get_kv_converted() for TQ dequant + fused RoPE,
                    // otherwise use raw cache tensors directly.

                    // =====================================================================
                    // Fused Q8_1 GPU attention path: pass raw Q8_1 tensors to the CUDA
                    // kernel which does inline int8→float dequant in the attention loop.
                    // This eliminates the Q8_1→FP16/FP32 workspace conversion entirely.
                    //
                    // Conditions: decode mode, NOT rope_on_read, Q8_1 cache format
                    // =====================================================================
                    const bool is_q8_gpu_decode = (effective_kv_len > params_.seq_len &&
                                                   !params_.apply_rope_to_k &&
                                                   params_.seq_len == 1);
                    if (is_q8_gpu_decode)
                    {
                        const auto kp = params_.kv_cache->k_precision();
                        const auto vp = params_.kv_cache->v_precision();
                        if (kp == ActivationPrecision::Q8_1 && vp == ActivationPrecision::Q8_1)
                        {
                            ITensor *raw_k = params_.kv_cache->get_k(params_.layer_idx, 0);
                            ITensor *raw_v = params_.kv_cache->get_v(params_.layer_idx, 0);
                            if (raw_k && raw_v)
                            {
                                effective_K = raw_k;
                                effective_V = raw_v;
                                LOG_TRACE("[AttentionComputeStage] GPU fused Q8_1 path: passing raw Q8_1 tensors for layer "
                                          << params_.layer_idx << " kv_len=" << effective_kv_len);
                            }
                        }
                    }

                    if (effective_K == params_.K &&
                        (params_.apply_rope_to_k || effective_kv_len > params_.seq_len))
                    {
                        LOG_DEBUG("[AttentionComputeStage] GPU KV CONVERTED PATH layer=" << params_.layer_idx
                                                                                         << " apply_rope=" << params_.apply_rope_to_k
                                                                                         << " rope_theta=" << params_.rope_theta
                                                                                         << " eff_kv_len=" << effective_kv_len
                                                                                         << " seq_len=" << params_.seq_len);
                        IKVCache::KVReadParams read_params;
                        if (params_.apply_rope_to_k)
                        {
                            read_params.rope_theta = params_.rope_theta;
                            read_params.position_start = 0;
                            read_params.rope_dim = static_cast<int>(
                                params_.partial_rotary_factor * params_.head_dim);
                        }
                        read_params.n_kv_heads = params_.n_kv_heads;
                        read_params.head_dim = params_.head_dim;
                        read_params.turboquant_ctx = params_.turboquant_ctx;
                        read_params.gpu_stream = gpuStream();

                        int kv_len_out = 0;
                        if (params_.kv_cache->get_kv_converted(
                                params_.layer_idx, 0,
                                ActivationPrecision::FP16,
                                &cache_k, &cache_v, &kv_len_out,
                                &read_params))
                        {
                            effective_K = cache_k;
                            effective_V = cache_v;
                            LOG_TRACE("[AttentionComputeStage] GPU: Using cache get_kv_converted<FP16> ("
                                      << cache_k->dtype_name() << ") for layer " << params_.layer_idx
                                      << " kv_len=" << kv_len_out);
                        }
                        else
                        {
                            LOG_WARN("[AttentionComputeStage] GPU: get_kv_converted failed, falling back to raw cache");
                            cache_k = params_.kv_cache->get_k(params_.layer_idx, 0);
                            cache_v = params_.kv_cache->get_v(params_.layer_idx, 0);
                            if (cache_k && cache_v)
                            {
                                effective_K = cache_k;
                                effective_V = cache_v;
                            }
                        }
                    }
                    else
                    {
                        cache_k = params_.kv_cache->get_k(params_.layer_idx, 0);
                        cache_v = params_.kv_cache->get_v(params_.layer_idx, 0);
                        if (cache_k && cache_v)
                        {
                            effective_K = cache_k;
                            effective_V = cache_v;
                            LOG_TRACE("[AttentionComputeStage] Using cache K/V ("
                                      << cache_k->dtype_name() << ") for layer " << params_.layer_idx);
                        }
                        else
                        {
                            LOG_TRACE("[AttentionComputeStage] Cache K/V not available yet, using wired tensors");
                        }
                    }
                }
            }
        }

        // Detect attention mode if auto-detection enabled
        AttentionMode mode = params_.attention_mode;
        if (params_.auto_detect_mode)
        {
            mode = detect_attention_mode(params_.batch_size, params_.seq_len, effective_kv_len);
        }

        const bool is_decode_mode = (mode == AttentionMode::DECODE ||
                                     (params_.seq_len < effective_kv_len && params_.batch_size == 1));

        LOG_DEBUG("[AttentionComputeStage] Execute: batch=" << params_.batch_size
                                                            << " seq_len=" << params_.seq_len
                                                            << " kv_len=" << effective_kv_len
                                                            << " n_heads=" << params_.n_heads
                                                            << " n_kv_heads=" << params_.n_kv_heads
                                                            << " head_dim=" << params_.head_dim
                                                            << " position_offset=" << params_.position_offset
                                                            << " mode=" << attention_mode_name(mode)
                                                            << " Q_type=" << (params_.Q ? params_.Q->dtype_name() : "null")
                                                            << " K_type=" << (effective_K ? effective_K->dtype_name() : "null")
                                                            << " V_type=" << (effective_V ? effective_V->dtype_name() : "null")
                                                            << " output=" << (void *)params_.output);

        // Validate inputs
        if (!ensureRequiredPointers("AttentionComputeStage", {
                                                                 {"Q", params_.Q},
                                                                 {"K", effective_K},
                                                                 {"V", effective_V},
                                                                 {"output", params_.output},
                                                             }))
        {
            return false;
        }

        if (params_.seq_len <= 0 || effective_kv_len <= 0 ||
            params_.n_heads <= 0 || params_.n_kv_heads <= 0 || params_.head_dim <= 0)
        {
            LOG_ERROR("[AttentionComputeStage] Invalid dimensions");
            return false;
        }

        if (params_.n_heads % params_.n_kv_heads != 0)
        {
            LOG_ERROR("[AttentionComputeStage] n_heads (" << params_.n_heads
                                                          << ") must be divisible by n_kv_heads (" << params_.n_kv_heads << ")");
            return false;
        }

        // Use cached kernel (enables workspace binding for GPU kernels)
        auto *kernel = getOrCreateKernel();
        if (!kernel)
        {
            LOG_ERROR("[AttentionComputeStage] Failed to get attention kernel");
            return false;
        }
        bindStageStream(kernel);

        // Get device index using proper ordinal for GPU devices (0-based), not legacy index
        int device_idx = params_.device_id.toKernelDeviceIndex();

        // Build proper causal mask for decode mode
        // Key insight: For single-token decode (seq_len=1), the query at position
        // (kv_len-1) can attend to ALL positions, so the mask is all-zeros.
        // Skip mask construction entirely and pass nullptr + causal=false.
        // Only build an explicit mask for multi-token decode (seq_len > 1 but < kv_len).
        std::unique_ptr<FP32Tensor> decode_mask;
        ITensor *mask_to_use = params_.workspace_mask;

        if (params_.causal && is_decode_mode)
        {
            if (gpuStream() == nullptr && params_.seq_len > 1)
            {
                // Multi-token decode on CPU: some tokens need causal masking
                const int base_pos = (params_.position_offset > 0)
                                         ? params_.position_offset
                                         : (effective_kv_len - params_.seq_len);

                decode_mask = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{static_cast<size_t>(params_.seq_len * effective_kv_len)});
                float *mask_data = decode_mask->mutable_data();

                for (int q = 0; q < params_.seq_len; ++q)
                {
                    const int q_pos = base_pos + q;
                    for (int k = 0; k < effective_kv_len; ++k)
                    {
                        mask_data[q * effective_kv_len + k] = (k <= q_pos)
                                                                  ? 0.0f
                                                                  : -std::numeric_limits<float>::infinity();
                    }
                }

                mask_to_use = decode_mask.get();
            }
            else
            {
                // Single-token decode (seq_len=1) or GPU stream: mask is all-zeros,
                // equivalent to nullptr + causal=false (attend to all positions).
                mask_to_use = nullptr;
            }
        }

        // Dispatch to kernel's compute method
        // IMPORTANT: For decode with explicit mask, we pass causal=false to avoid
        // double-masking (kernel would apply "n > m" on top of our mask)
        const bool kernel_causal = params_.causal && !is_decode_mode;

        // Device-agnostic unified path using compute_tensor()
        // The kernel factory creates the appropriate kernel (CPU or GPU) based on dev_type,
        // and compute_tensor() handles type dispatch internally.
        // Since compute_tensor() now takes ITensor*, we can pass Q/K/V directly without casting.
        // This allows GPU tensor wrappers (like GpuTensorView from CUDA KV cache) to work.

        if (!ensureRequiredPointers("AttentionComputeStage", {
                                                                 {"Q", params_.Q},
                                                                 {"K", params_.K},
                                                                 {"V", params_.V},
                                                                 {"output", params_.output},
                                                             }))
        {
            return false;
        }

        // Device coherence is now handled automatically by DeviceGraphExecutor at stage boundaries
        // based on the stage's coherencePolicy() (FULL by default)

        LOG_DEBUG("[AttentionComputeStage] Executing kernel: Q_type=" << params_.Q->dtype_name()
                                                                      << " device=" << params_.device_id.to_string()
                                                                      << " device_idx=" << device_idx);

        // =====================================================================
        // DEBUG: Dump effective K/V to binary files for Python analysis
        // Enable with LLAMINAR_DUMP_EFFECTIVE_KV=1
        // Dumps layer 0 data (or all layers with LLAMINAR_DUMP_EFFECTIVE_KV_ALL=1)
        // =====================================================================
        {
            static const bool dump_enabled = (std::getenv("LLAMINAR_DUMP_EFFECTIVE_KV") &&
                                              std::atoi(std::getenv("LLAMINAR_DUMP_EFFECTIVE_KV")) != 0);
            static const bool dump_all = (std::getenv("LLAMINAR_DUMP_EFFECTIVE_KV_ALL") &&
                                          std::atoi(std::getenv("LLAMINAR_DUMP_EFFECTIVE_KV_ALL")) != 0);

            if (dump_enabled && (dump_all || params_.layer_idx == 0) && is_decode_mode)
            {
                static int dump_iteration = 0;
                const std::string dump_dir = "/tmp/effective_kv_dump/layer" +
                                             std::to_string(params_.layer_idx) +
                                             "_iter" + std::to_string(dump_iteration);
                std::filesystem::create_directories(dump_dir);

                // Write metadata
                {
                    std::ofstream meta(dump_dir + "/meta.txt");
                    meta << "layer=" << params_.layer_idx << "\n"
                         << "iteration=" << dump_iteration << "\n"
                         << "seq_len=" << params_.seq_len << "\n"
                         << "kv_len=" << effective_kv_len << "\n"
                         << "n_heads=" << params_.n_heads << "\n"
                         << "n_kv_heads=" << params_.n_kv_heads << "\n"
                         << "head_dim=" << params_.head_dim << "\n"
                         << "batch_size=" << params_.batch_size << "\n"
                         << "mode=" << attention_mode_name(mode) << "\n"
                         << "Q_type=" << (params_.Q ? params_.Q->dtype_name() : "null") << "\n"
                         << "K_type=" << (effective_K ? effective_K->dtype_name() : "null") << "\n"
                         << "V_type=" << (effective_V ? effective_V->dtype_name() : "null") << "\n"
                         << "K_is_converted=" << (effective_K && dynamic_cast<FP32Tensor *>(effective_K) ? 1 : 0) << "\n"
                         << "V_is_converted=" << (effective_V && dynamic_cast<FP32Tensor *>(effective_V) ? 1 : 0) << "\n"
                         << "K_ptr=" << (void *)effective_K << "\n"
                         << "V_ptr=" << (void *)effective_V << "\n";
                }

                // Dump Q (FP32)
                if (auto *q_fp32 = dynamic_cast<FP32Tensor *>(params_.Q))
                {
                    const size_t q_elems = static_cast<size_t>(params_.seq_len) *
                                           params_.n_heads * params_.head_dim;
                    std::ofstream f(dump_dir + "/Q.bin", std::ios::binary);
                    f.write(reinterpret_cast<const char *>(q_fp32->data()),
                            q_elems * sizeof(float));
                }

                // Dump effective K (should be FP32 after dequant)
                if (auto *k_fp32 = dynamic_cast<FP32Tensor *>(effective_K))
                {
                    const size_t k_elems = static_cast<size_t>(effective_kv_len) *
                                           params_.n_kv_heads * params_.head_dim;
                    std::ofstream f(dump_dir + "/K_effective.bin", std::ios::binary);
                    f.write(reinterpret_cast<const char *>(k_fp32->data()),
                            k_elems * sizeof(float));
                }

                // Dump effective V (should be FP32 after dequant)
                if (auto *v_fp32 = dynamic_cast<FP32Tensor *>(effective_V))
                {
                    const size_t v_elems = static_cast<size_t>(effective_kv_len) *
                                           params_.n_kv_heads * params_.head_dim;
                    std::ofstream f(dump_dir + "/V_effective.bin", std::ios::binary);
                    f.write(reinterpret_cast<const char *>(v_fp32->data()),
                            v_elems * sizeof(float));
                }

                // Dump raw TQ4 K cache if available
                if (params_.kv_cache && params_.layer_idx >= 0)
                {
                    ITensor *cache_k = params_.kv_cache->get_k(params_.layer_idx, 0);
                    if (auto *k_tq4 = dynamic_cast<TQ4Tensor *>(cache_k))
                    {
                        const size_t raw_bytes = static_cast<size_t>(effective_kv_len) *
                                                 k_tq4->blocks_per_row() * k_tq4->block_bytes();
                        std::ofstream f(dump_dir + "/K_cache_tq4.bin", std::ios::binary);
                        f.write(reinterpret_cast<const char *>(k_tq4->typed_data()),
                                raw_bytes);
                        std::ofstream m(dump_dir + "/K_cache_meta.txt");
                        m << "blocks_per_row=" << k_tq4->blocks_per_row() << "\n"
                          << "block_bytes=" << k_tq4->block_bytes() << "\n"
                          << "head_dim=" << k_tq4->head_dim() << "\n"
                          << "rows=" << effective_kv_len << "\n";
                    }
                }

                LOG_DEBUG("[AttentionComputeStage] Dumped effective K/V to " << dump_dir);
                if (params_.layer_idx == 0)
                    dump_iteration++;
            }
        }

        // =====================================================================
        // Fused TQ GPU decode: rotation trick + centroid attention
        // Bypasses normal compute_tensor() since TQ requires ring buffer
        // metadata, rotation matrices, and codebook access.
        // NOTE: Currently slower than dequant+flash path. Enable with
        // LLAMINAR_ENABLE_FUSED_TQ_ATTN=1 for testing.
        // =====================================================================
#if defined(HAVE_CUDA)
        if (is_decode_mode && params_.device_id.is_gpu() && params_.kv_cache && std::getenv("LLAMINAR_ENABLE_FUSED_TQ_ATTN"))
        {
            const auto kp = params_.kv_cache->k_precision();
            const auto vp = params_.kv_cache->v_precision();
            if (kp == ActivationPrecision::TQ8 && vp == ActivationPrecision::TQ4)
            {
                auto *tq_cache = dynamic_cast<CUDARingKVCacheTQ *>(params_.kv_cache);
                auto *cuda_kernel = dynamic_cast<cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> *>(kernel);
                if (tq_cache && cuda_kernel)
                {
                    const auto &rots = tq_cache->rotations();
                    const size_t layer_offset = static_cast<size_t>(params_.layer_idx) *
                                                rots.n_kv_heads * rots.head_dim * rots.head_dim;
                    const float *R = rots.d_rotations + layer_offset;
                    const float *Rt = rots.d_rotations_t + layer_offset;

                    bool tq_success = cuda_kernel->compute_tensor_tq_decode(
                        params_.Q, params_.output,
                        tq_cache->raw_k_cache(params_.layer_idx, 0),
                        tq_cache->raw_v_cache(params_.layer_idx, 0),
                        R, Rt,
                        params_.batch_size,
                        effective_kv_len,
                        params_.n_heads,
                        params_.n_kv_heads,
                        params_.head_dim,
                        tq_cache->max_seq_len(),
                        tq_cache->ring_tail(params_.layer_idx, 0),
                        static_cast<int>(tq_cache->k_block_size()),
                        static_cast<int>(tq_cache->v_block_size()));

                    if (!tq_success)
                    {
                        LOG_ERROR("[AttentionComputeStage] Fused TQ GPU decode kernel failed");
                        return false;
                    }
                    LOG_TRACE("[AttentionComputeStage] Fused TQ GPU decode for layer " << params_.layer_idx);
                    return true;
                }
            }
        }
#endif

        // =================================================================
        // KV rotation: rotate Q before attention, inverse-rotate output after.
        // When K/V were rotated by ActivationRotation before Q16_1 quantization
        // in KVCacheAppendStage, Q must be rotated by the same matrix so that
        // Q@R @ (K@R)^T = Q@K^T (rotation cancels in the score). The output
        // (sum of alpha_i * V_i@R) must then be inverse-rotated to recover
        // the correct unrotated attention result.
        //
        // During PREFILL, effective_K/V are the raw projection outputs (unrotated),
        // since the cache override is only triggered for decode (kv_len > seq_len).
        // In this case we must also rotate K/V to match the rotated Q.
        // During DECODE, effective_K/V come from the cache (already rotated),
        // so only Q rotation is needed.
        // =================================================================
        const auto *kv_rot = params_.kv_rotation;
        const int q_dim = params_.n_heads * params_.head_dim;
        const int kv_dim = params_.n_kv_heads * params_.head_dim;
        const bool projections_need_rotation = kv_rot &&
                                               (effective_K == params_.K) &&
                                               (effective_V == params_.V);

        if (kv_rot)
        {
            float *q_fp32 = params_.Q->mutable_data();
            if (q_fp32)
            {
                kv_rot->rotate_rows_inplace(q_fp32, params_.seq_len, q_dim);
            }

            // Prefill path: K/V are original projections, not from cache.
            // Rotate them so Q_rot @ K_rot^T = Q @ K^T and V_rot gives R*context.
            if (projections_need_rotation)
            {
                float *k_fp32 = effective_K->mutable_data();
                float *v_fp32 = effective_V->mutable_data();
                if (k_fp32)
                    kv_rot->rotate_rows_inplace(k_fp32, params_.seq_len, kv_dim);
                if (v_fp32)
                    kv_rot->rotate_rows_inplace(v_fp32, params_.seq_len, kv_dim);
            }
        }

        bool success = kernel->compute_tensor(
            params_.Q, effective_K, effective_V, params_.output,
            params_.batch_size,
            params_.seq_len,
            effective_kv_len,
            params_.n_heads,
            params_.n_kv_heads,
            params_.head_dim,
            kernel_causal, // Pass false for decode (we built the mask explicitly)
            params_.window_size,
            params_.workspace_scores,
            mask_to_use, // Use our decode mask if we built one
            params_.mpi_ctx,
            device_idx,
            params_.head_start,
            -1, // local_n_heads (n_heads is already local)
            -1, // local_n_kv_heads (n_kv_heads is already local)
            params_.gqa_n_rep);

        if (kv_rot)
        {
            // Inverse-rotate attention output to undo the V rotation.
            // output = sum(alpha_i * V_i@R) → R^T * output = sum(alpha_i * V_i)
            float *out_fp32 = params_.output->mutable_data();
            if (out_fp32)
            {
                kv_rot->inverse_rotate_rows_inplace(out_fp32,
                                                    params_.seq_len, q_dim);
            }

            // Q, K, V do NOT need inverse-rotation:
            // - Q is overwritten by next layer's QKV projection
            // - K/V during prefill: KVCacheAppend already ran (depends on rope/proj),
            //   and no downstream stage reads these buffers after attention
        }

        return true;
    }

    size_t AttentionComputeStage::estimatedFlops() const
    {
        // Attention FLOPs:
        // Q @ K^T: 2 * batch * n_heads * seq_len * kv_len * head_dim
        // softmax: ~4 * batch * n_heads * seq_len * kv_len
        // scores @ V: 2 * batch * n_heads * seq_len * kv_len * head_dim
        const size_t qk_flops = 2ULL * params_.batch_size * params_.n_heads *
                                params_.seq_len * params_.kv_len * params_.head_dim;
        const size_t softmax_flops = 4ULL * params_.batch_size * params_.n_heads *
                                     params_.seq_len * params_.kv_len;
        const size_t sv_flops = qk_flops;
        return qk_flops + softmax_flops + sv_flops;
    }

    size_t AttentionComputeStage::estimatedMemoryBytes() const
    {
        // Workspace for attention scores: n_heads * seq_len * kv_len
        return static_cast<size_t>(params_.n_heads) * params_.seq_len * params_.kv_len * sizeof(float);
    }

    bool AttentionComputeStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
#if defined(HAVE_CUDA)
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#if defined(HAVE_ROCM)
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    StageDumpInfo AttentionComputeStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Input: Q, K, V tensors
        // Q shape: [batch_size * seq_len, n_heads * head_dim]
        // K/V shape: [batch_size * kv_len, n_kv_heads * head_dim]
        const size_t total_q_tokens = static_cast<size_t>(params_.batch_size * params_.seq_len);
        const size_t total_kv_tokens = static_cast<size_t>(params_.batch_size * params_.kv_len);

        if (params_.Q)
        {
            info.addInput("Q", params_.Q, total_q_tokens, params_.n_heads * params_.head_dim);
        }
        if (params_.K)
        {
            info.addInput("K", params_.K, total_kv_tokens, params_.n_kv_heads * params_.head_dim);
        }
        if (params_.V)
        {
            info.addInput("V", params_.V, total_kv_tokens, params_.n_kv_heads * params_.head_dim);
        }

        // Output: attention context
        // Output shape: [batch_size * seq_len, n_heads * head_dim]
        if (params_.output)
        {
            LOG_DEBUG("[AttentionComputeStage::getDumpInfo] output=" << (void *)params_.output
                                                                     << " type=" << params_.output->dtype_name()
                                                                     << " batch_size=" << params_.batch_size
                                                                     << " seq_len=" << params_.seq_len
                                                                     << " total_tokens=" << total_q_tokens
                                                                     << " n_heads*head_dim=" << (params_.n_heads * params_.head_dim));
            // Use ITensor* overload to enable coherence tracking
            info.addOutput("output", params_.output, total_q_tokens, params_.n_heads * params_.head_dim);
        }
        else
        {
            LOG_DEBUG("[AttentionComputeStage::getDumpInfo] output is NULL");
        }

        // Scalars capture all necessary info for debugging
        info.addScalarInt("batch_size", params_.batch_size);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("kv_len", params_.kv_len);
        info.addScalarInt("n_heads", params_.n_heads);
        info.addScalarInt("n_kv_heads", params_.n_kv_heads);
        info.addScalarInt("head_dim", params_.head_dim);
        info.addScalarBool("causal", params_.causal);
        info.addScalarInt("window_size", params_.window_size);
        info.addScalarInt("device_id", params_.device_id.toKernelDeviceIndex());

        // Add attention mode info (as int - PREFILL=0, DECODE=1, BATCHED_DECODE=2, CHUNKED_PREFILL=3)
        AttentionMode mode = params_.auto_detect_mode
                                 ? detect_attention_mode(params_.batch_size, params_.seq_len, params_.kv_len)
                                 : params_.attention_mode;
        info.addScalarInt("attention_mode", static_cast<int>(mode));
        info.addScalarBool("auto_detect_mode", params_.auto_detect_mode);

        return info;
    }

    StageBufferRequirements AttentionComputeStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // Input: Q (query)
        if (params_.Q)
        {
            const size_t q_rows = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t q_cols = static_cast<size_t>(params_.n_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.Q->native_type());
            reqs.addInput("Q", {q_rows, q_cols}, buf_type);
        }

        // Input: K (key - may have different kv_len than Q's seq_len)
        if (params_.K)
        {
            const size_t k_rows = static_cast<size_t>(params_.batch_size * params_.kv_len);
            const size_t k_cols = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.K->native_type());
            reqs.addInput("K", {k_rows, k_cols}, buf_type);
        }

        // Input: V (value)
        if (params_.V)
        {
            const size_t v_rows = static_cast<size_t>(params_.batch_size * params_.kv_len);
            const size_t v_cols = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.V->native_type());
            reqs.addInput("V", {v_rows, v_cols}, buf_type);
        }

        // Output: attention output
        if (params_.output)
        {
            const size_t out_rows = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t out_cols = static_cast<size_t>(params_.n_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.output->native_type());
            reqs.addOutput("output", {out_rows, out_cols}, buf_type);
        }

        // Scratch: workspace buffers (if pre-allocated)
        if (params_.workspace_scores)
        {
            reqs.addScratch("workspace_scores", params_.workspace_scores->shape(),
                            toBufferTensorType(params_.workspace_scores->native_type()));
        }
        if (params_.workspace_context)
        {
            reqs.addScratch("workspace_context", params_.workspace_context->shape(),
                            toBufferTensorType(params_.workspace_context->native_type()));
        }

        return reqs;
    }

    StageBufferContract AttentionComputeStage::bufferContract() const
    {
        if (!params_.q_buffer_id || !params_.output_buffer_id)
            return {};

        auto contract = StageBufferContract::build()
                            .addInput(*params_.q_buffer_id)
                            .addOutput(*params_.output_buffer_id);

        if (params_.workspace_scores_buffer_id)
            contract.addInOut(*params_.workspace_scores_buffer_id);
        if (params_.workspace_context_buffer_id)
            contract.addInOut(*params_.workspace_context_buffer_id);

        return contract;
    }

} // namespace llaminar2
