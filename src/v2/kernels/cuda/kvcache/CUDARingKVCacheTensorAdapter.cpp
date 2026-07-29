/**
 * @file CUDARingKVCacheTensorAdapter.cpp
 * @brief ITensor adapter for CUDA KV Cache
 *
 * This file is compiled by the regular C++ compiler (not nvcc) so it can
 * include heavy headers like CPUTensors.h without MPI header issues.
 *
 * Provides:
 * - ICUDARingKVCache::append(ITensor*) implementation
 * - CUDARingKVCache<>::get_k() and get_v() implementations using GpuTensorView
 */

#include "CUDARingKVCache.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../tensors/GpuTensorView.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../backends/DeviceId.h"
#include "../../../utils/Logger.h"
#include "../../../utils/KVCacheProfiler.h"

#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace llaminar2
{

    extern "C" bool cuda_convert_tensor_to_fp16(
        const void *d_src,
        TensorType src_type,
        uint16_t *d_dst,
        int count,
        cudaStream_t stream);

    extern "C" bool cuda_convert_tensor_to_q8_1(
        const void *d_src,
        TensorType src_type,
        Q8_1Block *d_dst,
        int rows,
        int cols,
        cudaStream_t stream);

    // =========================================================================
    // ICUDARingKVCache destructor + conversion scratch buffer management
    // =========================================================================

    ICUDARingKVCache::~ICUDARingKVCache()
    {
        freeConvScratch();
    }

    bool ICUDARingKVCache::ensureConvScratch(size_t bytes)
    {
        if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(this))
        {
            DeviceWorkspaceManager *workspace = consumer->getWorkspace();
            if (workspace && workspace->isAllocated())
            {
                void *workspace_k = workspace->getBuffer(KVCacheWorkspaceBuffers::CONV_SCRATCH_K);
                void *workspace_v = workspace->getBuffer(KVCacheWorkspaceBuffers::CONV_SCRATCH_V);
                const size_t workspace_k_size = workspace->getBufferSize(KVCacheWorkspaceBuffers::CONV_SCRATCH_K);
                const size_t workspace_v_size = workspace->getBufferSize(KVCacheWorkspaceBuffers::CONV_SCRATCH_V);
                if (!workspace_k || !workspace_v || workspace_k_size < bytes || workspace_v_size < bytes)
                {
                    LOG_ERROR("[ICUDARingKVCache] Bound workspace is missing conversion scratch: required="
                              << bytes << " bytes, K_available=" << workspace_k_size
                              << " V_available=" << workspace_v_size);
                    return false;
                }

                conv_scratch_k_ = workspace_k;
                conv_scratch_v_ = workspace_v;
                conv_scratch_capacity_ = std::min(workspace_k_size, workspace_v_size);
                conv_scratch_workspace_backed_ = true;
                return true;
            }
        }

        LOG_ERROR("[ICUDARingKVCache] Conversion requires pre-bound graph workspace: required="
                  << bytes << " bytes per K/V buffer");
        return false;
    }

    void ICUDARingKVCache::freeConvScratch()
    {
        // Conversion scratch is always workspace-owned.
        conv_scratch_k_ = nullptr;
        conv_scratch_v_ = nullptr;
        conv_scratch_capacity_ = 0;
        conv_scratch_workspace_backed_ = false;
    }

    // =========================================================================
    // ICUDARingKVCache::append(ITensor*) implementation
    // =========================================================================
    // NOTE: This is in a separate .cpp file (not .cu) because nvcc has issues
    // with some C++ headers. The ITensor interface is lightweight and doesn't
    // require heavy includes.

    bool ICUDARingKVCache::append(int layer, int seq_idx,
                                  const ITensor *K, const ITensor *V,
                                  int num_tokens)
    {
        (void)layer;
        (void)seq_idx;
        (void)K;
        (void)V;
        (void)num_tokens;
        LOG_ERROR("[ICUDARingKVCache::append(ITensor)] Explicit CUDA stream required; use appendWithStream()");
        return false;
    }

    bool ICUDARingKVCache::appendWithStream(int layer, int seq_idx,
                                            const ITensor *K, const ITensor *V,
                                            int num_tokens, void *gpu_stream)
    {
        if (!K || !V)
        {
            LOG_DEBUG("[ICUDARingKVCache::appendWithStream] Null K or V tensor");
            return false;
        }
        if (!gpu_stream)
        {
            LOG_ERROR("[ICUDARingKVCache::appendWithStream] Null CUDA stream is not allowed");
            return false;
        }

        const auto target = DeviceId::cuda(device_id());

        const auto prepare_input = [&](const ITensor *tensor, const char *label)
        {
            if (const auto *prepared =
                    dynamic_cast<const PreparedGpuTensorView *>(tensor))
            {
                if (!prepared->isPreparedFor(target, gpu_stream))
                {
                    throw std::runtime_error(
                        std::string("[ICUDARingKVCache::appendWithStream] Prepared ") +
                        label + " slice does not match the CUDA consumer device/stream");
                }
                return;
            }

            /*
             * An ordinary tensor must publish its producer event through the
             * canonical transfer owner. Only the stage-created prepared slice
             * above may bypass this join, because it names the exact stream on
             * which its parent was already ordered.
             */
            TransferEngine::prepareDeviceInput(
                const_cast<ITensor *>(tensor), target, gpu_stream);
        };
        prepare_input(K, "K");
        prepare_input(V, "V");

        const void *d_k = K->gpu_data_ptr();
        const void *d_v = V->gpu_data_ptr();

        if (!d_k || !d_v)
        {
            LOG_ERROR("[ICUDARingKVCache::appendWithStream] K or V tensor lacks GPU data after TransferEngine preparation");
            return false;
        }

        const auto stream = static_cast<cudaStream_t>(gpu_stream);

        // NOTE: We use k_precision() for both K and V conversion paths below.
        // For symmetric caches (k_precision() == v_precision(), which is the
        // default), this is correct. For asymmetric caches such as
        // CUDARingKVCacheTQ (TQ8 K + TQ4 V) the TQ path is not handled by
        // these blocks at all - those tensors fall through to the native
        // append() call below. If a future asymmetric FP16/Q8_1 cache is
        // added, the conversion paths must be split into separate K and V
        // gates using k_precision() / v_precision() respectively.
        const ActivationPrecision destination_precision = k_precision();
        const bool floating_cache =
            destination_precision == ActivationPrecision::FP32 ||
            destination_precision == ActivationPrecision::FP16 ||
            destination_precision == ActivationPrecision::BF16;
        const bool source_matches_cache =
            (destination_precision == ActivationPrecision::FP32 &&
             K->native_type() == TensorType::FP32 &&
             V->native_type() == TensorType::FP32) ||
            (destination_precision == ActivationPrecision::FP16 &&
             K->native_type() == TensorType::FP16 &&
             V->native_type() == TensorType::FP16) ||
            (destination_precision == ActivationPrecision::BF16 &&
             K->native_type() == TensorType::BF16 &&
             V->native_type() == TensorType::BF16);

        if (floating_cache && !source_matches_cache)
        {
            const auto &k_shape = K->shape();
            const auto &v_shape = V->shape();
            if (k_shape.size() < 2 || v_shape.size() < 2)
            {
                LOG_ERROR("[ICUDARingKVCache::appendWithStream] Invalid K/V shape for floating cache conversion");
                return false;
            }

            const int kv_dim = static_cast<int>(k_shape[1]);
            const int elements = num_tokens * kv_dim;
            if (elements <= 0)
            {
                return append(layer, seq_idx, d_k, d_v, num_tokens, stream);
            }
            if (K->native_type() != V->native_type())
            {
                LOG_ERROR("[ICUDARingKVCache::appendWithStream] Asymmetric K/V source types are unsupported for fused floating append: K="
                          << static_cast<int>(K->native_type())
                          << " V=" << static_cast<int>(V->native_type()));
                return false;
            }

            const auto append_start = std::chrono::high_resolution_clock::now();
            const bool ok = appendConvertedWithStream(layer, seq_idx, d_k, d_v,
                                                      K->native_type(), num_tokens, stream);
            const auto append_end = std::chrono::high_resolution_clock::now();
            {
                auto to_ns = [](auto d) -> uint64_t
                {
                    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
                };
                const uint64_t append_ns = to_ns(append_end - append_start);
                const uint64_t destination_element_bytes =
                    destination_precision == ActivationPrecision::FP32
                        ? sizeof(float)
                        : sizeof(uint16_t);
                const uint64_t bytes =
                    static_cast<uint64_t>(elements) *
                    destination_element_bytes * 2;
                KVCacheProfiler::record(KVCacheOpType::APPEND, append_ns, static_cast<uint64_t>(num_tokens), bytes);
            }

            return ok;
        }

        if (k_precision() == ActivationPrecision::Q8_1 &&
            (K->native_type() != TensorType::Q8_1 || V->native_type() != TensorType::Q8_1))
        {
            const auto &k_shape = K->shape();
            const auto &v_shape = V->shape();
            if (k_shape.size() < 2 || v_shape.size() < 2)
            {
                LOG_ERROR("[ICUDARingKVCache::appendWithStream] Invalid K/V shape for Q8_1 conversion");
                return false;
            }

            const int kv_dim = static_cast<int>(k_shape[1]);
            const int blocks_per_row = (kv_dim + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
            const size_t block_count = static_cast<size_t>(num_tokens) * static_cast<size_t>(blocks_per_row);
            if (block_count == 0)
            {
                return append(layer, seq_idx, d_k, d_v, num_tokens, stream);
            }

            // --- Profiling: ensure scratch buffers ---
            const auto alloc_start = std::chrono::high_resolution_clock::now();

            const size_t buf_bytes = block_count * sizeof(Q8_1Block);
            if (!ensureConvScratch(buf_bytes))
            {
                LOG_ERROR("[ICUDARingKVCache::appendWithStream] Failed to ensure Q8_1 conversion scratch");
                return false;
            }
            auto *d_k_q8 = static_cast<Q8_1Block *>(conv_scratch_k_);
            auto *d_v_q8 = static_cast<Q8_1Block *>(conv_scratch_v_);

            const auto alloc_end = std::chrono::high_resolution_clock::now();

            // --- Profiling: Q8_1 conversion kernels ---
            const auto conv_start = std::chrono::high_resolution_clock::now();

            const bool k_ok = cuda_convert_tensor_to_q8_1(d_k, K->native_type(), d_k_q8, num_tokens, kv_dim, stream);
            const bool v_ok = cuda_convert_tensor_to_q8_1(d_v, V->native_type(), d_v_q8, num_tokens, kv_dim, stream);
            if (!k_ok || !v_ok)
            {
                LOG_ERROR("[ICUDARingKVCache::appendWithStream] GPU Q8_1 conversion failed");
                return false;
            }

            const auto conv_end = std::chrono::high_resolution_clock::now();

            // --- Profiling: ring buffer append ---
            const auto append_start = std::chrono::high_resolution_clock::now();
            const bool ok = append(layer, seq_idx, d_k_q8, d_v_q8, num_tokens, stream);
            const auto append_end = std::chrono::high_resolution_clock::now();

            // Record profiling breakdown
            {
                auto to_ns = [](auto d) -> uint64_t
                {
                    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
                };
                const uint64_t alloc_ns = to_ns(alloc_end - alloc_start);
                const uint64_t conv_ns = to_ns(conv_end - conv_start);
                const uint64_t append_ns = to_ns(append_end - append_start);
                const uint64_t bytes = block_count * sizeof(Q8_1Block) * 2;
                KVCacheProfiler::record(KVCacheOpType::GPU_ALLOC, alloc_ns);
                KVCacheProfiler::record(KVCacheOpType::CONVERT_TO_Q8_1, conv_ns, static_cast<uint64_t>(num_tokens), bytes);
                KVCacheProfiler::record(KVCacheOpType::APPEND, append_ns, static_cast<uint64_t>(num_tokens), bytes);
            }

            return ok;
        }

        // No conversion needed - profile just the append
        {
            const auto start = std::chrono::high_resolution_clock::now();
            const bool ok = append(layer, seq_idx, d_k, d_v, num_tokens, stream);
            const auto end = std::chrono::high_resolution_clock::now();
            const uint64_t ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
            KVCacheProfiler::record(KVCacheOpType::APPEND, ns, static_cast<uint64_t>(num_tokens), 0);
            return ok;
        }
    }

    // =========================================================================
    // CUDARingKVCache<Precision>::get_k() / get_v() implementations
    // =========================================================================
    // These create GpuTensorView wrappers around the cached device buffers.
    // Views are stored in tensor_views_ and reused on subsequent calls.

    template <ActivationPrecision Precision>
    ITensor *CUDARingKVCache<Precision>::get_k(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_WARN("[CUDARingKVCache::get_k] Invalid layer=" << layer
                                                               << " seq_idx=" << seq_idx);
            return nullptr;
        }

        // Get device pointers via non-virtual get_kv_typed to avoid double-mapping
        // when called from a derived class (e.g. CUDAHybridRingKVCache) that overrides
        // the virtual get_kv_for_attention with its own layer remapping.
        const DataT *d_k_typed = nullptr;
        const DataT *d_v_typed = nullptr;
        int kv_len = 0;

        if (!get_kv_typed(layer, seq_idx, &d_k_typed, &d_v_typed, &kv_len, 0))
        {
            LOG_WARN("[CUDARingKVCache::get_k] get_kv_typed failed for layer="
                     << layer << " seq_idx=" << seq_idx);
            return nullptr;
        }

        const void *d_k = d_k_typed;
        if (!d_k || kv_len == 0)
        {
            // Empty cache - valid state, return nullptr
            return nullptr;
        }

        // Convert ActivationPrecision to TensorType at compile time
        constexpr TensorType tensor_type = []() constexpr
        {
            if constexpr (Precision == ActivationPrecision::FP16)
                return TensorType::FP16;
            else if constexpr (Precision == ActivationPrecision::BF16)
                return TensorType::BF16;
            else if constexpr (Precision == ActivationPrecision::Q8_1)
                return TensorType::Q8_1;
            else
                return TensorType::FP32;
        }();

        const size_t view_cols = (Precision == ActivationPrecision::Q8_1)
                                     ? static_cast<size_t>(kv_storage_dim_)
                                     : static_cast<size_t>(kv_dim_);

        // Create or update the view
        auto &view = tensor_views_[layer][seq_idx][0]; // Index 0 = K

        // Check if view needs to be created or updated (pointer or size changed)
        if (!view ||
            view->gpu_data_ptr() != d_k ||
            view->rows() != static_cast<size_t>(kv_len))
        {
            // Create new view wrapping the device buffer
            view = std::make_unique<GpuTensorView>(
                const_cast<void *>(d_k), // GpuTensorView needs non-const for interface
                static_cast<size_t>(kv_len),
                view_cols,
                tensor_type,
                device_id_);

            LOG_TRACE("[CUDARingKVCache::get_k] Created view for layer=" << layer
                                                                         << " seq=" << seq_idx << " kv_len=" << kv_len);
        }

        return view.get();
    }

    template <ActivationPrecision Precision>
    const ITensor *CUDARingKVCache<Precision>::get_k(int layer, int seq_idx) const
    {
        // Delegate to non-const version (tensor_views_ is mutable)
        return const_cast<CUDARingKVCache<Precision> *>(this)->get_k(layer, seq_idx);
    }

    template <ActivationPrecision Precision>
    ITensor *CUDARingKVCache<Precision>::get_v(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_WARN("[CUDARingKVCache::get_v] Invalid layer=" << layer
                                                               << " seq_idx=" << seq_idx);
            return nullptr;
        }

        // Get device pointers via non-virtual get_kv_typed to avoid double-mapping
        // when called from a derived class (e.g. CUDAHybridRingKVCache) that overrides
        // the virtual get_kv_for_attention with its own layer remapping.
        const DataT *d_k_typed = nullptr;
        const DataT *d_v_typed = nullptr;
        int kv_len = 0;

        if (!get_kv_typed(layer, seq_idx, &d_k_typed, &d_v_typed, &kv_len, 0))
        {
            LOG_WARN("[CUDARingKVCache::get_v] get_kv_typed failed for layer="
                     << layer << " seq_idx=" << seq_idx);
            return nullptr;
        }

        const void *d_v = d_v_typed;
        if (!d_v || kv_len == 0)
        {
            // Empty cache - valid state, return nullptr
            return nullptr;
        }

        // Convert ActivationPrecision to TensorType at compile time
        constexpr TensorType tensor_type = []() constexpr
        {
            if constexpr (Precision == ActivationPrecision::FP16)
                return TensorType::FP16;
            else if constexpr (Precision == ActivationPrecision::BF16)
                return TensorType::BF16;
            else if constexpr (Precision == ActivationPrecision::Q8_1)
                return TensorType::Q8_1;
            else
                return TensorType::FP32;
        }();

        const size_t view_cols = (Precision == ActivationPrecision::Q8_1)
                                     ? static_cast<size_t>(kv_storage_dim_)
                                     : static_cast<size_t>(kv_dim_);

        // Create or update the view
        auto &view = tensor_views_[layer][seq_idx][1]; // Index 1 = V

        // Check if view needs to be created or updated (pointer or size changed)
        if (!view ||
            view->gpu_data_ptr() != d_v ||
            view->rows() != static_cast<size_t>(kv_len))
        {
            // Create new view wrapping the device buffer
            view = std::make_unique<GpuTensorView>(
                const_cast<void *>(d_v), // GpuTensorView needs non-const for interface
                static_cast<size_t>(kv_len),
                view_cols,
                tensor_type,
                device_id_);

            LOG_TRACE("[CUDARingKVCache::get_v] Created view for layer=" << layer
                                                                         << " seq=" << seq_idx << " kv_len=" << kv_len);
        }

        return view.get();
    }

    template <ActivationPrecision Precision>
    const ITensor *CUDARingKVCache<Precision>::get_v(int layer, int seq_idx) const
    {
        // Delegate to non-const version (tensor_views_ is mutable)
        return const_cast<CUDARingKVCache<Precision> *>(this)->get_v(layer, seq_idx);
    }

    // =========================================================================
    // get_kv(): single-pass K+V ITensor access
    // =========================================================================

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::get_kv(
        int layer, int seq_idx,
        ITensor **out_k, ITensor **out_v,
        int *out_kv_len)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return false;

        // Single call gets both K and V device pointers (non-virtual to avoid
        // double-mapping when called from derived class with layer remapping)
        const DataT *d_k_typed = nullptr;
        const DataT *d_v_typed = nullptr;
        int kv_len = 0;

        if (!get_kv_typed(layer, seq_idx, &d_k_typed, &d_v_typed, &kv_len, 0))
            return false;

        const void *d_k = d_k_typed;
        const void *d_v = d_v_typed;

        if (kv_len == 0 || !d_k || !d_v)
        {
            if (out_kv_len)
                *out_kv_len = 0;
            return true;
        }

        constexpr TensorType tensor_type = []() constexpr
        {
            if constexpr (Precision == ActivationPrecision::FP16)
                return TensorType::FP16;
            else if constexpr (Precision == ActivationPrecision::BF16)
                return TensorType::BF16;
            else if constexpr (Precision == ActivationPrecision::Q8_1)
                return TensorType::Q8_1;
            else
                return TensorType::FP32;
        }();

        const size_t view_cols = (Precision == ActivationPrecision::Q8_1)
                                     ? static_cast<size_t>(kv_storage_dim_)
                                     : static_cast<size_t>(kv_dim_);
        const size_t rows = static_cast<size_t>(kv_len);

        // Update K view (index 0)
        auto &k_view = tensor_views_[layer][seq_idx][0];
        if (!k_view || k_view->gpu_data_ptr() != d_k || k_view->rows() != rows)
        {
            k_view = std::make_unique<GpuTensorView>(
                const_cast<void *>(d_k), rows, view_cols, tensor_type, device_id_);
        }

        // Update V view (index 1)
        auto &v_view = tensor_views_[layer][seq_idx][1];
        if (!v_view || v_view->gpu_data_ptr() != d_v || v_view->rows() != rows)
        {
            v_view = std::make_unique<GpuTensorView>(
                const_cast<void *>(d_v), rows, view_cols, tensor_type, device_id_);
        }

        if (out_k)
            *out_k = k_view.get();
        if (out_v)
            *out_v = v_view.get();
        if (out_kv_len)
            *out_kv_len = kv_len;
        return true;
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::get_kv(
        int layer, int seq_idx,
        const ITensor **out_k, const ITensor **out_v,
        int *out_kv_len) const
    {
        // Delegate to non-const version (tensor_views_ is mutable)
        ITensor *k = nullptr;
        ITensor *v = nullptr;
        bool ok = const_cast<CUDARingKVCache<Precision> *>(this)->get_kv(layer, seq_idx, &k, &v, out_kv_len);
        if (ok)
        {
            if (out_k)
                *out_k = k;
            if (out_v)
                *out_v = v;
        }
        return ok;
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::get_kv_snapshot_view(
        int layer, int seq_idx,
        int token_count,
        ITensor **out_k, ITensor **out_v,
        int *out_kv_len)
    {
        if (out_k)
            *out_k = nullptr;
        if (out_v)
            *out_v = nullptr;
        if (out_kv_len)
            *out_kv_len = 0;

        if (layer < 0 || layer >= n_layers_ ||
            seq_idx < 0 || seq_idx >= batch_size_ ||
            token_count <= 0 || token_count > max_seq_len_)
        {
            return false;
        }

        const EntryT &entry = entries_[layer][seq_idx];
        KVCacheSequenceState state;
        if (!entry.d_K || !entry.d_V ||
            !observeDeviceSequenceState(layer, seq_idx, &state) ||
            token_count < state.cached_tokens)
            return false;

        const int tail =
            (state.implementation_head - state.cached_tokens + max_seq_len_) %
            max_seq_len_;
        if (state.cached_tokens > 0 && tail != 0)
            return false;
        if (state.implementation_head != state.cached_tokens)
            return false;

        constexpr TensorType tensor_type = []() constexpr
        {
            if constexpr (Precision == ActivationPrecision::FP16)
                return TensorType::FP16;
            else if constexpr (Precision == ActivationPrecision::BF16)
                return TensorType::BF16;
            else if constexpr (Precision == ActivationPrecision::Q8_1)
                return TensorType::Q8_1;
            else
                return TensorType::FP32;
        }();

        if (snapshot_tensor_views_.empty())
        {
            snapshot_tensor_views_.resize(n_layers_);
            for (int l = 0; l < n_layers_; ++l)
                snapshot_tensor_views_[l].resize(batch_size_);
        }

        const size_t rows = static_cast<size_t>(token_count);
        const size_t view_cols = (Precision == ActivationPrecision::Q8_1)
                                     ? static_cast<size_t>(kv_storage_dim_)
                                     : static_cast<size_t>(kv_dim_);

        auto &k_view = snapshot_tensor_views_[layer][seq_idx][0];
        if (!k_view || k_view->gpu_data_ptr() != entry.d_K || k_view->rows() != rows)
        {
            k_view = std::make_unique<GpuTensorView>(
                static_cast<void *>(entry.d_K), rows, view_cols, tensor_type, device_id_);
        }

        auto &v_view = snapshot_tensor_views_[layer][seq_idx][1];
        if (!v_view || v_view->gpu_data_ptr() != entry.d_V || v_view->rows() != rows)
        {
            v_view = std::make_unique<GpuTensorView>(
                static_cast<void *>(entry.d_V), rows, view_cols, tensor_type, device_id_);
        }

        if (out_k)
            *out_k = k_view.get();
        if (out_v)
            *out_v = v_view.get();
        if (out_kv_len)
            *out_kv_len = token_count;
        return true;
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::get_kv_snapshot_view(
        int layer, int seq_idx,
        int token_count,
        const ITensor **out_k, const ITensor **out_v,
        int *out_kv_len) const
    {
        ITensor *k = nullptr;
        ITensor *v = nullptr;
        const bool ok = const_cast<CUDARingKVCache<Precision> *>(this)->get_kv_snapshot_view(
            layer, seq_idx, token_count, &k, &v, out_kv_len);
        if (ok)
        {
            if (out_k)
                *out_k = k;
            if (out_v)
                *out_v = v;
        }
        return ok;
    }

    // Explicit template instantiations
    template ITensor *CUDARingKVCache<ActivationPrecision::FP32>::get_k(int, int);
    template const ITensor *CUDARingKVCache<ActivationPrecision::FP32>::get_k(int, int) const;
    template ITensor *CUDARingKVCache<ActivationPrecision::FP32>::get_v(int, int);
    template const ITensor *CUDARingKVCache<ActivationPrecision::FP32>::get_v(int, int) const;

    template ITensor *CUDARingKVCache<ActivationPrecision::FP16>::get_k(int, int);
    template const ITensor *CUDARingKVCache<ActivationPrecision::FP16>::get_k(int, int) const;
    template ITensor *CUDARingKVCache<ActivationPrecision::FP16>::get_v(int, int);
    template const ITensor *CUDARingKVCache<ActivationPrecision::FP16>::get_v(int, int) const;

    template ITensor *CUDARingKVCache<ActivationPrecision::BF16>::get_k(int, int);
    template const ITensor *CUDARingKVCache<ActivationPrecision::BF16>::get_k(int, int) const;
    template ITensor *CUDARingKVCache<ActivationPrecision::BF16>::get_v(int, int);
    template const ITensor *CUDARingKVCache<ActivationPrecision::BF16>::get_v(int, int) const;

    template ITensor *CUDARingKVCache<ActivationPrecision::Q8_1>::get_k(int, int);
    template const ITensor *CUDARingKVCache<ActivationPrecision::Q8_1>::get_k(int, int) const;
    template ITensor *CUDARingKVCache<ActivationPrecision::Q8_1>::get_v(int, int);
    template const ITensor *CUDARingKVCache<ActivationPrecision::Q8_1>::get_v(int, int) const;

    template bool CUDARingKVCache<ActivationPrecision::FP32>::get_kv(int, int, ITensor **, ITensor **, int *);
    template bool CUDARingKVCache<ActivationPrecision::FP32>::get_kv(int, int, const ITensor **, const ITensor **, int *) const;
    template bool CUDARingKVCache<ActivationPrecision::FP16>::get_kv(int, int, ITensor **, ITensor **, int *);
    template bool CUDARingKVCache<ActivationPrecision::FP16>::get_kv(int, int, const ITensor **, const ITensor **, int *) const;
    template bool CUDARingKVCache<ActivationPrecision::BF16>::get_kv(int, int, ITensor **, ITensor **, int *);
    template bool CUDARingKVCache<ActivationPrecision::BF16>::get_kv(int, int, const ITensor **, const ITensor **, int *) const;
    template bool CUDARingKVCache<ActivationPrecision::Q8_1>::get_kv(int, int, ITensor **, ITensor **, int *);
    template bool CUDARingKVCache<ActivationPrecision::Q8_1>::get_kv(int, int, const ITensor **, const ITensor **, int *) const;
    template bool CUDARingKVCache<ActivationPrecision::FP32>::get_kv_snapshot_view(int, int, int, ITensor **, ITensor **, int *);
    template bool CUDARingKVCache<ActivationPrecision::FP32>::get_kv_snapshot_view(int, int, int, const ITensor **, const ITensor **, int *) const;
    template bool CUDARingKVCache<ActivationPrecision::FP16>::get_kv_snapshot_view(int, int, int, ITensor **, ITensor **, int *);
    template bool CUDARingKVCache<ActivationPrecision::FP16>::get_kv_snapshot_view(int, int, int, const ITensor **, const ITensor **, int *) const;
    template bool CUDARingKVCache<ActivationPrecision::BF16>::get_kv_snapshot_view(int, int, int, ITensor **, ITensor **, int *);
    template bool CUDARingKVCache<ActivationPrecision::BF16>::get_kv_snapshot_view(int, int, int, const ITensor **, const ITensor **, int *) const;
    template bool CUDARingKVCache<ActivationPrecision::Q8_1>::get_kv_snapshot_view(int, int, int, ITensor **, ITensor **, int *);
    template bool CUDARingKVCache<ActivationPrecision::Q8_1>::get_kv_snapshot_view(int, int, int, const ITensor **, const ITensor **, int *) const;

    // =========================================================================
    // get_kv_converted(): FP16 shadow buffers with optional RoPE
    // =========================================================================

    extern "C" bool cuda_rope_apply_fp16(
        __half *d_K, int count,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        cudaStream_t stream, int rope_dim = 0);

    extern "C" bool cuda_rope_apply_fp32(
        float *d_K, int count,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        cudaStream_t stream, int rope_dim = 0);

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::ensureRoPEShadow(int layer, int seq_idx)
    {
        // Lazy init the outer vectors
        if (rope_shadows_.empty())
        {
            rope_shadows_.resize(n_layers_);
            for (auto &layer_shadows : rope_shadows_)
                layer_shadows.resize(batch_size_);
        }

        auto &shadow = rope_shadows_[layer][seq_idx];
        const size_t fp16_bytes =
            static_cast<size_t>(max_seq_len_) *
            static_cast<size_t>(kv_dim_) * sizeof(__half);
        if (!ensureConvScratch(fp16_bytes))
        {
            throw std::runtime_error(
                "[CUDARingKVCache] RoPE conversion requires bound K/V workspace");
        }
        shadow.d_K = static_cast<__half *>(conv_scratch_k_);
        shadow.d_V = static_cast<__half *>(conv_scratch_v_);
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::invalidateRoPEShadow(int layer, int seq_idx) const
    {
        if (rope_shadows_.empty())
            return;
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return;

        auto &shadow = rope_shadows_[layer][seq_idx];
        shadow.k_view.reset();
        shadow.v_view.reset();
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::get_kv_converted(
        int layer, int seq_idx,
        ActivationPrecision target,
        ITensor **out_k, ITensor **out_v,
        int *out_kv_len,
        const KVReadParams *rope)
    {
        (void)target; // We always produce FP16 on GPU

        if (out_k)
            *out_k = nullptr;
        if (out_v)
            *out_v = nullptr;
        if (out_kv_len)
            *out_kv_len = 0;
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return false;
        if (isGraphCaptureActive())
        {
            LOG_ERROR("[CUDARingKVCache::get_kv_converted] Scalar conversion is forbidden during graph capture; use get_kv_batched_converted_device_view");
            return false;
        }

        const auto &entry = entries_[layer][seq_idx];
        KVCacheSequenceState state;
        if (!observeDeviceSequenceState(layer, seq_idx, &state))
            return false;
        const int requested_token_count =
            rope && rope->requested_token_count > 0
                ? rope->requested_token_count
                : 0;
        if (requested_token_count > 0 &&
            requested_token_count != state.cached_tokens)
        {
            LOG_ERROR("[CUDARingKVCache::get_kv_converted] Requested scalar span does not match canonical device count"
                      << " requested=" << requested_token_count
                      << " device_count=" << state.cached_tokens);
            return false;
        }
        const int read_count = state.cached_tokens;
        if (read_count == 0)
            return true;

        // If no RoPE is requested, the raw cache tensors already have the
        // required representation.
        // Use qualified call to avoid virtual dispatch — CUDAHybridRingKVCache
        // overrides get_kv() with layer remapping, but the layer index passed here
        // has already been remapped by the hybrid override of get_kv_converted().
        const bool want_rope = (rope && rope->rope_theta > 0.0f);
        if (!want_rope)
            return CUDARingKVCache::get_kv(layer, seq_idx, out_k, out_v, out_kv_len);

        cudaSetDevice(device_id_);
        const cudaStream_t stream = getEffectiveStream(
            rope ? static_cast<cudaStream_t>(rope->gpu_stream) : nullptr);

        ensureRoPEShadow(layer, seq_idx);
        auto &shadow = rope_shadows_[layer][seq_idx];
        if (!shadow.d_K || !shadow.d_V || read_count > max_seq_len_)
        {
            LOG_ERROR("[CUDARingKVCache::get_kv_converted] Scalar conversion storage is unavailable");
            return false;
        }

        /*
         * This API is deliberately a complete observation-boundary rebuild.
         * Incremental validity cannot be inferred from host counters once graph
         * replay owns append/publication. The production grouped API performs
         * its fixed-shape conversion directly from canonical device metadata.
         */
        if constexpr (Precision == ActivationPrecision::FP16)
        {
            launch_linearize_kernel(
                entry, state.implementation_head, read_count,
                shadow.d_K, shadow.d_V, stream);
            if (!cuda_rope_apply_fp16(
                    shadow.d_K, read_count, local_n_kv_heads_, head_dim_,
                    rope->rope_theta, rope->position_start, stream,
                    rope->rope_dim))
                return false;
        }
        else if constexpr (Precision == ActivationPrecision::FP32)
        {
            auto *d_temp_k = entry.d_K_scratch;
            auto *d_temp_v = entry.d_V_scratch;
            launch_linearize_kernel(
                entry, state.implementation_head, read_count,
                d_temp_k, d_temp_v, stream);
            if (!cuda_rope_apply_fp32(
                    d_temp_k, read_count, local_n_kv_heads_, head_dim_,
                    rope->rope_theta, rope->position_start, stream,
                    rope->rope_dim) ||
                !cuda_convert_tensor_to_fp16(
                    d_temp_k, TensorType::FP32,
                    reinterpret_cast<uint16_t *>(shadow.d_K),
                    read_count * kv_dim_, stream) ||
                !cuda_convert_tensor_to_fp16(
                    d_temp_v, TensorType::FP32,
                    reinterpret_cast<uint16_t *>(shadow.d_V),
                    read_count * kv_dim_, stream))
                return false;
        }
        else if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            auto *d_temp_k = entry.d_K_scratch;
            auto *d_temp_v = entry.d_V_scratch;
            launch_linearize_kernel(
                entry, state.implementation_head, read_count,
                d_temp_k, d_temp_v, stream);
            if (!cuda_convert_tensor_to_fp16(
                    d_temp_k, TensorType::Q8_1,
                    reinterpret_cast<uint16_t *>(shadow.d_K),
                    read_count * kv_dim_, stream) ||
                !cuda_convert_tensor_to_fp16(
                    d_temp_v, TensorType::Q8_1,
                    reinterpret_cast<uint16_t *>(shadow.d_V),
                    read_count * kv_dim_, stream) ||
                !cuda_rope_apply_fp16(
                    shadow.d_K, read_count, local_n_kv_heads_, head_dim_,
                    rope->rope_theta, rope->position_start, stream,
                    rope->rope_dim))
                return false;
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            auto *d_temp_k = entry.d_K_scratch;
            auto *d_temp_v = entry.d_V_scratch;
            launch_linearize_kernel(
                entry, state.implementation_head, read_count,
                d_temp_k, d_temp_v, stream);
            if (!cuda_convert_tensor_to_fp16(
                    d_temp_k, TensorType::BF16,
                    reinterpret_cast<uint16_t *>(shadow.d_K),
                    read_count * kv_dim_, stream) ||
                !cuda_convert_tensor_to_fp16(
                    d_temp_v, TensorType::BF16,
                    reinterpret_cast<uint16_t *>(shadow.d_V),
                    read_count * kv_dim_, stream) ||
                !cuda_rope_apply_fp16(
                    shadow.d_K, read_count, local_n_kv_heads_, head_dim_,
                    rope->rope_theta, rope->position_start, stream,
                    rope->rope_dim))
                return false;
        }

        if (cudaGetLastError() != cudaSuccess)
            return false;

        // Create/update GpuTensorViews
        if (!shadow.k_view || shadow.k_view->shape()[0] != static_cast<size_t>(read_count))
        {
            shadow.k_view = std::make_unique<GpuTensorView>(
                shadow.d_K, read_count, kv_dim_,
                TensorType::FP16, device_id_);
        }
        if (!shadow.v_view || shadow.v_view->shape()[0] != static_cast<size_t>(read_count))
        {
            shadow.v_view = std::make_unique<GpuTensorView>(
                shadow.d_V, read_count, kv_dim_,
                TensorType::FP16, device_id_);
        }

        if (out_k)
            *out_k = shadow.k_view.get();
        if (out_v)
            *out_v = shadow.v_view.get();
        if (out_kv_len)
            *out_kv_len = read_count;

        return true;
    }

    // Explicit template instantiations for get_kv_converted
    template bool CUDARingKVCache<ActivationPrecision::FP32>::get_kv_converted(int, int, ActivationPrecision, ITensor **, ITensor **, int *, const KVReadParams *);
    template bool CUDARingKVCache<ActivationPrecision::FP16>::get_kv_converted(int, int, ActivationPrecision, ITensor **, ITensor **, int *, const KVReadParams *);
    template bool CUDARingKVCache<ActivationPrecision::BF16>::get_kv_converted(int, int, ActivationPrecision, ITensor **, ITensor **, int *, const KVReadParams *);
    template bool CUDARingKVCache<ActivationPrecision::Q8_1>::get_kv_converted(int, int, ActivationPrecision, ITensor **, ITensor **, int *, const KVReadParams *);

    // Explicit template instantiations for shadow helpers
    template void CUDARingKVCache<ActivationPrecision::FP32>::ensureRoPEShadow(int, int);
    template void CUDARingKVCache<ActivationPrecision::FP16>::ensureRoPEShadow(int, int);
    template void CUDARingKVCache<ActivationPrecision::BF16>::ensureRoPEShadow(int, int);
    template void CUDARingKVCache<ActivationPrecision::Q8_1>::ensureRoPEShadow(int, int);

    template void CUDARingKVCache<ActivationPrecision::FP32>::invalidateRoPEShadow(int, int) const;
    template void CUDARingKVCache<ActivationPrecision::FP16>::invalidateRoPEShadow(int, int) const;
    template void CUDARingKVCache<ActivationPrecision::BF16>::invalidateRoPEShadow(int, int) const;
    template void CUDARingKVCache<ActivationPrecision::Q8_1>::invalidateRoPEShadow(int, int) const;

} // namespace llaminar2
