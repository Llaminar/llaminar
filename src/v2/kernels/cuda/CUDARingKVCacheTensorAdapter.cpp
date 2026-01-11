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
#include "../../tensors/GpuTensorView.h"
#include "../../utils/Logger.h"

namespace llaminar2
{

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
        if (!K || !V)
        {
            LOG_DEBUG("[ICUDARingKVCache::append(ITensor)] Null K or V tensor");
            return false;
        }

        // Get GPU data pointers from tensors via ITensor interface
        const void *d_k = K->gpu_data_ptr();
        const void *d_v = V->gpu_data_ptr();

        if (!d_k || !d_v)
        {
            // Tensors don't have GPU data - caller should have called ensureOnDevice()
            LOG_ERROR("[ICUDARingKVCache::append(ITensor)] K or V tensor lacks GPU data. "
                      << "Call ensureOnDevice() before append.");
            return false;
        }

        // Delegate to the device pointer version (with default stream 0)
        return append(layer, seq_idx, d_k, d_v, num_tokens, 0);
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

        // Get device pointers via get_kv_for_attention
        const void *d_k = nullptr;
        const void *d_v = nullptr;
        int kv_len = 0;

        if (!get_kv_for_attention(layer, seq_idx, &d_k, &d_v, &kv_len, 0))
        {
            LOG_WARN("[CUDARingKVCache::get_k] get_kv_for_attention failed for layer="
                     << layer << " seq_idx=" << seq_idx);
            return nullptr;
        }

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
            else
                return TensorType::FP32;
        }();

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
                static_cast<size_t>(kv_dim_),
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

        // Get device pointers via get_kv_for_attention
        const void *d_k = nullptr;
        const void *d_v = nullptr;
        int kv_len = 0;

        if (!get_kv_for_attention(layer, seq_idx, &d_k, &d_v, &kv_len, 0))
        {
            LOG_WARN("[CUDARingKVCache::get_v] get_kv_for_attention failed for layer="
                     << layer << " seq_idx=" << seq_idx);
            return nullptr;
        }

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
            else
                return TensorType::FP32;
        }();

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
                static_cast<size_t>(kv_dim_),
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

} // namespace llaminar2
