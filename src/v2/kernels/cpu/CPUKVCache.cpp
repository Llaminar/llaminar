/**
 * @file CPUKVCache.cpp
 * @brief Unified KV cache implementation
 * @author David Sanftenberg
 * @date December 2025
 */

#include "CPUKVCache.h"
#include "../../utils/Logger.h"
#include "../../tensors/SIMDHelpers.h"
#include <cstring>
#include <algorithm>
#include <omp.h>

namespace llaminar2
{

    // =========================================================================
    // Legacy Device Index Conversion Helper
    // =========================================================================

    namespace
    {
        /**
         * @brief Convert legacy device index to DeviceId
         *
         * Legacy convention: 0 = CPU, 1+ = GPU (1=GPU0, 2=GPU1, etc.)
         * This matches the old DeviceManager indexing.
         *
         * TODO: Remove this when CPUKVCache API is updated to take vector<DeviceId>
         */
        DeviceId deviceIdFromLegacyIndex(int legacy_idx)
        {
            if (legacy_idx <= 0)
            {
                return DeviceId::cpu();
            }
            // Legacy: GPU indices start at 1, DeviceId::cuda expects 0-based ordinal
            return DeviceId::cuda(legacy_idx - 1);
        }
    } // anonymous namespace

    // =========================================================================
    // Helper Specializations: Tensor Allocation
    // =========================================================================

    template <>
    std::shared_ptr<FP32Tensor> CPUKVCache<ActivationPrecision::FP32>::allocate_tensor(
        size_t rows, size_t cols, DeviceId device)
    {
        (void)device;
        return tensor_factory_->createFP32({rows, cols}, device);
    }

    template <>
    std::shared_ptr<BF16Tensor> CPUKVCache<ActivationPrecision::BF16>::allocate_tensor(
        size_t rows, size_t cols, DeviceId /* device */)
    {
        return tensor_factory_->createBF16({rows, cols});
    }

    template <>
    std::shared_ptr<FP16Tensor> CPUKVCache<ActivationPrecision::FP16>::allocate_tensor(
        size_t rows, size_t cols, DeviceId /* device */)
    {
        return tensor_factory_->createFP16({rows, cols});
    }

    template <>
    std::shared_ptr<Q8_1Tensor> CPUKVCache<ActivationPrecision::Q8_1>::allocate_tensor(
        size_t rows, size_t cols, DeviceId /* device */)
    {
        return tensor_factory_->createQ8_1({rows, cols});
    }

    template <>
    std::shared_ptr<Q16_1Tensor> CPUKVCache<ActivationPrecision::Q16_1>::allocate_tensor(
        size_t rows, size_t cols, DeviceId device)
    {
        // Auto-select optimal Q16 block size based on head dimension for 1-block-per-head
        // This eliminates per-block scale tracking overhead in integer attention
        const Q16BlockSize block_size = optimal_q16_block_size(head_dim_);
        return tensor_factory_->createQ16_1({rows, cols}, block_size, device);
    }

    // =========================================================================
    // Helper Specializations: Copy Append Data
    // =========================================================================

    template <>
    void CPUKVCache<ActivationPrecision::FP32>::copy_append_data(
        FP32Tensor *dst, const FP32Tensor *src, int offset_tokens, int new_tokens)
    {
        float *dst_data = dst->mutable_data();
        const float *src_data = src->data();

        if (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
        {
            // HEAD_MAJOR: scatter from [new_tokens, n_kv_heads * head_dim] to [n_kv_heads * max_seq_len, head_dim]
            // Input: src[t * kv_dim_ + h * head_dim_ + d]
            // Output: dst[(h * max_seq_len_ + offset_tokens + t) * head_dim_ + d]
            for (int t = 0; t < new_tokens; ++t)
            {
                for (int h = 0; h < local_n_kv_heads_; ++h)
                {
                    const float *src_head = src_data + t * kv_dim_ + h * head_dim_;
                    float *dst_head = dst_data + (h * max_seq_len_ + offset_tokens + t) * head_dim_;
                    std::memcpy(dst_head, src_head, head_dim_ * sizeof(float));
                }
            }
        }
        else
        {
            // POSITION_MAJOR: simple contiguous copy
            size_t offset = static_cast<size_t>(offset_tokens) * kv_dim_;
            size_t copy_size = static_cast<size_t>(new_tokens) * kv_dim_ * sizeof(float);
            std::memcpy(dst_data + offset, src_data, copy_size);
        }
    }

    template <>
    void CPUKVCache<ActivationPrecision::BF16>::copy_append_data(
        BF16Tensor *dst, const BF16Tensor *src, int offset_tokens, int new_tokens)
    {
        uint16_t *dst_data = dst->mutable_typed_data();
        const uint16_t *src_data = src->typed_data();

        if (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
        {
            // HEAD_MAJOR: scatter from [new_tokens, n_kv_heads * head_dim] to [n_kv_heads * max_seq_len, head_dim]
            for (int t = 0; t < new_tokens; ++t)
            {
                for (int h = 0; h < local_n_kv_heads_; ++h)
                {
                    const uint16_t *src_head = src_data + t * kv_dim_ + h * head_dim_;
                    uint16_t *dst_head = dst_data + (h * max_seq_len_ + offset_tokens + t) * head_dim_;
                    std::memcpy(dst_head, src_head, head_dim_ * sizeof(uint16_t));
                }
            }
        }
        else
        {
            // POSITION_MAJOR: simple contiguous copy
            size_t offset = static_cast<size_t>(offset_tokens) * kv_dim_;
            size_t copy_size = static_cast<size_t>(new_tokens) * kv_dim_ * sizeof(uint16_t);
            std::memcpy(dst_data + offset, src_data, copy_size);
        }
    }

    template <>
    void CPUKVCache<ActivationPrecision::FP16>::copy_append_data(
        FP16Tensor *dst, const FP16Tensor *src, int offset_tokens, int new_tokens)
    {
        uint16_t *dst_data = dst->mutable_typed_data();
        const uint16_t *src_data = src->typed_data();

        if (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
        {
            // HEAD_MAJOR: scatter from [new_tokens, n_kv_heads * head_dim] to [n_kv_heads * max_seq_len, head_dim]
            for (int t = 0; t < new_tokens; ++t)
            {
                for (int h = 0; h < local_n_kv_heads_; ++h)
                {
                    const uint16_t *src_head = src_data + t * kv_dim_ + h * head_dim_;
                    uint16_t *dst_head = dst_data + (h * max_seq_len_ + offset_tokens + t) * head_dim_;
                    std::memcpy(dst_head, src_head, head_dim_ * sizeof(uint16_t));
                }
            }
        }
        else
        {
            // POSITION_MAJOR: simple contiguous copy
            size_t offset = static_cast<size_t>(offset_tokens) * kv_dim_;
            size_t copy_size = static_cast<size_t>(new_tokens) * kv_dim_ * sizeof(uint16_t);
            std::memcpy(dst_data + offset, src_data, copy_size);
        }
    }

    template <>
    void CPUKVCache<ActivationPrecision::Q8_1>::copy_append_data(
        Q8_1Tensor *dst, const Q8_1Tensor *src, int offset_tokens, int new_tokens)
    {
        Q8_1Block *dst_blocks = dst->mutable_typed_data();
        const Q8_1Block *src_blocks = src->typed_data();

        if (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
        {
            // HEAD_MAJOR: scatter from [new_tokens, n_kv_heads * head_dim] to [n_kv_heads * max_seq_len, head_dim]
            // Input has blocks_per_row = kv_dim / 32, output has blocks_per_head = head_dim / 32
            const size_t blocks_per_head = (head_dim_ + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
            const size_t src_blocks_per_row = (kv_dim_ + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;

            for (int t = 0; t < new_tokens; ++t)
            {
                for (int h = 0; h < local_n_kv_heads_; ++h)
                {
                    // Source: position-major [t][h] -> block index: t * src_blocks_per_row + h * blocks_per_head
                    const Q8_1Block *src_head = src_blocks + t * src_blocks_per_row + h * blocks_per_head;
                    // Dest: head-major [h][offset+t] -> block index: (h * max_seq_len_ + offset + t) * blocks_per_head
                    Q8_1Block *dst_head = dst_blocks + (h * max_seq_len_ + offset_tokens + t) * blocks_per_head;
                    std::memcpy(dst_head, src_head, blocks_per_head * sizeof(Q8_1Block));
                }
            }
        }
        else
        {
            // POSITION_MAJOR: simple contiguous copy
            size_t blocks_per_row = (kv_dim_ + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
            size_t offset_blocks = static_cast<size_t>(offset_tokens) * blocks_per_row;
            size_t copy_blocks = static_cast<size_t>(new_tokens) * blocks_per_row;
            size_t copy_size = copy_blocks * sizeof(Q8_1Block);
            std::memcpy(dst_blocks + offset_blocks, src_blocks, copy_size);
        }
    }

    template <>
    void CPUKVCache<ActivationPrecision::Q16_1>::copy_append_data(
        Q16_1Tensor *dst, const Q16_1Tensor *src, int offset_tokens, int new_tokens)
    {
        // Use tensor's actual block size for variable-size Q16 blocks
        const Q16BlockSize bs = dst->q16_block_size();
        const size_t block_bytes = q16_block_size_bytes(bs);
        const size_t block_elements = q16_block_size_elements(bs);

        uint8_t *dst_bytes = static_cast<uint8_t *>(dst->raw_mutable_data());
        const uint8_t *src_bytes = static_cast<const uint8_t *>(src->raw_data());

        if (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
        {
            // HEAD_MAJOR: scatter from [new_tokens, n_kv_heads * head_dim] to [n_kv_heads * max_seq_len, head_dim]
            // Input has blocks_per_row = kv_dim / block_size, output has blocks_per_head = head_dim / block_size
            const size_t blocks_per_head = (head_dim_ + block_elements - 1) / block_elements;
            const size_t src_blocks_per_row = (kv_dim_ + block_elements - 1) / block_elements;

            for (int t = 0; t < new_tokens; ++t)
            {
                for (int h = 0; h < local_n_kv_heads_; ++h)
                {
                    // Source: position-major [t][h] -> block index: t * src_blocks_per_row + h * blocks_per_head
                    const uint8_t *src_head = src_bytes + (t * src_blocks_per_row + h * blocks_per_head) * block_bytes;
                    // Dest: head-major [h][offset+t] -> block index: (h * max_seq_len_ + offset + t) * blocks_per_head
                    uint8_t *dst_head = dst_bytes + (h * max_seq_len_ + offset_tokens + t) * blocks_per_head * block_bytes;
                    std::memcpy(dst_head, src_head, blocks_per_head * block_bytes);
                }
            }
        }
        else
        {
            // POSITION_MAJOR: simple contiguous copy
            size_t blocks_per_row = (kv_dim_ + block_elements - 1) / block_elements;
            size_t offset_blocks = static_cast<size_t>(offset_tokens) * blocks_per_row;
            size_t copy_blocks = static_cast<size_t>(new_tokens) * blocks_per_row;
            size_t offset_bytes_val = offset_blocks * block_bytes;
            size_t copy_size = copy_blocks * block_bytes;
            std::memcpy(dst_bytes + offset_bytes_val, src_bytes, copy_size);
        }
    }

    // =========================================================================
    // Helper Specializations: Shift Evict Data
    // =========================================================================

    template <>
    void CPUKVCache<ActivationPrecision::FP32>::shift_evict_data(
        FP32Tensor *tensor, int tokens_to_evict, int tokens_to_keep)
    {
        float *data = tensor->mutable_data();

        if (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
        {
            // HEAD_MAJOR: shift each head's data independently
            // Layout: [n_kv_heads * max_seq_len, head_dim]
            // Each head has [max_seq_len, head_dim] contiguous block
            for (int h = 0; h < local_n_kv_heads_; ++h)
            {
                float *head_data = data + h * max_seq_len_ * head_dim_;
                size_t evict_offset = static_cast<size_t>(tokens_to_evict) * head_dim_;
                size_t keep_size = static_cast<size_t>(tokens_to_keep) * head_dim_ * sizeof(float);
                std::memmove(head_data, head_data + evict_offset, keep_size);
            }
        }
        else
        {
            // POSITION_MAJOR: simple contiguous shift
            size_t evict_offset = static_cast<size_t>(tokens_to_evict) * kv_dim_;
            size_t keep_size = static_cast<size_t>(tokens_to_keep) * kv_dim_ * sizeof(float);
            std::memmove(data, data + evict_offset, keep_size);
        }
    }

    template <>
    void CPUKVCache<ActivationPrecision::BF16>::shift_evict_data(
        BF16Tensor *tensor, int tokens_to_evict, int tokens_to_keep)
    {
        uint16_t *data = tensor->mutable_typed_data();

        if (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
        {
            // HEAD_MAJOR: shift each head's data independently
            for (int h = 0; h < local_n_kv_heads_; ++h)
            {
                uint16_t *head_data = data + h * max_seq_len_ * head_dim_;
                size_t evict_offset = static_cast<size_t>(tokens_to_evict) * head_dim_;
                size_t keep_size = static_cast<size_t>(tokens_to_keep) * head_dim_ * sizeof(uint16_t);
                std::memmove(head_data, head_data + evict_offset, keep_size);
            }
        }
        else
        {
            // POSITION_MAJOR: simple contiguous shift
            size_t evict_offset = static_cast<size_t>(tokens_to_evict) * kv_dim_;
            size_t keep_size = static_cast<size_t>(tokens_to_keep) * kv_dim_ * sizeof(uint16_t);
            std::memmove(data, data + evict_offset, keep_size);
        }
    }

    template <>
    void CPUKVCache<ActivationPrecision::FP16>::shift_evict_data(
        FP16Tensor *tensor, int tokens_to_evict, int tokens_to_keep)
    {
        uint16_t *data = tensor->mutable_typed_data();

        if (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
        {
            // HEAD_MAJOR: shift each head's data independently
            for (int h = 0; h < local_n_kv_heads_; ++h)
            {
                uint16_t *head_data = data + h * max_seq_len_ * head_dim_;
                size_t evict_offset = static_cast<size_t>(tokens_to_evict) * head_dim_;
                size_t keep_size = static_cast<size_t>(tokens_to_keep) * head_dim_ * sizeof(uint16_t);
                std::memmove(head_data, head_data + evict_offset, keep_size);
            }
        }
        else
        {
            // POSITION_MAJOR: simple contiguous shift
            size_t evict_offset = static_cast<size_t>(tokens_to_evict) * kv_dim_;
            size_t keep_size = static_cast<size_t>(tokens_to_keep) * kv_dim_ * sizeof(uint16_t);
            std::memmove(data, data + evict_offset, keep_size);
        }
    }

    template <>
    void CPUKVCache<ActivationPrecision::Q8_1>::shift_evict_data(
        Q8_1Tensor *tensor, int tokens_to_evict, int tokens_to_keep)
    {
        Q8_1Block *blocks = tensor->mutable_typed_data();

        if (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
        {
            // HEAD_MAJOR: shift each head's data independently
            const size_t blocks_per_head = (head_dim_ + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
            for (int h = 0; h < local_n_kv_heads_; ++h)
            {
                Q8_1Block *head_blocks = blocks + h * max_seq_len_ * blocks_per_head;
                size_t evict_blocks = static_cast<size_t>(tokens_to_evict) * blocks_per_head;
                size_t keep_blocks = static_cast<size_t>(tokens_to_keep) * blocks_per_head;
                size_t keep_size = keep_blocks * sizeof(Q8_1Block);
                std::memmove(head_blocks, head_blocks + evict_blocks, keep_size);
            }
        }
        else
        {
            // POSITION_MAJOR: simple contiguous shift
            size_t blocks_per_row = (kv_dim_ + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
            size_t evict_blocks = static_cast<size_t>(tokens_to_evict) * blocks_per_row;
            size_t keep_blocks = static_cast<size_t>(tokens_to_keep) * blocks_per_row;
            size_t keep_size = keep_blocks * sizeof(Q8_1Block);
            std::memmove(blocks, blocks + evict_blocks, keep_size);
        }
    }

    template <>
    void CPUKVCache<ActivationPrecision::Q16_1>::shift_evict_data(
        Q16_1Tensor *tensor, int tokens_to_evict, int tokens_to_keep)
    {
        // Use tensor's actual block size for variable-size Q16 blocks
        const Q16BlockSize bs = tensor->q16_block_size();
        const size_t block_bytes = q16_block_size_bytes(bs);
        const size_t block_elements = q16_block_size_elements(bs);

        uint8_t *data = static_cast<uint8_t *>(tensor->raw_mutable_data());

        if (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
        {
            // HEAD_MAJOR: shift each head's data independently
            const size_t blocks_per_head = (head_dim_ + block_elements - 1) / block_elements;
            for (int h = 0; h < local_n_kv_heads_; ++h)
            {
                uint8_t *head_data = data + h * max_seq_len_ * blocks_per_head * block_bytes;
                size_t evict_bytes = static_cast<size_t>(tokens_to_evict) * blocks_per_head * block_bytes;
                size_t keep_bytes = static_cast<size_t>(tokens_to_keep) * blocks_per_head * block_bytes;
                std::memmove(head_data, head_data + evict_bytes, keep_bytes);
            }
        }
        else
        {
            // POSITION_MAJOR: simple contiguous shift
            size_t blocks_per_row = (kv_dim_ + block_elements - 1) / block_elements;
            size_t evict_blocks = static_cast<size_t>(tokens_to_evict) * blocks_per_row;
            size_t keep_blocks = static_cast<size_t>(tokens_to_keep) * blocks_per_row;
            size_t evict_offset = evict_blocks * block_bytes;
            size_t keep_size = keep_blocks * block_bytes;
            std::memmove(data, data + evict_offset, keep_size);
        }
    }

    // =========================================================================
    // Constructor Implementations
    // =========================================================================

    template <ActivationPrecision Precision>
    CPUKVCache<Precision>::CPUKVCache(
        const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim, DeviceId device, KVCacheLayoutMode layout_mode)
        : n_layers_(n_layers), batch_size_(batch_size), max_seq_len_(max_seq_len),
          n_kv_heads_(n_kv_heads), local_n_kv_heads_(n_kv_heads),
          kv_head_start_(0), head_dim_(head_dim),
          kv_dim_(n_kv_heads * head_dim), is_sharded_(false),
          layout_mode_(layout_mode)
    {
        tensor_factory_ = std::make_unique<TensorFactory>(mpi_ctx);
        layer_devices_.resize(n_layers_, device);
        entries_.resize(n_layers_);

// Parallelize layer initialization - each layer is independent
#pragma omp parallel for schedule(static) if (n_layers_ >= 4)
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            initialize_layer(layer, device);
        }

        LOG_DEBUG("CPUKVCache: n_layers=" << n_layers_ << ", batch_size=" << batch_size_
                                          << ", max_seq_len=" << max_seq_len_ << ", kv_dim=" << kv_dim_
                                          << ", precision=" << static_cast<int>(Precision)
                                          << ", layout_mode=" << (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR ? "HEAD_MAJOR" : "POSITION_MAJOR"));
    }

    template <ActivationPrecision Precision>
    CPUKVCache<Precision>::CPUKVCache(
        const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim, const std::vector<int> &attention_devices,
        KVCacheLayoutMode layout_mode)
        : n_layers_(n_layers), batch_size_(batch_size), max_seq_len_(max_seq_len),
          n_kv_heads_(n_kv_heads), local_n_kv_heads_(n_kv_heads),
          kv_head_start_(0), head_dim_(head_dim),
          kv_dim_(n_kv_heads * head_dim), is_sharded_(false),
          layout_mode_(layout_mode)
    {
        tensor_factory_ = std::make_unique<TensorFactory>(mpi_ctx);

        // Use provided device list or default to CPU
        if (attention_devices.size() >= static_cast<size_t>(n_layers_))
        {
            layer_devices_.reserve(n_layers_);
            for (int i = 0; i < n_layers_; ++i)
            {
                layer_devices_.push_back(deviceIdFromLegacyIndex(attention_devices[i]));
            }
        }
        else
        {
            layer_devices_.resize(n_layers_, DeviceId::cpu());
            LOG_WARN("CPUKVCache: attention_devices size mismatch, defaulting all layers to CPU");
        }

        entries_.resize(n_layers_);

// Parallelize layer initialization - each layer is independent
#pragma omp parallel for schedule(static) if (n_layers_ >= 4)
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            initialize_layer(layer, layer_devices_[layer]);
        }

        LOG_DEBUG("CPUKVCache: n_layers=" << n_layers_ << ", batch_size=" << batch_size_
                                          << ", max_seq_len=" << max_seq_len_ << ", kv_dim=" << kv_dim_
                                          << ", precision=" << static_cast<int>(Precision)
                                          << ", layout_mode=" << (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR ? "HEAD_MAJOR" : "POSITION_MAJOR")
                                          << ", per-layer devices");
    }

    // =========================================================================
    // Sharded Constructor Implementations (for Tensor Parallelism)
    // =========================================================================

    template <ActivationPrecision Precision>
    CPUKVCache<Precision>::CPUKVCache(
        const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, DeviceId device, KVCacheLayoutMode layout_mode)
        : n_layers_(n_layers), batch_size_(batch_size), max_seq_len_(max_seq_len),
          n_kv_heads_(n_kv_heads), local_n_kv_heads_(local_n_kv_heads),
          kv_head_start_(kv_head_start), head_dim_(head_dim),
          kv_dim_(local_n_kv_heads * head_dim), is_sharded_(true),
          layout_mode_(layout_mode)
    {
        tensor_factory_ = std::make_unique<TensorFactory>(mpi_ctx);
        layer_devices_.resize(n_layers_, device);
        entries_.resize(n_layers_);

// Parallelize layer initialization - each layer is independent
#pragma omp parallel for schedule(static) if (n_layers_ >= 4)
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            initialize_layer(layer, device);
        }

        LOG_DEBUG("CPUKVCache (sharded): n_layers=" << n_layers_ << ", batch_size=" << batch_size_
                                                    << ", max_seq_len=" << max_seq_len_
                                                    << ", n_kv_heads=" << n_kv_heads_
                                                    << ", local_n_kv_heads=" << local_n_kv_heads_
                                                    << ", kv_head_start=" << kv_head_start_
                                                    << ", kv_dim=" << kv_dim_
                                                    << ", precision=" << static_cast<int>(Precision)
                                                    << ", layout_mode=" << (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR ? "HEAD_MAJOR" : "POSITION_MAJOR"));
    }

    template <ActivationPrecision Precision>
    CPUKVCache<Precision>::CPUKVCache(
        const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, const std::vector<int> &attention_devices,
        KVCacheLayoutMode layout_mode)
        : n_layers_(n_layers), batch_size_(batch_size), max_seq_len_(max_seq_len),
          n_kv_heads_(n_kv_heads), local_n_kv_heads_(local_n_kv_heads),
          kv_head_start_(kv_head_start), head_dim_(head_dim),
          kv_dim_(local_n_kv_heads * head_dim), is_sharded_(true),
          layout_mode_(layout_mode)
    {
        tensor_factory_ = std::make_unique<TensorFactory>(mpi_ctx);

        // Use provided device list or default to CPU
        if (attention_devices.size() >= static_cast<size_t>(n_layers_))
        {
            layer_devices_.reserve(n_layers_);
            for (int i = 0; i < n_layers_; ++i)
            {
                layer_devices_.push_back(deviceIdFromLegacyIndex(attention_devices[i]));
            }
        }
        else
        {
            layer_devices_.resize(n_layers_, DeviceId::cpu());
            LOG_WARN("CPUKVCache (sharded): attention_devices size mismatch, defaulting all layers to CPU");
        }

        entries_.resize(n_layers_);

// Parallelize layer initialization - each layer is independent
#pragma omp parallel for schedule(static) if (n_layers_ >= 4)
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            initialize_layer(layer, layer_devices_[layer]);
        }

        LOG_DEBUG("CPUKVCache (sharded): n_layers=" << n_layers_ << ", batch_size=" << batch_size_
                                                    << ", max_seq_len=" << max_seq_len_
                                                    << ", n_kv_heads=" << n_kv_heads_
                                                    << ", local_n_kv_heads=" << local_n_kv_heads_
                                                    << ", kv_head_start=" << kv_head_start_
                                                    << ", kv_dim=" << kv_dim_
                                                    << ", precision=" << static_cast<int>(Precision)
                                                    << ", layout_mode=" << (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR ? "HEAD_MAJOR" : "POSITION_MAJOR")
                                                    << ", per-layer devices");
    }

    template <ActivationPrecision Precision>
    void CPUKVCache<Precision>::initialize_layer(int layer, DeviceId device)
    {
        entries_[layer].resize(batch_size_);

// Parallelize sequence initialization for batched mode
// Note: allocate_tensor must be thread-safe (TensorFactory uses thread-local or atomic allocation)
#pragma omp parallel for schedule(static) if (batch_size_ >= 4)
        for (int seq_idx = 0; seq_idx < batch_size_; ++seq_idx)
        {
            auto &entry = entries_[layer][seq_idx];
            if (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
            {
                // HEAD_MAJOR: [n_kv_heads * max_seq_len, head_dim]
                // Logically: [n_kv_heads][max_seq_len][head_dim] - head-contiguous
                entry.K = allocate_tensor(local_n_kv_heads_ * max_seq_len_, head_dim_, device);
                entry.V = allocate_tensor(local_n_kv_heads_ * max_seq_len_, head_dim_, device);
            }
            else
            {
                // POSITION_MAJOR: [max_seq_len, n_kv_heads * head_dim]
                // Logically: [max_seq_len][n_kv_heads][head_dim] - position-contiguous
                entry.K = allocate_tensor(max_seq_len_, kv_dim_, device);
                entry.V = allocate_tensor(max_seq_len_, kv_dim_, device);
            }
            entry.cached_tokens = 0;
        }
    }

    // =========================================================================
    // Accessor Implementations
    // =========================================================================

    template <ActivationPrecision Precision>
    int CPUKVCache<Precision>::get_cached_tokens(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return 0;
        }
        return entries_[layer][seq_idx].cached_tokens;
    }

    // =========================================================================
    // Unified KV Access (preferred interface)
    // =========================================================================

    template <ActivationPrecision Precision>
    bool CPUKVCache<Precision>::get_kv(int layer, int seq_idx,
                                       ITensor **out_k, ITensor **out_v,
                                       int *out_kv_len)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = 0;
            return false;
        }

        auto &entry = entries_[layer][seq_idx];
        if (out_k)
            *out_k = entry.K.get();
        if (out_v)
            *out_v = entry.V.get();
        if (out_kv_len)
            *out_kv_len = entry.cached_tokens;
        return true;
    }

    template <ActivationPrecision Precision>
    bool CPUKVCache<Precision>::get_kv(int layer, int seq_idx,
                                       const ITensor **out_k, const ITensor **out_v,
                                       int *out_kv_len) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = 0;
            return false;
        }

        const auto &entry = entries_[layer][seq_idx];
        if (out_k)
            *out_k = entry.K.get();
        if (out_v)
            *out_v = entry.V.get();
        if (out_kv_len)
            *out_kv_len = entry.cached_tokens;
        return true;
    }

    // =========================================================================
    // Legacy Individual Accessors (deprecated - use get_kv() instead)
    // =========================================================================

    template <ActivationPrecision Precision>
    ITensor *CPUKVCache<Precision>::get_k(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return nullptr;
        }
        return entries_[layer][seq_idx].K.get();
    }

    template <ActivationPrecision Precision>
    const ITensor *CPUKVCache<Precision>::get_k(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return nullptr;
        }
        return entries_[layer][seq_idx].K.get();
    }

    template <ActivationPrecision Precision>
    ITensor *CPUKVCache<Precision>::get_v(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return nullptr;
        }
        return entries_[layer][seq_idx].V.get();
    }

    template <ActivationPrecision Precision>
    const ITensor *CPUKVCache<Precision>::get_v(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return nullptr;
        }
        return entries_[layer][seq_idx].V.get();
    }

    template <ActivationPrecision Precision>
    std::shared_ptr<typename CPUKVCache<Precision>::TensorT>
    CPUKVCache<Precision>::get_k_typed(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return nullptr;
        }
        const auto &entry = entries_[layer][seq_idx];
        if (!entry.K || entry.cached_tokens == 0)
        {
            return nullptr;
        }
        return entry.K;
    }

    template <ActivationPrecision Precision>
    std::shared_ptr<typename CPUKVCache<Precision>::TensorT>
    CPUKVCache<Precision>::get_v_typed(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return nullptr;
        }
        const auto &entry = entries_[layer][seq_idx];
        if (!entry.V || entry.cached_tokens == 0)
        {
            return nullptr;
        }
        return entry.V;
    }

    template <ActivationPrecision Precision>
    DeviceId CPUKVCache<Precision>::get_layer_device(int layer) const
    {
        if (layer < 0 || layer >= n_layers_)
        {
            return DeviceId::cpu();
        }
        return layer_devices_[layer];
    }

    // =========================================================================
    // Append Implementations
    // =========================================================================

    template <ActivationPrecision Precision>
    bool CPUKVCache<Precision>::append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v)
    {
        // Use tensor shape to determine token count
        if (!new_k || !new_v)
        {
            return false;
        }
        int num_tokens = static_cast<int>(new_k->shape()[0]);
        return append_kv(layer, seq_idx, new_k, new_v, num_tokens);
    }

    template <ActivationPrecision Precision>
    bool CPUKVCache<Precision>::append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v, int num_tokens)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_ERROR("CPUKVCache::append_kv: invalid layer=" << layer << " or seq_idx=" << seq_idx);
            return false;
        }

        // Type checking and casting
        const TensorT *typed_k = dynamic_cast<const TensorT *>(new_k);
        const TensorT *typed_v = dynamic_cast<const TensorT *>(new_v);

        if (!typed_k || !typed_v)
        {
            LOG_ERROR("CPUKVCache::append_kv: tensor type mismatch for layer " << layer);
            return false;
        }

        return append_kv_impl(layer, seq_idx, typed_k, typed_v, num_tokens);
    }

    template <ActivationPrecision Precision>
    bool CPUKVCache<Precision>::append_kv_typed(int layer, int seq_idx, const TensorT *new_k, const TensorT *new_v)
    {
        if (!new_k || !new_v)
        {
            return false;
        }
        int num_tokens = static_cast<int>(new_k->shape()[0]);
        return append_kv_impl(layer, seq_idx, new_k, new_v, num_tokens);
    }

    template <ActivationPrecision Precision>
    bool CPUKVCache<Precision>::append_kv_typed(int layer, int seq_idx, const TensorT *new_k, const TensorT *new_v, int num_tokens)
    {
        return append_kv_impl(layer, seq_idx, new_k, new_v, num_tokens);
    }

    template <ActivationPrecision Precision>
    bool CPUKVCache<Precision>::append_kv_impl(int layer, int seq_idx, const TensorT *new_k, const TensorT *new_v, int num_tokens)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return false;
        }

        auto &entry = entries_[layer][seq_idx];
        int current_tokens = entry.cached_tokens;

        // Check capacity
        if (current_tokens + num_tokens > max_seq_len_)
        {
            LOG_ERROR("CPUKVCache: capacity exceeded for layer " << layer << " seq " << seq_idx
                                                                 << " (current=" << current_tokens << " + new=" << num_tokens
                                                                 << " > max=" << max_seq_len_ << ")");
            return false;
        }

        // Copy new data to cache
        copy_append_data(entry.K.get(), new_k, current_tokens, num_tokens);
        copy_append_data(entry.V.get(), new_v, current_tokens, num_tokens);

        entry.cached_tokens = current_tokens + num_tokens;

        LOG_TRACE("CPUKVCache: layer " << layer << " seq " << seq_idx
                                       << " appended " << num_tokens << " tokens (total=" << entry.cached_tokens << ")");

        return true;
    }

    // =========================================================================
    // Clear Implementations
    // =========================================================================

    template <ActivationPrecision Precision>
    void CPUKVCache<Precision>::clear()
    {
// Parallelize clear across layers - simple scalar writes to independent locations
#pragma omp parallel for schedule(static) if (n_layers_ >= 4)
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            for (int seq_idx = 0; seq_idx < batch_size_; ++seq_idx)
            {
                entries_[layer][seq_idx].cached_tokens = 0;
            }
        }
        LOG_DEBUG("CPUKVCache: cleared all layers and sequences");
    }

    template <ActivationPrecision Precision>
    void CPUKVCache<Precision>::clear_sequence(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_)
        {
            return;
        }
        if (seq_idx < 0 || seq_idx >= batch_size_)
        {
            return;
        }
        entries_[layer][seq_idx].cached_tokens = 0;
        LOG_DEBUG("CPUKVCache: cleared layer " << layer << " sequence " << seq_idx);
    }

    template <ActivationPrecision Precision>
    void CPUKVCache<Precision>::clear_layer(int layer)
    {
        if (layer < 0 || layer >= n_layers_)
        {
            return;
        }

        for (int seq_idx = 0; seq_idx < batch_size_; ++seq_idx)
        {
            entries_[layer][seq_idx].cached_tokens = 0;
        }
        LOG_DEBUG("CPUKVCache: cleared layer " << layer);
    }

    // =========================================================================
    // Eviction Implementations
    // =========================================================================

    template <ActivationPrecision Precision>
    void CPUKVCache<Precision>::evict_oldest(int tokens_to_evict)
    {
// Parallelize eviction across sequences for batched inference
#pragma omp parallel for schedule(static) if (batch_size_ >= 4)
        for (int seq_idx = 0; seq_idx < batch_size_; ++seq_idx)
        {
            evict_oldest_from_sequence(seq_idx, tokens_to_evict);
        }
    }

    template <ActivationPrecision Precision>
    void CPUKVCache<Precision>::evict_oldest_from_sequence(int seq_idx, int tokens_to_evict)
    {
        if (seq_idx < 0 || seq_idx >= batch_size_ || tokens_to_evict <= 0)
        {
            return;
        }

        int total_evicted_local = 0;

// Parallelize eviction across layers - each layer's memmove is independent
#pragma omp parallel for schedule(static) reduction(+ : total_evicted_local) if (n_layers_ >= 4)
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            auto &entry = entries_[layer][seq_idx];
            if (entry.cached_tokens <= 0)
            {
                continue;
            }

            int actual_evict = std::min(tokens_to_evict, entry.cached_tokens);
            int tokens_to_keep = entry.cached_tokens - actual_evict;

            if (tokens_to_keep > 0)
            {
                shift_evict_data(entry.K.get(), actual_evict, tokens_to_keep);
                shift_evict_data(entry.V.get(), actual_evict, tokens_to_keep);
            }

            entry.cached_tokens = tokens_to_keep;
            total_evicted_local += actual_evict;
        }

        total_evicted_ += total_evicted_local;

        LOG_DEBUG("CPUKVCache: evicted " << tokens_to_evict << " tokens from sequence " << seq_idx);
    }

    // =========================================================================
    // Gather K/V Batched Implementation
    // =========================================================================

    template <ActivationPrecision Precision>
    int CPUKVCache<Precision>::gather_kv_batched(
        int layer, int num_sequences, TensorBase *out_k, TensorBase *out_v,
        std::vector<int> &out_kv_lens)
    {
        if (layer < 0 || layer >= n_layers_)
        {
            LOG_ERROR("CPUKVCache::gather_kv_batched: invalid layer=" << layer);
            return -1;
        }

        if (num_sequences > batch_size_)
        {
            LOG_ERROR("CPUKVCache::gather_kv_batched: num_sequences=" << num_sequences
                                                                      << " > batch_size=" << batch_size_);
            return -1;
        }

        // Type check output tensors
        TensorT *typed_k = dynamic_cast<TensorT *>(out_k);
        TensorT *typed_v = dynamic_cast<TensorT *>(out_v);
        if (!typed_k || !typed_v)
        {
            LOG_ERROR("CPUKVCache::gather_kv_batched: output tensor type mismatch");
            return -1;
        }

        // Resize output vector
        out_kv_lens.resize(num_sequences);

        // First pass: find max kv_len and populate out_kv_lens
        int max_kv_len = 0;
        for (int seq_idx = 0; seq_idx < num_sequences; ++seq_idx)
        {
            int kv_len = entries_[layer][seq_idx].cached_tokens;
            out_kv_lens[seq_idx] = kv_len;
            max_kv_len = std::max(max_kv_len, kv_len);
        }

        if (max_kv_len == 0)
        {
            LOG_DEBUG("CPUKVCache::gather_kv_batched: all sequences have 0 cached tokens");
            return 0;
        }

        // Validate output tensor dimensions
        size_t expected_rows = static_cast<size_t>(num_sequences) * max_kv_len;
        size_t expected_cols = kv_dim_;
        if (typed_k->shape()[0] < expected_rows || typed_k->shape()[1] != expected_cols)
        {
            LOG_ERROR("CPUKVCache::gather_kv_batched: out_k shape mismatch. Expected at least ["
                      << expected_rows << ", " << expected_cols << "], got ["
                      << typed_k->shape()[0] << ", " << typed_k->shape()[1] << "]");
            return -1;
        }
        if (typed_v->shape()[0] < expected_rows || typed_v->shape()[1] != expected_cols)
        {
            LOG_ERROR("CPUKVCache::gather_kv_batched: out_v shape mismatch");
            return -1;
        }

        // Second pass: copy data from each sequence to batched output
        // Parallelize across sequences - each copy is independent
#pragma omp parallel for schedule(static) if (num_sequences >= 4)
        for (int seq_idx = 0; seq_idx < num_sequences; ++seq_idx)
        {
            int kv_len = out_kv_lens[seq_idx];
            if (kv_len == 0)
            {
                continue;
            }

            const TensorT *src_k = entries_[layer][seq_idx].K.get();
            const TensorT *src_v = entries_[layer][seq_idx].V.get();

            // Copy K
            if constexpr (Precision == ActivationPrecision::FP32)
            {
                const float *src_data = src_k->data();
                float *dst_data = typed_k->mutable_data();
                size_t dst_offset = static_cast<size_t>(seq_idx) * max_kv_len * kv_dim_;
                size_t copy_bytes = static_cast<size_t>(kv_len) * kv_dim_ * sizeof(float);
                std::memcpy(dst_data + dst_offset, src_data, copy_bytes);
            }
            else if constexpr (Precision == ActivationPrecision::BF16)
            {
                const uint16_t *src_data = src_k->typed_data();
                uint16_t *dst_data = typed_k->mutable_typed_data();
                size_t dst_offset = static_cast<size_t>(seq_idx) * max_kv_len * kv_dim_;
                size_t copy_bytes = static_cast<size_t>(kv_len) * kv_dim_ * sizeof(uint16_t);
                std::memcpy(dst_data + dst_offset, src_data, copy_bytes);
            }
            else if constexpr (Precision == ActivationPrecision::FP16)
            {
                const uint16_t *src_data = src_k->typed_data();
                uint16_t *dst_data = typed_k->mutable_typed_data();
                size_t dst_offset = static_cast<size_t>(seq_idx) * max_kv_len * kv_dim_;
                size_t copy_bytes = static_cast<size_t>(kv_len) * kv_dim_ * sizeof(uint16_t);
                std::memcpy(dst_data + dst_offset, src_data, copy_bytes);
            }
            else if constexpr (Precision == ActivationPrecision::Q8_1)
            {
                const Q8_1Block *src_blocks = src_k->typed_data();
                Q8_1Block *dst_blocks = typed_k->mutable_typed_data();
                size_t blocks_per_row = (kv_dim_ + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                size_t dst_offset_blocks = static_cast<size_t>(seq_idx) * max_kv_len * blocks_per_row;
                size_t copy_blocks = static_cast<size_t>(kv_len) * blocks_per_row;
                size_t copy_bytes = copy_blocks * sizeof(Q8_1Block);
                std::memcpy(dst_blocks + dst_offset_blocks, src_blocks, copy_bytes);
            }
            else if constexpr (Precision == ActivationPrecision::Q16_1)
            {
                const Q16_1Block *src_blocks = src_k->typed_data();
                Q16_1Block *dst_blocks = typed_k->mutable_typed_data();
                size_t blocks_per_row = (kv_dim_ + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;
                size_t dst_offset_blocks = static_cast<size_t>(seq_idx) * max_kv_len * blocks_per_row;
                size_t copy_blocks = static_cast<size_t>(kv_len) * blocks_per_row;
                size_t copy_bytes = copy_blocks * sizeof(Q16_1Block);
                std::memcpy(dst_blocks + dst_offset_blocks, src_blocks, copy_bytes);
            }

            // Copy V
            if constexpr (Precision == ActivationPrecision::FP32)
            {
                const float *src_data = src_v->data();
                float *dst_data = typed_v->mutable_data();
                size_t dst_offset = static_cast<size_t>(seq_idx) * max_kv_len * kv_dim_;
                size_t copy_bytes = static_cast<size_t>(kv_len) * kv_dim_ * sizeof(float);
                std::memcpy(dst_data + dst_offset, src_data, copy_bytes);
            }
            else if constexpr (Precision == ActivationPrecision::BF16)
            {
                const uint16_t *src_data = src_v->typed_data();
                uint16_t *dst_data = typed_v->mutable_typed_data();
                size_t dst_offset = static_cast<size_t>(seq_idx) * max_kv_len * kv_dim_;
                size_t copy_bytes = static_cast<size_t>(kv_len) * kv_dim_ * sizeof(uint16_t);
                std::memcpy(dst_data + dst_offset, src_data, copy_bytes);
            }
            else if constexpr (Precision == ActivationPrecision::FP16)
            {
                const uint16_t *src_data = src_v->typed_data();
                uint16_t *dst_data = typed_v->mutable_typed_data();
                size_t dst_offset = static_cast<size_t>(seq_idx) * max_kv_len * kv_dim_;
                size_t copy_bytes = static_cast<size_t>(kv_len) * kv_dim_ * sizeof(uint16_t);
                std::memcpy(dst_data + dst_offset, src_data, copy_bytes);
            }
            else if constexpr (Precision == ActivationPrecision::Q8_1)
            {
                const Q8_1Block *src_blocks = src_v->typed_data();
                Q8_1Block *dst_blocks = typed_v->mutable_typed_data();
                size_t blocks_per_row = (kv_dim_ + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                size_t dst_offset_blocks = static_cast<size_t>(seq_idx) * max_kv_len * blocks_per_row;
                size_t copy_blocks = static_cast<size_t>(kv_len) * blocks_per_row;
                size_t copy_bytes = copy_blocks * sizeof(Q8_1Block);
                std::memcpy(dst_blocks + dst_offset_blocks, src_blocks, copy_bytes);
            }
            else if constexpr (Precision == ActivationPrecision::Q16_1)
            {
                const Q16_1Block *src_blocks = src_v->typed_data();
                Q16_1Block *dst_blocks = typed_v->mutable_typed_data();
                size_t blocks_per_row = (kv_dim_ + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;
                size_t dst_offset_blocks = static_cast<size_t>(seq_idx) * max_kv_len * blocks_per_row;
                size_t copy_blocks = static_cast<size_t>(kv_len) * blocks_per_row;
                size_t copy_bytes = copy_blocks * sizeof(Q16_1Block);
                std::memcpy(dst_blocks + dst_offset_blocks, src_blocks, copy_bytes);
            }
        }

        LOG_DEBUG("CPUKVCache::gather_kv_batched: layer " << layer
                                                          << ", gathered " << num_sequences << " sequences, max_kv_len=" << max_kv_len);

        return max_kv_len;
    }

    // =========================================================================
    // Explicit Template Instantiations
    // =========================================================================

    template class CPUKVCache<ActivationPrecision::FP32>;
    template class CPUKVCache<ActivationPrecision::BF16>;
    template class CPUKVCache<ActivationPrecision::FP16>;
    template class CPUKVCache<ActivationPrecision::Q8_1>;
    template class CPUKVCache<ActivationPrecision::Q16_1>;

    // =========================================================================
    // Factory Functions
    // =========================================================================

    std::unique_ptr<ICPUKVCache> createCPUKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        DeviceId device, KVCacheLayoutMode layout_mode)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<CPUKVCacheFP32>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device, layout_mode);
        case ActivationPrecision::BF16:
            return std::make_unique<CPUKVCacheBF16>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device, layout_mode);
        case ActivationPrecision::FP16:
            return std::make_unique<CPUKVCacheFP16>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device, layout_mode);
        case ActivationPrecision::Q8_1:
            return std::make_unique<CPUKVCacheQ8_1>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device, layout_mode);
        case ActivationPrecision::Q16_1:
            return std::make_unique<CPUKVCacheQ16_1>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device, layout_mode);
        default:
            LOG_ERROR("createCPUKVCache: unsupported precision " << static_cast<int>(precision));
            return nullptr;
        }
    }

    std::unique_ptr<ICPUKVCache> createCPUKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        const std::vector<int> &attention_devices, KVCacheLayoutMode layout_mode)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<CPUKVCacheFP32>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::BF16:
            return std::make_unique<CPUKVCacheBF16>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::FP16:
            return std::make_unique<CPUKVCacheFP16>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::Q8_1:
            return std::make_unique<CPUKVCacheQ8_1>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::Q16_1:
            return std::make_unique<CPUKVCacheQ16_1>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, attention_devices, layout_mode);
        default:
            LOG_ERROR("createCPUKVCache: unsupported precision " << static_cast<int>(precision));
            return nullptr;
        }
    }

    // =========================================================================
    // Sharded Factory Functions (for Tensor Parallelism)
    // =========================================================================

    std::unique_ptr<ICPUKVCache> createShardedCPUKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, DeviceId device, KVCacheLayoutMode layout_mode)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<CPUKVCacheFP32>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                    n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device, layout_mode);
        case ActivationPrecision::BF16:
            return std::make_unique<CPUKVCacheBF16>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                    n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device, layout_mode);
        case ActivationPrecision::FP16:
            return std::make_unique<CPUKVCacheFP16>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                    n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device, layout_mode);
        case ActivationPrecision::Q8_1:
            return std::make_unique<CPUKVCacheQ8_1>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                    n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device, layout_mode);
        case ActivationPrecision::Q16_1:
            return std::make_unique<CPUKVCacheQ16_1>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                     n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device, layout_mode);
        default:
            LOG_ERROR("createShardedCPUKVCache: unsupported precision " << static_cast<int>(precision));
            return nullptr;
        }
    }

    std::unique_ptr<ICPUKVCache> createShardedCPUKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim,
        const std::vector<int> &attention_devices, KVCacheLayoutMode layout_mode)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<CPUKVCacheFP32>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                    n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::BF16:
            return std::make_unique<CPUKVCacheBF16>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                    n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::FP16:
            return std::make_unique<CPUKVCacheFP16>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                    n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::Q8_1:
            return std::make_unique<CPUKVCacheQ8_1>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                    n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::Q16_1:
            return std::make_unique<CPUKVCacheQ16_1>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                     n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, attention_devices, layout_mode);
        default:
            LOG_ERROR("createShardedCPUKVCache: unsupported precision " << static_cast<int>(precision));
            return nullptr;
        }
    }

} // namespace llaminar2
