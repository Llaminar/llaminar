/**
 * @file GemmWeightCache.h
 * @brief Q8_0 block caching layer for integer GEMM operations
 *
 * This file provides a caching layer that prevents redundant decoding of quantized
 * weights during GEMM operations. Since all tensor types now have native decode_to_q8_0()
 * methods, this file focuses purely on the caching mechanism.
 *
 * Architecture:
 * 1. **Zero-copy path** (Q8_0 weights): Direct pointer access, no cache overhead
 * 2. **Cached path** (IQ4_NL/Q6_K/FP32/etc.): LRU cache to avoid redundant decoding
 *
 * The cache is critical for GEMM performance:
 *   for (ii in M_tiles) {           // Outer loop (OpenMP parallelized)
 *     for (jj in N_tiles) {
 *       for (kb in K_blocks) {
 *         // Without cache: decode weight(jj, kb) M_tiles times (catastrophic!)
 *         // With cache: decode once, reuse for all ii iterations
 *       }
 *     }
 *   }
 *
 * Memory vs Performance Trade-off:
 * - Q8_0 weights: Zero memory overhead, maximum performance
 * - Cached decode: Small cache overhead (~10-50MB), avoids redundant decode
 *
 * @author David Sanftenberg
 * @date November 2025
 */

#pragma once

#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <memory>
#include <mutex>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * @brief Base interface for providing Q8_0 blocks to GEMM kernels
             *
             * This interface abstracts whether blocks come from:
             * - Direct memory access (Q8_0 tensors - zero-copy)
             * - Cached decoded blocks (other formats - decode once, reuse)
             *
             * Thread Safety:
             * - All methods are thread-safe (cached implementations use mutex)
             * - Multiple threads can call get_q8_block() concurrently
             */
            class Q8_0BlockProvider
            {
            public:
                virtual ~Q8_0BlockProvider() = default;

                /**
                 * @brief Get Q8_0 block for GEMM operation
                 *
                 * Thread-safe: Can be called concurrently from OpenMP parallel regions.
                 *
                 * @param row_idx Row index in weight matrix
                 * @param k_block_offset K-dimension block offset (block index, not element index)
                 * @return Pointer to Q8_0Block (either cached or direct)
                 */
                virtual const Q8_0Block *get_q8_block(size_t row_idx, size_t k_block_offset) = 0;

                /**
                 * @brief Pre-decode multiple blocks in parallel (cache warming)
                 *
                 * Useful for pre-populating cache before GEMM loops to:
                 * 1. Parallelize decode overhead across multiple cores
                 * 2. Avoid cache contention during GEMM execution
                 * 3. Amortize decode cost when reusing weights
                 *
                 * Thread-safe: Uses OpenMP parallelization internally.
                 * No-op for zero-copy providers (Q8_0 weights).
                 *
                 * @param row_start First row to decode
                 * @param row_count Number of rows to decode
                 * @param k_block_start First K-block to decode
                 * @param k_block_count Number of K-blocks to decode
                 */
                virtual void warmup_cache(size_t row_start, size_t row_count,
                                          size_t k_block_start, size_t k_block_count)
                {
                    // Default: sequential decode (overridden in cached providers)
                    for (size_t r = row_start; r < row_start + row_count; ++r)
                    {
                        for (size_t kb = k_block_start; kb < k_block_start + k_block_count; ++kb)
                        {
                            get_q8_block(r, kb);
                        }
                    }
                }

                /**
                 * @brief Check if this provider uses zero-copy direct access
                 *
                 * @return true if Q8_0 weights (no decode needed), false otherwise
                 */
                virtual bool is_zero_copy() const = 0;

                /**
                 * @brief Get block size (number of elements per block)
                 *
                 * All weight formats must align to 32-element blocks (Q8_0 standard).
                 *
                 * @return Always 32 (Q8_0 block size)
                 */
                virtual size_t block_size() const { return 32; }

                /**
                 * @brief Get number of K-dimension blocks in weight matrix
                 */
                virtual size_t k_blocks() const = 0;

                /**
                 * @brief Get number of rows in weight matrix
                 */
                virtual size_t num_rows() const = 0;

                /**
                 * @brief Clear decode cache (if applicable)
                 *
                 * Called between GEMM operations to free cache memory.
                 * No-op for zero-copy providers.
                 */
                virtual void clear_cache() {}
            };

            // ========== Cache Infrastructure ==========

            /**
             * @brief LRU cache for decoded Q8_0 blocks
             *
             * Implements a thread-safe LRU cache to avoid redundant decoding across M-tiles.
             * Cache key: (row_idx, k_block_offset) → decoded Q8_0Block
             *
             * Cache sizing strategy:
             * - Default: 1024 entries (~34KB with 32-byte Q8_0Block)
             * - Adjustable via constructor for different memory/performance trade-offs
             *
             * Eviction policy: LRU (least recently used)
             * - Typical GEMM access pattern has good temporal locality
             * - Same (row, k_block) accessed for all M-tiles before moving to next
             */
            class WeightDecodeCache
            {
            public:
                explicit WeightDecodeCache(size_t max_entries = 1024)
                    : max_entries_(max_entries)
                {
                    cache_.reserve(max_entries);
                }

                /**
                 * @brief Get cached block or decode and cache
                 *
                 * @param key Cache key (row_idx, k_block_offset)
                 * @param decoder Function to decode block if not cached
                 * @return Pointer to cached Q8_0Block
                 */
                const Q8_0Block *get_or_decode(
                    uint64_t key,
                    const std::function<void(Q8_0Block *)> &decoder)
                {
                    std::lock_guard<std::mutex> lock(mutex_);

                    auto it = cache_.find(key);
                    if (it != cache_.end())
                    {
                        // Cache hit: update LRU timestamp
                        it->second.timestamp = ++current_timestamp_;
                        return &it->second.block;
                    }

                    // Cache miss: evict LRU entry if at capacity
                    if (cache_.size() >= max_entries_)
                    {
                        evict_lru();
                    }

                    // Decode and insert
                    CacheEntry entry;
                    decoder(&entry.block);
                    entry.timestamp = ++current_timestamp_;
                    auto result = cache_.emplace(key, entry);
                    return &result.first->second.block;
                }

                void clear()
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    cache_.clear();
                    current_timestamp_ = 0;
                }

                size_t size() const
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    return cache_.size();
                }

            private:
                struct CacheEntry
                {
                    Q8_0Block block;
                    uint64_t timestamp;
                };

                void evict_lru()
                {
                    // Find oldest entry
                    auto oldest = cache_.begin();
                    uint64_t min_timestamp = oldest->second.timestamp;

                    for (auto it = cache_.begin(); it != cache_.end(); ++it)
                    {
                        if (it->second.timestamp < min_timestamp)
                        {
                            min_timestamp = it->second.timestamp;
                            oldest = it;
                        }
                    }

                    cache_.erase(oldest);
                }

                std::unordered_map<uint64_t, CacheEntry> cache_;
                size_t max_entries_;
                uint64_t current_timestamp_ = 0;
                mutable std::mutex mutex_;
            };

            /**
             * @brief Helper to create cache key from (row, k_block)
             */
            inline uint64_t make_cache_key(size_t row_idx, size_t k_block_offset)
            {
                return (static_cast<uint64_t>(row_idx) << 32) | static_cast<uint64_t>(k_block_offset);
            }

            // ========== Concrete Implementations ==========

            /**
             * @brief Zero-copy provider for Q8_0 weights (no decoding needed)
             *
             * This is the optimal path: weights are already in Q8_0 format,
             * so we just return direct pointers with no cache overhead.
             */
            class Q8_0DirectProvider : public Q8_0BlockProvider
            {
            public:
                explicit Q8_0DirectProvider(const Q8_0Tensor *tensor)
                    : tensor_(tensor)
                {
                    const auto &shape = tensor->shape();
                    num_rows_ = shape[0];
                    k_elements_ = shape[1];
                    k_blocks_ = (k_elements_ + 31) / 32;
                }

                const Q8_0Block *get_q8_block(size_t row_idx, size_t k_block_offset) override
                {
                    // Zero-copy: return direct pointer (no cache, no allocation)
                    return static_cast<const Q8_0Block *>(tensor_->get_raw_block_at(row_idx, k_block_offset));
                }

                void warmup_cache(size_t /*row_start*/, size_t /*row_count*/,
                                  size_t /*k_block_start*/, size_t /*k_block_count*/) override
                {
                    // No-op: Zero-copy provider doesn't need cache warming
                }

                bool is_zero_copy() const override { return true; }
                size_t k_blocks() const override { return k_blocks_; }
                size_t num_rows() const override { return num_rows_; }

            private:
                const Q8_0Tensor *tensor_;
                size_t num_rows_;
                size_t k_elements_;
                size_t k_blocks_;
            };

            /**
             * @brief Template-based cached provider for quantized tensors
             *
             * This generic implementation works for quantized tensor types that implement
             * the IQ8_0Decodable interface (IQ4_NL, Q6_K, Q4_K, Q8_0, etc.).
             *
             * Decoding is delegated to the tensor's native implementation, which uses
             * SIMD-optimized paths (AVX512/AVX2/scalar).
             *
             * Thread-safe: Cache access is protected by mutex.
             *
             * @tparam TensorType Quantized tensor class implementing IQ8_0Decodable interface
             */
            template <typename TensorType>
            class CachedQ8Provider : public Q8_0BlockProvider
            {
                // Static assertion to ensure TensorType implements IQ8_0Decodable
                static_assert(std::is_base_of<IQ8_0Decodable, TensorType>::value,
                              "CachedQ8Provider requires TensorType to implement IQ8_0Decodable interface");

            public:
                explicit CachedQ8Provider(const TensorType *tensor, size_t cache_size = 1024)
                    : tensor_(tensor), cache_(cache_size)
                {
                    const auto &shape = tensor->shape();
                    num_rows_ = shape[0];
                    k_elements_ = shape[1];
                    k_blocks_ = (k_elements_ + 31) / 32;
                }

                const Q8_0Block *get_q8_block(size_t row_idx, size_t k_block_offset) override
                {
                    uint64_t key = make_cache_key(row_idx, k_block_offset);

                    return cache_.get_or_decode(key, [this, row_idx, k_block_offset](Q8_0Block *output)
                                                {
                        // Delegate decode to tensor's SIMD-optimized implementation
                        tensor_->decode_to_q8_0(row_idx, k_block_offset, output); });
                }

                void warmup_cache(size_t row_start, size_t row_count,
                                  size_t k_block_start, size_t k_block_count) override
                {
                    // OpenMP-parallelized cache warming
                    // Each thread decodes different blocks in parallel
                    const size_t total_blocks = row_count * k_block_count;

#pragma omp parallel for schedule(dynamic, 8) if (total_blocks > 64)
                    for (size_t idx = 0; idx < total_blocks; ++idx)
                    {
                        const size_t r = row_start + (idx / k_block_count);
                        const size_t kb = k_block_start + (idx % k_block_count);
                        get_q8_block(r, kb); // Decode and cache
                    }
                }

                bool is_zero_copy() const override { return false; }
                size_t k_blocks() const override { return k_blocks_; }
                size_t num_rows() const override { return num_rows_; }
                void clear_cache() override { cache_.clear(); }

            private:
                const TensorType *tensor_;
                size_t num_rows_;
                size_t k_elements_;
                size_t k_blocks_;
                WeightDecodeCache cache_;
            };

            /**
             * @brief Factory function to create appropriate provider based on tensor type
             *
             * Automatically detects tensor format and returns:
             * - Zero-copy provider for Q8_0 tensors (optimal path)
             * - Cached provider for other formats (decode once, reuse across M-tiles)
             *
             * @param tensor Input tensor (any quantized format)
             * @param cache_size LRU cache size for non-Q8_0 formats (default: 1024 entries ≈ 34KB)
             * @return Unique pointer to appropriate provider
             *
             * Usage:
             * ```cpp
             * auto provider = createWeightProvider(weight_tensor);
             * const Q8_0Block* block = provider->get_q8_block(row, k_block);
             * ```
             */
            inline std::unique_ptr<Q8_0BlockProvider> createWeightProvider(
                const TensorBase *tensor,
                size_t cache_size = 1024)
            {
                if (!tensor)
                    return nullptr;

                // Q8_0: Zero-copy path (optimal - no cache needed)
                if (tensor->native_type() == TensorType::Q8_0)
                {
                    return std::make_unique<Q8_0DirectProvider>(
                        static_cast<const Q8_0Tensor *>(tensor));
                }

                // All other formats: Use template-based cached provider
                // The tensor's decode_to_q8_0() method handles format-specific logic
                switch (tensor->native_type())
                {
                case TensorType::IQ4_NL:
                    return std::make_unique<CachedQ8Provider<IQ4_NLTensor>>(
                        static_cast<const IQ4_NLTensor *>(tensor), cache_size);

                case TensorType::Q6_K:
                    return std::make_unique<CachedQ8Provider<Q6_KTensor>>(
                        static_cast<const Q6_KTensor *>(tensor), cache_size);

                case TensorType::Q4_K:
                    return std::make_unique<CachedQ8Provider<Q4_KTensor>>(
                        static_cast<const Q4_KTensor *>(tensor), cache_size);

                case TensorType::Q5_K:
                    return std::make_unique<CachedQ8Provider<Q5_KTensor>>(
                        static_cast<const Q5_KTensor *>(tensor), cache_size);

                case TensorType::Q2_K:
                    return std::make_unique<CachedQ8Provider<Q2_KTensor>>(
                        static_cast<const Q2_KTensor *>(tensor), cache_size);

                case TensorType::Q3_K:
                    return std::make_unique<CachedQ8Provider<Q3_KTensor>>(
                        static_cast<const Q3_KTensor *>(tensor), cache_size);

                case TensorType::Q8_K:
                    return std::make_unique<CachedQ8Provider<Q8_KTensor>>(
                        static_cast<const Q8_KTensor *>(tensor), cache_size);

                case TensorType::Q4_0:
                    return std::make_unique<CachedQ8Provider<Q4_0Tensor>>(
                        static_cast<const Q4_0Tensor *>(tensor), cache_size);

                case TensorType::Q4_1:
                    return std::make_unique<CachedQ8Provider<Q4_1Tensor>>(
                        static_cast<const Q4_1Tensor *>(tensor), cache_size);

                case TensorType::Q5_0:
                    return std::make_unique<CachedQ8Provider<Q5_0Tensor>>(
                        static_cast<const Q5_0Tensor *>(tensor), cache_size);

                case TensorType::Q5_1:
                    return std::make_unique<CachedQ8Provider<Q5_1Tensor>>(
                        static_cast<const Q5_1Tensor *>(tensor), cache_size);

                case TensorType::FP32:
                    return std::make_unique<CachedQ8Provider<FP32Tensor>>(
                        static_cast<const FP32Tensor *>(tensor), cache_size);

                case TensorType::FP16:
                    return std::make_unique<CachedQ8Provider<FP16Tensor>>(
                        static_cast<const FP16Tensor *>(tensor), cache_size);

                // IQ series formats
                case TensorType::IQ4_XS:
                    return std::make_unique<CachedQ8Provider<IQ4_XSTensor>>(
                        static_cast<const IQ4_XSTensor *>(tensor), cache_size);

                case TensorType::IQ2_XXS:
                    return std::make_unique<CachedQ8Provider<IQ2_XXSTensor>>(
                        static_cast<const IQ2_XXSTensor *>(tensor), cache_size);

                case TensorType::IQ2_XS:
                    return std::make_unique<CachedQ8Provider<IQ2_XSTensor>>(
                        static_cast<const IQ2_XSTensor *>(tensor), cache_size);

                case TensorType::IQ3_XXS:
                    return std::make_unique<CachedQ8Provider<IQ3_XXSTensor>>(
                        static_cast<const IQ3_XXSTensor *>(tensor), cache_size);

                case TensorType::IQ2_S:
                    return std::make_unique<CachedQ8Provider<IQ2_STensor>>(
                        static_cast<const IQ2_STensor *>(tensor), cache_size);

                case TensorType::IQ3_S:
                    return std::make_unique<CachedQ8Provider<IQ3_STensor>>(
                        static_cast<const IQ3_STensor *>(tensor), cache_size);

                case TensorType::IQ1_S:
                    return std::make_unique<CachedQ8Provider<IQ1_STensor>>(
                        static_cast<const IQ1_STensor *>(tensor), cache_size);

                default:
                    return nullptr; // Unsupported format
                }
            }

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
