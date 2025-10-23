// QuantSlabCache.h
// Provides caching for decoded quantization slabs (K x N_tile slices) to amortize
// decode cost across multiple matmul invocations. Intended replacement for the
// fused decode+GEMM path which has been retired due to poor performance.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include <list>
#include <mutex>
#include <functional>

// Forward declare QuantizedTensor to avoid heavy include
namespace llaminar
{
    class QuantizedTensor;
}

#include "../utils/BFloat16.h" // bfloat16 helper (placed before namespace to avoid double nested namespace)
#include "../Logger.h"         // For LOG_DEBUG in template functions
#include <chrono>

namespace llaminar
{

    // ===========================
    // Generalized Tensor Cache
    // ===========================

    /**
     * @brief Type of cached decoded data
     */
    enum class CachedDataType : uint8_t
    {
        FP32 = 0,
        BF16 = 1
    };

    /**
     * @brief Cache key for full tensor decode (not column slices)
     *
     * Used for activation tensors that need full decode on access.
     * Separate from QuantSlabKey which is for weight column slices.
     */
    struct TensorCacheKey
    {
        const void *tensor_ptr;   // Tensor identity (pointer address)
        CachedDataType data_type; // FP32 or BF16
        size_t element_count;     // For validation

        bool operator==(const TensorCacheKey &other) const noexcept
        {
            return tensor_ptr == other.tensor_ptr &&
                   data_type == other.data_type &&
                   element_count == other.element_count;
        }
    };

    /**
     * @brief Hasher for TensorCacheKey
     */
    struct TensorCacheKeyHasher
    {
        size_t operator()(const TensorCacheKey &k) const noexcept
        {
            auto h1 = std::hash<const void *>()(k.tensor_ptr);
            auto h2 = static_cast<size_t>(k.data_type);
            auto h3 = std::hash<size_t>()(k.element_count);
            return ((h1 ^ (h2 << 1)) >> 1) ^ (h3 << 1);
        }
    };

    /**
     * @brief Cached decoded tensor data
     *
     * Stores decoded tensor in type-erased format with metadata.
     * Used for pull-through cache pattern on TensorBase::data_fp32/data_bf16.
     */
    struct CachedTensorData
    {
        CachedDataType type;
        size_t element_count;
        size_t bytes;
        std::vector<uint8_t> data; // Type-erased storage
        std::chrono::steady_clock::time_point last_access;

        template <typename T>
        const T *typed_data() const
        {
            return reinterpret_cast<const T *>(data.data());
        }

        size_t memory_bytes() const { return data.size(); }
    };

    // ===========================
    // Weight Column Slab Cache (Existing)
    // ===========================

    struct QuantSlabKey
    {
        const void *weight_ptr; // identity of backing quantized raw buffer
        size_t col_start;       // starting column in logical weight matrix
        size_t col_count;       // number of columns covered by the slab

        bool operator==(const QuantSlabKey &other) const noexcept
        {
            return weight_ptr == other.weight_ptr && col_start == other.col_start && col_count == other.col_count;
        }
    };

    struct QuantSlabKeyHasher
    {
        size_t operator()(const QuantSlabKey &k) const noexcept
        {
            // mix pointer and columns
            auto h1 = std::hash<const void *>()(k.weight_ptr);
            auto h2 = std::hash<size_t>()(k.col_start);
            auto h3 = std::hash<size_t>()(k.col_count);
            return ((h1 ^ (h2 << 1)) >> 1) ^ (h3 << 1);
        }
    };

    // Slab stores decoded weights in BF16 (bfloat16) to preserve dynamic range
    // while gaining bandwidth savings. Expansion to FP32 performed at matmul time.
    struct QuantSlab
    {
        size_t k{0};
        size_t n{0};
        std::vector<bfloat16> data;
    };

    class QuantSlabCache
    {
    public:
        static QuantSlabCache &instance();

        // ===========================
        // Full Tensor Decode Cache (NEW)
        // ===========================

        /**
         * @brief Get or decode a full tensor with pull-through caching
         *
         * @tparam T Data type (float or bfloat16)
         * @param tensor_ptr Pointer to source tensor (identity)
         * @param element_count Number of elements in tensor
         * @param type Requested cached data type (FP32 or BF16)
         * @param decode_fn Function to decode tensor: void(T* dst)
         * @return const T* Pointer to cached decoded data (valid until eviction)
         *
         * This implements the pull-through cache pattern:
         * - Cache hit: Return cached pointer, update LRU
         * - Cache miss: Allocate, decode via decode_fn, insert, return pointer
         */
        template <typename T>
        const T *getOrDecodeTensor(
            const void *tensor_ptr,
            size_t element_count,
            CachedDataType type,
            const std::function<void(T *)> &decode_fn);

        /**
         * @brief Invalidate all cache entries for a specific tensor
         *
         * Call this when a tensor is destroyed or its data mutated.
         */
        void invalidateTensor(const void *tensor_ptr);

        /**
         * @brief Get cache statistics for diagnostics
         */
        struct CacheStats
        {
            size_t tensor_cache_hits = 0;
            size_t tensor_cache_misses = 0;
            size_t tensor_evictions = 0;
            size_t slab_cache_hits = 0;
            size_t slab_cache_misses = 0;
            size_t slab_evictions = 0;
            size_t total_cached_bytes = 0;

            double tensor_hit_rate() const
            {
                size_t total = tensor_cache_hits + tensor_cache_misses;
                return total > 0 ? static_cast<double>(tensor_cache_hits) / total : 0.0;
            }

            double slab_hit_rate() const
            {
                size_t total = slab_cache_hits + slab_cache_misses;
                return total > 0 ? static_cast<double>(slab_cache_hits) / total : 0.0;
            }
        };

        CacheStats getStats() const;
        void resetStats();

        // ===========================
        // Weight Column Slab Cache (EXISTING)
        // ===========================

        // Retrieve an existing slab or decode a new one from a QuantizedTensor.
        // Returns shared_ptr to slab; sets reused flag via return bool when using overload below.
        std::shared_ptr<QuantSlab> getOrDecode(const QuantSlabKey &key, size_t k, size_t n,
                                               const std::function<void(QuantSlab &)> &decode_fn);

        // Convenience overload matching benchmark expectations: decodes a slab covering
        // columns [col_start, col_start+col_count) for the given QuantizedTensor. If reuse_allowed
        // is false, forces a fresh decode (overwriting any existing cache entry). Returns true if
        // cached reuse occurred (and decode was skipped).
        bool getOrDecode(const QuantizedTensor &tensor, size_t col_start, size_t col_count,
                         QuantSlab &out_slab, bool reuse_allowed);

        void setCapacityBytes(size_t bytes);
        size_t capacityBytes() const { return capacity_bytes_; }
        size_t currentBytes() const { return current_bytes_; }
        size_t size() const { return map_.size(); }

        void clear();

    private:
        QuantSlabCache();
        void touch(std::list<QuantSlabKey>::iterator it);
        void enforceCapacity();

        // LRU eviction for tensor cache
        void touchTensor(std::list<TensorCacheKey>::iterator it);
        void evictLRUTensor();

        // ===========================
        // Weight Slab Cache (Existing)
        // ===========================

        struct Entry
        {
            std::shared_ptr<QuantSlab> slab;
            std::list<QuantSlabKey>::iterator lru_iter;
        };

        std::unordered_map<QuantSlabKey, Entry, QuantSlabKeyHasher> map_;
        std::list<QuantSlabKey> lru_list_;

        // ===========================
        // Tensor Cache (NEW)
        // ===========================

        struct TensorEntry
        {
            CachedTensorData cached_data;
            std::list<TensorCacheKey>::iterator lru_iter;
        };

        std::unordered_map<TensorCacheKey, TensorEntry, TensorCacheKeyHasher> tensor_map_;
        std::list<TensorCacheKey> tensor_lru_list_;

        // ===========================
        // Shared State
        // ===========================

        size_t capacity_bytes_ = 4096ULL * 1024 * 1024; // default 4GB (shared by both caches) - increased to prevent eviction during active operations
        size_t current_bytes_ = 0;                      // total bytes (slabs + tensors)
        mutable std::mutex mutex_;                      // mutable for const methods like getStats()

        // Statistics
        mutable size_t tensor_cache_hits_ = 0;
        mutable size_t tensor_cache_misses_ = 0;
        mutable size_t tensor_evictions_ = 0;
        mutable size_t slab_cache_hits_ = 0;
        mutable size_t slab_cache_misses_ = 0;
        mutable size_t slab_evictions_ = 0;
    };

    // ===========================
    // Template Implementation
    // ===========================

    /**
     * @brief Template implementation of getOrDecodeTensor
     *
     * Must be in header for template instantiation.
     */
    template <typename T>
    const T *QuantSlabCache::getOrDecodeTensor(
        const void *tensor_ptr,
        size_t element_count,
        CachedDataType type,
        const std::function<void(T *)> &decode_fn)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        TensorCacheKey key{tensor_ptr, type, element_count};

        LOG_DEBUG("QuantSlabCache::getOrDecodeTensor - tensor=" << tensor_ptr
                                                                << " type=" << (int)type << " elements=" << element_count);

        // Cache hit?
        auto it = tensor_map_.find(key);
        if (it != tensor_map_.end())
        {
            // Update LRU
            touchTensor(it->second.lru_iter);
            it->second.cached_data.last_access = std::chrono::steady_clock::now();
            tensor_cache_hits_++;
            LOG_DEBUG("QuantSlabCache CACHE HIT - returning ptr=" << (void *)it->second.cached_data.template typed_data<T>());
            return it->second.cached_data.template typed_data<T>();
        }

        // Cache miss: Allocate and decode
        tensor_cache_misses_++;

        size_t bytes_needed = element_count * sizeof(T);
        LOG_DEBUG("QuantSlabCache CACHE MISS - will allocate " << bytes_needed << " bytes");

        // Evict if needed (both tensor and slab caches share capacity)
        while (current_bytes_ + bytes_needed > capacity_bytes_ &&
               (!tensor_map_.empty() || !map_.empty()))
        {

            // Determine which cache has the oldest entry
            bool evict_tensor = false;
            if (!tensor_lru_list_.empty() && map_.empty())
            {
                evict_tensor = true;
            }
            else if (tensor_lru_list_.empty() && !map_.empty())
            {
                evict_tensor = false;
            }
            else if (!tensor_lru_list_.empty() && !map_.empty())
            {
                // Both caches have entries - compare timestamps
                auto tensor_oldest_key = tensor_lru_list_.back();
                auto tensor_entry_it = tensor_map_.find(tensor_oldest_key);

                auto slab_oldest_key = lru_list_.back();
                auto slab_entry_it = map_.find(slab_oldest_key);

                // Evict whichever has older last_access time
                // For slabs, we don't track access time, so just alternate or prefer tensor
                evict_tensor = true; // Prefer evicting tensor cache (simpler for now)
            }
            else
            {
                break; // Both empty
            }

            if (evict_tensor)
            {
                evictLRUTensor();
            }
            else
            {
                // Evict from slab cache
                auto oldest_slab_key = lru_list_.back();
                auto slab_it = map_.find(oldest_slab_key);
                if (slab_it != map_.end())
                {
                    current_bytes_ -= slab_it->second.slab->data.size() * sizeof(bfloat16);
                    lru_list_.erase(slab_it->second.lru_iter);
                    map_.erase(slab_it);
                    slab_evictions_++;
                }
                lru_list_.pop_back();
            }
        }

        // Create cached entry
        CachedTensorData cached;
        cached.type = type;
        cached.element_count = element_count;
        cached.bytes = bytes_needed;

        cached.data.resize(bytes_needed);
        cached.last_access = std::chrono::steady_clock::now();

        // Decode into cache
        decode_fn(reinterpret_cast<T *>(cached.data.data()));

        // Insert into LRU list (front = most recent)
        tensor_lru_list_.push_front(key);

        // Insert into map
        TensorEntry entry;
        entry.cached_data = std::move(cached);
        entry.lru_iter = tensor_lru_list_.begin();

        auto [inserted_it, success] = tensor_map_.emplace(key, std::move(entry));
        current_bytes_ += bytes_needed;

        return inserted_it->second.cached_data.template typed_data<T>();
    }

} // namespace llaminar