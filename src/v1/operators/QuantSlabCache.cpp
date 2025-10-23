#include "QuantSlabCache.h"
#include "../tensors/TensorFactory.h"
#include <algorithm>
#include <cstring>
#include <functional>

namespace llaminar
{

    QuantSlabCache &QuantSlabCache::instance()
    {
        static QuantSlabCache cache;
        return cache;
    }

    QuantSlabCache::QuantSlabCache()
    {
        // capacity may later be overridden by environment snapshot (DebugEnv)
    }

    void QuantSlabCache::touch(std::list<QuantSlabKey>::iterator it)
    {
        // move accessed key to front of LRU list
        lru_list_.splice(lru_list_.begin(), lru_list_, it);
    }

    void QuantSlabCache::setCapacityBytes(size_t bytes)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        capacity_bytes_ = bytes;
        enforceCapacity();
    }

    void QuantSlabCache::enforceCapacity()
    {
        while (current_bytes_ > capacity_bytes_ && !lru_list_.empty())
        {
            auto last_it = std::prev(lru_list_.end());
            auto key = *last_it;
            auto map_it = map_.find(key);
            if (map_it != map_.end())
            {
                current_bytes_ -= map_it->second.slab->data.size() * sizeof(bfloat16);
                map_.erase(map_it);
                slab_evictions_++;
            }
            lru_list_.erase(last_it);
        }
    }

    std::shared_ptr<QuantSlab> QuantSlabCache::getOrDecode(const QuantSlabKey &key, size_t k, size_t n,
                                                           const std::function<void(QuantSlab &)> &decode_fn)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = map_.find(key);
        if (it != map_.end())
        {
            touch(it->second.lru_iter);
            slab_cache_hits_++;
            return it->second.slab;
        }
        slab_cache_misses_++;
        auto slab = std::make_shared<QuantSlab>();
        slab->k = k;
        slab->n = n;
        slab->data.resize(k * n);
        decode_fn(*slab);
        lru_list_.push_front(key);
        Entry e;
        e.slab = slab;
        e.lru_iter = lru_list_.begin();
        map_.emplace(key, std::move(e));
        current_bytes_ += slab->data.size() * sizeof(bfloat16);
        enforceCapacity();
        return slab;
    }

    bool QuantSlabCache::getOrDecode(const QuantizedTensor &tensor, size_t col_start, size_t col_count,
                                     QuantSlab &out_slab, bool reuse_allowed)
    {
        const auto &layout = tensor.layout();
        size_t k = layout.original_shape[0];
        size_t n = col_count;
        QuantSlabKey key{tensor.raw(), col_start, col_count};
        if (reuse_allowed)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = map_.find(key);
            if (it != map_.end())
            {
                touch(it->second.lru_iter);
                slab_cache_hits_++;
                out_slab = *it->second.slab; // copy metadata + data
                return true;                 // reused
            }
        }
        slab_cache_misses_++;
        // Perform full decode of requested column span
        out_slab.k = k;
        out_slab.n = n;
        out_slab.data.resize(k * n);
        int block_elems = layout.block_desc.elements_per_block;
        int blocks_per_row = (layout.original_shape[1] + block_elems - 1) / block_elems;
        std::vector<float> tmp(block_elems);
        for (size_t row = 0; row < k; ++row)
        {
            for (int b = 0; b < blocks_per_row; ++b)
            {
                int col0 = b * block_elems;
                int span = std::min(block_elems, layout.original_shape[1] - col0);
                if (col0 + span <= (int)col_start)
                    continue; // before requested span
                if (col0 >= (int)(col_start + col_count))
                    break; // beyond requested span
                size_t bi = row * blocks_per_row + b;
                tensor.decodeBlock(bi, tmp.data());
                int write_start = std::max<int>(col0, col_start);
                int write_end = std::min<int>(col0 + span, col_start + col_count);
                int local_span = write_end - write_start;
                if (local_span <= 0)
                    continue;
                bfloat16 *dst = out_slab.data.data() + row * n + (write_start - col_start);
                const float *src = tmp.data() + (write_start - col0);
                for (int i = 0; i < local_span; ++i)
                    dst[i] = bfloat16::from_float(src[i]);
            }
        }
        if (reuse_allowed)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            lru_list_.push_front(key);
            Entry e;
            auto slab_ptr = std::make_shared<QuantSlab>(out_slab);
            e.slab = slab_ptr;
            e.lru_iter = lru_list_.begin();
            map_[key] = e;
            current_bytes_ += out_slab.data.size() * sizeof(bfloat16);
            enforceCapacity();
        }
        return false; // not reused
    }

    void QuantSlabCache::clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        map_.clear();
        lru_list_.clear();
        tensor_map_.clear();
        tensor_lru_list_.clear();
        current_bytes_ = 0;
    }

    // ===========================
    // Tensor Cache Methods (NEW)
    // ===========================

    void QuantSlabCache::touchTensor(std::list<TensorCacheKey>::iterator it)
    {
        // Move accessed key to front of LRU list (most recent)
        tensor_lru_list_.splice(tensor_lru_list_.begin(), tensor_lru_list_, it);
    }

    void QuantSlabCache::evictLRUTensor()
    {
        if (tensor_lru_list_.empty())
            return;

        auto last_it = std::prev(tensor_lru_list_.end());
        auto key = *last_it;
        auto map_it = tensor_map_.find(key);

        if (map_it != tensor_map_.end())
        {
            current_bytes_ -= map_it->second.cached_data.memory_bytes();
            tensor_map_.erase(map_it);
            tensor_evictions_++;
        }

        tensor_lru_list_.erase(last_it);
    }

    void QuantSlabCache::invalidateTensor(const void *tensor_ptr)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Remove all entries for this tensor (both FP32 and BF16)
        auto it = tensor_map_.begin();
        while (it != tensor_map_.end())
        {
            if (it->first.tensor_ptr == tensor_ptr)
            {
                current_bytes_ -= it->second.cached_data.memory_bytes();
                tensor_lru_list_.erase(it->second.lru_iter);
                it = tensor_map_.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    QuantSlabCache::CacheStats QuantSlabCache::getStats() const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        CacheStats stats;
        stats.tensor_cache_hits = tensor_cache_hits_;
        stats.tensor_cache_misses = tensor_cache_misses_;
        stats.tensor_evictions = tensor_evictions_;
        stats.slab_cache_hits = slab_cache_hits_;
        stats.slab_cache_misses = slab_cache_misses_;
        stats.slab_evictions = slab_evictions_;
        stats.total_cached_bytes = current_bytes_;

        return stats;
    }

    void QuantSlabCache::resetStats()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        tensor_cache_hits_ = 0;
        tensor_cache_misses_ = 0;
        tensor_evictions_ = 0;
        slab_cache_hits_ = 0;
        slab_cache_misses_ = 0;
        slab_evictions_ = 0;
    }

} // namespace llaminar