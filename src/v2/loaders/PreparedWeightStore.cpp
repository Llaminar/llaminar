#include "PreparedWeightStore.h"

#include "../tensors/TensorClasses.h"
#include "../tensors/TensorKernels.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace llaminar2
{
    namespace
    {
        void validateBindingForStore(const WeightBinding &binding, ModelContextId model_id, PreparedWeightKind kind)
        {
            if (binding.binding_id == 0)
                throw std::runtime_error("PreparedWeightStore requires a non-zero binding id");
            if (kind == PreparedWeightKind::None)
                throw std::runtime_error("PreparedWeightStore cannot register PreparedWeightKind::None");
            if (model_id.value != 0 && binding.identity.model_id.value != 0 &&
                binding.identity.model_id != model_id)
            {
                throw std::runtime_error("PreparedWeightStore binding model id mismatch");
            }
        }

        bool samePreparedRef(const PreparedWeightRef &stored, const PreparedWeightRef &requested)
        {
            return stored.model_id == requested.model_id &&
                   stored.binding_id == requested.binding_id &&
                   stored.kind == requested.kind &&
                   stored.device == requested.device;
        }
    }

    PreparedWeightStore::PreparedWeightStore(ModelContextId model_id)
        : model_id_(model_id)
    {
    }

    ModelContextId PreparedWeightStore::modelId() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return model_id_;
    }

    bool PreparedWeightStore::bindModelIdIfUnset(ModelContextId model_id)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (model_id.value == 0)
            return model_id_.value == 0;
        if (model_id_.value == 0)
        {
            model_id_ = model_id;
            return true;
        }
        return model_id_ == model_id;
    }

    PreparedWeightKind PreparedWeightStore::inferPreparedKind(DeviceId device) const
    {
        if (device.is_cuda())
            return PreparedWeightKind::CudaInt8PackedGemm;
        if (device.is_rocm())
            return PreparedWeightKind::RocmInt8PackedGemm;
        return PreparedWeightKind::CpuPackedGemm;
    }

    PreparedWeightRef PreparedWeightStore::makeRef(uint64_t binding_id, PreparedWeightKind kind, DeviceId device) const
    {
        PreparedWeightRef ref;
        ref.model_id = model_id_;
        ref.binding_id = binding_id;
        ref.kind = kind;
        ref.device = device;
        return ref;
    }

    PreparedWeightRef PreparedWeightStore::prepareGemm(const WeightBinding &binding)
    {
        if (!binding.tensor)
            throw std::runtime_error("PreparedWeightStore::prepareGemm requires a tensor binding: " + binding.identity.canonical_name);

        const DeviceId device = binding.residency.resident_device.value_or(binding.residency.home_device);
        const PreparedWeightKind kind = inferPreparedKind(device);
        validateBindingForStore(binding, model_id_, kind);

        auto owned = llaminar::v2::kernels::KernelFactory::prepareGemmHandleLocal(
            binding.tensor, device);
        if (!owned)
            throw std::runtime_error("PreparedWeightStore::prepareGemm failed for: " + binding.identity.canonical_name);

        auto ref = makeRef(binding.binding_id, kind, device);
        WeightBinding stored = binding;
        stored.prepared = ref;

        std::lock_guard<std::mutex> lock(mutex_);
        entries_[ref.binding_id] = Entry{std::move(stored), ref, std::move(owned)};
        WeightLifecycleTrace::record(
            WeightLifecycleEventType::RegisterPrepared,
            binding.identity.canonical_name,
            binding.identity.role,
            binding.identity.layer,
            device,
            toString(kind));
        return ref;
    }

    PreparedWeightRef PreparedWeightStore::registerPreparedForTest(
        const WeightBinding &binding,
        PreparedWeightKind kind,
        DeviceId device)
    {
        validateBindingForStore(binding, model_id_, kind);
        auto ref = makeRef(binding.binding_id, kind, device);
        WeightBinding stored = binding;
        stored.prepared = ref;

        std::lock_guard<std::mutex> lock(mutex_);
        entries_[ref.binding_id] = Entry{std::move(stored), ref, nullptr};
        WeightLifecycleTrace::record(
            WeightLifecycleEventType::RegisterPrepared,
            binding.identity.canonical_name,
            binding.identity.role,
            binding.identity.layer,
            device,
            toString(kind));
        return ref;
    }

    PreparedWeightRef PreparedWeightStore::registerPreparedGemmHandle(
        const WeightBinding &binding,
        PreparedWeightKind kind,
        DeviceId device,
        std::shared_ptr<llaminar::v2::kernels::KernelFactory::PreparedGemmHandle> handle)
    {
        validateBindingForStore(binding, model_id_, kind);
        auto ref = makeRef(binding.binding_id, kind, device);
        WeightBinding stored = binding;
        stored.prepared = ref;

        if (stored.tensor)
            stored.tensor->has_prepared_device_state_ = true;

        std::lock_guard<std::mutex> lock(mutex_);
        entries_[ref.binding_id] = Entry{std::move(stored), ref, std::move(handle)};
        WeightLifecycleTrace::record(
            WeightLifecycleEventType::RegisterPrepared,
            binding.identity.canonical_name,
            binding.identity.role,
            binding.identity.layer,
            device,
            toString(kind));
        return ref;
    }

    bool PreparedWeightStore::adoptPreparedGemmForBinding(
        const WeightBinding &binding,
        DeviceId device)
    {
        if (!binding.tensor || !binding.prepared.has_value())
            return false;

        std::lock_guard<std::mutex> lock(mutex_);
        if (entries_.find(binding.binding_id) != entries_.end())
            return true;

        for (const auto &[id, entry] : entries_)
        {
            (void)id;
            if (entry.ref.device != device)
                continue;

            const bool same_tensor = entry.binding.tensor == binding.tensor;
            const bool same_canonical_name = !entry.binding.identity.canonical_name.empty() &&
                                             entry.binding.identity.canonical_name == binding.identity.canonical_name;
            if (!same_tensor && !same_canonical_name)
                continue;

            auto handle = entry.owned_handle;
            if (!handle || !handle->prepared_weights)
                continue;

            validateBindingForStore(binding, model_id_, entry.ref.kind);
            WeightBinding stored = binding;
            auto ref = makeRef(binding.binding_id, entry.ref.kind, device);
            stored.prepared = ref;
            if (stored.tensor)
                stored.tensor->has_prepared_device_state_ = true;
            entries_[ref.binding_id] = Entry{std::move(stored), ref, std::move(handle)};
            return true;
        }

        return false;
    }

    ITensorGemm *PreparedWeightStore::gemmKernel(const PreparedWeightRef &ref) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(ref.binding_id);
        if (it == entries_.end() || !samePreparedRef(it->second.ref, ref))
            return nullptr;
        const auto *handle = it->second.activeHandle();
        if (!handle || !handle->prepared_weights)
            return nullptr;
        // Phase 8: Direct kernel resolution — no KernelFactory delegation.
        return handle->prepared_weights->kernel;
    }

    std::optional<PreparedWeightRef> PreparedWeightStore::preparedRefForBinding(
        uint64_t binding_id,
        DeviceId device) const
    {
        if (binding_id == 0)
            return std::nullopt;

        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(binding_id);
        if (it != entries_.end() && it->second.ref.device == device)
            return it->second.ref;

        auto emb_it = embedding_entries_.find(binding_id);
        if (emb_it != embedding_entries_.end() && emb_it->second.ref.device == device)
            return emb_it->second.ref;

        return std::nullopt;
    }

    ITensorFusedGateUpGemm *PreparedWeightStore::fusedGateUpKernel(
        const PreparedWeightRef &gate_ref,
        const PreparedWeightRef &up_ref) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto gate_it = entries_.find(gate_ref.binding_id);
        auto up_it = entries_.find(up_ref.binding_id);
        if (gate_it == entries_.end() || up_it == entries_.end())
            return nullptr;
        if (!samePreparedRef(gate_it->second.ref, gate_ref) ||
            !samePreparedRef(up_it->second.ref, up_ref))
            return nullptr;
        auto *gate_tensor = gate_it->second.binding.tensor;
        auto *up_tensor = up_it->second.binding.tensor;
        if (!gate_tensor || !up_tensor)
            return nullptr;

        // Phase 8: Check local fused cache first
        FusedCacheKey fkey{gate_ref.binding_id, up_ref.binding_id};
        auto fc_it = fused_cache_.find(fkey);
        if (fc_it != fused_cache_.end())
            return fc_it->second.get();

        auto fused = llaminar::v2::kernels::KernelFactory::createFusedGateUpGemmLocal(
            gate_it->second.activeHandle(), up_it->second.activeHandle(), gate_ref.device);
        if (!fused)
            return nullptr;
        auto *raw = fused.get();
        fused_cache_[fkey] = std::move(fused);
        return raw;
    }

    // =========================================================================
    // Embedding Preparation & Resolution
    // =========================================================================

    PreparedWeightRef PreparedWeightStore::prepareEmbedding(
        const WeightBinding &binding,
        int d_model,
        size_t vocab_offset,
        size_t total_vocab)
    {
        if (!binding.tensor)
            throw std::runtime_error("PreparedWeightStore::prepareEmbedding requires a tensor binding: " + binding.identity.canonical_name);

        const DeviceId device = binding.residency.resident_device.value_or(binding.residency.home_device);

        auto owned = llaminar::v2::kernels::KernelFactory::prepareEmbeddingHandleLocal(
            binding.tensor, d_model, device, vocab_offset, total_vocab);
        if (!owned)
            throw std::runtime_error("PreparedWeightStore::prepareEmbedding failed for: " + binding.identity.canonical_name);

        auto ref = makeRef(binding.binding_id, PreparedWeightKind::PreparedEmbedding, device);
        WeightBinding stored = binding;
        stored.prepared = ref;

        std::lock_guard<std::mutex> lock(mutex_);
        embedding_entries_[ref.binding_id] = EmbeddingEntry{std::move(stored), ref, std::move(owned), nullptr};

        // Mark tensor as having prepared device state
        binding.tensor->has_prepared_device_state_ = true;

        return ref;
    }

    PreparedWeightRef PreparedWeightStore::registerPreparedEmbeddingFromPipeline(
        const WeightBinding &binding,
        DeviceId device,
        const PreparedEmbeddingHandle *handle)
    {
        auto ref = makeRef(binding.binding_id, PreparedWeightKind::PreparedEmbedding, device);
        WeightBinding stored = binding;
        stored.prepared = ref;

        std::shared_ptr<PreparedEmbeddingHandle> owned;
        if (handle)
        {
            owned = std::make_shared<PreparedEmbeddingHandle>();
            owned->tensor = handle->tensor;
            owned->device_id = handle->device_id;
            owned->weights = handle->weights;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        embedding_entries_[ref.binding_id] = EmbeddingEntry{std::move(stored), ref, std::move(owned), handle};

        return ref;
    }

    const PreparedEmbeddingHandle *PreparedWeightStore::embeddingHandle(const PreparedWeightRef &ref) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = embedding_entries_.find(ref.binding_id);
        if (it == embedding_entries_.end() || !samePreparedRef(it->second.ref, ref))
            return nullptr;
        return it->second.activeHandle();
    }

    // =========================================================================
    // Sliced GEMM (TP row-range) Resolution
    // =========================================================================

    ITensorGemm *PreparedWeightStore::slicedGemmKernel(
        const PreparedWeightRef &ref,
        size_t row_start,
        size_t row_end) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto entry_it = entries_.find(ref.binding_id);
        if (entry_it == entries_.end() || !samePreparedRef(entry_it->second.ref, ref))
            return nullptr;

        const TensorBase *tensor = entry_it->second.binding.tensor;
        if (!tensor)
            return nullptr;

        SlicedKey key{ref.binding_id, nullptr, row_start, row_end};
        auto cache_it = sliced_cache_.find(key);
        if (cache_it != sliced_cache_.end())
            return cache_it->second.get();

        auto kernel = llaminar::v2::kernels::KernelFactory::createGemmSlicedLocal(
            tensor, row_start, row_end);
        if (kernel)
        {
            auto *raw = kernel.get();
            sliced_cache_[key] = std::move(kernel);
            return raw;
        }
        return nullptr;
    }

    bool PreparedWeightStore::contains(const PreparedWeightRef &ref) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(ref.binding_id);
        if (it != entries_.end())
        {
            return samePreparedRef(it->second.ref, ref);
        }

        auto emb_it = embedding_entries_.find(ref.binding_id);
        if (emb_it == embedding_entries_.end())
            return false;
        return samePreparedRef(emb_it->second.ref, ref);
    }

    std::optional<WeightBinding> PreparedWeightStore::binding(const PreparedWeightRef &ref) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(ref.binding_id);
        if (it != entries_.end() && samePreparedRef(it->second.ref, ref))
            return it->second.binding;

        auto emb_it = embedding_entries_.find(ref.binding_id);
        if (emb_it == embedding_entries_.end() || !samePreparedRef(emb_it->second.ref, ref))
            return std::nullopt;
        return emb_it->second.binding;
    }

    size_t PreparedWeightStore::size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.size();
    }

    void PreparedWeightStore::dumpEntries(const char *prefix) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        LOG_ERROR(prefix << " PreparedWeightStore dump (" << entries_.size() << " entries):");
        for (const auto &[id, entry] : entries_)
        {
            LOG_ERROR(prefix << "   id=" << id
                             << " name='" << entry.binding.identity.canonical_name << "'"
                             << " tensor_ptr=" << (void *)entry.binding.tensor
                             << " has_handle=" << (entry.activeHandle() != nullptr));
        }
    }

    void PreparedWeightStore::clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.clear();
        fused_cache_.clear();
        embedding_entries_.clear();
        sliced_cache_.clear();
        expert_slabs_.clear();
    }

    PreparedWeightStore::~PreparedWeightStore()
    {
        // Automatically release prepared state on destruction.
        // This ensures model teardown cleans global registries before tensors die.
        releaseAllPreparedState();
    }

    void PreparedWeightStore::releaseAllPreparedState()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (entries_.empty() && fused_cache_.empty() &&
            embedding_entries_.empty() && sliced_cache_.empty() &&
            expert_slabs_.empty())
            return;

        // This store owns prepared handles directly via shared_ptr. Clearing entries_
        // releases owned kernels through PreparedGemmWeights::owned_kernel. Expert
        // slabs own MoE expert GEMM engines through ExpertEntry::engine_lifetime.

        entries_.clear();
        fused_cache_.clear();
        embedding_entries_.clear();
        sliced_cache_.clear();
        expert_slabs_.clear();
    }

    // =========================================================================
    // MoE Expert Slab Implementation
    // =========================================================================

    ExpertSlabRef PreparedWeightStore::registerExpertSlab(const ExpertSlabDescriptor &desc)
    {
        if (desc.num_experts <= 0)
            throw std::runtime_error("ExpertSlabDescriptor requires num_experts > 0");
        if (desc.layer_idx < 0)
            throw std::runtime_error("ExpertSlabDescriptor requires layer_idx >= 0");

        ExpertSlabRef ref;
        ref.model_id = model_id_;
        ref.layer_idx = desc.layer_idx;
        ref.role = desc.role;
        ref.device = desc.device;

        auto entry = std::make_shared<ExpertSlabEntry>();
        entry->descriptor = desc;
        entry->experts.resize(static_cast<size_t>(desc.num_experts));

        std::lock_guard<std::mutex> lock(mutex_);
        ref.slab_id = next_slab_id_++;
        entry->ref = ref;
        expert_slabs_[ref.slab_id] = std::move(entry);
        return ref;
    }

    std::optional<ExpertSlabRef> PreparedWeightStore::findExpertSlab(const ExpertSlabDescriptor &desc) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto &[_, entry] : expert_slabs_)
        {
            const auto &candidate = entry->descriptor;
            if (candidate.layer_idx == desc.layer_idx &&
                candidate.role == desc.role &&
                candidate.device == desc.device &&
                candidate.num_experts == desc.num_experts &&
                candidate.local_expert_start == desc.local_expert_start &&
                candidate.local_expert_count == desc.local_expert_count &&
                candidate.rows_per_expert == desc.rows_per_expert &&
                candidate.cols_per_expert == desc.cols_per_expert)
            {
                return entry->ref;
            }
        }
        return std::nullopt;
    }

    ITensorGemm *PreparedWeightStore::expertGemmKernel(const ExpertSlabRef &slab, int expert_id) const
    {
        // Hot path: find slab entry under brief outer lock, then release outer lock
        // before taking per-slab shared_lock (allows concurrent reads across slabs).
        // Hold shared_ptr to keep slab alive if releaseExpertSlab() races.
        std::shared_ptr<ExpertSlabEntry> slab_ptr;
        {
            std::lock_guard<std::mutex> outer_lock(mutex_);
            auto it = expert_slabs_.find(slab.slab_id);
            if (it == expert_slabs_.end())
                return nullptr;
            slab_ptr = it->second;
        }

        std::shared_lock<std::shared_mutex> slab_lock(slab_ptr->slab_mutex);
        if (expert_id < 0 || expert_id >= static_cast<int>(slab_ptr->experts.size()))
            return nullptr;
        const auto &expert = slab_ptr->experts[static_cast<size_t>(expert_id)];
        return expert.available ? expert.engine : nullptr;
    }

    std::vector<int> PreparedWeightStore::registerArrivedExperts(
        const ExpertSlabRef &slab,
        const std::vector<ExpertArrival> &arrivals)
    {
        std::lock_guard<std::mutex> outer_lock(mutex_);
        auto it = expert_slabs_.find(slab.slab_id);
        if (it == expert_slabs_.end())
            throw std::runtime_error("registerArrivedExperts: slab not found (id=" + std::to_string(slab.slab_id) + ")");

        auto &entry = *it->second;
        std::unique_lock<std::shared_mutex> slab_lock(entry.slab_mutex);

        std::vector<int> actually_new;
        actually_new.reserve(arrivals.size());

        for (const auto &arrival : arrivals)
        {
            if (arrival.expert_id < 0 || arrival.expert_id >= static_cast<int>(entry.experts.size()))
                continue;

            auto &slot = entry.experts[static_cast<size_t>(arrival.expert_id)];
            if (slot.available)
                continue; // Already populated — skip

            slot.engine = arrival.engine;
            slot.engine_lifetime = arrival.engine_lifetime;
            slot.view_lifetime = arrival.view_lifetime;
            slot.derivation = arrival.derivation;
            slot.source_device = arrival.source_device;
            slot.available = true;
            actually_new.push_back(arrival.expert_id);
        }
        return actually_new;
    }

    void PreparedWeightStore::releaseDepartedExperts(
        const ExpertSlabRef &slab,
        const std::vector<int> &expert_ids)
    {
        std::lock_guard<std::mutex> outer_lock(mutex_);
        auto it = expert_slabs_.find(slab.slab_id);
        if (it == expert_slabs_.end())
            return;

        auto &entry = *it->second;
        std::unique_lock<std::shared_mutex> slab_lock(entry.slab_mutex);

        for (int expert_id : expert_ids)
        {
            if (expert_id < 0 || expert_id >= static_cast<int>(entry.experts.size()))
                continue;

            auto &slot = entry.experts[static_cast<size_t>(expert_id)];
            slot.engine = nullptr;
            slot.engine_lifetime.reset();
            slot.view_lifetime.reset();
            slot.source_device.reset();
            slot.available = false;
        }
    }

    std::vector<bool> PreparedWeightStore::expertAvailabilityMask(const ExpertSlabRef &slab) const
    {
        std::shared_ptr<ExpertSlabEntry> slab_ptr;
        {
            std::lock_guard<std::mutex> outer_lock(mutex_);
            auto it = expert_slabs_.find(slab.slab_id);
            if (it == expert_slabs_.end())
                return {};
            slab_ptr = it->second;
        }

        std::shared_lock<std::shared_mutex> slab_lock(slab_ptr->slab_mutex);
        std::vector<bool> mask(slab_ptr->experts.size(), false);
        for (size_t i = 0; i < slab_ptr->experts.size(); ++i)
            mask[i] = slab_ptr->experts[i].available;
        return mask;
    }

    void PreparedWeightStore::releaseExpertSlab(const ExpertSlabRef &slab)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        expert_slabs_.erase(slab.slab_id);
    }

    size_t PreparedWeightStore::expertSlabCount() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return expert_slabs_.size();
    }

    size_t PreparedWeightStore::totalPopulatedExperts() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t count = 0;
        for (const auto &[id, entry] : expert_slabs_)
        {
            std::shared_lock<std::shared_mutex> slab_lock(entry->slab_mutex);
            for (const auto &expert : entry->experts)
            {
                if (expert.available)
                    ++count;
            }
        }
        return count;
    }

    PreparedWeightStore::MemoryStats PreparedWeightStore::memoryStats() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        MemoryStats stats;

        for (const auto &[_, entry] : entries_)
        {
            const auto *handle = entry.activeHandle();
            auto *kernel = (handle && handle->prepared_weights) ? handle->prepared_weights->kernel : nullptr;
            if (!kernel && handle && handle->prepared_weights && handle->prepared_weights->owned_kernel)
                kernel = handle->prepared_weights->owned_kernel.get();
            if (!kernel)
                continue;
            ++stats.gemm_entries;
            stats.gemm_bytes += kernel->packedWeightBytes();
        }

        for (const auto &[_, kernel] : sliced_cache_)
        {
            if (!kernel)
                continue;
            ++stats.sliced_entries;
            stats.sliced_bytes += kernel->packedWeightBytes();
        }

        stats.expert_slabs = expert_slabs_.size();
        for (const auto &[_, entry] : expert_slabs_)
        {
            std::shared_lock<std::shared_mutex> slab_lock(entry->slab_mutex);
            for (const auto &expert : entry->experts)
            {
                if (!expert.available)
                    continue;
                ++stats.expert_engines;
                if (expert.engine)
                    stats.expert_bytes += expert.engine->packedWeightBytes();
                if (expert.view_lifetime)
                {
                    ++stats.expert_view_lifetimes;
                    if (expert.view_lifetime->is_view())
                        ++stats.expert_borrowed_views;
                    else if (!expert.view_lifetime->is_raw_data_released())
                        ++stats.expert_view_raw_live;
                }
            }
        }

        for (const auto &[_, entry] : embedding_entries_)
        {
            const auto *handle = entry.activeHandle();
            if (!handle || !handle->weights)
                continue;
            ++stats.embedding_entries;
            stats.embedding_bytes += handle->weights->byte_size;
        }

        return stats;
    }

    void PreparedWeightStore::logMemorySummary(const char *context) const
    {
        const auto stats = memoryStats();
        LOG_INFO("[PreparedWeightStore] Memory summary"
                 << (context ? std::string(" (") + context + ")" : std::string{})
                 << ": total=" << (stats.totalBytes() >> 20) << " MB"
                 << " gemm=" << (stats.gemm_bytes >> 20) << " MB/" << stats.gemm_entries
                 << " sliced=" << (stats.sliced_bytes >> 20) << " MB/" << stats.sliced_entries
                 << " experts=" << (stats.expert_bytes >> 20) << " MB/" << stats.expert_engines
                 << " slabs=" << stats.expert_slabs
                 << " expert_views=" << stats.expert_view_lifetimes
                 << " borrowed_views=" << stats.expert_borrowed_views
                 << " raw_live_views=" << stats.expert_view_raw_live
                 << " embeddings=" << (stats.embedding_bytes >> 20) << " MB/" << stats.embedding_entries);
    }
}
