#include "PreparedWeightStore.h"

#include "../tensors/TensorClasses.h"
#include "../tensors/TensorKernels.h"

#include <stdexcept>
#include <unordered_set>
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
            if (binding.identity.model_id.value != 0 && binding.identity.model_id != model_id)
            {
                throw std::runtime_error(
                    "PreparedWeightStore model id mismatch for: " + binding.identity.canonical_name);
            }
        }
    }

    PreparedWeightStore::PreparedWeightStore(ModelContextId model_id)
        : model_id_(model_id)
    {
    }

    PreparedWeightKind PreparedWeightStore::inferPreparedKind(DeviceId device) const
    {
        if (device.is_cuda()) return PreparedWeightKind::CudaInt8PackedGemm;
        if (device.is_rocm()) return PreparedWeightKind::RocmInt8PackedGemm;
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

        const auto *handle = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(
            binding.tensor, device);
        if (!handle)
            throw std::runtime_error("PreparedWeightStore::prepareGemm failed for: " + binding.identity.canonical_name);

        auto ref = makeRef(binding.binding_id, kind, device);
        WeightBinding stored = binding;
        stored.prepared = ref;

        std::lock_guard<std::mutex> lock(mutex_);
        entries_[ref.binding_id] = Entry{std::move(stored), ref, handle};
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
        return registerPreparedGemmFromPipeline(binding, kind, device, nullptr);
    }

    PreparedWeightRef PreparedWeightStore::registerPreparedGemmFromPipeline(
        const WeightBinding &binding,
        PreparedWeightKind kind,
        DeviceId device,
        const llaminar::v2::kernels::KernelFactory::PreparedGemmHandle *handle)
    {
        validateBindingForStore(binding, model_id_, kind);
        auto ref = makeRef(binding.binding_id, kind, device);
        WeightBinding stored = binding;
        stored.prepared = ref;

        std::lock_guard<std::mutex> lock(mutex_);
        entries_[ref.binding_id] = Entry{std::move(stored), ref, handle};
        WeightLifecycleTrace::record(
            WeightLifecycleEventType::RegisterPrepared,
            binding.identity.canonical_name,
            binding.identity.role,
            binding.identity.layer,
            device,
            toString(kind));
        return ref;
    }

    ITensorGemm *PreparedWeightStore::gemmKernel(const PreparedWeightRef &ref) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(ref.binding_id);
        if (it == entries_.end() || !it->second.gemm_handle)
            return nullptr;
        return llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(it->second.gemm_handle);
    }

    ITensorGemm *PreparedWeightStore::gemmKernelForTensor(const TensorBase *tensor) const
    {
        if (!tensor)
            return nullptr;
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto &[id, entry] : entries_)
        {
            if (entry.binding.tensor == tensor && entry.gemm_handle)
                return llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(entry.gemm_handle);
        }
        return nullptr;
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
        auto *gate_tensor = gate_it->second.binding.tensor;
        auto *up_tensor = up_it->second.binding.tensor;
        if (!gate_tensor || !up_tensor)
            return nullptr;
        return llaminar::v2::kernels::KernelFactory::getOrCreateFusedGateUpGemm(
            gate_tensor, up_tensor, gate_ref.device);
    }

    ITensorFusedGateUpGemm *PreparedWeightStore::fusedGateUpKernelForTensors(
        const TensorBase *gate_tensor,
        const TensorBase *up_tensor,
        DeviceId device) const
    {
        if (!gate_tensor || !up_tensor)
            return nullptr;
        return llaminar::v2::kernels::KernelFactory::getOrCreateFusedGateUpGemm(
            gate_tensor, up_tensor, device);
    }

    bool PreparedWeightStore::contains(const PreparedWeightRef &ref) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(ref.binding_id);
        if (it == entries_.end())
            return false;
        return it->second.ref.model_id == ref.model_id &&
               it->second.ref.kind == ref.kind &&
               it->second.ref.device == ref.device;
    }

    std::optional<WeightBinding> PreparedWeightStore::binding(const PreparedWeightRef &ref) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(ref.binding_id);
        if (it == entries_.end() ||
            it->second.ref.model_id != ref.model_id ||
            it->second.ref.kind != ref.kind ||
            it->second.ref.device != ref.device)
            return std::nullopt;
        return it->second.binding;
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
                      << " has_handle=" << (entry.gemm_handle != nullptr));
        }
    }

    void PreparedWeightStore::clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.clear();
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

        if (entries_.empty())
            return;

        // Collect unique tensors managed by this store
        std::unordered_set<const TensorBase *> managed_tensors;
        managed_tensors.reserve(entries_.size());

        for (const auto &[id, entry] : entries_)
        {
            if (entry.binding.tensor)
            {
                managed_tensors.insert(entry.binding.tensor);
            }
        }

        // Clear prepared state from KernelFactory for each managed tensor.
        // This removes prepared_gemm_registry_, fused_gate_up_cache_,
        // sliced_cache_, and prepared_embedding_registry_ entries.
        for (const auto *tensor : managed_tensors)
        {
            llaminar::v2::kernels::KernelFactory::clearPreparedStateForTensor(tensor);
        }

        entries_.clear();
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
}
