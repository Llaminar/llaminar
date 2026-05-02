#include "PreparedWeightStore.h"

#include "../tensors/TensorKernels.h"

#include <stdexcept>
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

    void PreparedWeightStore::clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.clear();
    }
}
