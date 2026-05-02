#pragma once

#include "WeightPlan.h"
#include "../kernels/KernelFactory.h"

#include <mutex>
#include <optional>
#include <unordered_map>

namespace llaminar2
{
    class ITensorGemm;

    class PreparedWeightStore
    {
    public:
        explicit PreparedWeightStore(ModelContextId model_id = {});

        PreparedWeightRef prepareGemm(const WeightBinding &binding);
        PreparedWeightRef registerPreparedForTest(
            const WeightBinding &binding,
            PreparedWeightKind kind,
            DeviceId device);
        PreparedWeightRef registerPreparedGemmFromPipeline(
            const WeightBinding &binding,
            PreparedWeightKind kind,
            DeviceId device,
            const llaminar::v2::kernels::KernelFactory::PreparedGemmHandle *handle);
        ITensorGemm *gemmKernel(const PreparedWeightRef &ref) const;
        bool contains(const PreparedWeightRef &ref) const;
        std::optional<WeightBinding> binding(const PreparedWeightRef &ref) const;
        size_t size() const;
        void clear();

    private:
        struct Entry
        {
            WeightBinding binding;
            PreparedWeightRef ref;
            const llaminar::v2::kernels::KernelFactory::PreparedGemmHandle *gemm_handle = nullptr;
        };

        PreparedWeightKind inferPreparedKind(DeviceId device) const;
        PreparedWeightRef makeRef(uint64_t binding_id, PreparedWeightKind kind, DeviceId device) const;

        ModelContextId model_id_;
        mutable std::mutex mutex_;
        std::unordered_map<uint64_t, Entry> entries_;
    };
}
