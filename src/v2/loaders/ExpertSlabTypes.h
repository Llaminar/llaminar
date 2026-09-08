#pragma once

#include "WeightIdentity.h"
#include "../backends/DeviceId.h"

#include <memory>
#include <optional>
#include <vector>

namespace llaminar2
{
    class ITensorGemm;
    class TensorBase;

    /// Completion handle for an asynchronous same-backend GPU expert transfer.
    ///
    /// The transfer stream records `ready_event` after packed-weight peer copies
    /// are enqueued.  Any stage that consumes the arrived expert must make its
    /// explicit compute stream wait on `ready_event` before launching GEMMs.
    struct GpuDirectTransferCompletion
    {
        DeviceId device_id;
        int device_ordinal = -1;
        std::shared_ptr<void> ready_event;
        std::shared_ptr<void> source_ready_event;
        std::shared_ptr<void> transfer_stream;
        std::vector<std::shared_ptr<void>> transient_lifetimes;

        bool valid() const
        {
            return device_id.is_gpu() &&
                   device_ordinal >= 0 &&
                   ready_event != nullptr;
        }
    };

    /// Identifies a "slab" of expert GEMM weights for one weight group × one layer.
    struct ExpertSlabRef
    {
        ModelContextId model_id;
        uint64_t slab_id = 0;
        int layer_idx = -1;
        WeightRole role = WeightRole::Other;
        DeviceId device = DeviceId::cpu();

        bool operator==(const ExpertSlabRef &other) const
        {
            return model_id == other.model_id && slab_id == other.slab_id;
        }
    };

    /// Descriptor for registering a new expert slab.
    struct ExpertSlabDescriptor
    {
        int layer_idx = -1;
        WeightRole role = WeightRole::Other;
        DeviceId device = DeviceId::cpu();
        int num_experts = 0;
        int local_expert_start = 0;
        int local_expert_count = 0;
        size_t rows_per_expert = 0;
        size_t cols_per_expert = 0;
        WeightIdentity source_identity; // Identity of the 3D parent tensor
    };

    /// Describes one expert arriving (from initial load or rebalance transfer).
    struct ExpertArrival
    {
        int expert_id = -1;
        ITensorGemm *engine = nullptr;
        std::shared_ptr<ITensorGemm> engine_lifetime;
        std::shared_ptr<TensorBase> view_lifetime;
        WeightDerivationKind derivation = WeightDerivationKind::ExpertSlice;
        std::optional<DeviceId> source_device; // Non-null for RebalancedExpertReplica
        std::optional<GpuDirectTransferCompletion> gpu_direct_completion;
    };
}
