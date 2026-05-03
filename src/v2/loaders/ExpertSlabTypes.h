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
    };
}
