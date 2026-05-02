#pragma once

#include "WeightIdentity.h"

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{
    class TensorBase;

    struct WeightMetadata
    {
        WeightIdentity identity;
        WeightSliceSpec slice;
        WeightResidency residency;
    };

    class WeightMetadataRegistry
    {
    public:
        bool registerSource(
            const TensorBase *tensor,
            const std::string &canonical_name,
            DeviceId home_device = DeviceId::cpu());

        bool registerWeight(
            const TensorBase *tensor,
            WeightIdentity identity,
            WeightSliceSpec slice = {},
            WeightResidency residency = {});

        bool registerDerived(
            const TensorBase *tensor,
            const TensorBase *source,
            WeightDerivationKind derivation,
            WeightSliceSpec slice = {},
            DeviceId home_device = DeviceId::cpu());

        bool has(const TensorBase *tensor) const;
        std::optional<WeightMetadata> metadata(const TensorBase *tensor) const;
        std::optional<WeightIdentity> identity(const TensorBase *tensor) const;
        std::optional<WeightSliceSpec> slice(const TensorBase *tensor) const;
        std::optional<WeightResidency> residency(const TensorBase *tensor) const;
        void updateResidency(const TensorBase *tensor, WeightResidency residency);
        std::string describe(const TensorBase *tensor) const;
        std::vector<WeightMetadata> snapshot() const;
        size_t size() const;
        void clear();

    private:
        uint64_t nextInstanceIdLocked();

        mutable std::mutex mutex_;
        uint64_t next_instance_id_ = 1;
        std::unordered_map<const TensorBase *, WeightMetadata> metadata_;
    };
}
