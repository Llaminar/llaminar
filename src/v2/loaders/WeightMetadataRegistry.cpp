/**
 * @file WeightMetadataRegistry.cpp
 * @brief Implements tensor-pointer metadata tracking for graph weight bindings.
 */

#include "WeightMetadataRegistry.h"

#include "../tensors/Tensors.h"

#include <sstream>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Join host policies without losing a live CPU consumer.
         *
         * The ordinary lifecycle gate resolves every transient `RequiredUntil*`
         * policy before the release sweep. CPU execution is different: its
         * floating-point engines continue reading the source bytes during every
         * inference. It consequently dominates every reclaimable policy.
         */
        WeightHostPolicy mergeHostPolicies(
            WeightHostPolicy current,
            WeightHostPolicy incoming)
        {
            if (current == WeightHostPolicy::RequiredForCPUExecution ||
                incoming == WeightHostPolicy::RequiredForCPUExecution)
            {
                return WeightHostPolicy::RequiredForCPUExecution;
            }
            return incoming;
        }
    }

    uint64_t WeightMetadataRegistry::nextInstanceIdLocked()
    {
        return next_instance_id_++;
    }

    bool WeightMetadataRegistry::registerSource(
        const TensorBase *tensor,
        const std::string &canonical_name,
        DeviceId home_device)
    {
        if (!tensor)
            return false;

        std::lock_guard<std::mutex> lock(mutex_);
        WeightResidency residency;
        residency.home_device = home_device;
        residency.resident_device = home_device.is_valid() ? std::optional<DeviceId>(home_device) : std::nullopt;

        auto it = metadata_.find(tensor);
        if (it != metadata_.end() &&
            it->second.identity.derivation == WeightDerivationKind::Source &&
            it->second.identity.canonical_name == canonical_name)
        {
            residency.host_policy = mergeHostPolicies(
                it->second.residency.host_policy,
                residency.host_policy);
            it->second.residency = residency;
            return false;
        }

        metadata_[tensor] =
            WeightMetadata{makeSourceWeightIdentity(canonical_name, {}, nextInstanceIdLocked()), {}, residency};
        return true;
    }

    bool WeightMetadataRegistry::registerWeight(
        const TensorBase *tensor,
        WeightIdentity identity,
        WeightSliceSpec slice,
        WeightResidency residency)
    {
        if (!tensor)
            return false;

        std::lock_guard<std::mutex> lock(mutex_);
        if (identity.instance_id == 0)
            identity.instance_id = nextInstanceIdLocked();
        if (identity.logical_id == 0 && !identity.canonical_name.empty())
            identity.logical_id = stableWeightLogicalId(identity.canonical_name);
        if (const auto existing = metadata_.find(tensor);
            existing != metadata_.end())
        {
            residency.host_policy = mergeHostPolicies(
                existing->second.residency.host_policy,
                residency.host_policy);
        }
        metadata_[tensor] = WeightMetadata{std::move(identity), slice, residency};
        return true;
    }

    bool WeightMetadataRegistry::registerDerived(
        const TensorBase *tensor,
        const TensorBase *source,
        WeightDerivationKind derivation,
        WeightSliceSpec slice,
        DeviceId home_device)
    {
        if (!tensor || !source)
            return false;

        std::lock_guard<std::mutex> lock(mutex_);
        auto source_it = metadata_.find(source);
        if (source_it == metadata_.end())
            return false;

        WeightIdentity identity = source_it->second.identity;
        identity.source_instance_id = source_it->second.identity.instance_id;
        identity.instance_id = nextInstanceIdLocked();
        identity.derivation = derivation;

        WeightResidency residency = source_it->second.residency;
        residency.home_device = home_device;
        residency.resident_device = home_device.is_valid() ? std::optional<DeviceId>(home_device) : std::nullopt;

        metadata_[tensor] = WeightMetadata{std::move(identity), slice, residency};
        return true;
    }

    bool WeightMetadataRegistry::has(const TensorBase *tensor) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return metadata_.find(tensor) != metadata_.end();
    }

    std::optional<WeightMetadata> WeightMetadataRegistry::metadata(const TensorBase *tensor) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = metadata_.find(tensor);
        if (it == metadata_.end())
            return std::nullopt;
        return it->second;
    }

    std::optional<WeightIdentity> WeightMetadataRegistry::identity(const TensorBase *tensor) const
    {
        auto data = metadata(tensor);
        if (!data)
            return std::nullopt;
        return data->identity;
    }

    std::optional<WeightSliceSpec> WeightMetadataRegistry::slice(const TensorBase *tensor) const
    {
        auto data = metadata(tensor);
        if (!data)
            return std::nullopt;
        return data->slice;
    }

    std::optional<WeightResidency> WeightMetadataRegistry::residency(const TensorBase *tensor) const
    {
        auto data = metadata(tensor);
        if (!data)
            return std::nullopt;
        return data->residency;
    }

    void WeightMetadataRegistry::updateResidency(const TensorBase *tensor, WeightResidency residency)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = metadata_.find(tensor);
        if (it != metadata_.end())
        {
            residency.host_policy = mergeHostPolicies(
                it->second.residency.host_policy,
                residency.host_policy);
            it->second.residency = residency;
        }
    }

    bool WeightMetadataRegistry::mergeHostPolicy(
        const TensorBase *tensor,
        WeightHostPolicy policy)
    {
        if (!tensor)
            return false;

        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = metadata_.find(tensor);
        if (it == metadata_.end())
            return false;

        it->second.residency.host_policy = mergeHostPolicies(
            it->second.residency.host_policy,
            policy);
        return true;
    }

    std::string WeightMetadataRegistry::describe(const TensorBase *tensor) const
    {
        auto data = metadata(tensor);
        if (!data)
            return "(unregistered weight)";

        std::ostringstream out;
        out << data->identity.canonical_name
            << " role=" << toString(data->identity.role)
            << " derivation=" << toString(data->identity.derivation)
            << " instance=" << data->identity.instance_id;
        if (data->identity.source_instance_id)
            out << " source=" << *data->identity.source_instance_id;
        out << " home=" << data->residency.home_device.to_string();
        return out.str();
    }

    std::vector<WeightMetadata> WeightMetadataRegistry::snapshot() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<WeightMetadata> values;
        values.reserve(metadata_.size());
        for (const auto &entry : metadata_)
            values.push_back(entry.second);
        return values;
    }

    size_t WeightMetadataRegistry::size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return metadata_.size();
    }

    void WeightMetadataRegistry::clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        metadata_.clear();
        next_instance_id_ = 1;
    }
}
