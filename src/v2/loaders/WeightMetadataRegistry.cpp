#include "WeightMetadataRegistry.h"

#include "../tensors/Tensors.h"

#include <sstream>

namespace llaminar2
{
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
        return metadata_.emplace(
            tensor,
            WeightMetadata{makeSourceWeightIdentity(canonical_name, {}, nextInstanceIdLocked()), {}, residency}).second;
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
            it->second.residency = residency;
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
