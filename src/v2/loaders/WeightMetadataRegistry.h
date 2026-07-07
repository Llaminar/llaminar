/**
 * @file WeightMetadataRegistry.h
 * @brief Lifetime-scoped metadata registry for tensor weight identities.
 *
 * Weight tensors can be loaded, sliced, cloned, and rebound under graph-local
 * aliases before a DeviceGraphOrchestrator consumes them.  This registry records
 * the source identity, derivation, slice, and residency attached to each live
 * TensorBase pointer so later planning code can preserve slice/residency facts
 * without guessing from names alone.
 */

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
        /**
         * @brief Register or refresh metadata for a source tensor.
         *
         * Tensor allocation addresses may be reused after a previous runner is
         * destroyed.  Source registration is therefore an upsert: if the pointer
         * already has stale metadata for a different canonical weight, the old
         * identity is replaced with a fresh source identity for this load.
         *
         * @param tensor Live tensor pointer that owns or views the source data.
         * @param canonical_name Canonical GGUF graph weight name.
         * @param home_device Device that owns the source tensor residency.
         * @return true when a new entry was created or stale identity was replaced.
         */
        bool registerSource(
            const TensorBase *tensor,
            const std::string &canonical_name,
            DeviceId home_device = DeviceId::cpu());

        /**
         * @brief Install exact metadata for a graph-derived tensor.
         *
         * Use this when the caller has already constructed the full identity and
         * slice, for example for row slices, tied aliases, or overlay expert
         * views.  Unlike registerSource(), this deliberately overwrites any
         * existing pointer entry because derived views are graph-binding scoped.
         */
        bool registerWeight(
            const TensorBase *tensor,
            WeightIdentity identity,
            WeightSliceSpec slice = {},
            WeightResidency residency = {});

        /**
         * @brief Register metadata for a tensor derived from another registered tensor.
         *
         * The derived tensor inherits the source canonical identity and records
         * the source instance id.  This fails fast by returning false when the
         * source tensor has no registered metadata, allowing callers to install a
         * source identity explicitly instead of inheriting stale state.
         */
        bool registerDerived(
            const TensorBase *tensor,
            const TensorBase *source,
            WeightDerivationKind derivation,
            WeightSliceSpec slice = {},
            DeviceId home_device = DeviceId::cpu());

        /// Return whether metadata is registered for @p tensor.
        bool has(const TensorBase *tensor) const;

        /// Return full metadata for @p tensor, if the pointer is registered.
        std::optional<WeightMetadata> metadata(const TensorBase *tensor) const;

        /// Return only the identity portion of the registered metadata.
        std::optional<WeightIdentity> identity(const TensorBase *tensor) const;

        /// Return only the slice portion of the registered metadata.
        std::optional<WeightSliceSpec> slice(const TensorBase *tensor) const;

        /// Return only the residency portion of the registered metadata.
        std::optional<WeightResidency> residency(const TensorBase *tensor) const;

        /// Replace residency metadata for an already registered tensor pointer.
        void updateResidency(const TensorBase *tensor, WeightResidency residency);

        /// Render a compact diagnostic string for the tensor metadata.
        std::string describe(const TensorBase *tensor) const;

        /// Return a point-in-time copy of all registered metadata records.
        std::vector<WeightMetadata> snapshot() const;

        /// Return the number of registered tensor-pointer metadata records.
        size_t size() const;

        /// Remove all metadata and reset instance id allocation.
        void clear();

    private:
        /// Allocate the next registry-local identity instance id.
        uint64_t nextInstanceIdLocked();

        mutable std::mutex mutex_;
        uint64_t next_instance_id_ = 1;
        std::unordered_map<const TensorBase *, WeightMetadata> metadata_;
    };
}
