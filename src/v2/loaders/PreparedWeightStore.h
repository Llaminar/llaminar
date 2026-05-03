#pragma once

#include "ExpertSlabTypes.h"
#include "WeightPlan.h"
#include "../kernels/KernelFactory.h"

#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>

namespace llaminar2
{
    class ITensorGemm;
    class ITensorFusedGateUpGemm;
    class TensorBase;

    class PreparedWeightStore
    {
    public:
        explicit PreparedWeightStore(ModelContextId model_id = {});
        ~PreparedWeightStore();

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
        ITensorGemm *gemmKernelForTensor(const TensorBase *tensor) const;
        ITensorFusedGateUpGemm *fusedGateUpKernel(
            const PreparedWeightRef &gate_ref,
            const PreparedWeightRef &up_ref) const;
        ITensorFusedGateUpGemm *fusedGateUpKernelForTensors(
            const TensorBase *gate_tensor,
            const TensorBase *up_tensor,
            DeviceId device) const;
        bool contains(const PreparedWeightRef &ref) const;
        std::optional<WeightBinding> binding(const PreparedWeightRef &ref) const;
        size_t size() const;
        void clear();

        /**
         * @brief Release all prepared weight state from global registries (Phase 8/10)
         *
         * This proactively removes all prepared GEMM, fused gate/up, sliced, and
         * embedding entries from KernelFactory's static registries for every tensor
         * managed by this store.
         *
         * Phase 10: This is the EXCLUSIVE path for KernelFactory registry cleanup.
         * TensorBase destructors never touch global registries.
         *
         * After this call, the store is empty and gemmKernel()/fusedGateUpKernel()
         * will return nullptr for all refs.
         */
        void releaseAllPreparedState();

        // Debug: dump all entries with tensor pointers for diagnosing lookup failures
        void dumpEntries(const char *prefix) const;

        // =========================================================================
        // MoE Expert Slab API
        // =========================================================================

        /// Register a new expert slab (one weight group × one layer × one device).
        /// Returns a ref that can be used to look up individual expert engines.
        ExpertSlabRef registerExpertSlab(const ExpertSlabDescriptor &desc);

        /// Get the GEMM engine for a specific expert within a slab.
        /// Returns nullptr if the expert is not populated.
        ITensorGemm *expertGemmKernel(const ExpertSlabRef &slab, int expert_id) const;

        /// Register newly-arrived expert engines (from initial load or rebalance transfer).
        /// Returns the expert IDs that were actually new (not already populated).
        std::vector<int> registerArrivedExperts(
            const ExpertSlabRef &slab,
            const std::vector<ExpertArrival> &arrivals);

        /// Release departed expert engines (free memory for evicted/migrated experts).
        void releaseDepartedExperts(
            const ExpertSlabRef &slab,
            const std::vector<int> &expert_ids);

        /// Query which experts in this slab have prepared engines.
        std::vector<bool> expertAvailabilityMask(const ExpertSlabRef &slab) const;

        /// Release an entire slab (model unload). Frees all engines and removes the slab.
        void releaseExpertSlab(const ExpertSlabRef &slab);

        /// Number of expert slabs registered.
        size_t expertSlabCount() const;

        /// Total number of populated expert engines across all slabs.
        size_t totalPopulatedExperts() const;

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

        // =========================================================================
        // Expert Slab Storage
        // =========================================================================

        struct ExpertEntry
        {
            ITensorGemm *engine = nullptr;
            std::shared_ptr<ITensorGemm> engine_lifetime;
            std::shared_ptr<TensorBase> view_lifetime;
            WeightDerivationKind derivation = WeightDerivationKind::ExpertSlice;
            std::optional<DeviceId> source_device;
            bool available = false;
        };

        struct ExpertSlabEntry
        {
            ExpertSlabDescriptor descriptor;
            ExpertSlabRef ref;
            std::vector<ExpertEntry> experts; // Indexed by expert_id
            mutable std::shared_mutex slab_mutex; // Per-slab: shared for reads, exclusive for writes
        };

        uint64_t next_slab_id_ = 1;
        std::unordered_map<uint64_t, std::shared_ptr<ExpertSlabEntry>> expert_slabs_;
        // Note: expert_slabs_ itself is protected by the existing mutex_ for insert/erase.
        // Per-slab reads/writes use slab_mutex inside ExpertSlabEntry.
        // shared_ptr ensures slab entries outlive concurrent readers even during erase.
    };
}
