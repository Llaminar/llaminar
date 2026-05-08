#pragma once

#include "ExpertSlabTypes.h"
#include "WeightPlan.h"
#include "../kernels/KernelFactory.h"
#include "../kernels/common/PreparedEmbeddingWeights.h"

#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace llaminar2
{
    class ITensorGemm;
    class ITensorFusedGateUpGemm;
    class TensorBase;

    /**
     * @brief Model-context-owned prepared weight store (Phase 8 KernelFactory slimming).
     *
     * This store owns all model-weight-lifetime state: prepared GEMM handles, fused
     * gate/up kernels, and embedding handles. KernelFactory is reduced to a
     * KernelRegistry role (device-scoped kernel selection, no model-weight ownership).
     *
     * Kernel resolution is direct: gemmKernel() returns the kernel from the owned
     * handle without delegating to KernelFactory's static registries.
     */
    class PreparedWeightStore
    {
    public:
        explicit PreparedWeightStore(ModelContextId model_id = {});
        ~PreparedWeightStore();

        ModelContextId modelId() const;

        /// Bind a previously-unbound store to a model id. Returns true when the
        /// store was unbound or already bound to the same id; false on mismatch.
        bool bindModelIdIfUnset(ModelContextId model_id);

        // =========================================================================
        // GEMM Preparation & Resolution
        // =========================================================================

        PreparedWeightRef prepareGemm(const WeightBinding &binding);
        PreparedWeightRef registerPreparedForTest(
            const WeightBinding &binding,
            PreparedWeightKind kind,
            DeviceId device);
        PreparedWeightRef registerPreparedGemmHandle(
            const WeightBinding &binding,
            PreparedWeightKind kind,
            DeviceId device,
            std::shared_ptr<llaminar::v2::kernels::KernelFactory::PreparedGemmHandle> handle);
        bool adoptPreparedGemmForBinding(
            const WeightBinding &binding,
            DeviceId device);

        /// Resolve GEMM kernel from a prepared ref. O(1) lookup by binding_id.
        /// Returns nullptr if ref not found or handle missing.
        ITensorGemm *gemmKernel(const PreparedWeightRef &ref) const;

        /// Resolve a prepared ref by frozen binding identity. O(1) lookup by binding id.
        /// Returns std::nullopt if the binding has not been prepared for this device.
        std::optional<PreparedWeightRef> preparedRefForBinding(
            uint64_t binding_id,
            DeviceId device) const;

        // =========================================================================
        // Fused Gate/Up Kernel Resolution
        // =========================================================================

        ITensorFusedGateUpGemm *fusedGateUpKernel(
            const PreparedWeightRef &gate_ref,
            const PreparedWeightRef &up_ref) const;

        // =========================================================================
        // Embedding Preparation & Resolution
        // =========================================================================

        /// Prepare embedding weights on a target device. Delegates creation to
        /// KernelFactory but owns the handle lifetime.
        PreparedWeightRef prepareEmbedding(
            const WeightBinding &binding,
            int d_model,
            size_t vocab_offset = 0,
            size_t total_vocab = 0);

        /// Register an already-prepared embedding handle from the pipeline.
        PreparedWeightRef registerPreparedEmbeddingFromPipeline(
            const WeightBinding &binding,
            DeviceId device,
            const PreparedEmbeddingHandle *handle);

        /// Look up prepared embedding weights by ref. O(1).
        const PreparedEmbeddingHandle *embeddingHandle(const PreparedWeightRef &ref) const;

        // =========================================================================
        // Sliced GEMM (TP row-range) Resolution
        // =========================================================================

        /// Get or create a sliced GEMM kernel for a prepared weight ref.
        /// Uses the binding id as the store-owned cache identity.
        ITensorGemm *slicedGemmKernel(
            const PreparedWeightRef &ref,
            size_t row_start,
            size_t row_end) const;

        // =========================================================================
        // Query & Lifecycle
        // =========================================================================

        bool contains(const PreparedWeightRef &ref) const;
        std::optional<WeightBinding> binding(const PreparedWeightRef &ref) const;
        size_t size() const;
        void clear();

        /**
         * @brief Release all owned prepared weight state.
         *
         * Phase 8: This store OWNS prepared handles and fused kernels directly.
         * No delegation to KernelFactory static registries for cleanup.
         * TensorBase destructors never touch prepared state.
         */
        void releaseAllPreparedState();

        // Debug: dump all entries with tensor pointers for diagnosing lookup failures
        void dumpEntries(const char *prefix) const;

        // =========================================================================
        // MoE Expert Slab API
        // =========================================================================

        /// Register a new expert slab (one weight group × one layer × one device).
        ExpertSlabRef registerExpertSlab(const ExpertSlabDescriptor &desc);

        /// Get the GEMM engine for a specific expert within a slab.
        ITensorGemm *expertGemmKernel(const ExpertSlabRef &slab, int expert_id) const;

        /// Register newly-arrived expert engines (from initial load or rebalance transfer).
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
            // Phase 8: Store owns the prepared handle directly (not borrowed from KernelFactory).
            std::shared_ptr<llaminar::v2::kernels::KernelFactory::PreparedGemmHandle> owned_handle;

            /// Resolve the active handle (owned or legacy).
            const llaminar::v2::kernels::KernelFactory::PreparedGemmHandle *activeHandle() const
            {
                return owned_handle.get();
            }
        };

        /// Fused gate/up kernel cache (Phase 8: owned by this store, not KernelFactory)
        struct FusedCacheKey
        {
            uint64_t gate_binding_id = 0;
            uint64_t up_binding_id = 0;
            bool operator==(const FusedCacheKey &o) const
            {
                return gate_binding_id == o.gate_binding_id && up_binding_id == o.up_binding_id;
            }
        };
        struct FusedCacheHash
        {
            size_t operator()(const FusedCacheKey &k) const
            {
                return std::hash<uint64_t>{}(k.gate_binding_id) ^ (std::hash<uint64_t>{}(k.up_binding_id) << 32);
            }
        };

        PreparedWeightKind inferPreparedKind(DeviceId device) const;
        PreparedWeightRef makeRef(uint64_t binding_id, PreparedWeightKind kind, DeviceId device) const;

        ModelContextId model_id_;
        mutable std::mutex mutex_;
        std::unordered_map<uint64_t, Entry> entries_;
        mutable std::unordered_map<FusedCacheKey, std::unique_ptr<ITensorFusedGateUpGemm>, FusedCacheHash> fused_cache_;

        // =========================================================================
        // Embedding Storage (Phase 8: owned by this store)
        // =========================================================================

        struct EmbeddingEntry
        {
            WeightBinding binding;
            PreparedWeightRef ref;
            std::shared_ptr<PreparedEmbeddingHandle> owned_handle;
            const PreparedEmbeddingHandle *legacy_handle = nullptr;

            const PreparedEmbeddingHandle *activeHandle() const
            {
                return owned_handle ? owned_handle.get() : legacy_handle;
            }
        };

        std::unordered_map<uint64_t, EmbeddingEntry> embedding_entries_;

        // =========================================================================
        // Sliced GEMM Cache (Phase 8: owned by this store)
        // =========================================================================

        struct SlicedKey
        {
            uint64_t binding_id = 0;
            const TensorBase *tensor = nullptr;
            size_t row_start = 0;
            size_t row_end = 0;
            bool operator==(const SlicedKey &o) const
            {
                return binding_id == o.binding_id &&
                       tensor == o.tensor &&
                       row_start == o.row_start &&
                       row_end == o.row_end;
            }
        };
        struct SlicedKeyHash
        {
            size_t operator()(const SlicedKey &k) const
            {
                auto h = std::hash<uint64_t>{}(k.binding_id);
                h ^= std::hash<const void *>{}(k.tensor) << 8;
                h ^= std::hash<size_t>{}(k.row_start) << 16;
                h ^= std::hash<size_t>{}(k.row_end) << 32;
                return h;
            }
        };

        mutable std::unordered_map<SlicedKey, std::unique_ptr<ITensorGemm>, SlicedKeyHash> sliced_cache_;

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
