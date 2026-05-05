#pragma once

#include "backends/DeviceId.h"

#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    class ITensorGemm;

    /// Registry for pre-prepared MoE expert GEMM engines.
    /// Populated by the unified GPU weight pipeline, queried by graph builders.
    /// Thread-safe for concurrent reads, exclusive writes.
    class ExpertGemmRegistry
    {
    public:
        enum class WeightRole : uint8_t
        {
            GATE = 0,
            UP = 1,
            DOWN = 2
        };

        ExpertGemmRegistry() = default;
        ~ExpertGemmRegistry() = default;

        // Non-copyable, non-movable (owned by WeightManager)
        ExpertGemmRegistry(const ExpertGemmRegistry &) = delete;
        ExpertGemmRegistry &operator=(const ExpertGemmRegistry &) = delete;

        /// Register a single expert GEMM engine.
        /// @param device Target device
        /// @param layer Layer index
        /// @param expert Expert index
        /// @param role Gate/Up/Down
        /// @param engine Raw pointer (for fast lookup)
        /// @param ownership Shared pointer keeping VRAM pool alive
        void registerEngine(DeviceId device, int layer, int expert, WeightRole role,
                            ITensorGemm *engine, std::shared_ptr<ITensorGemm> ownership);

        /// Look up a single expert GEMM engine. Returns nullptr if not found.
        ITensorGemm *getEngine(DeviceId device, int layer, int expert, WeightRole role) const;

        /// Check if a full role is registered for every expert in a layer.
        bool hasCompleteRole(DeviceId device, int layer, int num_experts, WeightRole role) const;

        /// Check if gate/up/down engines are registered for every expert in a layer.
        bool hasCompleteLayer(DeviceId device, int layer, int num_experts) const;

        /// Bulk populate vectors for a graph builder's MoE stage params.
        /// Resizes output vectors to num_experts and fills with registered engines (nullptr for missing).
        /// Returns true only when all gate/up/down engines for all requested experts are present.
        bool populateExpertEngines(DeviceId device, int layer, int num_experts,
                                   std::vector<ITensorGemm *> &gate_out,
                                   std::vector<ITensorGemm *> &up_out,
                                   std::vector<ITensorGemm *> &down_out) const;

        /// Replace an existing engine (for dynamic rebalancing arrival).
        /// If no existing engine, equivalent to registerEngine.
        void replaceEngine(DeviceId device, int layer, int expert, WeightRole role,
                           ITensorGemm *engine, std::shared_ptr<ITensorGemm> ownership);

        /// Remove an engine (for dynamic rebalancing departure).
        /// Returns true if an engine was found and removed.
        bool removeEngine(DeviceId device, int layer, int expert, WeightRole role);

        /// Total number of registered engines.
        size_t size() const;

        /// Check if any engines are registered for a specific device and layer.
        bool hasEnginesForLayer(DeviceId device, int layer) const;

        /// Clear all entries, releasing all shared_ptr ownership.
        void clear();

    private:
        struct Key
        {
            DeviceId device;
            int layer;
            int expert;
            WeightRole role;

            bool operator==(const Key &other) const;
        };

        struct KeyHash
        {
            size_t operator()(const Key &k) const;
        };

        struct Entry
        {
            ITensorGemm *engine = nullptr;
            std::shared_ptr<ITensorGemm> ownership;
        };

        mutable std::shared_mutex mutex_;
        std::unordered_map<Key, Entry, KeyHash> engines_;
    };

} // namespace llaminar2
