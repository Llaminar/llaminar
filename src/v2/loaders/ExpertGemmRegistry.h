/**
 * @file ExpertGemmRegistry.h
 * @brief Thread-safe prepared MoE GEMM identity and lifetime registry.
 *
 * Weight preparation publishes device-, domain-, or participant-scoped
 * engines here. Graph construction may borrow raw pointers for fixed model
 * lifetime execution, while epoch-indexed ExpertOverlay residency explicitly
 * acquires shared ownership so old banks survive concurrent migration.
 */

#pragma once

#include "backends/DeviceId.h"

#include <memory>
#include <shared_mutex>
#include <span>
#include <string>
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

        /**
         * @brief One participant/layer whose complete resident set is replaced.
         *
         * Context-reuse sealing supplies every process-local scope, including
         * an empty layer, so stale expert keys cannot survive a migration. The
         * exact participant identity mirrors the keys used by graph lowering.
         */
        struct ParticipantLayerScope
        {
            std::string domain_name;
            DeviceId device = DeviceId::invalid();
            int participant_world_rank = -1;
            int participant_index = -1;
            int layer = -1;

            /** @brief Compare the complete participant/layer key. */
            bool operator==(const ParticipantLayerScope &) const = default;
        };

        /**
         * @brief Complete prepared engine triplet for one restored resident.
         *
         * Shared pointers may be aliases whose control block owns a recyclable
         * CPU/GPU slot. Retaining those exact pointers in this registry makes
         * the prepared model context the post-run lifetime authority without
         * copying or repacking weights again.
         */
        struct ParticipantExpertBinding
        {
            ParticipantLayerScope scope;
            int expert = -1;
            std::shared_ptr<ITensorGemm> gate;
            std::shared_ptr<ITensorGemm> up;
            std::shared_ptr<ITensorGemm> down;

            /** @return Whether the binding names all three prepared roles. */
            [[nodiscard]] bool complete() const noexcept
            {
                return gate != nullptr && up != nullptr && down != nullptr;
            }
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

        /// Register a single expert GEMM engine under a logical overlay domain.
        void registerEngineForDomain(const std::string &domain_name,
                         DeviceId device, int layer, int expert, WeightRole role,
                         ITensorGemm *engine, std::shared_ptr<ITensorGemm> ownership);

        /// Register a single expert GEMM engine under a logical overlay domain and participant.
        void registerEngineForParticipant(const std::string &domain_name,
                 DeviceId device, int participant_world_rank, int participant_index,
                 int layer, int expert, WeightRole role,
                 ITensorGemm *engine, std::shared_ptr<ITensorGemm> ownership);

        /// Alias an existing device-scoped engine into a logical overlay domain.
        /// Copies the owning shared_ptr so the aliased key preserves lifetime.
        bool aliasEngineForDomainFromDevice(const std::string &domain_name,
             DeviceId device, int layer, int expert, WeightRole role);

        /// Alias an existing device-scoped engine into a logical overlay domain participant.
        bool aliasEngineForParticipantFromDevice(const std::string &domain_name,
             DeviceId device, int participant_world_rank, int participant_index,
             int layer, int expert, WeightRole role);

        /// Look up a single expert GEMM engine. Returns nullptr if not found.
        ITensorGemm *getEngine(DeviceId device, int layer, int expert, WeightRole role) const;

        /// Look up a single expert GEMM engine in a logical overlay domain.
        ITensorGemm *getEngineForDomain(const std::string &domain_name,
                        DeviceId device, int layer, int expert, WeightRole role) const;

        ITensorGemm *getEngineForParticipant(const std::string &domain_name,
                DeviceId device, int participant_world_rank, int participant_index,
                int layer, int expert, WeightRole role) const;

        /**
         * @brief Acquire shared ownership of one device-scoped engine.
         * @param device Exact execution device.
         * @param layer Transformer layer index.
         * @param expert Global expert id.
         * @param role Gate, up, or down projection.
         * @return Exact engine lifetime, or null when absent or not owned.
         */
        [[nodiscard]] std::shared_ptr<ITensorGemm> getEngineLifetime(
            DeviceId device,
            int layer,
            int expert,
            WeightRole role) const;

        /**
         * @brief Acquire shared ownership of one domain-scoped engine.
         * @param domain_name Logical overlay execution domain.
         * @param device Exact execution device.
         * @param layer Transformer layer index.
         * @param expert Global expert id.
         * @param role Gate, up, or down projection.
         * @return Exact engine lifetime, or null when absent or not owned.
         */
        [[nodiscard]] std::shared_ptr<ITensorGemm>
        getEngineLifetimeForDomain(
            const std::string &domain_name,
            DeviceId device,
            int layer,
            int expert,
            WeightRole role) const;

        /**
         * @brief Acquire shared ownership of one participant-scoped engine.
         * @param domain_name Logical overlay execution domain.
         * @param device Exact execution device.
         * @param participant_world_rank Resolved MPI owner rank, or -1.
         * @param participant_index Stable index inside the logical domain.
         * @param layer Transformer layer index.
         * @param expert Global expert id.
         * @param role Gate, up, or down projection.
         * @return Exact engine lifetime, or null when absent or not owned.
         */
        [[nodiscard]] std::shared_ptr<ITensorGemm>
        getEngineLifetimeForParticipant(
            const std::string &domain_name,
            DeviceId device,
            int participant_world_rank,
            int participant_index,
            int layer,
            int expert,
            WeightRole role) const;

        /// Check if a full role is registered for every expert in a layer.
        bool hasCompleteRole(DeviceId device, int layer, int num_experts, WeightRole role) const;

        bool hasCompleteRoleForDomain(const std::string &domain_name,
                          DeviceId device, int layer, int num_experts, WeightRole role) const;

        /// Check if a role is registered for a specific expert subset.
        bool hasCompleteRoleForExperts(DeviceId device, int layer,
                                       const std::vector<int> &expert_ids,
                                       WeightRole role) const;

        bool hasCompleteRoleForExpertsInDomain(const std::string &domain_name,
                               DeviceId device, int layer,
                               const std::vector<int> &expert_ids,
                               WeightRole role) const;

        /// Check if gate/up/down engines are registered for every expert in a layer.
        bool hasCompleteLayer(DeviceId device, int layer, int num_experts) const;

        bool hasCompleteLayerInDomain(const std::string &domain_name,
                          DeviceId device, int layer, int num_experts) const;

        /// Complete experts for a layer, where gate/up/down are all present.
        std::vector<int> completeExpertsForLayer(DeviceId device, int layer, int num_experts) const;
        std::vector<int> completeExpertsForLayerInDomain(const std::string &domain_name,
                                 DeviceId device, int layer, int num_experts) const;
        size_t countCompleteExpertsForLayer(DeviceId device, int layer, int num_experts) const;
        size_t countCompleteExpertsForLayerInDomain(const std::string &domain_name,
                                DeviceId device, int layer, int num_experts) const;

        /// Number of engines registered for a device, optionally constrained to a layer.
        size_t countEnginesForDevice(DeviceId device) const;
        size_t countEnginesForDeviceInDomain(const std::string &domain_name, DeviceId device) const;
        /**
         * @brief Count owned engine records for a device across every scope.
         *
         * ExpertOverlay publishes domain- and participant-scoped keys rather
         * than the legacy unscoped key. This model-lifetime accounting query
         * intentionally includes all of them; aliases may count more than once,
         * so callers use it only as residency evidence, never a byte estimate.
         */
        size_t countOwnedEnginesForDeviceAcrossScopes(DeviceId device) const;
        size_t countEnginesForLayer(DeviceId device, int layer) const;
        size_t countEnginesForLayerInDomain(const std::string &domain_name, DeviceId device, int layer) const;

        /// Bulk populate vectors for a graph builder's MoE stage params.
        /// Resizes output vectors to num_experts and fills with registered engines (nullptr for missing).
        /// Returns true only when all gate/up/down engines for all requested experts are present.
        bool populateExpertEngines(DeviceId device, int layer, int num_experts,
                                   std::vector<ITensorGemm *> &gate_out,
                                   std::vector<ITensorGemm *> &up_out,
                                   std::vector<ITensorGemm *> &down_out) const;

        bool populateExpertEnginesForDomain(const std::string &domain_name,
                            DeviceId device, int layer, int num_experts,
                            std::vector<ITensorGemm *> &gate_out,
                            std::vector<ITensorGemm *> &up_out,
                            std::vector<ITensorGemm *> &down_out) const;

        bool populateExpertEnginesForParticipant(const std::string &domain_name,
                            DeviceId device, int participant_world_rank, int participant_index,
                            int layer, int num_experts,
                            std::vector<ITensorGemm *> &gate_out,
                            std::vector<ITensorGemm *> &up_out,
                            std::vector<ITensorGemm *> &down_out) const;

        /**
         * @brief Atomically replace process-local ExpertOverlay residency keys.
         *
         * Every supplied scope is first removed from a private copy of the
         * registry, then rebuilt from @p bindings. Domain aliases are rebuilt
         * from the same triplets so setup-time format discovery and
         * participant graph lowering observe one identity. The live map is
         * swapped only after complete validation and allocation succeed; a
         * caller can never observe a partially rebound model context.
         *
         * This is a terminal model-lifecycle operation. Inference and graph
         * construction must already be quiescent, although ordinary readers
         * remain protected by the registry mutex.
         *
         * @param scopes Complete local participant/layer replacement surface.
         * @param bindings Exact resident triplets contained by those scopes.
         * @param error Optional precise validation/allocation diagnostic.
         * @return True after one atomic replacement, false without mutation.
         */
        [[nodiscard]] bool replaceParticipantResidency(
            std::span<const ParticipantLayerScope> scopes,
            std::span<const ParticipantExpertBinding> bindings,
            std::string *error = nullptr) noexcept;

        /// Replace an existing engine (for dynamic rebalancing arrival).
        /// If no existing engine, equivalent to registerEngine.
        void replaceEngine(DeviceId device, int layer, int expert, WeightRole role,
                           ITensorGemm *engine, std::shared_ptr<ITensorGemm> ownership);
        void replaceEngineForDomain(const std::string &domain_name,
                        DeviceId device, int layer, int expert, WeightRole role,
                        ITensorGemm *engine, std::shared_ptr<ITensorGemm> ownership);

        /// Remove an engine (for dynamic rebalancing departure).
        /// Returns true if an engine was found and removed.
        bool removeEngine(DeviceId device, int layer, int expert, WeightRole role);
        bool removeEngineForDomain(const std::string &domain_name,
                       DeviceId device, int layer, int expert, WeightRole role);

        /// Total number of registered engines.
        size_t size() const;

        /// Check if any engines are registered for a specific device and layer.
        bool hasEnginesForLayer(DeviceId device, int layer) const;
        bool hasEnginesForLayerInDomain(const std::string &domain_name, DeviceId device, int layer) const;

        /// Clear all entries, releasing all shared_ptr ownership.
        void clear();

    private:
        struct Key
        {
            std::string domain_name;
            DeviceId device;
            int layer;
            int expert;
            WeightRole role;
            int participant_world_rank = -1;
            int participant_index = -1;

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
