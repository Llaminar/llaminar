/**
 * @file ExpertGemmRegistry.h
 * @brief Thread-safe prepared MoE GEMM identity and lifetime registry.
 *
 * Weight preparation publishes device-, domain-, or participant-scoped
 * engines here. Graph construction may borrow raw pointers for fixed model
 * lifetime execution, while epoch-indexed ExpertOverlay residency explicitly
 * acquires shared ownership so old banks survive concurrent migration. Static
 * within-expert matrices additionally retain exact frozen source identities and
 * are available only through complete participant-scoped acquisition.
 */

#pragma once

#include "backends/DeviceId.h"
#include "WeightPlan.h"
#include "execution/moe/MoEExpertProjectionOwnership.h"

#include <memory>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{
    class PreparedWeightStore;
    struct MoEOverlayReusableContextSeal;
    class MoEOverlayParticipantResidencyRegistry;
    struct MoEOverlayParticipantResidencyBank;
    namespace test { struct ExpertGemmRegistryResidencyTestAccess; }


    class ITensorGemm;

    struct MoEExpertOwnerParticipant;

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

        /** @brief One acquired static layer; exact source and GEMM owners survive registry retirement. */
        struct StaticProjectionLayer
        {
            ParticipantLayerScope scope;
            MoEExpertProjectionOwnership projection;
            std::vector<ParticipantExpertBinding> experts; ///< Original ordinal E order; CPU unselected entries remain empty.
            std::vector<WeightBinding> sources; ///< Original frozen views, in selected expert-major G/U/D order.
            std::shared_ptr<const CPUExpertShardSourceSelection> cpu_joint_source; ///< Closed CPU fragment selection; dense experts has empty unselected entries.
        };

        /**
         * @brief Resolve an explicitly declared static source participant; never infer a domain.
         * @param source Original marked immutable parent or matrix binding.
         * @return Existing participant/layer key, including actual target device.
         * @throws std::invalid_argument for absent/foreign scope or non-static source ownership.
         */
        /** @brief Check exact original native source/window geometry, including retired metadata.
         * @param source Original frozen static matrix binding.
         * @note This validates source identity only; it grants no prepared-engine readiness.
         */
        static void validateStaticProjectionSource(const WeightBinding &source);
        [[nodiscard]] static ParticipantLayerScope staticProjectionScope(const WeightBinding &source);

        /**
         * @brief Register one original static matrix without publishing domain/device aliases.
         * @param source Exact frozen native 2D source with original N/K/E coordinates and scope.
         * @param engine Actual locally prepared engine, whose N/K must match the source contract.
         * @param owner Exact shared engine owner retaining its canonical packed pool.
         * @throws std::invalid_argument for invalid source, geometry or engine ownership.
         * @throws std::logic_error for any duplicate key or replacement by another source.
         */
        /**
         * @brief Publish a streamed CPU fragment only from its actual committed model store.
         * @param source Original retained identity/cohort and committed final binding.
         * @param store Same original model/PMA store issuing the completed binding/ref.
         * @throws std::invalid_argument For passive metadata, foreign source or another engine owner.
         * @note GPU registration retains its readable original-parent law below.
         */
        void registerCommittedCPUProjection(const WeightBinding &source, const PreparedWeightStore &store);

        void registerStaticProjection(const WeightBinding &source, ITensorGemm *engine,
                                      std::shared_ptr<ITensorGemm> owner);

        /**
         * @brief Acquire the exact full-E GPU or closed selected-ID CPU triplets from original frozen declarations.
         * @param scope Explicit original domain/device/rank/participant/layer.
         * @param projection Complete expected original F partition, not just local dimensions.
         * @param sources Exact frozen matrices: 3E for GPU or three per originally selected CPU expert.
         * @return Owned source-bound layer, or nullopt when no engine has been registered.
         * @throws std::invalid_argument for duplicate/incomplete/foreign source declarations.
         * @throws std::logic_error for a stale source owner, different partition or partial published layer.
         *
         * No unscoped or overlay alias is consulted. A failed query publishes no partial result.
         */
        [[nodiscard]] std::optional<StaticProjectionLayer> acquireStaticProjectionLayer(
            const ParticipantLayerScope &scope, const MoEExpertProjectionOwnership &projection,
            std::span<const WeightBinding> sources) const;

        /**
         * @brief Borrow only already-prepared CPU fragments from their retained original source declarations.
         * @param original_model Original model store lifetime, never an unscoped engine lookup.
         * @param original_participant Exact original rank/NUMA/domain/tier/member descriptor.
         * @param layer Original model layer, not a compact source index.
         * @return Source-bound selected layer lease, or nullopt when wholly absent.
         * @throws std::invalid_argument For invalid model/member or foreign/whole-owner sources.
         * @throws std::logic_error For an incomplete/stale prepared layer.
         *
         * Graph callers have no frozen tensor inventory of their own. This reads
         * the SAME declarations retained at registration and delegates complete
         * acquisition; it never reconstructs IDs from compact tensor geometry.
         */
        [[nodiscard]] std::optional<StaticProjectionLayer> acquireCPUJointLayer(
            ModelContextId original_model,const MoEExpertOwnerParticipant& original_participant,int layer) const;

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

        /**
         * @brief Publish a domain alias with the exact optional projection layout.
         * @param domain_name Logical overlay domain, or empty for legacy device scope.
         * @param device Physical execution device.
         * @param layer Model layer index.
         * @param expert Global router expert ID.
         * @param role Gate, up or down projection.
         * @param engine Borrowed execution handle published by this key.
         * @param ownership Exact shared owner retaining the handle and weight allocation.
         * @param projection Complete source/partition identity, absent for whole experts.
         * @throws std::invalid_argument for inconsistent partition/engine identity.
         * @throws std::logic_error if this key already names another layout.
         */
        void registerEngineForDomain(const std::string &domain_name,
                         DeviceId device, int layer, int expert, WeightRole role,
                         ITensorGemm *engine, std::shared_ptr<ITensorGemm> ownership,
                         std::optional<MoEExpertProjectionOwnership> projection = std::nullopt);

        /**
         * @brief Publish a participant engine and its immutable physical projection identity.
         * @param domain_name Logical overlay execution domain.
         * @param device Physical execution device.
         * @param participant_world_rank Resolved MPI owner rank, or -1 for a domain alias.
         * @param participant_index Global domain coordinate, or -1 for a domain alias.
         * @param layer Model layer index.
         * @param expert Global router expert ID.
         * @param role Gate, up or down projection.
         * @param engine Borrowed execution handle published by this key.
         * @param ownership Shared owner retaining that exact handle and allocation.
         * @param projection Exact full-source/partition contract; absent for complete experts.
         * @throws std::invalid_argument for inconsistent partition/engine identity.
         * @throws std::logic_error when an existing key names a different physical layout.
         *
         * A down slice must never be adopted by an ordinary whole-expert caller.
         * Shape alone cannot authenticate the original output-column interval.
         */
        void registerEngineForParticipant(const std::string &domain_name,
                 DeviceId device, int participant_world_rank, int participant_index,
                 int layer, int expert, WeightRole role,
                 ITensorGemm *engine, std::shared_ptr<ITensorGemm> ownership,
                 std::optional<MoEExpertProjectionOwnership> projection = std::nullopt);

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

        /**
         * @brief Borrow an engine from a layout-authenticated domain alias.
         * @param domain_name Logical overlay domain.
         * @param device Physical execution device.
         * @param layer Model layer index.
         * @param expert Global router expert ID.
         * @param role Gate, up or down projection.
         * @param projection Expected source/partition identity; absent means a whole expert.
         * @return Borrowed handle, or null only when the key is absent.
         * @throws std::logic_error if an existing key has another physical layout or requires static full-source acquisition.
         */
        ITensorGemm *getEngineForDomain(const std::string &domain_name,
                        DeviceId device, int layer, int expert, WeightRole role,
                        const std::optional<MoEExpertProjectionOwnership> &projection = std::nullopt) const;

        /**
         * @brief Borrow an engine only from the requested participant and layout.
         * @param domain_name Logical overlay domain.
         * @param device Physical execution device.
         * @param participant_world_rank Resolved MPI owner rank, or -1.
         * @param participant_index Stable global domain coordinate.
         * @param layer Model layer index.
         * @param expert Global router expert ID.
         * @param role Gate, up or down projection.
         * @param projection Expected source/partition identity; absent means a whole expert.
         * @return Borrowed handle, or null only when the key is absent.
         * @throws std::logic_error if an existing key has another physical layout.
         */
        ITensorGemm *getEngineForParticipant(const std::string &domain_name,
                DeviceId device, int participant_world_rank, int participant_index,
                int layer, int expert, WeightRole role,
                const std::optional<MoEExpertProjectionOwnership> &projection = std::nullopt) const;

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
         * @param projection Exact expected layout; absent requires a complete expert.
         * @return Exact engine lifetime, or null when absent or not owned.
         * @throws std::logic_error if an existing key has a different layout or requires static full-source acquisition.
         */
        [[nodiscard]] std::shared_ptr<ITensorGemm>
        getEngineLifetimeForDomain(
            const std::string &domain_name,
            DeviceId device,
            int layer,
            int expert,
            WeightRole role,
            const std::optional<MoEExpertProjectionOwnership> &projection = std::nullopt) const;

        /**
         * @brief Acquire shared ownership of one participant-scoped engine.
         * @param domain_name Logical overlay execution domain.
         * @param device Exact execution device.
         * @param participant_world_rank Resolved MPI owner rank, or -1.
         * @param participant_index Stable index inside the logical domain.
         * @param layer Transformer layer index.
         * @param expert Global expert id.
         * @param role Gate, up, or down projection.
         * @param projection Exact expected layout; absent requires a complete expert.
         * @return Exact engine lifetime, or null when absent or not owned.
         * @throws std::logic_error if an existing key has a different layout or requires static full-source acquisition.
         */
        [[nodiscard]] std::shared_ptr<ITensorGemm>
        getEngineLifetimeForParticipant(
            const std::string &domain_name,
            DeviceId device,
            int participant_world_rank,
            int participant_index,
            int layer,
            int expert,
            WeightRole role,
            const std::optional<MoEExpertProjectionOwnership> &projection = std::nullopt) const;

        /// Check complete whole-expert projections only; partitioned entries never satisfy this query.
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

        /// Check complete gate/up/down triplets only, excluding projection-partitioned layouts.
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
         * @brief Atomically rebind original prepared identities from their actual terminal seal.
         * @param seal Complete physically canonicalized initial-placement banks.
         * @param residency Same original participant/fixed-cohort declaration authority.
         * @param error Receives a precise failure; validation publishes no partial map.
         * @return True only after one atomic publication of all process-local scopes.
         *
         * Whole payloads keep their original domain aliases. CPU fragments keep
         * exact participant keys and original source/Store/layout declarations;
         * only their actual sealed engine owners change. No whole-expert alias
         * or source identity is manufactured from a fragment triplet. Other
         * fixed GPU projection slices remain unsupported and reject unchanged.
         * This terminal operation requires original inference/graph quiescence.
         */
        [[nodiscard]] bool replaceParticipantResidencyFromSeal(
            const MoEOverlayReusableContextSeal &seal,
            const MoEOverlayParticipantResidencyRegistry &residency,
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
        friend struct test::ExpertGemmRegistryResidencyTestAccess;
        /**
         * @brief Stage the single registry replacement; unsealed inputs are test-only.
         * @param scopes Complete checked participant/layer surface.
         * @param bindings Checked whole payloads; never CPU fragment adapters.
         * @param banks Canonical banks already authorized by the public seal; component tests may corrupt this input.
         * @param residency Original declaration authority paired with the banks, null only for untyped corruption tests.
         * @param error Receives validation/allocation failure before atomic swap.
         * @return Whether one complete replacement was published.
         */
        bool replaceParticipantResidencyImpl(
            std::span<const ParticipantLayerScope> scopes,
            std::span<const ParticipantExpertBinding> bindings,
            std::span<const MoEOverlayParticipantResidencyBank> banks,
            const MoEOverlayParticipantResidencyRegistry *residency,
            std::string *error) noexcept;

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
            std::optional<MoEExpertProjectionOwnership> projection_ownership;
            std::shared_ptr<const WeightBinding> static_source = {}; ///< Exact original frozen source, absent for old modes.
        };

        mutable std::shared_mutex mutex_;
        std::unordered_map<Key, Entry, KeyHash> engines_;
    };

} // namespace llaminar2
