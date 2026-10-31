/**
 * @file PreparedWeightPhysicalUses.h
 * @brief Lower logical weight uses to physical creators and authenticated alias edges.
 *
 * This setup-only compiler retains source identities and edges, never bytes,
 * allocator state or residency credit. The existing estimator prices creators
 * into the ordinary PhysicalMemoryAuthority BOM. The original profile and its
 * immutable inventory must outlive one compilation; separate profiles cannot
 * donate roots merely because their tensor names or dimensions agree.
 */
#pragma once
#include "backends/DeviceId.h"
#include "planning/WeightMemoryEstimator.h"
#include "loaders/PreparedWeightIdentityContract.h"
#include <optional>
#include <variant>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace llaminar2
{
    struct ModelMemoryProfile;
    struct TensorSizeInfo;
    class WeightShardGeometry;
    enum class WeightSourceStorageView;
    /** @brief Primary publication precedes any attempt to borrow its exact root. */
    enum class PreparedWeightUseIntent { PublishCreator, ReuseExistingOrCreate };
    /** @brief A required independently priced physical creator. */
    struct PreparedWeightPhysicalCreator { size_t root; };
    /** @brief A logical use retaining the earlier exact creator rather than allocating again. */
    struct PreparedWeightPhysicalAlias { size_t root; };
    /** @brief Inspectable immutable decision; allocation authority remains exclusively PMA. */
    using PreparedWeightPhysicalUse = std::variant<PreparedWeightPhysicalCreator, PreparedWeightPhysicalAlias>;

    /** @brief Original-inventory/device-bound physical-use compiler for one BOM. */
    class PreparedWeightPhysicalUses final
    {
    public:
        /**
         * @brief Bind the original immutable source inventory and physical participant.
         * @param profile Canonical original model metadata, not a synthesized subset.
         * @param device Participant whose physical creator roots are being compiled.
         * @param components Actual graph-owned global components, shared with runtime selection.
         */
        PreparedWeightPhysicalUses(const ModelMemoryProfile &profile, DeviceId device,
            WeightComponentScope components = WeightComponentScope::EmbeddingAndTerminal);
        /**
         * @brief Publish a creator or retain an earlier proved complete GEMM root.
         * @param profile Must be the exact originally bound model metadata object.
         * @param device Must be the exact originally bound physical participant.
         * @param tensor Original inventory entry; synthesized/foreign entries cannot alias.
         * @param geometry Exact schema-directed shape for this particular logical use.
         * @param source_view Actual source-loading declaration resolved by WeightShardGeometryResolver.
         * @param intent Primary publication or subsequent alias-or-create intent.
         * @return Typed creator/alias edge, also retained in edges() for inspection.
         * @throws std::invalid_argument on foreign profile/device compilation scope.
         *
         * Partial or projected geometry without original interval/owner evidence
         * remains a distinct required creator. Equal names alone are never proof.
         */
        PreparedWeightPhysicalUse record(const ModelMemoryProfile &profile, DeviceId device,
            const TensorSizeInfo &tensor, const WeightShardGeometry &geometry, WeightSourceStorageView source_view,
            PreparedWeightUseIntent intent);
        /** @return Ordered decisions in exact logical-use compilation order. */
        const std::vector<PreparedWeightPhysicalUse> &edges() const noexcept { return edges_; }
        /** @return Required physical creator count; no byte/occupancy accounting. */
        size_t creatorCount() const noexcept { return roots_.size(); }
        /** @return Exact original inventory entry, or null for a synthesized/unproven source. */
        const TensorSizeInfo *originalSource(size_t root) const { return roots_.at(root).original; }
    private:
        friend class WeightMemoryEstimator;
        /** @brief Reject a different model, allocator or component authority before pricing any use. */
        void requireScope(const ModelMemoryProfile &profile, DeviceId device, WeightComponentScope components) const;
        /** @brief One original source and representation, retained solely for exact alias matching. */
        struct Root
        {
            const TensorSizeInfo *original;
            std::optional<PreparedWeightIdentityContract> identity;
        };
        /** @brief Produce a shareable identity only for a proved complete original GEMM view. */
        std::optional<PreparedWeightIdentityContract> identity(const TensorSizeInfo &tensor,
            const WeightShardGeometry &geometry, WeightSourceStorageView source_view) const;
        const ModelMemoryProfile &profile_;
        DeviceId device_;
        WeightComponentScope components_;
        std::unordered_set<std::string> terminal_sources_;
        std::unordered_set<const TensorSizeInfo *> originals_;
        std::unordered_map<const TensorSizeInfo *, std::vector<size_t>> roots_by_source_;
        std::vector<Root> roots_;
        std::vector<PreparedWeightPhysicalUse> edges_;
    };
}
