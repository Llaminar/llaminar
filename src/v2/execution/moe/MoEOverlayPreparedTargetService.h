/**
 * @file MoEOverlayPreparedTargetService.h
 * @brief Prepared-input to complete-canonical-output evidence over original targets.
 *
 * The retained snapshot/catalog declare metadata only. They grant no prepared
 * engine, native-domain or publication permit. CPU/GPU producers retain their
 * original authorities. One cohort is one price, never a set of member prices.
 * Dispatch/ingress is excluded; cohort start release/gather/fold is included.
 */
#pragma once
#include "MoEOverlayRouteTarget.h"
#include "ExpertHistogramSource.h"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <compare>
#include <span>
#include <vector>

namespace llaminar2 {
class MoEOverlayResidencySnapshot;
class MoEOverlayEconomyCalibrationLayerCatalog;
class MoEExpertShardCohort;
struct ExpertHistogramProductionTopology;

/** @brief Complete logical execution vertex and original model layer. */
struct MoEOverlayTargetServiceCoordinate {
    MoEOverlayRouteTarget target;
    int layer=-1;
    /** @return Exact signed-word/layer ordering, not representative participant order. */
    auto operator<=>(const MoEOverlayTargetServiceCoordinate& other) const noexcept
    {return std::pair{target.word(),layer}<=>std::pair{other.target.word(),other.layer};}
    /** @return Exact target/layer equality. */
    bool operator==(const MoEOverlayTargetServiceCoordinate& other) const noexcept
    {return target.word()==other.target.word() && layer==other.layer;}
};

/**
 * @brief Prepared-only actual samples; not convertible from live packet totals.
 * Every populated phase comes from one declared sample layer/global expert.
 * Durations are elapsed service times; CPU monotonic/GPU event clocks are not
 * subtracted across participants. No sender supplies target geometry.
 */
struct MoEOverlayPreparedTargetServiceTotals {
    MoEOverlayTargetServiceCoordinate coordinate;
    std::array<uint64_t,kExpertHistogramProductionSourceCount> total_nanoseconds{};
    std::array<uint64_t,kExpertHistogramProductionSourceCount> activation_count{};
    std::array<uint64_t,kExpertHistogramProductionSourceCount> sample_count{};
    std::array<int,kExpertHistogramProductionSourceCount> expert_ids = [] {
        std::array<int,kExpertHistogramProductionSourceCount> result{};result.fill(-1);return result;
    }();
};

/** @brief Immutable original metadata driving setup jobs, wire coverage and normalization. */
class MoEOverlayPreparedTargetServicePlan final {
public:
    /** @brief One canonical target/layer row and its declared equivalent phase samples. */
    struct Entry {
        MoEOverlayTargetServiceCoordinate coordinate;
        int tier=-1;
        int root_world_rank=-1;
        std::array<int,kExpertHistogramProductionSourceCount> sample_layers{};
        std::array<uint64_t,kExpertHistogramProductionSourceCount> sample_rows{};
        uint64_t measured_invocations=0;
    };
    /**
     * @brief Check complete original geometry without minting execution readiness.
     * @param snapshot Original sealed setup placement/source definition.
     * @param catalog Original full-source equivalence catalog.
     * @param topology Exact retained recurring phase reachability.
     * @return One metadata owner, shared through producer/exchange/composer.
     * @throws std::exception For foreign/incomplete/overlapping target geometry.
     */
    static std::shared_ptr<const MoEOverlayPreparedTargetServicePlan> checked(
        std::shared_ptr<const MoEOverlayResidencySnapshot> snapshot,
        std::shared_ptr<const MoEOverlayEconomyCalibrationLayerCatalog> catalog,
        const ExpertHistogramProductionTopology& topology);
    /** @param borrow Proposed retained metadata owner. @return Exact original object/control-block ownership. */
    bool authenticates(const std::shared_ptr<const MoEOverlayPreparedTargetServicePlan>& borrow) const noexcept {
        const auto original=original_owner_.lock();return original && borrow.get()==this &&
            !original.owner_before(borrow) && !borrow.owner_before(original);
    }
    /** @return One canonical target/layer list; physical members remain in the snapshot. */
    const std::vector<Entry>& entries() const noexcept {return entries_;}
    /** @return Exact original source snapshot; no candidate or mutable owner map. */
    const std::shared_ptr<const MoEOverlayResidencySnapshot>& snapshot() const noexcept {return snapshot_;}
    /** @return Original native/shape class partition. */
    const std::shared_ptr<const MoEOverlayEconomyCalibrationLayerCatalog>& catalog() const noexcept {return catalog_;}
    /** @return Original retained graph/economy phase declaration. */
    const ExpertHistogramProductionTopology& topology() const noexcept;
    /** @param coordinate Original target/layer. @return Its checked declaration. @throws std::invalid_argument For absence. */
    const Entry& entry(MoEOverlayTargetServiceCoordinate coordinate) const;
    /** @param entry Original declaration. @return Same original cohort or null for whole targets. */
    std::shared_ptr<const MoEExpertShardCohort> cohort(const Entry& entry) const;
    /**
     * @brief Validate actual root-owned sampled coordinates before any exchange/normalization.
     * @param rows Exact live local rows. @param world_rank Original observer rank.
     * @throws std::invalid_argument For missing/duplicate/wrong-root/foreign-phase samples.
     */
    void requireLocalSamples(std::span<const MoEOverlayPreparedTargetServiceTotals> rows,int world_rank) const;
    /** @param entry Original target/layer declaration. @param phase Priced production phase.
     * @return Canonical original resident expert for this actual sample job, never a compact/member ID. */
    int sampleExpert(const Entry& entry,size_t phase) const;
    /** @return Deterministic lightweight metadata identity; never a payload hash. */
    const std::string& identity() const noexcept {return identity_;}
private:
    /** @brief Closed metadata construction after all declarations validate. */
    MoEOverlayPreparedTargetServicePlan(std::shared_ptr<const MoEOverlayResidencySnapshot> snapshot,
        std::shared_ptr<const MoEOverlayEconomyCalibrationLayerCatalog> catalog,
        std::shared_ptr<const ExpertHistogramProductionTopology> topology,std::vector<Entry> entries,std::string identity)
        :snapshot_(std::move(snapshot)),catalog_(std::move(catalog)),topology_(std::move(topology)),
         entries_(std::move(entries)),identity_(std::move(identity)){}
    std::weak_ptr<const MoEOverlayPreparedTargetServicePlan> original_owner_;
    std::shared_ptr<const MoEOverlayResidencySnapshot> snapshot_;
    std::shared_ptr<const MoEOverlayEconomyCalibrationLayerCatalog> catalog_;
    std::shared_ptr<const ExpertHistogramProductionTopology> topology_;
    std::vector<Entry> entries_;
    std::string identity_;
};
} // namespace llaminar2
