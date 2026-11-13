/**
 * @file MoEOptimizationMovementEndpoint.h
 * @brief One immutable completed-movement endpoint with whole or original cohort ownership.
 *
 * A cohort borrows its complete original source/member/cut definition. No member
 * represents the cohort and no copied owner table or runtime permit lives here.
 * Scalar native evidence must explicitly request the checked whole projection.
 */
#pragma once
#include "backends/DeviceId.h"
#include "MoEOverlayRouteTarget.h"
#include <memory>
#include <optional>
#include <variant>
namespace llaminar2 {
class MoEExpertShardCohort;
struct MoEExpertOwner;
/** @brief Closed complete-target alternatives retained by the authority's immutable ledger. */
class MoEOptimizationMovementEndpoint final {
public:
    /** @brief Original single participant and its observed rank identity. */
    struct Whole {
        int participant;
        DeviceId device;
        std::optional<int> world_rank;
        bool operator==(const Whole&) const = default;
    };
    /** @brief Original complete cohort, in its canonical reduction/member order. */
    struct CPUCohort {
        int tier;
        std::shared_ptr<const MoEExpertShardCohort> original;
    };
    /** @brief Unbound records cannot pass edge validation. */
    MoEOptimizationMovementEndpoint() = default;
    /** @param participant Original scalar participant. @param device Original physical allocator.
     * @param world_rank Actual known rank, or explicit absence.
     * @return Checked whole endpoint. @throws std::invalid_argument For invalid identity. */
    static MoEOptimizationMovementEndpoint whole(int participant,DeviceId device,std::optional<int> world_rank=std::nullopt);
    /** @param owner Original canonical placement, including its original cohort borrow.
     * @return Exact complete endpoint. @throws std::invalid_argument For incomplete ownership. */
    static MoEOptimizationMovementEndpoint fromOwner(const MoEExpertOwner& owner);
    /** @return Whether this endpoint is an original whole participant. */
    bool isWhole() const noexcept {return std::holds_alternative<Whole>(value_);}
    /** @return Whether this endpoint is one complete original CPU cohort. */
    bool isCPUCohort() const noexcept {return std::holds_alternative<CPUCohort>(value_);}
    /** @return Checked scalar projection. @throws std::logic_error For absent or cohort ownership. */
    const Whole& whole() const;
    /** @return Complete original metadata. @throws std::logic_error For absent or whole ownership. */
    const CPUCohort& cohort() const;
    /** @return Same canonical typed route alternative; unbound remains absent. */
    MoEOverlayRouteTarget target() const noexcept;
    /** @return Whether the immutable alternative is complete and originally bound. */
    bool valid() const noexcept;
    /** @param layer Original model layer. @return Whole identity or exact original cohort source-layer agreement. */
    bool validForLayer(int layer) const noexcept;
    /** @param other Original endpoint. @return Structural source/member/cut equality, never representative equality. */
    bool operator==(const MoEOptimizationMovementEndpoint& other) const;
private:
    /** @brief Seal one checked alternative without exposing mutable scalar arms. */
    explicit MoEOptimizationMovementEndpoint(Whole value):value_(std::move(value)){}
    /** @brief Seal the original immutable cohort borrow after source checks. */
    explicit MoEOptimizationMovementEndpoint(CPUCohort value):value_(std::move(value)){}
    std::variant<std::monostate,Whole,CPUCohort> value_;
};
}
