/**
 * @file Qwen4ExpReplicatedPLEQuery.h
 * @brief Immutable projection of an actual DGO query producer into its Rank cohort.
 *
 * This value borrows existing native tokens/count/cache; it owns no request,
 * publication cursor or mutable history. Only DGO can mint it after checking
 * its canonical input bank. Graph construction may copy the metadata while
 * retaining exactly the original count owner, including physical M1/live0.
 */
#pragma once
#include "kernels/common/DeviceRowRange.h"
#include "backends/DeviceId.h"
#include <cstdint>
#include <stdexcept>
namespace llaminar2 {
class DeviceGraphOrchestrator;class Qwen4ExpReplicatedPLECohort;class IKVCache;
enum class ForwardExecutionRole : std::uint8_t;enum class ForwardExecutionPhase : std::uint8_t;
/** @brief Actual producer role, independent of the current values behind its pointers. */
enum class Qwen4ExpPLEQuerySource { Request, PrefillChunk, OrdinaryFrontier, Verifier, ConditionFrontier, WorkspaceDeclaration };
/** @brief Read-only local input proof; copying it cannot relabel another bank or cache. */
class Qwen4ExpReplicatedPLEQuery final {
public:
    /** @return Whether this projection describes workspace geometry without granting executable query authority. */
    bool isWorkspaceDeclaration() const noexcept {return source_==Qwen4ExpPLEQuerySource::WorkspaceDeclaration;}
    /** @brief Reject direct stage preparation/submission of a geometry-only declaration. */
    void requireExecutable() const
    {
        if(isWorkspaceDeclaration())throw std::invalid_argument("Fixed-head workspace query declaration is not executable");
    }
    /** @return Original complete query rows, never a transport-derived count. */
    const DeviceRowRange &rows() const noexcept {return rows_;}
    /** @return Actual native token producer, retained by its DGO arena. */
    const std::int32_t *tokens() const noexcept {return tokens_;}
    /** @return Canonical main cache whose accepted lexical history is consumed. */
    const IKVCache *cache() const noexcept {return cache_;}
    /** @return Frozen semantic source role; not a user-selected replication claim. */
    Qwen4ExpPLEQuerySource source() const noexcept {return source_;}
    /** @return Actual immutable Rank cohort, never a caller boolean. */
    const Qwen4ExpReplicatedPLECohort *cohort() const noexcept {return cohort_;}
    /** @return Actual participant device, retained with its DGO producer. */
    DeviceId device() const noexcept {return device_;}
    /** @return Original collective/head coordinate, independent of physical ordinal. */
    int participant() const noexcept {return participant_;}
    /** @return Complete declaration identity; changing token/count/history values are excluded. */
    std::uint64_t captureIdentity() const noexcept {
        std::uint64_t value=1469598103934665603ULL;
        const auto mix=[&](std::uint64_t word){value^=word;value*=1099511628211ULL;};
        mix(1);mix(reinterpret_cast<std::uintptr_t>(cohort_));mix(reinterpret_cast<std::uintptr_t>(owner_));
        mix(static_cast<std::uint64_t>(device_.type));mix(device_.ordinal);mix(participant_);
        mix(reinterpret_cast<std::uintptr_t>(tokens_));mix(reinterpret_cast<std::uintptr_t>(cache_));
        mix(static_cast<std::uint64_t>(role_));mix(static_cast<std::uint64_t>(phase_));mix(static_cast<std::uint64_t>(source_));
        const auto rows=rows_.captureIdentity();mix(reinterpret_cast<std::uintptr_t>(rows.count));mix(rows.count_capacity);
        mix(rows.first_physical_row);mix(rows.physical_rows);mix(rows.repeats_per_row);mix(rows.schema_version);
        return value?value:1;
    }
private:
    friend class DeviceGraphOrchestrator;friend class Qwen4ExpReplicatedPLECohort;
    /** @brief Mint only after the actual DGO validates its exact canonical bank and role. */
    Qwen4ExpReplicatedPLEQuery(const Qwen4ExpReplicatedPLECohort *cohort,const DeviceGraphOrchestrator *owner,
        DeviceId device,int participant,const std::int32_t *tokens,DeviceRowRange rows,const IKVCache *cache,
        ForwardExecutionRole role,ForwardExecutionPhase phase,Qwen4ExpPLEQuerySource source)
        :cohort_(cohort),owner_(owner),device_(device),participant_(participant),tokens_(tokens),rows_(rows),
         cache_(cache),role_(role),phase_(phase),source_(source){}
    const Qwen4ExpReplicatedPLECohort *cohort_;const DeviceGraphOrchestrator *owner_;
    DeviceId device_;int participant_;const std::int32_t *tokens_;DeviceRowRange rows_;const IKVCache *cache_;
    ForwardExecutionRole role_;ForwardExecutionPhase phase_;Qwen4ExpPLEQuerySource source_;
};
} // namespace llaminar2
