/**
 * @file PooledImmutableRetirementAuthority.h
 * @brief Closed original cohort authorities authorizing distributed source reset and terminal publication.
 *
 * Only the complete component cohort and the model-owned retained participants
 * can construct this base. Callers cannot mint a successful reset capability by
 * supplying a callback, executable pointer or arbitrary derived implementation.
 */
#pragma once
#include <memory>
#include <chrono>
namespace llaminar2 {class RetainedGraphParticipantParentStream;}
namespace llaminar2::qwen4exp {
class PooledImmutableCompleteCohort;
class PooledImmutableRetainedParticipants;
class PooledImmutableSourcePreparedStorage;
/** @brief Closed shared authority for actual original participant capture and reader lifetime. */
class PooledImmutableRetirementAuthority {
public:
    /** @brief Keep actual captures and roots until their original native retirement frontier. */
    virtual ~PooledImmutableRetirementAuthority()=default;
    /** @brief Authenticate actual original graph, roots, worker generation, code and recording streams. */
    virtual void requireCurrent()const=0;
    /** @brief Authenticate the actual source storage and explicit source stream before native close/reset. */
    virtual void requireSource(const PooledImmutableSourcePreparedStorage&,void* stream)const=0;
    /** @brief Reject a guessed, stale or unrelated source submission/recording stream. */
    virtual void requireSourceStream(void* stream)const=0;
    /** @brief Retain the exact existing parent submission stream owner during cold materialization. */
    virtual void bindSourceSubmissionStream(const RetainedGraphParticipantParentStream& owner)=0;
    /** @brief Observe all original native participants only in the existing whole-request terminal join. */
    virtual void awaitNativeTerminal(std::chrono::steady_clock::time_point deadline)const=0;
    PooledImmutableRetirementAuthority(const PooledImmutableRetirementAuthority&)=delete;
    PooledImmutableRetirementAuthority& operator=(const PooledImmutableRetirementAuthority&)=delete;
private:
    friend class PooledImmutableCompleteCohort;
    friend class PooledImmutableRetainedParticipants;
    PooledImmutableRetirementAuthority()=default;
};
}
