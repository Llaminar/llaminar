/**
 * @file PooledImmutableLocalHotPoolQuery.h
 * @brief Closed compact-query producer for the unchanged original pool Begin.
 *
 * Only the original DirectedTransport can bind its union count and merged keys
 * (or source-self borrowed miss keys). This is not a second query/count owner:
 * the metadata kernels remain their sole writers, and the original coordinator
 * still opens, coalesces and retires the actual L2 interval.
 */
#pragma once
#include "PooledImmutableNativeTransfer.h"
namespace llaminar2::qwen4exp {
class PooledImmutableDirectedTransport;
class PooledImmutableLocalHotPreparedStorage;
/** @brief Retain exact original compact producers without inventing ordinary tensor roles. */
class PooledImmutableLocalHotPoolQuery final {
public:
    /** @brief Seal existing prepared original producer addresses before native capture. */
    void seal();
    /** @brief Validate cold source/coordinator geometry before physical producer addresses are sealed.
     * @note Graph declaration must not materialize addresses; capture preparation seals them later. */
    void requireDeclaration(const PooledImmutableCoordinatorNativePlan&)const;
    /** @brief Reject unsealed producers or source/coordinator/root/runtime drift before native use. */
    void require(const PooledImmutableCoordinatorNativePlan&)const;
    /** @return Immutable cold declaration identity, excluding all live values. */
    std::uint64_t declarationIdentity()const;
    /** @return Exact native producer identity after its original storage seal. */
    std::uint64_t captureIdentity()const;
private:
    friend class PooledImmutableDirectedTransport;
    friend class IPooledImmutableNativeKernels;
    /** @brief Only the actual original transport may bind its own private producer subranges. */
    PooledImmutableLocalHotPoolQuery(std::shared_ptr<const PooledImmutableDirectedTransport>,
        std::shared_ptr<PooledImmutableLocalHotPreparedStorage>,PooledNativeTransferSpan,PooledNativeTransferSpan,PooledNativeTransferSpan,
        std::uint32_t maximumOriginal);
    /** @return Same original counted descriptor consumed by the unchanged coalescer. */
    pooled_native::CoordinatorQuery native()const;
    std::shared_ptr<const PooledImmutableDirectedTransport> transport_;
    std::shared_ptr<PooledImmutableLocalHotPreparedStorage> source_local_;
    PooledNativeTransferSpan ids_,count_,admission_;
    std::uint32_t capacity_;
    const void* sealed_ids_=nullptr;
    const std::int32_t* sealed_count_=nullptr;
    const std::uint64_t* sealed_admission_=nullptr;
};
} // namespace llaminar2::qwen4exp
