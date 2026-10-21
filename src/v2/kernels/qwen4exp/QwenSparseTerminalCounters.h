/**
 * @file QwenSparseTerminalCounters.h
 * @brief Exact request-owned QSA work observations in the existing terminal control row.
 *
 * Native request-owned writers use checked aligned UINT64 atomics; host
 * terminal projection reads low/high words without aliasing. No host mirror or
 * additional readback is created.
 * Counter values are observational and cannot affect KV/frontier acceptance.
 */
#pragma once
#include "kernels/common/DeviceGenerationControlLayout.h"
#include "kernels/common/NativeCounter64.h"
#include <cstdint>
#include <cstddef>
#include <climits>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_QSA_COUNTER_HD __host__ __device__ inline
#else
#define LLAMINAR_QSA_COUNTER_HD inline
#endif
namespace llaminar2::qwen4exp {
/** @brief One operation's actual metadata and live payload bytes. */
struct QwenSparseCounterDelta {
    std::uint64_t prior_header=0,prior_payload=0,candidate_payload=0,accepted_payload=0;
};
/** @brief Distinct native observations; these indices never select generation behavior. */
enum class QwenSparseWorkCounter : int {
    PriorHeader=sampling_math::kDeviceGenerationControlQsaPriorHeaderBytesLow,
    PriorPayload=sampling_math::kDeviceGenerationControlQsaPriorPayloadBytesLow,
    CandidatePayload=sampling_math::kDeviceGenerationControlQsaCandidatePayloadBytesLow,
    AcceptedPayload=sampling_math::kDeviceGenerationControlQsaAcceptedPayloadBytesLow
};
/** @brief Borrow one original canonical request row; disabled observations are explicit. */
struct QwenSparseWorkCounterRow {
    std::int32_t*control=nullptr;
    int words=0;
    /** @return Whether a caller supplied any row-binding metadata. */
    LLAMINAR_QSA_COUNTER_HD bool bound()const{return control || words!=0;}
    /** @return Whether every even word-pair can use native aligned UINT64 atomics. */
    LLAMINAR_QSA_COUNTER_HD bool valid()const {
        return control && words==sampling_math::kDeviceGenerationControlCount &&
            reinterpret_cast<std::uintptr_t>(control)%alignof(std::uint64_t)==0;
    }
    /** @brief Name a metric pair without reading mutable words.
     * @param kind Closed metric identity, never a raw caller-provided offset.
     * @return Original aligned pair, or an invalid disabled binding on malformed input.
     */
    LLAMINAR_QSA_COUNTER_HD NativeCounter64 metric(QwenSparseWorkCounter kind)const {
        if(!valid())return {};
        switch(kind){
        case QwenSparseWorkCounter::PriorHeader:case QwenSparseWorkCounter::PriorPayload:
        case QwenSparseWorkCounter::CandidatePayload:case QwenSparseWorkCounter::AcceptedPayload:
            return {control+static_cast<int>(kind)};
        }
        return {};
    }
};
static_assert(sampling_math::kDeviceGenerationControlCount%2==0);
/** @brief Prove totals fit for every canonical controller transaction before capture.
 * @param layers Admitted local layer bound, conservatively including non-sparse layers.
 * @param rows Admitted verifier physical row bound, independent of live row words.
 * @return False for malformed geometry or an overflowing lifetime upper bound.
 * @details Controller response/transaction counters are signed INT32 and each
 * committed transaction emits at least one response. INT_MAX+1 includes the
 * initial transaction conservatively. One snapshot, retain and acceptance per
 * layer/transaction are canonical; diagnostic revised publications still use
 * checked atomic addition and cannot wrap.
 */
LLAMINAR_QSA_COUNTER_HD bool validQwenSparseCounterLifetimeBound(int layers,int rows) {
    if(layers<=0 || rows<=0)return false;
    const std::uint64_t transactions=std::uint64_t(INT_MAX)+1;
    const std::uint64_t copies=transactions*static_cast<std::uint64_t>(layers);
    const std::uint64_t maximumPayload=static_cast<std::uint64_t>(rows)*128*4;
    return copies<=UINT64_MAX/16 && copies<=UINT64_MAX/(3*128*4) &&
        copies<=UINT64_MAX/maximumPayload;
}
/** @brief Explicit terminal projection; never a live host execution-state mirror. */
using QwenSparseCounterTotals=QwenSparseCounterDelta;
/** @brief Reject incomplete or misaligned native control-row spans without observing state.
 * @param row Original request-owned controller row.
 * @param words Exact canonical row width, independent of request stride.
 * @return Whether individual INT32 word accesses are valid.
 */
LLAMINAR_QSA_COUNTER_HD bool validQwenSparseCounterRow(const std::int32_t*row,int words) {
    return row && words==sampling_math::kDeviceGenerationControlCount &&
        (reinterpret_cast<std::uintptr_t>(row)%alignof(std::int32_t))==0;
}
/** @brief Decode two fixed-width words without requiring UINT64 alignment.
 * @param row Valid canonical control row.
 * @param low Low-word index of one checked metric pair.
 * @return Exact unsigned byte total, including low-word carry bits.
 */
LLAMINAR_QSA_COUNTER_HD std::uint64_t qwenSparseCounterPair(const std::int32_t*row,int low) {
    return std::uint64_t(static_cast<std::uint32_t>(row[low])) |
        (std::uint64_t(static_cast<std::uint32_t>(row[low+1]))<<32);
}
/** @brief Project immutable terminal words into exact native UINT64 totals.
 * @param row Original controller row after its terminal ordering boundary.
 * @param words Canonical control row width.
 * @param out Result written only after complete binding validation.
 * @return False for malformed binding, leaving out unchanged.
 */
LLAMINAR_QSA_COUNTER_HD bool readQwenSparseTerminalCounters(const std::int32_t*row,int words,QwenSparseCounterTotals&out) {
    if(!validQwenSparseCounterRow(row,words))return false;
    out={qwenSparseCounterPair(row,sampling_math::kDeviceGenerationControlQsaPriorHeaderBytesLow),
        qwenSparseCounterPair(row,sampling_math::kDeviceGenerationControlQsaPriorPayloadBytesLow),
        qwenSparseCounterPair(row,sampling_math::kDeviceGenerationControlQsaCandidatePayloadBytesLow),
        qwenSparseCounterPair(row,sampling_math::kDeviceGenerationControlQsaAcceptedPayloadBytesLow)};
    return true;
}
/** @brief Add one host-owned operation, validating every sum before any write.
 * @param row Original request controller row owned by the enclosing transaction.
 * @param words Exact row width; request stride remains a separate address calculation.
 * @param delta Actual operation bytes, never allocation capacity.
 * @return False on malformed binding or UINT64 overflow, preserving every row word.
 * @details The CPU execution owner serializes host updates; GPU callers use native
 * atomic additions through NativeCounter64 instead. Empty deltas write no words.
 */
inline bool addQwenSparseHostCounters(std::int32_t*row,int words,QwenSparseCounterDelta delta) {
    QwenSparseCounterTotals old;
    if(!readQwenSparseTerminalCounters(row,words,old))return false;
    const std::uint64_t before[]{old.prior_header,old.prior_payload,old.candidate_payload,old.accepted_payload};
    const std::uint64_t added[]{delta.prior_header,delta.prior_payload,delta.candidate_payload,delta.accepted_payload};
    const int low[]{sampling_math::kDeviceGenerationControlQsaPriorHeaderBytesLow,
        sampling_math::kDeviceGenerationControlQsaPriorPayloadBytesLow,
        sampling_math::kDeviceGenerationControlQsaCandidatePayloadBytesLow,
        sampling_math::kDeviceGenerationControlQsaAcceptedPayloadBytesLow};
    for(int i=0;i<4;++i)if(added[i]>UINT64_MAX-before[i])return false;
    for(int i=0;i<4;++i)if(added[i]) {
        const auto value=before[i]+added[i];
        row[low[i]]=static_cast<std::int32_t>(static_cast<std::uint32_t>(value));
        row[low[i]+1]=static_cast<std::int32_t>(static_cast<std::uint32_t>(value>>32));
    }
    return true;
}
} // namespace llaminar2::qwen4exp
#undef LLAMINAR_QSA_COUNTER_HD
