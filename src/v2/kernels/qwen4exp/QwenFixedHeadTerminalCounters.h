/**
 * @file QwenFixedHeadTerminalCounters.h
 * @brief Request-owned fixed-head lookup work and actual native contribution observations.
 *
 * Calls/consensus failures count once per participant and lookup. Live IDs and
 * native bytes describe owned source rows actually packed: L*(16/P), multiplied
 * by original native rowBytes. They exclude per-consumer decode/receive expansion.
 * Header/payload native-send pairs are actual outgoing ring primitive bytes,
 * accumulated by NCCL/RCCL themselves. Only the complete terminal projection
 * authenticates P-1 send multiplicity before exposing logical source-contribution
 * counters. Empty/error participation still sends the required8B header on each
 * ring step and contributes zero payload. No receive/decode expansion is counted.
 */
#pragma once
#include "kernels/common/DeviceGenerationControlLayout.h"
#include "kernels/common/NativeCounter64.h"
#include <climits>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_FIXED_HEAD_COUNTER_HD __host__ __device__ inline
#else
#define LLAMINAR_FIXED_HEAD_COUNTER_HD inline
#endif
namespace llaminar2::qwen4exp {
/** @brief Closed request-observation pairs, never generation decision fields. */
enum class QwenFixedHeadCounter : int {
    Calls=sampling_math::kDeviceGenerationControlFixedHeadCallsLow,
    LiveIds=sampling_math::kDeviceGenerationControlFixedHeadLiveIdsLow,
    NativeBytes=sampling_math::kDeviceGenerationControlFixedHeadNativeBytesLow,
    Failures=sampling_math::kDeviceGenerationControlFixedHeadFailuresLow,
    HeaderNativeSendBytes=sampling_math::kDeviceGenerationControlFixedHeadHeaderNativeSendBytesLow,
    PayloadNativeSendBytes=sampling_math::kDeviceGenerationControlFixedHeadPayloadNativeSendBytesLow
};
/** @brief Borrow only the original canonical controller, never own storage or counter values. */
struct QwenFixedHeadCounterRow {
    std::int32_t *control=nullptr;int words=0;
    /** @return Whether a caller requested observation. */
    LLAMINAR_FIXED_HEAD_COUNTER_HD bool bound()const{return control || words!=0;}
    /** @return Whether the exact row and every UINT64 pair are native aligned. */
    LLAMINAR_FIXED_HEAD_COUNTER_HD bool valid()const {
        return control && words==sampling_math::kDeviceGenerationControlCount &&
            reinterpret_cast<std::uintptr_t>(control)%alignof(std::uint64_t)==0;
    }
    /** @return Original closed metric pair without reading mutable words. */
    LLAMINAR_FIXED_HEAD_COUNTER_HD NativeCounter64 metric(QwenFixedHeadCounter kind)const {
        if(!valid())return {};
        switch(kind) {
        case QwenFixedHeadCounter::Calls:case QwenFixedHeadCounter::LiveIds:case QwenFixedHeadCounter::NativeBytes:
        case QwenFixedHeadCounter::Failures:case QwenFixedHeadCounter::HeaderNativeSendBytes:case QwenFixedHeadCounter::PayloadNativeSendBytes:
            return {control+static_cast<int>(kind)};
        }
        return {};
    }
    /** @return Exact immutable row metadata, excluding every live value. */
    bool operator==(const QwenFixedHeadCounterRow &)const=default;
};
/** @brief Bound every legal request/window native contribution before capture.
 * @details INT_MAX response transactions plus one context's prefill are bounded
 * conservatively by3*(INT_MAX+1) lookup calls. Payload byte overflow is rejected
 * before native receipts, whose storage is the same original request row.
 */
LLAMINAR_FIXED_HEAD_COUNTER_HD bool validQwenFixedHeadCounterLifetime(int rows,int heads,std::size_t rowBytes) {
    if(rows<=0 || (heads!=2 && heads!=4 && heads!=8) || !rowBytes)return false;
    const std::uint64_t calls=3*(std::uint64_t(INT_MAX)+1);
    const std::uint64_t sends=16/heads-1;
    return calls<=UINT64_MAX/(8*sends) && rowBytes<=UINT64_MAX/static_cast<std::uint64_t>(heads)/sends &&
        static_cast<std::uint64_t>(rows)<=UINT64_MAX/calls/(static_cast<std::uint64_t>(heads)*rowBytes*sends);
}
/** @brief Raw actual native ring send observations, never logical contribution bytes.
 * @details Both dependency patches restrict device-counted AllGather to native
 * ring. Their ncclLlaminarPayload records the exact shortened primitive count
 * multiplied by outgoing sends=P-1 using atomicAdd. This source-owned contract
 * is required by the projection below; changing it requires coherent updates
 * and native byte qualification, not guessing a different transport multiplier.
 */
struct QwenFixedHeadNativeSendTotals {
    std::uint64_t calls=0,live_ids=0,native_bytes=0,failures=0,header_send_bytes=0,payload_send_bytes=0;
    /** @return Complete immutable raw native observation. */
    bool operator==(const QwenFixedHeadNativeSendTotals &)const=default;
};
/** @brief Immutable terminal totals; never a host execution-state mirror. */
struct QwenFixedHeadCounterTotals {
    std::uint64_t calls=0,live_ids=0,native_bytes=0,failures=0,header_bytes=0,payload_bytes=0;
    /** @return Exact observed terminal value identity. */
    bool operator==(const QwenFixedHeadCounterTotals &)const=default;
};
/** @brief Check exact contribution relations without inferring link or receive traffic.
 * @param value Observed complete terminal pairs.
 * @param rowBytes Original source native row stride, never decoded width or capacity.
 * @param ownedHeads Exact source heads per participant (16 divided by actual cohort degree).
 * @return Whether every observed identity relation is exact and overflow-free.
 */
inline bool validQwenFixedHeadContributions(const QwenFixedHeadCounterTotals &value,std::size_t rowBytes,int ownedHeads) {
    return (ownedHeads==2 || ownedHeads==4 || ownedHeads==8) && value.live_ids%ownedHeads==0 && rowBytes && value.calls<=UINT64_MAX/8 && value.header_bytes==value.calls*8 &&
        value.failures<=value.calls && value.live_ids<=UINT64_MAX/rowBytes &&
        value.native_bytes==value.live_ids*rowBytes && value.payload_bytes==value.native_bytes;
}
static_assert(sampling_math::kDeviceGenerationControlFixedHeadCallsLow%2==0);
static_assert(sampling_math::kDeviceGenerationControlFixedHeadPayloadNativeSendBytesHigh + 1 ==
              sampling_math::kDeviceGenerationControlAdaptiveCallsLow);
static_assert(sampling_math::kDeviceGenerationControlCount%2==0);

/** @brief Authenticate raw sends against exact local packing before exposing logical contributions.
 * @param raw Actual complete terminal pairs, with unmodified outgoing ring totals.
 * @param participants Actual authenticated homogeneous cohort degree2/4/8.
 * @param rowBytes Original source stride, never decoded or capacity bytes.
 * @param out Logical source contributions, changed only after every exact relation passes.
 * @return False for missing/excess/nondivisible sends, malformed geometry or overflow.
 * @details This is not acceptance of arbitrary divisible traffic: raw header must
 * equal8*calls*(P-1), and raw payload must equal packedNativeBytes*(P-1).
 * Equal live counts/head counts/rowBytes were already authenticated by the real
 * cohort consensus. The native primitive counter proves that the actual ring
 * path submitted that work; there is no host live-count probe or extra copy.
 */
inline bool projectQwenFixedHeadTerminalContributions(const QwenFixedHeadNativeSendTotals &raw,
    int participants,std::size_t rowBytes,QwenFixedHeadCounterTotals &out) {
    if(participants!=2 && participants!=4 && participants!=8)return false;
    const std::uint64_t sends=participants-1;
    const QwenFixedHeadCounterTotals logical{raw.calls,raw.live_ids,raw.native_bytes,raw.failures,
        raw.calls<=UINT64_MAX/8?raw.calls*8:0,raw.native_bytes};
    if(!validQwenFixedHeadContributions(logical,rowBytes,16/participants) ||
        logical.header_bytes>UINT64_MAX/sends || logical.native_bytes>UINT64_MAX/sends ||
        raw.header_send_bytes!=logical.header_bytes*sends ||
        raw.payload_send_bytes!=logical.native_bytes*sends)return false;
    out=logical;return true;
}
/** @brief Decode RAW ring-send pairs only after the existing complete terminal frontier/copy.
 * @return False for an invalid row, leaving out untouched. Logical metric publication
 * must additionally use projectQwenFixedHeadTerminalContributions with the actual cohort.
 */
inline bool readQwenFixedHeadNativeSendCounters(const std::int32_t *row,int words,QwenFixedHeadNativeSendTotals &out) {
    if(!row || words!=sampling_math::kDeviceGenerationControlCount ||
        reinterpret_cast<std::uintptr_t>(row)%alignof(std::int32_t))return false;
    const auto value=[&](QwenFixedHeadCounter kind){return readNativeCounter64(row+static_cast<int>(kind));};
    out={value(QwenFixedHeadCounter::Calls),value(QwenFixedHeadCounter::LiveIds),value(QwenFixedHeadCounter::NativeBytes),
        value(QwenFixedHeadCounter::Failures),value(QwenFixedHeadCounter::HeaderNativeSendBytes),value(QwenFixedHeadCounter::PayloadNativeSendBytes)};
    return true;
}
} // namespace llaminar2::qwen4exp

#undef LLAMINAR_FIXED_HEAD_COUNTER_HD
