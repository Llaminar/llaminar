/**
 * @file PooledImmutableLocalHotNativeTypes.h
 * @brief Decoded native local-cache launch arguments, never serving/materialization permits.
 */
#pragma once
#include "PooledImmutableLocalHotOffsets.h"
#include "PooledImmutableLocalHotOps.h"
#include "PooledImmutableLocalHotTerminal.h"
#include "PooledImmutableNativeTypes.h"
#include "kernels/common/NativeEmbeddingIQTables.h"
namespace llaminar2::qwen4exp::local_hot {
/** @brief Closed launch/schema version captured with every actual native input. */
inline constexpr std::uint64_t NativeVersion=3;
/** @brief Static projection of the original prepared geometry and physical lifetime. */
struct Parameters {
    PooledImmutableLocalHotCacheOffsets cache;
    PooledImmutableLocalHotFrameOffsets frame;
    PooledImmutableLocalHotScratchOffsets scratch;
    std::uint64_t source_rows=0,table=0,source_generation=0,consumer_storage_identity=0;
    std::size_t row_bytes=0,slot_stride=0;
    std::uint32_t maximum_original=0,slots=0,lane_width=0,protected_limit=0,row_elements=0;
    adaptive_lookup::Policy policy{};
    adaptive_lookup::RowKeyWidth key_width=adaptive_lookup::RowKeyWidth::UInt32;
    TensorType format=TensorType::FP32;
};
/** @brief Original root aliases, used alone only by cold metadata initialization. */
struct NativeRoots {
    Parameters plan;
    std::byte* cache=nullptr;
    std::byte* frame=nullptr;
    std::byte* scratch=nullptr;
    std::uint64_t* ready=nullptr;
    NativeEmbeddingIQTables tables{};
};
/** @brief Whole-request terminal-only projection into separately admitted output storage. */
struct NativeTerminal : NativeRoots {
    const int* control=nullptr;
    std::byte* output=nullptr;
    int requests=0,control_words=0;
    std::size_t output_bytes=0;
};
/** @brief Same original active controller and actual reader coordinate, without dummy query/output. */
struct NativeFrame : NativeRoots {
    int* healthy=nullptr;
    int* error=nullptr;
    local_hot_transport::ConsumerGeometry consumer{};
};
/** @brief Complete actual original hash/count producer, present only for Probe. */
struct NativeQuery : NativeFrame {
    DeviceRowRange original_ids;
    const void* ids;
    pooled_native::OriginalIdWidth id_width;
    PooledImmutableLocalHotAccessIntent intent;
};
/** @brief Sole model output writer, independent of optional cache fill. */
struct NativeRead : NativeFrame {float* output=nullptr;std::uint32_t physical_original=0;};
/** @brief Original once-per-request local admission producer, never a per-query grant. */
struct NativeRequestReset : NativeFrame {const local_hot_transport::SourceAdmissionHeader* admission=nullptr;};
/** @brief Actual own immutable request producer, present only in response acquisition. */
struct NativeResult : NativeFrame {const local_hot_transport::RequestHeader* own_request=nullptr;};
/** @return One exact named root subrange; optional C0 root alone may be null. */
template<class T> QFN_ADAPTIVE_HD inline T* at(std::byte* root,std::size_t offset)
{return root?reinterpret_cast<T*>(root+offset):nullptr;}
/** @return The unchanged canonical metadata/frames with exact original source/root identity. */
template<class Slot> QFN_ADAPTIVE_HD inline View<Slot> view(const NativeRoots& a)
{
    const auto& p=a.plan;
    return {{at<Slot>(a.cache,p.cache.slots),at<adaptive_lookup::RadixNode>(a.cache,p.cache.nodes),
        at<adaptive_lookup::OwnerControl>(a.cache,p.cache.control),at<adaptive_lookup::FrameControl>(a.frame,p.frame.control),
        at<std::uint32_t>(a.cache,p.cache.epochs),at<std::uint16_t>(a.cache,p.cache.counts),
        p.slots,p.protected_limit,p.lane_width,p.source_rows,p.policy.aging_accesses},
        at<FrameIdentity>(a.frame,p.frame.identity),at<Counters>(a.frame,p.frame.counters),
        at<adaptive_lookup::UniqueRecord>(a.frame,p.frame.unique),at<std::uint32_t>(a.frame,p.frame.original_to_unique),
        at<std::uint64_t>(a.frame,p.frame.miss_keys),at<adaptive_lookup::KeyIndex>(a.scratch,p.scratch.keys_a),
        at<std::byte>(a.cache,p.cache.payload),at<std::byte>(a.frame,p.frame.response),
        p.maximum_original,p.row_bytes,p.slot_stride,p.table,p.source_generation,p.consumer_storage_identity};
}
/** @return Original actual completed-response metadata in the same local private native root. */
QFN_ADAPTIVE_HD inline local_hot_transport::DeviceResultView result(const NativeRoots& a)
{
    const auto& p=a.plan;
    return {at<local_hot_transport::ResultHeader>(a.frame,p.frame.result_header),
        at<std::byte>(a.frame,p.frame.response),at<std::uint64_t>(a.frame,p.frame.received_header_bytes),
        at<std::uint64_t>(a.frame,p.frame.received_payload_bytes),std::uint64_t(p.maximum_original)*p.row_bytes};
}
} // namespace llaminar2::qwen4exp::local_hot
