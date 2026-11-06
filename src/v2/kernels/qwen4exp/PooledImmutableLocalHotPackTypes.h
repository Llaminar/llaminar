/**
 * @file PooledImmutableLocalHotPackTypes.h
 * @brief Capture-only native row delivery descriptor over the original pool inverse map.
 */
#pragma once
#include "PooledImmutableLocalHotUnionOps.h"
namespace llaminar2::qwen4exp::local_hot_transport {
/** @brief Addresses issued only by the checked original transport/native-source bridge. */
struct NativePackDescriptor {
    const UnionControl* union_control=nullptr;
    const RequestHeader* request=nullptr;
    ResultHeader* result=nullptr;
    std::byte* rows=nullptr;
    std::uint64_t* payload_bytes=nullptr;
    std::uint64_t* local_header_bytes=nullptr; ///< Source-self original producer publication; remote receivers use native receipts.
    std::uint32_t consumer=0,consumer_count=0,maximum_rows=0,native_format=0;
    std::uint64_t consumer_storage_identity=0;
    std::uint64_t capacity_bytes=0;
};
/** @brief Copy only needed native rows on the original source CUDA stream, then seal exact metadata. */
void enqueueCUDAPooledLocalHotNativePack(pooled_source::DeviceArgs,NativePackDescriptor,void* exact_stream);
/** @brief Identical exact native pack and terminal publication on the original HIP stream. */
void enqueueROCmPooledLocalHotNativePack(pooled_source::DeviceArgs,NativePackDescriptor,void* exact_stream);
} // namespace llaminar2::qwen4exp::local_hot_transport
