/**
 * @file PooledImmutableLocalHotOffsets.h
 * @brief Immutable native local-cache subranges shared without host source/tensor headers.
 */
#pragma once
#include <cstddef>
#include <cstdint>
namespace llaminar2::qwen4exp {
/** @brief Named immutable subranges in the optional local physical cache root. */
struct PooledImmutableLocalHotCacheOffsets {
    std::size_t control=0,slots=0,nodes=0,epochs=0,counts=0,payload=0,bytes=0;
    bool operator==(const PooledImmutableLocalHotCacheOffsets&)const=default;
};
/** @brief Mandatory frame/output reconstruction, retained independently of hot slots. */
struct PooledImmutableLocalHotFrameOffsets {
    std::size_t control=0,identity=0,counters=0,unique=0,original_to_unique=0,miss_keys=0;
    std::size_t response=0,result_header=0,received_header_bytes=0,received_payload_bytes=0,bytes=0;
    bool operator==(const PooledImmutableLocalHotFrameOffsets&)const=default;
};
/** @brief Fixed cooperative sorting/scan scratch, never a communication extent. */
struct PooledImmutableLocalHotScratchOffsets {
    std::size_t keys_a=0,keys_b=0,radix_counts=0,radix_offsets=0,scan=0,bytes=0;
    std::uint32_t radix_blocks=0;
    bool operator==(const PooledImmutableLocalHotScratchOffsets&)const=default;
};
} // namespace llaminar2::qwen4exp
