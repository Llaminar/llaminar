/**
 * @file PooledImmutableLocalHotNativePack.h
 * @brief Closed original source-native delivery spans, without an expanded model output.
 *
 * The actual DirectedTransport alone chooses source-self L1 response or an
 * original remote pack window. Exact source pins and readable rows remain with
 * the original source program until every native delivery/ACK has retired.
 */
#pragma once
#include "PooledImmutableNativeTransfer.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotPackTypes.h"
namespace llaminar2::qwen4exp {
class PooledImmutableDirectedTransport;
class PooledImmutableSourcePreparedStorage;
class PooledImmutableLocalHotPreparedStorage;
/** @brief Immutable admitted source/consumer/native span binding; never a new read or source authority. */
class PooledImmutableLocalHotNativePack final {
public:
    /** @return Exact original bounded request/result/row/count/union subranges for graph transport binding. */
    const auto& spans()const noexcept{return spans_;}
    /** @brief Seal actual prepared original source/result/extent addresses before capture. */
    void seal();
    /** @brief Reject changed original source, transport, consumer or physical producer identities. */
    void require(const PooledImmutableSourcePreparedStorage&)const;
    /** @return Complete immutable declaration identity before native capture. */
    std::uint64_t declarationIdentity()const;
    /** @return Exact already-sealed original pointer identity. */
    std::uint64_t captureIdentity()const;
private:
    friend class PooledImmutableDirectedTransport;
    friend class IPooledImmutableSourceKernels;
    /** @brief Construct only from the actual original transport's bounded source spans. */
    PooledImmutableLocalHotNativePack(std::shared_ptr<const PooledImmutableDirectedTransport>,
        std::shared_ptr<PooledImmutableLocalHotPreparedStorage>,std::uint32_t,
        std::array<PooledNativeTransferSpan,5>,std::optional<PooledNativeTransferSpan>);
    /** @return Actual pack descriptors after all original subranges are physically sealed. */
    local_hot_transport::NativePackDescriptor native()const;
    std::shared_ptr<const PooledImmutableDirectedTransport> transport_;
    std::shared_ptr<PooledImmutableLocalHotPreparedStorage> consumer_;
    std::uint32_t coordinate_;
    std::array<PooledNativeTransferSpan,5> spans_; ///< Request/result/payload/byte-count/union, all source-local.
    std::optional<PooledNativeTransferSpan> local_header_bytes_;
    std::array<void*,5> sealed_{};
    void* sealed_header_bytes_=nullptr;
};
} // namespace llaminar2::qwen4exp
