/**
 * @file CacheAuxiliaryState.h
 * @brief Typed cache-owned lifecycle for admitted model auxiliary state.
 *
 * Implementations borrow already-admitted persistent arena storage. Endpoint
 * checkpoints contain only state needed to resume arithmetic; append-only
 * logical-block payload is archived separately, so rollback never copies a
 * full compressed-key bank. The canonical cache remains the frontier owner.
 */
#pragma once
#include "../../backends/DeviceId.h"
#include <cstddef>
#include <cstdint>

namespace llaminar2
{
class IKVCache;
/** @brief Versioned model state identity; wrong-state attachment is a setup error. */
enum class CacheAuxiliaryStateKind : std::uint32_t { QwenFlashNext = 1 };
/** @brief Exact source/destination authority for an explicit cache-state I/O boundary. */
enum class CacheAuxiliaryMemoryDomain : std::uint8_t { Host, Device };
/** @brief Caller-retained admitted native state payload, independent of live contents. */
struct CacheAuxiliaryMemory
{
    void *data;
    std::size_t capacity;
    CacheAuxiliaryMemoryDomain domain;
};
/** @brief Immutable logical block range; storage capacity is never its transfer extent. */
struct CacheAuxiliaryBlockRange
{
    int layer, sequence, firstToken, tokenCount;
};
/** @brief Explicit reset scope, retaining immutable state bindings and graph identity. */
struct CacheAuxiliaryResetRange
{
    int firstLayer, layerCount, firstSequence, sequenceCount;
};
/** @brief Model extension owned by one canonical cache and sealed before capture. */
class ICacheAuxiliaryState
{
public:
    /** @brief Retire metadata after its captured users and borrowed arena have retired. */
    virtual ~ICacheAuxiliaryState() = default;
    /** @brief Exact canonical cache that owns this extension's frontiers. */
    virtual const IKVCache &cache() const noexcept = 0;
    /** @brief Initialize admitted state once and publish exact producer ordering before capture. */
    virtual void initialize(void *exactStream) = 0;
    /** @brief Join setup/reset/import publications on an exact future execution stream. */
    virtual void requireReady(void *exactStream) const = 0;
    /** @brief Exact versioned model payload kind. */
    virtual CacheAuxiliaryStateKind kind() const noexcept = 0;
    /** @brief Exact CPU/GPU authority of every live value. */
    virtual DeviceId device() const noexcept = 0;
    /** @brief Admitted checkpoint capacity for one sequence; excludes append-only pools. */
    virtual std::size_t checkpointCapacity(int sequence) const = 0;
    /** @brief Actual logical-block archive extent from immutable token-range metadata. */
    virtual std::size_t blockArchiveBytes(CacheAuxiliaryBlockRange range) const = 0;
    /** @brief Reset live endpoint state through an ordered canonical cache boundary. */
    virtual void reset(CacheAuxiliaryResetRange range,void *exactStream) = 0;
    /** @brief Capture endpoint state without reading device counts on the host. */
    virtual void captureCheckpoint(int sequence,CacheAuxiliaryMemory destination,void *exactStream) const = 0;
    /** @brief Restore endpoint state; restored cache frontiers hide speculative pooled writes. */
    virtual void restoreCheckpoint(int sequence,CacheAuxiliaryMemory source,void *exactStream) = 0;
    /** @brief Compact endpoint archive extent for an immutable logical prefix length. */
    virtual std::size_t endpointArchiveBytes(int sequence,int logicalTokens) const = 0;
    /** @brief Export compact raw tails and accepted PLE histories at an explicit archive boundary. */
    virtual void exportEndpoint(int sequence,int logicalTokens,CacheAuxiliaryMemory destination,void *exactStream) const = 0;
    /** @brief Import compact raw tails and accepted PLE histories before frontier publication. */
    virtual void importEndpoint(int sequence,int logicalTokens,CacheAuxiliaryMemory source,void *exactStream) = 0;
    /** @brief Export only compressed pool rows completed in the requested logical block. */
    virtual void exportBlock(CacheAuxiliaryBlockRange range,CacheAuxiliaryMemory destination,void *exactStream) const = 0;
    /** @brief Import only the requested immutable archive block before publishing its frontier. */
    virtual void importBlock(CacheAuxiliaryBlockRange range,CacheAuxiliaryMemory source,void *exactStream) = 0;
};
} // namespace llaminar2
