/**
 * @file PooledImmutableNativeOperands.h
 * @brief Strong original producer spans and closed per-phase input schemas.
 *
 * Static capacities describe retained storage only. Native byte/count producers
 * determine each live extent. No constructor reads a GPU value or turns equal
 * integers into source/transport permission; graph bindings must retain and
 * acquire the actual producers under their original canonical arena roles.
 */
#pragma once
#include "PooledImmutableNativeStorage.h"
#include "execution/qwen4exp/Qwen4ExpPreparedStorage.h"
#include "memory/BufferId.h"
#include <array>
#include <optional>
namespace llaminar2 {class CapturedDeviceStorageLease;}
namespace llaminar2::qwen4exp
{
/** @brief Closed role set; owner and coordinator schemas reject unused, missing or substituted operands. */
enum class PooledNativeOperand : std::uint32_t
{
    Invocation,Records,RecordBytes,Frontier,Payload,PayloadBytes,Ids,Admission,OriginalCount,Source,Outcome,AllocationSlice,Count
};
inline constexpr auto PooledNativeOperandCount=static_cast<std::size_t>(PooledNativeOperand::Count);
/** @brief Original canonical root plus immutable bounded subrange and exact graph role. */
struct PooledNativeTensorSpan
{
    std::shared_ptr<TensorBase> root;
    BufferId buffer;
    std::size_t offset,bytes;
};
/** @brief One phase's exact required storage extent, never its communication count. */
struct PooledNativeOperandRequirement
{
    std::size_t bytes=0,alignment=1;
    bool operator==(const PooledNativeOperandRequirement&)const=default;
};
using PooledNativeOperandSchema=std::array<PooledNativeOperandRequirement,PooledNativeOperandCount>;
using PooledNativeOperandSpans=std::array<std::optional<PooledNativeTensorSpan>,PooledNativeOperandCount>;
/** @return Exact owner input declaration; static C0 requires no dummy record/count/payload roots. */
PooledNativeOperandSchema pooledNativeSchema(pooled_native::OwnerAction action,const pooled_native::OwnerParameters& plan,
    pooled_native::SourceCondition condition=pooled_native::SourceCondition::Always,
    pooled_native::RecordLayout records=pooled_native::RecordLayout::Compact);
/** @return Exact coordinator inputs; all fixed frontiers use the actual original P, not consumer count. */
PooledNativeOperandSchema pooledNativeSchema(pooled_native::CoordinatorAction action,
    const pooled_native::CoordinatorParameters& plan,std::optional<pooled_native::OriginalRows> original_rows,
    pooled_native::SourceCondition condition=pooled_native::SourceCondition::Always);
/** @brief Checked strong owner inputs, sealed once before capture and revalidated on every enqueue. */
class PooledImmutableOwnerOperands final
{
public:
    /**
     * @brief Retain exact original producer spans before native materialization.
     * @param action Closed owner operation.
     * @param plan Original checked native geometry.
     * @param spans Exact schema-owned canonical roots and subranges.
     * @param condition Fixed captured source branch; its actual outcome is a retained producer operand.
     * @param records Compact acquired records, or an exact same-device directed allocation-bank view with its actual slice producer.
     */
    PooledImmutableOwnerOperands(pooled_native::OwnerAction action,const PooledImmutableOwnerNativePlan& plan,
        PooledNativeOperandSpans spans,pooled_native::SourceCondition condition=pooled_native::SourceCondition::Always,
        pooled_native::RecordLayout records=pooled_native::RecordLayout::Compact);
    /** @brief Seal existing prepared addresses outside capture; allocate and publish no bytes. */
    void seal();
    /** @brief Reject changed role/type/extent/device/backend/runtime or operation before native submission. */
    void require(pooled_native::OwnerAction action,const PooledImmutableOwnerNativePlan& plan)const;
    /** @return Stable semantic declarations; a later prepared seal rejects every physical drift. */
    std::uint64_t declarationIdentity()const;
    /** @brief Validate declarations and, when sealed, original physical addresses/runtime. */
    void validateIfSealed()const;
    /** @return Every immutable input/root/offset/geometry field embedded in capture. */
    std::uint64_t captureIdentity()const;
    /** @return Original exact input roles for declarative graph contracts. */
    const PooledNativeOperandSpans& spans()const noexcept{return spans_;}
private:
    friend class IPooledImmutableNativeKernels;
    /** @return Only sealed native coordinates; no live count is dereferenced here. */
    pooled_native::OwnerInputs native()const;
    /** @brief Validate original declarations without reading mutable values. */
    void declarations()const;
    pooled_native::OwnerAction action_;
    pooled_native::SourceCondition condition_;
    pooled_native::RecordLayout records_;
    pooled_native::OwnerParameters plan_;
    PooledNativeOperandSchema schema_;
    PooledNativeOperandSpans spans_;
    std::optional<Qwen4ExpPreparedStorage> prepared_;
    std::array<std::shared_ptr<const CapturedDeviceStorageLease>,PooledNativeOperandCount> captured_; ///< Original allocator-record borrowers retire before strong Tensor owners.
};
/** @brief Exact coordinator input producer set, including the original complete counted row descriptor. */
class PooledImmutableCoordinatorOperands final
{
public:
    /**
     * @brief Retain actual original counted prefix and acquired peer/source spans.
     * @param action Closed coordinator operation.
     * @param plan Original immutable policy/owner-order geometry.
     * @param original_rows Complete unsliced counted query prefix; no mutable count is copied to the host.
     * @param spans Exact schema-owned canonical original/acquired producer roots.
     * @param condition Fixed captured source branch; only source publication/cancellation may be conditional.
     */
    PooledImmutableCoordinatorOperands(pooled_native::CoordinatorAction action,const PooledImmutableCoordinatorNativePlan& plan,
        std::optional<pooled_native::OriginalRows> original_rows,PooledNativeOperandSpans spans,
        pooled_native::SourceCondition condition=pooled_native::SourceCondition::Always);
    /** @brief Seal already-prepared addresses and actual count-owner identity before capture. */
    void seal();
    /** @brief Reject operation/geometry/root/count/runtime drift before recording native work. */
    void require(pooled_native::CoordinatorAction action,const PooledImmutableCoordinatorNativePlan& plan)const;
    /** @return Stable semantic declaration including the original count pointer/geometry. */
    std::uint64_t declarationIdentity()const;
    /** @brief Validate declarations and any installed original physical seal. */
    void validateIfSealed()const;
    /** @return Complete original count pointer, immutable expansion, spans and physical capture identity. */
    std::uint64_t captureIdentity()const;
    /** @return Exact original graph input roles. */
    const PooledNativeOperandSpans& spans()const noexcept{return spans_;}
private:
    friend class IPooledImmutableNativeKernels;
    /** @return Sealed addresses and original counted descriptor, never host-materialized values. */
    pooled_native::CoordinatorInputs native()const;
    /** @return The actual original query only for Begin; missing or unused query descriptors reject. */
    pooled_native::CoordinatorQuery query()const;
    /** @brief Validate immutable declarations and exact counted shape before a seal. */
    void declarations()const;
    pooled_native::CoordinatorAction action_;
    pooled_native::SourceCondition condition_;
    pooled_native::CoordinatorParameters plan_;
    std::optional<pooled_native::OriginalRows> rows_;
    PooledNativeOperandSchema schema_;
    PooledNativeOperandSpans spans_;
    std::optional<Qwen4ExpPreparedStorage> prepared_;
    std::array<std::shared_ptr<const CapturedDeviceStorageLease>,PooledNativeOperandCount> captured_; ///< Original allocator-record borrowers retire before strong Tensor owners.
};
}
