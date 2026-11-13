/**
 * @file PooledImmutableLocalHotKernels.h
 * @brief Checked exact-stream local native-cache phase submission, with one model output writer.
 *
 * Probe classifies/pins only. AcceptResult orders original compact native bytes;
 * LocalRead is the sole embedding producer. Policy, fills and retirement follow
 * the actual consumed/discarded graph endpoints, never an inferred receipt.
 */
#pragma once
#include "PooledImmutableLocalHotAccess.h"
#include "PooledImmutableLocalHotTransportProtocol.h"
#include "PooledImmutableLocalHotNativeTypes.h"
#include "kernels/common/NativeEmbeddingIQTables.h"
#include <memory>
namespace llaminar2 {class DeviceGenerationControlBinding;}
namespace llaminar2::qwen4exp {
class PooledNativeTransferSpan;
class PooledImmutableLocalHotPreparedStorage;
class PooledImmutableLocalHotOperands;

/** @brief Fixed native operations; each phase preserves its own original exact dependency. */
enum class PooledImmutableLocalHotAction : std::uint32_t {
    Probe,AcceptResult,LocalRead,ConsumeDemand,CopyFills,CompleteFills,
    Adopt,Discard,Cancel,Retire,ResetRequest
};
/** @brief Stateless backend code bundle; all live roots and ready/setup resources belong to storage. */
class IPooledImmutableLocalHotKernels {
public:
    virtual ~IPooledImmutableLocalHotKernels()=default;
    /** @brief Initialize exact admitted metadata outside capture on the original setup stream. */
    void initialize(PooledImmutableLocalHotPreparedStorage&,void* exact_setup_stream)const;
    /** @brief Enqueue one captured fixed phase; no allocation, host state/count read or fallback exists. */
    void enqueue(PooledImmutableLocalHotAction,PooledImmutableLocalHotPreparedStorage&,
        const PooledImmutableLocalHotOperands&,void* exact_stream)const;
    /** @brief Project actual lifetime totals only after the original whole parent and auxiliary body retired.
     * @note The original caller retains both arena output and local roots through its existing terminal event.
     */
    void packTerminal(PooledImmutableLocalHotPreparedStorage&,const DeviceGenerationControlBinding& original_control,
        int requests,const PooledNativeTransferSpan& original_terminal_output,void* exact_stream)const;
    /** @return Immutable kernel/layout/arithmetic version. */
    virtual std::uint64_t codeIdentity()const noexcept=0;
    /** @return One original GPU backend; no runtime discovery occurs. */
    virtual DeviceType backend()const noexcept=0;
protected:
    /** @brief Backend terminal pack shares one native implementation on CUDA and HIP. */
    virtual void launchTerminal(const local_hot::NativeTerminal&,void* exact_stream)const=0;
    /** @brief Bind source IQ tables from the existing backend before any capture. */
    virtual NativeEmbeddingIQTables prepareTables(DeviceId,TensorType)const=0;
    /** @brief Initialize canonical local metadata using exact raw arguments issued only by this bridge. */
    virtual void initializeNative(const local_hot::NativeRoots&,void* exact_stream)const=0;
    /** @brief Actual backend lowering of an already checked fixed action. */
    virtual void launchFrame(PooledImmutableLocalHotAction,const local_hot::NativeFrame&,void* exact_stream)const=0;
    /** @brief Actual original hash/count producer is carried only by Probe. */
    virtual void launchProbe(const local_hot::NativeQuery&,void* exact_stream)const=0;
    /** @brief Explicit next-prefill count specialization; original contiguous Probe is unchanged. */
    virtual void launchNextWindowProbe(const local_hot::NativeNextWindowQuery&,void* exact_stream)const=0;
    /**
     * @brief Cooperatively compare actual demand IDs against the original retained future frame.
     * @param demand Checked complete hash/count/request producer and original local roots.
     * @param exact_stream Same non-default stream selected by the enclosing capture.
     */
    virtual void launchAdoption(const local_hot::NativeAdoption& demand,void* exact_stream)const=0;
    /** @brief Native LocalRead receives its sole actual FP32 output. */
    virtual void launchRead(const local_hot::NativeRead&,void* exact_stream)const=0;
    /** @brief Once-per-request reset consumes its exact original local admission producer. */
    virtual void launchRequestReset(const local_hot::NativeRequestReset&,void* exact_stream)const=0;
    /** @brief Only response acquisition carries the original actual request producer. */
    virtual void launchResult(const local_hot::NativeResult&,void* exact_stream)const=0;
};
/** @return CUDA code bundle without device discovery, allocation, source/service construction. */
std::unique_ptr<IPooledImmutableLocalHotKernels> createCUDAPooledImmutableLocalHotKernels();
/** @return Symmetric HIP code bundle retaining the SAME native source formats and phase contracts. */
std::unique_ptr<IPooledImmutableLocalHotKernels> createROCmPooledImmutableLocalHotKernels();
} // namespace llaminar2::qwen4exp
