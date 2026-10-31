/**
 * @file PooledImmutableSourceKernels.h
 * @brief Real retained RAM/SSD provider bridge for accepted native pooled grants.
 *
 * This code bundle retains the existing service-issued source channel. Source
 * storage and owner/coordinator roots have their original PMA owners; canonical
 * request control and output remain graph-owned. Every source gate is a fixed
 * captured operation, with no host cache decisions or alternate delivery path.
 */
#pragma once
#include "PooledImmutableSourceGeometry.h"
#include "PooledImmutableSourceOps.h"
#include <memory>
namespace llaminar2 {class DeviceGenerationControlBinding;}
namespace llaminar2::qwen4exp {
class PooledImmutableSourcePreparedStorage;
/** @brief Distinct source producers/readers with explicit owner/coordinator graph edges between them. */
enum class PooledSourceAction:std::uint32_t {Plan,Publish,Acquire,Read,ReadersRetired,ObserveRetired,BeforeReady};
/** @brief Native output already retained/sealed by its actual reusable graph stage. */
struct PooledSourceQuery {
    float* output=nullptr; ///< Original canonical embedding output, without aliasing native authority.
    std::size_t elements=0; ///< This graph's exact physical FP32 envelope, no larger than the shared source declaration.
};
/** @brief Checked participant-local source code; arbitrary directed owner edges remain explicit transport contracts. */
class IPooledImmutableSourceKernels {
public:
    /** @brief Release only after graph terminal and final source readers; retained source bindings own mapped backing. */
    virtual ~IPooledImmutableSourceKernels()=default;
    /** @return Complete original source/frame/owner/coordinator declarations. */
    virtual const PooledImmutableSourceGeometry& geometry()const noexcept=0;
    /** @return Existing service-issued provider/channel binding, retained unchanged. */
    virtual const ImmutableSubordinateLookupBinding& source()const noexcept=0;
    /** @brief Initialize source protocol metadata once, retaining its exact setup event even on partial failure. */
    void initialize(PooledImmutableSourcePreparedStorage& storage,void* exact_stream)const;
    /**
     * @brief Publish only immutable I/O admission and its actual retained-worker ticket outside capture.
     * @param storage Same initialized original source roots.
     * @param generation Exact admitted service generation.
     * @param invocation Exact retained-worker dispatch ticket for that generation.
     * @param exact_stream Producer stream ordered before the complete captured parent.
     * @note Successful request reset preserves sequence history, warmed slots and graph topology.
     */
    void admit(PooledImmutableSourcePreparedStorage& storage,std::uint64_t generation,std::uint64_t invocation,void* exact_stream)const;
    /**
     * @brief Enqueue one fixed source operation over actual accepted grants and canonical controller.
     * @param action Explicit captured source producer/reader operation.
     * @param storage Original source-issued root owner.
     * @param control Existing canonical request authority, never a host-owned health mirror.
     * @param query Exact original FP32 output span for Read, absent for other operations.
     * @param exact_stream Actual non-default participant stream with graph producer edges already acquired.
     */
    void enqueue(PooledSourceAction action,PooledImmutableSourcePreparedStorage& storage,
        const DeviceGenerationControlBinding& control,PooledSourceQuery query,void* exact_stream)const;
    /** @brief Close only at the complete parent's canonical terminal, after source reader retirement. */
    void closeWhenTerminal(PooledImmutableSourcePreparedStorage& storage,
        const DeviceGenerationControlBinding& control,void* exact_stream)const;
    /** @brief Explicit event-ordered request reset closes after actual source readers, independent of the generation terminal predicate. */
    void closeForReset(PooledImmutableSourcePreparedStorage& storage,
        const DeviceGenerationControlBinding& control,void* exact_stream)const;
    /** @return Complete source/code/root/runtime identity; stages additionally bind exact controller/output roles. */
    std::uint64_t captureIdentity(const PooledImmutableSourcePreparedStorage& storage)const;
protected:
    /** @return All immutable codebook addresses, wall-clock deadline, format and launch policies. */
    virtual std::uint64_t codeIdentity()const noexcept=0;
    /** @brief Receive only checked original addresses; perform no allocation or host observation. */
    virtual void initializeNative(const pooled_source::DeviceArgs&,void*)const=0;
    /** @brief Preserve device I/O sequence history while publishing the original admission/ticket. */
    virtual void admitNative(const pooled_source::DeviceArgs&,std::uint64_t,std::uint64_t,void*)const=0;
    /** @brief Submit the fixed exact-live-byte source operation on the supplied exact stream. */
    virtual void launch(PooledSourceAction,const pooled_source::DeviceArgs&,PooledSourceQuery,void*)const=0;
    /** @brief Complete-parent terminal close uses only original canonical device predicates. */
    virtual void closeNative(const pooled_source::DeviceArgs&,const int*,void*)const=0;
private:
    /** @brief Reject foreign service/root associations before any native operation. */
    void requireStorage(const PooledImmutableSourcePreparedStorage&)const;
    /** @return Original sealed native coordinates, never live host-materialized counts. */
    pooled_source::DeviceArgs args(const PooledImmutableSourcePreparedStorage&,const DeviceGenerationControlBinding*)const;
};
/** @return CUDA source bridge with actual original codebooks and closed service binding. */
std::unique_ptr<IPooledImmutableSourceKernels> createCUDAPooledImmutableSourceKernels(PooledImmutableSourceGeometry,
    ImmutableSubordinateLookupBinding);
/** @return HIP source bridge under the same original service/protocol/byte/lifetime contract. */
std::unique_ptr<IPooledImmutableSourceKernels> createROCmPooledImmutableSourceKernels(PooledImmutableSourceGeometry,
    ImmutableSubordinateLookupBinding);
}
