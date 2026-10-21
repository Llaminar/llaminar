/**
 * @file AdaptiveImmutableLookupGeometry.h
 * @brief Compose existing GPU and subordinate CPU source-owned BOMs without another capacity resolver.
 *
 * These immutable projections price one serialized single-GPU/source domain.
 * Mandatory native frame/protocol, bounded source/grant scratch and exactly one
 * protocol2 mapped channel precede all row/quota decisions. Optional native GPU
 * and RAM cache storage is extended only in the final canonical PMA candidate.
 * No bytes are inferred from allocation capacity or sampled free-memory deltas.
 */
#pragma once
#include "CpuAdaptiveImmutableRowCache.h"
#include "kernels/qwen4exp/AdaptiveImmutableRowCacheGeometry.h"
#include "execution/qwen4exp/ImmutableLookupService.h"
#include <stdexcept>
namespace llaminar2
{
/** @brief Checked geometry projection, never a live allocation or admission authority. */
class AdaptiveImmutableLookupGeometry final
{
public:
    /**
     * @brief Project the sole resolved/candidate layout through actual physical owners.
     * @param layout Original source/endpoint/slot geometry; no fixed/replica arm is accepted.
     * @param source Original native metadata, including exact row elements and format.
     * @param request Original explicit GPU/CPU policies, separate from admitted slots.
     * @return Source-authored geometry using the same original-ID bound on CPU and GPU.
     * @throws std::exception For foreign source/endpoint, excessive live IDs or unsupported formats.
     */
    static AdaptiveImmutableLookupGeometry fromLayout(const ImmutableRowTableLayoutGeometry &layout,
        const ImmutableRowTableSource &source,const AdaptiveVRAMRAMSSDRequest &request)
    {
        if(layout.consumers().size()!=1 || layout.sourceServices().size()!=1 ||
            !layout.maximumInflightRows() || layout.maximumInflightRows()%16)
            throw std::invalid_argument("Adaptive lookup needs one GPU/source domain and complete original token-ID geometry");
        const auto gpu=qwen4exp::AdaptiveImmutableRowCacheGeometry::fromLayout(layout,layout.consumers().front(),source,
            static_cast<std::size_t>(layout.maximumInflightRows()/16),
            {request.gpu_policy.aging_accesses,request.gpu_policy.protected_numerator,request.gpu_policy.protected_denominator});
        const auto host=CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layout,
            layout.sourceServices().front().allocator,layout.consumers().front(),request.host.policy);
        return {gpu,host};
    }
    /** @return Exact existing GPU geometry, including optional slot and mandatory frame projections. */
    const qwen4exp::AdaptiveImmutableRowCacheGeometry &gpu() const noexcept { return gpu_; }
    /** @return Exact existing CPU source/cache geometry, with subordinate access enforced. */
    const CpuAdaptiveImmutableRowCacheGeometry &host() const noexcept { return host_; }
    /**
     * @brief Price one model-owned mandatory domain before expert/graph/KV decisions.
     * @param gpu Exact existing observed GPU resource.
     * @param host Exact existing observed CPU source resource.
     * @return Original source-owned canonical BOMs plus one existing mapped channel contribution.
     * @throws std::exception For inconsistent physical observations or arithmetic overflow.
     */
    PhysicalMemoryPlan mandatoryBOM(PhysicalMemoryResource gpu,PhysicalMemoryResource host) const
    {
        PhysicalMemoryPlanBuilder result;
        const auto native=gpu_.mandatoryBOM(gpu),source=host_.mandatoryBOM(host);
        for(const auto &bom:native.resources())result.add(bom);
        for(const auto &bom:source.resources())result.add(bom);
        result.add(host,PhysicalMemoryOwner::ImmutableLookupStaging,
            qwen4exp::ImmutableLookupService::hostBytes(gpu_.channelLayout()));
        return result.build();
    }
    /**
     * @brief Price optional actual native payload and metadata after higher priorities freeze.
     * @param gpu Original observed GPU allocator.
     * @param host Original observed CPU allocator.
     * @return Existing source-owned optional GPU/CPU cache contributions, empty for both zero capacities.
     */
    PhysicalMemoryPlan cacheBOM(PhysicalMemoryResource gpu,PhysicalMemoryResource host) const
    {
        PhysicalMemoryPlanBuilder result;
        const auto native=gpu_.cacheBOM(gpu),source=host_.cacheBOM(host);
        for(const auto &bom:native.resources())result.add(bom);
        for(const auto &bom:source.resources())result.add(bom);
        return result.build();
    }
private:
    /** @brief Store checked source-owned projections, not another layout/slot inventory. */
    AdaptiveImmutableLookupGeometry(qwen4exp::AdaptiveImmutableRowCacheGeometry gpu,
        CpuAdaptiveImmutableRowCacheGeometry host):gpu_(std::move(gpu)),host_(std::move(host)) {}
    qwen4exp::AdaptiveImmutableRowCacheGeometry gpu_;
    CpuAdaptiveImmutableRowCacheGeometry host_;
};
} // namespace llaminar2
