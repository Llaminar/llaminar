/**
 * @file GPUFlatGraphStructure.h
 * @brief Cold structural observations in caller-owned original-PMA scratch.
 *
 * Counts are observations and source-authored scope, never memory limits or a
 * General graph certificate. No graph/node handle escapes the setup operation.
 */
#pragma once
#include "backends/DeviceId.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace llaminar2 {
/** @brief Canonical inspected native kinds; child/control/host/allocation are unsupported. */
enum class GPUFlatGraphNodeKind:std::uint8_t { Kernel, Copy, Set, Empty, EventWait, EventRecord, Count };
/** @brief Immutable source-declared inspection envelope, not a caller-created admission permission. */
struct GPUFlatGraphLimits final {
    std::uint32_t maximumNodes=0,maximumEdges=0;
    std::uint32_t allowedKinds=0;
    /** @return Explicit canonical-kind bit for a finite closed set. */
    static constexpr std::uint32_t bit(GPUFlatGraphNodeKind kind){
        const auto ordinal=static_cast<unsigned>(kind);
        if(ordinal>=static_cast<unsigned>(GPUFlatGraphNodeKind::Count))throw std::invalid_argument("Unknown flat graph kind");
        return std::uint32_t{1}<<ordinal;
    }
    /** @return Whether every scratch extent/kind can be represented before native calls. */
    constexpr bool valid()const noexcept{
        constexpr auto all=(std::uint32_t{1}<<static_cast<unsigned>(GPUFlatGraphNodeKind::Count))-1;
        return maximumNodes && allowedKinds && !(allowedKinds&~all);
    }
    bool operator==(const GPUFlatGraphLimits&)const=default;
};
/** @brief Complete successful flat/default-edge DAG observation; no native resources are owned here. */
struct GPUFlatGraphStructure final {
    std::array<std::uint32_t,static_cast<unsigned>(GPUFlatGraphNodeKind::Count)> kinds{};
    std::uint32_t nodes=0,edges=0;
    bool linearChain=false; ///< Complete single-stream chain, including the one-node case.
    bool operator==(const GPUFlatGraphStructure&)const=default;
};
/** @brief Exact in-place inspection layout shared by declaration and the SDK-adapter algorithm. */
struct GPUFlatGraphScratchLayout final {
    std::size_t nodes=0,from=0,to=0,degree=0,queue=0,bytes=0;
    /** @brief Derive all pointer/index arrays without SDK calls or allocation.
     * @param limits Closed source scope used by the actual recording owner.
     * @return Exact logical scratch extent; its CPUStorageRoot supplies allocator rounding.
     * @throws std::invalid_argument For an empty/unknown scope.
     * @throws std::overflow_error For an unrepresentable host extent.
     */
    static GPUFlatGraphScratchLayout fromLimits(const GPUFlatGraphLimits& limits){
        if(!limits.valid())throw std::invalid_argument("Flat graph scratch requires a complete source scope");
        GPUFlatGraphScratchLayout out;std::size_t cursor=0;
        const auto append=[&](std::size_t count,std::size_t width){
            if(count>(std::numeric_limits<std::size_t>::max()-cursor)/width)
                throw std::overflow_error("Flat graph scratch extent overflow");
            const auto first=cursor;cursor+=count*width;return first;
        };
        out.nodes=append(limits.maximumNodes,sizeof(std::uintptr_t));
        out.from=append(limits.maximumEdges,sizeof(std::uintptr_t));
        out.to=append(limits.maximumEdges,sizeof(std::uintptr_t));
        out.degree=append(limits.maximumNodes,sizeof(std::uint32_t));
        out.queue=append(limits.maximumNodes,sizeof(std::uint32_t));out.bytes=cursor;return out;
    }
};
/** @return Actual compiled native adapter object extent, without creating a worker or querying an SDK.
 * @param device Exact compiled CUDA or ROCm class.
 * @throws std::invalid_argument For CPU/unsupported backends.
 */
std::size_t nativeGraphCaptureObjectBytes(DeviceId device);
}
