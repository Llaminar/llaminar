/**
 * @file NativeCooperativeKernel.cuh
 * @brief Native compiler adapter for cold admission and exact-stream cooperative submission.
 *
 * Include from the kernel's CUDA/HIP translation unit. Hardware discovery stays
 * in the existing once-only authority; only exact-function resources are queried
 * during preparation. Submission is one native cooperative launch and performs
 * no device/property query, allocation, synchronization or alternative launch.
 */
#pragma once
#include "NativeCooperativeKernel.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <limits>
#include <stdexcept>
#include <string>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include "backends/rocm/ROCmHardwareMetadata.h"
#define NCK_RT(Name) hip##Name
#define NCK_BACKEND ROCm
#else
#include <cuda_runtime.h>
#include "backends/cuda/CUDAHardwareMetadata.h"
#define NCK_RT(Name) cuda##Name
#define NCK_BACKEND CUDA
#endif
namespace llaminar2
{
/** @brief Native boundary specialized only by the owning GPU compiler. */
template<> class NativeCooperativeKernelBoundary<NativeCooperativeBackend::NCK_BACKEND> final
{
    /** @brief Preserve the exact native failure without retrying another path. */
    static void checked(NCK_RT(Error_t) status,const char* operation)
    {
        if(status!=NCK_RT(Success))throw std::runtime_error(std::string(operation)+": "+NCK_RT(GetErrorString)(status));
    }
public:
    /**
     * @brief Admit the exact loaded entry using existing immutable device facts.
     * @param device Original worker's native participant.
     * @param function Exact compiled entry, never a proxy with different resources.
     * @param threads Exact physical CTA width.
     * @param dynamic_shared Exact shared-memory bytes for this program.
     * @param generation Existing prepared owner lifecycle identity, required nonzero.
     * @param policy Actual submission mechanism; ordinary grids still require exact resident resources.
     * @return Sealed successful residency and compiler-resource evidence.
     * @throws std::invalid_argument For absent capability or malformed geometry.
     * @throws std::logic_error During capture or on a foreign current device.
     */
    static NativeCooperativeKernelAdmission prepare(DeviceId device,const void* function,unsigned threads,
        std::size_t dynamic_shared,std::uint64_t generation,
        NativeResidentLaunchPolicy policy=NativeResidentLaunchPolicy::Cooperative,
        NativeCoResidencyReservation co_residency=NativeCoResidencyReservation::none())
    {
        if(isGraphCaptureActive())throw std::logic_error("Cooperative program admission is forbidden during capture");
        if(!function||!threads||!generation)throw std::invalid_argument("Cooperative program has incomplete identity");
#if defined(__HIPCC__)
        if(!device.is_rocm())throw std::invalid_argument("HIP cooperative program has a foreign participant");
        const auto& hardware=rocmHardwareMetadata().at(device.ordinal);
        const int processors=hardware.compute_units;
        const std::size_t physical_shared=hardware.lds_per_block;
#else
        if(!device.is_cuda())throw std::invalid_argument("CUDA cooperative program has a foreign participant");
        const auto& hardware=cudaHardwareMetadata().at(device.ordinal);
        const int processors=hardware.sm_count;
        const std::size_t physical_shared=hardware.shared_memory_per_block;
#endif
        if((policy!=NativeResidentLaunchPolicy::Cooperative&&policy!=NativeResidentLaunchPolicy::Ordinary)||
            (policy==NativeResidentLaunchPolicy::Cooperative&&!hardware.can_launch_cooperative)||threads>unsigned(hardware.maximum_threads_per_block)||
            dynamic_shared>physical_shared||processors<=0||
            co_residency.reserved_processors>=unsigned(processors))
            throw std::invalid_argument("Cooperative program exceeds its admitted native physical capability");
        const unsigned available_processors=unsigned(processors)-co_residency.reserved_processors;
        int current=-1;
        checked(NCK_RT(GetDevice)(&current),"cooperative program current device");
        if(current!=device.ordinal)throw std::logic_error("Cooperative program admission has a foreign current worker");
        NCK_RT(FuncAttributes) attributes{};
        checked(NCK_RT(FuncGetAttributes)(&attributes,function),"cooperative program function resources");
        int blocks=0;
        checked(NCK_RT(OccupancyMaxActiveBlocksPerMultiprocessor)(&blocks,function,int(threads),dynamic_shared),
            "cooperative program native occupancy");
        if(blocks<=0||attributes.maxThreadsPerBlock<int(threads)||attributes.localSizeBytes!=0||
            attributes.sharedSizeBytes>physical_shared-dynamic_shared||
            std::uint64_t(available_processors)*unsigned(blocks)>std::numeric_limits<unsigned>::max())
            throw std::invalid_argument("Cooperative program has invalid residency or private memory");
        return {device,function,generation,threads,dynamic_shared,available_processors,
            available_processors*unsigned(blocks),
            {unsigned(attributes.numRegs),unsigned(attributes.maxThreadsPerBlock),unsigned(blocks),
             attributes.localSizeBytes,attributes.sharedSizeBytes},policy,co_residency};
    }
    /**
     * @brief Submit one exact native cooperative node, including during parent capture.
     * @param admission Same cold loaded-program observation retained by its owner.
     * @param parameters Original typed argument pointers, copied by the native API.
     * @param grid Positive grid no larger than the admitted resident bound.
     * @param stream Exact caller-owned non-default publication stream.
     * @param generation Same existing owner generation used during admission.
     * @throws std::invalid_argument For invalid stream, geometry or argument array.
     * @throws std::logic_error For a stale prepared owner generation.
     * @throws std::runtime_error On native submission failure, without retry.
     */
    static void launch(const NativeCooperativeKernelAdmission& admission,void** parameters,unsigned grid,
        void* stream,std::uint64_t generation)
    {
        const auto native=static_cast<NCK_RT(Stream_t)>(stream);
        if(admission.policy()!=NativeResidentLaunchPolicy::Cooperative||
            !native||native==NCK_RT(StreamLegacy)||native==NCK_RT(StreamPerThread)||!parameters||
            !grid||grid>admission.maximumGrid())
            throw std::invalid_argument("Cooperative launch has an invalid stream, arguments or resident grid");
        if(generation!=admission.generation())throw std::logic_error("Cooperative launch owner generation changed");
        checked(NCK_RT(LaunchCooperativeKernel)(admission.function(),dim3(grid),dim3(admission.threads()),parameters,
            admission.dynamicSharedBytes(),native),"native cooperative program launch");
    }
};
}
#undef NCK_RT
#undef NCK_BACKEND
