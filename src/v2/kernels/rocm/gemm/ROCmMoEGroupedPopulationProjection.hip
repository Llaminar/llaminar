/**
 * @file ROCmMoEGroupedPopulationProjection.hip
 * @brief Independent compilation owner for the device-population projection transaction.
 *
 * Gate/up publishes its decision and directory before its exclusive consumers; down
 * rebuilds the directory for its fused consumer after SwiGLU. Both roles borrow
 * canonical arena storage. This shard owns and initializes its private IQ2_S
 * lookup table through the existing backend generation transaction before capture.
 */
#include <hip/hip_runtime.h>
#include "ROCmMoEGroupedPopulationDecision.h"
#include "ROCmMoEGroupedPopulationWork.h"
#include "ROCmMoEGroupedPopulationPublication.h"
#include "ROCmNativeQuantisedDecodeEquivalentMath.h"
#include "kernels/common/MoEGroupedDirectory.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "kernels/common/NativeQuantisedGroupedDecodePolicy.h"
#include "kernels/rocm/common/ROCmSignedDot4.h"
#include "ROCmNativeQuantisedGemmDecode.hip.inc"

namespace llaminar2::rocm
{
/** @brief Publish selected row steps, producing the device decision in gate/up.
 * @tparam ROLE Actual projection role. @param group_counts Authoritative expert counts.
 * @param directory Borrowed canonical work bank. @param num_experts Exact capacity.
 * @param max_directory_tiles Conservative bound. @param admission Route authority.
 * @param selection Writable GU publisher or read-only down decision binding. */
template <MoEPrefillProjectionRole ROLE>
__global__ __launch_bounds__(moe_grouped_directory::kMaximumExperts, 1)
void moe_grouped_build_population_directory_kernel(
    const int* __restrict__ group_counts,
    uint32_t* __restrict__ directory,
    int num_experts,
    int max_directory_tiles,
    ROCmMoEGroupedRouteAdmission admission,
    std::conditional_t<ROLE == MoEPrefillProjectionRole::GateUp,
        MoEGroupedPopulationPublisher,MoEGroupedPopulationBinding> selection)
{
    const int tid = threadIdx.x;
    if (!admission.allows(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled))
    {
        // Empty and sparse replays still retire the previous selection. Down
        // borrows this publication but neither role consumes a tiled directory.
        if constexpr (ROLE == MoEPrefillProjectionRole::GateUp)
            if (tid == 0) selection.publish(MoEGroupedPopulationFamily::Dot4);
        return;
    }

    if (num_experts <= 0 || num_experts > moe_grouped_directory::kMaximumExperts ||
        max_directory_tiles <= 0)
    {
        if (tid == 0)
            __builtin_trap();
        return;
    }

    __shared__ int inclusive_tile_counts[moe_grouped_directory::kMaximumExperts];
    int count = tid < num_experts ? group_counts[tid] : 0;
    if (count < 0 || static_cast<uint32_t>(count) > llaminar2::moe_grouped_directory::kMaximumRows)
    {
        if (tid < num_experts)
            __builtin_trap();
        count = 0;
    }
    MoEGroupedPopulationFamily family;
    if constexpr (ROLE == MoEPrefillProjectionRole::GateUp)
    {
        // Reuse this planner's count loads. Every wave contributes four exact
        // integer statistics; a fixed small fold publishes one shared decision.
        __shared__ int wave_stats[moe_grouped_directory::kMaximumExperts / 64][4];
        __shared__ uint32_t selected;
        const int slots = admission.activeSlots();
        if (count > slots) __builtin_trap();
        int active = count > 0, minimum = count ? count : INT_MAX;
        int maximum = count, sum = count;
        for (int step = 32; step > 0; step /= 2)
        {
            active += __shfl_down(active,step,64);
            sum += __shfl_down(sum,step,64);
            minimum = min(minimum,__shfl_down(minimum,step,64));
            maximum = max(maximum,__shfl_down(maximum,step,64));
        }
        if (tid % 64 == 0)
        {
            wave_stats[tid / 64][0] = active;
            wave_stats[tid / 64][1] = minimum;
            wave_stats[tid / 64][2] = maximum;
            wave_stats[tid / 64][3] = sum;
        }
        __syncthreads();
        if (tid == 0)
        {
            active = sum = maximum = 0; minimum = INT_MAX;
            for (int wave = 0; wave < static_cast<int>(blockDim.x) / 64; ++wave)
            {
                active += wave_stats[wave][0];
                minimum = min(minimum,wave_stats[wave][1]);
                maximum = max(maximum,wave_stats[wave][2]);
                sum += wave_stats[wave][3];
            }
            if (sum != slots) __builtin_trap();
            const auto choice = groupedPopulationFamily(MoEGroupedPopulationDomain::Qwen,{active,minimum,maximum});
            selection.publish(choice);
            selected = static_cast<uint32_t>(choice);
        }
        __syncthreads();
        family = static_cast<MoEGroupedPopulationFamily>(selected);
    }
    else family = selection.load();
    const int row_step = populationProjectionRows(ROLE,family,count);
    if (count > 0 && row_step <= 0) __builtin_trap();
    const int expert_tiles = count > 0 ? (count + row_step - 1) / row_step : 0;
    inclusive_tile_counts[tid] = expert_tiles;
    __syncthreads();

    // Hillis-Steele is deliberately used for this tiny 256-element directory:
    // its regular barriers are graph-capture friendly and much cheaper than a
    // lane-zero serial scan on gfx906.
    for (int stride = 1; stride < static_cast<int>(blockDim.x); stride <<= 1)
    {
        const int addend = tid >= stride
                               ? inclusive_tile_counts[tid - stride]
                               : 0;
        __syncthreads();
        inclusive_tile_counts[tid] += addend;
        __syncthreads();
    }

    const int total_tiles = inclusive_tile_counts[blockDim.x - 1];
    if (total_tiles > max_directory_tiles)
    {
        if (tid == 0)
            __builtin_trap();
        return;
    }

    // Launch completion orders both compact payload and count before the consumer.
    // Unused capacity is intentionally untouched and can contain poisoned data.
    if (tid == 0)
        directory[MoEGroupedPopulationDirectoryLayout(max_directory_tiles).liveCountIndex()] = total_tiles;
    if (tid < num_experts)
    {
        const int first_tile = tid == 0 ? 0 : inclusive_tile_counts[tid - 1];
        for (int tile = 0; tile < expert_tiles; ++tile)
        {
            const uint32_t first_row = static_cast<uint32_t>(tile * row_step);
            directory[first_tile + tile] =
                (first_row << moe_grouped_directory::kExpertBits) |
                static_cast<uint32_t>(tid);
        }
    }
}



namespace grouped_wmma_detail
{
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__)
#include "ROCmMoEGroupedWmmaMath.hip.inc"
#include "ROCmMoEGroupedPopulationProjection.hip.inc"
#endif
}

bool submitMoEGroupedPopulation(MoEPrefillProjectionRole role, MoEGroupedPopulationWorkspace workspace,
    uint32_t codebooks,const MoEGroupedWmmaArguments &args,
    MoEGroupedPopulationControl control,void *stream,MoEGroupedPrefillRoleSubmission *submitted)
{
    if (submitted) *submitted = {};
    const auto native_stream = static_cast<hipStream_t>(stream);
    const bool gateup = role == MoEPrefillProjectionRole::GateUp;
    if ((role != MoEPrefillProjectionRole::GateUp && role != MoEPrefillProjectionRole::Down) ||
        !workspace.selection || workspace.selection != control.binding.address() ||
        !workspace.directory || workspace.directory != args.directory ||
        gateup != (args.second_descriptors != nullptr)) return false;
    if (!native_stream || native_stream == hipStreamLegacy || native_stream == hipStreamPerThread ||
        control.omitted != MoEGroupedPopulationFamily::Invalid ||
        args.branch_suppression != MoEGroupedBranchSuppression::None ||
        !args.activations || !args.activation_scales || !args.group_counts || !args.group_offsets ||
        !args.directory || !args.descriptors || !args.output || args.directory_capacity <= 0 ||
        args.rows != 512 || args.top_k != 8 || args.experts != 256 || args.device < 0 ||
        !nativeGroupedDevice(args.device) || args.n != (gateup ? 512 : 2048) || args.k != (gateup ? 2048 : 512) ||
        codebooks != (uint32_t{1} << (gateup ? NATIVE_QUANTISED_GEMM_IQ2_S : NATIVE_QUANTISED_GEMM_IQ4_XS_SOURCE)) ||
        args.input != (gateup ? MoEGroupedWmmaInput::OriginalTokens : MoEGroupedWmmaInput::GroupedRows) ||
        (gateup && (!args.group_token_indices || !args.second_output)) ||
        (!gateup && args.second_output) || !args.dot_admission.binds(role,args.n,args.k) ||
        !args.dot_admission.rowPlan().isWide() ||
        args.dot_admission.rowPlan().floor() != MoEGroupedWmmaRowFloor::Sixteen) return false;
    // Capture and replay share one source arithmetic contract.
    if (args.partitions.forCodebook(gateup ? 13 : 4) !=
        MoEProjectionNumericalContract::orderedKPartitionsForWidth(args.k)) return false;
    if (MoEGroupedPopulationDirectoryLayout(args.directory_capacity).words() > workspace.directory_words) return false;
    int stream_device = -1;
    if (hipStreamGetDevice(native_stream,&stream_device) != hipSuccess || stream_device != args.device) return false;
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__)
    const dim3 directory_threads(moe_grouped_directory::directoryThreads(args.experts));
    if (gateup)
        moe_grouped_build_population_directory_kernel<MoEPrefillProjectionRole::GateUp><<<1,directory_threads,0,native_stream>>>(
            args.group_counts,workspace.directory,args.experts,args.directory_capacity,args.admission,MoEGroupedPopulationPublisher(workspace.selection));
    else
        moe_grouped_build_population_directory_kernel<MoEPrefillProjectionRole::Down><<<1,directory_threads,0,native_stream>>>(
            args.group_counts,workspace.directory,args.experts,args.directory_capacity,args.admission,control.binding);
    if (hipGetLastError() != hipSuccess) return false;
    const dim3 grid(kMoEGroupedPopulationWorkers,1,gateup ? 2 : 1);
    using grouped_wmma_detail::PopulationArithmetic;
    MoEGroupedPrefillRoleSubmission pending;
    if (gateup)
    {
        // Independent allocations let cold GU retain its narrow ordered stage.
        // Both consumers borrow the same device decision and directory; every
        // tile has exactly one arithmetic owner, including one-row tails.
        grouped_wmma_detail::populationProjection<NATIVE_QUANTISED_GEMM_IQ2_S,true,PopulationArithmetic::Ordered>
            <<<grid,128,0,native_stream>>>(args,control);
        if (hipGetLastError() != hipSuccess) return false;
        pending.record({MoEGroupedPrefillEngine::OrderedDot4,MoEGroupedWmmaRowFloor::EveryExpert,12,128,128,0,1,0});
        grouped_wmma_detail::populationProjection<NATIVE_QUANTISED_GEMM_IQ2_S,true,PopulationArithmetic::Native>
            <<<grid,128,0,native_stream>>>(args,control);
        if (hipGetLastError() != hipSuccess) return false;
        pending.record({MoEGroupedPrefillEngine::WmmaM16N32,MoEGroupedWmmaRowFloor::EveryExpert,16,32,128,0,1,0});
    }
    else
    {
        grouped_wmma_detail::populationProjection<NATIVE_QUANTISED_GEMM_IQ4_XS_SOURCE,false><<<grid,128,0,native_stream>>>(args,control);
        if (hipGetLastError() != hipSuccess) return false;
        pending.record({MoEGroupedPrefillEngine::PopulationAdaptive,
            MoEGroupedWmmaRowFloor::Sixteen,16,32,128,16,1,128});
    }
    if (submitted) *submitted = pending;
    return true;
#else
    return false;
#endif
}


bool initializeMoEGroupedPopulationGrids(int device,const void *iq2s)
{
    if (!iq2s || HipDeviceGuard::setDevice(device) != hipSuccess) return false;
    return hipMemcpyToSymbol(d_iq2s_grid_gemm,iq2s,sizeof(d_iq2s_grid_gemm)) == hipSuccess;
}
}
