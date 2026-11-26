/**
 * @file CUDANativeQuantisedGemvDispatchHeuristic.cpp
 * @brief Single host C++ translation unit owning generated CUDA NativeQuantised dispatch tables.
 *
 * Compiles the untouched 76,381-line `CUDANativeQuantisedGemvDispatchHeuristicGenerated.inc`
 * once with the host C++ compiler instead of eight times inside nvcc GEMV shards.
 */
#include "kernels/cuda/gemm/CUDANativeQuantisedGemvDispatchHeuristic.h"
#include <cstddef>
#include <cstdint>

namespace llaminar2::cuda::native_quantised_dispatch::detail
{
using NativeGemvShape = llaminar2::cuda::native_quantised_dispatch::NativeGemvShape;
#include "kernels/cuda/gemm/CUDANativeQuantisedGemvDispatchHeuristicGenerated.inc"
} // namespace llaminar2::cuda::native_quantised_dispatch::detail

namespace llaminar2::cuda::native_quantised_dispatch
{

int resolveGeneratedTargetBlocksKBlocks(
    int grid_n, int k_groups, int target_blocks,
    int min_kgroups_per_cta, int max_kb)
{
    return detail::resolveGeneratedTargetBlocksKBlocks(
        grid_n, k_groups, target_blocks, min_kgroups_per_cta, max_kb);
}

int resolveGeneratedCanonicalTargetBlocksKBlocks(
    int grid_n, int k_groups, int target_blocks,
    int min_kgroups_per_cta, int max_kb)
{
    return detail::resolveGeneratedCanonicalTargetBlocksKBlocks(
        grid_n, k_groups, target_blocks, min_kgroups_per_cta, max_kb);
}

int resolveGeneratedBlocksPerPartitionKBlocks(
    int k_groups, int blocks_per_partition, int max_kb)
{
    return detail::resolveGeneratedBlocksPerPartitionKBlocks(
        k_groups, blocks_per_partition, max_kb);
}

bool selectGeneratedDispatchByCodebook(
    uint8_t policy_cb, bool graph_captured, int m, int n, int k,
    NativeGemvShape &shape, GeneratedDispatchTuning &tuning)
{
    detail::GeneratedDispatchTuning raw{};
    bool ok = false;
    switch (policy_cb)
    {
#define LLAMINAR_CUDA_GEN_CASE(CB) \
    case CB: ok = detail::selectGeneratedDispatch<CB>(graph_captured, m, n, k, shape, raw); break;
        LLAMINAR_CUDA_GEN_CASE(0)
        LLAMINAR_CUDA_GEN_CASE(4)
        LLAMINAR_CUDA_GEN_CASE(5)
        LLAMINAR_CUDA_GEN_CASE(6)
        LLAMINAR_CUDA_GEN_CASE(7)
        LLAMINAR_CUDA_GEN_CASE(8)
        LLAMINAR_CUDA_GEN_CASE(9)
        LLAMINAR_CUDA_GEN_CASE(10)
        LLAMINAR_CUDA_GEN_CASE(11)
        LLAMINAR_CUDA_GEN_CASE(12)
        LLAMINAR_CUDA_GEN_CASE(13)
        LLAMINAR_CUDA_GEN_CASE(14)
        LLAMINAR_CUDA_GEN_CASE(15)
        LLAMINAR_CUDA_GEN_CASE(16)
        LLAMINAR_CUDA_GEN_CASE(17)
        LLAMINAR_CUDA_GEN_CASE(19)
#undef LLAMINAR_CUDA_GEN_CASE
    default:
        return false;
    }
    if (!ok)
        return false;
    tuning = {raw.tile_n, raw.cpt, raw.target_waves, raw.mkg, raw.max_kb, raw.force_two_phase, raw.exact_kb};
    return true;
}

bool selectGeneratedGroupedTuningByCodebook(
    uint8_t policy_cb, bool graph_captured, int m, int n, int k,
    GeneratedGroupedTuning &tuning)
{
    detail::GeneratedGroupedTuning raw{};
    bool ok = false;
    switch (policy_cb)
    {
#define LLAMINAR_CUDA_GROUPED_CASE(CB) \
    case CB: ok = detail::selectGeneratedGroupedTuning<CB>(graph_captured, m, n, k, raw); break;
        LLAMINAR_CUDA_GROUPED_CASE(0)
        LLAMINAR_CUDA_GROUPED_CASE(4)
        LLAMINAR_CUDA_GROUPED_CASE(5)
        LLAMINAR_CUDA_GROUPED_CASE(6)
        LLAMINAR_CUDA_GROUPED_CASE(7)
        LLAMINAR_CUDA_GROUPED_CASE(8)
        LLAMINAR_CUDA_GROUPED_CASE(9)
        LLAMINAR_CUDA_GROUPED_CASE(10)
        LLAMINAR_CUDA_GROUPED_CASE(11)
        LLAMINAR_CUDA_GROUPED_CASE(12)
        LLAMINAR_CUDA_GROUPED_CASE(13)
        LLAMINAR_CUDA_GROUPED_CASE(14)
        LLAMINAR_CUDA_GROUPED_CASE(15)
        LLAMINAR_CUDA_GROUPED_CASE(16)
        LLAMINAR_CUDA_GROUPED_CASE(17)
        LLAMINAR_CUDA_GROUPED_CASE(19)
#undef LLAMINAR_CUDA_GROUPED_CASE
    default:
        return false;
    }
    if (!ok)
        return false;
    tuning = {static_cast<GeneratedGroupedKernel>(raw.kernel), raw.grouped_rows};
    return true;
}

} // namespace llaminar2::cuda::native_quantised_dispatch
