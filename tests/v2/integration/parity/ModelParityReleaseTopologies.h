/**
 * @file ModelParityReleaseTopologies.h
 * @brief The three physical topology requests in the bounded release matrix.
 *
 * Dense Qwen3.8 and QFN share these exact participants and native collective
 * boundaries. Model definitions own arithmetic, storage and MTP policy; no CI
 * runner reconstructs topology from a case name or adds storage/placement axes.
 */
#pragma once
#include "ModelParityDefinition.h"

namespace llaminar2::test::parity
{
/** @return The complete release topology inventory, independent of model identity. */
inline std::vector<ModelParityTopologyDefinition> modelParityReleaseTopologies()
{
    return {
        {.test_id="Release_ROCm4_TP", .kind=ModelParityTopologyKind::RankLocalTensorParallel,
         .participants={{GlobalDeviceAddress::rocm(0),0},{GlobalDeviceAddress::rocm(1),0},
                        {GlobalDeviceAddress::rocm(2),0},{GlobalDeviceAddress::rocm(3),0}},
         .collective=Collective::RCCL},
        {.test_id="Release_CUDA2_TP", .kind=ModelParityTopologyKind::RankLocalTensorParallel,
         .participants={{GlobalDeviceAddress::cuda(0),0},{GlobalDeviceAddress::cuda(1),0}},
         .collective=Collective::NCCL},
        {.test_id="Release_CUDA2_ROCm4_CPU_TPPP", .kind=ModelParityTopologyKind::RankLocalPipelineParallel,
         .participants={{GlobalDeviceAddress::cuda(0),0},{GlobalDeviceAddress::cuda(1),0},
                        {GlobalDeviceAddress::rocm(0),0},{GlobalDeviceAddress::rocm(1),0},
                        {GlobalDeviceAddress::rocm(2),0},{GlobalDeviceAddress::rocm(3),0},
                        {GlobalDeviceAddress::cpu(0),0}},
         .collective=Collective::HETEROGENEOUS, .pipeline_stage_sizes={2,4,1},
         .pipeline_weights={1.0f,2.0f,1.0f}, .tensor_parallel_collective=Collective::HETEROGENEOUS,
         .pipeline_tensor_parallel_collectives={Collective::NCCL,Collective::RCCL,Collective::HOST}}
    };
}
} // namespace llaminar2::test::parity
