/**
 * @file CPUStandaloneRouterTestAdmission.h
 * @brief Explicit exact Q8 publication root for isolated CPU router fixtures.
 *
 * A router tested without its expert graph has one physical payload: canonical
 * Q8 hidden rows. Paired graph fixtures instead collect the complete expert
 * manifest and replace this setup-oracle backing before actual paired replay.
 * No production allocation or default authority is installed by this helper.
 */
#pragma once
#include "execution/compute_stages/stages/MoERoutingStage.h"
#include "kernels/cpu/moe/CPUMoEKernel.h"
#include "CPURouterKernelTestAdmission.h"

namespace llaminar2::test
{
/**
 * @brief Author and materialize only an isolated router's actual Q8 root.
 * @param params Explicit router fixture source geometry, updated with its owner.
 * @note GPU and intentionally invalid scalar fixtures retain their own boundary.
 */
inline void admitCPUStandaloneRouterForTest(MoERoutingStage::Params &params)
{
    if (!params.device_id.is_cpu() || params.seq_len <= 0 || params.d_model <= 0)
        return;
    if(!params.routed_pipeline_kernel_owner)
        params.routed_pipeline_kernel_owner=std::make_shared<MoERoutedPipelineKernelOwner>();
    auto &owner=*params.routed_pipeline_kernel_owner;
    if(!owner.kernel) owner.kernel=std::make_unique<CPUMoEKernel>();
    auto *kernel=dynamic_cast<CPUMoEKernel *>(owner.kernel.get());
    if(!kernel) throw std::invalid_argument("Isolated CPU router requires an explicit CPU kernel");
    admitCPURouterKernelForTest(*kernel,params.seq_len,params.d_model,params.device_id);
}
}
