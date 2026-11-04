/**
 * @file CPUJointSourceTestPlan.h
 * @brief One original sparse-source fixture topology shared by loader lifetime and native-format window proofs.
 */
#pragma once
#include <algorithm>
#include "execution/moe/MoEExpertOwnerMap.h"
namespace llaminar2::test {
/** @return Original E512 tier placement:64 GPU expert IDs and448 CPU joint IDs; tests may author a sparse static assignment. */
inline MoERoutedExpertPlacementPlan cpuJointSourceTestPlan() {
 using namespace llaminar2;
 MoERoutedExpertPlacementPlan plan;plan.enabled=true;plan.first_model_layer=7;
 plan.topology=RoutedExpertPlacementTopology::TieredOverlay;
 plan.continuation_domain=plan.base_model_domain=plan.shared_expert_domain="compute";
 RoutedExpertDomain gpu;gpu.name="compute";gpu.scope=ExecutionDomainScope::RANK_LOCAL;
 gpu.backend=CollectiveBackendType::NCCL;gpu.owner_rank=0;gpu.participants={GlobalDeviceAddress::cuda(0)};
 gpu.routed_compute_policy=RoutedExpertComputePolicy::Apportioned;
 RoutedExpertDomain cpu;cpu.name="cpu_joint";cpu.scope=ExecutionDomainScope::NODE_LOCAL;
 cpu.backend=CollectiveBackendType::UPI;cpu.owner_rank=-1;
 cpu.participants={GlobalDeviceAddress::cpu(0),GlobalDeviceAddress::cpu(1)};cpu.world_ranks={0,1};
 cpu.routed_compute_policy=RoutedExpertComputePolicy::TensorSharded;
 cpu.routed_phase_policy=RoutedExpertPhasePolicy::Uniform;
 cpu.routed_decode_assignment_policy=cpu.routed_prefill_assignment_policy=RoutedExpertAssignmentPolicy::StaticOwner;
 plan.domains={gpu,cpu};
 plan.routed_tiers={{.name="gpu",.domain="compute",.priority=0,.max_experts_per_layer=64},
  {.name="cpu",.domain="cpu_joint",.priority=1,.max_experts_per_layer=448,.fallback=true}};
 std::vector<int> tiers(512,1);std::fill_n(tiers.begin(),64,0);
 plan.placements={{.layer=7,.routed_expert_tier=std::move(tiers)}};return plan;
}
}
