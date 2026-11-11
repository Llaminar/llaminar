/**
 * @file MoEOverlayRankBatchPreparation.h
 * @brief Reusable original-rank setup of the existing compact activation transport.
 *
 * Main-model and stage owners share the same geometry/factory and exact
 * endpoint first-touch. Every original parent rank visits every channel's
 * setup consensus; only its true source/target endpoints allocate the admitted
 * payload. This is cold setup, never another inference transport or probe.
 */
#pragma once
#include "MoEExpertOwnerMap.h"
#include "MoEOverlayInferenceTransaction.h"
#include "MoEOverlayRankBatchTransport.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "interfaces/IMPIContext.h"
namespace llaminar2 {
/** @brief Original immutable placement/geometry and the actual parent communicator. */
struct MoEOverlayRankBatchPreparationInput {
    const MoERoutedExpertPlacementPlan& placement;
    const MoEExpertOwnerMap& owner_map;
    const ClusterInventory& inventory;
    const MoEOverlayInferenceGraphFamilyIdentity& graph_family;
    std::shared_ptr<IMPIContext> original_context;
    std::shared_ptr<PhysicalMemoryAuthority> physical_memory_authority; ///< Same original rank memory owner as every prepared stage.
    std::shared_ptr<const MoEOverlayActivationBatchInventory> activation_batch_inventory; ///< Same selected source declaration as original admission.
    std::shared_ptr<const MoEExpertOwnerMap> original_cpu_joint_owners;
    int d_model;
    int top_k;
};
/**
 * @brief Create only the existing source-owned compact rank-batch channels.
 * @param input Original plan, complete original topology and admitted graph family.
 * @return Exact local registry, or absent when this declaration requires no channels.
 * @throws std::exception for invalid original identities or any endpoint setup failure.
 */
std::shared_ptr<MoEOverlayRankBatchTransportRegistry> prepareOriginalMoEOverlayRankBatchTransports(
    const MoEOverlayRankBatchPreparationInput& input);
}
