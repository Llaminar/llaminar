/**
 * @file CPUPerLayerPacketTestAdmission.h
 * @brief Explicit packet-root admission for positive nonserial CPU stage fixtures.
 *
 * Fixtures author the exact four-root allocation declaration. Existing supplied
 * or prepared-store authorities are preserved; missing packet capacity remains
 * an error rather than extending admission after publication.
 * Per-layer CPU packet roots retain their own exact backing allocation claims.
 */
#pragma once
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "loaders/PreparedWeightStore.h"
namespace llaminar2::test
{
/**
 * @brief Admit the exact packet owned by one positive fixture stage.
 * @param params Immutable setup dimensions and packet binding.
 * @return Parameters with one explicit CPU allocation issuer.
 */
inline MoELocalExpertStage::Params admitCPUPerLayerPacketForTest(MoELocalExpertStage::Params params)
{
    if(!params.input_rows && params.input_rows_lifetime)
        params.input_rows=params.input_rows_lifetime.get();
    if(!params.device_id.is_cpu() || params.serial_compact_buffer_arena || !params.input_rows ||
        params.physical_memory_authority) return params;
    if(params.prepared_store && params.prepared_store->physicalMemoryAuthority())
    {
        params.physical_memory_authority=params.prepared_store->physicalMemoryAuthority();
        return params;
    }
    const auto rows=params.graph_row_capacity?params.graph_row_capacity:params.input_rows->row_capacity;
    const auto layout=CPUCompactPacketStorageLayout::declare(rows,params.d_model,params.top_k,params.device_id);
    const auto bytes=layout.physicalBytes();
    PhysicalMemoryPlanBuilder plan;
    plan.add({0,params.device_id,bytes,bytes},PhysicalMemoryOwner::ActivationArena,bytes);
    params.physical_memory_authority=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
    return params;
}
}
