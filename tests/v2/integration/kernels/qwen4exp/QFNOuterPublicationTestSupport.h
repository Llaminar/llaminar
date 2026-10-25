/**
 * @file QFNOuterPublicationTestSupport.h
 * @brief Test the production outer MTP write contract with real retained hooks.
 *
 * No controller/KV execution is fabricated here. The native hooks execute under
 * the outer stage identity and its actual contract-derived capture ledger.
 */
#pragma once
#include "execution/compute_stages/stages/MTPSpeculativeStatePublicationStage.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "backends/IWorkerGPUContext.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <string_view>
#include "tensors/Tensors.h"

namespace qfn_outer_publication_test
{
using namespace llaminar2;
/**
 * @brief Resolve the actual outer contract into existing fixture tensor owners.
 * @param outer Production publisher whose contract grants the writes.
 * @param owners Exact already-materialized arena-equivalent tensor inventory.
 * @return Capture plan with no additional storage or independent write grants.
 */
inline GraphCaptureDependencyLedger::StagePlan plan(
    const MTPSpeculativeStatePublicationStage &outer,
    std::initializer_list<std::pair<BufferId, const TensorBase *>> owners)
{
    const auto contract = outer.bufferContract();
    if (!contract.inputs.empty() || !contract.inouts.empty() ||
        !contract.workspaces.empty() || !contract.weight_tensors.empty() ||
        !contract.prepared_weights.empty())
        throw std::logic_error("Follower retained publication acquired unrelated storage");
    GraphCaptureDependencyLedger::StagePlan result{
        .stage_identity = &outer, .stage_name = "outer_accepted_state_publication"};
    for (const auto &binding : contract.outputs)
    {
        if (binding.access != BufferAccess::WRITE || binding.prepare_write_storage)
            throw std::logic_error("Accepted state must use its preallocated write owner");
        const auto found = std::find_if(owners.begin(), owners.end(),
            [&](const auto &entry) { return entry.first == binding.id; });
        if (found == owners.end() || !binding.dtype ||
            std::string_view(binding.dtype) != tensorTypeName(found->second->native_type()))
            throw std::logic_error("Outer publication output has no exact typed fixture owner");
        const auto *owner = found->second->transferStorageOwner();
        if (!owner || owner->transferStorageOwner() != owner)
            throw std::logic_error("Outer publication has no canonical transfer storage owner");
        result.outputs.push_back(owner);
    }
    return result;
}
/**
 * @brief Prove the old missing-output contract fails in the real capture ledger.
 * @param worker Exact worker owning stream and graph capture.
 * @param device Exact device owning every supplied tensor.
 * @param stream Explicit nondefault publication stream.
 * @param original Correct outer plan whose write authority is removed.
 * @param restore Real native retained hook; it must reject before executing any captured work.
 */
template<class Restore>
void rejectMissingWrites(IWorkerGPUContext &worker, DeviceId device, void *stream,
    GraphCaptureDependencyLedger::StagePlan original, Restore restore)
{
    original.outputs.clear();
    GraphCaptureDependencyLedger ledger(device, stream, {original},
        "negative original missing accepted-state contract");
    auto graph = worker.createGraphCapture(stream);
    bool rejected = false;
    {
        ScopedBackendGraphCapture capture(worker, *graph,
            "negative original missing accepted-state contract", &ledger);
        if (!capture.begin())
            throw std::runtime_error("Negative publication capture could not begin");
        try
        {
            ScopedGraphCaptureStage stage(original.stage_identity);
            restore();
            stage.complete();
        }
        catch (const std::logic_error &error)
        {
            if (std::string_view(error.what()).find("published a tensor absent from its declared output contract") ==
                std::string_view::npos)
                throw;
            rejected = true;
        }
        // The capture scope closes even when the ledger rejects the write.
    }
    EXPECT_TRUE(rejected) << "The original empty output contract granted an undeclared write";
}
}
