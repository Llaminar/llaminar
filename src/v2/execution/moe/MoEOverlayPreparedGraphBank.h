/**
 * @file MoEOverlayPreparedGraphBank.h
 * @brief Borrow exact admitted expert engines for a participant-local graph family.
 *
 * Runtime, GraphScope and the canonical owner map authenticate setup identity;
 * the existing registry/payload resolver supplies prepared engines and shared
 * allocation lifetimes. This adapter performs no packing or allocation of
 * weights and owns no placement, capacity, or live-byte ledger.
 */
#pragma once
#include "MoEExpertOverlayGraphScope.h"
#include "MoEExpertOwnerMap.h"
#include "MoEOverlayParticipantResidency.h"
#include "loaders/ExpertGemmRegistry.h"
#include "tensors/TensorKernels.h"
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2
{
/** @brief Declarative logical ownership and global expert extent of one graph bank. */
struct MoEOverlayPreparedGraphBankRequest
{
    DeviceId device;
    const ITPContext *tp_context;
    int tp_participant;
    std::string domain;
    const MoEExpertOwnerParticipant *participant;
    int layer;
    int experts;
    int d_model;
    int expert_intermediate;
    int first_expert;
    int local_experts;
    std::reference_wrapper<const std::vector<bool>> expert_mask;
};

/** @brief Borrowed complete triplets with shared ownership of their actual execution allocations. */
struct MoEOverlayPreparedGraphBank
{
    std::vector<ITensorGemm *> gate;
    std::vector<ITensorGemm *> up;
    std::vector<ITensorGemm *> down;
    std::vector<std::shared_ptr<ITensorGemm>> owners;
};

/**
 * @brief Adopt an authenticated existing participant bank without repacking any source.
 * @param runtime Retained immutable original domain/rank/endpoint declarations.
 * @param registry Model-owned prepared engine and lifetime authority.
 * @param request Graph-owned identity, source layer, geometry and local execution mask.
 * @return Exact global-indexed engine triplets and their original shared lifetimes.
 * @throws std::invalid_argument For stale or ambiguous source/endpoint/geometry.
 * @throws std::runtime_error For a missing owned prepared projection.
 */
[[nodiscard]] inline MoEOverlayPreparedGraphBank adoptMoEOverlayPreparedGraphBank(
    const MoEExpertOverlayRuntimePlan &runtime, const ExpertGemmRegistry &registry,
    const MoEOverlayPreparedGraphBankRequest &request)
{
    const auto &mask = request.expert_mask.get();
    const int count = request.local_experts < 0 ? request.experts : request.local_experts;
    if (request.layer < 0 || request.experts <= 0 || request.d_model <= 0 ||
        request.expert_intermediate <= 0 || request.first_expert < 0 ||
        request.first_expert > request.experts || count < 0 || count > request.experts - request.first_expert ||
        (!mask.empty() && mask.size() != static_cast<size_t>(request.experts)))
        throw std::invalid_argument("ExpertOverlay prepared graph bank has invalid expert geometry");

    const auto scope = request.participant
        ? MoEExpertOverlayGraphScope::resolve(runtime, request.device, request.participant->address)
        : MoEExpertOverlayGraphScope::resolveForTPParticipant(
              runtime, request.device, request.tp_context, request.tp_participant);
    std::optional<MoEExpertOwnerParticipant> selected;
    if (request.participant)
    {
        const auto &owner = *request.participant;
        if (!owner.world_rank_known || owner.device != request.device ||
            (!request.domain.empty() && request.domain != owner.domain_name) ||
            !scope.contains(owner.domain_name, owner.device, owner.world_rank, owner.domain_participant_index))
            throw std::invalid_argument("ExpertOverlay prepared graph bank has a foreign explicit owner");
        selected = owner;
    }
    else
    {
        // The canonical owner map supplies complete typed descriptors. Scope
        // only selects among that source; it never synthesizes a second map.
        const auto owners = MoEExpertOwnerMap::build(runtime.sourcePlan());
        for (const auto &owner : owners.participants())
        {
            if ((!request.domain.empty() && owner.domain_name != request.domain) ||
                !scope.contains(owner.domain_name, owner.device, owner.world_rank, owner.domain_participant_index))
                continue;
            if (selected)
                throw std::invalid_argument("ExpertOverlay prepared graph bank has multiple logical owners");
            selected = owner;
        }
    }
    if (!selected)
        throw std::invalid_argument("ExpertOverlay prepared graph bank has no logical owner");

    std::vector<bool> required = mask;
    if (required.empty())
    {
        required.assign(static_cast<size_t>(request.experts), false);
        for (int expert = request.first_expert; expert < request.first_expert + count; ++expert)
            required[expert] = true;
    }
    std::vector<MoEOverlayPreparedExpertPayload> payloads;
    std::string error;
    if (!resolveMoEOverlayPreparedExpertPayloads(registry, *selected, request.layer,
            request.experts, required, payloads, &error))
        throw std::runtime_error("ExpertOverlay prepared graph bank is incomplete: " + error);

    MoEOverlayPreparedGraphBank bank;
    bank.gate.resize(request.experts, nullptr);
    bank.up.resize(request.experts, nullptr);
    bank.down.resize(request.experts, nullptr);
    for (int expert = 0; expert < request.experts; ++expert)
    {
        if (!required[expert])
            continue;
        const auto &payload = payloads[expert];
        // Projection roles are semantic geometry: a complete triplet from the
        // correct owner is still invalid if it belongs to another shard shape.
        const auto validate_projection = [](const ITensorGemm &engine, int n, int k)
        {
            if (engine.get_n() != n || engine.get_k() != k)
                throw std::invalid_argument("ExpertOverlay prepared graph bank projection N/K mismatch");
        };
        validate_projection(*payload.gate(), request.expert_intermediate, request.d_model);
        validate_projection(*payload.up(), request.expert_intermediate, request.d_model);
        validate_projection(*payload.down(), request.d_model, request.expert_intermediate);
        bank.gate[expert] = payload.gate().get();
        bank.up[expert] = payload.up().get();
        bank.down[expert] = payload.down().get();
        bank.owners.push_back(payload.gate());
        bank.owners.push_back(payload.up());
        bank.owners.push_back(payload.down());
    }
    return bank;
}
} // namespace llaminar2
