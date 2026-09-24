/**
 * @file PreparedOverlaySourceRetirement.cpp
 * @brief Identity-checked reclamation of prepared accelerator expert sources.
 *
 * Random placement compacts disjoint GGUF rows into large owned host tensors.
 * Once their exact engines have been uploaded and published, retaining those
 * copies until graph destruction needlessly doubles the model's host footprint.
 * This setup-only transaction proves replacement ownership before retirement;
 * it creates neither a second memory ledger nor a shadow execution authority.
 */
#include "PreparedOverlaySourceRetirement.h"

#include "ExpertGemmRegistry.h"
#include "WeightMetadataRegistry.h"
#include "WeightPlan.h"
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "tensors/Tensors.h"

#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace llaminar2
{
    std::size_t retirePreparedOverlaySources(
        const FrozenModelWeightSet &weights,
        const MoEExpertOverlayPreparationPlan &plan,
        const ExpertGemmRegistry &engines,
        const WeightMetadataRegistry &metadata)
    {
        std::vector<TensorBase *> sources;
        std::unordered_set<TensorBase *> visited;
        for (const auto &binding : weights.bindings())
        {
            const auto device = binding.residency.home_device;
            auto *const tensor = binding.tensor;
            if (!isRoutedExpertRole(binding.identity.role) || !device.is_gpu() ||
                binding.identity.derivation != WeightDerivationKind::ExpertSlice ||
                binding.slice.expert_ids.empty() ||
                !plan.hasRequestsForDevice(device) || !tensor ||
                tensor->is_raw_data_released() || tensor->is_view())
                continue;

            // Shared host consumers win over this binding's GPU-only intent.
            // In particular a CPU FP engine may still borrow these exact bytes.
            const auto residency = metadata.residency(tensor);
            if (!residency)
                throw std::logic_error("Prepared overlay source has no host policy: " +
                                       binding.identity.canonical_name);
            const auto releasable = [](WeightHostPolicy policy)
            {
                return policy == WeightHostPolicy::ReleasableAfterPreparation ||
                       policy == WeightHostPolicy::RequiredUntilPreparedOrTransferred;
            };
            if (!releasable(binding.residency.host_policy) ||
                !releasable(residency->host_policy))
                continue;

            if (!binding.slice.inner_is_presliced ||
                binding.tensor_owner.get() != tensor || tensor->shape().size() != 3u ||
                tensor->shape()[2] != binding.slice.expert_ids.size())
                throw std::logic_error("Prepared overlay source is not an exact owned expert selection: " +
                                       binding.identity.canonical_name);

            const auto role = binding.identity.role == WeightRole::MoEExpertGate
                                  ? ExpertGemmRegistry::WeightRole::GATE
                              : binding.identity.role == WeightRole::MoEExpertUp
                                  ? ExpertGemmRegistry::WeightRole::UP
                                  : ExpertGemmRegistry::WeightRole::DOWN;
            const auto &id = binding.identity;
            for (const int expert : binding.slice.expert_ids)
            {
                const auto *request = plan.requestForParticipant(
                    id.overlay_domain, device, id.overlay_participant_world_rank,
                    id.overlay_participant_index, id.layer, expert, role);
                const auto participant = engines.getEngineLifetimeForParticipant(
                    id.overlay_domain, device, id.overlay_participant_world_rank,
                    id.overlay_participant_index, id.layer, expert, role);
                const auto domain = engines.getEngineLifetimeForDomain(
                    id.overlay_domain, device, id.layer, expert, role);
                if (!request || !participant || !domain ||
                    participant.get() != domain.get() ||
                    participant.owner_before(domain) || domain.owner_before(participant))
                    throw std::logic_error("Prepared overlay source lacks complete participant ownership: " +
                                           id.canonical_name + " expert=" + std::to_string(expert) +
                                           " device=" + device.to_string());
            }
            if (visited.insert(tensor).second)
                sources.push_back(tensor);
        }

        // Validate the entire transaction before its first irreversible release.
        // Tensor metadata and binding identity remain alive for graph construction;
        // only the now-unused raw storage (and any host registration) is retired.
        std::size_t bytes = 0u;
        for (auto *tensor : sources)
        {
            bytes += tensor->size_bytes();
            tensor->release_host_weight_data();
        }
        return bytes;
    }
}
