/**
 * @file ExpertGemmRegistryResidencyTestAccess.h
 * @brief Narrow unsealed registry corruption access for existing device-free contracts.
 *
 * Production consumes only a complete original physical seal. Existing atomic
 * map/alias rejection tests deliberately submit incomplete scopes or bindings;
 * this accessor exposes that private validator without minting a seal, source,
 * bank, engine, PMA allocation or native readiness.
 */
#pragma once
#include "loaders/ExpertGemmRegistry.h"
#include "execution/moe/MoEOverlayParticipantResidency.h"
namespace llaminar2::test
{
    struct ExpertGemmRegistryResidencyTestAccess final
    {
        /** @brief Invoke only the unsealed map-validation branch of the actual implementation. */
        static bool replace(ExpertGemmRegistry &registry,
                            std::span<const ExpertGemmRegistry::ParticipantLayerScope> scopes,
                            std::span<const ExpertGemmRegistry::ParticipantExpertBinding> bindings,
                            std::string *error = nullptr) noexcept
        {
            return registry.replaceParticipantResidencyImpl(scopes, bindings, {}, nullptr, error);
        }
        /**
         * @brief Exercise the same private atomic CPU map validator with a genuine component bank.
         * @param registry Original engine registry, not a replacement readiness owner.
         * @param residency Original source/family declaration.
         * @param bank Component data input; no physical seal is issued or claimed.
         * @param error Validation failure before map publication.
         * @return Actual atomic map validation result only, never native/capture readiness.
         */
        static bool replaceCPUFragmentBank(ExpertGemmRegistry &registry,
            const MoEOverlayParticipantResidencyRegistry &residency,
            const MoEOverlayParticipantResidencyBank &bank,std::string *error=nullptr) noexcept
        {
            const auto *participant=residency.initialOwnerMap().participantForId(bank.participant_id);
            if(!participant)return false;
            std::vector<ExpertGemmRegistry::ParticipantLayerScope> scopes;
            for(std::size_t index=0;index<bank.layers.size();++index)
                scopes.push_back({participant->domain_name,participant->device,
                    participant->world_rank_known?participant->world_rank:-1,
                    participant->domain_participant_index,bank.first_model_layer+int(index)});
            return registry.replaceParticipantResidencyImpl(scopes,{},std::span(&bank,1),&residency,error);
        }
    };
}
