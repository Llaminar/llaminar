/**
 * @file ExpertGemmRegistryResidencyTestAccess.h
 * @brief Device-free registry identity economy and unsealed corruption contracts.
 *
 * Production consumes only a complete original physical seal. Existing atomic
 * map/alias rejection tests deliberately submit incomplete scopes or bindings;
 * this accessor exposes that private validator without minting a seal, source,
 * bank, engine, PMA allocation or native readiness. Hash diagnostics exercise the
 * original private key/map types without loading weights or creating engines.
 */
#pragma once
#include <chrono>
#include "loaders/ExpertGemmRegistry.h"
#include "execution/moe/MoEOverlayParticipantResidency.h"
namespace llaminar2::test
{
    struct ExpertGemmRegistryResidencyTestAccess final
    {
        /** @brief Actual registry hash/bucket statistics; no engines or device allocations. */
        struct HashDistribution
        {
            std::size_t keys=0, distinct_hashes=0, maximum_equal_hashes=0;
            std::size_t successful_lookup_comparisons=0;
            std::size_t measured_lookup_hits=0;
            std::uint64_t lookup_nanoseconds=0, negative_lookup_nanoseconds=0;
            std::size_t measured_lookup_misses=0;
        };
        /**
         * @brief Exercise original private keys/hash and actual unordered lookup buckets.
         * @param devices Arbitrary positive metadata-only participant population.
         * @param layers Original representative model layer count.
         * @param experts Original representative experts per layer.
         * @param resident_experts Optional sparse GPU subset; includes original domain+participant aliases.
         * @return Full-hash distribution and exact comparisons needed to find every key.
         * @throws std::invalid_argument If any declared dimension is zero.
         */
        static HashDistribution hashDistribution(std::size_t devices, int layers, int experts,
            std::optional<int> resident_experts={})
        {
            if(!devices || layers<=0 || experts<=0 ||
               (resident_experts && (*resident_experts<=0 || *resident_experts>=experts || devices>std::size_t(experts))))throw std::invalid_argument("Hash fixture requires positive geometry");
            using Key=ExpertGemmRegistry::Key;
            using Hash=ExpertGemmRegistry::KeyHash;
            std::unordered_map<Key,ExpertGemmRegistry::Entry,Hash> entries;
            entries.reserve(devices*std::size_t(layers)*std::size_t(resident_experts.value_or(experts))*3*(resident_experts?2:1));
            std::unordered_map<std::size_t,std::size_t> histogram;
            for(std::size_t device=0;device<devices;++device)
                for(int layer=0;layer<layers;++layer)
                    for(int expert=0;expert<resident_experts.value_or(experts);++expert)
                        for(int role=0;role<3;++role)
                        {
                            const auto type=resident_experts?DeviceType::ROCm:
                                device%3==0?DeviceType::CPU:device%3==1?DeviceType::CUDA:DeviceType::ROCm;
                            const Key key{resident_experts?"rocm_continuation":device%2?"routed":"replica",DeviceId{type,int(device)},layer,
                                resident_experts?int(device)*experts/int(devices)+expert:expert,
                                static_cast<ExpertGemmRegistry::WeightRole>(role),int(device/4),int(device%4)};
                            entries.emplace(key,ExpertGemmRegistry::Entry{});
                            ++histogram[Hash{}(key)];
                            if(resident_experts)
                            {
                                auto domain_key=key;
                                domain_key.participant_world_rank=-1;
                                domain_key.participant_index=-1;
                                entries.emplace(domain_key,ExpertGemmRegistry::Entry{});
                                ++histogram[Hash{}(domain_key)];
                            }
                        }
            HashDistribution result{.keys=entries.size(),.distinct_hashes=histogram.size()};
            for(const auto& [hash,count]:histogram)result.maximum_equal_hashes=std::max(result.maximum_equal_hashes,count);
            for(const auto& [key,value]:entries)
            {
                const auto bucket=entries.bucket(key);
                bool found=false;
                for(auto it=entries.begin(bucket);it!=entries.end(bucket);++it)
                {
                    ++result.successful_lookup_comparisons;
                    if(it->first==key){found=true;break;}
                }
                if(!found || entries.find(key)==entries.end())throw std::logic_error("Actual registry bucket lost its original key");
            }
            // Measure only actual successful hash-table lookups, excluding key
            // generation, allocation, histogram construction and bucket inspection.
            const auto begin=std::chrono::steady_clock::now();
            for(int pass=0;pass<2;++pass)
                for(const auto& [key,value]:entries)
                    result.measured_lookup_hits+=entries.find(key)!=entries.end();
            result.lookup_nanoseconds=std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now()-begin).count();
            const auto negative_begin=std::chrono::steady_clock::now();
            for(int pass=0;pass<2;++pass)
                for(const auto& [key,value]:entries)
                {
                    // Sparse case queries a legal but nonresident expert on
                    // the SAME device/scope; full cases query an absent layer.
                    auto missing=key;
                    if(resident_experts)missing.expert=(missing.expert+*resident_experts)%experts;
                    else missing.layer=layers;
                    result.measured_lookup_misses+=entries.find(missing)==entries.end();
                }
            result.negative_lookup_nanoseconds=std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now()-negative_begin).count();
            return result;
        }
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
