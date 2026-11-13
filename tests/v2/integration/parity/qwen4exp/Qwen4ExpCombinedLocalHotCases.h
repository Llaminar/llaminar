/**
 * @file Qwen4ExpCombinedLocalHotCases.h
 * @brief Bounded local-cache candidates from original CUDA/ROCm owner-subset requests.
 *
 * These cells retain the exact original source, topology, arithmetic, ordinal
 * expert policy and R3 predictor envelope. Combined C0 is an engaged local
 * request with mandatory native miss/scatter roots, not shared-only fallback.
 * Source admission, public ordinary generation and selected R3 native exact replay
 * are independent obligations. Declaration never qualifies native execution,
 * numerical accuracy, local hits, read-ahead benefit or performance.
 */
#pragma once
#include "Qwen4ExpOwnerSubsetParityCases.h"
#include "Qwen4ExpAllGPUQ6MTPParityCases.h"
#include <initializer_list>

namespace llaminar2::test::parity::qwen4exp {
/**
 * @brief Select explicit original R3 MTP modes without changing their source/topology.
 * @param selected Exact code-owned original request; no topology or owner is synthesized.
 * @param modes Nonempty distinct subset of the original Off/fixed1/fixed3/dynamic modes.
 * @return Original selected R3 requests with engaged local C0 and zero shared optional ceilings.
 * @throws std::invalid_argument For empty, repeated or unsupported mode declarations.
 * @throws std::logic_error If an original typed seed or selected R3 mode disappears.
 */
inline std::vector<ModelParityCase> qwen4ExpCombinedLocalHotC0CasesFromOriginalSeed(
    const Qwen4ExpOwnerSubsetModelRequest& selected, std::initializer_list<ModelParityMTP> modes)
{
    if (modes.size() == 0)
        throw std::invalid_argument("Combined local cache requires an explicit original MTP subset");
    for (const auto mode : modes)
    {
        if ((mode != ModelParityMTP::Off && mode != ModelParityMTP::Depth1 &&
             mode != ModelParityMTP::Depth3 && mode != ModelParityMTP::DynamicDepth) ||
            std::count(modes.begin(), modes.end(), mode) != 1)
            throw std::invalid_argument("Combined local cache requires distinct original Off/fixed1/fixed3/dynamic modes");
    }
    if (selected.movement != ModelParityExpertMovement::Static)
        throw std::invalid_argument("This combined-cache projection requires its original StaticOrdinal source");
    const auto& seeds=qwen4ExpOwnerSubsetSourceAdmissionSeeds();
    std::optional<std::string> original_identity;
    for(std::size_t i=0;i<kQwen4ExpOwnerSubsetModelRequests.size();++i) {
        const auto& request=kQwen4ExpOwnerSubsetModelRequests[i];
        if(request.target==selected.target && request.continuation_mask==selected.continuation_mask &&
           request.backing==selected.backing && request.cache==selected.cache &&
           request.movement==selected.movement) {
            if(original_identity)throw std::logic_error("Combined C0 source seed is ambiguous");
            original_identity=seeds.at(i).topology.test_id+"_R3";
        }
    }
    if(!original_identity)throw std::logic_error("Combined C0 lost its selected original source seed");
    std::vector<ModelParityCase> result;
    for(auto cell:qwen4ExpOwnerSubsetExecutionCases(ModelParityExpertOverlayPolicy{})) {
        if(cell.topology.test_id!=*original_identity ||
           std::find(modes.begin(),modes.end(),cell.mtp)==modes.end())continue;
        auto& policy=*cell.ngram_table_policy;
        policy.gpu_adaptive.local_hot_cache=AdaptiveGPULocalHotCacheRequest{};
        // C0 retains all mandatory roots and original owners while requesting
        // no optional shared/local cache payload. Positive siblings reuse the
        // same source and canonical budget projector below.
        for (auto& owner : policy.gpu_adaptive.placement.owners)
        {
            owner.maximum_cache_rows = 0;
            owner.cache_physical_byte_limit = 0;
        }
        policy.validate();
        cell.topology.test_id+="_CombinedLocalHotC0Candidate";
        result.push_back(std::move(cell));
    }
    if(result.size()!=modes.size())throw std::logic_error("Combined C0 lost its exact original R3 mode projection");
    return result;
}
/**
 * @brief Preserve the existing ROCm2 RAM/mask2 wrapper and every emitted identity.
 * @param modes Explicit original R3 mode subset.
 * @return Same original ROCm requests as before generic seed selection.
 */
inline std::vector<ModelParityCase> qwen4ExpCombinedLocalHotC0CasesForMTP(
    std::initializer_list<ModelParityMTP> modes)
{
    return qwen4ExpCombinedLocalHotC0CasesFromOriginalSeed(
        {Qwen4ExpHardwareTarget::ROCm2,2,AdaptiveImmutableRowBacking::RAM,
         Qwen4ExpOwnerSubsetCachePressure::Zero,ModelParityExpertMovement::Static}, modes);
}
/** @return The unchanged initial Off/fixed1 combined C0 projection. */
inline std::vector<ModelParityCase> qwen4ExpCombinedLocalHotC0Cases()
{
    return qwen4ExpCombinedLocalHotC0CasesForMTP({ModelParityMTP::Off,ModelParityMTP::Depth1});
}
/** @return One original ordinary request; no exact-MTP or numerical credit is inferred. */
inline std::vector<ModelParityCase> qwen4ExpCombinedLocalHotC0OrdinaryCases()
{
    auto cases=qwen4ExpCombinedLocalHotC0Cases();
    std::erase_if(cases,[](const auto& cell){return cell.mtp!=ModelParityMTP::Off;});
    return cases;
}
/** @return One original fixed1 exact request, retaining its full inherited parity gates. */
inline std::vector<ModelParityCase> qwen4ExpCombinedLocalHotC0ExactCases()
{
    auto cases=qwen4ExpCombinedLocalHotC0Cases();
    std::erase_if(cases,[](const auto& cell){return cell.mtp!=ModelParityMTP::Depth1;});
    return cases;
}
/** @brief Separate original physical policies with matched optional byte ceilings, never matched total PMA claims. */
enum class Qwen4ExpLocalHotBudgetProfile {
    SharedOnly, ConsumerLocal, CombinedEqual, CombinedUnequal, CombinedZeroPositive
};
/** @return Actual original continuation consumers in their authored domain order; no GPU count is inferred. */
inline std::vector<AdaptiveGPUCacheParticipant> qwen4ExpLocalHotOriginalConsumers(const ModelParityCase& cell)
{
    if(!cell.topology.expert_overlay_plan)throw std::logic_error("Local cache candidates lost their original expert topology");
    const auto& plan=*cell.topology.expert_overlay_plan;
    const auto domain=std::find_if(plan.domains.begin(),plan.domains.end(),[&](const auto& value){return value.name==plan.continuation_domain;});
    if(domain==plan.domains.end() || domain->owner_rank<0 || !domain->world_ranks.empty())
        throw std::logic_error("Local cache candidates require their original rank-local continuation authority");
    std::vector<AdaptiveGPUCacheParticipant> consumers;
    for(const auto& address:domain->participants) {
        if(address.device_type!=DeviceType::CUDA && address.device_type!=DeviceType::ROCm)
            throw std::logic_error("Local cache candidate has a non-GPU continuation consumer");
        consumers.push_back({domain->owner_rank,address.device_type,address.device_ordinal});
    }
    if(consumers.size()<2)throw std::logic_error("Unequal cache candidate requires at least two original consumers");
    return consumers;
}
/**
 * @brief Authored optional byte ceilings for uniform original cache roles.
 * Physical totals remain determined by canonical admission, not these caps.
 */
struct Qwen4ExpLocalHotExplicitOptionalBudgets
{
    std::size_t pool_owner_bytes; ///< Ceiling per original pool owner; zero for ConsumerLocal.
    std::size_t consumer_bytes; ///< Ceiling per original local consumer; zero for SharedOnly.
};
/**
 * @brief Apply one bounded optional-budget profile without changing source, arithmetic or expert ownership.
 * @param original Original combined C0 request, including its authored R3 and selected MTP policy.
 * @param profile Explicit physical cache hierarchy; no mode is entered as a fallback.
 * @param explicit_budgets Optional authored uniform ceilings; absent retains every historical128MiB identity.
 * @return Same real model case with historical128MiB ceilings or explicit authored caps.
 * @throws std::invalid_argument For unknown profiles or caps assigned to an absent role.
 * @throws std::logic_error If original ownership, divisibility or seed identity is lost.
 * Mandatory roots/event/source costs differ between hierarchies and remain PMA-admitted separately.
 * The rows ceiling is a correctness bound; only actual source admission proves fit and positive slots.
 */
inline ModelParityCase qwen4ExpLocalHotBudgetCase(ModelParityCase original,Qwen4ExpLocalHotBudgetProfile profile,
    std::optional<Qwen4ExpLocalHotExplicitOptionalBudgets> explicit_budgets = std::nullopt)
{
    constexpr std::size_t total=std::size_t{128}<<20,shared=total/2;
    constexpr std::uint64_t maximumRows=std::uint64_t{1}<<20;
    const auto consumers=qwen4ExpLocalHotOriginalConsumers(original);
    const auto degree=consumers.size();
    if(shared%degree || shared%(degree-1))throw std::logic_error("Declared optional budget must divide the actual original consumers exactly");
    auto& g=original.ngram_table_policy->gpu_adaptive;
    if(g.placement.mode!=AdaptiveGPUCachePlacementMode::DeclaredRankLocal || g.placement.owners.empty())
        throw std::logic_error("Local cache candidate lost its original shared storage owners");
    if(shared%g.placement.owners.size())throw std::logic_error("Shared budget must divide the original physical owners exactly");
    for(auto& owner:g.placement.owners){owner.maximum_cache_rows=maximumRows;owner.cache_physical_byte_limit=shared/g.placement.owners.size();}
    auto& local=g.local_hot_cache;local=AdaptiveGPULocalHotCacheRequest{
        .maximum_cache_rows=maximumRows,.cache_physical_byte_limit=shared/degree,.policy=g.gpu_policy};
    std::string suffix;
    switch(profile) {
    case Qwen4ExpLocalHotBudgetProfile::SharedOnly:
        local.reset();
        for(auto& owner:g.placement.owners)owner.cache_physical_byte_limit=total/g.placement.owners.size();
        suffix="SharedOnlyMatchedOptional128MiB";break;
    case Qwen4ExpLocalHotBudgetProfile::ConsumerLocal:
        local.reset();g.placement={.mode=AdaptiveGPUCachePlacementMode::ConsumerLocal};
        g.maximum_gpu_cache_rows=maximumRows;g.gpu_cache_physical_byte_limit=total/degree;
        suffix="ConsumerLocalMatchedOptional128MiB";break;
    case Qwen4ExpLocalHotBudgetProfile::CombinedEqual:
        suffix="CombinedEqualMatchedOptional128MiB";break;
    case Qwen4ExpLocalHotBudgetProfile::CombinedUnequal:
        for(const auto& consumer:consumers)local->consumer_overrides.push_back({consumer,maximumRows,shared/degree});
        local->consumer_overrides.front().cache_physical_byte_limit/=2;
        local->consumer_overrides.back().cache_physical_byte_limit+=shared/degree/2;
        suffix="CombinedUnequalMatchedOptional128MiB";break;
    case Qwen4ExpLocalHotBudgetProfile::CombinedZeroPositive:
        local->maximum_cache_rows=0;local->cache_physical_byte_limit=0;
        for(std::size_t index=1;index<degree;++index)local->consumer_overrides.push_back({consumers[index],maximumRows,shared/(degree-1)});
        suffix="CombinedZeroPositiveMatchedOptional128MiB";break;
    default:throw std::invalid_argument("Unknown original local cache budget profile");
    }
    if (explicit_budgets)
    {
        const auto ceilings = *explicit_budgets;
        switch (profile)
        {
        case Qwen4ExpLocalHotBudgetProfile::ConsumerLocal:
            if (ceilings.pool_owner_bytes != 0)
                throw std::invalid_argument("ConsumerLocal has no shared pool budget role");
            g.gpu_cache_physical_byte_limit = ceilings.consumer_bytes;
            suffix = "ConsumerLocalAuthoredOptionalBudgets";
            break;
        case Qwen4ExpLocalHotBudgetProfile::SharedOnly:
            if (ceilings.consumer_bytes != 0)
                throw std::invalid_argument("SharedOnly has no participant-local budget role");
            for (auto& owner : g.placement.owners)
                owner.cache_physical_byte_limit = ceilings.pool_owner_bytes;
            suffix = "SharedOnlyAuthoredOptionalBudgets";
            break;
        case Qwen4ExpLocalHotBudgetProfile::CombinedEqual:
            for (auto& owner : g.placement.owners)
                owner.cache_physical_byte_limit = ceilings.pool_owner_bytes;
            local->cache_physical_byte_limit = ceilings.consumer_bytes;
            suffix = "CombinedEqualAuthoredOptionalBudgets";
            break;
        default:
            throw std::invalid_argument("Uniform explicit budgets require a uniform original profile");
        }
        suffix += "_PoolB" + std::to_string(ceilings.pool_owner_bytes) +
            "_ConsumerB" + std::to_string(ceilings.consumer_bytes);
    }
    original.ngram_table_policy->validate();
    const std::string old="_CombinedLocalHotC0Candidate";
    if(!original.topology.test_id.ends_with(old))throw std::logic_error("Local budget projection replaced its original C0 seed");
    original.topology.test_id.resize(original.topology.test_id.size()-old.size());
    original.topology.test_id+="_"+suffix+"Candidate";
    return original;
}
/** @return Ten bounded source-owned requests: five hierarchy/budget profiles, ordinary and fixed1 only. */
inline std::vector<ModelParityCase> qwen4ExpLocalHotBudgetCases()
{
    std::vector<ModelParityCase> result;
    for(const auto& original:qwen4ExpCombinedLocalHotC0Cases())
        for(const auto profile:{Qwen4ExpLocalHotBudgetProfile::SharedOnly,Qwen4ExpLocalHotBudgetProfile::ConsumerLocal,
            Qwen4ExpLocalHotBudgetProfile::CombinedEqual,Qwen4ExpLocalHotBudgetProfile::CombinedUnequal,
            Qwen4ExpLocalHotBudgetProfile::CombinedZeroPositive})
            result.push_back(qwen4ExpLocalHotBudgetCase(original,profile));
    return result;
}
/** @return Five original ordinary generation requests; no numerical or speed qualification is implied. */
inline std::vector<ModelParityCase> qwen4ExpLocalHotBudgetOrdinaryCases()
{
    auto cases=qwen4ExpLocalHotBudgetCases();std::erase_if(cases,[](const auto& cell){return cell.mtp!=ModelParityMTP::Off;});return cases;
}
/** @return Five fixed1 native exact requests with original full-stage/reset contracts. */
inline std::vector<ModelParityCase> qwen4ExpLocalHotBudgetExactCases()
{
    auto cases=qwen4ExpLocalHotBudgetCases();std::erase_if(cases,[](const auto& cell){return cell.mtp!=ModelParityMTP::Depth1;});return cases;
}
/**
 * @return Two original ROCm2 RAM/Off policies with explicit diagnostic caps.
 * These are first measured-cost adjustment candidates, not matched physical
 * budgets or inherited numerical qualifications. All other seed fields remain.
 */
inline std::vector<ModelParityCase> qwen4ExpLocalHotMeasuredBudgetCandidates()
{
    const auto original = qwen4ExpCombinedLocalHotC0CasesForMTP({ModelParityMTP::Off}).front();
    return {
        qwen4ExpLocalHotBudgetCase(original, Qwen4ExpLocalHotBudgetProfile::ConsumerLocal,
            Qwen4ExpLocalHotExplicitOptionalBudgets{0, 104697856}),
        qwen4ExpLocalHotBudgetCase(original, Qwen4ExpLocalHotBudgetProfile::CombinedEqual,
            Qwen4ExpLocalHotExplicitOptionalBudgets{71344128, 35672064})};
}
/**
 * @return One separate original ROCm2 RAM/Off policy using the nearest canonical
 * ratio-preserving cap point. Fresh admission must validate the predicted64B
 * residual; all prior functional and measured candidates remain unchanged.
 */
inline std::vector<ModelParityCase> qwen4ExpLocalHotRefinedBudgetCandidates()
{
    const auto original = qwen4ExpCombinedLocalHotC0CasesForMTP({ModelParityMTP::Off}).front();
    return {qwen4ExpLocalHotBudgetCase(original, Qwen4ExpLocalHotBudgetProfile::CombinedEqual,
        Qwen4ExpLocalHotExplicitOptionalBudgets{70432384, 35216192})};
}
/**
 * @brief Project bounded SSD backing through original TP2 source seeds only.
 * @return Four original requests: CUDA/ROCm ordinary and fixed1, with first
 * consumer C0 and second positive. Original mmap windows and host cache policy
 * are retained; declaration does not qualify mmap execution or performance.
 */
inline std::vector<ModelParityCase> qwen4ExpLocalHotSSDCases()
{
    std::vector<ModelParityCase> result;
    for (const auto& request : {
        Qwen4ExpOwnerSubsetModelRequest{Qwen4ExpHardwareTarget::ROCm2, 1,
            AdaptiveImmutableRowBacking::SSD, Qwen4ExpOwnerSubsetCachePressure::Positive,
            ModelParityExpertMovement::Static},
        Qwen4ExpOwnerSubsetModelRequest{Qwen4ExpHardwareTarget::CUDA2, 1,
            AdaptiveImmutableRowBacking::SSD, Qwen4ExpOwnerSubsetCachePressure::Zero,
            ModelParityExpertMovement::Static}})
    {
        for (auto original : qwen4ExpCombinedLocalHotC0CasesFromOriginalSeed(
                 request, {ModelParityMTP::Off, ModelParityMTP::Depth1}))
        {
            result.push_back(qwen4ExpLocalHotBudgetCase(
                std::move(original), Qwen4ExpLocalHotBudgetProfile::CombinedZeroPositive));
        }
    }
    return result;
}
/** @return Two SSD ordinary/reset/reuse candidates with unchanged generation gates. */
inline std::vector<ModelParityCase> qwen4ExpLocalHotSSDOrdinaryCases()
{
    auto cases = qwen4ExpLocalHotSSDCases();
    std::erase_if(cases, [](const auto& cell) { return cell.mtp != ModelParityMTP::Off; });
    return cases;
}
/** @return Two SSD fixed1 candidates with unchanged serial-row byte gates. */
inline std::vector<ModelParityCase> qwen4ExpLocalHotSSDExactCases()
{
    auto cases = qwen4ExpLocalHotSSDCases();
    std::erase_if(cases, [](const auto& cell) { return cell.mtp != ModelParityMTP::Depth1; });
    return cases;
}
/**
 * @return Six focused recurrent requests: C0/equal/unequal, each fixed3 and dynamic.
 * @note Existing initial Off/fixed1 and five-profile comparisons remain unchanged.
 * These are candidates for standalone source admission and actual native exact
 * generation. Declaration grants no capture, execution, numerical or speed proof.
 */
inline std::vector<ModelParityCase> qwen4ExpLocalHotRecurrentCases()
{
    std::vector<ModelParityCase> result;
    for (const auto& original : qwen4ExpCombinedLocalHotC0CasesForMTP(
             {ModelParityMTP::Depth3,ModelParityMTP::DynamicDepth}))
    {
        result.push_back(original);
        result.push_back(qwen4ExpLocalHotBudgetCase(original,Qwen4ExpLocalHotBudgetProfile::CombinedEqual));
        result.push_back(qwen4ExpLocalHotBudgetCase(original,Qwen4ExpLocalHotBudgetProfile::CombinedUnequal));
    }
    return result;
}
/**
 * @return Four original CUDA2 RAM/mask2 requests: C0/equal, each Off/fixed1.
 * @note Only optional cache ceilings change. Original CPU expert capacity,
 *       CUDA continuation/coordinator, R3 sidecar and math remain untouched.
 */
inline std::vector<ModelParityCase> qwen4ExpCUDALocalHotCases()
{
    std::vector<ModelParityCase> result;
    for (const auto& original : qwen4ExpCombinedLocalHotC0CasesFromOriginalSeed(
             {Qwen4ExpHardwareTarget::CUDA2,2,AdaptiveImmutableRowBacking::RAM,
              Qwen4ExpOwnerSubsetCachePressure::Positive,ModelParityExpertMovement::Static},
             {ModelParityMTP::Off,ModelParityMTP::Depth1}))
    {
        result.push_back(original);
        result.push_back(qwen4ExpLocalHotBudgetCase(original,Qwen4ExpLocalHotBudgetProfile::CombinedEqual));
    }
    return result;
}
/** @return Two actual Off generation candidates; exact-MTP remains separate. */
inline std::vector<ModelParityCase> qwen4ExpCUDALocalHotOrdinaryCases()
{
    auto cases=qwen4ExpCUDALocalHotCases();
    std::erase_if(cases,[](const auto& cell){return cell.mtp!=ModelParityMTP::Off;});
    return cases;
}
/** @return Two active fixed1 requests for the existing native exact fixture. */
inline std::vector<ModelParityCase> qwen4ExpCUDALocalHotExactCases()
{
    auto cases=qwen4ExpCUDALocalHotCases();
    std::erase_if(cases,[](const auto& cell){return cell.mtp!=ModelParityMTP::Depth1;});
    return cases;
}
/**
 * @return Four GPU-only original Q6 requests: follower-subset C0/equal, each Off/fixed1.
 * @note RAM source, DynamicOrdinal expert movement, all six original GPU owners,
 *       the R1 predictor and retained3 remain unchanged. Shared storage is on
 *       explicit original follower coordinates 0 and 2, never a CPU expert tier.
 *       Source/native qualification is mandatory before support is claimed.
 */
inline std::vector<ModelParityCase> qwen4ExpAllGPUQ6LocalHotCases()
{
    const auto& originals=qwen4ExpAllGPUQ6MTPNativeExactCases();
    const auto first=std::find_if(originals.begin(),originals.end(),[](const auto& cell){
        return cell.mtp==ModelParityMTP::Depth1;});
    if(first==originals.end())throw std::logic_error("Combined Q6 lost its original fixed1 R3 parent");
    const auto& plan=*first->topology.expert_overlay_plan;
    if(resolveMoEOverlayAuthorityExecutionKind(plan)!=MoEOverlayAuthorityExecutionKind::DeviceResident)
        throw std::logic_error("Combined Q6 requires its original GPU-owned expert authority");
    const auto consumers=qwen4ExpLocalHotOriginalConsumers(*first);
    const auto root=plan.continuation_domain_spec.logical_root_participant;
    if(root<0 || static_cast<std::size_t>(root)>=consumers.size())
        throw std::logic_error("Combined Q6 lost its original continuation root");
    const RoutedExpertDomain* followers=nullptr;
    for(const auto& domain:plan.domains)
    {
        if(std::any_of(domain.participants.begin(),domain.participants.end(),[](const auto& participant){return participant.isCPU();}))
            throw std::logic_error("Combined Q6 cannot acquire a CPU expert tier");
        if(domain.name==plan.continuation_domain)continue;
        if(followers || domain.scope!=ExecutionDomainScope::RANK_LOCAL || domain.owner_rank<0 ||
           !domain.world_ranks.empty() || domain.participants.empty() ||
           std::any_of(domain.participants.begin(),domain.participants.end(),[](const auto& participant){return participant.device_type!=DeviceType::ROCm;}))
            throw std::logic_error("Combined Q6 lost its original rank-local ROCm follower domain");
        followers=&domain;
    }
    constexpr std::uint64_t selectedFollowers=5;
    if(!followers || followers->participants.size()>=std::numeric_limits<std::uint64_t>::digits ||
       (selectedFollowers>>followers->participants.size()))
        throw std::logic_error("Combined Q6 follower subset crossed its original roster");
    auto policy=qwen4ExpGPUAdaptiveStorageAxes().front();
    auto& g=policy.gpu_adaptive;
    g.source_backing=AdaptiveImmutableRowBacking::RAM;
    g.placement={.mode=AdaptiveGPUCachePlacementMode::DeclaredRankLocal,
        .owners={},.coordinator=consumers.at(static_cast<std::size_t>(root))};
    for(std::size_t index=0;index<followers->participants.size();++index)
        if(selectedFollowers&(std::uint64_t{1}<<index))
        {
            const auto& participant=followers->participants[index];
            g.placement.owners.push_back({{followers->owner_rank,participant.device_type,participant.device_ordinal},0,0});
        }
    g.local_hot_cache=AdaptiveGPULocalHotCacheRequest{};
    policy.validate();
    std::vector<ModelParityCase> result;
    for(const auto mode:{ModelParityMTP::Off,ModelParityMTP::Depth1})
    {
        auto cell=*first;
        // This is the existing same-configuration serial control: only request
        // execution changes, while the original R3 allocation remains retained.
        cell.mtp=mode;
        cell.ngram_table_policy=policy;
        cell.topology.test_id+="_FollowerOwnerSubset5Candidate_CombinedLocalHotC0Candidate";
        result.push_back(cell);
        result.push_back(qwen4ExpLocalHotBudgetCase(cell,Qwen4ExpLocalHotBudgetProfile::CombinedEqual));
    }
    return result;
}
/** @return Two original DynamicOrdinal Off requests for actual public generation. */
inline std::vector<ModelParityCase> qwen4ExpAllGPUQ6LocalHotOrdinaryCases()
{
    auto cases=qwen4ExpAllGPUQ6LocalHotCases();
    std::erase_if(cases,[](const auto& cell){return cell.mtp!=ModelParityMTP::Off;});return cases;
}
/** @return Two original fixed1 native exact requests with all movement/reset contracts intact. */
inline std::vector<ModelParityCase> qwen4ExpAllGPUQ6LocalHotExactCases()
{
    auto cases=qwen4ExpAllGPUQ6LocalHotCases();
    std::erase_if(cases,[](const auto& cell){return cell.mtp!=ModelParityMTP::Depth1;});return cases;
}
} // namespace llaminar2::test::parity::qwen4exp
