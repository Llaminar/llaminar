/**
 * @file AutomaticOrchestrationPlanner.cpp
 * @brief Cost selection without first-fit placement or duplicated memory admission.
 *
 * Proposals are admitted in bounded worker-team batches. Pure preparation and
 * quote owners remain worker-local; callbacks and winner selection drain in
 * original order. No Cartesian product of configurations/BOMs is retained. A cost evaluator receives the compiled rank geometry
 * and resolved expert quotas, never the original unadmitted proposal.
 */
#include "planning/AutomaticOrchestrationPlanner.h"
#include "planning/PlanningAdmissionPreparation.h"
#include "planning/PlanningMemoryQuoteReuse.h"
#include "planning/PhysicalMemoryCapacityExhausted.h"
#include "planning/ObservedPeerGrouping.h"
#include "utils/Logger.h"
#include <sstream>
#include <iomanip>
#include "config/OrchestrationConfigDocument.h"
#include <algorithm>
#include <cmath>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <omp.h>
#include <exception>

namespace llaminar2
{
    OrchestrationCostEstimate::OrchestrationCostEstimate(OrchestrationPlanningWorkload workload,
        double prefill_seconds, double decode_seconds_per_token, std::string evidence)
        : workload_(workload), prefill_seconds_(prefill_seconds),
          decode_seconds_per_token_(decode_seconds_per_token), evidence_(std::move(evidence))
    {
        if (!std::isfinite(prefill_seconds_) || prefill_seconds_ <= 0 ||
            !std::isfinite(decode_seconds_per_token_) || decode_seconds_per_token_ <= 0 ||
            !std::isfinite(requestSeconds()) || evidence_.find_first_not_of(" \r\n\t") == std::string::npos)
            throw std::invalid_argument("Automatic candidate cost needs finite positive phase estimates and input evidence");
    }

    double OrchestrationCostEstimate::requestSeconds() const noexcept
    {
        return prefill_seconds_ + decode_seconds_per_token_ * workload_.generationTokens();
    }

    SelectedAutomaticOrchestration::SelectedAutomaticOrchestration(
        AdmittedOrchestrationCandidate candidate, OrchestrationCostEstimate cost,
        OrchestrationSelectionCounts counts)
        : candidate_(std::move(candidate)), cost_(std::move(cost)), counts_(counts) {}

    namespace
    {
        /** @brief A candidate and its authenticated workload cost advance together. */
        struct PricedCandidate
        {
            AdmittedOrchestrationCandidate candidate;
            OrchestrationCostEstimate cost;
        };

        /**
         * @return Whether declared compute includes the preferred backend.
         *
         * Expert-only ranks' placeholder CPU control endpoints are not compute
         * participants. Overlay domains already own that distinction. Ordinary
         * plans expose their exact local TP/PP endpoints and primary device.
         */
        bool usesBackend(const AdmittedOrchestrationCandidate &candidate, DeviceType backend)
        {
            const auto matches = [backend](const auto &address) { return address.device_type == backend; };
            const auto &overlay = candidate.config().moe_routed_expert_plan;
            if (overlay && overlay->usesExpertOverlayAuthority())
            {
                const auto includes = [&](const auto &domains) {
                    return std::any_of(domains.begin(), domains.end(), [&](const auto &domain) {
                        return std::any_of(domain.participants.begin(), domain.participants.end(), matches);
                    });
                };
                return includes(overlay->domains) || includes(overlay->dense_domains);
            }
            return std::any_of(candidate.rankPlans().begin(), candidate.rankPlans().end(), [&](const auto &rank) {
                return matches(rank.primary_device) ||
                    std::any_of(rank.local_tp_devices.begin(), rank.local_tp_devices.end(), matches) ||
                    std::any_of(rank.local_pp_devices.begin(), rank.local_pp_devices.end(), matches);
            });
        }

        /** @return Number of explicit hints satisfied; hints never alter a cost estimate. */
        int hintMatches(const AdmittedOrchestrationCandidate &candidate, const AutomaticOrchestrationOptions &options)
        {
            return (options.prefer_backend && usesBackend(candidate, *options.prefer_backend) ? 1 : 0) +
                (options.prefer_strategy && candidate.strategy() == *options.prefer_strategy ? 1 : 0);
        }

        /** @brief One declared native domain projected to its original local rank members. */
        struct LocalPeerGroup {int rank;std::vector<DeviceId> devices;};
        /** @return Actual domain groups; never merge two PP stages into a fictitious collective. */
        std::vector<LocalPeerGroup> peerGroups(const AdmittedOrchestrationCandidate& candidate)
        {
            const auto& inventory=candidate.membership().inventory();
            std::vector<LocalPeerGroup> result;
            std::set<std::pair<int,std::vector<DeviceId>>> seen;
            const auto append=[&](const auto& addresses,std::optional<int> owner) {
                for(const auto& rank:inventory.ranks) {
                    if(owner && *owner!=rank.rank) continue;
                    for(auto backend:{DeviceType::CUDA,DeviceType::ROCm}) {
                        std::vector<DeviceId> devices;
                        for(const auto& address:addresses) {
                            if(address.device_type!=backend || address.hostname!=rank.hostname) continue;
                            const DeviceId id{backend,address.device_ordinal};
                            const auto found=std::find_if(rank.gpus.begin(),rank.gpus.end(),[&](const auto& gpu){
                                return gpu.type==id.type && gpu.local_device_id==id.ordinal;
                            });
                            if(found!=rank.gpus.end() && rankOwnsGPUDevice(rank,*found)) devices.push_back(id);
                        }
                        std::sort(devices.begin(),devices.end());
                        devices.erase(std::unique(devices.begin(),devices.end()),devices.end());
                        if(devices.size()>1 && seen.emplace(rank.rank,devices).second)
                            result.push_back({rank.rank,std::move(devices)});
                    }
                }
            };
            const auto appendDomain=[&](const ExecutionDomainDefinition& domain) {
                const bool local=domain.scope==ExecutionDomainScope::RANK_LOCAL ||
                    domain.scope==ExecutionDomainScope::SINGLE;
                append(domain.participants,local?domain.owner_rank:std::nullopt);
            };
            const auto& config=candidate.config();
            const auto& overlay=config.moe_routed_expert_plan;
            if(overlay && overlay->usesExpertOverlayAuthority()) {
                for(const auto& domain:overlay->domains) {
                    const auto typed=domain.toExecutionDomainDefinition();
                    appendDomain(typed);
                }
                for(const auto& domain:overlay->dense_domains)
                    appendDomain(domain);
            }
            else if(!config.domain_definitions.empty()) {
                for(const auto& domain:config.domain_definitions) appendDomain(domain.toExecutionDomainDefinition());
            }
            else for(const auto& rank:candidate.rankPlans()) append(rank.local_tp_devices,rank.rank);
            return result;
        }
        /** @return Residual preference over actual immutable links, never an invented latency. */
        ObservedPeerGroupQuality peerQuality(const AdmittedOrchestrationCandidate& candidate)
        {
            ObservedPeerGroupQuality result;
            for(const auto& group:peerGroups(candidate))
                result+=ObservedPeerGrouping(candidate.membership().inventory().ranks.at(group.rank)).quality(group.devices);
            return result;
        }
        /** @return Immutable physical identities for deterministic equal-topology ties. */
        std::vector<std::vector<std::pair<int,std::string>>> peerIdentities(const AdmittedOrchestrationCandidate& candidate)
        {
            std::vector<std::vector<std::pair<int,std::string>>> result;
            for(const auto& group:peerGroups(candidate)) {
                const auto& rank=candidate.membership().inventory().ranks.at(group.rank);
                const ObservedPeerGrouping facts(rank);
                std::vector<std::pair<int,std::string>> identities;
                for(const auto id:group.devices) identities.emplace_back(rank.node_id,facts.device(id).uuid);
                std::sort(identities.begin(),identities.end());result.push_back(std::move(identities));
            }
            return result;
        }
        /**
         * @brief Describe one admitted proposal and its exact cost on a single diagnostic line.
         *
         * Groups retain semantic visitor order and original discovery ownership;
         * newline escaping keeps evaluator evidence from impersonating another
         * record. Called only inside the DEBUG logging guard, never for ranking.
         */
        std::string candidateCostDiagnostic(const AdmittedOrchestrationCandidate &candidate,
            const OrchestrationCostEstimate &cost, std::size_t ordinal)
        {
            std::ostringstream text;
            text << std::setprecision(17) << "AutomaticCandidateCost candidate=" << ordinal
                << " strategy=" << orchestrationStrategyName(candidate.strategy()) << " groups=[";
            bool first_group = true;
            for (const auto &group : peerGroups(candidate))
            {
                if (!first_group) text << ';';
                first_group = false;
                text << "discovery_rank=" << candidate.membership().discoveryRanks().at(group.rank) << ":devices=[";
                for (std::size_t i = 0; i < group.devices.size(); ++i)
                {
                    if (i) text << ',';
                    text << deviceTypeToString(group.devices[i].type) << ':' << group.devices[i].ordinal;
                }
                text << ']';
            }
            std::string evidence;
            for (const char c : cost.evidence())
                if (c == '\\') evidence += "\\\\";
                else if (c == '\n') evidence += "\\n";
                else if (c == '\r') evidence += "\\r";
                else evidence += c;
            text << "] prompt_tokens=" << cost.workload().promptTokens()
                << " generation_tokens=" << cost.workload().generationTokens()
                << " request_seconds=" << cost.requestSeconds()
                << " prefill_seconds=" << cost.prefillSeconds()
                << " decode_seconds_per_token=" << cost.decodeSecondsPerToken()
                << " evidence=" << std::quoted(evidence);
            return text.str();
        }

        /** @brief Publish original chosen identities and pair facts through normal planning diagnostics. */
        void reportPeerGroups(const AdmittedOrchestrationCandidate& candidate)
        {
            for(const auto& group:peerGroups(candidate)) {
                const auto& rank=candidate.membership().inventory().ranks.at(group.rank);
                const ObservedPeerGrouping facts(rank);
                std::ostringstream text;
                text<<"Automatic native group original rank="<<candidate.membership().discoveryRanks().at(group.rank)
                    <<" NUMA="<<rank.cpu.numa_node<<" endpoints=";
                for(const auto id:group.devices) text<<deviceTypeToString(id.type)<<":"<<id.ordinal
                    <<"/"<<facts.device(id).uuid<<"/"<<facts.device(id).pci_bus_address
                    <<"/"<<facts.device(id).native_isa<<" ";
                text<<"pairs=";
                for(std::size_t i=0;i<group.devices.size();++i) for(std::size_t j=i+1;j<group.devices.size();++j) {
                    text<<group.devices[i].ordinal<<"->"<<group.devices[j].ordinal
                        <<":mutual="<<facts.mutuallyAccessible(group.devices[i],group.devices[j]);
                    if(const auto hops=facts.distance(group.devices[i],group.devices[j])) text<<",PCIhops="<<*hops;
                    else text<<",PCIhops=unobserved";
                    text<<" ";
                }
                LOG_INFO(text.str());
            }
        }

        /**
         * @return Number of distinct compiled compute endpoints, not MPI processes.
         *
         * A rank-local TP group has several devices but only one MPI rank. A
         * device may also serve several logical overlay roles: count that
         * endpoint once, without counting CPU control/staging allocations.
         * This is a tie-break only; it must never become a fabricated link cost.
         */
        size_t computeEndpoints(const AdmittedOrchestrationCandidate &candidate)
        {
            std::set<std::tuple<int, DeviceType, int>> endpoints;
            for (const auto &device : candidate.devicePlans())
                endpoints.emplace(device.world_rank, device.device.type, device.device.ordinal);
            if (endpoints.empty()) throw std::logic_error("Admitted candidate has no compute endpoints");
            return endpoints.size();
        }
    }

    SelectedAutomaticOrchestration AutomaticOrchestrationPlanner::select(const OrchestrationConfig &request,
        const PlanningModelSource &source, const ClusterInventory &inventory,
        const OrchestrationCandidateMemoryPolicy &memory, const OrchestrationPlanningWorkload &workload,
        const Evaluate &evaluate, const CapacityRejected &rejected)
    {
        const auto intent = resolveOrchestrationIntent(request);
        const auto *automatic = std::get_if<AutomaticOrchestrationRequest>(&intent);
        if (!automatic || !evaluate)
            throw std::invalid_argument("Automatic selection requires automatic intent and a cost evaluator");
        if (request.model_path != source.path() || request.shared_predictor != source.predictorSource())
            throw std::invalid_argument("Automatic selection model does not name its retained metadata source");
        workload.requireFitsContext(request.max_seq_len);
        if (automatic->options().workload && *automatic->options().workload != workload)
            throw std::invalid_argument("Automatic planning objective conflicts with the configured workload hint");

        std::optional<PricedCandidate> best;
        OrchestrationSelectionCounts counts;
        std::string last_capacity_error;
        if(omp_in_parallel())
            throw std::invalid_argument("Automatic planning cannot reenter an active OpenMP source/quote owner");
        const int workers=omp_get_max_threads();
        struct AdmissionSlot {
            OrchestrationStrategy strategy;
            std::optional<AutomaticOrchestrationCandidate> proposal;
            std::optional<AdmittedOrchestrationCandidate> admitted;
            std::exception_ptr error;
        };
        std::vector<AdmissionSlot> batch;
        std::vector<PlanningAdmissionPreparation*> preparations;
        std::vector<std::exception_ptr> initialization_errors;
        std::exception_ptr failure;
        const auto consider=[&](AdmissionSlot &slot) {
            if(slot.error) {
                try {std::rethrow_exception(slot.error);}
                catch(const PhysicalMemoryCapacityExhausted &error) {
                    ++counts.capacity_rejected;last_capacity_error=error.what();
                    if(rejected)rejected(slot.strategy,last_capacity_error);
                    return;
                }
            }
            auto &candidate=slot.admitted;
            // Evaluation and callbacks stay on the original caller, outside
            // admission's capacity catch, with exactly serial source ordering.
            auto cost = evaluate(*candidate, workload);
            if (cost.workload() != workload)
                throw std::invalid_argument("Automatic candidate cost describes a different workload");
            ++counts.evaluated;
            LOG_DEBUG(candidateCostDiagnostic(*candidate, cost, counts.evaluated));
            bool prefer = !best || cost.requestSeconds() < best->cost.requestSeconds();
            if (best && cost.requestSeconds() == best->cost.requestSeconds())
            {
                const int hints = hintMatches(*candidate, automatic->options());
                const int best_hints = hintMatches(best->candidate, automatic->options());
                // Explicit user preferences still break equal-cost ties first.
                // Otherwise choose the narrower execution footprint. In
                // particular, zero-link single-device execution wins an exact
                // tie against local TP, even though both use one MPI process.
                const auto endpoints = computeEndpoints(*candidate);
                const auto best_endpoints = computeEndpoints(best->candidate);
                // Containing-group communication proxies can tie across very
                // different physical partitions. Prefer observed mutual cliques
                // and shorter original PCI paths only after admitted measured
                // cost, explicit hints and footprint tie. Never discard a
                // valid no-P2P native candidate or override a better cost.
                if (hints != best_hints) prefer = hints > best_hints;
                else if (endpoints != best_endpoints) prefer = endpoints < best_endpoints;
                else {
                    const auto peers=peerQuality(*candidate);
                    const auto best_peers=peerQuality(best->candidate);
                    const auto identities=peerIdentities(*candidate);
                    const auto best_identities=peerIdentities(best->candidate);
                    if (peers != best_peers) prefer = peers > best_peers;
                    else if (identities != best_identities) prefer = identities < best_identities;
                    else prefer = serializeOrchestrationConfig(candidate->config()) <
                        serializeOrchestrationConfig(best->candidate.config());
                }
            }
            if (prefer)
            {
                best = PricedCandidate{std::move(*candidate), std::move(cost)};
            }
        };
        const auto flush=[&] {
            if(batch.empty())return;
            auto **local_preparations=preparations.data();
            const auto *original_source=&source;
            const auto *original_memory=&memory;
#pragma omp taskgroup
            {
                for(size_t index=0;index<batch.size();++index) {
                    auto *slot=&batch[index];
#pragma omp task firstprivate(slot,local_preparations,original_source,original_memory)
                    {
                        try {
                            slot->admitted.emplace(AdmittedOrchestrationCandidate::admit(std::move(*slot->proposal),
                                *original_source,*original_memory,local_preparations[omp_get_thread_num()]));
                        } catch(...) {slot->error=std::current_exception();}
                        slot->proposal.reset();
                    }
                }
            }
            try {for(auto &slot:batch)consider(slot);}
            catch(...) {batch.clear();throw;}
            batch.clear();
        };
#pragma omp parallel num_threads(workers)
        {
            // The configured budget is an upper bound. Dynamic/thread-limited
            // runtimes may legitimately form a smaller actual team.
#pragma omp master
            {
                try {
                    const auto width=static_cast<size_t>(omp_get_num_threads());
                    batch.reserve(width);preparations.resize(width,nullptr);initialization_errors.resize(width);
                } catch(...) {failure=std::current_exception();}
            }
#pragma omp barrier
            // These lexical owners are constructed and retired by the same
            // persistent worker. TLS quotes never migrate across OMP teams.
            const int worker=omp_get_thread_num();
            std::unique_ptr<PlanningAdmissionPreparation> preparation;
            std::unique_ptr<PlanningMemoryQuoteReuse> quotes;
            try { if(!failure) {
                preparation=std::make_unique<PlanningAdmissionPreparation>(source.loader(),source.metadata(),memory.weight_load);
                quotes=std::make_unique<PlanningMemoryQuoteReuse>(source.metadata().tensorIndex());
                preparations[static_cast<size_t>(worker)]=preparation.get();
            }} catch(...) {initialization_errors[static_cast<size_t>(worker)]=std::current_exception();}
#pragma omp barrier
#pragma omp master
            {
                try { if(!failure) {
                    for(const auto &error:initialization_errors)if(error)std::rethrow_exception(error);
                    std::exception_ptr enumeration_failure;
                    try {
                        visitAutomaticOrchestrationCandidates(request,source.metadata(),inventory,[&](auto proposal) {
                            ++counts.proposed;
                            const auto strategy=proposal.strategy;
                            batch.push_back({strategy,std::move(proposal),{}, {}});
                            if(batch.size()==static_cast<size_t>(omp_get_num_threads()))flush();
                        });
                    } catch(...) {enumeration_failure=std::current_exception();}
                    // If enumeration fails after emitting a partial batch, the
                    // preceding callbacks still happen before that later error.
                    flush();
                    if(enumeration_failure)std::rethrow_exception(enumeration_failure);
                }} catch(...) {failure=std::current_exception();}
            }
#pragma omp barrier
        }
        if(failure)std::rethrow_exception(failure);
        if (counts.proposed == 0)
            throw std::invalid_argument("No observed topology satisfies the automatic planning constraints");
        if (!best)
            throw PhysicalMemoryCapacityExhausted("No automatic candidate fits the complete physical admission: " + last_capacity_error);
        reportPeerGroups(best->candidate);
        return {std::move(best->candidate), std::move(best->cost), counts};
    }
}
