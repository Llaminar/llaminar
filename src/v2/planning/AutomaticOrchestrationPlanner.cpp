/**
 * @file AutomaticOrchestrationPlanner.cpp
 * @brief Cost selection without first-fit placement or duplicated memory admission.
 *
 * Proposals are admitted and discarded one at a time. Only the current winner
 * stays alive, so a large topology search does not retain a Cartesian product
 * of configurations/BOMs. A cost evaluator receives the compiled rank geometry
 * and resolved expert quotas, never the original unadmitted proposal.
 */
#include "planning/AutomaticOrchestrationPlanner.h"
#include "planning/PhysicalMemoryCapacityExhausted.h"
#include "config/OrchestrationConfigDocument.h"
#include <algorithm>
#include <cmath>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>

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
        if (request.model_path != source.path())
            throw std::invalid_argument("Automatic selection model does not name its retained metadata source");
        workload.requireFitsContext(request.max_seq_len);
        if (automatic->options().workload && *automatic->options().workload != workload)
            throw std::invalid_argument("Automatic planning objective conflicts with the configured workload hint");

        std::optional<PricedCandidate> best;
        OrchestrationSelectionCounts counts;
        std::string last_capacity_error;
        visitAutomaticOrchestrationCandidates(request, source.metadata(), inventory, [&](auto proposal) {
            ++counts.proposed;
            const auto strategy = proposal.strategy;
            std::optional<AdmittedOrchestrationCandidate> candidate;
            try
            {
                candidate = AdmittedOrchestrationCandidate::admit(std::move(proposal), source, memory);
            }
            catch (const PhysicalMemoryCapacityExhausted &error)
            {
                ++counts.capacity_rejected;
                last_capacity_error = error.what();
                if (rejected) rejected(strategy, last_capacity_error);
                return;
            }
            // Evaluator errors are outside the admission catch. A missing
            // measurement, including one mislabeled as exhaustion, is fatal;
            // it cannot select a different execution path without explanation.
            auto cost = evaluate(*candidate, workload);
            if (cost.workload() != workload)
                throw std::invalid_argument("Automatic candidate cost describes a different workload");
            ++counts.evaluated;
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
                prefer = hints > best_hints || (hints == best_hints &&
                    (endpoints < best_endpoints || (endpoints == best_endpoints &&
                        serializeOrchestrationConfig(candidate->config()) < serializeOrchestrationConfig(best->candidate.config()))));
            }
            if (prefer)
            {
                best = PricedCandidate{std::move(*candidate), std::move(cost)};
            }
        });
        if (counts.proposed == 0)
            throw std::invalid_argument("No observed topology satisfies the automatic planning constraints");
        if (!best)
            throw PhysicalMemoryCapacityExhausted("No automatic candidate fits the complete physical admission: " + last_capacity_error);
        return {std::move(best->candidate), std::move(best->cost), counts};
    }
}
