/**
 * @file NodeExpertOverlayParityEconomy.cpp
 * @brief Economy implementation of the shared node-overlay parity fixture.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 */
#include "NodeExpertOverlayParityFixture.h"

namespace llaminar2::test::parity::qwen35moe::node_overlay
{

    /** @return Monotonic elapsed nanoseconds, clamped away from zero. */
    auto Qwen35MoENodeExpertOverlayParityTest::elapsedNanoseconds(
        std::chrono::steady_clock::time_point start) noexcept -> std::uint64_t
    {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count();
        return static_cast<std::uint64_t>(std::max<std::int64_t>(elapsed, 1));
    }

    /**
     * @brief Snapshot cumulative timers needed to attribute an A/B cohort.
     *
     * Snapshot construction occurs outside every measured interval. Exact tags
     * retain layer, sparse endpoint, replay segment, and route identity so the
     * resulting CSV can locate a regression without adding clocks to the live
     * graph.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::convergenceTimerSnapshot() -> ConvergenceTimerSnapshot
    {
        ConvergenceTimerSnapshot result;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"forward_graph", "moe_overlay"}))
        {
            if (record.kind != PerfStatRecord::Kind::Timer ||
                (record.domain == "moe_overlay" &&
                 record.name != "compute"))
                continue;
            ConvergenceTimerRecord value{
                .domain = record.domain,
                .name = record.name,
                .phase = record.phase,
                .device = record.device,
                .tags = record.tags,
                .count = record.count,
                .total_ns = record.total_ns,
            };
            result.emplace(value, value);
        }
        return result;
    }

    /**
     * @brief Subtract two cumulative timer views without losing exact tags.
     *
     * @param before Cumulative snapshot immediately before ordinary requests.
     * @param after Cumulative snapshot immediately after ordinary requests.
     * @return Interval-local records; zero-occurrence keys are omitted.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::convergenceTimerDelta(
        const ConvergenceTimerSnapshot &before,
        const ConvergenceTimerSnapshot &after) -> ConvergenceTimerSnapshot
    {
        ConvergenceTimerSnapshot result;
        for (const auto &[key, terminal] : after)
        {
            const auto start = before.find(key);
            const std::uint64_t begin_count =
                start == before.end() ? 0u : start->second.count;
            const std::uint64_t begin_ns =
                start == before.end() ? 0u : start->second.total_ns;
            if (terminal.count < begin_count || terminal.total_ns < begin_ns)
            {
                throw std::logic_error(
                    "PerfStats timer regressed inside an immutable convergence request");
            }
            ConvergenceTimerRecord interval = key;
            interval.count = terminal.count - begin_count;
            interval.total_ns = terminal.total_ns - begin_ns;
            if (interval.count != 0u)
                result.emplace(interval, interval);
        }
        return result;
    }

    /**
     * @brief Write request-matched initial/converged timer evidence.
     *
     * Every CSV row belongs to one of the exact three prompt identities retained
     * on both sides of the economy assertion. Movement training owns a disjoint
     * cache-identity interval, so no overlap filtering is necessary.
     *
     * @param baseline Selected initial-epoch samples in comparison order.
     * @param converged Selected converged-epoch samples in comparison order.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::writeConvergenceTimerSamples(
        const std::vector<const ResidencyConvergenceTimings::RequestSample *>
            &baseline,
        const std::vector<ResidencyConvergenceTimings::RequestSample>
            &converged) -> void
    {
        if (baseline.size() != converged.size() ||
            baseline.size() != static_cast<std::size_t>(
                                   kConvergenceTimingMeasuredRequests))
        {
            throw std::logic_error(
                "ExpertOverlay convergence timer evidence is not request-matched");
        }
        const auto path =
            ensureResultsDir() / "expert_overlay_convergence_timers.csv";
        std::ofstream csv(path, std::ios::trunc);
        if (!csv.is_open())
        {
            throw std::runtime_error(
                "Cannot create ExpertOverlay convergence timer CSV at " +
                path.string());
        }
        csv << "cohort,prompt_identity,domain,name,phase,device,tags,count,total_ns,average_ns\n";
        const auto write_sample = [&csv](
                                      std::string_view cohort,
                                      const ResidencyConvergenceTimings::RequestSample
                                          &sample)
        {
            for (const auto &[key, interval] : sample.timer_deltas)
            {
                std::ostringstream tags;
                bool first = true;
                for (const auto &[name, value] : key.tags)
                {
                    if (!first)
                        tags << ';';
                    first = false;
                    tags << name << '=' << value;
                }
                csv << cohort << ',' << sample.prompt_identity << ','
                    << csvEscape(key.domain) << ','
                    << csvEscape(key.name) << ','
                    << csvEscape(key.phase) << ','
                    << csvEscape(key.device) << ','
                    << csvEscape(tags.str()) << ','
                    << interval.count << ',' << interval.total_ns << ','
                    << interval.total_ns / interval.count << '\n';
            }
        };
        for (const auto *sample : baseline)
        {
            if (!sample)
                throw std::logic_error(
                    "ExpertOverlay baseline timer sample is null");
            write_sample("initial_epoch", *sample);
        }
        for (const auto &sample : converged)
            write_sample("converged_epoch", sample);
        csv.flush();
        if (!csv.good())
        {
            throw std::runtime_error(
                "Failed to write ExpertOverlay convergence timer CSV at " +
                path.string());
        }
    }

    /**
     * @brief Build one cache-distinct valid-token service workload.
     *
     * Certification needs broad natural routing so every real sparse
     * participant and runtime phase contributes a measured service profile.
     * The deterministic SplitMix corpus changes only valid embedding rows and
     * enters the graph through the ordinary serving API. Its routed rows are
     * calibration evidence, not optimization demand: the production admission
     * state quarantines the complete corpus and discards its histogram bank
     * before the next public request boundary.
     *
     * @param request_index Stable request ordinal within the service corpus.
     * @return Prompt-sized deterministic token vector unique to this ordinal.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::makeEconomyWorkloadPrompt(int request_index) const -> std::vector<int32_t>
    {
        const int vocabulary_size = orch_runner_->vocabSize();
        if (config_.token_ids.empty() || request_index < 0 ||
            vocabulary_size <= 4'096)
        {
            throw std::logic_error(
                "Economy workload requires authenticated prompt geometry, a non-negative identity, and a valid vocabulary");
        }

        std::uint64_t state =
            0x9e3779b97f4a7c15ULL ^
            (static_cast<std::uint64_t>(request_index + 1) *
             0xbf58476d1ce4e5b9ULL);
        const auto usable_vocabulary =
            static_cast<std::uint64_t>(vocabulary_size - 2'048);
        std::vector<int32_t> varied(config_.token_ids.size(), 0);
        for (auto &token : varied)
        {
            state += 0x9e3779b97f4a7c15ULL;
            std::uint64_t mixed = state;
            mixed = (mixed ^ (mixed >> 30u)) *
                    0xbf58476d1ce4e5b9ULL;
            mixed = (mixed ^ (mixed >> 27u)) *
                    0x94d049bb133111ebULL;
            mixed ^= mixed >> 31u;
            token = static_cast<int32_t>(
                256u + mixed % usable_vocabulary);
        }
        varied.front() = economyPromptLeadingToken(
            EconomyPromptNamespace::ServiceCertification,
            request_index);
        return varied;
    }

    /**
     * @brief Map one workload identity to a collision-free prefix-cache block.
     *
     * The usable embedding interval is split between service certification and
     * stationary convergence. Within either typed half the mapping is
     * injective and skips the authenticated Hugging Face prompt's first token,
     * so neither calibration nor timing can seed the later parity prefix.
     *
     * @param prompt_namespace Typed economy-traffic namespace.
     * @param request_index Non-negative identity within that namespace.
     * @return Valid vocabulary row reserved for this exact request identity.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::economyPromptLeadingToken(
        EconomyPromptNamespace prompt_namespace,
        int request_index) const -> int32_t
    {
        const int vocabulary_size = orch_runner_->vocabSize();
        if (config_.token_ids.empty() || request_index < 0 ||
            vocabulary_size <= 4'096)
        {
            throw std::logic_error(
                "Economy prefix identity requires authenticated prompt geometry, a non-negative identity, and a valid vocabulary");
        }

        constexpr int kFirstOrdinaryEmbedding = 256;
        const int last_embedding_exclusive = vocabulary_size - 2'048;
        const int midpoint =
            kFirstOrdinaryEmbedding +
            (last_embedding_exclusive - kFirstOrdinaryEmbedding) / 2;
        const int interval_begin =
            prompt_namespace ==
                    EconomyPromptNamespace::ServiceCertification
                ? kFirstOrdinaryEmbedding
                : midpoint;
        const int interval_end =
            prompt_namespace ==
                    EconomyPromptNamespace::ServiceCertification
                ? midpoint
                : last_embedding_exclusive;
        const int reference_first_token = config_.token_ids.front();
        const bool excludes_reference_token =
            reference_first_token >= interval_begin &&
            reference_first_token < interval_end;
        const int available_identities =
            interval_end - interval_begin -
            (excludes_reference_token ? 1 : 0);
        if (request_index >= available_identities)
        {
            throw std::out_of_range(
                "Economy prefix identity exhausted its typed vocabulary namespace");
        }

        int token = interval_begin + request_index;
        if (excludes_reference_token && token >= reference_first_token)
            ++token;
        return static_cast<int32_t>(token);
    }

    /**
     * @brief Preserve the authenticated workload while changing its cache key.
     *
     * Timing requests use one cache-distinct leading token followed by repeated
     * authenticated rows, preserving the exact matched A/B geometry. A
     * movement-only request instead executes the leading causal rows of the
     * Hugging Face prompt itself. Any expert promoted from that admitted demand
     * is therefore exercised again by the later fixed parity prefill. If a
     * model's initial histogram window exceeds the reference prompt, the full
     * prompt is used rather than inventing unauthenticated suffix rows.
     *
     * @param request_index Stable request ordinal within the measured corpus.
     * @param role Typed economy phase that owns this request geometry.
     * @return Typed timing corpus or an exact causal prefix of the HF corpus.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::makeReferenceShapedEconomyPrompt(
        int request_index,
        ReferenceEconomyPromptRole role) const -> std::vector<int32_t>
    {
        const int vocabulary_size = orch_runner_->vocabSize();
        if (vocabulary_size <= 4'096 || request_index < 0 ||
            config_.token_ids.empty())
        {
            throw std::logic_error(
                "Reference-shaped economy prompt requires authenticated tokens, a non-negative identity, and a valid vocabulary");
        }
        const int selected_rows =
            role == ReferenceEconomyPromptRole::TimingCohort
                ? static_cast<int>(kConvergenceTimingPromptRows)
                : std::min(
                      activeModelParityCaseOrThrow()
                          .dynamic_rebalance.window_size,
                      static_cast<int>(config_.token_ids.size()));
        if (selected_rows <= 0)
        {
            throw std::logic_error(
                "Reference-shaped economy prompt requires a positive typed histogram width");
        }
        const std::size_t stationary_rows =
            static_cast<std::size_t>(selected_rows);
        if (role == ReferenceEconomyPromptRole::MovementProof)
        {
            return std::vector<int32_t>(
                config_.token_ids.begin(),
                config_.token_ids.begin() +
                    static_cast<std::ptrdiff_t>(stationary_rows));
        }
        std::vector<int32_t> prompt;
        prompt.reserve(stationary_rows);
        prompt.push_back(economyPromptLeadingToken(
            EconomyPromptNamespace::StationaryConvergence,
            request_index));
        for (std::size_t row = 1u;
             row < stationary_rows;
             ++row)
        {
            prompt.push_back(config_.token_ids.at(
                (row - 1u) % config_.token_ids.size()));
        }
        return prompt;
    }

    /**
     * @brief Build the exact stationary prefill that closes a demand bank.
     *
     * The authority supplies the remaining logical routed-row count. This
     * helper repeats authenticated model tokens for the requested causal
     * length. The caller purges the reusable cache through the public API;
     * changing a leading token would change every later routed expert and
     * prevent the short-window controller from converging on one workload.
     *
     * @param routed_rows Exact positive active-bank headroom to consume.
     * @return One valid model prompt with exactly @p routed_rows rows.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::makeDemandWindowClosurePrompt(
        std::uint64_t routed_rows) const -> std::vector<int32_t>
    {
        const auto &test_case = activeModelParityCaseOrThrow();
        if (config_.token_ids.empty() ||
            routed_rows == 0u ||
            routed_rows > static_cast<std::uint64_t>(
                              test_case.model.max_seq_len) ||
            routed_rows > static_cast<std::uint64_t>(
                              std::numeric_limits<std::size_t>::max()))
        {
            throw std::logic_error(
                "Demand-window closure requires a positive in-context production prompt");
        }

        return stationaryDemandWindowPrompt(
            config_.token_ids, static_cast<std::size_t>(routed_rows));
    }

    /** @return Median of a non-empty timing corpus without changing it. */
    auto Qwen35MoENodeExpertOverlayParityTest::medianNanoseconds(
        const std::vector<std::uint64_t> &samples) -> std::uint64_t
    {
        if (samples.empty())
            throw std::invalid_argument("Cannot take the median of no inference samples");
        auto ordered = samples;
        const std::size_t midpoint = ordered.size() / 2u;
        std::nth_element(
            ordered.begin(),
            ordered.begin() + static_cast<std::ptrdiff_t>(midpoint),
            ordered.end());
        if ((ordered.size() & 1u) != 0u)
            return ordered[midpoint];
        const auto lower = *std::max_element(
            ordered.begin(),
            ordered.begin() + static_cast<std::ptrdiff_t>(midpoint));
        return lower + (ordered[midpoint] - lower) / 2u;
    }

    /** @return Whether the central matrix assigned this cell the speed witness. */
    auto Qwen35MoENodeExpertOverlayParityTest::requiresObservedConvergenceSpeedup() const noexcept -> bool
    {
        const auto *test_case = activeModelParityCase();
        return test_case &&
               test_case->requiresObservedConvergenceSpeedup();
    }

    /**
     * @brief Select the one retained graph-family lifecycle this cell needs.
     *
     * A numerical/movement cell consumes diagnostic checkpoints throughout
     * its request, so preparing an additional lean family cannot contribute
     * evidence.  Only a centrally designated throughput witness must measure
     * ordinary inference without diagnostic D2D publication before switching
     * to the already-prepared diagnostic family for mathematical parity.
     * Keeping this decision beside the typed evidence role prevents every
     * Dynamic matrix cell from paying a second native graph capture merely
     * because it permits expert movement.
     *
     * @return Immediate diagnostics for ordinary cells, or a prepared lean to
     *         diagnostic transition for an observed-throughput witness.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::paritySnapshotSetupMode() const noexcept -> ParitySnapshotSetupMode
    {
        return requiresObservedConvergenceSpeedup()
                   ? ParitySnapshotSetupMode::PreparedInactive
                   : ParitySnapshotSetupMode::Enabled;
    }

    /**
     * @return Evidence-role and topology-specific convergence objective.
     *
     * Every movement-only cell needs a durable publication proving its
     * applicable axes, irrespective of model size. Only a centrally designated
     * speed witness requires the longer convergence workload before timing.
     * Consume the definition-owned role mapping here so a fixture cannot
     * silently impose a second model-specific evidence policy.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::dynamicResidencyConvergenceTarget() const -> DynamicResidencyConvergenceTarget
    {
        return {
            .minimum_published_waves = qwen35MoEMinimumMovementPublications(
                activeModelParityCaseOrThrow().dynamic_evidence),
            .axis_contract =
                dynamicMovementAxisContract(resolvedOverlayPlan()),
        };
    }

    /**
     * @brief Classify the complete topology plus mathematical-workload target.
     *
     * A broad service-certificate request may validly move experts that the
     * fixed parity prompt never selects. Such a wave remains in the movement
     * ledger and economy CSV, but it cannot close the numerical proof. The
     * exact reference prefill must author at least one later promotion so the
     * captured parity graph can execute that destination and compare its
     * canonical per-route contribution.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::classifyDynamicResidencyProofConvergence(
        const DynamicResidencyConvergenceTarget &target,
        const DynamicResidencyConvergenceOrigin &origin,
        const MoEOptimizationStatus &status,
        const MoEOptimizationMovementLedger &ledger) const noexcept -> DynamicResidencyConvergenceState
    {
        const auto topology = classifyDynamicResidencyConvergence(
            target, origin, status, ledger);
        const auto promotion =
            classifyAuthenticatedPromotionConvergence(
                target,
                origin,
                ledger,
                authenticated_movement_routes_);
        return requireAuthenticatedPromotion(topology, promotion);
    }

    /**
     * @return Passive status projected by the sole production authority.
     * @throws std::logic_error when the runner is absent or its lifecycle failed.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::optimizationStatus() const -> MoEOptimizationStatus
    {
        if (!orch_runner_)
            throw std::logic_error(
                "ExpertOverlay optimization status requires a live runner");
        auto status = orch_runner_->moeOptimizationStatus();
        if (status.failed())
        {
            throw std::logic_error(
                status.diagnostic.empty()
                    ? "ExpertOverlay optimization lifecycle failed"
                    : status.diagnostic);
        }
        return status;
    }

    /** @return Exact durable movement-wave count from the production owner. */
    auto Qwen35MoENodeExpertOverlayParityTest::localCommittedWaveCount() const -> std::uint64_t
    {
        return optimizationStatus().published_movement_waves;
    }

    /**
     * @brief Measure one exact production workload inside one residency epoch.
     *
     * Both sides use identical prompt IDs, boundary calls, decode budgets, and
     * greedy token trajectories. This convergence cell selects the production
     * invalidate-on-rebalance prefix policy, so publication advances the
     * fingerprint and the post-movement replay cannot restore an initial-epoch
     * entry. Both sides therefore execute full model compute. Ordinary
     * maintenance notifications remain enabled. Any publication during the
     * cohort is a hard protocol failure instead of a sample that can be hidden
     * by median selection.
     *
     * @param cohort Initial adversarial epoch or post-movement epoch.
     * @return True only after the complete workload ran in one exact epoch.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::collectInferenceTimings(ResidencyTimingCohort cohort) -> bool
    {
        if (!requiresObservedConvergenceSpeedup())
            return true;

        constexpr std::size_t kRequiredDecodeSamples =
            static_cast<std::size_t>(
                kConvergenceTimingMeasuredRequests) *
            kConvergenceTimingDecodeForwardsPerRequest;
        const bool initial =
            cohort == ResidencyTimingCohort::InitialEpoch;
        const DynamicResidencyProofPhase required_phase =
            initial
                ? DynamicResidencyProofPhase::EconomyCertified
                : DynamicResidencyProofPhase::
                      MovementBoundarySettled;
        if (dynamic_residency_proof_lifecycle_.phase() != required_phase)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Timing cohort crossed an invalid typed residency lifecycle transition: cohort="
                << (initial ? "initial" : "converged")
                << " required_phase="
                << static_cast<int>(required_phase)
                << " actual_phase="
                << static_cast<int>(
                       dynamic_residency_proof_lifecycle_.phase()));
            return false;
        }
        const std::uint64_t cohort_wave = localCommittedWaveCount();
        /* The typed phase above already proves the complete model-specific
         * wave and axis objective. This boundary checks only that the cohort
         * pins an initial or post-publication epoch; it must not reinterpret a
         * cycle count, histogram-window count, or another model's wave target. */
        if ((initial && cohort_wave != 0u) ||
            (!initial && cohort_wave == 0u))
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Timing cohort began in the wrong residency epoch: cohort="
                << (initial ? "initial" : "converged")
                << " committed_waves=" << cohort_wave);
            return false;
        }

        if (initial &&
            (!convergence_timings_.baseline_candidates.empty() ||
             !convergence_timings_.baseline_prefill_ns.empty()))
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Initial timing candidates were already populated");
            return false;
        }
        if (!initial &&
            (!convergence_timings_.converged_prefill_ns.empty() ||
             !convergence_timings_.converged_decode_ns.empty() ||
             !convergence_timings_.converged_samples.empty()))
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Converged timing samples were already populated");
            return false;
        }

        /*
         * Movement-driving traffic can finish immediately after publishing the
         * final placement fingerprint and legitimately archive one of these
         * same prompts under that fingerprint. Retire the reusable archive once
         * at the unmeasured cohort boundary. Every first prefill below must then
         * execute full production compute, while the repeated prefill inside
         * each request still proves ordinary RAM/disk prefix restore.
         */
        activeClearCache();
        if (!orch_runner_->purgePrefixCache())
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Controlled timing cohort could not retire reusable prefix state: "
                << orch_runner_->lastError());
            return false;
        }

        /*
         * Measurement identities are never used to drive movement. The
         * converged cohort therefore replays exactly the initial identities
         * after the typed quiescent boundary, with no publication-overlap
         * reserve, identity omission, or inferred race state.
         */
        const int requested_identities =
            kConvergenceTimingCorpusRequests;
        std::vector<int> prompt_identities;
        prompt_identities.reserve(
            static_cast<std::size_t>(requested_identities));
        for (int identity = 0; identity < requested_identities; ++identity)
        {
            prompt_identities.push_back(identity);
        }

        std::vector<const ResidencyConvergenceTimings::RequestSample *>
            selected_baseline_samples;
        if (!initial)
        {
            /* Materialize only the three initial candidates paired with the
             * selected converged identities. */
            convergence_timings_.baseline_prefill_ns.clear();
            convergence_timings_.baseline_decode_ns.clear();
            convergence_timings_.baseline_prefill_epochs.clear();
            convergence_timings_.baseline_decode_epochs.clear();
            convergence_timings_.baseline_decode_input_tokens.clear();
            selected_baseline_samples.reserve(
                kConvergenceTimingMeasuredRequests);
            for (int ordinal = kConvergenceTimingWarmupRequests;
                 ordinal < requested_identities;
                 ++ordinal)
            {
                const int identity = prompt_identities.at(
                    static_cast<std::size_t>(ordinal));
                const auto candidate = std::find_if(
                    convergence_timings_.baseline_candidates.begin(),
                    convergence_timings_.baseline_candidates.end(),
                    [identity](const auto &sample)
                    { return sample.prompt_identity == identity; });
                if (candidate ==
                    convergence_timings_.baseline_candidates.end())
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] No initial-epoch timing candidate matches converged prompt identity "
                        << identity);
                    return false;
                }
                convergence_timings_.baseline_prefill_ns.push_back(
                    candidate->prefill_ns);
                convergence_timings_.baseline_prefill_epochs.push_back(
                    candidate->prefill_epoch);
                convergence_timings_.baseline_decode_ns.insert(
                    convergence_timings_.baseline_decode_ns.end(),
                    candidate->decode_ns.begin(),
                    candidate->decode_ns.end());
                convergence_timings_.baseline_decode_epochs.insert(
                    convergence_timings_.baseline_decode_epochs.end(),
                    candidate->decode_epochs.begin(),
                    candidate->decode_epochs.end());
                convergence_timings_.baseline_decode_input_tokens.insert(
                    convergence_timings_.baseline_decode_input_tokens.end(),
                    candidate->decode_input_tokens.begin(),
                    candidate->decode_input_tokens.end());
                selected_baseline_samples.push_back(&*candidate);
            }
        }
        std::vector<int32_t> decode_input_tokens;

        for (int request = 0;
             request < requested_identities;
             ++request)
        {
            activeClearCache();
            const int prompt_identity = prompt_identities.at(
                static_cast<std::size_t>(request));
            const std::vector<int32_t> prompt =
                makeReferenceShapedEconomyPrompt(
                    prompt_identity,
                    ReferenceEconomyPromptRole::TimingCohort);
            const bool retain_sample =
                request >= kConvergenceTimingWarmupRequests;
            ResidencyConvergenceTimings::RequestSample request_sample;
            request_sample.prompt_identity = prompt_identity;
            const std::optional<ConvergenceTimerSnapshot> request_timer_begin =
                retain_sample
                    ? std::optional<ConvergenceTimerSnapshot>(
                          convergenceTimerSnapshot())
                    : std::nullopt;

            const std::uint64_t prefill_waves_before =
                localCommittedWaveCount();
            const auto prefill_start = std::chrono::steady_clock::now();
            if (!orch_runner_->prefill(prompt))
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Residency timing prefill failed: "
                    << orch_runner_->lastError());
                return false;
            }
            const std::uint64_t prefill_ns =
                elapsedNanoseconds(prefill_start);
            const std::uint64_t prefill_waves_after =
                localCommittedWaveCount();
            if (prefill_waves_before != cohort_wave ||
                prefill_waves_after != cohort_wave)
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Residency changed inside the controlled prefill cohort: expected="
                    << cohort_wave << " before=" << prefill_waves_before
                    << " after=" << prefill_waves_after);
                return false;
            }
            const PrefixRuntimeStateSnapshot prefix_state =
                orch_runner_->prefixStateProbe();
            if (prefix_state.prefix_request.hit ||
                prefix_state.prefix_request.matched_tokens != 0)
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Controlled timing prefill restored cached model state instead of executing the full prompt");
                return false;
            }
            if (retain_sample)
            {
                request_sample.prefill_ns = prefill_ns;
                request_sample.prefill_epoch = cohort_wave + 1u;
            }

            for (int step = 0;
                 step < kConvergenceTimingDecodeForwardsPerRequest;
                 ++step)
            {
                if (step > 0)
                {
                    /*
                     * Restore the exact prefill terminal state so every timed
                     * decode consumes the same autoregressive input. The first
                     * pass above also proves full-compute prefill economy; these
                     * untimed repetitions prove normal prefix reuse within one
                     * immutable placement epoch.
                     */
                    activeClearCache();
                    const std::uint64_t restore_waves_before =
                        localCommittedWaveCount();
                    if (!orch_runner_->prefill(prompt))
                    {
                        LOG_ERROR(
                            "[Qwen3.5 MoE GraphNative] Residency timing prefix restore failed: "
                            << orch_runner_->lastError());
                        return false;
                    }
                    const std::uint64_t restore_waves_after =
                        localCommittedWaveCount();
                    const PrefixRuntimeStateSnapshot restored_prefix_state =
                        orch_runner_->prefixStateProbe();
                    if (restore_waves_before != cohort_wave ||
                        restore_waves_after != cohort_wave ||
                        !restored_prefix_state.prefix_request.hit ||
                        restored_prefix_state.prefix_request.matched_tokens !=
                            static_cast<int>(prompt.size()))
                    {
                        LOG_ERROR(
                            "[Qwen3.5 MoE GraphNative] Controlled decode sample did not restore the exact epoch-stable prefix: expected_epoch="
                            << cohort_wave << " before="
                            << restore_waves_before << " after="
                            << restore_waves_after << " hit="
                            << restored_prefix_state.prefix_request.hit
                            << " matched_tokens="
                            << restored_prefix_state.prefix_request.matched_tokens
                            << " prompt_tokens=" << prompt.size());
                        return false;
                    }
                }

                /* Consume prefill logits outside the decode timing interval. */
                const std::uint64_t boundary_waves_before =
                    localCommittedWaveCount();
                orch_runner_->setDecodeStepTokenBudget(1);
                GenerationResult boundary_sample = orch_runner_->decodeStep();
                orch_runner_->setDecodeStepTokenBudget(0);
                const std::uint64_t boundary_waves_after =
                    localCommittedWaveCount();
                if (!boundary_sample.success() ||
                    boundary_sample.tokens.size() != 1u ||
                    boundary_waves_before != cohort_wave ||
                    boundary_waves_after != cohort_wave)
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Residency timing boundary sample was not one epoch-stable token: error="
                        << boundary_sample.error << " tokens="
                        << boundary_sample.tokens.size() << " expected_epoch="
                        << cohort_wave << " before=" << boundary_waves_before
                        << " after=" << boundary_waves_after);
                    return false;
                }
                if (retain_sample)
                {
                    request_sample.decode_input_tokens.push_back(
                        boundary_sample.tokens.front());
                }
                if (!orch_runner_->maybeApplyMoERebalance(1u))
                    return false;

                const std::uint64_t decode_waves_before =
                    localCommittedWaveCount();
                orch_runner_->setDecodeStepTokenBudget(1);
                const auto decode_start = std::chrono::steady_clock::now();
                GenerationResult decoded = orch_runner_->decodeStep();
                const std::uint64_t decode_ns =
                    elapsedNanoseconds(decode_start);
                orch_runner_->setDecodeStepTokenBudget(0);
                const std::uint64_t decode_waves_after =
                    localCommittedWaveCount();
                if (!decoded.success() || decoded.tokens.size() != 1u)
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Residency timing decode did not execute exactly one forward: error="
                        << decoded.error << " tokens="
                        << decoded.tokens.size());
                    return false;
                }
                if (decode_waves_before != cohort_wave ||
                    decode_waves_after != cohort_wave)
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Residency changed inside the controlled decode cohort: expected="
                        << cohort_wave << " before=" << decode_waves_before
                        << " after=" << decode_waves_after);
                    return false;
                }
                if (!orch_runner_->maybeApplyMoERebalance(1u))
                    return false;

                if (retain_sample)
                {
                    request_sample.decode_ns.push_back(decode_ns);
                    request_sample.decode_epochs.push_back(
                        cohort_wave + 1u);
                }
            }

            if (!retain_sample)
                continue;
            if (request_sample.decode_ns.size() !=
                    static_cast<std::size_t>(
                        kConvergenceTimingDecodeForwardsPerRequest) ||
                request_sample.decode_epochs.size() !=
                    request_sample.decode_ns.size() ||
                request_sample.decode_input_tokens.size() !=
                    request_sample.decode_ns.size())
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] A controlled timing request did not retain one complete decode sample group: identity="
                    << prompt_identity << " decode="
                    << request_sample.decode_ns.size() << " epochs="
                    << request_sample.decode_epochs.size() << " inputs="
                    << request_sample.decode_input_tokens.size());
                return false;
            }
            if (!request_timer_begin)
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] A retained timing request has no timer interval origin");
                return false;
            }
            request_sample.timer_deltas = convergenceTimerDelta(
                *request_timer_begin,
                convergenceTimerSnapshot());
            if (initial)
            {
                convergence_timings_.baseline_candidates.push_back(
                    std::move(request_sample));
            }
            else
            {
                convergence_timings_.converged_prefill_ns.push_back(
                    request_sample.prefill_ns);
                convergence_timings_.converged_prefill_epochs.push_back(
                    request_sample.prefill_epoch);
                convergence_timings_.converged_decode_ns.insert(
                    convergence_timings_.converged_decode_ns.end(),
                    request_sample.decode_ns.begin(),
                    request_sample.decode_ns.end());
                convergence_timings_.converged_decode_epochs.insert(
                    convergence_timings_.converged_decode_epochs.end(),
                    request_sample.decode_epochs.begin(),
                    request_sample.decode_epochs.end());
                decode_input_tokens.insert(
                    decode_input_tokens.end(),
                    request_sample.decode_input_tokens.begin(),
                    request_sample.decode_input_tokens.end());
                convergence_timings_.converged_samples.push_back(
                    std::move(request_sample));
            }
        }

        const bool complete = initial
                                  ? convergence_timings_
                                            .baseline_candidates.size() ==
                                        static_cast<std::size_t>(
                                            kConvergenceTimingMeasuredRequests)
                                  : convergence_timings_
                                                .converged_prefill_ns.size() ==
                                            static_cast<std::size_t>(
                                                kConvergenceTimingMeasuredRequests) &&
                                        convergence_timings_
                                                .converged_decode_ns.size() ==
                                            kRequiredDecodeSamples &&
                                        convergence_timings_
                                                .baseline_prefill_ns.size() ==
                                            static_cast<std::size_t>(
                                                kConvergenceTimingMeasuredRequests) &&
                                        convergence_timings_
                                                .baseline_decode_ns.size() ==
                                            kRequiredDecodeSamples &&
                                        convergence_timings_
                                                .converged_samples.size() ==
                                            static_cast<std::size_t>(
                                                kConvergenceTimingMeasuredRequests);
        if (localCommittedWaveCount() != cohort_wave || !complete)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Controlled timing cohort was incomplete or crossed an epoch: prefill="
                << (initial
                        ? convergence_timings_.baseline_candidates.size()
                        : convergence_timings_.converged_prefill_ns.size())
                << " decode="
                << (initial ? 0u
                            : convergence_timings_.converged_decode_ns.size())
                << " expected_epoch="
                << cohort_wave << " final_epoch="
                << localCommittedWaveCount());
            return false;
        }

        if (!initial &&
            decode_input_tokens !=
                 convergence_timings_.baseline_decode_input_tokens)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Before/after timing cohorts supplied different sampled inputs to a timed decode: baseline_inputs="
                << convergence_timings_.baseline_decode_input_tokens.size()
                << " converged_inputs=" << decode_input_tokens.size());
            return false;
        }
        if (!initial)
        {
            writeConvergenceTimerSamples(
                selected_baseline_samples,
                convergence_timings_.converged_samples);
        }
        if (initial)
            dynamic_residency_proof_lifecycle_.recordInitialCohort();
        else
            dynamic_residency_proof_lifecycle_.recordConvergedCohort();
        return true;
    }

    /** @return Stable diagnostic spelling for the active priority topology. */
    auto Qwen35MoENodeExpertOverlayParityTest::convergenceTopologyName() const -> std::string
    {
        return activeModelParityCaseOrThrow().topology.test_id;
    }

    /**
     * @brief Assert and serialize the real before/after convergence evidence.
     *
     * A two-percent floor is deliberately larger than timer quantization and
     * ordinary run-to-run jitter on this host. The median makes the gate robust
     * to background OS activity while retaining a directional performance
     * requirement for both time-to-first-token prefill and steady decode.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::assertAndWriteObservedConvergenceSpeedup() -> void
    {
        if (!requiresObservedConvergenceSpeedup() || !isRootParityRank())
            return;

        ASSERT_EQ(
            convergence_timings_.baseline_prefill_ns.size(),
            static_cast<std::size_t>(
                kConvergenceTimingMeasuredRequests));
        ASSERT_EQ(
            convergence_timings_.baseline_decode_ns.size(),
            static_cast<std::size_t>(
                kConvergenceTimingMeasuredRequests *
                kConvergenceTimingDecodeForwardsPerRequest));
        ASSERT_EQ(
            convergence_timings_.converged_prefill_ns.size(),
            convergence_timings_.baseline_prefill_ns.size());
        ASSERT_EQ(
            convergence_timings_.converged_decode_ns.size(),
            convergence_timings_.baseline_decode_ns.size());

        const std::uint64_t baseline_prefill = medianNanoseconds(
            convergence_timings_.baseline_prefill_ns);
        const std::uint64_t baseline_decode = medianNanoseconds(
            convergence_timings_.baseline_decode_ns);
        const std::uint64_t converged_prefill = medianNanoseconds(
            convergence_timings_.converged_prefill_ns);
        const std::uint64_t converged_decode = medianNanoseconds(
            convergence_timings_.converged_decode_ns);
        constexpr long double kMaximumConvergedRatio = 0.98L;
        const bool prefill_passed =
            static_cast<long double>(converged_prefill) <=
            static_cast<long double>(baseline_prefill) *
                kMaximumConvergedRatio;
        const bool decode_passed =
            static_cast<long double>(converged_decode) <=
            static_cast<long double>(baseline_decode) *
                kMaximumConvergedRatio;
        const auto improvement = [](std::uint64_t baseline,
                                    std::uint64_t converged)
        {
            return 100.0L *
                   (static_cast<long double>(baseline) -
                    static_cast<long double>(converged)) /
                   static_cast<long double>(baseline);
        };
        const long double prefill_improvement =
            improvement(baseline_prefill, converged_prefill);
        const long double decode_improvement =
            improvement(baseline_decode, converged_decode);

        const auto path =
            ensureResultsDir() / "expert_overlay_convergence.csv";
        std::ofstream csv(path, std::ios::trunc);
        ASSERT_TRUE(csv.is_open())
            << "Cannot create observed ExpertOverlay convergence CSV at "
            << path;
        csv << "backend,topology,phase,cohort,sample,residency_epoch,latency_ns,baseline_median_ns,converged_median_ns,improvement_percent,passed\n";
        csv << std::fixed << std::setprecision(4);
        const auto write_samples = [&](const char *phase,
                                       const char *cohort,
                                       const std::vector<std::uint64_t> &samples,
                                       const std::vector<std::uint64_t> &epochs,
                                       std::uint64_t baseline_median,
                                       std::uint64_t converged_median,
                                       long double improvement_percent,
                                       bool passed)
        {
            for (std::size_t index = 0; index < samples.size(); ++index)
            {
                const std::uint64_t epoch =
                    epochs.empty() ? 1u : epochs.at(index);
                csv << getBackendName() << ','
                    << convergenceTopologyName() << ','
                    << phase << ','
                    << cohort << ','
                    << index << ','
                    << epoch << ','
                    << samples[index] << ','
                    << baseline_median << ','
                    << converged_median << ','
                    << static_cast<double>(improvement_percent) << ','
                    << (passed ? "true" : "false") << '\n';
            }
        };
        write_samples(
            "prefill",
            "initial_epoch",
            convergence_timings_.baseline_prefill_ns,
            convergence_timings_.baseline_prefill_epochs,
            baseline_prefill,
            converged_prefill,
            prefill_improvement,
            prefill_passed);
        write_samples(
            "prefill",
            "converged_epoch",
            convergence_timings_.converged_prefill_ns,
            convergence_timings_.converged_prefill_epochs,
            baseline_prefill,
            converged_prefill,
            prefill_improvement,
            prefill_passed);
        write_samples(
            "decode",
            "initial_epoch",
            convergence_timings_.baseline_decode_ns,
            convergence_timings_.baseline_decode_epochs,
            baseline_decode,
            converged_decode,
            decode_improvement,
            decode_passed);
        write_samples(
            "decode",
            "converged_epoch",
            convergence_timings_.converged_decode_ns,
            convergence_timings_.converged_decode_epochs,
            baseline_decode,
            converged_decode,
            decode_improvement,
            decode_passed);
        csv.flush();
        ASSERT_TRUE(csv.good())
            << "Failed to write observed ExpertOverlay convergence CSV at "
            << path;

        LOG_INFO(
            "[Qwen3.5 MoE GraphNative] Observed convergence performance: topology="
            << convergenceTopologyName()
            << " prefill_initial_ns=" << baseline_prefill
            << " prefill_converged_ns=" << converged_prefill
            << " prefill_improvement_percent="
            << static_cast<double>(prefill_improvement)
            << " decode_initial_ns=" << baseline_decode
            << " decode_converged_ns=" << converged_decode
            << " decode_improvement_percent="
            << static_cast<double>(decode_improvement));
        EXPECT_TRUE(prefill_passed)
            << "Priority convergence did not improve observed prefill latency by at least 2%";
        EXPECT_TRUE(decode_passed)
            << "Priority convergence did not improve observed decode latency by at least 2%";
    }

    /**
     * @brief Retire Dynamic proof prefixes before the mathematical prefill.
     *
     * The movement proof is ordinary production traffic and intentionally
     * executes a causal prefix of the authenticated Hugging Face prompt. A
     * movement wave may publish before that request is harvested, making its
     * cache entry valid under the final placement epoch. Clearing request KV
     * alone cannot distinguish that entry from the fresh parity seed. Cross
     * the public coordinated purge boundary while worker ranks are still live,
     * then publish the typed lifecycle transition that admits parity traffic.
     *
     * Prefix caching remains enabled. The fresh parity prefill immediately
     * seeds the new entry and the standard decode phase must restore and
     * numerically certify it.
     *
     * @return True only after every coordinated participant has purged the
     *         reusable archive and the lifecycle is ready for numerical parity.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::prepareDynamicNumericalParityBoundary() -> bool
    {
        auto profile_scope = profileParityScope(
            "qwen122.dynamic.prepare_numerical_boundary");
        const DynamicResidencyProofPhase phase =
            dynamic_residency_proof_lifecycle_.phase();
        if (phase !=
                DynamicResidencyProofPhase::MovementBoundarySettled &&
            phase !=
                DynamicResidencyProofPhase::ConvergedCohortMeasured)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Numerical parity requested before the Dynamic movement boundary settled: phase="
                << static_cast<int>(phase));
            return false;
        }

        activeClearSnapshots();
        activeClearCache();
        if (!orch_runner_->purgePrefixCache())
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Numerical parity could not retire Dynamic proof prefix state: "
                << orch_runner_->lastError());
            return false;
        }
        dynamic_residency_proof_lifecycle_
            .recordNumericalParityIsolation();
        return true;
    }

    /**
     * @brief Execute one ordinary request prefill for economy evidence.
     *
     * @param tokens Real model tokens supplied through the serving API.
     * @param purpose Stable diagnostic name for the traffic lifecycle phase.
     * @return Whether production prefill completed successfully.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::runDynamicEconomyPrefill(
        const std::vector<int32_t> &tokens,
        const char *purpose) -> bool
    {
        activeClearSnapshots();
        activeClearCache();
        if (orch_runner_->prefill(tokens))
            return true;
        LOG_ERROR(
            "[Qwen3.5 MoE GraphNative] Dynamic "
            << (purpose ? purpose : "economy")
            << " prefill failed: " << orch_runner_->lastError());
        return false;
    }

    /**
     * @brief Execute one ordinary bounded decode and notify maintenance.
     *
     * The first generated token after prefill consumes existing logits; the
     * next budgeted call executes a real DecodeToken graph. Callers that need
     * decode-route evidence therefore issue the typed boundary/forward pair
     * without invoking any test-only routing surface.
     *
     * @param response_token_budget Positive serving response budget.
     * @param purpose Stable diagnostic name for the traffic lifecycle phase.
     * @return Completion state, or no value when inference/maintenance failed.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::runDynamicEconomyDecode(
        int response_token_budget,
        const char *purpose) -> std::optional<bool>
    {
        if (response_token_budget <= 0)
            return std::nullopt;
        orch_runner_->setDecodeStepTokenBudget(response_token_budget);
        const GenerationResult generated = orch_runner_->decodeStep();
        orch_runner_->setDecodeStepTokenBudget(0);
        if (!generated.success() || generated.tokens.empty())
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Dynamic "
                << (purpose ? purpose : "economy")
                << " decode failed: " << generated.error);
            return std::nullopt;
        }
        if (!orch_runner_->maybeApplyMoERebalance(generated.tokens.size()))
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Dynamic maintenance wake after "
                << (purpose ? purpose : "economy")
                << " decode failed: " << orch_runner_->lastError());
            return std::nullopt;
        }
        return generated.is_complete;
    }

    /**
     * @brief Replay one untimed request with the exact measured route shape.
     *
     * Every pair restores the same terminal prefill state, consumes its logits
     * with one boundary sample, and executes one DecodeToken forward. This is
     * deliberately the same production sequence as collectInferenceTimings(),
     * minus clocks and assertions, so the planner cannot optimize a longer
     * autoregressive trajectory than the one judged by the convergence gate.
     *
     * @param corpus_request Stable request ordinal in the timing corpus.
     * @return Whether all production requests and maintenance notifications ran.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::replayStationaryConvergenceRequest(int corpus_request) -> bool
    {
        constexpr ConvergenceTrainingTrafficPlan traffic =
            convergenceTrainingTrafficPlan();
        static_assert(traffic.valid());
        const std::vector<int32_t> prompt =
            makeReferenceShapedEconomyPrompt(
                corpus_request,
                ReferenceEconomyPromptRole::TimingCohort);
        if (prompt.size() != traffic.cold_prefill_rows)
        {
            throw std::logic_error(
                "Convergence training prompt no longer matches the measured cold-prefill geometry");
        }

        /*
         * The three measured identities deliberately repeat across movement
         * epochs. Retire their reusable archive through the public production
         * control surface before each training request so the first pass is a
         * genuine captured prefill, not a 196-MiB prefix restore followed by
         * two one-row decode transactions. This changes no model, histogram,
         * or placement state; it only makes the finite training corpus execute
         * the same cold-prefill/restore pair as collectInferenceTimings().
         */
        activeClearSnapshots();
        activeClearCache();
        if (!orch_runner_->purgePrefixCache())
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Convergence training could not retire its prior prefix archive: "
                << orch_runner_->lastError());
            return false;
        }
        std::optional<PrefixCacheRequestSummary> cold_request;
        for (int step = 0;
             step < kConvergenceTimingDecodeForwardsPerRequest;
             ++step)
        {
            if (step == 0)
            {
                if (!runDynamicEconomyPrefill(
                        prompt, "stationary-convergence"))
                {
                    return false;
                }
                const PrefixRuntimeStateSnapshot cold_prefix =
                    orch_runner_->prefixStateProbe();
                if (cold_prefix.prefix_request.hit ||
                    cold_prefix.prefix_request.matched_tokens != 0)
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Convergence training cold prefill unexpectedly restored prefix state: hit="
                        << cold_prefix.prefix_request.hit
                        << " matched_tokens="
                        << cold_prefix.prefix_request.matched_tokens);
                    return false;
                }
                cold_request = cold_prefix.prefix_request;
            }
            else
            {
                /* Request reset plus ordinary prefix restore reproduces the
                 * exact terminal state consumed by every measured pair. A
                 * publication before this lookup may legitimately invalidate
                 * the archive and turn it into a second cold prefill. A
                 * publication after admission does not invalidate a completed
                 * restore: the request's RCU lease protects the restored state
                 * through inference while the new bank becomes live. */
                if (!runDynamicEconomyPrefill(
                        prompt, "stationary-convergence-prefix-restore"))
                {
                    return false;
                }
                const PrefixRuntimeStateSnapshot restored_prefix =
                    orch_runner_->prefixStateProbe();
                if (!cold_request)
                {
                    throw std::logic_error(
                        "Convergence prefix replay lost its cold-request lifecycle result");
                }
                const bool exact_restore =
                    restored_prefix.prefix_request.hit &&
                    restored_prefix.prefix_request.matched_tokens ==
                        static_cast<int>(prompt.size());
                const bool movement_preceded_restore_admission =
                    cold_request->movementPrecededAdmissionOf(
                        restored_prefix.prefix_request);
                const bool movement_invalidated_cold_prefill =
                    movement_preceded_restore_admission &&
                    !restored_prefix.prefix_request.hit &&
                    restored_prefix.prefix_request.matched_tokens == 0;
                if (!exact_restore &&
                    !movement_invalidated_cold_prefill)
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Convergence training prefix replay was neither an exact admitted restore nor a movement-invalidated cold prefill: cold_admission_epoch="
                        << cold_request->admission_placement_epochs.earliest()
                        << " cold_admission_latest_epoch="
                        << cold_request->admission_placement_epochs.latest()
                        << " cold_completion_epoch="
                        << cold_request->completion_movement_epoch
                        << " restore_admission_epoch="
                        << restored_prefix.prefix_request
                               .admission_placement_epochs.earliest()
                        << " restore_admission_latest_epoch="
                        << restored_prefix.prefix_request
                               .admission_placement_epochs.latest()
                        << " restore_completion_epoch="
                        << restored_prefix.prefix_request
                               .completion_movement_epoch
                        << " hit="
                        << restored_prefix.prefix_request.hit
                        << " matched_tokens="
                        << restored_prefix.prefix_request.matched_tokens
                        << " prompt_tokens=" << prompt.size());
                    return false;
                }
            }

            if (!runDynamicEconomyDecode(
                    1, "stationary-convergence-boundary") ||
                !runDynamicEconomyDecode(
                    1, "stationary-convergence-forward"))
            {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Execute one movement request whose hot routes remain replayable.
     *
     * The prefill is the exact authenticated Hugging Face request. Its pending
     * progress is retired by the next ordinary production request before that
     * request submits another captured prefill. This is sufficient to close
     * every histogram bank and ensures later numerical parity replays precisely
     * the route family that selected each promoted expert.
     *
     * @param corpus_request Stable request ordinal used by the traffic driver.
     * @return True after the boundary and one canonical routed forward finish.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::replayStationaryMovementProofRequest(int corpus_request) -> bool
    {
        constexpr MovementProofTrafficPlan plan =
            movementProofTrafficPlan();
        if (!plan.authenticatedPrefillOnly())
        {
            throw std::logic_error(
                "Movement-proof traffic admitted non-prefill rows");
        }

        const std::vector<int32_t> prompt =
            makeReferenceShapedEconomyPrompt(
                corpus_request,
                ReferenceEconomyPromptRole::MovementProof);
        if (prompt.size() != plan.authenticated_prefill_rows)
        {
            throw std::logic_error(
                "Movement-proof prompt no longer matches authenticated reference geometry");
        }
        return runDynamicEconomyPrefill(prompt, "stationary-movement");
    }

    /**
     * @brief Learn and publish the measured service profile from real traffic.
     *
     * Transport preparation already completed through prepareForInference().
     * This phase supplies a bounded broad token corpus so every live sparse
     * participant receives natural prefill/decode work. Production publishes
     * the certificate while the corpus remains quarantined from optimization
     * demand. The following public request boundary discards that calibration
     * bank and admits the real convergence/parity workload.
     *
     * @return True after the sole production authority reports certification.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::certifyDynamicResidencyEconomy() -> bool
    {
        auto profile_scope = profileParityScope(
            "qwen122.dynamic.certify_economy");
        if (!isDynamicResidencyProductionTest())
            return true;
        if (!isRootParityRank())
        {
            throw std::logic_error(
                "Only the coordinated parity root may certify Dynamic residency economy");
        }
        if (dynamic_residency_proof_lifecycle_.phase() !=
            DynamicResidencyProofPhase::AwaitingEconomyCertification)
        {
            throw std::logic_error(
                "Dynamic residency economy certification entered out of order");
        }

        const int decode_steps_per_request =
            isQwen122ProductionTest() ? 8 : 9;
        std::uint64_t service_profile_forwards = 0;
        MoEOptimizationStatus optimization = optimizationStatus();
        if (!optimization.learning() && !optimization.active())
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Dynamic economy certification has no learning or active production authority");
            return false;
        }
        const MoEOptimizationMovementLedger certification_ledger =
            orch_runner_->moeOptimizationMovementLedger();
        if (!certification_ledger.complete())
        {
            throw std::logic_error(
                "Dynamic economy certification began with a truncated authoritative ledger");
        }
        /*
         * Freeze the frontier before the first ordinary service request. A
         * background transfer is allowed to publish while this corpus runs;
         * that publication is part of the convergence proof, not a new
         * baseline to be silently adopted by the later traffic driver.
         */
        dynamic_residency_proof_lifecycle_.beginEconomyCertification({
            .published_waves = optimization.published_movement_waves,
            .completed_transactions =
                optimization.completed_movement.transactions,
            .ledger_edges = certification_ledger.edges.size(),
            .host_admissions =
                certification_ledger.host_admissions.size(),
        });
        for (int request_index = 0;
             request_index < kMaximumServiceCertificationRequests &&
            !optimization.active();
             ++request_index)
        {
            if (!runDynamicEconomyPrefill(
                    makeEconomyWorkloadPrompt(request_index),
                    "service-certification"))
            {
                return false;
            }
            ++service_profile_forwards;

            bool request_complete = false;
            for (int step = 0;
                 !request_complete && step < decode_steps_per_request;
                 ++step)
            {
                const int response_token_budget =
                    isQwen122ProductionTest() &&
                            step < decode_steps_per_request / 2
                        ? 1
                        : (isQwen122ProductionTest() ? 4 : 2);
                const auto complete = runDynamicEconomyDecode(
                    response_token_budget,
                    "service-certification");
                if (!complete)
                    return false;
                ++service_profile_forwards;
                request_complete = *complete;
            }
            optimization = optimizationStatus();
        }

        if (!optimization.active())
        {
            struct ServiceTrafficTotals
            {
                double active_routes = 0.0;
                std::uint64_t completed_packets = 0u;
            };
            std::map<std::pair<std::string, std::string>,
                     ServiceTrafficTotals>
                service_traffic_by_participant_phase;
            for (const auto &record :
                 PerfStatsCollector::snapshot({"forward_graph"}))
            {
                if (record.kind != PerfStatRecord::Kind::Counter ||
                    record.domain != "forward_graph")
                {
                    continue;
                }
                const auto participant = record.tags.find("participant");
                const auto source = record.tags.find("service_source");
                if (participant == record.tags.end() ||
                    source == record.tags.end())
                {
                    continue;
                }
                auto &totals = service_traffic_by_participant_phase[
                    {participant->second, source->second}];
                if (record.name ==
                    "moe_overlay_local_expert_active_routes")
                {
                    totals.active_routes += record.value;
                }
                else if (record.name ==
                         "moe_overlay_local_expert_completions")
                {
                    totals.completed_packets += record.count;
                }
            }
            std::ostringstream service_traffic;
            service_traffic << "rank=" << (mpi_ctx_ ? mpi_ctx_->rank() : 0);
            for (const auto &[coordinate, totals] :
                 service_traffic_by_participant_phase)
            {
                service_traffic
                    << " p" << coordinate.first << '/'
                    << coordinate.second << "{routes="
                    << static_cast<std::uint64_t>(totals.active_routes)
                    << ",packets=" << totals.completed_packets << '}';
            }
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Dynamic economy certification did not complete after "
                << service_profile_forwards
                << " ordinary production forwards; service_traffic="
                << service_traffic.str() << "\n"
                << PerfStatsCollector::summaryString(
                       {"moe_overlay_residency"}));
            return false;
        }

        dynamic_residency_proof_lifecycle_.completeEconomyCertification();
        return true;
    }

}
