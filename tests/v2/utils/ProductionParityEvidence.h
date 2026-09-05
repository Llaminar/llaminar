/**
 * @file ProductionParityEvidence.h
 * @brief Typed interpretation of production-path PerfStats for parity gates.
 *
 * Numerical parity is meaningful only when the production implementation that
 * a campaign claims to certify actually ran.  This helper interprets the
 * request-local MTP controller records without coupling fast unit tests to the
 * large model-parity fixture. CUDA certifies one native conditional parent.
 * ROCm, whose HIP graph API has no conditional nodes, certifies a narrow
 * authenticated scheduler ticket and the retained captured transaction family
 * selected by that ticket. Both policies leave mutable generation state under
 * device-controller authority.
 */

#pragma once

#include "kernels/common/SamplingMath.h"
#include "utils/PerfStatsCollector.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace llaminar2::test::parity
{
    /**
     * @brief Authority that submitted the complete MTP generation loop.
     *
     * CUDA uses `NativeConditionalParent`. HIP uses
     * `HostScheduledCapturedTransactions`: the host observes only a fixed-size
     * immutable decision ticket and submits the already-captured branch named
     * by the device controller. The latter is a first-class policy, not eager
     * replay or host ownership of generation state.
     */
    enum class ProductionDeviceGenerationPolicy
    {
        NotObserved,
        NativeConditionalParent,
        HostScheduledCapturedTransactions,
        Unclassified,
        Inconsistent,
    };

    /**
     * @brief Return the stable CSV spelling for a generation policy.
     *
     * @param policy Typed policy inferred from request-local PerfStats.
     * @return A process-lifetime string literal suitable for diagnostics/CSV.
     */
    inline const char *productionDeviceGenerationPolicyName(
        ProductionDeviceGenerationPolicy policy)
    {
        switch (policy)
        {
        case ProductionDeviceGenerationPolicy::NotObserved:
            return "not_observed";
        case ProductionDeviceGenerationPolicy::NativeConditionalParent:
            return "native_conditional_parent";
        case ProductionDeviceGenerationPolicy::HostScheduledCapturedTransactions:
            return "host_scheduled_captured_transactions";
        case ProductionDeviceGenerationPolicy::Unclassified:
            return "unclassified";
        case ProductionDeviceGenerationPolicy::Inconsistent:
            return "inconsistent";
        }
        return "unclassified";
    }

    /** @brief Request-local proof of generation-loop authority and dispatch. */
    struct ProductionDeviceGenerationEvidence
    {
        bool controller_observed = false;
        ProductionDeviceGenerationPolicy policy =
            ProductionDeviceGenerationPolicy::NotObserved;
        bool hosted_ticket_boundary_certified = false;
        std::string certification_detail = "not_observed";

        /** @brief Return whether one native parent owned all transactions. */
        bool hasNativeParent() const
        {
            return controller_observed &&
                   policy ==
                       ProductionDeviceGenerationPolicy::NativeConditionalParent;
        }

        /**
         * @brief Return whether the backend-specific production loop is proven.
         *
         * CUDA is proven by native-parent policy evidence; its complete graph
         * capture/replay is checked by the surrounding parity fixture. ROCm is
         * proven only when the full ticket/transaction/ledger protocol below
         * agrees independently on every participating device.
         */
        bool hasCertifiedGenerationLoop() const
        {
            return hasNativeParent() ||
                   (controller_observed &&
                    policy == ProductionDeviceGenerationPolicy::
                                  HostScheduledCapturedTransactions &&
                    hosted_ticket_boundary_certified);
        }
    };

    /**
     * @brief Physical execution shape whose graph contract parity must prove.
     *
     * CPU execution is declarative but has no accelerator graph to capture. A
     * homogeneous GPU topology must execute one backend-native captured graph
     * policy. Any topology containing an accelerator plus another device type
     * crosses an explicitly declared captured-segment boundary; merely
     * entering the declarative graph runner is not sufficient evidence.
     */
    enum class ProductionParityExecutionTopology : std::uint8_t
    {
        CPUOnly,
        HomogeneousGPU,
        HeterogeneousAccelerator,
    };

    /** @return Stable diagnostic spelling for an execution topology. */
    inline const char *productionParityExecutionTopologyName(
        ProductionParityExecutionTopology topology)
    {
        switch (topology)
        {
        case ProductionParityExecutionTopology::CPUOnly:
            return "cpu_only";
        case ProductionParityExecutionTopology::HomogeneousGPU:
            return "homogeneous_gpu";
        case ProductionParityExecutionTopology::HeterogeneousAccelerator:
            return "heterogeneous_accelerator";
        }
        return "unknown";
    }

    /**
     * @brief Rank-local graph proof required by one global execution topology.
     *
     * A heterogeneous campaign has exactly one continuation/artifact authority
     * that proves the global segmented boundary. GPU follower ranks prove their
     * own native captured participant graph; CPU-only followers prove their
     * declarative participant graph. This distinction preserves a mandatory
     * global boundary proof without pretending every rank owns that boundary.
     */
    enum class ProductionParityGraphContract : std::uint8_t
    {
        CPUDeclarative,
        HomogeneousGPUCaptured,
        HeterogeneousCoordinatorSegmented,
        HeterogeneousGPUParticipantCaptured,
        HeterogeneousCPUParticipantDeclarative,
    };

    /** @return Stable diagnostic spelling for a rank-local graph contract. */
    inline const char *productionParityGraphContractName(
        ProductionParityGraphContract contract)
    {
        switch (contract)
        {
        case ProductionParityGraphContract::CPUDeclarative:
            return "cpu_declarative";
        case ProductionParityGraphContract::HomogeneousGPUCaptured:
            return "homogeneous_gpu_captured";
        case ProductionParityGraphContract::
            HeterogeneousCoordinatorSegmented:
            return "heterogeneous_coordinator_segmented";
        case ProductionParityGraphContract::
            HeterogeneousGPUParticipantCaptured:
            return "heterogeneous_gpu_participant_captured";
        case ProductionParityGraphContract::
            HeterogeneousCPUParticipantDeclarative:
            return "heterogeneous_cpu_participant_declarative";
        }
        return "unknown";
    }

    /**
     * @brief Classify a topology from its typed participant inventory.
     *
     * @param cpu_count Number of CPU participants.
     * @param cuda_count Number of CUDA participants.
     * @param rocm_count Number of ROCm participants.
     * @return The one graph-execution contract applicable to the inventory.
     */
    inline ProductionParityExecutionTopology
    classifyProductionParityExecutionTopology(
        std::size_t cpu_count,
        std::size_t cuda_count,
        std::size_t rocm_count)
    {
        const std::size_t gpu_count = cuda_count + rocm_count;
        if (gpu_count == 0)
            return ProductionParityExecutionTopology::CPUOnly;
        if (cpu_count == 0 && (cuda_count == 0 || rocm_count == 0))
            return ProductionParityExecutionTopology::HomogeneousGPU;
        return ProductionParityExecutionTopology::HeterogeneousAccelerator;
    }

    /**
     * @brief Resolve one rank's proof obligation from global typed ownership.
     *
     * @param topology Global participant topology for the matrix cell.
     * @param is_campaign_authority Whether this rank owns continuation output
     *        and canonical artifacts after inventory binding.
     * @param local_accelerator Whether this rank's production runner owns an
     *        accelerator graph.
     * @return The only graph contract valid for this rank.
     */
    inline ProductionParityGraphContract resolveProductionParityGraphContract(
        ProductionParityExecutionTopology topology,
        bool is_campaign_authority,
        bool local_accelerator)
    {
        switch (topology)
        {
        case ProductionParityExecutionTopology::CPUOnly:
            return ProductionParityGraphContract::CPUDeclarative;
        case ProductionParityExecutionTopology::HomogeneousGPU:
            return ProductionParityGraphContract::HomogeneousGPUCaptured;
        case ProductionParityExecutionTopology::HeterogeneousAccelerator:
            if (is_campaign_authority)
            {
                return ProductionParityGraphContract::
                    HeterogeneousCoordinatorSegmented;
            }
            return local_accelerator
                       ? ProductionParityGraphContract::
                             HeterogeneousGPUParticipantCaptured
                       : ProductionParityGraphContract::
                             HeterogeneousCPUParticipantDeclarative;
        }
        return ProductionParityGraphContract::CPUDeclarative;
    }

    /**
     * @brief Structured proof emitted by one live production parity campaign.
     *
     * Numerical CSVs remain the mathematical oracle. This record separately
     * proves that those values came from the production graph topology and,
     * for MTP, from the backend's complete device-generation policy.
     */
    struct ProductionParityEvidence
    {
        bool graph_execution = false;
        ProductionParityExecutionTopology execution_topology =
            ProductionParityExecutionTopology::CPUOnly;
        ProductionParityGraphContract graph_contract =
            ProductionParityGraphContract::CPUDeclarative;
        bool forward_full_graph_capture = false;
        bool forward_full_graph_replay = false;
        bool full_graph_capture = false;
        bool full_graph_replay = false;
        bool decode_graph_capture = false;
        bool decode_graph_replay = false;
        bool device_generation_controller = false;
        ProductionDeviceGenerationPolicy generation_execution_policy =
            ProductionDeviceGenerationPolicy::NotObserved;
        bool native_generation_parent = false;
        bool hosted_ticket_boundary_certified = false;
        bool generation_loop_certified = false;
        std::string generation_certification_detail = "not_observed";
        bool segmented_plan = false;
        bool segmented_capture = false;
        bool segmented_replay = false;
        bool model_context_reused = false;
        double elapsed_seconds = 0.0;
        double target_seconds = 4500.0;

        /** @return Whether at least one accelerator participates. */
        bool hasAccelerator() const
        {
            return execution_topology !=
                   ProductionParityExecutionTopology::CPUOnly;
        }

        /** @return Whether every participant is the same GPU backend. */
        bool usesHomogeneousGPU() const
        {
            return execution_topology ==
                   ProductionParityExecutionTopology::HomogeneousGPU;
        }
    };

    /**
     * @brief Recognize a completed retained-parent decode replay.
     *
     * The retained-parent executor historically exposed two counter spellings:
     * `retained_parent_replays` is the current production counter, while
     * `retained_parent_transaction_replays` remains valid evidence from the
     * explicit transaction path. Keeping the accepted producer vocabulary in
     * one typed predicate prevents parity fixtures from certifying segmented
     * replay but rejecting the same record as decode replay.
     *
     * @param records Request-local PerfStats snapshot.
     * @return `true` only for a positive decode-phase counter from either
     *         retained-parent replay producer.
     */
    inline bool productionParityHasRetainedParentDecodeReplay(
        const std::vector<PerfStatRecord> &records)
    {
        return std::any_of(
            records.begin(),
            records.end(),
            [](const PerfStatRecord &record)
            {
                return record.kind == PerfStatRecord::Kind::Counter &&
                       record.domain == "forward_graph" &&
                       record.phase == "decode" && record.value > 0.0 &&
                       (record.name == "retained_parent_replays" ||
                        record.name ==
                            "retained_parent_transaction_replays");
            });
    }

    /**
     * @brief Fail-closed result of validating graph-path evidence.
     *
     * A typed result makes the first violated lifecycle invariant explicit and
     * prevents independent boolean checks from accidentally omitting an entire
     * topology class.
     */
    enum class ProductionParityGraphCertification : std::uint8_t
    {
        Certified,
        InconsistentTopologyContract,
        MissingDeclarativeGraphExecution,
        MissingNativeCapturedGraph,
        MissingNativeDecodeGraph,
        UnexpectedNativeSegmentation,
        MissingHeterogeneousSegmentPlan,
        MissingHeterogeneousSegmentCapture,
        MissingHeterogeneousDecodeCapture,
        MissingHeterogeneousDecodeReplay,
        MissingHeterogeneousSegmentReplay,
    };

    /** @return Stable diagnostic spelling for a graph certification result. */
    inline const char *productionParityGraphCertificationName(
        ProductionParityGraphCertification certification)
    {
        switch (certification)
        {
        case ProductionParityGraphCertification::Certified:
            return "certified";
        case ProductionParityGraphCertification::
            InconsistentTopologyContract:
            return "inconsistent_topology_contract";
        case ProductionParityGraphCertification::
            MissingDeclarativeGraphExecution:
            return "missing_declarative_graph_execution";
        case ProductionParityGraphCertification::
            MissingNativeCapturedGraph:
            return "missing_native_captured_graph";
        case ProductionParityGraphCertification::
            MissingNativeDecodeGraph:
            return "missing_native_decode_graph";
        case ProductionParityGraphCertification::
            UnexpectedNativeSegmentation:
            return "unexpected_native_segmentation";
        case ProductionParityGraphCertification::
            MissingHeterogeneousSegmentPlan:
            return "missing_heterogeneous_segment_plan";
        case ProductionParityGraphCertification::
            MissingHeterogeneousSegmentCapture:
            return "missing_heterogeneous_segment_capture";
        case ProductionParityGraphCertification::
            MissingHeterogeneousDecodeCapture:
            return "missing_heterogeneous_decode_capture";
        case ProductionParityGraphCertification::
            MissingHeterogeneousDecodeReplay:
            return "missing_heterogeneous_decode_replay";
        case ProductionParityGraphCertification::
            MissingHeterogeneousSegmentReplay:
            return "missing_heterogeneous_segment_replay";
        }
        return "unknown";
    }

    /** @return Whether the selected generation policy ran captured work. */
    inline bool productionParityHasRequiredGenerationGraph(
        const ProductionParityEvidence &evidence)
    {
        if (evidence.device_generation_controller &&
            evidence.generation_execution_policy ==
                ProductionDeviceGenerationPolicy::
                    HostScheduledCapturedTransactions)
        {
            return evidence.forward_full_graph_capture ||
                   evidence.forward_full_graph_replay;
        }
        return evidence.full_graph_capture || evidence.full_graph_replay;
    }

    /**
     * @brief Validate the complete graph contract for one production cell.
     *
     * @return `Certified` only when the topology's production capture/replay
     *         lifecycle is explicitly evidenced.
     */
    inline ProductionParityGraphCertification
    certifyProductionParityGraphExecution(
        const ProductionParityEvidence &evidence)
    {
        if (!evidence.graph_execution)
        {
            return ProductionParityGraphCertification::
                MissingDeclarativeGraphExecution;
        }

        const bool contract_matches_topology =
            (evidence.execution_topology ==
                 ProductionParityExecutionTopology::CPUOnly &&
             evidence.graph_contract ==
                 ProductionParityGraphContract::CPUDeclarative) ||
            (evidence.execution_topology ==
                 ProductionParityExecutionTopology::HomogeneousGPU &&
             evidence.graph_contract == ProductionParityGraphContract::
                                            HomogeneousGPUCaptured) ||
            (evidence.execution_topology == ProductionParityExecutionTopology::
                                                HeterogeneousAccelerator &&
             (evidence.graph_contract == ProductionParityGraphContract::
                                             HeterogeneousCoordinatorSegmented ||
              evidence.graph_contract == ProductionParityGraphContract::
                                             HeterogeneousGPUParticipantCaptured ||
              evidence.graph_contract == ProductionParityGraphContract::
                                             HeterogeneousCPUParticipantDeclarative));
        if (!contract_matches_topology)
        {
            return ProductionParityGraphCertification::
                InconsistentTopologyContract;
        }

        switch (evidence.graph_contract)
        {
        case ProductionParityGraphContract::CPUDeclarative:
        case ProductionParityGraphContract::
            HeterogeneousCPUParticipantDeclarative:
            return ProductionParityGraphCertification::Certified;
        case ProductionParityGraphContract::HomogeneousGPUCaptured:
        case ProductionParityGraphContract::
            HeterogeneousGPUParticipantCaptured:
            if (evidence.segmented_plan || evidence.segmented_capture ||
                evidence.segmented_replay)
            {
                return ProductionParityGraphCertification::
                    UnexpectedNativeSegmentation;
            }
            if (!productionParityHasRequiredGenerationGraph(evidence))
            {
                return ProductionParityGraphCertification::
                    MissingNativeCapturedGraph;
            }
            if (!evidence.decode_graph_capture &&
                !evidence.decode_graph_replay)
            {
                return ProductionParityGraphCertification::
                    MissingNativeDecodeGraph;
            }
            return ProductionParityGraphCertification::Certified;
        case ProductionParityGraphContract::
            HeterogeneousCoordinatorSegmented:
            if (!evidence.segmented_plan)
            {
                return ProductionParityGraphCertification::
                    MissingHeterogeneousSegmentPlan;
            }
            if (!evidence.segmented_capture)
            {
                return ProductionParityGraphCertification::
                    MissingHeterogeneousSegmentCapture;
            }
            if (!evidence.decode_graph_capture)
            {
                return ProductionParityGraphCertification::
                    MissingHeterogeneousDecodeCapture;
            }
            if (!evidence.decode_graph_replay)
            {
                return ProductionParityGraphCertification::
                    MissingHeterogeneousDecodeReplay;
            }
            if (!evidence.segmented_replay)
            {
                return ProductionParityGraphCertification::
                    MissingHeterogeneousSegmentReplay;
            }
            return ProductionParityGraphCertification::Certified;
        }
        return ProductionParityGraphCertification::
            InconsistentTopologyContract;
    }

    namespace detail
    {
        /** @brief Classify one known execution-policy tag value. */
        inline ProductionDeviceGenerationPolicy classifyGenerationPolicyTag(
            std::string_view value)
        {
            if (value == "native_conditional_graph" ||
                value == "native_device_controlled_while" ||
                value == "native_device_controlled_selector_while" ||
                value == "single_async_native_while_launch" ||
                value == "single_async_native_selector_while_launch")
            {
                return ProductionDeviceGenerationPolicy::NativeConditionalParent;
            }
            if (value == "host_scheduled_captured_transactions" ||
                value ==
                    "hosted_captured_transactions_with_ticket_only_dispatch" ||
                value == "hosted_ticket_selected_captured_transactions")
            {
                return ProductionDeviceGenerationPolicy::
                    HostScheduledCapturedTransactions;
            }
            return ProductionDeviceGenerationPolicy::Unclassified;
        }

        /** @brief Return whether an aggregated PerfStats record did work. */
        inline bool exercised(const PerfStatRecord &record)
        {
            return record.value > 0.0 || record.count > 0 ||
                   record.total_ns > 0;
        }

        /** @brief Match one exact tag without inserting into the tag map. */
        inline bool tagEquals(
            const PerfStatRecord &record,
            std::string_view key,
            std::string_view expected)
        {
            const auto it = record.tags.find(std::string(key));
            return it != record.tags.end() && it->second == expected;
        }

        /** @brief Parse and validate one strictly positive integral tag. */
        inline bool positiveIntegerTag(
            const PerfStatRecord &record,
            std::string_view key)
        {
            const auto it = record.tags.find(std::string(key));
            if (it == record.tags.end() || it->second.empty())
                return false;

            long long parsed = 0;
            const char *const begin = it->second.data();
            const char *const end = begin + it->second.size();
            const auto [next, error] =
                std::from_chars(begin, end, parsed, 10);
            return error == std::errc{} && next == end && parsed > 0;
        }

        /** @brief Parse and validate one non-negative integral tag. */
        inline bool nonNegativeIntegerTag(
            const PerfStatRecord &record,
            std::string_view key)
        {
            const auto it = record.tags.find(std::string(key));
            if (it == record.tags.end() || it->second.empty())
                return false;

            long long parsed = 0;
            const char *const begin = it->second.data();
            const char *const end = begin + it->second.size();
            const auto [next, error] =
                std::from_chars(begin, end, parsed, 10);
            return error == std::errc{} && next == end && parsed >= 0;
        }

        /**
         * @brief Per-participant ledger used to certify HIP ticket dispatch.
         *
         * Boolean fields validate the shape and provenance of the captured
         * family. Numeric fields are independent aggregates whose equality
         * proves that no transaction disappeared between device publication,
         * host observation, captured submission, and terminal accounting.
         */
        struct HostedTicketDeviceEvidence
        {
            bool materialization_observed = false;
            bool materializations_valid = true;
            bool launch_observed = false;
            bool launches_valid = true;
            double launches = 0.0;
            bool ticket_observed = false;
            bool tickets_valid = true;
            bool observation_observed = false;
            bool observations_valid = true;
            bool transaction_submission_observed = false;
            bool transaction_submissions_valid = true;
            bool terminal_submission_observed = false;
            bool terminal_submissions_valid = true;
            bool compact_reducer_observed = false;
            bool compact_reducers_valid = true;
            double ticket_submissions = 0.0;
            double ticket_observations = 0.0;
            double transaction_submissions = 0.0;
            double terminal_submissions = 0.0;
            double terminal_transactions = 0.0;
            double compact_reductions = 0.0;
            double terminal_response_bridges = 0.0;

            /**
             * @brief Return whether one participant's protocol is complete.
             *
             * @param distributed_launch_certified Whether the rank-level
             *        launch record proves this participant was submitted
             *        before the first immutable-ticket wait.
             * @return `true` only when graph-family, launch, ticket, captured
             *         transaction, compact reduction, and terminal ledgers
             *         are mutually consistent.
             */
            bool certified(bool distributed_launch_certified) const
            {
                /*
                 * A request that completes in its first captured transaction
                 * has no non-terminal graph to submit.  Absence of the
                 * continuation counter is therefore the canonical zero value,
                 * not missing evidence.  Once a continuation exists, its
                 * tagged submission record remains mandatory and fail-closed.
                 */
                const bool continuation_submissions_certified =
                    transaction_submissions == 0.0 ||
                    (transaction_submission_observed &&
                     transaction_submissions_valid);
                return materialization_observed && materializations_valid &&
                       ((launch_observed && launches_valid &&
                         launches == terminal_response_bridges) ||
                        distributed_launch_certified) &&
                       ticket_observed &&
                       tickets_valid && observation_observed &&
                       observations_valid &&
                       continuation_submissions_certified &&
                       terminal_submission_observed &&
                       terminal_submissions_valid &&
                       compact_reducer_observed && compact_reducers_valid &&
                       terminal_transactions > 0.0 &&
                       ticket_submissions == terminal_transactions &&
                       ticket_observations == terminal_transactions &&
                       transaction_submissions + terminal_submissions ==
                           terminal_transactions &&
                       terminal_submissions > 0.0 &&
                       compact_reductions == terminal_transactions &&
                       terminal_response_bridges > 0.0;
            }
        };

        /**
         * @brief Recognize an intermediate host-read boundary conservatively.
         *
         * The two terminal D2H records are the compact final response. The
         * dispatch ticket is separately authenticated below. Any future MTP
         * D2H/host-materialization spelling fails closed until reviewed.
         */
        inline bool isForbiddenHostedGenerationHostRead(
            const PerfStatRecord &record)
        {
            const std::string_view name(record.name);
            const bool host_read =
                name == "host_logits_access" ||
                name.find("d2h") != std::string_view::npos ||
                name.find("host_bridge") != std::string_view::npos ||
                name.find("host_materialization") != std::string_view::npos;
            if (!host_read)
                return false;

            return name !=
                       "device_generation_dispatch_ticket_d2h_submissions" &&
                   name != "device_generation_terminal_d2h_enqueue" &&
                   name != "device_generation_terminal_d2h_wait";
        }

        /**
         * @brief Certify HIP's immutable-ticket captured-transaction protocol.
         *
         * Participants are discovered from a hosted graph-family
         * materialization or an exact retained-executable reuse, never inferred
         * from a requested topology. Every participant must publish an
         * internally consistent ledger, and all mirrored participants must
         * agree on transaction cardinalities.
         *
         * @param records Request-local PerfStats emitted at the public runner
         *        lifecycle boundary.
         * @param failure_detail Optional destination for a stable fail-closed
         *        diagnostic or the successful certification spelling.
         * @return `true` only when the complete HIP hosted-ticket protocol is
         *         independently proven for every participant.
         */
        inline bool certifyHostedTicketBoundary(
            const std::vector<PerfStatRecord> &records,
            std::string *failure_detail)
        {
            const auto fail = [failure_detail](std::string detail)
            {
                if (failure_detail)
                    *failure_detail = std::move(detail);
                return false;
            };
            std::map<std::string, HostedTicketDeviceEvidence> devices;
            bool policy_selection_observed = false;
            bool policy_selections_valid = true;
            bool rank_launch_observed = false;
            bool rank_launch_valid = true;
            double rank_launches = 0.0;

            // First establish the exact ROCm participants. A reuse record is
            // equally authoritative here: production emits it only after the
            // executable's complete identity matches the current request.
            // This prevents a malformed later ledger record from inventing a
            // device solely to satisfy counter equality.
            for (const PerfStatRecord &record : records)
            {
                if (record.kind != PerfStatRecord::Kind::Counter ||
                    record.domain != "mtp" || !exercised(record) ||
                    (record.name !=
                         "device_generation_loop_graph_materializations" &&
                     record.name !=
                         "device_generation_loop_graph_reuses") ||
                    !std::string_view(record.device).starts_with("ROCm:"))
                {
                    continue;
                }
                devices.try_emplace(record.device);
            }
            if (devices.empty())
                return fail("no_rocm_hosted_graph_family_evidence");

            for (const PerfStatRecord &record : records)
            {
                if (record.kind != PerfStatRecord::Kind::Counter ||
                    record.domain != "mtp" || !exercised(record))
                {
                    continue;
                }
                if (isForbiddenHostedGenerationHostRead(record))
                {
                    return fail(
                        "forbidden_host_read:" + record.device + ":" +
                        record.name);
                }

                if (record.name ==
                    "device_generation_execution_policy_selections")
                {
                    policy_selection_observed = true;
                    policy_selections_valid =
                        policy_selections_valid &&
                        tagEquals(
                            record,
                            "policy",
                            "host_scheduled_captured_transactions") &&
                        tagEquals(
                            record,
                            "selection_boundary",
                            "pre_first_draft") &&
                        (tagEquals(record, "topology", "fixed_depth") ||
                         tagEquals(record, "topology", "dynamic_depth"));
                }
                else if (
                    record.name == "rank_device_generation_parent_launches")
                {
                    rank_launch_observed = true;
                    rank_launch_valid =
                        rank_launch_valid &&
                        tagEquals(
                            record,
                            "execution",
                            "hosted_ticket_selected_captured_transactions") &&
                        tagEquals(
                            record,
                            "launch_order",
                            "all_participants_before_ticket_wait") &&
                        tagEquals(
                            record,
                            "participants",
                            std::to_string(devices.size()));
                    rank_launches += record.value;
                }

                const auto device_it = devices.find(record.device);
                if (device_it == devices.end())
                    continue;
                HostedTicketDeviceEvidence &device = device_it->second;

                if (record.name ==
                    "device_generation_loop_graph_materializations")
                {
                    device.materialization_observed = true;
                    device.materializations_valid =
                        device.materializations_valid &&
                        tagEquals(record, "backend", "HIP") &&
                        tagEquals(
                            record,
                            "execution",
                            "hosted_captured_transactions_with_ticket_only_dispatch") &&
                        positiveIntegerTag(record, "fragments") &&
                        tagEquals(record, "conditional_fragments", "0") &&
                        nonNegativeIntegerTag(
                            record,
                            "ticket_conditioned_fragments");
                }
                else if (
                    record.name == "device_generation_loop_graph_reuses")
                {
                    /*
                     * Reuse is stronger than a topology assertion. The graph
                     * owner emits it only after checking executable presence,
                     * workspace generation, request/depth/sampling geometry,
                     * fragment count, execution kind, and source-fragment
                     * identity. Revalidate the externally visible portion so
                     * stale or incomplete telemetry fails closed.
                     */
                    device.materialization_observed = true;
                    device.materializations_valid =
                        device.materializations_valid &&
                        tagEquals(record, "backend", "HIP") &&
                        tagEquals(
                            record,
                            "execution_policy",
                            "host_scheduled_captured_transactions") &&
                        positiveIntegerTag(record, "requests") &&
                        positiveIntegerTag(record, "draft_depth") &&
                        positiveIntegerTag(record, "minimum_draft_depth") &&
                        positiveIntegerTag(record, "maximum_draft_depth") &&
                        positiveIntegerTag(record, "fragments") &&
                        positiveIntegerTag(record, "workspace_generation") &&
                        tagEquals(record, "conditional_fragments", "0") &&
                        nonNegativeIntegerTag(
                            record,
                            "ticket_conditioned_fragments") &&
                        (tagEquals(record, "depth_policy", "fixed_width") ||
                         tagEquals(record, "depth_policy", "dynamic")) &&
                        (tagEquals(record, "sampling_mode", "greedy") ||
                         tagEquals(record, "sampling_mode", "stochastic"));
                }
                else if (
                    record.name == "device_generation_loop_graph_launches")
                {
                    device.launch_observed = true;
                    device.launches_valid =
                        device.launches_valid &&
                        tagEquals(record, "backend", "HIP") &&
                        tagEquals(
                            record,
                            "execution",
                            "hosted_ticket_selected_captured_transactions") &&
                        positiveIntegerTag(record, "fragments") &&
                        tagEquals(record, "conditional_fragments", "0") &&
                        nonNegativeIntegerTag(
                            record,
                            "ticket_conditioned_fragments");
                    device.launches += record.value;
                }
                else if (
                    record.name ==
                    "device_generation_dispatch_ticket_d2h_submissions")
                {
                    device.ticket_observed = true;
                    device.tickets_valid =
                        device.tickets_valid &&
                        tagEquals(
                            record,
                            "bytes",
                            std::to_string(
                                sampling_math::
                                    DeviceGenerationDispatchTicket::
                                        kWireBytes)) &&
                        tagEquals(
                            record,
                            "abi_version",
                            std::to_string(
                                sampling_math::
                                    DeviceGenerationDispatchTicket::
                                        kABIVersion)) &&
                        tagEquals(
                            record,
                            "word_count",
                            std::to_string(
                                sampling_math::
                                    DeviceGenerationDispatchTicket::
                                        kWordCount)) &&
                        tagEquals(
                            record,
                            "authority",
                            "immutable_scheduler_snapshot") &&
                        tagEquals(record, "state_payload", "false");
                    device.ticket_submissions += record.value;
                }
                else if (
                    record.name ==
                    "device_generation_dispatch_tickets_observed")
                {
                    device.observation_observed = true;
                    device.observations_valid =
                        device.observations_valid &&
                        positiveIntegerTag(record, "transaction") &&
                        positiveIntegerTag(record, "next_depth") &&
                        (tagEquals(record, "complete", "true") ||
                         tagEquals(record, "complete", "false")) &&
                        (tagEquals(record, "maintenance_due", "true") ||
                         tagEquals(record, "maintenance_due", "false"));
                    device.ticket_observations += record.value;
                }
                else if (
                    record.name ==
                    "hosted_device_generation_transaction_submissions")
                {
                    device.transaction_submission_observed = true;
                    device.transaction_submissions_valid =
                        device.transaction_submissions_valid &&
                        positiveIntegerTag(record, "depth") &&
                        positiveIntegerTag(record, "fragments") &&
                        tagEquals(
                            record,
                            "dynamic_depth_source",
                            "device_controller_ticket");
                    device.transaction_submissions += record.value;
                }
                else if (
                    record.name ==
                    "hosted_device_generation_terminal_submissions")
                {
                    device.terminal_submission_observed = true;
                    device.terminal_submissions_valid =
                        device.terminal_submissions_valid &&
                        positiveIntegerTag(record, "transactions");
                    device.terminal_submissions += record.value;
                }
                else if (
                    record.name == "device_generation_terminal_transactions")
                {
                    device.terminal_transactions += record.value;
                }
                else if (
                    record.name ==
                    "device_generation_terminal_compact_outcome_reductions")
                {
                    device.compact_reducer_observed = true;
                    device.compact_reducers_valid =
                        device.compact_reducers_valid &&
                        tagEquals(
                            record,
                            "authority",
                            "device_generation_controller") &&
                        tagEquals(
                            record,
                            "accounting_role",
                            "captured_graph_replay_multiplier") &&
                        (tagEquals(
                             record,
                             "source",
                             "captured_greedy_compact_outcome") ||
                         tagEquals(
                             record,
                             "source",
                             "captured_stochastic_compact_outcome")) &&
                        tagEquals(
                            record,
                            "execution",
                            "host_scheduled_captured_transactions");
                    device.compact_reductions += record.value;
                }
                else if (
                    record.name ==
                    "device_generation_terminal_response_bridges")
                {
                    device.terminal_response_bridges += record.value;
                }
            }

            if (!policy_selection_observed || !policy_selections_valid)
            {
                return fail(
                    !policy_selection_observed
                        ? "missing_pre_first_draft_policy_selection"
                        : "malformed_pre_first_draft_policy_selection");
            }

            // Mirrored participants must report exactly the same controller
            // ledger. An equality break proves rank divergence even if each
            // participant looked locally plausible.
            const HostedTicketDeviceEvidence &first = devices.begin()->second;
            const bool distributed_launch_certified =
                rank_launch_observed && rank_launch_valid &&
                rank_launches == first.terminal_response_bridges;
            for (const auto &[device_name, device] : devices)
            {
                if (!device.certified(distributed_launch_certified))
                {
                    std::ostringstream detail;
                    detail << "incomplete_participant:" << device_name
                           << ":materialization="
                           << device.materialization_observed << '/'
                           << device.materializations_valid
                           << ":launch=" << device.launch_observed << '/'
                           << device.launches_valid << '/' << device.launches
                           << ":rank_launch=" << rank_launch_observed << '/'
                           << rank_launch_valid << '/' << rank_launches
                           << ":ticket=" << device.ticket_observed << '/'
                           << device.tickets_valid
                           << ":observation="
                           << device.observation_observed << '/'
                           << device.observations_valid
                           << ":transaction_submission="
                           << device.transaction_submission_observed << '/'
                           << device.transaction_submissions_valid
                           << ":terminal_submission="
                           << device.terminal_submission_observed << '/'
                           << device.terminal_submissions_valid
                           << ":compact="
                           << device.compact_reducer_observed << '/'
                           << device.compact_reducers_valid
                           << ":ledger=" << device.ticket_submissions << '/'
                           << device.ticket_observations << '/'
                           << device.transaction_submissions << '+'
                           << device.terminal_submissions << '/'
                           << device.terminal_transactions << '/'
                           << device.compact_reductions
                           << ":response_bridges="
                           << device.terminal_response_bridges;
                    return fail(detail.str());
                }
                if (device.ticket_submissions != first.ticket_submissions ||
                    device.ticket_observations != first.ticket_observations ||
                    device.transaction_submissions !=
                        first.transaction_submissions ||
                    device.terminal_submissions !=
                        first.terminal_submissions ||
                    device.terminal_transactions !=
                        first.terminal_transactions ||
                    device.compact_reductions != first.compact_reductions)
                {
                    return fail("mirrored_participant_ledger_mismatch:" +
                                device_name);
                }
            }
            if (failure_detail)
                *failure_detail = "certified_rocm_ticket_boundary";
            return true;
        }
    } // namespace detail

    /**
     * @brief Infer outer-loop ownership from request-local MTP PerfStats.
     *
     * Every positive request-lifecycle `mtp.device_generation_*` counter proves
     * that the device generation controller participated. Setup-only stream and
     * storage initialization is deliberately excluded: ordinary MTP-off decode
     * may construct persistent infrastructure without admitting a controller.
     * Materialization, reuse, launch, and terminal-reduction records publish an
     * `execution`, `execution_policy`, or `policy` tag. All observed
     * policy-bearing records must agree. A future producer that omits or invents
     * a spelling becomes `Unclassified`, while simultaneous native and hosted
     * evidence becomes `Inconsistent`; neither state can accidentally certify a
     * homogeneous production campaign.
     *
     * @param records PerfStats snapshot containing the `mtp` domain.
     * @return Typed controller-presence and outer-loop authority evidence.
     */
    inline ProductionDeviceGenerationEvidence
    collectProductionDeviceGenerationEvidence(
        const std::vector<PerfStatRecord> &records)
    {
        bool native_policy_observed = false;
        bool hosted_policy_observed = false;
        ProductionDeviceGenerationEvidence evidence;

        for (const PerfStatRecord &record : records)
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "mtp" || record.value <= 0.0 ||
                !std::string_view(record.name).starts_with("device_generation_") ||
                record.phase == "initialization")
            {
                continue;
            }
            evidence.controller_observed = true;

            // Different lifecycle records use one of these three typed tag
            // keys.  Inspect only those keys so unrelated descriptive tags do
            // not turn otherwise complete evidence into an unknown policy.
            for (const std::string_view key : {
                     std::string_view("execution"),
                     std::string_view("execution_policy"),
                     std::string_view("policy")})
            {
                const auto tag = record.tags.find(std::string(key));
                if (tag == record.tags.end())
                    continue;

                const auto classified =
                    detail::classifyGenerationPolicyTag(tag->second);
                if (classified ==
                    ProductionDeviceGenerationPolicy::NativeConditionalParent)
                {
                    native_policy_observed = true;
                }
                else if (
                    classified == ProductionDeviceGenerationPolicy::
                                      HostScheduledCapturedTransactions)
                {
                    hosted_policy_observed = true;
                }
            }
        }

        if (!evidence.controller_observed)
        {
            evidence.policy = ProductionDeviceGenerationPolicy::NotObserved;
            evidence.certification_detail = "not_observed";
        }
        else if (native_policy_observed && hosted_policy_observed)
        {
            evidence.policy = ProductionDeviceGenerationPolicy::Inconsistent;
            evidence.certification_detail = "inconsistent_policy_evidence";
        }
        else if (native_policy_observed)
        {
            evidence.policy =
                ProductionDeviceGenerationPolicy::NativeConditionalParent;
            evidence.certification_detail = "native_conditional_parent";
        }
        else if (hosted_policy_observed)
        {
            evidence.policy = ProductionDeviceGenerationPolicy::
                HostScheduledCapturedTransactions;
            evidence.hosted_ticket_boundary_certified =
                detail::certifyHostedTicketBoundary(
                    records,
                    &evidence.certification_detail);
        }
        else
        {
            // Either no lifecycle record carried a policy tag or a future
            // spelling was not recognized. Both states are uncertified.
            evidence.policy = ProductionDeviceGenerationPolicy::Unclassified;
            evidence.certification_detail = "unclassified_policy_evidence";
        }
        return evidence;
    }
} // namespace llaminar2::test::parity
