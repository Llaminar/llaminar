/**
 * @file MTPGraphOwnerPlan.h
 * @brief Canonical runtime geometry and native-graph owner inventory for MTP.
 *
 * MTP setup creates several bounded graph directories in addition to the
 * complete condition and verifier forwards.  Historically the memory planner
 * priced a semantic device-generation fragment count while the orchestrator
 * independently sized the directories that can actually retain native graph
 * executables.  The two quantities are unrelated: fragments describe how one
 * parent transaction is composed, whereas cache slots are independent owners
 * that may remain resident together.
 *
 * This value object is the sole arithmetic authority for those directories.
 * DeviceGraphOrchestrator uses it to allocate every cache owner and the memory
 * planner charges only role-eligible native executables; metadata registry
 * capacities remain available independently of physical executable ownership.  Adding a new directory
 * therefore requires extending this type rather than copying another formula
 * into setup or preflight.
 */

#pragma once

#include "execution/config/RuntimeConfig.h"
#include "backends/GPUGraphMemoryContract.h"
#include "kernels/common/SparseQueryExecutionProgram.h"
#include "kernels/common/DeviceRequestRowRanges.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    /**
     * @brief Immutable MTP cache geometry and retained auxiliary graph owners.
     *
     * Runtime storage keeps a small minimum stochastic workspace even when MTP
     * execution is disabled.  Native executable admission, however, is zero
     * unless the model context retains MTP graph capacity.  Keeping both facts
     * in one object preserves existing non-MTP sampling geometry without
     * pretending that inactive MTP graph owners consume driver memory.
     */
    class MTPGraphOwnerPlan final
    {
    public:
        /** Minimum target rows needed for condition and bonus sampling scratch. */
        static constexpr int kMinimumStochasticTargetRows = 4;
        /** Minimum draft rows retained by scalar stochastic sampling scratch. */
        static constexpr int kMinimumStochasticDraftRows = 3;
        /** Full/chained/KV-only cache owners split by host/device token source. */
        static constexpr std::size_t kFixedSidecarGraphSlots = 6u;
        /** Standalone and device-generation-controlled verifier preparations. */
        static constexpr std::size_t kVerifierControlPolicyCount = 2u;
        /** Query-program-independent stochastic outcome and target-distribution families. */
        static constexpr std::size_t kSamplerHelperFamilyCount = 2u;
        /** Four standalone preparation nodes/request plus two fixed nodes. */
        static constexpr std::size_t kMaximumBoundedVerifierRequests =
            (GPUGraphMemoryContract::kBoundedFlatHelperMaxNodes - 2u) / 4u;
        /** One replaceable nonzero contiguous selector used by GPU catch-up. */
        static constexpr std::size_t
            kGenericTerminalHiddenGraphSlots = 1u;

        /**
         * @brief Compile retained native owner counts from frozen MTP execution ownership.
         * @param config Runtime execution and retained-capacity configuration.
         * @param family Required execution-owned verifier identities shared with admission.
         * @param parent_role Controller or pipeline follower owning complete native transactions.
         * @param query_programs Immutable model-geometry main-query programs; sidecars remain shared.
         * @throws std::invalid_argument For an unknown family/parent role or invalid query/draft geometry.
         * @throws std::overflow_error when a configured geometry cannot be
         *         represented by the runtime's integer or size types.
         */
        explicit MTPGraphOwnerPlan(const MTPRuntimeConfig &config, MTPVerifierGraphFamily family,
                                   MTPGenerationParentRole parent_role, std::size_t query_programs = 1u)
            : parent_role_(parent_role),
              retains_graph_capacity_(retainsMTPGraphCapacity(config)),
              query_programs_(query_programs),
              draft_depth_(resolveMTPMaximumDraftDepth(config)),
              request_capacity_(std::max(1, config.max_request_batch))
        {
            if (query_programs_ == 0 || query_programs_ > kSparseQueryExecutionProgramCount)
                throw std::invalid_argument("MTP graph inventory has an unknown query-program family");
            if (draft_depth_ <= 0)
            {
                throw std::invalid_argument(
                    "MTP graph owner plan requires a positive runtime draft capacity");
            }

            verifier_rows_per_request_ = checkedIntAdd(
                draft_depth_,
                1,
                "MTP verifier rows per request");
            flattened_target_rows_ = checkedIntMultiply(
                request_capacity_,
                verifier_rows_per_request_,
                "MTP flattened target rows");
            stochastic_target_rows_ = std::max(
                kMinimumStochasticTargetRows,
                flattened_target_rows_);
            stochastic_draft_rows_ = std::max(
                kMinimumStochasticDraftRows,
                checkedIntMultiply(
                    request_capacity_,
                    draft_depth_,
                    "MTP flattened draft rows"));

            // A hosted transaction embeds one publication child per retained
            // exact width. Replacing a single child's cache during setup would
            // leave earlier complete transactions with duplicated ownership.
            const auto family_widths = mtpVerifierGraphFamilyWidthCount(verifier_rows_per_request_, family);
            if (parent_role != MTPGenerationParentRole::Controller &&
                parent_role != MTPGenerationParentRole::PipelineFollower)
                throw std::invalid_argument("MTP generation parent role is unknown");
            if (!retains_graph_capacity_)
                return;
            sampler_helper_widths_ = family == MTPVerifierGraphFamily::ExactRows ? family_widths : 1u;
            sampler_helper_graph_slots_ = checkedSizeMultiply(kSamplerHelperFamilyCount,
                sampler_helper_widths_, "MTP row-specific sampler helper graph slots");
            generation_parent_widths_ = family == MTPVerifierGraphFamily::ExactRows &&
                parent_role == MTPGenerationParentRole::PipelineFollower ? family_widths : 1u;
            generation_parent_graph_slots_ = checkedSizeMultiply(
                query_programs_, generation_parent_widths_, "MTP complete generation parent graph slots");
            speculative_publication_widths_ = family == MTPVerifierGraphFamily::ExactRows
                ? family_widths : 1u;
            speculative_publication_graph_slots_ = checkedSizeMultiply(
                query_programs_, speculative_publication_widths_, "MTP speculative publication graph slots");
            // Direct output commits share scalar target slot zero. Only the
            // terminal controller owns sampling history; followers own none.
            // This flat helper is shared across widths and query programs.
            direct_token_history_graph_slots_ =
                parent_role == MTPGenerationParentRole::Controller ? 1u : 0u;
            controller_graph_slots_ = checkedSizeAdd(direct_token_history_graph_slots_,
                checkedSizeAdd(sampler_helper_graph_slots_,
                    checkedSizeAdd(generation_parent_graph_slots_, speculative_publication_graph_slots_,
                        "MTP parent and publication graph slots"), "MTP sampler and controller graph slots"),
                "MTP direct output history and controller graph slots");

            /* Row one has a dedicated KV-only cache. Rows [2, N] own one
             * additional cache apiece, matching MTPSidecarCaptureLayout. */
            sidecar_kv_only_batch_graph_slots_ =
                static_cast<std::size_t>(flattened_target_rows_ - 1);
            sidecar_graph_slots_ = checkedSizeAdd(
                kFixedSidecarGraphSlots,
                sidecar_kv_only_batch_graph_slots_,
                "MTP sidecar graph slots");

            const std::size_t requests =
                static_cast<std::size_t>(request_capacity_);
            const std::size_t flattened =
                static_cast<std::size_t>(flattened_target_rows_);
            const std::size_t shifted_prefill_slots = checkedSizeMultiply(
                requests,
                flattened,
                "MTP shifted-prefill terminal-hidden graph slots");
            terminal_hidden_contiguous_graph_slots_ = flattened;
            terminal_hidden_device_accepted_graph_slots_ = requests;
            terminal_hidden_request_terminal_graph_slots_ = requests;
            terminal_hidden_shifted_prefill_graph_slots_ =
                shifted_prefill_slots;
            terminal_hidden_graph_slots_ = checkedSizeAdd(
                checkedSizeAdd(
                    checkedSizeAdd(
                        kGenericTerminalHiddenGraphSlots,
                        terminal_hidden_contiguous_graph_slots_,
                        "MTP contiguous terminal-hidden graph slots"),
                    checkedSizeAdd(
                        terminal_hidden_device_accepted_graph_slots_,
                        terminal_hidden_request_terminal_graph_slots_,
                        "MTP request-indexed terminal-hidden graph slots"),
                    "MTP terminal-hidden graph slots"),
                terminal_hidden_shifted_prefill_graph_slots_,
                "MTP terminal-hidden graph slots");

            draft_publication_graph_slots_ =
                static_cast<std::size_t>(stochastic_draft_rows_);
            verifier_preparation_graph_slots_ = checkedSizeMultiply(
                kVerifierControlPolicyCount,
                checkedSizeMultiply(
                    requests,
                    static_cast<std::size_t>(verifier_rows_per_request_),
                    "MTP verifier preparation geometry"),
                "MTP verifier preparation graph slots");
        }

        /** @return Whether setup retains an MTP-capable native graph family. */
        [[nodiscard]] bool retainsGraphCapacity() const noexcept
        {
            return retains_graph_capacity_;
        }

        /** @return Largest draft depth whose runtime storage is preallocated. */
        [[nodiscard]] int draftDepth() const noexcept { return draft_depth_; }

        /** @return Maximum logical requests represented by one MTP transaction. */
        [[nodiscard]] int requestCapacity() const noexcept
        {
            return request_capacity_;
        }

        /** @return Draft comparison rows plus one bonus row for one request. */
        [[nodiscard]] int verifierRowsPerRequest() const noexcept
        {
            return verifier_rows_per_request_;
        }

        /** @return Flattened verifier/sidecar row capacity across requests. */
        [[nodiscard]] int flattenedTargetRows() const noexcept
        {
            return flattened_target_rows_;
        }

        /**
         * @brief Publish original retained verifier request partitions for GPU setup.
         * @return Exact request capacity and physical per-request row stride from this owner.
         * @throws std::logic_error When this owner retains no verifier graph family.
         * @details Generic prefill batching is deliberately not an input. The
         * constructor already checked R*M; this cold value reads no live count
         * and borrows no payload or independent allocation/accounting authority.
         */
        [[nodiscard]] DeviceRequestRowRanges retainedVerifierRequestRows() const
        {
            if (!retains_graph_capacity_)
                throw std::logic_error("Verifier workspace rows require a retained MTP graph owner");
            return DeviceRequestRowRanges::fullyActiveRequests(
                request_capacity_, verifier_rows_per_request_);
        }

        /** @return Target-distribution scratch rows retained by the arena. */
        [[nodiscard]] int stochasticTargetRows() const noexcept
        {
            return stochastic_target_rows_;
        }

        /** @return Draft-distribution scratch rows retained by the arena. */
        [[nodiscard]] int stochasticDraftRows() const noexcept
        {
            return stochastic_draft_rows_;
        }

        /** @return Independently resident MTP sidecar executable slots. */
        [[nodiscard]] std::size_t sidecarGraphSlots() const noexcept
        {
            return ownsControllerExecutables() ? sidecar_graph_slots_ : 0u;
        }

        /** @return Metadata registry capacity only; native executable eligibility is accounted by the role-gated family total. */
        [[nodiscard]] std::size_t sidecarKVOnlyBatchRegistryCapacity() const noexcept
        {
            return sidecar_kv_only_batch_graph_slots_;
        }

        /** @return Independently resident terminal-hidden publication slots. */
        [[nodiscard]] std::size_t terminalHiddenGraphSlots() const noexcept
        {
            return ownsControllerExecutables() ? terminal_hidden_graph_slots_ : 0u;
        }

        /** @return Metadata registry capacity only; native executable eligibility is accounted by the role-gated family total. */
        [[nodiscard]] std::size_t
        terminalHiddenContiguousRegistryCapacity() const noexcept
        {
            return terminal_hidden_contiguous_graph_slots_;
        }

        /** @return Replaceable nonzero-contiguous terminal-hidden graph slots. */
        [[nodiscard]] std::size_t
        genericTerminalHiddenGraphSlots() const noexcept
        {
            return retains_graph_capacity_ && ownsControllerExecutables()
                       ? kGenericTerminalHiddenGraphSlots
                       : 0u;
        }

        /** @return Metadata registry capacity only; native executable eligibility is accounted by the role-gated family total. */
        [[nodiscard]] std::size_t
        terminalHiddenDeviceAcceptedRegistryCapacity() const noexcept
        {
            return terminal_hidden_device_accepted_graph_slots_;
        }

        /** @return Metadata registry capacity only; native executable eligibility is accounted by the role-gated family total. */
        [[nodiscard]] std::size_t
        terminalHiddenRequestTerminalRegistryCapacity() const noexcept
        {
            return terminal_hidden_request_terminal_graph_slots_;
        }

        /** @return Metadata registry capacity only; native executable eligibility is accounted by the role-gated family total. */
        [[nodiscard]] std::size_t
        terminalHiddenShiftedPrefillRegistryCapacity() const noexcept
        {
            return terminal_hidden_shifted_prefill_graph_slots_;
        }

        /** @return Independently resident draft-token publication slots. */
        [[nodiscard]] std::size_t draftPublicationGraphSlots() const noexcept
        {
            return ownsControllerExecutables() ? draft_publication_graph_slots_ : 0u;
        }

        /** @return Independently resident grouped-verifier preparation slots. */
        [[nodiscard]] std::size_t verifierPreparationGraphSlots() const noexcept
        {
            return ownsControllerExecutables() ? verifier_preparation_graph_slots_ : 0u;
        }

        /** @return Metadata width capacity of shared sampler directories; followers retain no native sampler executable. */
        [[nodiscard]] std::size_t samplerHelperWidthCount() const noexcept
        {
            return sampler_helper_widths_;
        }

        /** @return Outcome plus target-distribution owners across every admitted helper width. */
        [[nodiscard]] std::size_t samplerHelperGraphSlots() const noexcept
        {
            return ownsControllerExecutables() ? sampler_helper_graph_slots_ : 0u;
        }

        /** @return Complete parent widths per query program; exact followers retain every member. */
        [[nodiscard]] std::size_t generationParentWidthCount() const noexcept
        {
            return generation_parent_widths_;
        }

        /** @return Independently retained complete parent graph owners across query programs. */
        [[nodiscard]] std::size_t generationParentGraphSlots() const noexcept
        {
            return generation_parent_graph_slots_;
        }

        /** @return Width identities per query program; bounded parents share one envelope. */
        [[nodiscard]] std::size_t speculativePublicationWidthCount() const noexcept
        {
            return speculative_publication_widths_;
        }

        /** @return Accepted-state publication owners, indexed by admitted width and query program. */
        [[nodiscard]] std::size_t speculativePublicationGraphSlots() const noexcept
        {
            return speculative_publication_graph_slots_;
        }

        /**
         * @return One retained flat direct-token history publisher for an MTP
         *         controller, zero for followers or absent graph capacity.
         *
         * The graph embeds only the existing scalar target/history pointers.
         * It owns no tensor payload, has one kernel, and is reused across
         * request/reset, query-program and verifier-width changes. Retained
         * capacity requires its admission even when the first request is Off.
         */
        [[nodiscard]] std::size_t directTokenHistoryGraphSlots() const noexcept
        {
            return direct_token_history_graph_slots_;
        }

        /** @return Controller owners including shared direct history, parent and publication programs. */
        [[nodiscard]] std::size_t controllerGraphSlots() const noexcept
        {
            // Each outcome/distribution width remains shared across query
            // programs; parent and publication directories retain their own owners.
            // Direct scalar history publication is one controller-only owner.
            return ownsControllerExecutables() ? controller_graph_slots_ :
                generation_parent_graph_slots_ + speculative_publication_graph_slots_;
        }

        /**
         * @brief Declare the verifier-preparation shape before native capture.
         * @param requests Logical requests represented by this cache slot.
         * @return Certified helper class, or the general class for larger batches.
         * @throws std::invalid_argument for a non-positive request count.
         *
         * Standalone preparation records at most three token operations and
         * one KV checkpoint per request, plus geometry and base-count copies.
         * Controlled preparation is smaller (one budget, two nodes/request).
         * This bound is format-independent; native capture enforces the final
         * shape as well, including nodes introduced by diagnostics.
         */
        [[nodiscard]] static GPUGraphExecutableClass
        verifierPreparationExecutableClass(int requests)
        {
            if (requests <= 0)
                throw std::invalid_argument("Verifier graph class requires positive requests");
            return static_cast<std::size_t>(requests) <= kMaximumBoundedVerifierRequests
                       ? GPUGraphExecutableClass::BoundedFlatHelper
                       : GPUGraphExecutableClass::General;
        }

        /**
         * @return Small native owners charged by the same contract as capture.
         *
         * Terminal selectors, individual draft publications and the scalar
         * direct-token history publisher stay bounded.
         * Verifier directories are indexed by request count: only the prefix
         * with a certified node bound receives the helper charge. Larger
         * request counts remain full first-class graphs, not an alternate path.
         */
        [[nodiscard]] std::size_t boundedHelperExecutableSlotCount() const
        {
            if (!retains_graph_capacity_ || !ownsControllerExecutables())
                return 0u;
            const auto requests = std::min(
                static_cast<std::size_t>(request_capacity_), kMaximumBoundedVerifierRequests);
            const auto verifier_slots = checkedSizeMultiply(
                kVerifierControlPolicyCount,
                checkedSizeMultiply(requests,
                    static_cast<std::size_t>(verifier_rows_per_request_),
                    "MTP bounded verifier geometries"),
                "MTP bounded verifier slots");
            return checkedSizeAdd(direct_token_history_graph_slots_,
                checkedSizeAdd(
                    checkedSizeAdd(terminal_hidden_graph_slots_,
                        draft_publication_graph_slots_, "MTP bounded publication slots"),
                    verifier_slots, "MTP bounded verifier and publication helper slots"),
                "MTP bounded helper slots");
        }

        /** @return General auxiliary owners, excluding every bounded helper. */
        [[nodiscard]] std::size_t generalAuxiliaryExecutableSlotCount() const
        {
            return auxiliaryExecutableSlotCount() - boundedHelperExecutableSlotCount();
        }

        /**
         * @brief Return every auxiliary native executable owner retained together.
         * @return Checked sum of the cache directories described above.
         */
        [[nodiscard]] std::size_t auxiliaryExecutableSlotCount() const
        {
            return checkedSizeAdd(
                checkedSizeAdd(
                    sidecarGraphSlots(),
                    terminalHiddenGraphSlots(),
                    "MTP auxiliary graph slots"),
                checkedSizeAdd(
                    draftPublicationGraphSlots(),
                    checkedSizeAdd(
                        verifierPreparationGraphSlots(),
                        controllerGraphSlots(),
                        "MTP auxiliary graph slots"),
                    "MTP auxiliary graph slots"),
                "MTP auxiliary graph slots");
        }

        /** @return Frozen native execution owner; followers never own predictor/sampler helpers. */
        [[nodiscard]] MTPGenerationParentRole parentRole() const noexcept { return parent_role_; }
        /** @return Whether this participant may materialize terminal predictor/controller helpers. */
        [[nodiscard]] bool ownsControllerExecutables() const noexcept
        { return parent_role_ == MTPGenerationParentRole::Controller; }
        /** @return Metadata directory capacity, independent of role-eligible native executable count. */
        [[nodiscard]] std::size_t verifierPreparationRegistryCapacity() const noexcept
        { return verifier_preparation_graph_slots_; }
        /** @return Metadata directory capacity preserved even when followers own no draft executable. */
        [[nodiscard]] std::size_t draftPublicationRegistryCapacity() const noexcept
        { return draft_publication_graph_slots_; }

    private:
        MTPGenerationParentRole parent_role_ = MTPGenerationParentRole::Controller; ///< Immutable native ownership.
        /** Add two size values without allowing admission arithmetic to wrap. */
        [[nodiscard]] static std::size_t checkedSizeAdd(
            std::size_t left,
            std::size_t right,
            const char *description)
        {
            if (right > std::numeric_limits<std::size_t>::max() - left)
            {
                throw std::overflow_error(
                    std::string(description) + " overflows size_t");
            }
            return left + right;
        }

        /** Multiply two size values without allowing admission arithmetic to wrap. */
        [[nodiscard]] static std::size_t checkedSizeMultiply(
            std::size_t left,
            std::size_t right,
            const char *description)
        {
            if (left != 0u &&
                right > std::numeric_limits<std::size_t>::max() / left)
            {
                throw std::overflow_error(
                    std::string(description) + " overflows size_t");
            }
            return left * right;
        }

        /** Add two positive runtime dimensions without signed overflow. */
        [[nodiscard]] static int checkedIntAdd(
            int left,
            int right,
            const char *description)
        {
            if (left > std::numeric_limits<int>::max() - right)
            {
                throw std::overflow_error(
                    std::string(description) + " overflows int");
            }
            return left + right;
        }

        /** Multiply two positive runtime dimensions without signed overflow. */
        [[nodiscard]] static int checkedIntMultiply(
            int left,
            int right,
            const char *description)
        {
            if (left <= 0 || right <= 0 ||
                left > std::numeric_limits<int>::max() / right)
            {
                throw std::overflow_error(
                    std::string(description) + " overflows int");
            }
            return left * right;
        }

        bool retains_graph_capacity_ = false;
        std::size_t query_programs_ = 1u; ///< Closed immutable main-query program inventory.
        int draft_depth_ = 1;
        int request_capacity_ = 1;
        int verifier_rows_per_request_ = 2;
        int flattened_target_rows_ = 2;
        int stochastic_target_rows_ = kMinimumStochasticTargetRows;
        int stochastic_draft_rows_ = kMinimumStochasticDraftRows;
        std::size_t sidecar_graph_slots_ = 0u;
        std::size_t sidecar_kv_only_batch_graph_slots_ = 0u;
        std::size_t terminal_hidden_graph_slots_ = 0u;
        std::size_t terminal_hidden_contiguous_graph_slots_ = 0u;
        std::size_t terminal_hidden_device_accepted_graph_slots_ = 0u;
        std::size_t terminal_hidden_request_terminal_graph_slots_ = 0u;
        std::size_t terminal_hidden_shifted_prefill_graph_slots_ = 0u;
        std::size_t draft_publication_graph_slots_ = 0u;
        std::size_t verifier_preparation_graph_slots_ = 0u;
        std::size_t speculative_publication_widths_ = 0u; ///< Immutable width directory per query program.
        std::size_t speculative_publication_graph_slots_ = 0u;
        std::size_t direct_token_history_graph_slots_ = 0u; ///< Shared scalar output-history publisher, controller only.
        std::size_t controller_graph_slots_ = 0u;
        std::size_t sampler_helper_widths_ = 0u; ///< Retained static row geometry per shared helper family.
        std::size_t sampler_helper_graph_slots_ = 0u;
        std::size_t generation_parent_widths_ = 0u; ///< Complete transaction family per query program.
        std::size_t generation_parent_graph_slots_ = 0u;
    };
} // namespace llaminar2
