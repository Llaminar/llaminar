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
 * planner charges the resulting retained slot count.  Adding a new directory
 * therefore requires extending this type rather than copying another formula
 * into setup or preflight.
 */

#pragma once

#include "execution/config/RuntimeConfig.h"

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
        /** Spec publication, generation parent, serial outcome, target distribution. */
        static constexpr std::size_t kControllerGraphSlots = 4u;
        /** One replaceable nonzero contiguous selector used by GPU catch-up. */
        static constexpr std::size_t
            kGenericTerminalHiddenGraphSlots = 1u;

        /**
         * @brief Compile the exact bounded owner plan from frozen MTP policy.
         * @param config Runtime execution and retained-capacity configuration.
         * @throws std::overflow_error when a configured geometry cannot be
         *         represented by the runtime's integer or size types.
         */
        explicit MTPGraphOwnerPlan(const MTPRuntimeConfig &config)
            : retains_graph_capacity_(retainsMTPGraphCapacity(config)),
              draft_depth_(resolveMTPMaximumDraftDepth(config)),
              request_capacity_(std::max(1, config.max_request_batch))
        {
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

            if (!retains_graph_capacity_)
                return;

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
            return sidecar_graph_slots_;
        }

        /** @return Shape-indexed multi-row KV-only sidecar cache slots. */
        [[nodiscard]] std::size_t sidecarKVOnlyBatchGraphSlots() const noexcept
        {
            return sidecar_kv_only_batch_graph_slots_;
        }

        /** @return Independently resident terminal-hidden publication slots. */
        [[nodiscard]] std::size_t terminalHiddenGraphSlots() const noexcept
        {
            return terminal_hidden_graph_slots_;
        }

        /** @return Fixed-contiguous terminal-hidden graph slots. */
        [[nodiscard]] std::size_t
        terminalHiddenContiguousGraphSlots() const noexcept
        {
            return terminal_hidden_contiguous_graph_slots_;
        }

        /** @return Replaceable nonzero-contiguous terminal-hidden graph slots. */
        [[nodiscard]] std::size_t
        genericTerminalHiddenGraphSlots() const noexcept
        {
            return retains_graph_capacity_
                       ? kGenericTerminalHiddenGraphSlots
                       : 0u;
        }

        /** @return Device-accepted-row graph slots indexed by request count. */
        [[nodiscard]] std::size_t
        terminalHiddenDeviceAcceptedGraphSlots() const noexcept
        {
            return terminal_hidden_device_accepted_graph_slots_;
        }

        /** @return Request-terminal graph slots indexed by request count. */
        [[nodiscard]] std::size_t
        terminalHiddenRequestTerminalGraphSlots() const noexcept
        {
            return terminal_hidden_request_terminal_graph_slots_;
        }

        /** @return Shifted-prefill graph slots indexed by request and row count. */
        [[nodiscard]] std::size_t
        terminalHiddenShiftedPrefillGraphSlots() const noexcept
        {
            return terminal_hidden_shifted_prefill_graph_slots_;
        }

        /** @return Independently resident draft-token publication slots. */
        [[nodiscard]] std::size_t draftPublicationGraphSlots() const noexcept
        {
            return draft_publication_graph_slots_;
        }

        /** @return Independently resident grouped-verifier preparation slots. */
        [[nodiscard]] std::size_t verifierPreparationGraphSlots() const noexcept
        {
            return verifier_preparation_graph_slots_;
        }

        /** @return Fixed singleton controller/publication executable slots. */
        [[nodiscard]] std::size_t controllerGraphSlots() const noexcept
        {
            return retains_graph_capacity_ ? kControllerGraphSlots : 0u;
        }

        /**
         * @brief Return every auxiliary native executable owner retained together.
         * @return Checked sum of the cache directories described above.
         */
        [[nodiscard]] std::size_t auxiliaryExecutableSlotCount() const
        {
            return checkedSizeAdd(
                checkedSizeAdd(
                    sidecar_graph_slots_,
                    terminal_hidden_graph_slots_,
                    "MTP auxiliary graph slots"),
                checkedSizeAdd(
                    draft_publication_graph_slots_,
                    checkedSizeAdd(
                        verifier_preparation_graph_slots_,
                        controllerGraphSlots(),
                        "MTP auxiliary graph slots"),
                    "MTP auxiliary graph slots"),
                "MTP auxiliary graph slots");
        }

    private:
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
    };
} // namespace llaminar2
