/**
 * @file ServingWindowDecodeBridge.h
 * @brief ENGINE-OWNED DECODE BRIDGE (S3.2b continuous-serving integration).
 *
 * The row owner (the serving engine) advances one row's exact intrinsic
 * window through the TYPED ABI. The bridge OWNS the command factory (shared
 * ownership — a temporary/short-lived factory cannot be bound), mints the
 * execution command from the row's OWN progress (full issuance pins verified
 * at mint), invokes the runner's decodeServingWindow (MANDATORY ingress
 * before the private virtual implementation; the production implementation
 * performs the AUTHENTIC completion), and BINDS THE RESULT BACK TO THIS CALL:
 * correspondence compares the result's COMPLETE invocation binding AND the
 * COMPLETE window intent (authority UID, transaction epoch, law admission
 * identity + law issue identity, slot, generation, grant id, serving
 * instance, window start, quantum, response limit, seed and the terminal
 * flags) against the command captured BEFORE dispatch. A cross-operation or
 * intent-mismatched result is REJECTED with a precise diagnostic — the bridge
 * neither re-completes nor synthesizes a replacement.
 *
 * STATE CONTRACT (explicit): the ORIGINAL issuance identity (authority UID,
 * grant id and live state) is captured before dispatch. A pure failure is a
 * STATE VIOLATION only when the original issuance was consumed/replaced (the
 * authority UID no longer matches, the grant is no longer live/outstanding,
 * or the grant id changed) AND the command was valid at dispatch (a command
 * invalid upfront is an ingress rejection, never attributed to the
 * implementation). The diagnostic reports the actual state; nothing is
 * rolled back.
 *
 * SCOPE (narrowed honestly): typed forwarding + enclosing ingress,
 * correspondence and state checks on the accepted ABI mint/ingress machinery.
 * It is NOT production grouped decode, NOT engine lifecycle and NOT
 * production output publication. It is the retained DecodeWindow seam for
 * GPU serving; CPU continuous batching executes through OrchestrationRunner's
 * slot-addressed serving rows.
 */

#pragma once

#include "execution/runner/IOrchestrationRunner.h"
#include "execution/serving/ServingWindowDecode.h"

#include <memory>

namespace llaminar2
{
    /** @brief CHECKED ABI DECODE (shared by the bridge AND the engine): the
     *         command (minted by the row owner) is dispatched through the
     *         mandatory ingress; the result is bound back to THIS call (full
     *         invocation + window-intent correspondence) and the post-state
     *         contract is assessed (violations reported, never rolled back,
     *         re-completed or synthesized). */
    [[nodiscard]] inline ServingWindowDecodeResult abiDecodeChecked(
        IOrchestrationRunner &runner, const ServingWindowDecodeInput &command)
    {
        // GUARD before any dereference: a null/moved-from law node is an invalid
        // command (precise diagnostic BEFORE ingress, never a dereference).
        if (command.lawNode() == nullptr)
            return ServingWindowDecodeResult::failure("command has no law node (invalid)");
        const auto expected_law_issue = command.lawNode()->issueIdentity();
        const auto expected_law_admission = command.lawNode()->admissionIdentity();
        const auto expected_window = command.window_start;
        const auto expected_quantum = command.authority.current().window.quantum;
        const auto expected_epoch = command.window_epoch;
        const auto expected_limit = command.authority.current().window.response_limit;
        const bool valid_before = command.valid();
        const auto observed = runner.decodeServingWindow(command);
        const bool original_consumed =
            command.authority.authorityIdentity() != command.expected_authority ||
            !command.authority.mayExecute() ||
            command.authority.current().grant_id != command.expected_grant_id;
        if (observed.ok() || observed.cancelled())
        {
            if (observed.authorityIdentity() != command.expected_authority ||
                observed.issuanceNonce() != command.expected_nonce ||
                observed.transactionEpoch() != command.transaction_epoch ||
                observed.lawIssueIdentity() != expected_law_issue ||
                observed.lawAdmissionIdentity() != expected_law_admission ||
                observed.slot() != command.expected_slot ||
                observed.generation() != command.expected_generation ||
                observed.grantId() != command.expected_grant_id ||
                observed.servingInstance() != command.expected_instance ||
                observed.windowStart() != expected_window ||
                observed.windowEpoch() != expected_epoch ||
                observed.windowQuantum() != expected_quantum ||
                observed.responseLimit() != expected_limit ||
                observed.seed() != command.seed ||
                observed.finalPartial() != command.final_partial ||
                observed.firstPrefillCarry() != command.first_prefill_carry)
            {
                const std::string base = observed.cancelled()
                    ? "cancellation does not correspond to this call "
                      "(cross-operation or intent mismatch"
                    : "result does not correspond to this call "
                      "(cross-operation or intent mismatch";
                return ServingWindowDecodeResult::failure(
                    original_consumed
                        ? base + "; original grant consumed (state violation))"
                        : base + ")");
            }
            return observed;
        }
        if (valid_before && original_consumed)
            return ServingWindowDecodeResult::failure(
                "grant consumed without a completion (state violation)");
        return observed;
    }

    /** @brief Row-owner bridge: one row window through the typed ABI. */
    class ServingWindowDecodeBridge
    {
    public:
        /** @brief SHARED ownership of the command factory: a temporary or
         *         stack-only factory CANNOT be bound (a shared owner is
         *         required — the bridge and the engine co-own it). */
        explicit ServingWindowDecodeBridge(
            std::shared_ptr<ServingWindowCommandFactory> factory,
            IOrchestrationRunner &runner)
            : factory_(std::move(factory)), runner_(runner)
        {
            if (!factory_)
                throw std::logic_error(
                    "Decode bridge requires an owned command factory (no temporary)");
        }
        ServingWindowDecodeBridge(const ServingWindowDecodeBridge &) = delete;
        ServingWindowDecodeBridge &operator=(const ServingWindowDecodeBridge &) = delete;
        ServingWindowDecodeBridge(ServingWindowDecodeBridge &&) = delete;
        ServingWindowDecodeBridge &operator=(ServingWindowDecodeBridge &&) = delete;

        /** @brief Advance one row by its exact intrinsic window (checked decode). */
        [[nodiscard]] ServingWindowDecodeResult decodeWindow(
            const ServingRowProgress &progress) const
        {
            auto command = factory_->forRow(progress); // engine-owned position; pins verified
            return abiDecodeChecked(runner_, command);
        }

    private:
        std::shared_ptr<ServingWindowCommandFactory> factory_; ///< Shared ownership.
        IOrchestrationRunner &runner_;
    };
} // namespace llaminar2
