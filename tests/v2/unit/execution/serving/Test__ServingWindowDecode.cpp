/**
 * @file Test__ServingWindowDecode.cpp
 * @brief Per-row serving-window decode ABI contract (device-free).
 *
 * The input is an EXECUTION COMMAND bound to the OWNER-ISSUED one-window
 * grant (non-copyable authority, live query), the FULL binding/issuance
 * identity (authority identity, generation, grant id, issuance nonce), the
 * ADMITTED per-row sampling table (PER-ISSUE move-only RAII binding with
 * cross-admission refusal + capacity + clean pre-publication failure), the
 * resolved seed (§7.2), the transaction epoch (§8), the row-owner window
 * start and the terminal flags with INTRINSIC-GRID endpoint semantics.
 * The public entry is non-virtual (mandatory ingress before the private
 * virtual dispatch). Completion is an AUTHENTIC FAILURE-ATOMIC COMMIT:
 * completeWindow verifies the full binding, fails without consuming on an
 * invalid produced, resolves exactly once, and never accepts a
 * foreign/stale/repeated completion.
 */
#include <gtest/gtest.h>

#include "execution/runner/IOrchestrationRunner.h"
#include "execution/serving/ServingWindowDecode.h"
#include "execution/serving/ServingWindowDecodeBridge.h"
#include "mocks/MockOrchestrationRunner.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

namespace llaminar2 {
        struct AuthorityIdentityExhaustionProbe
        {
            static void arm()
            {
                WindowGrantAuthority::next_authority_identity_.store(
                    std::numeric_limits<std::uint64_t>::max() - 1);
                WindowGrantAuthority::next_authority_identity_exhausted_.store(false);
            }
            static bool sticky()
            {
                return WindowGrantAuthority::next_authority_identity_exhausted_.load();
            }
            static void clear()
            {
                WindowGrantAuthority::next_authority_identity_exhausted_.store(false);
                WindowGrantAuthority::next_authority_identity_.store(0);
            }
        };
        struct AdmissionIdentityExhaustionProbe
        {
            static void arm()
            {
                ServingLawAdmission::next_admission_identity_.store(
                    std::numeric_limits<std::uint64_t>::max() - 1);
                ServingLawAdmission::next_admission_identity_exhausted_.store(false);
            }
            static bool sticky()
            {
                return ServingLawAdmission::next_admission_identity_exhausted_.load();
            }
            static void clear()
            {
                ServingLawAdmission::next_admission_identity_exhausted_.store(false);
                ServingLawAdmission::next_admission_identity_.store(0);
            }
        };
} // namespace llaminar2

namespace llaminar2::test
{
    namespace
    {
        /** @brief Owner-issued backed grant authority (one outstanding grant,
         *         destination backed, quantum 16 / limit 32). */
        WindowGrantAuthority makeGrantAuthority(WindowGrant grant = WindowGrant{})
        {
            if (grant.grant_id == 0)
            {
                grant.grant_id = 7;
                grant.slot = 0;
                grant.generation = 1;
                grant.window = IntrinsicWindowPlan{16, 32};
                grant.destination_reservation = 9;
                grant.serving_instance = 1;
                grant.state = GrantState::Outstanding;
            }
            WindowGrantAuthority authority(1);
            authority.issue(grant);
            return authority;
        }

        /** @brief Live admitted law binding (epoch 3, token 11, schema 2, cap 32). */
        std::pair<std::shared_ptr<ServingLawAdmission>, ServingLawBinding> makeLaw()
        {
            auto admission = std::make_shared<ServingLawAdmission>(32);
            auto binding = admission->issue(3, 11, 2, 32);
            return {admission, std::move(binding)};
        }

        ServingWindowDecodeInput makeInput(WindowGrantAuthority &authority,
                                           const ServingLawBinding &law,
                                           std::uint64_t window_start = 0,
                                           bool final_partial = false,
                                           bool first_prefill_carry = false)
        {
            return ServingWindowDecodeInput(authority, law, 5, 4,
                                            authority.authorityIdentity(),
                                            authority.current().generation,
                                            authority.current().grant_id,
                                            authority.issuanceNonce(), 1, window_start,
                                            final_partial, first_prefill_carry);
        }

        /** @brief Subclass whose implementation completes for ANY valid input:
         *         proves mandatory ingress precedes dispatch and that a
         *         completing override runs against the REAL authority. */
        class CompletingRunner : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                return completeWindow(request.authority, request, 16,
                                      WindowTermination::Continue);
            }
        };
    }




    /** @brief The base (unimplemented) entry is fail-closed for VALID input. */
    TEST(ServingWindowDecode, DefaultIsFailClosedForValidInput)
    {
        MockOrchestrationRunner runner;
        EXPECT_FALSE(runner.supportsServingWindowDecode());
        auto authority = makeGrantAuthority();
        auto [admission, law] = makeLaw();
        auto input = makeInput(authority, law);
        EXPECT_TRUE(input.valid());
        const auto result = runner.decodeServingWindow(input);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.error().find("unsupported"), std::string::npos);
        EXPECT_EQ(result.produced(), 0u);
        EXPECT_FALSE(result.grantConsumed());
        EXPECT_EQ(result.servingInstance(), 0u);
    }

    /** @brief MANDATORY ingress: precise named diagnostics for every defect. */
    TEST(ServingWindowDecode, EntryRejectsInvalidInputWithPreciseDiagnostics)
    {
        MockOrchestrationRunner runner;
        const auto expectReject = [&](ServingWindowDecodeInput &&input, const char *needle) {
            const auto result = runner.decodeServingWindow(input);
            EXPECT_FALSE(result.ok());
            EXPECT_EQ(result.produced(), 0u);
            EXPECT_NE(result.error().find(needle), std::string::npos) << result.error();
        };
        {
            auto authority = makeGrantAuthority();
            authority.resolve();
            auto [admission, law] = makeLaw();
            expectReject(makeInput(authority, law), "grant not outstanding/backed");
        }
        {
            WindowGrant g;
            g.grant_id = 8;
            g.slot = 0;
            g.generation = 0; // generation zero is refused
            g.window = IntrinsicWindowPlan{16, 32};
            g.destination_reservation = 9;
            g.serving_instance = 1;
            g.state = GrantState::Outstanding;
            WindowGrantAuthority authority(1);
            authority.issue(g);
            auto [admission, law] = makeLaw();
            expectReject(makeInput(authority, law), "generation zero");
        }
        {
            auto authority = makeGrantAuthority();
            auto [admission, law] = makeLaw();
            admission->release(law);
            expectReject(makeInput(authority, law), "admitted law binding not live");
        }
        {
            auto authority = makeGrantAuthority();
            auto [admission, law] = makeLaw();
            ServingWindowDecodeInput input(authority, law, 0, 4,
                                           authority.authorityIdentity(), 1,
                                           authority.current().grant_id,
                                           authority.issuanceNonce(), 1, 0, false, false);
            expectReject(std::move(input), "resolved seed zero");
        }
        {
            auto authority = makeGrantAuthority();
            auto [admission, law] = makeLaw();
            ServingWindowDecodeInput input(authority, law, 5, 0,
                                           authority.authorityIdentity(), 1,
                                           authority.current().grant_id,
                                           authority.issuanceNonce(), 1, 0, false, false);
            expectReject(std::move(input), "transaction epoch zero");
        }
        {
            auto authority = makeGrantAuthority();
            auto [admission, law] = makeLaw();
            ServingWindowDecodeInput input(authority, law, 5, 4,
                                           authority.authorityIdentity(), 99,
                                           authority.current().grant_id,
                                           authority.issuanceNonce(), 1, 0, false, false);
            expectReject(std::move(input), "stale issuance");
        }
        {
            auto authority = makeGrantAuthority();
            auto [admission, law] = makeLaw();
            ServingWindowDecodeInput input(authority, law, 5, 4,
                                           authority.authorityIdentity(), 1,
                                           authority.current().grant_id,
                                           authority.issuanceNonce(), 1, 32, false, false);
            expectReject(std::move(input), "window start at/after response limit");
        }
        {
            // c31 / W16 / L40: the CURRENT intrinsic window ends at the next grid
            // boundary (32) — NOT terminal (the old c+W test wrongly demanded
            // final_partial); with grid semantics this is a valid continuation.
            WindowGrantAuthority a(1);
            WindowGrant g;
            g.grant_id = 9;
            g.slot = 0;
            g.generation = 1;
            g.window = IntrinsicWindowPlan{16, 40};
            g.destination_reservation = 9;
            g.serving_instance = 1;
            g.state = GrantState::Outstanding;
            a.issue(g);
            auto [admission, law] = makeLaw();
            ServingWindowDecodeInput input(a, law, 5, 4, a.authorityIdentity(), 1, 9,
                                           a.issuanceNonce(), 1, 31, false, false);
            EXPECT_TRUE(input.valid()); // accepted without final_partial (grid)
            const auto result = runner.decodeServingWindow(std::move(input));
            EXPECT_NE(result.error().find("unsupported"), std::string::npos);
        }
    }

    /** @brief Stale issuance: resolve -> reissue (same generation) is rejected by
     *         the full pin (no reuse across replacement). */
    TEST(ServingWindowDecode, ResolveThenReissueIsStaleAndRejected)
    {
        MockOrchestrationRunner runner;
        WindowGrantAuthority authority(1);
        WindowGrant g;
        g.grant_id = 7;
        g.slot = 0;
        g.generation = 1;
        g.window = IntrinsicWindowPlan{16, 32};
        g.destination_reservation = 9;
        g.serving_instance = 1;
        g.state = GrantState::Outstanding;
        authority.issue(g);
        authority.resolve();
        WindowGrant g2 = g;
        g2.grant_id = 8; // grant IDs are never reused; SAME generation, new issuance
        g2.state = GrantState::Outstanding;
        authority.issue(g2);
        auto [admission, law] = makeLaw();
        // The stale caller still pins generation 1 / grant 7 / nonce 1.
        ServingWindowDecodeInput stale(authority, std::move(law), 5, 4,
                                       authority.authorityIdentity(), 1, 7, 1, 1, 0, false,
                                       false);
        const auto result = runner.decodeServingWindow(stale);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.error().find("stale issuance"), std::string::npos);
        auto [admission2, law2] = makeLaw();
        auto current = makeInput(authority, law2);
        EXPECT_TRUE(current.valid());
    }

    /** @brief PER-ISSUE law binding: multi-handle isolation, cross-admission
     *         refusal, capacity enforcement, clean pre-publication failure for
     *         a non-shared admission, and exact retirement. */
    TEST(ServingWindowDecode, LawBindingsArePerIssueWithExactRetirement)
    {
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto a = admission->issue(1, 11, 2, 32);
        auto b = admission->issue(2, 22, 2, 32);
        EXPECT_TRUE(a.valid());
        EXPECT_TRUE(b.valid());
        EXPECT_EQ(admission->liveCount(), 2u);
        admission->release(a);
        EXPECT_FALSE(a.valid());
        EXPECT_TRUE(b.valid());
        auto c = admission->issue(3, 33, 2, 32);
        EXPECT_TRUE(c.valid());
        EXPECT_FALSE(a.valid());
        EXPECT_EQ(admission->liveCount(), 2u);
        // Cross-admission release is REFUSED (both owners intact).
        auto other = std::make_shared<ServingLawAdmission>(32);
        auto other_binding = other->issue(4, 44, 2, 32);
        EXPECT_THROW(other->release(b), std::logic_error);
        EXPECT_TRUE(b.valid());
        EXPECT_EQ(admission->liveCount(), 2u);
        EXPECT_EQ(other->liveCount(), 1u);
        // Capacity enforced.
        EXPECT_THROW(admission->issue(5, 55, 2, 64), std::logic_error);
        // Non-shared (stack) admission fails cleanly BEFORE publication.
        ServingLawAdmission stack(32);
        EXPECT_THROW(stack.issue(6, 66, 2, 32), std::logic_error);
        // Move: source retires, destination live (single live handle).
        ServingLawBinding moved{std::move(b)};
        EXPECT_FALSE(b.valid());
        EXPECT_TRUE(moved.valid());
        // Destruction retires exactly once.
        auto hold = std::make_shared<ServingLawBinding>(std::move(c));
        EXPECT_EQ(admission->liveCount(), 2u);
        hold.reset();
        EXPECT_EQ(admission->liveCount(), 1u);
        (void)other_binding;
    }

    /** @brief Authority moves carry the FULL identity (nonce + authority id):
     *         moving into a resolved destination keeps the old generation/nonce
     *         from matching a different grant. */
    TEST(ServingWindowDecode, AuthorityMoveCarriesFullIssuanceIdentity)
    {
        auto authority = makeGrantAuthority();
        const auto nonce = authority.issuanceNonce();
        const auto identity = authority.authorityIdentity();
        WindowGrantAuthority moved{std::move(authority)};
        EXPECT_EQ(moved.issuanceNonce(), nonce);
        EXPECT_EQ(moved.authorityIdentity(), identity);
        EXPECT_EQ(authority.issuanceNonce(), 0u); // moved-from is dead
        EXPECT_NE(authority.authorityIdentity(), identity); // never rebinds to a live id
    }

    /** @brief A COPIED Outstanding grant record does NOT survive the real
     *         resolution (the ABI queries the live authority). */
    TEST(ServingWindowDecode, CopiedOutstandingStateDoesNotSurviveResolution)
    {
        MockOrchestrationRunner runner;
        auto authority = makeGrantAuthority();
        const WindowGrant copied = authority.current();
        EXPECT_EQ(copied.state, GrantState::Outstanding);
        authority.resolve();
        auto [admission, law] = makeLaw();
        auto input = makeInput(authority, law);
        const auto result = runner.decodeServingWindow(input);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.error().find("grant not outstanding/backed"), std::string::npos);
    }

    /** @brief MANDATORY INGRESS + authentic completion: an override cannot
     *         bypass the entry checks, and completion resolves the REAL
     *         authority exactly once. */
    TEST(ServingWindowDecode, OverrideCannotBypassIngressAndCompletionIsAuthentic)
    {
        CompletingRunner runner;
        {
            auto authority = makeGrantAuthority();
            auto [admission, law] = makeLaw();
            auto input = makeInput(authority, law);
            const auto result = runner.decodeServingWindow(input);
            EXPECT_TRUE(result.ok());
            EXPECT_EQ(result.produced(), 16u);
            EXPECT_TRUE(result.grantConsumed());
            EXPECT_EQ(result.grantId(), 7u);
            EXPECT_FALSE(authority.mayExecute()); // resolved by completion
        }
        {
            auto authority = makeGrantAuthority();
            authority.resolve();
            auto [admission, law] = makeLaw();
            auto input = makeInput(authority, law);
            const auto result = runner.decodeServingWindow(input);
            EXPECT_FALSE(result.ok()); // ingress rejects BEFORE the override runs
            EXPECT_NE(result.error().find("grant not outstanding/backed"), std::string::npos);
        }
    }

    /** @brief AUTHENTIC FAILURE-ATOMIC COMMIT: invalid produced does NOT consume
     *         the grant; repeated/foreign/stale completion cannot be minted;
     *         the bound is the INTRINSIC-GRID endpoint. */
    TEST(ServingWindowDecode, CompletionIsAuthenticFailureAtomicAndGridBound)
    {
        auto authority = makeGrantAuthority();
        auto [admission, law] = makeLaw();
        auto input = makeInput(authority, law);
        // First commit succeeds with identity from the authority's grant.
        const auto ok = completeWindow(authority, input, 16, WindowTermination::Continue);
        EXPECT_TRUE(ok.ok());
        EXPECT_EQ(ok.produced(), 16u);
        EXPECT_EQ(ok.servingInstance(), 1u);
        EXPECT_EQ(ok.generation(), 1u);
        EXPECT_EQ(ok.grantId(), 7u);
        EXPECT_TRUE(ok.grantConsumed());
        EXPECT_EQ(ok.termination(), WindowTermination::Continue);
        // Repeated completion on the same authority throws (already resolved).
        EXPECT_THROW(completeWindow(authority, input, 16, WindowTermination::Continue),
                     std::logic_error);
        // FAILURE-ATOMIC refusals do NOT consume (grant stays mayExecute):
        {
            auto a2 = makeGrantAuthority();
            auto [ad2, law2] = makeLaw();
            auto in2 = makeInput(a2, std::move(law2));
            EXPECT_THROW(completeWindow(a2, in2, 0, WindowTermination::Continue),
                         std::logic_error);
            EXPECT_TRUE(a2.mayExecute()); // NOT consumed
            // The same command then completes correctly (state-after-refusal).
            const auto after = completeWindow(a2, in2, 16, WindowTermination::Continue);
            EXPECT_TRUE(after.ok());
        }
        {
            // Foreign authority completion: a command bound to a DIFFERENT
            // authority is refused (never consumes).
            auto a3 = makeGrantAuthority();
            auto a4 = makeGrantAuthority();
            auto [ad3, law3] = makeLaw();
            auto in3 = makeInput(a3, std::move(law3));
            EXPECT_THROW(completeWindow(a4, in3, 16, WindowTermination::Continue),
                         std::logic_error);
            EXPECT_TRUE(a3.mayExecute());
            EXPECT_TRUE(a4.mayExecute());
        }
        // INTRINSIC-GRID bound: W16/L32 at start 1 -> 15 (not 16) — the current
        // intrinsic window ends at the next grid boundary.
        {
            auto a5 = makeGrantAuthority();
            auto [ad5, law5] = makeLaw();
            auto in5 = makeInput(a5, std::move(law5), 1);
            EXPECT_EQ(servingWindowBound(1, 16, 32), 15u);
            EXPECT_THROW(completeWindow(a5, in5, 16, WindowTermination::Continue),
                         std::logic_error);
            EXPECT_TRUE(a5.mayExecute()); // refused without consuming
            const auto after = completeWindow(a5, in5, 15, WindowTermination::Continue);
            EXPECT_TRUE(after.ok());
        }
        // Overflow safety: start near limit (c < L), W-c%W cannot overflow.
        EXPECT_EQ(servingWindowBound(31, 16, 32), 1u);
        EXPECT_EQ(servingWindowBound(0, 16, 5), 5u); // partial-final grid at c0
        // Failure: non-empty diagnostic, zero output, no identity, no consumption.
        const auto fail = ServingWindowDecodeResult::failure("diagnostic");
        EXPECT_FALSE(fail.ok());
        EXPECT_EQ(fail.produced(), 0u);
        EXPECT_EQ(fail.servingInstance(), 0u);
        EXPECT_FALSE(fail.grantConsumed());
        EXPECT_EQ(fail.termination(), WindowTermination::Continue);
        EXPECT_THROW(ServingWindowDecodeResult::failure(std::string{}), std::logic_error);
    }

    /** @brief Bounded law admission: the admitted live-binding capacity is
     *         enforced (full admission refuses a new binding). */
    TEST(ServingWindowDecode, LawAdmissionIsBounded)
    {
        auto admission = std::make_shared<ServingLawAdmission>(1); // capacity 1
        auto a = admission->issue(1, 11, 2, 1);
        EXPECT_THROW(admission->issue(2, 22, 2, 1), std::logic_error); // full
        EXPECT_EQ(admission->liveCount(), 1u);
        admission->release(a);
        auto b = admission->issue(3, 33, 2, 1); // capacity freed -> ok
        EXPECT_TRUE(b.valid());
    }

    /** @brief GRID terminal helper: no start+W anywhere; near-UINT64_MAX does
     *         not wrap; c31/W16/L40 is NOT the terminal window. */
    TEST(ServingWindowDecode, GridTerminalIsOverflowSafe)
    {
        EXPECT_FALSE(servingWindowIsTerminal(31, 16, 40)); // next grid 32 < 40
        EXPECT_TRUE(servingWindowIsTerminal(32, 16, 40));  // grid end 48 >= 40? -> terminal
        EXPECT_TRUE(servingWindowIsTerminal(3, 16, 4));    // grid end 16 >= 4 (partial final)
        const auto near_max = std::numeric_limits<std::uint64_t>::max();
        // (near_max-2) % 16 == 13 -> grid_remaining 3 >= (limit - start) == 1
        EXPECT_TRUE(servingWindowIsTerminal(near_max - 2, 16, near_max - 1));
        EXPECT_EQ(servingWindowBound(near_max - 2, 16, near_max - 1), 1u); // no wrap
    }

    /** @brief EXACT-WINDOW completion: a normal window must be FULL (produced ==
     *         the intrinsic-grid bound); under-production (e.g., 1/Continue for a
     *         bound of 16) is REFUSED and does NOT consume. */
    TEST(ServingWindowDecode, CompletionRequiresFullWindow)
    {
        auto authority = makeGrantAuthority();
        auto [admission, law] = makeLaw();
        auto input = makeInput(authority, law);
        EXPECT_THROW(completeWindow(authority, input, 1, WindowTermination::Continue),
                     std::logic_error);
        EXPECT_TRUE(authority.mayExecute()); // not consumed
        const auto after = completeWindow(authority, input, 16, WindowTermination::Continue);
        EXPECT_TRUE(after.ok());
        EXPECT_EQ(after.produced(), 16u);
        EXPECT_THROW(completeWindow(authority, input, 16, WindowTermination::Continue),
                     std::logic_error); // already resolved
    }

    /** @brief MOVED-FROM AUTHORITY LIFECYCLE: post-move issue is refused (dead
     *         identity), move-assign carries the identity and the moved-from
     *         side cannot issue again. */
    TEST(ServingWindowDecode, MovedFromAuthorityCannotIssueAgain)
    {
        auto src = makeGrantAuthority();
        const auto identity = src.authorityIdentity();
        WindowGrantAuthority moved{std::move(src)};
        EXPECT_EQ(moved.authorityIdentity(), identity);
        // Post-move issue on the moved-from side is refused.
        WindowGrant g;
        g.grant_id = 20;
        g.slot = 0;
        g.generation = 1;
        g.window = IntrinsicWindowPlan{16, 32};
        g.destination_reservation = 9;
        g.serving_instance = 1;
        g.state = GrantState::Outstanding;
        EXPECT_THROW(src.issue(g), std::logic_error);
        // Move-ASSIGN: identity transfers; the moved-from side is dead.
        auto src2 = makeGrantAuthority();
        const auto id2 = src2.authorityIdentity();
        WindowGrantAuthority dst(1);
        dst = std::move(src2);
        EXPECT_EQ(dst.authorityIdentity(), id2);
        EXPECT_THROW(src2.issue(g), std::logic_error); // moved-from cannot issue
    }

    /** @brief ROW-OWNER COMMAND FACTORY: the engine construction site — the
     *         command is minted for the row's OWN committed position (engine-
     *         owned; the factory carries the pins), and the mint is admitted. */
    TEST(ServingWindowDecode, RowOwnerFactoryMintsEngineOwnedCommand)
    {
        auto authority = makeGrantAuthority();
        auto [admission, law] = makeLaw();
        ServingWindowCommandFactory factory(authority, law, 5, 4, 1);
        ServingRowProgress r0(authority, 0, false, false);
        auto cmd = factory.forRow(r0);
        EXPECT_TRUE(cmd.valid()); // sealed epoch 1
        EXPECT_EQ(cmd.window_start, 0u); // engine-owned position
        EXPECT_EQ(cmd.expected_authority, authority.authorityIdentity());
        EXPECT_EQ(cmd.expected_grant_id, authority.current().grant_id);
        EXPECT_EQ(cmd.expected_nonce, authority.issuanceNonce());
        // Mid-window command (engine row position 1) is grid-admissible.
        ServingRowProgress r1(authority, 1, false, false);
        auto cmd1 = factory.forRow(r1);
        EXPECT_TRUE(cmd1.valid());
        EXPECT_EQ(cmd1.window_start, 1u);
    }


    /** @brief #137 PIN: a minted command RETAINS the exact law node — replacing
     *         the wrapper binding (or a temporary issue argument) can never
     *         retarget or dangle the minted command. */
    TEST(ServingWindowDecode, CommandPinsExactLawNode)
    {
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto law = admission->issue(1, 11, 2, 32);
        auto authority = makeGrantAuthority();
        // Factory minted from a TEMPORARY issue: the wrapper dies immediately,
        // the retained node keeps the command valid (no dangling) AND IS USED.
        ServingWindowCommandFactory temp_factory(authority, admission->issue(2, 22, 2, 32), 5, 4, 1);
        ServingRowProgress rtmp(authority, 0, false, false);
        auto cmd_temp = temp_factory.forRow(rtmp);
        EXPECT_TRUE(cmd_temp.valid());
        ServingWindowCommandFactory factory(authority, law, 5, 4, 1);
        ServingRowProgress r0(authority, 0, false, false);
        auto cmd = factory.forRow(r0);
        const auto pinned_identity = cmd.lawNode()->issueIdentity();
        EXPECT_TRUE(cmd.valid());
        // RELEASE THE ORIGINAL LAW (the node retained by cmd): the command pin
        // keeps the node valid; the wrapper storage is consumed.
        admission->release(law);
        EXPECT_TRUE(law.released());
        EXPECT_EQ(law.tableToken(), 0u); // storage reset (no lingering shared_ptr)
        EXPECT_TRUE(cmd.valid());        // retained node still live for the command
        EXPECT_EQ(cmd.lawNode()->issueIdentity(), pinned_identity);
        // Replace the wrapper (move-assign a fresh issue).
        law = admission->issue(3, 33, 2, 32);
        EXPECT_TRUE(cmd.valid()); // not retargeted by the new wrapper
        EXPECT_EQ(cmd.lawNode()->issueIdentity(), pinned_identity);
    }

    /** @brief AUTHENTICATED OUTCOMES: EOS/Forced/Stop allow under-production
     *         (0 < produced <= bound) as OWNER-authorized exceptions; the
     *         response-limit boundary requires Terminal (not Continue); the
     *         partial-final window at c0/W16/L5 is exact at 5 for Terminal. */
    TEST(ServingWindowDecode, AuthenticatedOutcomesAndBoundaryRules)
    {
        // Boundary: W16/L5 c0 is terminal -> Continue is refused.
        WindowGrantAuthority a(1);
        WindowGrant g;
        g.grant_id = 9;
        g.slot = 0;
        g.generation = 1;
        g.window = IntrinsicWindowPlan{16, 5};
        g.destination_reservation = 9;
        g.serving_instance = 1;
        g.state = GrantState::Outstanding;
        a.issue(g);
        auto [admission, law] = makeLaw();
        auto in = makeInput(a, law);
        EXPECT_THROW(completeWindow(a, in, 5, WindowTermination::Continue),
                     std::logic_error);
        EXPECT_TRUE(a.mayExecute()); // refused without consuming
        const auto term = completeWindow(a, in, 5, WindowTermination::Terminal);
        EXPECT_TRUE(term.ok());
        EXPECT_EQ(term.produced(), 5u); // exact partial-final bound
        // Authenticated exceptions on a normal window: EOS/Forced/Stop allow
        // 0 < produced <= bound; 0 is refused.
        auto a2 = makeGrantAuthority();
        auto [ad2, law2] = makeLaw();
        auto in2 = makeInput(a2, law2);
        EXPECT_THROW(completeWindow(a2, in2, 0, WindowTermination::EndOfSequence),
                     std::logic_error);
        const auto eos = completeWindow(a2, in2, 3, WindowTermination::EndOfSequence);
        EXPECT_TRUE(eos.ok());
        EXPECT_EQ(eos.produced(), 3u);
        EXPECT_EQ(eos.termination(), WindowTermination::EndOfSequence);
        auto a3 = makeGrantAuthority();
        auto [ad3, law3] = makeLaw();
        auto in3 = makeInput(a3, law3);
        const auto forced = completeWindow(a3, in3, 1, WindowTermination::Forced);
        EXPECT_TRUE(forced.ok());
        auto a4 = makeGrantAuthority();
        auto [ad4, law4] = makeLaw();
        auto in4 = makeInput(a4, law4);
        EXPECT_THROW(completeWindow(a4, in4, 3, WindowTermination::Cancelled),
                     std::logic_error); // separate path
        // AUTHENTICATED cancellation: exactly once; repeated throws.
        const auto cancelled = cancelWindow(a4, in4);
        EXPECT_FALSE(cancelled.ok());
        EXPECT_THROW(cancelWindow(a4, in4), std::logic_error);
    }

    /** @brief IDENTITY EXHAUSTION: issue identities are monotonic and never
     *         reused; issuedCount is exact. */
    TEST(ServingWindowDecode, LawIssueIdentityIsNeverReused)
    {
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto x = admission->issue(1, 11, 2, 32);
        const auto first = x.issueIdentity();
        admission->release(x);
        auto y = admission->issue(2, 22, 2, 32);
        EXPECT_GT(y.issueIdentity(), first); // never reused
        EXPECT_EQ(admission->issuedCount(), 2u);
    }




    /** @brief Terminal-before-L, unknown outcome (255) and a FILLED window with
     *         an EOS observation are all rejected synchronously. */
    TEST(ServingWindowDecode, OutcomeConsistencyRejectsContradictions)
    {
        // Terminal before the response limit is refused (not at the boundary).
        auto a = makeGrantAuthority();
        auto [ad, law] = makeLaw();
        auto in = makeInput(a, law, 1); // c1: bound 15, not at limit boundary
        EXPECT_THROW(completeWindow(a, in, 15, WindowTermination::Terminal),
                     std::logic_error);
        EXPECT_TRUE(a.mayExecute());
        // Unknown outcome value bypasses nothing (default case rejects).
        auto a2 = makeGrantAuthority();
        auto [ad2, law2] = makeLaw();
        auto in2 = makeInput(a2, law2);
        EXPECT_THROW(completeWindow(a2, in2, 16,
                                    static_cast<WindowTermination>(255)),
                     std::logic_error);
        EXPECT_TRUE(a2.mayExecute());
        // EOS AT THE BOUNDARY IS LEGITIMATE: token 16 at W16/L32/c0 fills the
        // window AND the model observed EOS at the last token (a short count is
        // NOT the authentication; the production decode path is).
        auto a3 = makeGrantAuthority();
        auto [ad3, law3] = makeLaw();
        auto in3 = makeInput(a3, law3);
        const auto eos = completeWindow(a3, in3, 16, WindowTermination::EndOfSequence);
        EXPECT_TRUE(eos.ok());
        EXPECT_EQ(eos.produced(), 16u);
        EXPECT_EQ(eos.termination(), WindowTermination::EndOfSequence);
    }

    /** @brief AUTHENTICATED CANCELLATION RECEIPT: identity-bound, no consumption,
     *         outcome Cancelled; repeated cancellation refused. */
    TEST(ServingWindowDecode, CancellationReturnsBoundReceipt)
    {
        auto a = makeGrantAuthority();
        auto [ad, law] = makeLaw();
        auto in = makeInput(a, law);
        const auto receipt = cancelWindow(a, in);
        EXPECT_TRUE(receipt.cancelled());
        EXPECT_FALSE(receipt.ok());
        EXPECT_EQ(receipt.produced(), 0u);
        EXPECT_EQ(receipt.grantId(), 7u); // bound to the cancelled grant
        EXPECT_EQ(receipt.generation(), 1u);
        EXPECT_FALSE(receipt.grantConsumed());
        EXPECT_EQ(receipt.termination(), WindowTermination::Cancelled);
        EXPECT_THROW(cancelWindow(a, in), std::logic_error); // exactly once
    }

    /** @brief ISSUE-IDENTITY EXHAUSTION: the admission's bounded identity cap is
     *         enforced (cap 2 -> third issue refused; never reused). */
    TEST(ServingWindowDecode, LawIssueIdentityExhaustionIsEnforced)
    {
        auto admission = std::make_shared<ServingLawAdmission>(32, 2); // cap 2
        auto x = admission->issue(1, 11, 2, 32);
        auto y = admission->issue(2, 22, 2, 32);
        EXPECT_THROW(admission->issue(3, 33, 2, 32), std::logic_error);
        EXPECT_EQ(admission->issuedCount(), 2u);
        EXPECT_GT(y.issueIdentity(), x.issueIdentity());
    }


    /** @brief FABRICATION IS CLOSED AT THE TYPE LEVEL: the node is not publicly
     *         constructible (a fabricated metadata object cannot exist) and the
     *         input only accepts the exact node issued by a LIVE admission (the
     *         input's node-is-null path is rejected by the ingress). A node from
     *         a live-but-zero-issues admission cannot be attached without owning
     *         that admission's issued object (ownership = authenticated()). */
    TEST(ServingWindowDecode, FabricationIsClosedAtTypeAndOwnership)
    {
        // A node can only come from issue(); the input requires the exact
        // admitted node. Ownership is proven by authenticated() (registry).
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto law = admission->issue(1, 11, 2, 32);
        auto authority = makeGrantAuthority();
        MockOrchestrationRunner runner;
        // No public Node constructor exists (compile-time closure).
        static_assert(!std::is_default_constructible_v<ServingLawBindingNode>);
        EXPECT_TRUE(law.valid());
    }

    /** @brief FOREIGN PROGRESS is refused: B's progress cannot be attached to
     *         A's pins (the progress is authority-bound). */
    TEST(ServingWindowDecode, ForeignRowProgressRefused)
    {
        auto a = makeGrantAuthority();
        auto b = makeGrantAuthority();
        auto [ad, law] = makeLaw();
        ServingWindowCommandFactory factory(a, law, 5, 4, 1);
        ServingRowProgress progressB(b, 0, false, false);
        EXPECT_THROW(factory.forRow(progressB), std::logic_error);
    }

    /** @brief TRUE FREE BOUNDARY: the physical node count defers to the last
     *         shared_ptr holder (wrapper + factory + command pins); it does NOT
     *         decrement early and the factory count is observable. */
    TEST(ServingWindowDecode, PhysicalNodeFreesAtLastHolder)
    {
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto law = admission->issue(1, 11, 2, 32);
        EXPECT_EQ(admission->physicalNodeCount(), 1u);
        {
            auto authority = makeGrantAuthority();
            ServingWindowCommandFactory factory(authority, law, 5, 4, 1);
            ServingRowProgress r0(authority, 0, false, false);
            {
                auto cmd = factory.forRow(r0);
                // Release the wrapper: the command/factory pins keep the node alive.
                admission->release(law);
                EXPECT_EQ(admission->physicalNodeCount(), 1u);
                EXPECT_TRUE(cmd.valid());
                {
                    auto cmd2 = factory.forRow(r0);
                    (void)cmd2;
                }
                EXPECT_EQ(admission->physicalNodeCount(), 1u);
            } // command dies; the factory still pins the node
            EXPECT_EQ(admission->physicalNodeCount(), 1u);
        } // factory destroyed -> last pin gone -> node freed
        EXPECT_EQ(admission->physicalNodeCount(), 0u);
    }

    /** @brief An EMPTY authority consumes NO global identity (lazy assignment at
     *         first issue only). */
    TEST(ServingWindowDecode, EmptyAuthorityConsumesNoIdentity)
    {
        WindowGrantAuthority empty(1);
        EXPECT_EQ(empty.authorityIdentity(), 0u); // unissued -> no identity
        auto authority = makeGrantAuthority();    // issued once -> identity assigned
        EXPECT_NE(authority.authorityIdentity(), 0u);
    }


    /** @brief STALE PROGRESS: retain old progress; resolve + reissue on the SAME
     *         authority (new + grant id, same generation) — the old progress is
     *         refused (issuance pins, not just authority id). */
    TEST(ServingWindowDecode, StaleProgressAfterResolveReissueRefused)
    {
        WindowGrantAuthority authority(1);
        WindowGrant g;
        g.grant_id = 7;
        g.slot = 0;
        g.generation = 1;
        g.window = IntrinsicWindowPlan{16, 32};
        g.destination_reservation = 9;
        g.serving_instance = 1;
        g.state = GrantState::Outstanding;
        authority.issue(g);
        ServingRowProgress old_progress(authority, 0, false, false);
        authority.resolve();
        WindowGrant g2 = g;
        g2.grant_id = 8; // same generation, new issuance
        g2.state = GrantState::Outstanding;
        authority.issue(g2);
        auto [ad, law] = makeLaw();
        ServingWindowCommandFactory factory(authority, law, 5, 4, 1);
        EXPECT_THROW(factory.forRow(old_progress), std::logic_error); // stale pins
        ServingRowProgress fresh(authority, 0, false, false);
        auto cmd = factory.forRow(fresh); // current issuance admitted
        EXPECT_TRUE(cmd.valid());
    }

    /** @brief MOVED-FROM RESET: assign a live authority INTO a moved-from
     *         destination — the destination becomes live (can issue again). */
    TEST(ServingWindowDecode, MovedFromFlagResetsOnLiveAssignment)
    {
        auto a = makeGrantAuthority();
        WindowGrantAuthority moved{std::move(a)}; // a moved-from
        auto b = makeGrantAuthority();
        a = std::move(b); // live authority assigned into the moved-from destination
        // A is now a LIVE authority (adopts b's outstanding grant; not dead).
        EXPECT_TRUE(a.mayExecute());
        WindowGrant g;
        g.grant_id = 21;
        EXPECT_THROW(b.issue(g), std::logic_error); // source is dead
        a.resolve();
        // A can issue again normally (the flag was reset on assignment).
        g.slot = 0;
        g.generation = 1;
        g.window = IntrinsicWindowPlan{16, 32};
        g.destination_reservation = 9;
        g.serving_instance = 1;
        g.state = GrantState::Outstanding;
        EXPECT_NO_THROW(a.issue(g));
        EXPECT_TRUE(a.mayExecute());
    }


    /** @brief PINNED CAPACITY-1 RE-ISSUE: while a command/factory pin holds the
     *         live node, a capacity-1 admission REFUSES the re-issue; after the
     *         LAST holder dies the capacity is free again (true free boundary,
     *         not a logical-live-only count). */
    TEST(ServingWindowDecode, PinnedCapacityOneBlocksReissueUntilLastHolder)
    {
        auto admission = std::make_shared<ServingLawAdmission>(1); // capacity 1
        auto law = admission->issue(1, 11, 2, 1);
        auto authority = makeGrantAuthority();
        {
            ServingWindowCommandFactory factory(authority, law, 5, 4, 1);
            ServingRowProgress r0(authority, 0, false, false);
            auto cmd = factory.forRow(r0);
            admission->release(law); // wrapper gone; the pin holds the node
            EXPECT_TRUE(cmd.valid());
            EXPECT_THROW(admission->issue(2, 22, 2, 1), std::logic_error); // occupied
            {
                auto cmd2 = factory.forRow(r0);
                (void)cmd2;
            }
            EXPECT_THROW(admission->issue(3, 33, 2, 1), std::logic_error); // still pinned
        } // factory + command die -> last holder gone
        EXPECT_EQ(admission->physicalNodeCount(), 0u);
        auto again = admission->issue(4, 44, 2, 1); // capacity free again
        EXPECT_TRUE(again.valid());
    }


    /** @brief STICKY EXHAUSTION: once the identity space trips, EVERY later
     *         request is refused (repeated, not just the single overflow) —
     *         authority and admission allocators. */
    TEST(ServingWindowDecode, IdentityExhaustionIsSticky)
    {
        // Authority allocator.
        AuthorityIdentityExhaustionProbe::arm();
        WindowGrant g;
        g.grant_id = 30;
        g.slot = 0;
        g.generation = 1;
        g.window = IntrinsicWindowPlan{16, 32};
        g.destination_reservation = 9;
        g.serving_instance = 1;
        g.state = GrantState::Outstanding;
        {
            WindowGrantAuthority a1(1);
            EXPECT_THROW(a1.issue(g), std::logic_error);
            EXPECT_TRUE(AuthorityIdentityExhaustionProbe::sticky());
            WindowGrantAuthority a2(1);
            EXPECT_THROW(a2.issue(g), std::logic_error); // repeated refusal
            WindowGrantAuthority a3(1);
            EXPECT_THROW(a3.issue(g), std::logic_error); // still refused
        }
        AuthorityIdentityExhaustionProbe::clear();
        {
            WindowGrantAuthority a4(1);
            EXPECT_NO_THROW(a4.issue(g)); // restored after clearing the probe
        }
        // Admission allocator.
        AdmissionIdentityExhaustionProbe::arm();
        EXPECT_THROW(std::make_shared<ServingLawAdmission>(32), std::logic_error);
        EXPECT_TRUE(AdmissionIdentityExhaustionProbe::sticky());
        EXPECT_THROW(std::make_shared<ServingLawAdmission>(32), std::logic_error);
        EXPECT_THROW(std::make_shared<ServingLawAdmission>(32), std::logic_error);
        AdmissionIdentityExhaustionProbe::clear();
    }

    /** @brief PHYSICAL AUTHORITY GATE: the capacity bound follows the PHYSICAL
     *         node count (the live-count retire-then-free window cannot admit a
     *         second physical node at capacity 1). */
    TEST(ServingWindowDecode, PhysicalAuthorityGateBlocksSecondPhysicalNode)
    {
        auto admission = std::make_shared<ServingLawAdmission>(1);
        auto law = admission->issue(1, 11, 2, 1);
        // MOVE the wrapper (the same physical node is retained by the moved
        // holder): the capacity-1 bound follows the PHYSICAL node.
        auto held = std::move(law);
        EXPECT_TRUE(held.valid());
        EXPECT_EQ(admission->physicalNodeCount(), 1u);
        EXPECT_THROW(admission->issue(2, 22, 2, 1), std::logic_error);
        // Hold the FREE BOUNDARY through a command pin, then observe the actual
        // free: release the wrapper with a command alive -> physical 1; command
        // dies -> still 1 (factory? no factory here; use the wrapper drop path)...
        auto authority = makeGrantAuthority();
        {
            ServingWindowCommandFactory factory(authority, held, 5, 4, 1);
            ServingRowProgress r0(authority, 0, false, false);
            auto cmd = factory.forRow(r0);
            admission->release(held); // wrapper released; command pin keeps the node
            EXPECT_EQ(admission->physicalNodeCount(), 1u);
            EXPECT_THROW(admission->issue(3, 33, 2, 1), std::logic_error);
            {
                auto cmd2 = factory.forRow(r0);
                (void)cmd2;
                EXPECT_EQ(admission->physicalNodeCount(), 1u);
            }
        } // command + factory die -> actual free boundary
        EXPECT_EQ(admission->physicalNodeCount(), 0u);
        auto law2 = admission->issue(4, 44, 2, 1); // capacity free again
        EXPECT_TRUE(law2.valid());
    }

    /** @brief ACYCLIC RETAINED AUTHORITY: external drop keeps the owner alive
     *         through the leases; FINAL-COMMAND and FINAL-FACTORY ownership
     *         tested separately; weak owner expiry observed; an explicit release
     *         while the dead wrapper stays in scope retains no owner. */
    TEST(ServingWindowDecode, OwnerSurvivesLeaseAndExpiresWithFinalHolder)
    {
        std::weak_ptr<ServingLawAdmission> weak;
        {
            auto admission = std::make_shared<ServingLawAdmission>(32);
            weak = admission; // observe expiry
            auto law = admission->issue(1, 11, 2, 32);
            auto authority = makeGrantAuthority();
            ServingWindowCommandFactory factory(authority, law, 5, 4, 1);
            ServingRowProgress r0(authority, 0, false, false);
            {
                auto cmd = factory.forRow(r0);
                admission.reset(); // external drop; leases retain the owner
                EXPECT_TRUE(law.valid());
                EXPECT_TRUE(cmd.valid());
                EXPECT_EQ(law.admissionOwner()->liveCount(), 1u);
            } // FINAL-COMMAND dies; the factory still retains the owner + node
            EXPECT_TRUE(law.valid());
            EXPECT_FALSE(weak.expired()); // owner alive via the factory
            // EXPLICIT release while the dead wrapper remains in scope.
            law.admissionOwner()->release(law);
            EXPECT_TRUE(law.released());
            EXPECT_FALSE(law.valid());
            EXPECT_EQ(law.admissionOwner(), nullptr); // no retention by the dead wrapper
            EXPECT_FALSE(weak.expired());            // the factory still retains the owner
        } // FINAL-FACTORY dies -> node freed -> owner expires
        EXPECT_TRUE(weak.expired()); // owner expiry after the final lease
    }


    /** @brief SOLE-COMMAND and SOLE-FACTORY free-boundary witness: the free
     *         event is recorded with owner-aliveness at the exact moment
     *         (deterministic deallocating witness); the final command/factory
     *         are the unique last leases and the owner expires only after the
     *         actual free. */
    TEST(ServingWindowDecode, SoleCommandFreeBoundaryWitness)
    {
        auto observer = std::make_shared<LawFreeObserver>();
        std::weak_ptr<ServingLawAdmission> weak;
        std::optional<ServingWindowDecodeInput> cmd;
        auto admission = std::make_shared<ServingLawAdmission>(1); // capacity 1
        weak = admission;
        admission->observeFree(observer);
        auto law = admission->issue(1, 11, 2, 1);
        auto authority = makeGrantAuthority();
        ServingRowProgress r0(authority, 0, false, false);
        {
            ServingWindowCommandFactory factory(authority, law, 5, 4, 1);
            cmd.emplace(factory.forRow(r0));
        } // FACTORY dies: the COMMAND is now the unique remaining lease
        law = ServingLawBinding{};
        // HELD-FREE REFUSAL with the single lease: only the command holds the
        // physical node; a new issue at capacity 1 REFUSES (physical gate).
        EXPECT_THROW(admission->issue(2, 22, 2, 1), std::logic_error);
        admission.reset(); // external dropped; only the command holds the owner
        EXPECT_FALSE(weak.expired());
        EXPECT_EQ(observer->frees.load(), 0u); // nothing freed while the cmd pins
        cmd.reset(); // SOLE-COMMAND dies -> actual free
        EXPECT_EQ(observer->frees.load(), 1u);
        EXPECT_TRUE(observer->owner_alive_at_free.load()); // owner alive AT the free
        EXPECT_TRUE(weak.expired()); // owner expires only after the free
    }

    TEST(ServingWindowDecode, SoleFactoryFreeBoundaryWitness)
    {
        auto observer = std::make_shared<LawFreeObserver>();
        std::weak_ptr<ServingLawAdmission> weak;
        auto admission = std::make_shared<ServingLawAdmission>(1); // capacity 1
        weak = admission;
        admission->observeFree(observer);
        auto law = admission->issue(1, 11, 2, 1);
        auto authority = makeGrantAuthority();
        ServingRowProgress r0(authority, 0, false, false);
        {
            ServingWindowCommandFactory factory(authority, law, 5, 4, 1);
            std::optional<ServingWindowDecodeInput> cmd;
            cmd.emplace(factory.forRow(r0));
            admission->release(law); // only factory + cmd remain
            cmd.reset();             // cmd dies: the FACTORY is the unique lease
            // HELD-FREE REFUSAL: only the factory holds the physical node.
            EXPECT_THROW(admission->issue(2, 22, 2, 1), std::logic_error);
            EXPECT_EQ(observer->frees.load(), 0u);
            // DROP THE EXTERNAL ADMISSION BEFORE THE FACTORY DESTRUCTION: the
            // factory is the last strong owner at the block end (ordering is NOT
            // masked by an external strong reference).
            admission.reset();
            EXPECT_FALSE(weak.expired());
        } // SOLE-FACTORY dies -> actual free, then owner expiry
        EXPECT_EQ(observer->frees.load(), 1u);
        EXPECT_TRUE(observer->owner_alive_at_free.load());
        EXPECT_TRUE(weak.expired());
    }


    /** @brief S3.2b BRIDGE: the row owner advances one row through the TYPED ABI
     *         — mint (engine-owned progress), ingress-validated decode,
     *         AUTHENTIC completion (grant resolved exactly once by the bridge),
     *         and the produced count is the exact/exception window. */
    TEST(ServingWindowDecode, BridgeAdvancesOneRowThroughTheTypedAbl)
    {
        auto authority = makeGrantAuthority();
        auto [admission, law] = makeLaw();
        auto factory = std::make_shared<ServingWindowCommandFactory>(authority, law, 5, 4, 1);
        CompletingRunner runner;
        ServingWindowDecodeBridge bridge(factory, runner);
        ServingRowProgress r0(authority, 0, false, false);
        const auto result = bridge.decodeWindow(r0);
        EXPECT_TRUE(result.ok());
        EXPECT_EQ(result.produced(), 16u);
        EXPECT_TRUE(result.grantConsumed());
        EXPECT_EQ(result.grantId(), 7u);
        EXPECT_FALSE(authority.mayExecute()); // resolved exactly once
        // A TEMPORARY factory cannot be bound (requires shared ownership).
        auto a_tmp = makeGrantAuthority();
        auto [adTmp, lawTmp] = makeLaw();
        EXPECT_THROW(
            (ServingWindowDecodeBridge(std::shared_ptr<ServingWindowCommandFactory>{},
                                       runner)),
            std::logic_error);
        // Foreign progress is refused AT MINT (no bypass through the bridge).
        auto authority2 = makeGrantAuthority();
        auto authority3 = makeGrantAuthority();
        auto [ad2, law2] = makeLaw();
        auto factory2 = std::make_shared<ServingWindowCommandFactory>(authority2, law2, 5, 4, 1);
        ServingWindowDecodeBridge bridge2(factory2, runner);
        EXPECT_THROW(bridge2.decodeWindow(ServingRowProgress(authority3, 0, false, false)),
                     std::logic_error); // foreign row progress
        // Stale progress (minted before a resolve+reissue on the same authority).
        auto authority4 = makeGrantAuthority();
        auto [ad4, law4] = makeLaw();
        ServingRowProgress stale_r(authority4, 0, false, false);
        authority4.resolve();
        WindowGrant g4 = authority4.current();
        g4.grant_id = 70;
        g4.state = GrantState::Outstanding;
        authority4.issue(g4);
        auto factory4 = std::make_shared<ServingWindowCommandFactory>(authority4, law4, 5, 4, 1);
        ServingWindowDecodeBridge bridge4(factory4, runner);
        EXPECT_THROW(bridge4.decodeWindow(stale_r), std::logic_error);
        // A fail-closed runner returns the explicit diagnostic WITHOUT consuming.
        auto authority5 = makeGrantAuthority();
        auto [ad5, law5] = makeLaw();
        auto factory5 = std::make_shared<ServingWindowCommandFactory>(authority5, law5, 5, 4, 1);
        MockOrchestrationRunner plain_runner;
        ServingWindowDecodeBridge bridge5(factory5, plain_runner);
        ServingRowProgress r5(authority5, 0, false, false);
        const auto refused = bridge5.decodeWindow(r5);
        EXPECT_FALSE(refused.ok());
        EXPECT_NE(refused.error().find("unsupported"), std::string::npos);
        EXPECT_TRUE(authority5.mayExecute()); // NOT consumed on refusal (asserted on the SAME authority)
    }

    /** @brief OPERATION CORRESPONDENCE (#134) — runner fixture + regression. */
    class MismatchRunner : public MockOrchestrationRunner
    {
    public:
        MismatchRunner(ServingWindowCommandFactory &other_factory,
                       WindowGrantAuthority &other_authority)
            : other_factory_(other_factory), other_authority_(other_authority)
        {
        }
    private:
        ServingWindowDecodeResult decodeServingWindowImpl(
            const ServingWindowDecodeInput &request) override
        {
            (void)request;
            // Authentically complete B (a DIFFERENT row/grant) and return B's
            // receipt for A's call.
            auto cmdB = other_factory_.forRow(
                ServingRowProgress(other_authority_, 0, false, false));
            return completeWindow(other_authority_, cmdB, 16,
                                  WindowTermination::Continue);
        }
        ServingWindowCommandFactory &other_factory_;
        WindowGrantAuthority &other_authority_;
    };

    TEST(ServingWindowDecode, BridgeRejectsCrossOperationReceipt)
    {
        auto authorityA = makeGrantAuthority();
        WindowGrant gB;
        gB.grant_id = 88; // DISTINCT grant identity for B (receipt must not match A)
        gB.slot = 1;
        gB.generation = 1;
        gB.window = IntrinsicWindowPlan{16, 32};
        gB.destination_reservation = 9;
        gB.serving_instance = 1;
        gB.state = GrantState::Outstanding;
        auto authorityB = makeGrantAuthority(gB);
        auto [adA, lawA] = makeLaw();
        auto [adB, lawB] = makeLaw();
        auto factoryB = std::make_shared<ServingWindowCommandFactory>(authorityB, lawB, 5, 4, 1);
        MismatchRunner runner(*factoryB, authorityB);
        auto factoryA = std::make_shared<ServingWindowCommandFactory>(authorityA, lawA, 5, 4, 1);
        ServingWindowDecodeBridge bridge(factoryA, runner);
        ServingRowProgress rA(authorityA, 0, false, false);
        const auto result = bridge.decodeWindow(rA);
        EXPECT_FALSE(result.ok()); // cross-operation receipt rejected
        EXPECT_NE(result.error().find("cross-operation or intent mismatch"), std::string::npos);
        EXPECT_EQ(result.error().find("state violation"), std::string::npos); // untouched: no state claim
        EXPECT_TRUE(authorityA.mayExecute());   // A remains outstanding (impl did NOT touch A)
        EXPECT_FALSE(authorityB.mayExecute());  // B was consumed by the impl
    }

    /** @brief SLOT IN THE CORRESPONDENCE TUPLE: B with grant_id 7 (same as A) but a
     *         DIFFERENT slot is REJECTED (the full immutable invocation tuple
     *         includes slot). */
    TEST(ServingWindowDecode, BridgeCorrespondenceIncludesSlot)
    {
        auto authorityA = makeGrantAuthority();
        WindowGrant gB;
        gB.grant_id = 7; // same grant id as A
        gB.slot = 1;     // DIFFERENT slot (this was the counterexample)
        gB.generation = 1;
        gB.window = IntrinsicWindowPlan{16, 32};
        gB.destination_reservation = 9;
        gB.serving_instance = 1;
        gB.state = GrantState::Outstanding;
        auto authorityB = makeGrantAuthority(gB);
        auto [adA, lawA] = makeLaw();
        auto [adB, lawB] = makeLaw();
        auto factoryB = std::make_shared<ServingWindowCommandFactory>(authorityB, lawB, 5, 4, 1);
        MismatchRunner runner(*factoryB, authorityB);
        auto factoryA = std::make_shared<ServingWindowCommandFactory>(authorityA, lawA, 5, 4, 1);
        ServingWindowDecodeBridge bridge(factoryA, runner);
        ServingRowProgress rA(authorityA, 0, false, false);
        const auto result = bridge.decodeWindow(rA);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.error().find("cross-operation or intent mismatch"), std::string::npos);
        EXPECT_TRUE(authorityA.mayExecute());
    }

    /** @brief CANCELLATION IS VALIDATED SEPARATELY: a cross-operation
     *         CANCELLATION is rejected (cancellation is a completion outcome,
     *         NOT a pass-through failure). */
    TEST(ServingWindowDecode, BridgeValidatesCancellationCorrespondence)
    {
        auto authorityA = makeGrantAuthority();
        WindowGrant gB;
        gB.grant_id = 7;
        gB.slot = 1;
        gB.generation = 1;
        gB.window = IntrinsicWindowPlan{16, 32};
        gB.destination_reservation = 9;
        gB.serving_instance = 1;
        gB.state = GrantState::Outstanding;
        auto authorityB = makeGrantAuthority(gB);
        auto [adA, lawA] = makeLaw();
        auto [adB, lawB] = makeLaw();
        class CancelMismatchRunner : public MockOrchestrationRunner
        {
        public:
            CancelMismatchRunner(ServingWindowCommandFactory &f, WindowGrantAuthority &a)
                : f_(f), a_(a) {}
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                (void)request;
                auto cmdB = f_.forRow(ServingRowProgress(a_, 0, false, false));
                return cancelWindow(a_, cmdB); // B's cancellation returned for A's call
            }
            ServingWindowCommandFactory &f_;
            WindowGrantAuthority &a_;
        };
        auto factoryB = std::make_shared<ServingWindowCommandFactory>(authorityB, lawB, 5, 4, 1);
        CancelMismatchRunner runner(*factoryB, authorityB);
        auto factoryA = std::make_shared<ServingWindowCommandFactory>(authorityA, lawA, 5, 4, 1);
        ServingWindowDecodeBridge bridge(factoryA, runner);
        ServingRowProgress rA(authorityA, 0, false, false);
        const auto result = bridge.decodeWindow(rA);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.error().find("cross-operation or intent mismatch"), std::string::npos);
        EXPECT_EQ(result.error().find("state violation"), std::string::npos); // untouched
        EXPECT_TRUE(authorityA.mayExecute());
    }

    /** @brief OWNERSHIP LIFETIME PROOF: the bridge OWNS the factory — use after
     *         the external shared_ptr reset still works, and the factory expires
     *         only after the bridge's destruction. */
    TEST(ServingWindowDecode, BridgeOwnsFactoryPastExternalReset)
    {
        auto authority = makeGrantAuthority();
        auto [ad, law] = makeLaw();
        std::weak_ptr<ServingWindowCommandFactory> weak;
        std::shared_ptr<ServingWindowCommandFactory> external =
            std::make_shared<ServingWindowCommandFactory>(authority, law, 5, 4, 1);
        weak = external;
        CompletingRunner runner;
        {
            ServingWindowDecodeBridge bridge(external, runner);
            external.reset(); // EXTERNAL owner dropped; the bridge retains the factory
            EXPECT_FALSE(weak.expired());
            ServingRowProgress r0(authority, 0, false, false);
            const auto result = bridge.decodeWindow(r0);
            EXPECT_TRUE(result.ok());
            EXPECT_EQ(result.produced(), 16u);
        } // bridge destroyed -> the factory expires
        EXPECT_TRUE(weak.expired());
    }


    /** @brief EQUAL-VISIBLE-TUPLE INDEPENDENT A/B: same slot/gen/id/instance but a
     *         DIFFERENT authority UID/nonce/law/transaction — the receipt's full
     *         invocation binding rejects it (no one-field match). */
    TEST(ServingWindowDecode, BridgeRejectsEqualTupleDifferentInvocation)
    {
        auto authorityA = makeGrantAuthority();
        WindowGrant gB;
        gB.grant_id = 7; // SAME visible tuple as A
        gB.slot = 0;
        gB.generation = 1;
        gB.window = IntrinsicWindowPlan{16, 32};
        gB.destination_reservation = 9;
        gB.serving_instance = 1;
        gB.state = GrantState::Outstanding;
        auto authorityB = makeGrantAuthority(gB);
        auto [adA, lawA] = makeLaw();
        auto [adB, lawB] = makeLaw();
        auto factoryB = std::make_shared<ServingWindowCommandFactory>(authorityB, lawB, 5, 4, 1);
        MismatchRunner runner(*factoryB, authorityB);
        auto factoryA = std::make_shared<ServingWindowCommandFactory>(authorityA, lawA, 5, 4, 1);
        ServingWindowDecodeBridge bridge(factoryA, runner);
        ServingRowProgress rA(authorityA, 0, false, false);
        const auto result = bridge.decodeWindow(rA);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.error().find("cross-operation or intent mismatch"), std::string::npos);
        EXPECT_EQ(result.error().find("state violation"), std::string::npos); // untouched
        EXPECT_TRUE(authorityA.mayExecute());
    }

    /** @brief CONSUME-THEN-ERROR STATE VIOLATION: an impl that completes the
     *         row's OWN grant then returns a plain failure is reported as a state
     *         violation (never rolled back or re-completed). */
    TEST(ServingWindowDecode, BridgeReportsConsumeThenErrorStateViolation)
    {
        auto authority = makeGrantAuthority();
        auto [ad, law] = makeLaw();
        class ConsumeThenErrorRunner : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                // Consume the row's own grant, then return a plain failure.
                auto consumed = completeWindow(request.authority, request, 16,
                                               WindowTermination::Continue);
                (void)consumed;
                return ServingWindowDecodeResult::failure("internal error");
            }
        };
        auto factory = std::make_shared<ServingWindowCommandFactory>(authority, law, 5, 4, 1);
        ConsumeThenErrorRunner runner;
        ServingWindowDecodeBridge bridge(factory, runner);
        ServingRowProgress r0(authority, 0, false, false);
        const auto result = bridge.decodeWindow(r0);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.error().find("grant consumed without a completion"),
                  std::string::npos);
        EXPECT_FALSE(authority.mayExecute()); // consumed (reported, not rolled back)
    }

    /** @brief LEGITIMATE SAME-CALL CANCELLATION: an impl cancelling ITS OWN
     *         command's authority passes the correspondence (a legitimate
     *         cancellation is preserved, not misreported). */
    TEST(ServingWindowDecode, BridgePreservesSameCallCancellation)
    {
        auto authority = makeGrantAuthority();
        auto [ad, law] = makeLaw();
        class SameCallCancelRunner : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                return cancelWindow(request.authority, request);
            }
        };
        auto factory = std::make_shared<ServingWindowCommandFactory>(authority, law, 5, 4, 1);
        SameCallCancelRunner runner;
        ServingWindowDecodeBridge bridge(factory, runner);
        ServingRowProgress r0(authority, 0, false, false);
        const auto result = bridge.decodeWindow(r0);
        EXPECT_TRUE(result.cancelled()); // same-call cancellation preserved
        EXPECT_EQ(result.grantId(), 7u);
        EXPECT_FALSE(authority.mayExecute());
    }


    /** @brief LAW ADMISSION IS PART OF THE BINDING: the same authority + tx with
     *         a law from a DIFFERENT admission (whose first issue also has serial
     *         1) is REJECTED (law admission identity distinguishes). */
    TEST(ServingWindowDecode, BridgeRejectsCrossAdmissionLaw)
    {
        auto authority = makeGrantAuthority();
        auto [adA, lawA] = makeLaw(); // admission A, law serial 1
        auto otherAdmission = std::make_shared<ServingLawAdmission>(32);
        auto lawB = otherAdmission->issue(3, 11, 2, 32); // DIFFERENT admission, serial 1
        // The malicious impl completes B (another law admission) for A's call.
        class CrossLawRunner : public MockOrchestrationRunner
        {
        public:
            CrossLawRunner(ServingWindowCommandFactory &f, WindowGrantAuthority &a)
                : f_(f), a_(a) {}
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                (void)request;
                auto cmdB = f_.forRow(ServingRowProgress(a_, 0, false, false));
                return completeWindow(a_, cmdB, 16, WindowTermination::Continue);
            }
            ServingWindowCommandFactory &f_;
            WindowGrantAuthority &a_;
        };
        auto factoryB = std::make_shared<ServingWindowCommandFactory>(authority, lawB, 5, 4, 1);
        CrossLawRunner runner(*factoryB, authority);
        auto factoryA = std::make_shared<ServingWindowCommandFactory>(authority, lawA, 5, 4, 1);
        ServingWindowDecodeBridge bridge(factoryA, runner);
        ServingRowProgress r0(authority, 0, false, false);
        const auto result = bridge.decodeWindow(r0);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.error().find("cross-operation or intent mismatch"),
                  std::string::npos);
        EXPECT_NE(result.error().find("state violation"), std::string::npos); // joined report
        EXPECT_FALSE(authority.mayExecute()); // the impl DID complete A
    }

    /** @brief WINDOW INTENT IS BOUND: an impl completing a DIFFERENT window
     *         (start 1 -> bound 15) for a start-0 call is REJECTED (the complete
     *         command intent, not only the identity tuple). */
    TEST(ServingWindowDecode, BridgeRejectsWindowIntentMismatch)
    {
        auto authority = makeGrantAuthority();
        auto [ad, law] = makeLaw();
        class WindowMismatchRunner2 : public MockOrchestrationRunner
        {
        public:
            WindowMismatchRunner2(ServingWindowCommandFactory &f) : f_(f) {}
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                (void)request;
                ServingRowProgress r1(request.authority, 1, false, false);
                auto cmd = f_.forRow(r1); // same authority/law/tx, start 1
                return completeWindow(request.authority, cmd, 15,
                                      WindowTermination::Continue);
            }
            ServingWindowCommandFactory &f_;
        };
        auto factory = std::make_shared<ServingWindowCommandFactory>(authority, law, 5, 4, 1);
        WindowMismatchRunner2 runner(*factory);
        ServingWindowDecodeBridge bridge(factory, runner);
        ServingRowProgress r0(authority, 0, false, false);
        const auto result = bridge.decodeWindow(r0);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.error().find("cross-operation or intent mismatch"),
                  std::string::npos);
        EXPECT_NE(result.error().find("state violation"), std::string::npos); // joined report
        EXPECT_FALSE(authority.mayExecute()); // the impl DID complete A
    }

    /** @brief ORIGINAL-ISSUANCE STATE: completing grant 7 then issuing a FRESH
     *         grant 8 and returning a failure IS reported as a state violation
     *         (mayExecute alone would misread it as untouched). */
    TEST(ServingWindowDecode, BridgeReportsConsumeThenRefreshAsViolation)
    {
        auto authority = makeGrantAuthority();
        auto [ad, law] = makeLaw();
        class ConsumeAndRefreshRunner : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                auto consumed = completeWindow(request.authority, request, 16,
                                               WindowTermination::Continue);
                (void)consumed;
                WindowGrant g2 = request.authority.current();
                g2.grant_id = 8;
                g2.state = GrantState::Outstanding;
                request.authority.issue(g2); // fresh outstanding grant (original consumed)
                return ServingWindowDecodeResult::failure("internal error");
            }
        };
        auto factory = std::make_shared<ServingWindowCommandFactory>(authority, law, 5, 4, 1);
        ConsumeAndRefreshRunner runner;
        ServingWindowDecodeBridge bridge(factory, runner);
        ServingRowProgress r0(authority, 0, false, false);
        const auto result = bridge.decodeWindow(r0);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.error().find("grant consumed without a completion"),
                  std::string::npos);
        EXPECT_TRUE(authority.mayExecute()); // fresh grant present — but ORIGINAL consumed
    }

    /** @brief INGRESS REJECTIONS ARE NOT MISATTRIBUTED: an already-resolved
     *         command is an ingress rejection (valid_before false) — the bridge
     *         passes it through without a state-violation label. */
    TEST(ServingWindowDecode, BridgeDoesNotMisattributeIngressRejection)
    {
        auto authority = makeGrantAuthority();
        authority.resolve();
        auto [ad, law] = makeLaw();
        auto factory = std::make_shared<ServingWindowCommandFactory>(authority, law, 5, 4, 1);
        MockOrchestrationRunner runner;
        ServingWindowDecodeBridge bridge(factory, runner);
        ServingRowProgress r0(authority, 0, false, false);
        const auto result = bridge.decodeWindow(r0);
        EXPECT_FALSE(result.ok());
        EXPECT_EQ(result.error().find("state violation"), std::string::npos);
        EXPECT_NE(result.error().find("grant not outstanding"), std::string::npos);
    }


    /** @brief STATIC REPLACEMENT COUNTEREXAMPLE: complete A (grant 7), then
     *         legally move-assign a same-instance B (grant id 7, Outstanding)
     *         into A, then return a plain failure — the ORIGINAL issuance (its
     *         authority UID) is gone, so the state violation is reported. */
    TEST(ServingWindowDecode, BridgeReportsStaticReplacementAsViolation)
    {
        auto authorityA = makeGrantAuthority();
        auto [adA, lawA] = makeLaw();
        class ReplaceRunner : public MockOrchestrationRunner
        {
        public:
            ReplaceRunner(WindowGrantAuthority &replacement) : replacement_(replacement) {}
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                // Consume A's grant, then legally replace A with B (same instance,
                // same grant id 7, Outstanding).
                auto consumed = completeWindow(request.authority, request, 16,
                                               WindowTermination::Continue);
                (void)consumed;
                request.authority = std::move(replacement_);
                return ServingWindowDecodeResult::failure("internal error");
            }
            WindowGrantAuthority &replacement_;
        };
        WindowGrant gB;
        gB.grant_id = 7; // SAME grant id as A
        gB.slot = 0;
        gB.generation = 1;
        gB.window = IntrinsicWindowPlan{16, 32};
        gB.destination_reservation = 9;
        gB.serving_instance = 1;
        gB.state = GrantState::Outstanding;
        auto authorityB = makeGrantAuthority(gB); // the replacement (same instance)
        auto factory = std::make_shared<ServingWindowCommandFactory>(authorityA, lawA, 5, 4, 1);
        ReplaceRunner runner(authorityB);
        ServingWindowDecodeBridge bridge(factory, runner);
        ServingRowProgress r0(authorityA, 0, false, false);
        const auto result = bridge.decodeWindow(r0);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.error().find("state violation"), std::string::npos);
        EXPECT_NE(result.error().find("grant consumed without a completion"),
                  std::string::npos);
    }


    /** @brief CHANGED-ORIGINAL CANCELLATION CONTROL: the impl consumes A, then
     *         cancels B and returns B's cancellation — the joined diagnostic
     *         reports BOTH the intent mismatch AND the consumed original. */
    TEST(ServingWindowDecode, BridgeReportsChangedOriginalCancellation)
    {
        auto authorityA = makeGrantAuthority();
        WindowGrant gB;
        gB.grant_id = 7;
        gB.slot = 1;
        gB.generation = 1;
        gB.window = IntrinsicWindowPlan{16, 32};
        gB.destination_reservation = 9;
        gB.serving_instance = 1;
        gB.state = GrantState::Outstanding;
        auto authorityB = makeGrantAuthority(gB);
        auto [adA, lawA] = makeLaw();
        auto [adB, lawB] = makeLaw();
        class ConsumeThenCancelB : public MockOrchestrationRunner
        {
        public:
            ConsumeThenCancelB(ServingWindowCommandFactory &f, WindowGrantAuthority &a)
                : f_(f), a_(a) {}
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                auto consumed = completeWindow(request.authority, request, 16,
                                               WindowTermination::Continue);
                (void)consumed;
                auto cmdB = f_.forRow(ServingRowProgress(a_, 0, false, false));
                return cancelWindow(a_, cmdB); // B's cancellation for A's call
            }
            ServingWindowCommandFactory &f_;
            WindowGrantAuthority &a_;
        };
        auto factoryB = std::make_shared<ServingWindowCommandFactory>(authorityB, lawB, 5, 4, 1);
        ConsumeThenCancelB runner(*factoryB, authorityB);
        auto factoryA = std::make_shared<ServingWindowCommandFactory>(authorityA, lawA, 5, 4, 1);
        ServingWindowDecodeBridge bridge(factoryA, runner);
        ServingRowProgress r0(authorityA, 0, false, false);
        const auto result = bridge.decodeWindow(r0);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.error().find("cross-operation or intent mismatch"),
                  std::string::npos);
        EXPECT_NE(result.error().find("state violation"), std::string::npos); // joined (A consumed)
        EXPECT_FALSE(authorityA.mayExecute());
    }

} // namespace llaminar2::test

/** @brief TEST-ONLY exhaustion probes (friends; definitions live HERE, not in
     *         production headers — no public rewind controls). */




