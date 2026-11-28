/**
 * @file Test__ServingLifetimes.cpp
 * @brief Unit tests for typed serving identity and lifetime state machines (design §3.1/§4).
 */

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "execution/config/ServingLifetimes.h"

namespace llaminar2::test
{
    // Value types are immutable after construction; the lease machines are
    // single-owner: copy and move construction are deleted so a second handle
    // can never independently advance the same lifetime.
    static_assert(!std::is_copy_assignable_v<ServingRequestHandle>);
    static_assert(!std::is_move_assignable_v<ServingRequestHandle>);
    static_assert(!std::is_copy_assignable_v<ServingSlotBinding>);
    static_assert(!std::is_move_assignable_v<ServingSlotBinding>);
    static_assert(!std::is_copy_constructible_v<ResponseLease>);
    static_assert(!std::is_move_constructible_v<ResponseLease>);
    static_assert(!std::is_copy_assignable_v<ResponseLease>);
    static_assert(!std::is_move_assignable_v<ResponseLease>);
    static_assert(!std::is_copy_constructible_v<ExecutionLease>);
    static_assert(!std::is_move_constructible_v<ExecutionLease>);
    static_assert(!std::is_copy_assignable_v<ExecutionLease>);
    static_assert(!std::is_move_assignable_v<ExecutionLease>);
    static_assert(!std::is_copy_constructible_v<PublicationLease>);
    static_assert(!std::is_move_constructible_v<PublicationLease>);
    static_assert(!std::is_copy_assignable_v<PublicationLease>);
    static_assert(!std::is_move_assignable_v<PublicationLease>);
    static_assert(!std::is_copy_constructible_v<ServingSlotGenerationBank>);
    static_assert(!std::is_move_constructible_v<ServingSlotGenerationBank>);

    TEST(ServingLifetimes, RequestHandleIsStableExternalIdentity)
    {
        const auto handle = ServingRequestHandle::fromExternalId(42);
        EXPECT_EQ(handle.externalId(), 42u);
        EXPECT_EQ(handle, (ServingRequestHandle::fromExternalId(42)));
        EXPECT_NE(handle, (ServingRequestHandle::fromExternalId(43)));
    }

    TEST(ServingLifetimes, SlotBindingValidatesAndUnifiesIdentity)
    {
        const auto binding = ServingSlotBinding::from(7, 3, 9);
        EXPECT_EQ(binding.servingInstance(), 7u);
        EXPECT_EQ(binding.slot(), 3);
        EXPECT_EQ(binding.generation(), 9u);
        EXPECT_EQ(binding, (ServingSlotBinding::from(7, 3, 9)));
        EXPECT_THROW(ServingSlotBinding::from(7, -1, 9), std::invalid_argument);
        EXPECT_THROW(ServingSlotBinding::from(7, 3, 0), std::invalid_argument);
    }

    TEST(ServingLifetimes, GenerationBankIsMonotonicAndNeverReuses)
    {
        ServingSlotGenerationBank bank(1, 4);
        EXPECT_EQ(bank.current(2), 0u);
        EXPECT_FALSE(bank.hasOpenGeneration(2));
        const auto first = bank.next(2);
        EXPECT_EQ(first, 1u);
        EXPECT_TRUE(bank.hasOpenGeneration(2));
        EXPECT_TRUE(bank.isCurrent(ServingSlotBinding::from(1, 2, first)));
        // Consecutive mints without retirement are refused.
        EXPECT_THROW((void)bank.next(2), std::logic_error);
        bank.retire(ServingSlotBinding::from(1, 2, first));
        EXPECT_FALSE(bank.hasOpenGeneration(2));
        // The retired generation is stale; the slot itself is safely reusable.
        EXPECT_FALSE(bank.isCurrent(ServingSlotBinding::from(1, 2, first)));
        const auto second = bank.next(2);
        EXPECT_EQ(second, 2u);
        EXPECT_TRUE(bank.isCurrent(ServingSlotBinding::from(1, 2, second)));
        EXPECT_FALSE(bank.isCurrent(ServingSlotBinding::from(1, 2, first)));
        // A foreign serving instance with the same slot/generation is rejected.
        EXPECT_FALSE(bank.isCurrent(ServingSlotBinding::from(2, 2, second)));
        EXPECT_THROW(bank.next(-1), std::invalid_argument);
        // The bank never grows beyond its admitted slot count.
        EXPECT_THROW(bank.next(9), std::out_of_range);
        EXPECT_THROW(bank.current(9), std::out_of_range);
    }

    TEST(ServingLifetimes, GenerationBankExhaustionFailsClosed)
    {
        // Explicit small ceiling: exhaustion must refuse before any wrap and
        // the bank must never reuse a generation.
        ServingSlotGenerationBank bank(1, 4, 3);
        EXPECT_EQ(bank.next(0), 1u);
        bank.retire(ServingSlotBinding::from(1, 0, 1));
        EXPECT_EQ(bank.next(0), 2u);
        bank.retire(ServingSlotBinding::from(1, 0, 2));
        EXPECT_EQ(bank.next(0), 3u);
        bank.retire(ServingSlotBinding::from(1, 0, 3));
        EXPECT_THROW((void)bank.next(0), std::overflow_error);
        EXPECT_EQ(bank.current(0), 3u);
        EXPECT_TRUE(bank.isCurrent(ServingSlotBinding::from(1, 0, 3)) == false);
    }

    TEST(ServingLifetimes, GenerationBankRequiresPositiveAdmittedSlots)
    {
        EXPECT_THROW((ServingSlotGenerationBank(1, 0)), std::invalid_argument);
    }

    TEST(ServingLifetimes, GenerationBankRetireAuthenticatesTheCurrentGeneration)
    {
        ServingSlotGenerationBank bank(7, 2);
        EXPECT_EQ(bank.servingInstance(), 7u);
        const auto first = bank.next(0);
        // A foreign instance, a wrong generation or an unopened slot cannot
        // retire the open generation.
        EXPECT_THROW(bank.retire(ServingSlotBinding::from(9, 0, first)), std::invalid_argument);
        EXPECT_THROW(bank.retire(ServingSlotBinding::from(7, 0, first + 1)), std::logic_error);
        EXPECT_THROW(bank.retire(ServingSlotBinding::from(7, 1, first)), std::logic_error);
        EXPECT_TRUE(bank.hasOpenGeneration(0));
        bank.retire(ServingSlotBinding::from(7, 0, first));
        EXPECT_FALSE(bank.hasOpenGeneration(0));
        // Retiring the same generation again is refused: it is already retired.
        EXPECT_THROW(bank.retire(ServingSlotBinding::from(7, 0, first)), std::logic_error);
        // The slot is reusable after retirement; other slots are independent.
        EXPECT_EQ(bank.next(0), 2u);
        EXPECT_EQ(bank.next(1), 1u);
    }

    TEST(ServingLifetimes, ResponseLeaseFollowsNamedTransitionsOnly)
    {
        auto lease = ResponseLease::preparing();
        EXPECT_EQ(lease.state(), ResponseLease::State::Preparing);
        EXPECT_THROW(lease.close(), std::logic_error);
        lease.transitionToQueued();
        EXPECT_THROW(lease.transitionToDelivering(), std::logic_error);
        lease.transitionToAttached();
        lease.transitionToDelivering();
        lease.close();
        EXPECT_EQ(lease.state(), ResponseLease::State::Closed);
        EXPECT_THROW(lease.transitionToQueued(), std::logic_error);
    }

    TEST(ServingLifetimes, ResponseLeaseSupportsPreExecutionCancellation)
    {
        auto lease = ResponseLease::preparing();
        lease.cancel();
        EXPECT_EQ(lease.state(), ResponseLease::State::Cancelled);
        EXPECT_THROW(lease.transitionToQueued(), std::logic_error);

        auto queued = ResponseLease::preparing();
        queued.transitionToQueued();
        queued.abandon();
        EXPECT_EQ(queued.state(), ResponseLease::State::Cancelled);
        EXPECT_THROW(queued.transitionToAttached(), std::logic_error);
    }

    TEST(ServingLifetimes, ExecutionLeaseLifecycle)
    {
        auto lease = ExecutionLease::unbound();
        EXPECT_THROW(lease.release(), std::logic_error);
        lease.admit();
        lease.install();
        lease.observeTerminal();
        lease.beginRetire();
        lease.release();
        EXPECT_EQ(lease.state(), ExecutionLease::State::Released);
    }

    TEST(ServingLifetimes, PublicationLeaseLifecycle)
    {
        auto lease = PublicationLease::reserved();
        EXPECT_THROW(lease.drain(), std::logic_error);
        lease.begin();
        lease.detach();
        lease.drain();
        EXPECT_EQ(lease.state(), PublicationLease::State::Drained);
    }

    TEST(ServingLifetimes, ValidCompositeAllowsDeliveringResponseWithReleasedExecution)
    {
        auto response = ResponseLease::preparing();
        response.transitionToQueued();
        response.transitionToAttached();
        response.transitionToDelivering();
        auto execution = ExecutionLease::unbound();
        execution.admit();
        execution.install();
        execution.observeTerminal();
        execution.beginRetire();
        execution.release();
        PublicationLease publications[]{PublicationLease::reserved()};
        publications[0].begin();
        publications[0].detach();

        RespondingComposite composite(response, execution, publications);
        EXPECT_TRUE(composite.isValid());
    }

    TEST(ServingLifetimes, InvalidCompositeRejectsClosedResponseWithResidentExecution)
    {
        auto response = ResponseLease::preparing();
        response.transitionToQueued();
        response.transitionToAttached();
        response.transitionToDelivering();
        response.close();
        auto execution = ExecutionLease::unbound();
        execution.admit();
        execution.install(); // resident

        // Construction of an invalid composite fails closed.
        EXPECT_THROW((RespondingComposite(response, execution, {})), std::logic_error);
    }

    TEST(ServingLifetimes, CompositeRejectsClosedResponseWithUndrainedPublications)
    {
        auto response = ResponseLease::preparing();
        response.transitionToQueued();
        response.transitionToAttached();
        response.transitionToDelivering();
        response.close();
        auto execution = ExecutionLease::unbound();
        execution.admit();
        execution.install();
        execution.observeTerminal();
        execution.beginRetire();
        execution.release();
        PublicationLease publications[]{PublicationLease::reserved()};
        publications[0].begin(); // in flight, not drained

        EXPECT_THROW((RespondingComposite(response, execution, publications)), std::logic_error);
        RespondingComposite empty(response, execution, {});
        EXPECT_TRUE(empty.isValid());
    }

    TEST(ServingLifetimes, CompositeAcceptsPreExecutionCancellation)
    {
        // Pre-execution cancellation: nothing was ever bound, so nothing needs
        // releasing or draining.
        auto response = ResponseLease::preparing();
        response.cancel();
        RespondingComposite valid(response, ExecutionLease::unbound(), {});
        EXPECT_TRUE(valid.isValid());
        // Holding a reserved publication while cancelling pre-execution is
        // rejected: it could never be drained from Reserved.
        PublicationLease held[]{PublicationLease::reserved()};
        EXPECT_THROW((RespondingComposite(response, ExecutionLease::unbound(), held)), std::logic_error);
        // A cancelled response may not coexist with a live execution lease.
        auto execution = ExecutionLease::unbound();
        execution.admit();
        EXPECT_THROW((RespondingComposite(response, execution, {})), std::logic_error);
    }
} // namespace llaminar2::test
