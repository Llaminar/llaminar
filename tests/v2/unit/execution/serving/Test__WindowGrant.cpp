/** @file Test__WindowGrant.cpp
 * @brief §5.2 one-window grant authority + §11 saturating age contracts.
 *
 * Sole non-copyable (ownership-transfer-only) authority: at most one
 * outstanding grant, no same-ID reuse, no unbacked/wrong-instance issuance,
 * default grants never execute and cannot be resolved/cancelled; age
 * saturates instead of wrapping. Device-free.
 */

#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

#include "execution/serving/WindowGrant.h"

namespace llaminar2::test
{
    static WindowGrant makeGrant(std::uint64_t id, std::uint32_t slot, std::uint64_t generation)
    {
        WindowGrant grant;
        grant.grant_id = id;
        grant.slot = slot;
        grant.generation = generation;
        grant.window = IntrinsicWindowPlan{16, 128};
        grant.destination_reservation = 0xabcd;
        grant.serving_instance = 7;
        return grant;
    }

    TEST(WindowGrant, AuthorityIsSoleNonCopyablePerBindingOwner)
    {
        WindowGrantAuthority authority(7);
        EXPECT_FALSE(authority.mayExecute()); // default (unissued) grant never executes
        authority.issue(makeGrant(1, 0, 1));
        EXPECT_TRUE(authority.mayExecute());
        // Sole authority: NO outstanding replacement, even cross-binding.
        EXPECT_THROW(authority.issue(makeGrant(2, 1, 1)), std::logic_error);
        EXPECT_THROW(authority.issue(makeGrant(1, 0, 1)), std::logic_error); // same-ID reuse
        authority.resolve();
        EXPECT_FALSE(authority.mayExecute());
        authority.issue(makeGrant(3, 0, 1)); // after resolution: re-issue allowed
        EXPECT_TRUE(authority.mayExecute());
        // NO copies; ownership-transfer moves are the sole way the sole authority
        // moves (e.g., into per-row storage).
        EXPECT_TRUE(std::is_copy_constructible_v<WindowGrantAuthority> == false);
        EXPECT_TRUE(std::is_move_constructible_v<WindowGrantAuthority> == true);
    }

    TEST(WindowGrant, DefaultGrantIsInvalidAndExactlyOnceTransitions)
    {
        WindowGrant g;
        EXPECT_FALSE(grantMayExecute(g)); // id 0 / no quantum
        g = makeGrant(1, 0, 1);
        EXPECT_TRUE(grantMayExecute(g));
        resolveGrant(g);
        EXPECT_FALSE(grantMayExecute(g));
        EXPECT_THROW(resolveGrant(g), std::logic_error);
        g = makeGrant(4, 0, 1);
        cancelGrant(g);
        EXPECT_THROW(cancelGrant(g), std::logic_error);
        EXPECT_FALSE(grantMayExecute(g));
    }

    TEST(WindowGrant, RejectsZeroIdAndZeroQuantum)
    {
        WindowGrantAuthority authority(7);
        EXPECT_THROW(authority.issue(makeGrant(0, 0, 1)), std::logic_error);
        auto g = makeGrant(1, 0, 1);
        g.window.quantum = 0;
        EXPECT_THROW(authority.issue(g), std::logic_error);
    }

    TEST(ServingAge, SaturatesInsteadOfWrapping)
    {
        ServingAge age;
        age.value = std::numeric_limits<std::uint64_t>::max();
        age.tick();
        EXPECT_EQ(age.value, std::numeric_limits<std::uint64_t>::max());
        age.value = 0;
        age.tick();
        EXPECT_EQ(age.value, 1u);
    }

    TEST(WindowGrant, IDReuseSequenceIsRefused)
    {
        WindowGrantAuthority authority(7);
        authority.issue(makeGrant(1, 0, 1));
        authority.resolve();
        authority.issue(makeGrant(2, 0, 1)); // new ID after resolution
        authority.resolve();
        EXPECT_THROW(authority.issue(makeGrant(1, 0, 1)), std::logic_error); // ID1 again
    }

    TEST(WindowGrant, CrossBindingOutstandingReplacementIsRefused)
    {
        WindowGrantAuthority authority(7);
        authority.issue(makeGrant(1, 0, 1));
        EXPECT_THROW(authority.issue(makeGrant(2, 1, 1)), std::logic_error); // different binding
        EXPECT_THROW(authority.issue(makeGrant(3, 0, 2)), std::logic_error); // new generation
    }
} // namespace llaminar2::test
