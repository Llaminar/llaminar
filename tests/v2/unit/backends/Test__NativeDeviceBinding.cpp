/**
 * @file Test__NativeDeviceBinding.cpp
 * @brief Device-free replay of actual thread-device binding across native collective kinds.
 *
 * The native state oracle is deliberately independent of prior operation names.
 * External selection and reset/reuse change that oracle directly. Production
 * checked binding must observe it on every call while retaining exact query/set
 * errors. Frozen original coordinator bodies supply original-red controls.
 */
#include "backends/NativeDeviceBinding.h"
#include <gtest/gtest.h>
#include <array>
#include <string>
#include <thread>

namespace
{
    /** @brief Recorded native thread state and exact query/selection statuses. */
    struct RecordedNative
    {
        int current = 0, query_status = 0, selection_status = 0, queries = 0, selections = 0;
        /** @brief A failed native query may still overwrite the output with zero. */
        int query(int *ordinal) { ++queries; *ordinal = query_status ? 0 : current; return query_status; }
        /** @brief A failed native selection leaves the prior actual state untouched. */
        int select(int ordinal) { ++selections; if (!selection_status) current = ordinal; return selection_status; }
    };
    /** @brief Run the same actual production decision used by both coordinators and HIP guard. */
    auto bind(RecordedNative &native, int ordinal)
    {
        return llaminar2::bindNativeDevice(ordinal,
            [&](int *observed) { return native.query(observed); },
            [&](int selected) { return native.select(selected); });
    }
#include "fixtures/OriginalDeviceBindingBodies.inc"
}

/** @test Alternating operation kinds cannot reuse another operation's remembered device. */
TEST(NativeDeviceBinding, AlternatingCollectivesAndExternalSelectionKeepExactOwner)
{
    struct Call { const char *kind; int ordinal; };
    constexpr std::array calls{Call{"allreduce", 0}, Call{"allgather", 1}, Call{"allreduce", 0},
        Call{"broadcast", 8}, Call{"reduce", 0}, Call{"grouped-p2p", 128}, Call{"allgather", 0}};
    RecordedNative native;
    for (int replay = 0; replay < 20; ++replay)
        for (const auto call : calls)
        {
            SCOPED_TRACE(call.kind);
            const int previous = native.current;
            const int selections = native.selections;
            const int queries = native.queries;
            const auto result = bind(native, call.ordinal);
            ASSERT_TRUE(result.succeeded());
            EXPECT_EQ(native.current, call.ordinal);
            EXPECT_EQ(native.queries, queries + 1);
            EXPECT_EQ(native.selections, selections + (previous != call.ordinal));
            native.current = 71; // Another public runtime/library path selected a foreign participant.
            ASSERT_TRUE(bind(native, call.ordinal).succeeded());
            EXPECT_EQ(native.current, call.ordinal);
        }
}

/** @test Reset/reuse never leaves a successful hit in host shadow state. */
TEST(NativeDeviceBinding, ResetReuseObservesActualStateAndSameOwnerSkipsOnlySelection)
{
    RecordedNative native;
    ASSERT_TRUE(bind(native, 64).succeeded());
    const int selected = native.selections;
    EXPECT_EQ(bind(native, 64).step, llaminar2::NativeDeviceBindingStep::AlreadySelected);
    EXPECT_EQ(native.selections, selected);
    native.current = 0; // A fresh/reset runtime has its own native current state.
    ASSERT_TRUE(bind(native, 64).succeeded());
    EXPECT_EQ(native.selections, selected + 1);
    EXPECT_EQ(native.current, 64);
}

/** @test Get failure is not hidden by set; set failure does not manufacture a successful remembered owner. */
TEST(NativeDeviceBinding, NativeQueryAndSelectionFailuresRemainExact)
{
    RecordedNative native;
    native.current = 128;
    native.query_status = 7;
    auto result = bind(native, 0);
    EXPECT_FALSE(result.succeeded());
    EXPECT_EQ(result.step, llaminar2::NativeDeviceBindingStep::CurrentDeviceQuery);
    EXPECT_EQ(result.native_status, 7);
    EXPECT_EQ(native.current, 128);
    EXPECT_EQ(native.selections, 0);
    native.query_status = 0;
    native.selection_status = 23;
    result = bind(native, 0);
    EXPECT_FALSE(result.succeeded());
    EXPECT_EQ(result.step, llaminar2::NativeDeviceBindingStep::DeviceSelection);
    EXPECT_EQ(result.native_status, 23);
    EXPECT_EQ(native.current, 128);
    native.selection_status = 0;
    ASSERT_TRUE(bind(native, 0).succeeded());
    EXPECT_EQ(native.current, 0);
    EXPECT_THROW((void)bind(native, -1), std::invalid_argument);
    EXPECT_THROW((void)llaminar2::bindNativeDevice(0,
        [](int *) { return 0; }, [](int) { return 0; }), std::runtime_error);
}

/** @test Frozen original native coordinator bodies accept the call but leave the wrong current device. */
TEST(NativeDeviceBinding, OriginalCoordinatorCachesFailAlternatingKindsNegativeControl)
{
    const auto replay = [](auto allreduce, auto allgather) {
        // Fresh native host thread also gives the original function-local TLS a fresh lifetime.
        std::thread worker([&] {
            RecordedNative native;
            ASSERT_TRUE(allreduce(native, 0));
            ASSERT_TRUE(allgather(native, 1));
            ASSERT_TRUE(allreduce(native, 0));
            EXPECT_EQ(native.current, 1);
            EXPECT_NE(native.current, 0);
            EXPECT_EQ(native.selections, 2);
        });
        worker.join();
    };
    replay(originalCUDAAllreduce, originalCUDAAllgather);
    replay(originalHIPAllreduce, originalHIPAllgather);
}
