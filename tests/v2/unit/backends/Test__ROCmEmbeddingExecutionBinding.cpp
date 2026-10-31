/**
 * @file Test__ROCmEmbeddingExecutionBinding.cpp
 * @brief Device-free replay of the production embedding owner's binding state machine.
 *
 * Recorded stream-owner observations exercise the actual production-used class;
 * fake workspace records are borrowed only for identity, never native execution.
 * Failed observations and foreign-device rebinding cannot replace valid state.
 */
#include "kernels/rocm/ops/ROCmEmbeddingExecutionBinding.h"
#include <gtest/gtest.h>
#include <array>

namespace llaminar2::test
{
    namespace
    {
        struct WorkspaceRecord { int value; }; ///< Replay-only borrowed identity, with no runtime.
        using Binding = ROCmEmbeddingExecutionBinding<WorkspaceRecord>;
    }

    /** @test Explicit ownership replaces ordinal-zero and ambient-device substitution. */
    TEST(ROCmEmbeddingExecutionBinding, ExplicitOwnerAndLaunchOrdinalRemainExact)
    {
        EXPECT_THROW((void)Binding(-1), std::invalid_argument);
        EXPECT_THROW((void)Binding(-2), std::invalid_argument);
        for (int owner : {0, 7, 8, 63, 64, 127})
        {
            Binding binding(owner);
            EXPECT_EQ(binding.ordinal(), owner);
            EXPECT_EQ(binding.resolveOrdinal(-1), owner);
            EXPECT_EQ(binding.resolveOrdinal(owner), owner);
            EXPECT_THROW((void)binding.resolveOrdinal(owner + 1), std::invalid_argument);
            EXPECT_THROW((void)binding.resolveOrdinal(-2), std::invalid_argument);
            EXPECT_THROW((void)binding.stream(), std::runtime_error);
            EXPECT_EQ(binding.workspace(), nullptr);
        }
    }

    /** @test Native query failure cannot publish even an apparently matching returned ordinal. */
    TEST(ROCmEmbeddingExecutionBinding, FailedObservationPreservesPriorExactBinding)
    {
        Binding binding(127);
        std::array<int, 2> streams{};
        void *original = &streams[0], *candidate = &streams[1];
        binding.bindStream(original, [](void *, int *owner) { *owner = 127; return 0; });
        for (bool writes_owner : {false, true})
        {
            unsigned observations = 0;
            try
            {
                binding.bindStream(candidate, [&](void *stream, int *owner) {
                    ++observations; EXPECT_EQ(stream, candidate);
                    if (writes_owner) *owner = 127;
                    return 702;
                });
                FAIL() << "Failed stream-owner query replaced admitted binding";
            }
            catch (const std::runtime_error &error)
            {
                EXPECT_STREQ(error.what(), "hipStreamGetDevice failed before ROCm embedding binding: status=702");
            }
            EXPECT_EQ(observations, 1u);
            EXPECT_EQ(binding.stream(), original);
        }
        unsigned observations = 0;
        EXPECT_THROW(binding.bindStream(nullptr, [&](void *, int *) {
            ++observations; return 0;
        }), std::invalid_argument);
        EXPECT_EQ(observations, 0u);
        EXPECT_EQ(binding.stream(), original);
    }

    /** @test Foreign streams/workspaces cannot become a last-binding substitute. */
    TEST(ROCmEmbeddingExecutionBinding, ForeignRebindPreservesStreamAndWorkspace)
    {
        Binding binding(64);
        int first = 0, other = 1;
        WorkspaceRecord local{0}, foreign{1};
        binding.bindStream(&first, [](void *, int *owner) { *owner = 64; return 0; });
        binding.bindWorkspace(&local, DeviceId::rocm(64));
        EXPECT_THROW(binding.bindStream(&other, [](void *, int *owner) {
            *owner = 0; return 0;
        }), std::invalid_argument);
        EXPECT_THROW(binding.bindWorkspace(&foreign, DeviceId::rocm(0)), std::invalid_argument);
        EXPECT_THROW(binding.bindWorkspace(&foreign, DeviceId::cuda(64)), std::invalid_argument);
        EXPECT_THROW(binding.bindWorkspace(&foreign, DeviceId::cpu()), std::invalid_argument);
        EXPECT_EQ(binding.stream(), &first);
        EXPECT_EQ(binding.workspace(), &local);
        // The old map miss/query-failure branch returned the most recently
        // assigned pointer. This recorded original-code control names exactly
        // the wrong identity that production now rejects before committing.
        const auto old_lookup = [](auto *local_binding, auto *last, auto query) {
            int current = -1;
            if (query(&current) == 0 && current == 64)
                return local_binding; // Replay the original exact map hit.
            return last; // Original failed-current-query/missing-entry substitution.
        };
        const auto *former_last_workspace = old_lookup(&local, &foreign, [](int *) { return 702; });
        EXPECT_NE(binding.workspace(), former_last_workspace);
        void *former_last_stream = old_lookup(&first, &other, [](int *current) { *current = 127; return 0; });
        EXPECT_NE(binding.stream(), former_last_stream);
    }

    /** @test Clear/rebind changes borrowed state without changing topology or participant. */
    TEST(ROCmEmbeddingExecutionBinding, ClearAndRepeatedRebindKeepImmutableOwner)
    {
        Binding binding(8);
        WorkspaceRecord workspace{8};
        int first = 0, next = 1;
        unsigned observations = 0;
        const auto query = [&](void *, int *owner) { ++observations; *owner = 8; return 0; };
        binding.bindWorkspace(&workspace, DeviceId::rocm(8));
        for (int replay = 0; replay < 20; ++replay)
        {
            binding.bindStream(replay % 2 ? &first : &next, query);
            const auto before = observations;
            EXPECT_NE(binding.stream(), nullptr);
            EXPECT_EQ(binding.workspace(), &workspace);
            EXPECT_EQ(observations, before); // Reads perform no query.
            binding.clearStream();
            EXPECT_THROW((void)binding.stream(), std::runtime_error);
            EXPECT_EQ(binding.workspace(), &workspace);
            EXPECT_EQ(binding.ordinal(), 8);
        }
        binding.bindWorkspace(nullptr, DeviceId::invalid());
        EXPECT_EQ(binding.workspace(), nullptr);
        EXPECT_EQ(observations, 20u);
    }
}
