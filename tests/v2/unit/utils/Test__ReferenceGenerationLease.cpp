/**
 * @file Test__ReferenceGenerationLease.cpp
 * @brief Process-level lifecycle tests for parity-reference writer ownership.
 */

#include "../../utils/ReferenceGenerationLease.h"

#include <gtest/gtest.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <optional>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

namespace llaminar2::test::parity
{
    namespace
    {
        /** @brief Write one synchronization byte or terminate the child. */
        void childWriteByte(int fd, std::uint8_t value)
        {
            if (::write(fd, &value, sizeof(value)) != sizeof(value))
                _exit(2);
        }
    } // namespace

    /**
     * @brief A second process cannot publish until the current writer retires.
     *
     * The child is forked before the parent opens the lease so it cannot inherit
     * the locked file description. Pipes make the attempted/acquired edges
     * explicit and avoid timing assumptions about process scheduling.
     */
    TEST(ReferenceGenerationLeaseTest, SerializesIndependentWriterProcesses)
    {
        std::array<int, 2> begin_pipe{};
        std::array<int, 2> state_pipe{};
        ASSERT_EQ(::pipe(begin_pipe.data()), 0);
        ASSERT_EQ(::pipe(state_pipe.data()), 0);

        const pid_t child = ::fork();
        ASSERT_GE(child, 0);
        if (child == 0)
        {
            (void)::close(begin_pipe[1]);
            (void)::close(state_pipe[0]);

            std::uint8_t begin = 0;
            if (::read(begin_pipe[0], &begin, sizeof(begin)) != sizeof(begin))
                _exit(3);
            childWriteByte(state_pipe[1], 1u); // Attempting acquisition.
            ReferenceGenerationLease lease;
            childWriteByte(state_pipe[1], 2u); // Acquired after retirement.
            _exit(0);
        }

        (void)::close(begin_pipe[0]);
        (void)::close(state_pipe[1]);
        std::optional<ReferenceGenerationLease> parent_lease;
        parent_lease.emplace();

        std::uint8_t begin = 1u;
        ASSERT_EQ(::write(begin_pipe[1], &begin, sizeof(begin)), sizeof(begin));
        (void)::close(begin_pipe[1]);

        std::uint8_t state = 0;
        ASSERT_EQ(::read(state_pipe[0], &state, sizeof(state)), sizeof(state));
        ASSERT_EQ(state, 1u);

        pollfd wait_for_acquire{
            .fd = state_pipe[0],
            .events = POLLIN,
            .revents = 0,
        };
        EXPECT_EQ(::poll(&wait_for_acquire, 1, 100), 0)
            << "child acquired the reference writer lease before retirement";

        parent_lease.reset();
        ASSERT_EQ(::poll(&wait_for_acquire, 1, 2000), 1);
        ASSERT_EQ(::read(state_pipe[0], &state, sizeof(state)), sizeof(state));
        EXPECT_EQ(state, 2u);
        (void)::close(state_pipe[0]);

        int child_status = 0;
        ASSERT_EQ(::waitpid(child, &child_status, 0), child);
        ASSERT_TRUE(WIFEXITED(child_status));
        EXPECT_EQ(WEXITSTATUS(child_status), 0);
    }

} // namespace llaminar2::test::parity
