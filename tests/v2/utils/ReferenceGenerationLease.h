/**
 * @file ReferenceGenerationLease.h
 * @brief Node-local serialization for heavyweight parity-reference writers.
 *
 * Production parity campaigns may run disjoint CPU, CUDA, and ROCm inference
 * jobs concurrently, but their Hugging Face reference generators all consume
 * host cores and substantial RAM. This lease gives that shared writer phase
 * one node-local authority while leaving validated-pack readers and production
 * inference fully concurrent.
 */

#pragma once

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <unistd.h>

namespace llaminar2::test::parity
{

    /**
     * @brief Exclusive node-local lease for one heavyweight reference writer.
     *
     * The lock is process-scoped through its open file description and remains
     * held until this RAII object is destroyed. Callers must validate the pack
     * before acquiring the lease, acquire it only when publication appears
     * necessary, and validate again after acquisition so a waiting process can
     * consume a peer's completed immutable publication without regenerating it.
     */
    class ReferenceGenerationLease final
    {
    public:
        /**
         * @brief Acquire the node-wide writer lease, retrying interrupted waits.
         * @throws std::runtime_error when the lock file cannot be opened or locked.
         */
        ReferenceGenerationLease()
        {
            const auto lock_path =
                std::filesystem::temp_directory_path() /
                "llaminar-production-parity-reference-generation.lock";
            fd_ = ::open(
                lock_path.c_str(), O_CREAT | O_CLOEXEC | O_RDWR, 0600);
            if (fd_ < 0)
            {
                throw std::runtime_error(
                    "cannot open parity reference-generation lease '" +
                    lock_path.string() + "': " + std::strerror(errno));
            }

            // An interrupted blocking flock has made no ownership transition;
            // retry that exact operation instead of exposing a partial state.
            while (::flock(fd_, LOCK_EX) != 0)
            {
                if (errno == EINTR)
                    continue;
                const std::string detail = std::strerror(errno);
                (void)::close(fd_);
                fd_ = -1;
                throw std::runtime_error(
                    "cannot acquire parity reference-generation lease: " +
                    detail);
            }
        }

        ReferenceGenerationLease(const ReferenceGenerationLease &) = delete;
        ReferenceGenerationLease &operator=(
            const ReferenceGenerationLease &) = delete;
        ReferenceGenerationLease(ReferenceGenerationLease &&) = delete;
        ReferenceGenerationLease &operator=(
            ReferenceGenerationLease &&) = delete;

        /** @brief Release the writer lease and close its file description. */
        ~ReferenceGenerationLease()
        {
            if (fd_ >= 0)
            {
                (void)::flock(fd_, LOCK_UN);
                (void)::close(fd_);
            }
        }

    private:
        int fd_ = -1; ///< Open file description carrying the advisory lease.
    };

} // namespace llaminar2::test::parity
