/**
 * @file MPISessionOwner.h
 * @brief Ancestor ownership of MPI runtime files and shared-memory backing.
 *
 * Production executable entrypoints and CTest share this CPU-only owner before
 * initializing MPI or runtime state. One guardian retains each actual process
 * subtree through retirement. Managed ranks reuse the launcher; external MPI
 * parents retain responsibility for their own files and live peers.
 */
#pragma once
#include <filesystem>
#include <cstdint>
#include <memory>
#include <utility>
#include <optional>
#include <vector>
#include <string>

namespace llaminar2
{
    /** @brief One explicit process-entry boundary, never a restart inside MPI initialization. */
    class MPISessionOwner final
    {
    public:
        /** @brief Original same-filesystem identity; never a PID or environment-only namespace. */
        struct SharedIdentity final {
            std::uint64_t device=0, inode=0, uid=0;
            /** @return Whether both facts identify the same original filesystem object. */
            bool operator==(const SharedIdentity &other) const noexcept
            { return device==other.device && inode==other.inode && uid==other.uid; }
            bool operator!=(const SharedIdentity &other) const noexcept { return !(*this==other); }
        };
        class SharedFile;
        /**
         * @brief Immutable borrow of the existing guardian-owned shared directory.
         *
         * This is a descriptor/name capability, not a process owner or byte ledger.
         * The ancestor retains the directory until its actual descendants exit.
         */
        class SharedDirectory final {
        public:
            /** @return Exact original directory identity for startup collective validation. */
            [[nodiscard]] SharedIdentity identity() const;
            /** @return Original diagnostic path; creation uses the retained directory descriptor. */
            [[nodiscard]] const std::string &path() const;
            /** @brief Create one exact basename with O_EXCL/O_NOFOLLOW and mode0600. */
            [[nodiscard]] SharedFile createExclusive(const std::string &basename) const;
            /** @brief Open an existing exact basename without following a symlink. */
            [[nodiscard]] SharedFile openExisting(const std::string &basename, SharedIdentity creator,
                                                  std::uint64_t expected_bytes) const;
        private:
            friend class MPISessionOwner;
            friend class SharedFile;
            struct State;
            explicit SharedDirectory(std::shared_ptr<const State> state) : state_(std::move(state)) {}
            std::shared_ptr<const State> state_;
        };
        /**
         * @brief One original mmap backing descriptor and its retained directory identity.
         *
         * The caller owns sizing/mapping/NUMA and PMA bytes. Only the creator
         * retries name retirement on destruction; every unlink authenticates the
         * original inode and never removes a replacement object.
         */
        class SharedFile final {
        public:
            SharedFile() noexcept;
            ~SharedFile();
            SharedFile(SharedFile &&) noexcept;
            SharedFile &operator=(SharedFile &&) noexcept;
            SharedFile(const SharedFile &)=delete;
            SharedFile &operator=(const SharedFile &)=delete;
            /** @return Borrowed original file descriptor; ownership stays in this object. */
            [[nodiscard]] int descriptor() const noexcept;
            /** @return Original object identity for protocol/diagnostic assertions. */
            [[nodiscard]] SharedIdentity identity() const;
            /** @return Full original diagnostic pathname, never used as an unvalidated open. */
            [[nodiscard]] const std::string &path() const;
            /** @brief Retire only this exact name/inode; mappings and descriptor stay valid. */
            void unlink();
            /** @brief Require the original descriptor and exact live mapping extent before mmap. */
            void requireExtent(std::uint64_t expected_bytes) const;
        private:
            friend class SharedDirectory;
            struct State;
            explicit SharedFile(std::unique_ptr<State> state);
            void reset() noexcept;
            std::unique_ptr<State> state_;
        };
        /**
         * @brief Borrow original owned shared backing after existing startup authentication.
         * @return No directory for a genuinely unowned external launch; never adopt its paths.
         * @throws For present broken ownership/native publication evidence.
         */
        [[nodiscard]] static std::optional<SharedDirectory> sharedDirectory();
        /**
         * @brief Enter the original frontend under a retained ancestor before runtime work.
         * @param argc Original entrypoint argument count.
         * @param argv Original entrypoint arguments, before parsing or initialization.
         * @return Terminal child status for the outer frontend; nullopt continues an already owned/external rank.
         *
         * Only executable main entrypoints call this. An in-progress initializer
         * must reject missing ownership instead of restarting its caller.
         */
        [[nodiscard]] static std::optional<int> enterProcess(int argc, char **argv);
        /**
         * @brief Require an actual guardian-issued borrow or explicit external MPI launch.
         * @throws std::logic_error before MPI_Init if the caller bypassed the startup boundary.
         */
        static void requireProcessOwner();
        /**
         * @brief Retire only original startup loans immediately before native MPI exec.
         *
         * The same ancestor retains its files/process tree. Native MPI owns
         * rank descriptor setup; app startup loans must not partly propagate.
         */
        static void retireStartupLoansForMPILaunch();
        /** @return Helper next to the actual executable, never an arbitrary PATH match. */
        [[nodiscard]] static std::filesystem::path launcherExecutable();
        /**
         * @brief Run exact launcher argv under one independently retained ancestor.
         * @param command Original command and every authored argument.
         * @param base Existing parent of fresh mode-0700 runtime directories.
         * @param helper Actual installed helper used for the independent guardian.
         * @return Original exit status, or 128+signal for requested/actual termination.
         */
        static int launch(const std::vector<std::string> &command,
                          const std::filesystem::path &base,
                          const std::filesystem::path &helper);
        /** @brief Thin executable entry; public grammar is --session-base PATH -- command argv. */
        static int launcherMain(int argc, char **argv);
    };
}
