/**
 * @file GPUDeviceProcessLifetime.h
 * @brief Lexical process-entry authority for native GPU context retirement.
 *
 * Native runtimes may register process-exit callbacks during late device or
 * library initialization. A main-frame owner retires the existing context pool
 * before returning from main, independently of callback registration order.
 * Construction never probes a GPU. This authority governs lifecycle only;
 * PhysicalMemoryAuthority remains the sole physical admission/release ledger.
 */
#pragma once
#include <utility>

namespace llaminar2
{
    /**
     * @brief Own native GPU admission until one terminal process retirement.
     *
     * Establish exactly one instance in the owned child entry frame, before
     * application/test startup. Nested owners and reopening retired admission
     * are rejected. All graph/worker borrowers must retire before this scope.
     */
    class GPUDeviceProcessLifetime final
    {
    public:
        /** @brief One native worker admission held through its complete cleanup. */
        class NativeContextLease final
        {
        public:
            NativeContextLease() noexcept = default;
            NativeContextLease(const NativeContextLease &) = delete;
            NativeContextLease &operator=(const NativeContextLease &) = delete;
            /** @brief Transfer the single cleanup obligation without duplicating it. */
            NativeContextLease(NativeContextLease &&other) noexcept
                : active_(std::exchange(other.active_, false)) {}
            NativeContextLease &operator=(NativeContextLease &&) = delete;
            /** @brief Release only after native worker cleanup/member destruction. */
            ~NativeContextLease() noexcept;
        private:
            friend class GPUDeviceProcessLifetime;
            explicit NativeContextLease(bool active) noexcept : active_(active) {}
            bool active_ = false;
        };
        /** @brief Atomically admit a native context or reject terminal retirement. */
        [[nodiscard]] static NativeContextLease acquireNativeContext();

        /** @brief Establish process-entry ownership without native runtime queries. */
        GPUDeviceProcessLifetime();
        GPUDeviceProcessLifetime(const GPUDeviceProcessLifetime &) = delete;
        GPUDeviceProcessLifetime &operator=(const GPUDeviceProcessLifetime &) = delete;
        GPUDeviceProcessLifetime(GPUDeviceProcessLifetime &&) = delete;
        GPUDeviceProcessLifetime &operator=(GPUDeviceProcessLifetime &&) = delete;
        /** @brief Retire native owners before C/C++ process-exit callbacks execute. */
        ~GPUDeviceProcessLifetime() noexcept;

        /**
         * @brief Close admission and retire the canonical pool exactly once.
         *
         * Repeated calls are idempotent. A concurrent caller waits for the same
         * retirement; recursive retirement is fatal. Outstanding borrowers or a
         * failed native release are fatal, never translated into successful exit.
         */
        void retire() noexcept;
        /** @brief Reject native worker admission outside the live entry scope. */
        static void requireActive();
        /**
         * @brief Retire the active owner at the canonical pre-MPI-finalize boundary.
         *
         * This is a no-op if no scope was ever established, so CPU-only shutdown
         * does not manufacture a GPU owner. It never reopens terminal admission.
         */
        static void retireActive() noexcept;
    private:
        /** @brief Perform one shared terminal transition for the exact lexical owner. */
        static void retireOwner(const GPUDeviceProcessLifetime *expected) noexcept;
        /** @brief Discharge one exact native worker admission after cleanup. */
        static void releaseNativeContext() noexcept;
    };
}
