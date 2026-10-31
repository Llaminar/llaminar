/**
 * @file Test__NativeDeviceBindingNative.cpp
 * @brief Native current-device replay for the actual checked collective selector and HIP guard.
 *
 * Independent native get-device observations authenticate every requested owner
 * while actual set-device calls alternate visible participants and bypass the
 * helper deliberately. This fixture creates no allocations, streams, kernels,
 * communicators or context resets. Its canonical entry scope and short lifecycle
 * borrow enclose every native observation and the final selection restoration.
 */
#include "backends/NativeDeviceBinding.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "utils/DebugEnv.h"
#include <gtest/gtest.h>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

#if defined(GPU_CONTEXT_TEST_BACKEND_CUDA)
#include <cuda_runtime.h>
#elif defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
#include <hip/hip_runtime.h>
#include "backends/rocm/HipDeviceGuard.h"
#else
#error Native device binding replay requires an explicit backend target
#endif

namespace llaminar2::test
{
    namespace
    {
#if defined(GPU_CONTEXT_TEST_BACKEND_CUDA)
        constexpr const char *backend_name = "CUDA";
        constexpr int no_device = static_cast<int>(cudaErrorNoDevice);
        /** @brief Read the actual process-visible native inventory count. */
        int nativeCount(int *count) { return static_cast<int>(cudaGetDeviceCount(count)); }
        /** @brief Read actual current-thread selection, independently of the production helper. */
        int nativeCurrent(int *ordinal) { return static_cast<int>(cudaGetDevice(ordinal)); }
        /** @brief Perform a real runtime selection, including deliberate external selection. */
        int nativeSelect(int ordinal) { return static_cast<int>(cudaSetDevice(ordinal)); }
        /** @brief Preserve the exact runtime error description. */
        const char *nativeError(int status) { return cudaGetErrorString(static_cast<cudaError_t>(status)); }
        /** @return Explicit startup permission before any native discovery. */
        bool startupEnabled() { return debugEnv().backend_startup.cudaEnabled(); }
#else
        constexpr const char *backend_name = "ROCm";
        constexpr int no_device = static_cast<int>(hipErrorNoDevice);
        /** @brief Read the actual process-visible native inventory count. */
        int nativeCount(int *count) { return static_cast<int>(hipGetDeviceCount(count)); }
        /** @brief Read actual current-thread selection, independently of the production helper. */
        int nativeCurrent(int *ordinal) { return static_cast<int>(hipGetDevice(ordinal)); }
        /** @brief Perform a real runtime selection, including deliberate external selection. */
        int nativeSelect(int ordinal) { return static_cast<int>(hipSetDevice(ordinal)); }
        /** @brief Preserve the exact runtime error description. */
        const char *nativeError(int status) { return hipGetErrorString(static_cast<hipError_t>(status)); }
        /** @return Explicit startup permission before any native discovery. */
        bool startupEnabled() { return debugEnv().backend_startup.rocmEnabled(); }
#endif
        /** @brief Reject a failed actual native operation without manufacturing a device. */
        void requireNative(int status, const char *operation)
        {
            if (status != 0)
                throw std::runtime_error(std::string(backend_name) + " native binding replay " + operation +
                    " failed: status=" + std::to_string(status) + " error=" + nativeError(status));
        }
        /** @brief Restore exact original thread selection before its lifecycle borrow retires. */
        class OriginalSelection final
        {
        public:
            /** @brief Capture actual original selection after successful visible-device discovery. */
            OriginalSelection() { requireNative(nativeCurrent(&original_), "get original device"); }
            OriginalSelection(const OriginalSelection &) = delete;
            OriginalSelection &operator=(const OriginalSelection &) = delete;
            /** @brief A failed restoration is fatal, including on assertion or exception unwinding. */
            ~OriginalSelection()
            {
                const int status = nativeSelect(original_);
                if (status != 0)
                {
                    std::fprintf(stderr, "[ERROR] %s native binding restoration device=%d status=%d error=%s\n",
                        backend_name, original_, status, nativeError(status));
                    std::terminate();
                }
            }
        private:
            int original_ = -1;
        };
        /**
         * @brief Replay the actual production helper through real native operations.
         * @param ordinal Exact actually visible participant.
         * @note Independent reads verify the selected state; counters observe the
         * exact native get/set adapters, never substitute statuses or hardware.
         */
        void proveBinding(int ordinal)
        {
            int before = -1, after = -1;
            requireNative(nativeCurrent(&before), "get prior device");
            unsigned queries = 0, selections = 0;
            const auto result = bindNativeDevice(ordinal,
                [&](int *current) { ++queries; return nativeCurrent(current); },
                [&](int selected) { ++selections; return nativeSelect(selected); });
            requireNative(result.native_status,
                result.step == NativeDeviceBindingStep::CurrentDeviceQuery ? "query current device" : "select requested device");
            EXPECT_EQ(queries, 1u);
            EXPECT_EQ(selections, before == ordinal ? 0u : 1u);
            requireNative(nativeCurrent(&after), "get selected device");
            EXPECT_EQ(after, ordinal);
        }
    }

    /** @test Actual native selectors and the HIP guard cannot reuse a stale operation-local device. */
    TEST(NativeDeviceBindingNative, VisibleAlternationAndRawExternalSelectionRetainExactOwner)
    {
        if (!startupEnabled())
            GTEST_SKIP() << "UNQUALIFIED: " << backend_name << " startup was explicitly disabled";
        auto native_lifetime = GPUDeviceProcessLifetime::acquireNativeContext();
        int count = -1;
        const int status = nativeCount(&count);
        if (status == no_device)
            GTEST_SKIP() << "UNQUALIFIED: " << backend_name << " device alternation has no visible devices";
        requireNative(status, "get device count");
        ASSERT_GE(count, 0) << "Successful native enumeration returned an invalid count";
        if (count < 2)
            GTEST_SKIP() << "UNQUALIFIED: " << backend_name << " device alternation requires two visible devices";
        OriginalSelection original;
        for (int replay = 0; replay < 4; ++replay)
            for (int ordinal = 1; ordinal < count; ++ordinal)
            {
                SCOPED_TRACE(::testing::Message() << backend_name << " visible=" << count << " ordinal=" << ordinal);
                proveBinding(0);
                proveBinding(ordinal); // Includes actual visible 0 -> 1 -> 0 on every replay.
                proveBinding(0);
                requireNative(nativeSelect(ordinal), "external raw selection");
                proveBinding(0);
                proveBinding(0); // A true same-owner observation skips native set, while still querying actual state.
#if defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
                requireNative(HipDeviceGuard::forceSetDevice(0), "guard forced device zero");
                requireNative(nativeSelect(ordinal), "external raw selection bypassing HIP guard");
                requireNative(HipDeviceGuard::setDevice(0), "HIP guard after external raw selection");
                int observed = -1;
                requireNative(nativeCurrent(&observed), "get HIP guard selected device");
                EXPECT_EQ(observed, 0);
                requireNative(nativeSelect(0), "external raw device zero");
                requireNative(HipDeviceGuard::setDevice(ordinal), "HIP guard alternate device");
                requireNative(nativeCurrent(&observed), "get HIP guard alternate device");
                EXPECT_EQ(observed, ordinal);
#endif
            }
    }
}
