/**
 * @file NativeVNNITestPartialStorage.h
 * @brief Explicit partial storage for raw NativeVNNI correctness fixtures.
 *
 * Raw arithmetic tests own their arrays directly; participant-level tests use
 * CPUProjectionTestWorkspace and PhysicalMemoryAuthority. This helper has no
 * global or thread-local state. Performance fixtures must retain it outside
 * warmup, timing and profiler intervals, never construct it per timed call.
 */
#pragma once
#include "kernels/cpu/gemm/CPUNativeVNNIGemv.h"

namespace llaminar2::test
{
    /** @brief Exact, invocation-owned producer/reducer bank for a raw fixture. */
    class NativeVNNITestPartialStorage final
    {
    public:
        /**
         * @brief Allocate for the fixture's current worker and output-partition geometry.
         * @param packed Immutable prepared matrix selecting the serial arithmetic tree.
         * @param rows Largest row count that will reuse this bank.
         */
        NativeVNNITestPartialStorage(const cpu::native_vnni::CPUNativeVNNIPackedWeights &packed, int rows)
            : NativeVNNITestPartialStorage(cpu::native_vnni::nativeVNNIProjectionPartialEnvelopeFloats(packed, rows)) {}
        /** @brief Allocate a caller-composed exact maximum of serial projection demands. */
        explicit NativeVNNITestPartialStorage(size_t floats)
        {
            storage_.resize_uninitialized(floats);
        }
        /**
         * @brief Own the bounded worker slab for a raw descriptor bundle.
         * @param descriptors Actual immutable packed members (or CPU batched descriptors).
         * @param count Number of live descriptors, not the array's spare capacity.
         * @param rows Default row capacity; explicit routed counts override it.
         *
         * Four physical rows cover force-selected AVX512 and AVX2 policies in
         * raw tests. Performance callers construct this once before timing.
         */
        template<class Descriptor>
        NativeVNNITestPartialStorage(const Descriptor *descriptors, int count, int rows)
            : NativeVNNITestPartialStorage(bundleFloats(descriptors, count, rows)) {}
        /** @return One address shared by every worker entering the invocation. */
        std::span<float> span() noexcept { return {storage_.data(), storage_.size()}; }

        /** @return Maximum of each actual member's serial/fused slab contribution. */
        template<class Descriptor>
        static size_t bundleFloats(const Descriptor *descriptors, int count, int rows)
        {
            using namespace cpu::native_vnni;
            if (!descriptors || count <= 0 || rows <= 0)
                throw std::invalid_argument("Raw fused fixture requires complete descriptor geometry");
            size_t maximum = 0;
            for (int i = 0; i < count; ++i)
            {
                const auto &d = descriptors[i];
                const auto *packed = [&]() {
                    if constexpr (requires { d.packed; }) return d.packed;
                    else return d.kernel ? &d.kernel->packedWeights() : nullptr;
                }();
                if (!packed) throw std::invalid_argument("Raw fused fixture has an absent prepared member");
                int member_rows = rows;
                if constexpr (requires { d.rows; }) if (d.rows > 0) member_rows = d.rows;
                const int workers = nativeVNNIInvocationWorkerCount();
                const auto config = serialTileConfigForPackedMatrix(*packed,
                    cpuNativeVNNISerialEquivalentPolicyN(packed->N), packed->K, workers);
                maximum = std::max(maximum, cpuProjectionPartialEnvelopeFloats(
                    member_rows, packed->N, config.k_tiles, workers, 4));
            }
            return maximum;
        }
    private:
        AlignedVector<float> storage_; ///< Lifetime is explicit in the caller's fixture.
    };
}
