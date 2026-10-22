/**
 * @file ROCmNativeVNNISerialGeometry.h
 * @brief Fixed-arithmetic native-Q8 serial launch geometry and construction evidence.
 *
 * Output columns are independent work. These five geometries only change their
 * lane/workgroup ownership; source decoding, activation precision, K partitions
 * and ordered partial folding remain the production serial-M1 program. An
 * explicit diagnostic scope requires pre-capture resource admission and belongs
 * to one host construction thread. Ordinary inference keeps TN128/CPT1.
 * Receipts describe enqueued/captured topology, never live execution state or
 * replay counts. No API here allocates execution storage or installs a winner.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace llaminar2::rocm
{
/** @brief Reviewed arithmetic-neutral specializations of the existing serial kernel. */
enum class NativeQ8SerialGeometry : std::uint8_t
{
    TN64CPT1, TN128CPT1, TN256CPT1, TN128CPT2, TN256CPT2,
};

/** @brief Complete forceable geometry inventory; no nominal/no-op candidates. */
inline constexpr std::array kNativeQ8SerialGeometries{
    NativeQ8SerialGeometry::TN64CPT1, NativeQ8SerialGeometry::TN128CPT1,
    NativeQ8SerialGeometry::TN256CPT1, NativeQ8SerialGeometry::TN128CPT2,
    NativeQ8SerialGeometry::TN256CPT2};

/** @brief Physical geometry with an exact, derived workgroup width. */
struct NativeQ8SerialGeometrySpec
{
    int tile_columns; ///< Independent output columns per workgroup.
    int columns_per_thread; ///< Independent scalar accumulator chains per lane.
    int block_threads; ///< tile_columns / columns_per_thread, never rounded.
};

/**
 * @brief Resolve one reviewed geometry without touching a device.
 * @param geometry Exact enum value, not a requested occupancy heuristic.
 * @return Compiled N tiling and exact workgroup width.
 * @throws std::invalid_argument for an unknown value, never substitutes a default.
 */
[[nodiscard]] constexpr NativeQ8SerialGeometrySpec nativeQ8SerialGeometrySpec(
    NativeQ8SerialGeometry geometry)
{
    switch (geometry)
    {
    case NativeQ8SerialGeometry::TN64CPT1: return {64, 1, 64};
    case NativeQ8SerialGeometry::TN128CPT1: return {128, 1, 128};
    case NativeQ8SerialGeometry::TN256CPT1: return {256, 1, 256};
    case NativeQ8SerialGeometry::TN128CPT2: return {128, 2, 64};
    case NativeQ8SerialGeometry::TN256CPT2: return {256, 2, 128};
    }
    throw std::invalid_argument("unknown ROCm native-Q8 serial geometry");
}

/** @brief Loaded specialization resources observed outside captured execution. */
struct NativeQ8SerialKernelResources
{
    const void *function = nullptr; ///< Exact loaded function passed to HIP.
    const char *specialization = nullptr; ///< Static template spelling; pointer is authority.
    int block_threads = 0; ///< Width used for the occupancy query.
    int registers_per_thread = 0; ///< HIP compiler allocation for this function.
    std::size_t local_memory_bytes_per_thread = 0; ///< Must be zero for admission.
    std::size_t static_shared_memory_bytes = 0; ///< Compiler-owned LDS.
    int max_threads_per_block = 0; ///< Compiled launch ceiling.
    int max_active_blocks_per_sm = 0; ///< Residency at the exact block width.
};

class NativeQ8SerialGeometryAdmission;

/**
 * @brief Authenticate both exact symbols before a diagnostic capture/timing run.
 * @param geometry Reviewed geometry to inspect.
 * @param device_ordinal Exact HIP device owning the loaded functions.
 * @return Sealed immutable resource proof, without launching any kernel.
 * @throws std::invalid_argument for unknown geometry or invalid device.
 * @throws std::logic_error during capture.
 * @throws std::runtime_error for incomplete, nonresident, or private-memory resources.
 */
[[nodiscard]] NativeQ8SerialGeometryAdmission admitNativeQ8SerialGeometry(
    NativeQ8SerialGeometry geometry, int device_ordinal);

/** @brief Only the successful native resource query can construct this capability. */
class NativeQ8SerialGeometryAdmission final
{
public:
    /** @return Authenticated geometry, unchanged when copied into a scope. */
    [[nodiscard]] NativeQ8SerialGeometry geometry() const noexcept { return geometry_; }
    /** @return Device on which HIP authenticated both functions. */
    [[nodiscard]] int deviceOrdinal() const noexcept { return device_ordinal_; }
    /** @return Exact producer allocation and residency observation. */
    [[nodiscard]] const NativeQ8SerialKernelResources &producerResources() const noexcept { return producer_; }
    /** @return Exact ordered reducer allocation and residency observation. */
    [[nodiscard]] const NativeQ8SerialKernelResources &reducerResources() const noexcept { return reducer_; }

private:
    friend NativeQ8SerialGeometryAdmission admitNativeQ8SerialGeometry(NativeQ8SerialGeometry, int);
    /**
     * @brief Seal complete resources after the sole native admission succeeds.
     * @param geometry Exact reviewed specialization pair.
     * @param device Native resource owner.
     * @param producer Authenticated producer allocation and residency.
     * @param reducer Authenticated ordered reducer allocation and residency.
     */
    NativeQ8SerialGeometryAdmission(NativeQ8SerialGeometry geometry, int device,
        NativeQ8SerialKernelResources producer, NativeQ8SerialKernelResources reducer)
        : geometry_(geometry), device_ordinal_(device), producer_(producer), reducer_(reducer) {}
    NativeQ8SerialGeometry geometry_;
    int device_ordinal_;
    NativeQ8SerialKernelResources producer_;
    NativeQ8SerialKernelResources reducer_;
};

/** @brief Actual function/grid/block emitted by one serial construction call. */
struct NativeQ8SerialKernelLaunch
{
    const void *function = nullptr; ///< Exact HIP kernel function, absent for no reducer.
    const char *specialization = nullptr; ///< Stable readable compiled specialization.
    int grid_x = 0, grid_y = 0, grid_z = 0; ///< Actual dispatch geometry.
    int block_x = 0, block_y = 0, block_z = 0; ///< Actual workgroup geometry.
};

/** @brief Fixed-size immutable observation of one actual captured/enqueued operation. */
struct NativeQ8SerialLaunch
{
    NativeQ8SerialGeometry geometry = NativeQ8SerialGeometry::TN128CPT1;
    int physical_n = 0; ///< Actual descriptor/output stride; never policy N.
    int arithmetic_policy_n = 0; ///< Source serial-partition selector width.
    int k = 0; ///< Full unchanged reduction width.
    int device_ordinal = -1; ///< Exact HIP owner.
    std::uint8_t execution_codebook = 0; ///< This surface accepts only native19.
    std::uint8_t arithmetic_policy_codebook = 0; ///< Independent retained source policy.
    int k_partitions = 0; ///< Actual production-resolved count, not a request.
    int blocks_per_partition = 0; ///< ceil((K/32)/KB), including empty planes.
    int nonempty_partitions = 0; ///< Evidence only; never compacts the physical tree.
    std::size_t partial_elements = 0; ///< KB*N iff split; allocation capacity is separate.
    void *stream = nullptr; ///< Exact non-null construction/producer stream.
    NativeQ8SerialKernelLaunch producer;
    NativeQ8SerialKernelLaunch reducer;
};

/**
 * @brief Reject a scoped foreign-device/default-stream binding before any preparation enqueue.
 * @param device_ordinal Exact execution owner to compare with the sealed scope.
 * @param stream Exact explicit stream; null and HIP legacy/per-thread sentinels are invalid.
 * @throws std::logic_error for an incompatible active scope, clearing stale launch evidence.
 * @note Pure host validation; an absent diagnostic scope leaves production unchanged.
 */
void validateNativeQ8SerialScopeBinding(int device_ordinal, void *stream);

/** @brief Begin a thread-local launch observation; graph replay does not republish it. */
void clearNativeQ8SerialLaunch() noexcept;
/** @return Calling construction thread's latest actual native-Q8 launch, if any. */
[[nodiscard]] std::optional<NativeQ8SerialLaunch> lastNativeQ8SerialLaunch() noexcept;
/** @return Explicit calling-thread geometry scope, absent in ordinary production. */
[[nodiscard]] std::optional<NativeQ8SerialGeometry> currentNativeQ8SerialGeometry() noexcept;

/**
 * @brief Override geometry on one construction thread, preserving production arithmetic.
 *
 * Admission queries resources first; this scope performs no HIP calls. Nesting
 * restores the prior scope in strict LIFO order. Scope construction during
 * graph recording, movement/copy, foreign-device use, and KB/TW override mixing
 * are invalid. The recorded graph retains function pointers after scope exit.
 */
class ScopedNativeQ8SerialGeometry final
{
public:
    /**
     * @brief Copy a sealed resource identity into the calling construction thread.
     * @param admission Exact pre-capture loaded-symbol resource capability.
     * @throws std::logic_error During capture or when KB/TW/generated-arithmetic controls conflict.
     */
    explicit ScopedNativeQ8SerialGeometry(const NativeQ8SerialGeometryAdmission &admission);
    /** @brief Restore the calling thread's previous scope; invalid lifetime is fatal. */
    ~ScopedNativeQ8SerialGeometry() noexcept;
    ScopedNativeQ8SerialGeometry(const ScopedNativeQ8SerialGeometry &) = delete;
    ScopedNativeQ8SerialGeometry &operator=(const ScopedNativeQ8SerialGeometry &) = delete;
    ScopedNativeQ8SerialGeometry(ScopedNativeQ8SerialGeometry &&) = delete;
    ScopedNativeQ8SerialGeometry &operator=(ScopedNativeQ8SerialGeometry &&) = delete;
    /** @return This scope's immutable native resource admission. */
    [[nodiscard]] const NativeQ8SerialGeometryAdmission &admission() const noexcept { return admission_; }

private:
    NativeQ8SerialGeometryAdmission admission_;
    const ScopedNativeQ8SerialGeometry *previous_ = nullptr;
};
} // namespace llaminar2::rocm
