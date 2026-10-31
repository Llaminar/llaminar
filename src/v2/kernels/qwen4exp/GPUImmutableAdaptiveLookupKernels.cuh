/**
 * @file GPUImmutableAdaptiveLookupKernels.cuh
 * @brief Captured single-GPU adaptive row lookup using the explicit subordinate source ABI.
 *
 * Parallel validation/sort/scan/probe precedes the sole ordered policy writer.
 * Every initial hit is pinned before any victim is selected. Only compact live
 * source misses cross the mapped channel; all-hit and empty frames perform no
 * handshake. Acquired native bytes and ACKs are copied privately before mapped
 * retirement, and current admission/reservation is checked again before Ready.
 * This component adds no serving selector, topology or physical byte ledger.
 * Factory setup borrows the admitted physical clock for the unchanged deadline.
 */
#pragma once
#if defined(QFN_ADAPTIVE_CUDA)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/atomic>
#define QFN_ADAPTIVE_STREAM cudaStream_t
#define QFN_ADAPTIVE_ERROR cudaGetLastError
#define QFN_ADAPTIVE_SUCCESS cudaSuccess
#define QFN_ADAPTIVE_ERROR_TEXT cudaGetErrorString
#define QFN_ADAPTIVE_NAMESPACE cuda_adaptive_lookup
#define QFN_ADAPTIVE_FACTORY createCUDAImmutableAdaptiveLookupKernels
#else
#include <hip/hip_runtime.h>
#include "ImmutableLookupHardwareBinding.h"
#include <hip/hip_fp16.h>
#define QFN_ADAPTIVE_STREAM hipStream_t
#define QFN_ADAPTIVE_ERROR hipGetLastError
#define QFN_ADAPTIVE_SUCCESS hipSuccess
#define QFN_ADAPTIVE_ERROR_TEXT hipGetErrorString
#define QFN_ADAPTIVE_NAMESPACE rocm_adaptive_lookup
#define QFN_ADAPTIVE_FACTORY createROCmImmutableAdaptiveLookupKernels
#endif
#include "ImmutableAdaptiveLookupKernels.h"
#include "AdaptiveImmutableRowCacheOps.h"
#include "AdaptiveImmutableRowCacheControlBorrow.h"
#include "QwenAdaptiveTerminalCounters.h"
#include "ImmutableLookupDecode.h"
#include "../common/SamplingMath.h"
#include <algorithm>
#include <stdexcept>
#include <string>
namespace llaminar2::qwen4exp
{
namespace QFN_ADAPTIVE_NAMESPACE
{
using namespace adaptive_lookup;
/** @brief Native system-scope acquire of the CPU source's release publication. */
__device__ __forceinline__ std::uint64_t acquire(const std::uint64_t *p)
{
#if defined(QFN_ADAPTIVE_CUDA)
    cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> word(*const_cast<std::uint64_t *>(p));
    return word.load(cuda::memory_order_acquire);
#else
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
}
/** @brief System-release after every mapped reader/writer has completed its ordered operation. */
__device__ __forceinline__ void release(std::uint64_t *p, std::uint64_t value)
{
#if defined(QFN_ADAPTIVE_CUDA)
    cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> word(*p);
    word.store(value, cuda::memory_order_release);
#else
    __hip_atomic_store(p, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
}
/** @return Fresh acquired mapped payload, without a stale native device cache line. */
template <class T> __device__ __forceinline__ T fresh(const T *p)
{
#if defined(QFN_ADAPTIVE_CUDA)
    return __ldcv(p);
#else
    return __builtin_nontemporal_load(p);
#endif
}
/** @return Fixed-rate native wall clock for the standard30-second outstanding-response deadline. */
__device__ __forceinline__ std::uint64_t clock()
{
#if defined(QFN_ADAPTIVE_CUDA)
    std::uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
#else
    return wall_clock64();
#endif
}
/** @brief Trap an impossible overlapping frame rather than mutate another frame's pins. */
__device__ __forceinline__ void invalid()
{
#if defined(QFN_ADAPTIVE_CUDA)
    __trap();
#else
    __builtin_trap();
#endif
}
/** @return A typed subarray in an already-admitted stable byte root. */
template <class T> __device__ __forceinline__ T *at(std::byte *root, std::size_t offset)
{
    return reinterpret_cast<T *>(root + offset);
}
/** @return Sole mandatory phase authority, including the zero-optional-slot geometry. */
__device__ __forceinline__ FrameControl &frame(const AdaptiveLookupDeviceArgs &a)
{
    return *at<FrameControl>(a.storage.frame, a.plan.frame.control);
}
/** @return Actual shared CPU/device metadata operations over the exact checked roots. */
template <class Slot>
__device__ __forceinline__ Metadata<Slot> metadata(const AdaptiveNativeParameters &p,
                                                       AdaptiveLookupDeviceStorage s)
{
    return Metadata<Slot>{p.slots ? at<Slot>(s.cache, p.cache.slots) : nullptr,
                             p.slots > 1 ? at<RadixNode>(s.cache, p.cache.nodes) : nullptr,
                             p.slots ? at<OwnerControl>(s.cache, p.cache.owner) : nullptr,
                             at<FrameControl>(s.frame, p.frame.control),
                             p.slots ? at<std::uint32_t>(s.cache, p.cache.epochs) : nullptr,
                             p.slots ? at<std::uint16_t>(s.cache, p.cache.counts) : nullptr, p.slots,
                             p.protected_limit, p.lane_width, p.source_rows, p.policy.aging_accesses};
}
/** @return Actual global control/array operations for nonborrowed parallel phases. */
template <class Slot> __device__ __forceinline__ Operations<Slot> operations(
    const AdaptiveNativeParameters &p, const AdaptiveLookupDeviceStorage &s)
{
    return Operations<Slot>(metadata<Slot>(p, s));
}
/** @brief Preserve the first canonical inference failure, independent of cold prefill health0. */
__device__ __forceinline__ void failure(const AdaptiveLookupDeviceArgs &a, Status status)
{
    atomicCAS(reinterpret_cast<unsigned *>(&frame(a).status), static_cast<unsigned>(Status::Success),
              static_cast<unsigned>(status));
    atomicCAS(a.error, 0, static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
    *a.healthy = 0;
}
/** @brief Initialize only virgin metadata; native payload is never zero-filled or requantized. */
template <class Slot>
__global__ __launch_bounds__(256, 2) void initializeMetadata(AdaptiveNativeParameters p,
                                                             AdaptiveLookupDeviceStorage s)
{
    auto op = operations<Slot>(p, s);
    const auto lane = static_cast<std::uint64_t>(blockIdx.x) * 256 + threadIdx.x;
    const auto stride = static_cast<std::uint64_t>(gridDim.x) * 256;
    for (auto i = lane; i < p.slots; i += stride)
    {
        op.initializeSlot(static_cast<unsigned>(i));
        op.initializeNode(static_cast<unsigned>(i));
    }
    for (auto i = lane; i < static_cast<std::uint64_t>(p.lane_width) * 4; i += stride)
    {
        at<std::uint32_t>(s.cache, p.cache.epochs)[i] = 0;
        at<std::uint16_t>(s.cache, p.cache.counts)[i] = 0;
    }
    if (lane == 0)
    {
        op.initializeFrame();
        op.initializeOwner();
    }
}
/** @brief Publish original immutable admission, preserving warmed rows and monotonic I/O sequence. */
__global__ void admission(AdaptiveLookupDeviceStorage s, std::uint64_t generation, bool first)
{
    if (first)
        *s.protocol = ImmutableLookupDeviceState{};
    s.protocol->admissionGeneration = generation;
}
/** @brief Resolve original live count before reading a single ID or touching policy. */
template <class Slot> __global__ void begin(AdaptiveLookupDeviceArgs a)
{
    auto op = operations<Slot>(a.plan, a.storage);
    auto &f = frame(a);
    if (f.phase != Phase::Idle)
        invalid();
    // A failed begin must not recycle the previous Idle frame's reporting or
    // unique-row metadata. This resets only the new mandatory frame, not cache.
    op.initializeFrame();
    const auto rows = a.query.token_rows.activeRows();
    if (!op.begin(static_cast<std::uint64_t>(rows) * 16))
    {
        failure(a, f.status);
        return;
    }
    auto &s = *a.storage.protocol;
    if (!s.admissionGeneration || s.issuedSequence != s.consumedSequence)
    {
        failure(a, Status::Invariant);
        return;
    }
    if (s.reserved)
    {
        failure(a, Status::SourceFailure);
        return;
    }
    // Even U0 authenticates the original current interval again before Ready or
    // output; this scalar stays device-owned, never a caller-maintained copy.
    s.generation = s.admissionGeneration;
    if (*a.error)
        failure(a, Status::Cancelled);
}
/** @brief Validate all live IDs before cache mutation, output writes or source metadata publication. */
__global__ __launch_bounds__(256, 2) void validateKeys(AdaptiveLookupDeviceArgs a)
{
    auto &f = frame(a);
    if (f.status != Status::Success)
        return;
    auto *keys = at<KeyIndex>(a.storage.frame, a.plan.frame.keys_a);
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * 256 + threadIdx.x; i < f.original_ids;
         i += static_cast<std::uint64_t>(gridDim.x) * 256)
    {
        const auto id = a.query.ids[i];
        if (id < 0 || static_cast<std::uint64_t>(id) >= a.plan.source_rows)
            failure(a, Status::InvalidGeometry);
        else
            keys[i] = {static_cast<std::uint64_t>(id), static_cast<std::uint32_t>(i), 0};
    }
}
/** @return Stable total sort order on row then original coordinate; no native payload is sorted. */
__device__ __forceinline__ bool less(KeyIndex a, KeyIndex b)
{
    return a.row != b.row ? a.row < b.row : a.original_coordinate < b.original_coordinate;
}
/** @brief Independently sort fixed256-entry tiles in shared metadata; inactive lanes never touch global
 * suffix. */
__global__ __launch_bounds__(256, 2) void sortTiles(AdaptiveLookupDeviceArgs a)
{
    __shared__ KeyIndex tile[256];
    auto &f = frame(a);
    if (f.status != Status::Success)
        return;
    auto *keys = at<KeyIndex>(a.storage.frame, a.plan.frame.keys_a);
    for (auto block = static_cast<std::uint64_t>(blockIdx.x); block < (f.original_ids + 255) / 256;
         block += gridDim.x)
    {
        const auto original = block * 256 + threadIdx.x;
        tile[threadIdx.x] = original < f.original_ids ? keys[original] : KeyIndex{~std::uint64_t{0}, None, 0};
        __syncthreads();
        for (unsigned width = 2; width <= 256; width *= 2)
            for (unsigned step = width / 2; step; step /= 2)
            {
                const auto peer = threadIdx.x ^ step;
                if (peer > threadIdx.x)
                {
                    const bool ascending = (threadIdx.x & width) == 0;
                    const auto x = tile[threadIdx.x], y = tile[peer];
                    if (ascending ? less(y, x) : less(x, y))
                    {
                        tile[threadIdx.x] = y;
                        tile[peer] = x;
                    }
                }
                __syncthreads();
            }
        if (original < f.original_ids)
            keys[original] = tile[threadIdx.x];
        __syncthreads();
    }
}
/** @brief Stable bounded parallel merge; each original key owns one unique output rank. */
__global__ __launch_bounds__(256, 2) void merge(AdaptiveLookupDeviceArgs a, std::size_t source,
                                                std::size_t destination, std::uint32_t run)
{
    auto &f = frame(a);
    if (f.status != Status::Success)
        return;
    const auto *in = at<KeyIndex>(a.storage.frame, source);
    auto *out = at<KeyIndex>(a.storage.frame, destination);
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * 256 + threadIdx.x; i < f.original_ids;
         i += static_cast<std::uint64_t>(gridDim.x) * 256)
    {
        const auto pair = i / (static_cast<std::uint64_t>(run) * 2) * (static_cast<std::uint64_t>(run) * 2);
        const bool right = i >= pair + run;
        const auto own = right ? pair + run : pair;
        const auto other = right ? pair : pair + run;
        auto end = other + run;
        if (end > f.original_ids)
            end = f.original_ids;
        auto lo = other > f.original_ids ? f.original_ids : other, hi = end;
        const auto key = in[i];
        while (lo < hi)
        {
            const auto mid = (lo + hi) / 2;
            if (less(in[mid], key))
                lo = mid + 1;
            else
                hi = mid;
        }
        out[pair + (i - own) + (lo - (other > f.original_ids ? f.original_ids : other))] = key;
    }
}
/** @brief Recover each row's first original coordinate without serial search over the full frame. */
__global__ __launch_bounds__(256, 2) void firstCoordinates(AdaptiveLookupDeviceArgs a, std::size_t sorted)
{
    auto &f = frame(a);
    if (f.status != Status::Success)
        return;
    const auto *keys = at<KeyIndex>(a.storage.frame, sorted);
    auto *first = at<std::uint32_t>(a.storage.frame, a.plan.frame.first);
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * 256 + threadIdx.x; i < f.original_ids;
         i += static_cast<std::uint64_t>(gridDim.x) * 256)
    {
        const auto row = keys[i].row;
        std::uint64_t lo = 0, hi = i;
        while (lo < hi)
        {
            const auto mid = (lo + hi) / 2;
            if (keys[mid].row < row)
                lo = mid + 1;
            else
                hi = mid;
        }
        first[keys[i].original_coordinate] = keys[lo].original_coordinate;
    }
}
/** @brief Produce0/1 uniqueness flags in original policy order, preserving duplicates' original coordinates.
 */
__global__ __launch_bounds__(256, 2) void firstFlags(AdaptiveLookupDeviceArgs a)
{
    auto &f = frame(a);
    if (f.status != Status::Success)
        return;
    const auto *first = at<std::uint32_t>(a.storage.frame, a.plan.frame.first);
    auto *flags = at<std::uint32_t>(a.storage.frame, a.plan.frame.first_prefix);
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * 256 + threadIdx.x; i < f.original_ids;
         i += static_cast<std::uint64_t>(gridDim.x) * 256)
        flags[i] = first[i] == i;
}
/** @brief In-place exclusive block scan plus one exact block-total word per tile. */
__global__ __launch_bounds__(256, 2) void scanBlocks(AdaptiveLookupDeviceArgs a, std::size_t array,
                                                     std::size_t totals, std::uint32_t divisor)
{
    __shared__ std::uint32_t values[256];
    const auto &f = frame(a);
    if (f.status != Status::Success)
        return;
    const auto n = (f.original_ids + divisor - 1) / divisor;
    auto *data = at<std::uint32_t>(a.storage.frame, array);
    for (auto block = static_cast<std::uint64_t>(blockIdx.x); block < (n + 255) / 256; block += gridDim.x)
    {
        const auto i = block * 256 + threadIdx.x;
        const auto value = i < n ? data[i] : 0;
        values[threadIdx.x] = value;
        __syncthreads();
        for (unsigned step = 1; step < 256; step *= 2)
        {
            const auto add = threadIdx.x >= step ? values[threadIdx.x - step] : 0;
            __syncthreads();
            values[threadIdx.x] += add;
            __syncthreads();
        }
        if (i < n)
            data[i] = values[threadIdx.x] - value;
        if (threadIdx.x == 255)
            at<std::uint32_t>(a.storage.frame, totals)[block] = values[255];
        __syncthreads();
    }
}
/** @brief Apply scanned parent totals without reading or mutating inactive frame suffix. */
__global__ __launch_bounds__(256, 2) void addScan(AdaptiveLookupDeviceArgs a, std::size_t array,
                                                  std::size_t parent, std::uint32_t divisor)
{
    const auto &f = frame(a);
    if (f.status != Status::Success)
        return;
    const auto n = (f.original_ids + divisor - 1) / divisor;
    auto *data = at<std::uint32_t>(a.storage.frame, array);
    const auto *prefix = at<std::uint32_t>(a.storage.frame, parent);
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * 256 + threadIdx.x; i < n;
         i += static_cast<std::uint64_t>(gridDim.x) * 256)
        data[i] += prefix[i / 256];
}
/** @brief Materialize unique records in first-occurrence order and original-coordinate restoration map. */
__global__ __launch_bounds__(256, 2) void uniqueRecords(AdaptiveLookupDeviceArgs a)
{
    auto &f = frame(a);
    if (f.status != Status::Success)
        return;
    const auto *first = at<std::uint32_t>(a.storage.frame, a.plan.frame.first);
    const auto *prefix = at<std::uint32_t>(a.storage.frame, a.plan.frame.first_prefix);
    auto *mapping = at<std::uint32_t>(a.storage.frame, a.plan.frame.original_to_unique);
    auto *unique = at<UniqueRecord>(a.storage.frame, a.plan.frame.unique);
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * 256 + threadIdx.x; i < f.original_ids;
         i += static_cast<std::uint64_t>(gridDim.x) * 256)
    {
        const auto u = prefix[first[i]];
        mapping[i] = u;
        if (first[i] == i)
            unique[u] = {
                static_cast<std::uint64_t>(a.query.ids[i]), 0, static_cast<std::uint32_t>(i), None, None, 0};
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
        f.unique_rows = f.original_ids
                            ? prefix[f.original_ids - 1] + (first[f.original_ids - 1] == f.original_ids - 1)
                            : 0;
}
/** @brief Parallel read-only initial-hit discovery; no cache owner mutates before this complete frontier. */
template <class Slot> __global__ __launch_bounds__(256, 2) void probe(AdaptiveLookupDeviceArgs a)
{
    auto &f = frame(a);
    if (f.status != Status::Success)
        return;
    auto op = operations<Slot>(a.plan, a.storage);
    auto *unique = at<UniqueRecord>(a.storage.frame, a.plan.frame.unique);
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * 256 + threadIdx.x; i < f.unique_rows;
         i += static_cast<std::uint64_t>(gridDim.x) * 256)
    {
        auto &u = unique[i];
        const auto slot = op.find(u.row);
        if (f.status != Status::Success)
            continue;
        u.flags |= Probed; // Probed: actual lookup completed, even if a later frame operation fails.
        if (slot != None)
        {
            u.slot = slot;
            u.expected_slot_generation = at<Slot>(a.storage.cache, a.plan.cache.slots)[slot].generation;
            u.flags |= InitialHit;
        }
    }
}
inline constexpr std::uint32_t Held = 4U;
/** @brief Flush exclusive controls before the existing canonical first-error operation.
 * @param a Actual original frame/controller, never borrowed controller state.
 * @param borrow Lexical ordered metadata working set.
 * @param status Actual failure from this operation.
 */
template<class Slot> __device__ __forceinline__ void failBorrowed(
    const AdaptiveLookupDeviceArgs &a, OrderedControlBorrow<Slot> &borrow, Status status)
{
    borrow.publish();
    failure(a, status);
    borrow.refreshStatus();
}
/** @brief Pin the full initial frontier and retain the exact ordered queue/sketch algorithm.
 * @note Only these two small control records live in CTA-shared memory. Slots,
 * radix and sketch remain original global arrays. The one writer publishes on
 * every exit; no downstream observer runs before this kernel retires.
 */
template <class Slot> __global__ void policy(AdaptiveLookupDeviceArgs a)
{
    if (frame(a).status != Status::Success) return;
    __shared__ FrameControl workingFrame;
    __shared__ OwnerControl workingOwner;
    OrderedControlBorrow<Slot> borrow(metadata<Slot>(a.plan, a.storage), workingFrame, workingOwner);
    auto &f = borrow.frame();
    auto op = Operations<Slot>(borrow.metadata());
    auto *unique = at<UniqueRecord>(a.storage.frame, a.plan.frame.unique);
    for (std::uint64_t i = 0; i < f.unique_rows; ++i)
    {
        auto &u = unique[i];
        if (u.flags & InitialHit)
        {
            if (!op.pinInitial(u.slot, u.expected_slot_generation))
            {
                failBorrowed(a, borrow, f.status);
                return;
            }
            u.flags |= Held;
        }
    }
    if (!op.beginPolicy(f.unique_rows))
    {
        failBorrowed(a, borrow, f.status);
        return;
    }
    const auto *map = at<std::uint32_t>(a.storage.frame, a.plan.frame.original_to_unique);
    for (std::uint64_t i = 0; i < f.original_ids; ++i)
    {
        auto &u = unique[map[i]];
        const auto d = op.observe(u.row, {u.slot, u.expected_slot_generation});
        if (f.status != Status::Success)
        {
            failBorrowed(a, borrow, f.status);
            return;
        }
        u.slot = d.slot;
        u.expected_slot_generation = d.generation;
        if (d.slot != None) u.flags |= Held;
        if (!(u.flags & InitialHit) && d.slot != None) u.flags |= ReservedFill;
    }
    if (!op.prepare()) failBorrowed(a, borrow, f.status);
}
/** @brief Compact final unique source misses, after duplicates may have changed admission decisions. */
__global__ __launch_bounds__(256, 2) void missFlags(AdaptiveLookupDeviceArgs a)
{
    auto &f = frame(a);
    if (f.status != Status::Success)
        return;
    const auto *u = at<UniqueRecord>(a.storage.frame, a.plan.frame.unique);
    auto *flags = at<std::uint32_t>(a.storage.frame, a.plan.frame.miss_prefix);
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * 256 + threadIdx.x; i < f.original_ids;
         i += static_cast<std::uint64_t>(gridDim.x) * 256)
        flags[i] = i < f.unique_rows && !(u[i].flags & InitialHit);
}
/** @brief Resolve exact miss counts and original private source coordinates before publishing any metadata.
 */
template <class Slot> __global__ void beginSource(AdaptiveLookupDeviceArgs a)
{
    auto &f = frame(a);
    if (f.status != Status::Success)
        return;
    auto op = operations<Slot>(a.plan, a.storage);
    auto *u = at<UniqueRecord>(a.storage.frame, a.plan.frame.unique);
    const auto *prefix = at<std::uint32_t>(a.storage.frame, a.plan.frame.miss_prefix);
    f.source_rows =
        f.unique_rows ? prefix[f.unique_rows - 1] + (!(u[f.unique_rows - 1].flags & InitialHit)) : 0;
    auto &s = *a.storage.protocol;
    if (!s.admissionGeneration || s.generation != s.admissionGeneration ||
        s.issuedSequence != s.consumedSequence)
    {
        failure(a, Status::Invariant);
        return;
    }
    if (*a.error)
    {
        failure(a, Status::Cancelled);
        return;
    }
    if (!f.source_rows)
    {
        op.transition(Phase::Prepared, Phase::Validated);
        return;
    }
    if (s.issuedSequence == ~std::uint64_t{0})
    {
        failure(a, Status::Exhausted);
        return;
    }
    s.generation = s.admissionGeneration;
    ++s.issuedSequence;
    s.idCount = f.source_rows;
    s.nativeBytes = f.source_rows * a.plan.row_bytes;
    s.status = 0;
    auto &r = *a.channel.base.request;
    const auto seq = s.issuedSequence;
    r.sequenceComplement = ~seq;
    r.generation = s.generation;
    r.generationComplement = ~s.generation;
    r.table = a.channel.base.table;
    r.tableComplement = ~r.table;
    r.idCount = s.idCount;
    r.idCountComplement = ~s.idCount;
    r.idBytes = s.idCount * 8;
    r.idBytesComplement = ~r.idBytes;
    r.rowBytes = a.plan.row_bytes;
    r.rowBytesComplement = ~r.rowBytes;
    r.protocol = kImmutableLookupSubordinateProtocol;
    r.protocolComplement = ~r.protocol;
    __threadfence_system();
}
/** @brief Send only unique misses and their exact16B intents; every writer fences before release. */
__global__ __launch_bounds__(256, 2) void sourceRecords(AdaptiveLookupDeviceArgs a)
{
    auto &f = frame(a);
    if (f.status != Status::Success || !f.source_rows)
        return;
    auto *u = at<UniqueRecord>(a.storage.frame, a.plan.frame.unique);
    const auto *prefix = at<std::uint32_t>(a.storage.frame, a.plan.frame.miss_prefix);
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * 256 + threadIdx.x; i < f.unique_rows;
         i += static_cast<std::uint64_t>(gridDim.x) * 256)
        if (!(u[i].flags & InitialHit))
        {
            const auto response = prefix[i];
            u[i].response_coordinate = response;
            a.channel.base.ids[response] = static_cast<std::int64_t>(u[i].row);
            a.channel.intents[response] = {u[i].flags & ReservedFill
                                               ? ImmutableRowReadIntent::ReadAndInvalidate
                                               : ImmutableRowReadIntent::Read,
                                           u[i].flags & ReservedFill ? u[i].expected_slot_generation : 0};
        }
    __threadfence_system();
}
/** @brief Publish only a nonempty compact source frame; U0 leaves issued/consumed untouched. */
template <class Slot> __global__ void publish(AdaptiveLookupDeviceArgs a)
{
    auto &f = frame(a);
    if (f.status != Status::Success || !f.source_rows)
        return;
    auto op = operations<Slot>(a.plan, a.storage);
    if (!op.transition(Phase::Prepared, Phase::Published))
    {
        failure(a, f.status);
        return;
    }
    release(&a.channel.base.request->publication, a.storage.protocol->issuedSequence);
}
/** @brief One waiting CTA authenticates only112 live completion bytes, never aligned tail padding. */
__global__ __launch_bounds__(256, 2) void waitResponse(AdaptiveLookupDeviceArgs a, std::uint64_t timeout)
{
    auto &f = frame(a);
    if (f.phase != Phase::Published)
        return;
    alignas(64) __shared__ std::uint64_t words[16];
    __shared__ std::uint64_t acquired;
    auto &s = *a.storage.protocol;
    if (threadIdx.x == 0)
    {
        const auto start = clock();
        std::uint64_t seq;
        while ((seq = acquire(&a.channel.base.completion->publication)) < s.issuedSequence &&
               clock() - start < timeout)
        {
#if defined(QFN_ADAPTIVE_CUDA)
            __nanosleep(256);
#else
            __builtin_amdgcn_s_sleep(1);
#endif
        }
        acquired = seq;
    }
    __syncthreads();
    if (threadIdx.x < kLookupCompletionLiveBytes / 8)
        words[threadIdx.x] =
            fresh(reinterpret_cast<const std::uint64_t *>(a.channel.base.completion) + threadIdx.x);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const auto &body = *reinterpret_cast<const ImmutableLookupCompletion *>(words);
        const auto status = acquired < s.issuedSequence
                                ? ImmutableLookupStatus::TimedOut
                                : ImmutableLookupProtocol::validateCompletion(
                                      body, acquired, s.issuedSequence, s.generation, a.channel.base.table,
                                      f.source_rows, a.plan.row_bytes, kImmutableLookupSubordinateProtocol);
        s.status = static_cast<std::uint64_t>(status);
        if (status != ImmutableLookupStatus::Success)
            failure(a, Status::SourceFailure);
    }
}
/** @brief Copy all acquired live native bytes and ACK records privately before mapped retirement. */
__global__ __launch_bounds__(256, 2) void copyResponse(AdaptiveLookupDeviceArgs a)
{
    auto &f = frame(a);
    if (f.phase != Phase::Published || f.status != Status::Success)
        return;
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * 256 + threadIdx.x,
               stride = static_cast<std::uint64_t>(gridDim.x) * 256;
    auto *native = at<std::byte>(a.storage.frame, a.plan.frame.native_response);
    const auto bytes = f.source_rows * a.plan.row_bytes;
    for (auto i = index; i < bytes / 4; i += stride)
        reinterpret_cast<std::uint32_t *>(native)[i] =
            fresh(reinterpret_cast<const std::uint32_t *>(a.channel.base.nativeRows) + i);
    if (index < bytes % 4)
        reinterpret_cast<unsigned char *>(native)[bytes / 4 * 4 + index] =
            fresh(reinterpret_cast<const unsigned char *>(a.channel.base.nativeRows) + bytes / 4 * 4 + index);
    auto *acks = at<ImmutableRowInvalidationAcknowledgement>(a.storage.frame, a.plan.frame.acknowledgements);
    for (auto i = index; i < f.source_rows; i += stride)
        acks[i].destination_grant = fresh(&a.channel.acknowledgements[i].destination_grant);
}
/** @brief Authenticate every private ACK/reservation and current admission before releasing mapped ownership.
 */
template <class Slot> __global__ void validateResponse(AdaptiveLookupDeviceArgs a)
{
    auto &f = frame(a);
    if (f.phase != Phase::Published)
        return;
    auto op = operations<Slot>(a.plan, a.storage);
    auto &s = *a.storage.protocol;
    if (f.status != Status::Success)
        return;
    if (s.generation != s.admissionGeneration)
    {
        failure(a, Status::Invariant);
        return;
    }
    const auto *u = at<UniqueRecord>(a.storage.frame, a.plan.frame.unique);
    const auto *acks =
        at<ImmutableRowInvalidationAcknowledgement>(a.storage.frame, a.plan.frame.acknowledgements);
    for (std::uint64_t i = 0; i < f.unique_rows; ++i)
        if (!(u[i].flags & InitialHit))
        {
            const bool reserved = u[i].flags & ReservedFill;
            const auto token = reserved ? u[i].expected_slot_generation : 0;
            if (u[i].response_coordinate >= f.source_rows ||
                acks[u[i].response_coordinate].destination_grant != token)
            {
                failure(a, Status::SourceFailure);
                return;
            }
            if (reserved)
            {
                if (u[i].slot >= a.plan.slots)
                {
                    failure(a, Status::Invariant);
                    return;
                }
                const auto &slot = at<Slot>(a.storage.cache, a.plan.cache.slots)[u[i].slot];
                if (slot.generation != token || static_cast<std::uint64_t>(slot.row) != u[i].row ||
                    Operations<Slot>::state(slot) != SlotState::Reserved || !Operations<Slot>::pinned(slot))
                {
                    failure(a, Status::Invariant);
                    return;
                }
            }
        }
    if (!op.transition(Phase::Published, Phase::ResponseCopied) ||
        !op.transition(Phase::ResponseCopied, Phase::Validated))
        failure(a, f.status);
    if (f.status == Status::Success && *a.error)
        failure(a, Status::Cancelled);
}
/** @brief Retire mapped bytes only after all private copies and validation; failed frames never publish
 * Ready. */
__global__ void acknowledge(AdaptiveLookupDeviceArgs a)
{
    const auto &f = frame(a);
    if (!f.source_rows || a.storage.protocol->issuedSequence == a.storage.protocol->consumedSequence)
        return;
    a.storage.protocol->consumedSequence = a.storage.protocol->issuedSequence;
    release(&a.channel.base.acknowledgement->consumedSequence, a.storage.protocol->consumedSequence);
}
/** @brief Fill only original native bytes of actual reservations, never slab padding or inactive slots. */
__global__ __launch_bounds__(256, 2) void fill(AdaptiveLookupDeviceArgs a)
{
    const auto &f = frame(a);
    if (f.phase != Phase::Validated || f.status != Status::Success)
        return;
    const auto *u = at<UniqueRecord>(a.storage.frame, a.plan.frame.unique);
    const auto *native = at<std::byte>(a.storage.frame, a.plan.frame.native_response);
    const auto size = f.unique_rows * a.plan.row_bytes;
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * 256 + threadIdx.x; i < size;
         i += static_cast<std::uint64_t>(gridDim.x) * 256)
    {
        const auto row = i / a.plan.row_bytes, byte = i % a.plan.row_bytes;
        if (u[row].flags & ReservedFill)
            a.storage.cache[a.plan.cache.payload +
                            static_cast<std::size_t>(u[row].slot) * a.plan.slot_stride + byte] =
                native[static_cast<std::size_t>(u[row].response_coordinate) * a.plan.row_bytes + byte];
    }
}
/** @brief Publish Ready only after fill retirement and a final current admission/slot generation check. */
template <class Slot> __global__ void ready(AdaptiveLookupDeviceArgs a)
{
    auto &f = frame(a);
    if (f.phase != Phase::Validated || f.status != Status::Success)
        return;
    auto op = operations<Slot>(a.plan, a.storage);
    const auto &s = *a.storage.protocol;
    if (s.generation != s.admissionGeneration)
    {
        failure(a, Status::Invariant);
        return;
    }
    if (*a.error)
    {
        failure(a, Status::Cancelled);
        return;
    }
    const auto *u = at<UniqueRecord>(a.storage.frame, a.plan.frame.unique);
    for (std::uint64_t i = 0; i < f.unique_rows; ++i)
        if (u[i].flags & ReservedFill)
        {
            // A reservation is not publication permission for a later interval.
            // Recheck the actual canonical words for each individual Ready
            // transition, including a cancellation between two filled rows.
            if (*reinterpret_cast<const volatile int *>(a.error))
            {
                failure(a, Status::Cancelled);
                return;
            }
            if (*reinterpret_cast<const volatile std::uint64_t *>(&s.admissionGeneration) != s.generation)
            {
                failure(a, Status::Invariant);
                return;
            }
            if (u[i].slot >= a.plan.slots)
            {
                failure(a, Status::Invariant);
                return;
            }
            const auto &slot = at<Slot>(a.storage.cache, a.plan.cache.slots)[u[i].slot];
            if (slot.generation != u[i].expected_slot_generation ||
                static_cast<std::uint64_t>(slot.row) != u[i].row ||
                !op.makeReady({u[i].slot, u[i].expected_slot_generation}))
            {
                failure(a, Status::Invariant);
                return;
            }
        }
}
/** @brief Decode each original coordinate from its pinned initial row or private exact source response. */
template <TensorType Format>
__global__ __launch_bounds__(256, 2) void decode(AdaptiveLookupDeviceArgs a, NativeEmbeddingIQTables tables)
{
    const auto &f = frame(a);
    if (f.phase != Phase::Validated || f.status != Status::Success || *a.error)
        return;
    const auto *u = at<UniqueRecord>(a.storage.frame, a.plan.frame.unique);
    const auto *map = at<std::uint32_t>(a.storage.frame, a.plan.frame.original_to_unique);
    const auto size = f.original_ids * static_cast<unsigned>(a.plan.row_elements);
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * 256 + threadIdx.x; i < size;
         i += static_cast<std::uint64_t>(gridDim.x) * 256)
    {
        const auto &row = u[map[i / a.plan.row_elements]];
        const auto *bytes = row.flags & InitialHit
                                ? a.storage.cache + a.plan.cache.payload +
                                      static_cast<std::size_t>(row.slot) * a.plan.slot_stride
                                : a.storage.frame + a.plan.frame.native_response +
                                      static_cast<std::size_t>(row.response_coordinate) * a.plan.row_bytes;
        a.query.output[i] = lookup_detail::decode<Format>(bytes, i % a.plan.row_elements, tables);
    }
}
/** @brief Release each unique pin once after decode; cancellation drops only uncommitted reservations. */
template <class Slot> __global__ void finish(AdaptiveLookupDeviceArgs a)
{
    __shared__ FrameControl workingFrame;
    __shared__ OwnerControl workingOwner;
    OrderedControlBorrow<Slot> borrow(metadata<Slot>(a.plan, a.storage), workingFrame, workingOwner);
    auto &f = borrow.frame();
    auto &s = *a.storage.protocol;
    auto op = Operations<Slot>(borrow.metadata());
    auto *u = at<UniqueRecord>(a.storage.frame, a.plan.frame.unique);
    QwenAdaptiveCounterTotals observed;
    observed.calls = 1;
    observed.promotions = f.promotions;
    observed.evictions = f.evictions;
    if (f.phase == Phase::Validated)
    {
        observed.source_rows = f.source_rows;
        observed.source_bytes = f.source_rows * a.plan.row_bytes;
    }
    bool consistent = true;
    if (f.unique_rows > f.original_ids || f.original_ids > a.plan.maximum_ids)
        invalid();
    for (std::uint64_t i = 0; i < f.unique_rows; ++i)
    {
        if (u[i].flags & Probed)
        {
            if (u[i].flags & InitialHit) ++observed.hit_rows;
            else ++observed.miss_rows;
        }
        if (u[i].flags & Held)
        {
            if (u[i].slot >= a.plan.slots)
            {
                consistent = false;
                failBorrowed(a, borrow, Status::Invariant);
                continue;
            }
            const auto &slot = at<Slot>(a.storage.cache, a.plan.cache.slots)[u[i].slot];
            if (static_cast<std::uint64_t>(slot.row) != u[i].row ||
                !op.release({u[i].slot, u[i].expected_slot_generation}))
            {
                consistent = false;
                failBorrowed(a, borrow, Status::Invariant);
            }
            u[i].flags &= ~Held;
        }
    }
    if (f.status == Status::Success && *a.error)
        failBorrowed(a, borrow, Status::Cancelled);
    // The frame itself proves its cancellation frontier. Before publication it
    // has no outstanding source operation. Afterwards only validated private
    // ACKs plus mapped retirement and exact admission may preserve Ready rows.
    // A controller error alone is never a source/metadata consistency proof.
    const bool beforePolicy = f.phase == Phase::Building && !f.policy_started && !f.unique_rows;
    const bool beforePublication =
        f.phase == Phase::Prepared && f.policy_started && f.observed_ids == f.original_ids;
    const bool afterValidation =
        f.phase == Phase::Validated && f.policy_started && f.observed_ids == f.original_ids;
    const bool cleanCancellation = f.status == Status::Cancelled && consistent && !f.held_pins &&
                                   (beforePolicy || beforePublication || afterValidation) &&
                                   s.admissionGeneration && s.generation == s.admissionGeneration &&
                                   s.issuedSequence == s.consumedSequence && !s.reserved &&
                                   (beforePolicy || beforePublication || !f.source_rows || s.status == 0);
    if (f.status != Status::Success && !cleanCancellation)
    {
        op.poison();
        s.reserved = 1;
        failBorrowed(a, borrow, f.status);
    }
    const auto status = f.status;
    (void)op.finish();
    if (f.phase != Phase::Idle || f.held_pins || (status == Status::Success && f.status != Status::Success))
    {
        op.poison();
        s.reserved = 1;
        failBorrowed(a, borrow, Status::Invariant);
    }
    observed.failed_frames = f.status != Status::Success;
    observed.completed_live_ids = observed.failed_frames ? 0 : f.original_ids;
    if (!accumulateQwenAdaptiveCounters(a.control, sampling_math::kDeviceGenerationControlCount, observed))
    {
        op.poison();
        s.reserved = 1;
        failBorrowed(a, borrow, Status::Exhausted);
    }
}
/** @brief Conditional retained-ticket close derives completion solely from canonical controller words. */
__global__ void close(AdaptiveLookupDeviceStorage s, std::size_t frameControl,
                      ImmutableSubordinateLookupDeviceView source, const int *healthy, const int *complete)
{
    if (healthy && *healthy != 0 && *complete == 0)
        return;
    const auto &f = *at<FrameControl>(s.frame, frameControl);
    if (f.phase != Phase::Idle || f.held_pins || !s.protocol->admissionGeneration ||
        s.protocol->issuedSequence != s.protocol->consumedSequence)
        invalid();
    release(&source.base.acknowledgement->closedGeneration, s.protocol->admissionGeneration);
}
/** @brief Reject asynchronous launch submission failures without blocking capture or replay. */
void check()
{
    const auto error = QFN_ADAPTIVE_ERROR();
    if (error != QFN_ADAPTIVE_SUCCESS)
        throw std::runtime_error(std::string("Adaptive native lookup launch failed: ") +
                                 QFN_ADAPTIVE_ERROR_TEXT(error));
}
/** @return Bounded launch grid; physical capacity never becomes communication size. */
unsigned blocks(std::uint64_t size)
{
    return static_cast<unsigned>(
        std::min<std::uint64_t>(65535, std::max<std::uint64_t>(1, (size + 255) / 256)));
}
/**
 * @brief Bounded hierarchical scan using only the checked mandatory scratch root.
 *
 * Arrays and launch counts depend on captured physical geometry; all native
 * reads/writes use the original live count. Four levels cover every admitted
 * uint31 ID geometry. The same scratch is reused by disjoint uniqueness/miss
 * phases; no temporary allocator or source-sized directory exists.
 */
void scan(const AdaptiveLookupDeviceArgs &a, std::size_t target, std::uint32_t maximum,
          QFN_ADAPTIVE_STREAM stream)
{
    std::size_t arrays[5] = {target}, cursor = a.plan.frame.scan;
    std::uint32_t sizes[5] = {maximum}, divisors[5] = {1};
    unsigned levels = 0;
    for (;;)
    {
        const auto total = (sizes[levels] + 255U) / 256U;
        scanBlocks<<<blocks(sizes[levels]), 256, 0, stream>>>(a, arrays[levels], cursor, divisors[levels]);
        if (sizes[levels] <= 256)
            break;
        arrays[levels + 1] = cursor;
        sizes[levels + 1] = total;
        divisors[levels + 1] = divisors[levels] * 256U;
        cursor += static_cast<std::size_t>(total) * 8;
        ++levels;
    }
    while (levels)
    {
        --levels;
        addScan<<<blocks(sizes[levels]), 256, 0, stream>>>(a, arrays[levels], arrays[levels + 1],
                                                           divisors[levels]);
    }
}
/** @brief Immutable setup codebook/source bundle; mutable state exists only in its original prepared roots.
 */
template <TensorType Format> class Kernels final : public IImmutableAdaptiveLookupKernels
{
public:
    /** @brief Freeze original source format/codebooks, checked policy/geometry and native timeout. */
    Kernels(AdaptiveImmutableRowCacheNativePlan p, ImmutableSubordinateLookupBinding source,
            NativeEmbeddingIQTables tables, std::uint64_t timeout)
        : plan_(p), source_(std::move(source)), tables_(tables), timeout_(timeout)
    {
    }
    const AdaptiveImmutableRowCacheNativePlan &plan() const noexcept override
    {
        return plan_;
    }
    const ImmutableSubordinateLookupBinding &source() const noexcept override
    {
        return source_;
    }

protected:
    /** @brief Include every embedded codebook address, deadline and this general hierarchical launch policy.
     */
    std::uint64_t nativeCaptureIdentity() const noexcept override
    {
        std::uint64_t h = 1469598103934665603ULL;
        const auto mix = [&](std::uint64_t word)
        {
            h ^= word;
            h *= 1099511628211ULL;
        };
        mix(3); // Exclusive CTA-shared control access belongs to native launch identity.
        mix(sampling_math::kDeviceGenerationControlCount);
        mix(timeout_);
        mix(static_cast<std::uint64_t>(Format));
        mix(reinterpret_cast<std::uintptr_t>(tables_.iq3s));
        mix(reinterpret_cast<std::uintptr_t>(tables_.iq3xxs));
        mix(reinterpret_cast<std::uintptr_t>(tables_.iq2s));
        mix(reinterpret_cast<std::uintptr_t>(tables_.iq2xs));
        mix(reinterpret_cast<std::uintptr_t>(tables_.iq2xxs));
        mix(reinterpret_cast<std::uintptr_t>(tables_.iq1s));
        return h ? h : 1;
    }
    /** @brief Initialize the exact row-key variant once, with no payload allocation or fill. */
    void initializeNative(AdaptiveLookupDeviceStorage s, void *stream) const override
    {
        const auto &p = plan_.parameters();
        const auto n = std::max<std::uint64_t>(p.slots, static_cast<std::uint64_t>(p.lane_width) * 4);
        if (p.row_width == RowKeyWidth::UInt32)
            initializeMetadata<Slot32><<<blocks(n), 256, 0, static_cast<QFN_ADAPTIVE_STREAM>(stream)>>>(p, s);
        else
            initializeMetadata<Slot64><<<blocks(n), 256, 0, static_cast<QFN_ADAPTIVE_STREAM>(stream)>>>(p, s);
        check();
    }
    /** @brief Preserve protocol history on every request reset, including legal C0. */
    void admitNative(AdaptiveLookupDeviceStorage s, std::uint64_t generation,
                     ImmutableLookupAdmissionArgs::ProtocolInitialization init, void *stream) const override
    {
        admission<<<1, 1, 0, static_cast<QFN_ADAPTIVE_STREAM>(stream)>>>(
            s, generation, init == ImmutableLookupAdmissionArgs::ProtocolInitialization::InitializeOnce);
        check();
    }
    /** @brief Select the checked full source key width; every positive captured M uses the same complete
     * path. */
    void gatherNative(const AdaptiveLookupDeviceArgs &a, void *stream) const override
    {
        if (a.plan.row_width == RowKeyWidth::UInt32)
            enqueue<Slot32>(a, static_cast<QFN_ADAPTIVE_STREAM>(stream));
        else
            enqueue<Slot64>(a, static_cast<QFN_ADAPTIVE_STREAM>(stream));
        check();
    }
    /** @brief Unconditional reset close or the actual canonical failed/complete retained-ticket predicate. */
    void closeNative(AdaptiveLookupDeviceStorage s, const int *healthy, const int *complete,
                     void *stream) const override
    {
        QFN_ADAPTIVE_NAMESPACE::close<<<1, 1, 0, static_cast<QFN_ADAPTIVE_STREAM>(stream)>>>(
            s, plan_.parameters().frame.control, source_.deviceView(), healthy, complete);
        check();
    }

private:
    /** @brief Record one fixed dependency-ordered transaction; no allocation, readback, sync or rebound
     * pointer. */
    template <class Slot> void enqueue(const AdaptiveLookupDeviceArgs &a, QFN_ADAPTIVE_STREAM stream) const
    {
        const auto maximum = static_cast<std::uint32_t>(a.query.token_rows.physicalRows()) * 16U;
        begin<Slot><<<1, 1, 0, stream>>>(a);
        validateKeys<<<blocks(maximum), 256, 0, stream>>>(a);
        sortTiles<<<blocks(maximum), 256, 0, stream>>>(a);
        auto sorted = a.plan.frame.keys_a, other = a.plan.frame.keys_b;
        for (std::uint32_t run = 256; run < maximum; run *= 2)
        {
            merge<<<blocks(maximum), 256, 0, stream>>>(a, sorted, other, run);
            std::swap(sorted, other);
        }
        firstCoordinates<<<blocks(maximum), 256, 0, stream>>>(a, sorted);
        firstFlags<<<blocks(maximum), 256, 0, stream>>>(a);
        scan(a, a.plan.frame.first_prefix, maximum, stream);
        uniqueRecords<<<blocks(maximum), 256, 0, stream>>>(a);
        probe<Slot><<<blocks(maximum), 256, 0, stream>>>(a);
        policy<Slot><<<1, 1, 0, stream>>>(a);
        missFlags<<<blocks(maximum), 256, 0, stream>>>(a);
        scan(a, a.plan.frame.miss_prefix, maximum, stream);
        beginSource<Slot><<<1, 1, 0, stream>>>(a);
        sourceRecords<<<blocks(maximum), 256, 0, stream>>>(a);
        publish<Slot><<<1, 1, 0, stream>>>(a);
        waitResponse<<<1, 256, 0, stream>>>(a, timeout_);
        copyResponse<<<blocks(static_cast<std::uint64_t>(maximum) * a.plan.row_bytes / 4 + maximum), 256, 0,
                       stream>>>(a);
        validateResponse<Slot><<<1, 1, 0, stream>>>(a);
        acknowledge<<<1, 1, 0, stream>>>(a);
        fill<<<blocks(static_cast<std::uint64_t>(maximum) * a.plan.row_bytes), 256, 0, stream>>>(a);
        ready<Slot><<<1, 1, 0, stream>>>(a);
        decode<Format><<<blocks(static_cast<std::uint64_t>(maximum) * a.plan.row_elements), 256, 0, stream>>>(
            a, tables_);
        finish<Slot><<<1, 1, 0, stream>>>(a);
    }
    AdaptiveImmutableRowCacheNativePlan plan_;
    ImmutableSubordinateLookupBinding source_;
    NativeEmbeddingIQTables tables_;
    std::uint64_t timeout_;
};
} // namespace QFN_ADAPTIVE_NAMESPACE
/** @brief Setup-only format/codebook construction; source/PMA/domain identity remains separately
 * authenticated. */
std::unique_ptr<IImmutableAdaptiveLookupKernels>
QFN_ADAPTIVE_FACTORY(AdaptiveImmutableRowCacheNativePlan plan, ImmutableSubordinateLookupBinding source)
{
    const auto &p = plan.parameters();
    source.requireFor(p);
    source.requirePreparedStorage(plan);
    lookup_detail::validateGeometry(p.format, p.row_elements, p.row_bytes);
    NativeEmbeddingIQTables tables;
    std::uint64_t timeout = 30'000'000'000ULL;
#if defined(QFN_ADAPTIVE_CUDA)
    if (!p.device.is_cuda() || !prepareNativeEmbeddingTablesCUDA(p.device.ordinal, p.format, tables))
        throw std::invalid_argument("Adaptive CUDA source/device/codebook mismatch");
#else
    if (!p.device.is_rocm() || !prepareNativeEmbeddingTablesROCm(p.device.ordinal, p.format, tables))
        throw std::invalid_argument("Adaptive HIP source/device/codebook/wall-clock mismatch");
    timeout = rocmImmutableLookupTimeoutTicks(rocmHardwareMetadata().at(p.device.ordinal));
#endif
    using namespace QFN_ADAPTIVE_NAMESPACE;
    switch (p.format)
    {
#define QFN_ADAPTIVE_CASE(F)                                                                                 \
    case TensorType::F:                                                                                      \
        return std::make_unique<Kernels<TensorType::F>>(plan, std::move(source), tables, timeout);
        QFN_ADAPTIVE_CASE(FP32)
        QFN_ADAPTIVE_CASE(FP16)
        QFN_ADAPTIVE_CASE(BF16)
        QFN_LOOKUP_QUANT_FORMATS(QFN_ADAPTIVE_CASE)
#undef QFN_ADAPTIVE_CASE
    default:
        throw std::invalid_argument("Adaptive native source format unsupported");
    }
}
} // namespace llaminar2::qwen4exp
#undef QFN_ADAPTIVE_STREAM
#undef QFN_ADAPTIVE_ERROR
#undef QFN_ADAPTIVE_SUCCESS
#undef QFN_ADAPTIVE_ERROR_TEXT
#undef QFN_ADAPTIVE_NAMESPACE
#undef QFN_ADAPTIVE_FACTORY
