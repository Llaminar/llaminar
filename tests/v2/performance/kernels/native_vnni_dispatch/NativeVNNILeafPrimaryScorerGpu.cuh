/**
 * @file NativeVNNILeafPrimaryScorerGpu.cuh
 * @brief Shared CUDA/HIP session, leaf diagnostic, and complete policy search.
 *
 * This file is included by one CUDA and one HIP translation unit.  Exactly one
 * of `LLAMINAR_NATIVE_VNNI_SCORER_CUDA` or
 * `LLAMINAR_NATIVE_VNNI_SCORER_ROCM` must be defined by the including source.
 * Keeping the shared session and diagnostic primitive here ensures both GPU
 * backends execute the same comparisons, threshold predicate, memory layout,
 * and survivor logic.
 */

#pragma once

#include "NativeVNNILeafPrimaryScorer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>

#if defined(LLAMINAR_NATIVE_VNNI_SCORER_CUDA)
#include <cuda_runtime.h>
using NativeVNNIScorerError = cudaError_t;
constexpr NativeVNNIScorerError kNativeVNNIScorerSuccess = cudaSuccess;
#define LLAMINAR_SCORER_GET_DEVICE_COUNT cudaGetDeviceCount
#define LLAMINAR_SCORER_SET_DEVICE cudaSetDevice
#define LLAMINAR_SCORER_MALLOC cudaMalloc
#define LLAMINAR_SCORER_FREE cudaFree
#define LLAMINAR_SCORER_MEMCPY_ASYNC cudaMemcpyAsync
#define LLAMINAR_SCORER_MEMSET_ASYNC cudaMemsetAsync
#define LLAMINAR_SCORER_STREAM_CREATE cudaStreamCreateWithFlags
#define LLAMINAR_SCORER_STREAM_DESTROY cudaStreamDestroy
#define LLAMINAR_SCORER_STREAM_SYNC cudaStreamSynchronize
#define LLAMINAR_SCORER_LAST_ERROR cudaGetLastError
#define LLAMINAR_SCORER_ERROR_STRING cudaGetErrorString
#define LLAMINAR_SCORER_HOST_TO_DEVICE cudaMemcpyHostToDevice
#define LLAMINAR_SCORER_DEVICE_TO_HOST cudaMemcpyDeviceToHost
using NativeVNNIScorerStream = cudaStream_t;
using NativeVNNIScorerGraph = cudaGraph_t;
using NativeVNNIScorerGraphExec = cudaGraphExec_t;
constexpr unsigned int kNativeVNNIScorerStreamFlags = cudaStreamNonBlocking;
constexpr const char* kNativeVNNIScorerBackend = "cuda";
#elif defined(LLAMINAR_NATIVE_VNNI_SCORER_ROCM)
#include <hip/hip_runtime.h>
using NativeVNNIScorerError = hipError_t;
constexpr NativeVNNIScorerError kNativeVNNIScorerSuccess = hipSuccess;
#define LLAMINAR_SCORER_GET_DEVICE_COUNT hipGetDeviceCount
#define LLAMINAR_SCORER_SET_DEVICE hipSetDevice
#define LLAMINAR_SCORER_MALLOC hipMalloc
#define LLAMINAR_SCORER_FREE hipFree
#define LLAMINAR_SCORER_MEMCPY_ASYNC hipMemcpyAsync
#define LLAMINAR_SCORER_MEMSET_ASYNC hipMemsetAsync
#define LLAMINAR_SCORER_STREAM_CREATE hipStreamCreateWithFlags
#define LLAMINAR_SCORER_STREAM_DESTROY hipStreamDestroy
#define LLAMINAR_SCORER_STREAM_SYNC hipStreamSynchronize
#define LLAMINAR_SCORER_LAST_ERROR hipGetLastError
#define LLAMINAR_SCORER_ERROR_STRING hipGetErrorString
#define LLAMINAR_SCORER_HOST_TO_DEVICE hipMemcpyHostToDevice
#define LLAMINAR_SCORER_DEVICE_TO_HOST hipMemcpyDeviceToHost
using NativeVNNIScorerStream = hipStream_t;
using NativeVNNIScorerGraph = hipGraph_t;
using NativeVNNIScorerGraphExec = hipGraphExec_t;
constexpr unsigned int kNativeVNNIScorerStreamFlags = hipStreamNonBlocking;
constexpr const char* kNativeVNNIScorerBackend = "rocm";
#else
#error "NativeVNNI leaf scorer requires a CUDA or ROCm backend definition"
#endif

/**
 * Backend-owned context retained across a sequence of regret matrices.
 *
 * The stream, matrix high-water allocation, tree workspace, and captured graphs
 * have session lifetime. Diagnostic subset arrays have the capacity recorded in
 * `subsetCapacity` and grow together, preserving a simple invariant that every
 * pointer can serve any batch no larger than that value for the current matrix
 * geometry.
 */
struct LlaminarNativeVNNILeafPrimaryScorerSession {
    int deviceOrdinal = -1;
    std::uint32_t pointCount = 0;
    std::uint32_t candidateCount = 0;
    std::uint32_t subsetWordCount = 0;
    std::uint32_t candidateWordCount = 0;
    std::size_t matrixCapacity = 0;
    std::uint32_t subsetCapacity = 0;
    NativeVNNIScorerStream stream = nullptr;
    double* deviceFittingRegrets = nullptr;
    double* deviceMeasuredP95Regrets = nullptr;
    std::uint64_t* deviceSubsetMasks = nullptr;
    double* deviceCandidateFittingP95 = nullptr;
    std::uint32_t* deviceCandidateFailedLeaves = nullptr;
    double* deviceBestFittingP95 = nullptr;
    std::uint32_t* deviceBestFailedLeaves = nullptr;
    std::uint64_t* deviceSurvivors = nullptr;
    void* treeScratch = nullptr;
    bool graphCaptureActive = false;
    LlaminarNativeVNNITreeRuntimeStats runtimeStats = {};
};

/** Invalidate captured graphs after a persistent pointer changes. */
void invalidateNativeVNNITreeGraphs(
    LlaminarNativeVNNILeafPrimaryScorerSession* session);

/** Release all tree-search scratch owned by one session. */
NativeVNNIScorerError releaseNativeVNNITreeScratch(
    LlaminarNativeVNNILeafPrimaryScorerSession* session);

namespace {

constexpr std::uint32_t kNativeVNNIScorerThreads = 256;
constexpr std::uint32_t kNativeVNNIScorerMaximumBlocks = 65'535;
constexpr double kNativeVNNIRegretBudget = 0.05;
constexpr double kNativeVNNIInfinity = static_cast<double>(INFINITY);
constexpr std::uint32_t kNativeVNNIInvalidFailedLeafCount =
    ~std::uint32_t{0};

/** Write one bounded error message without throwing across the C ABI. */
void writeNativeVNNIScorerError(
    char* error,
    std::size_t errorCapacity,
    const char* message) {
    if (error == nullptr || errorCapacity == 0) {
        return;
    }
    std::snprintf(error, errorCapacity, "%s", message);
}

/** Convert one backend error into the common nonzero ABI result. */
int failNativeVNNIScorer(
    char* error,
    std::size_t errorCapacity,
    const char* operation,
    NativeVNNIScorerError backendError) {
    if (error != nullptr && errorCapacity > 0) {
        std::snprintf(
            error,
            errorCapacity,
            "%s failed: %s",
            operation,
            LLAMINAR_SCORER_ERROR_STRING(backendError));
    }
    return 2;
}

/** Return whether A precedes B under the exact two-key leaf objective. */
__device__ __forceinline__ bool primaryScoreLess(
    double fittingP95A,
    std::uint32_t failedLeavesA,
    double fittingP95B,
    std::uint32_t failedLeavesB) {
    return failedLeavesA < failedLeavesB
        || (failedLeavesA == failedLeavesB && fittingP95A < fittingP95B);
}

/**
 * Select one conservative nearest-rank p95 directly from an input column.
 *
 * P95 is near the upper tail, so selecting from the largest value downward
 * requires at most `N - ceil(0.95*N) + 1` complete scans. This avoids a
 * point-count-dependent local array and returns one original FP64 value exactly.
 */
__device__ __forceinline__ double nearestRankP95(
    const double* values,
    std::uint32_t pointCount,
    std::uint32_t candidateCount,
    std::uint32_t candidate,
    const std::uint64_t* subsetMask,
    std::uint32_t subsetWordCount,
    std::uint32_t selectedPointCount) {
    const std::uint64_t rank =
        (95ULL * selectedPointCount + 99ULL) / 100ULL;
    std::uint32_t remainingFromTop =
        selectedPointCount - static_cast<std::uint32_t>(rank);
    double exclusiveUpper = kNativeVNNIInfinity;

    while (true) {
        double nextLargest = -kNativeVNNIInfinity;
        std::uint32_t multiplicity = 0;
        for (std::uint32_t word = 0; word < subsetWordCount; ++word) {
            std::uint64_t membership = subsetMask[word];
            while (membership != 0) {
                const std::uint32_t bit = static_cast<std::uint32_t>(
                    __ffsll(static_cast<long long>(membership)) - 1);
                const std::uint32_t point = word * 64U + bit;
                membership &= membership - 1U;
                if (point >= pointCount) {
                    continue;
                }
                const double value =
                    values[static_cast<std::uint64_t>(point) * candidateCount
                           + candidate];
                if (value >= exclusiveUpper) {
                    continue;
                }
                if (value > nextLargest) {
                    nextLargest = value;
                    multiplicity = 1;
                } else if (value == nextLargest) {
                    ++multiplicity;
                }
            }
        }
        if (remainingFromTop < multiplicity) {
            return nextLargest;
        }
        remainingFromTop -= multiplicity;
        exclusiveUpper = nextLargest;
    }
}

/** Score every `(subset, candidate)` pair independently. */
__global__ void scoreNativeVNNILeafCandidates(
    const double* fittingRegrets,
    const double* measuredP95Regrets,
    std::uint32_t pointCount,
    std::uint32_t candidateCount,
    const std::uint64_t* subsetMasks,
    std::uint32_t subsetCount,
    std::uint32_t subsetWordCount,
    double* candidateFittingP95,
    std::uint32_t* candidateFailedLeaves) {
    const std::uint64_t pairCount =
        static_cast<std::uint64_t>(subsetCount) * candidateCount;
    const std::uint64_t gridStride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t pairIndex =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pairIndex < pairCount;
         pairIndex += gridStride) {
        const std::uint32_t subset =
            static_cast<std::uint32_t>(pairIndex / candidateCount);
        const std::uint32_t candidate =
            static_cast<std::uint32_t>(pairIndex % candidateCount);
        const std::uint64_t* subsetMask =
            subsetMasks + static_cast<std::uint64_t>(subset) * subsetWordCount;
        std::uint32_t selectedPointCount = 0;
        bool valid = true;

        for (std::uint32_t word = 0; word < subsetWordCount; ++word) {
            std::uint64_t membership = subsetMask[word];
            while (membership != 0) {
                const std::uint32_t bit = static_cast<std::uint32_t>(
                    __ffsll(static_cast<long long>(membership)) - 1);
                const std::uint32_t point = word * 64U + bit;
                membership &= membership - 1U;
                if (point >= pointCount) {
                    continue;
                }
                ++selectedPointCount;
                const std::uint64_t index =
                    static_cast<std::uint64_t>(point) * candidateCount + candidate;
                if (!isfinite(fittingRegrets[index])
                    || !isfinite(measuredP95Regrets[index])) {
                    valid = false;
                    break;
                }
            }
            if (!valid) {
                break;
            }
        }

        double fittingP95 = kNativeVNNIInfinity;
        std::uint32_t failedLeaves = kNativeVNNIInvalidFailedLeafCount;
        if (selectedPointCount != 0 && valid) {
            fittingP95 = nearestRankP95(
                fittingRegrets,
                pointCount,
                candidateCount,
                candidate,
                subsetMask,
                subsetWordCount,
                selectedPointCount);
            const double measuredP95 = nearestRankP95(
                measuredP95Regrets,
                pointCount,
                candidateCount,
                candidate,
                subsetMask,
                subsetWordCount,
                selectedPointCount);
            failedLeaves = measuredP95 >= kNativeVNNIRegretBudget ? 1U : 0U;
        }
        candidateFittingP95[pairIndex] = fittingP95;
        candidateFailedLeaves[pairIndex] = failedLeaves;
    }
}

/** Reduce all candidate primary keys for one subset in a fixed block tree. */
__global__ void reduceNativeVNNILeafPrimaryScores(
    const double* candidateFittingP95,
    const std::uint32_t* candidateFailedLeaves,
    std::uint32_t candidateCount,
    std::uint32_t subsetCount,
    double* bestFittingP95,
    std::uint32_t* bestFailedLeaves) {
    __shared__ double sharedFittingP95[kNativeVNNIScorerThreads];
    __shared__ std::uint32_t sharedFailedLeaves[kNativeVNNIScorerThreads];

    for (std::uint32_t subset = blockIdx.x;
         subset < subsetCount;
         subset += gridDim.x) {
        double localFittingP95 = kNativeVNNIInfinity;
        std::uint32_t localFailedLeaves = kNativeVNNIInvalidFailedLeafCount;
        for (std::uint32_t candidate = threadIdx.x;
             candidate < candidateCount;
             candidate += blockDim.x) {
            const std::uint64_t index =
                static_cast<std::uint64_t>(subset) * candidateCount + candidate;
            const double candidateP95 = candidateFittingP95[index];
            const std::uint32_t candidateFailure = candidateFailedLeaves[index];
            if (primaryScoreLess(
                    candidateP95,
                    candidateFailure,
                    localFittingP95,
                    localFailedLeaves)) {
                localFittingP95 = candidateP95;
                localFailedLeaves = candidateFailure;
            }
        }
        sharedFittingP95[threadIdx.x] = localFittingP95;
        sharedFailedLeaves[threadIdx.x] = localFailedLeaves;
        __syncthreads();

        for (std::uint32_t width = blockDim.x / 2U; width > 0; width >>= 1U) {
            if (threadIdx.x < width && primaryScoreLess(
                    sharedFittingP95[threadIdx.x + width],
                    sharedFailedLeaves[threadIdx.x + width],
                    sharedFittingP95[threadIdx.x],
                    sharedFailedLeaves[threadIdx.x])) {
                sharedFittingP95[threadIdx.x] =
                    sharedFittingP95[threadIdx.x + width];
                sharedFailedLeaves[threadIdx.x] =
                    sharedFailedLeaves[threadIdx.x + width];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            bestFittingP95[subset] = sharedFittingP95[0];
            bestFailedLeaves[subset] = sharedFailedLeaves[0];
        }
        __syncthreads();
    }
}

/** Materialize every exact primary-key tie without atomics or ordering effects. */
__global__ void markNativeVNNILeafPrimarySurvivors(
    const double* candidateFittingP95,
    const std::uint32_t* candidateFailedLeaves,
    std::uint32_t candidateCount,
    const double* bestFittingP95,
    const std::uint32_t* bestFailedLeaves,
    std::uint32_t candidateWordCount,
    std::uint32_t subsetCount,
    std::uint64_t* survivorMasks) {
    const std::uint64_t wordCount =
        static_cast<std::uint64_t>(subsetCount) * candidateWordCount;
    const std::uint64_t gridStride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t wordIndex =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         wordIndex < wordCount;
         wordIndex += gridStride) {
        const std::uint32_t subset =
            static_cast<std::uint32_t>(wordIndex / candidateWordCount);
        const std::uint32_t candidateWord =
            static_cast<std::uint32_t>(wordIndex % candidateWordCount);
        const std::uint32_t firstCandidate = candidateWord * 64U;
        std::uint64_t mask = 0;
        if (isfinite(bestFittingP95[subset])) {
            for (std::uint32_t bit = 0; bit < 64U; ++bit) {
                const std::uint32_t candidate = firstCandidate + bit;
                if (candidate >= candidateCount) {
                    break;
                }
                const std::uint64_t scoreIndex =
                    static_cast<std::uint64_t>(subset) * candidateCount + candidate;
                if (candidateFittingP95[scoreIndex] == bestFittingP95[subset]
                    && candidateFailedLeaves[scoreIndex]
                        == bestFailedLeaves[subset]) {
                    mask |= std::uint64_t{1} << bit;
                }
            }
        }
        survivorMasks[wordIndex] = mask;
    }
}

/** Allocate one device buffer and record successful runtime allocation calls. */
NativeVNNIScorerError allocateNativeVNNIScorerBuffer(
    LlaminarNativeVNNILeafPrimaryScorerSession* session,
    void** pointer,
    std::size_t bytes) {
    const NativeVNNIScorerError result = LLAMINAR_SCORER_MALLOC(pointer, bytes);
    if (result == kNativeVNNIScorerSuccess) {
        ++session->runtimeStats.device_allocation_count;
    }
    return result;
}

/** Free one optional device allocation and record successful runtime releases. */
NativeVNNIScorerError freeNativeVNNIScorerBuffer(
    LlaminarNativeVNNILeafPrimaryScorerSession* session,
    void* pointer) {
    if (pointer == nullptr) {
        return kNativeVNNIScorerSuccess;
    }
    const NativeVNNIScorerError result = LLAMINAR_SCORER_FREE(pointer);
    if (result == kNativeVNNIScorerSuccess) {
        ++session->runtimeStats.device_free_count;
    }
    return result;
}

/**
 * Publish or retrieve bytes on the session stream with exact direction counts.
 *
 * The graph-capture count is an executable invariant: it remains zero today,
 * and will immediately expose any future transfer inserted between capture
 * begin/end instead of relying on a source-level audit.
 */
template <typename CopyKind>
NativeVNNIScorerError copyNativeVNNIScorerBytesAsync(
    LlaminarNativeVNNILeafPrimaryScorerSession* session,
    void* destination,
    const void* source,
    std::size_t bytes,
    CopyKind kind) {
    const NativeVNNIScorerError result = LLAMINAR_SCORER_MEMCPY_ASYNC(
        destination, source, bytes, kind, session->stream);
    if (result != kNativeVNNIScorerSuccess) {
        return result;
    }
    if (kind == LLAMINAR_SCORER_HOST_TO_DEVICE) {
        ++session->runtimeStats.h2d_copy_count;
        session->runtimeStats.h2d_bytes += bytes;
    } else if (kind == LLAMINAR_SCORER_DEVICE_TO_HOST) {
        ++session->runtimeStats.d2h_copy_count;
        session->runtimeStats.d2h_bytes += bytes;
    }
    if (session->graphCaptureActive) {
        ++session->runtimeStats.captured_graph_transfer_count;
    }
    return result;
}

/** Synchronize only the explicit session stream and account for the boundary. */
NativeVNNIScorerError synchronizeNativeVNNIScorerStream(
    LlaminarNativeVNNILeafPrimaryScorerSession* session) {
    const NativeVNNIScorerError result =
        LLAMINAR_SCORER_STREAM_SYNC(session->stream);
    if (result == kNativeVNNIScorerSuccess) {
        ++session->runtimeStats.stream_sync_count;
    }
    return result;
}

/**
 * Release subset-scoring scratch after matrix geometry changes.
 *
 * The diagnostic scorer sizes several arrays as `subset_count * point_words`
 * or `subset_count * candidate_count`. A changed matrix geometry therefore
 * invalidates the old single-capacity invariant even when the next subset count
 * is numerically smaller. Tree-search scratch is independent and remains owned.
 */
void discardNativeVNNILeafDiagnosticScratch(
    LlaminarNativeVNNILeafPrimaryScorerSession* session) {
    (void)freeNativeVNNIScorerBuffer(session, session->deviceSubsetMasks);
    (void)freeNativeVNNIScorerBuffer(session, session->deviceCandidateFittingP95);
    (void)freeNativeVNNIScorerBuffer(session, session->deviceCandidateFailedLeaves);
    (void)freeNativeVNNIScorerBuffer(session, session->deviceBestFittingP95);
    (void)freeNativeVNNIScorerBuffer(session, session->deviceBestFailedLeaves);
    (void)freeNativeVNNIScorerBuffer(session, session->deviceSurvivors);
    session->deviceSubsetMasks = nullptr;
    session->deviceCandidateFittingP95 = nullptr;
    session->deviceCandidateFailedLeaves = nullptr;
    session->deviceBestFittingP95 = nullptr;
    session->deviceBestFailedLeaves = nullptr;
    session->deviceSurvivors = nullptr;
    session->subsetCapacity = 0;
}

/** Return whether an element count can be represented as a byte count. */
bool nativeVNNIScorerByteCountFits(
    std::size_t elementCount,
    std::size_t elementSize) {
    return elementCount <= std::numeric_limits<std::size_t>::max() / elementSize;
}

/** Release every session allocation without reporting secondary errors. */
void discardNativeVNNIScorerSessionStorage(
    LlaminarNativeVNNILeafPrimaryScorerSession* session) {
    if (session == nullptr) {
        return;
    }
    (void)freeNativeVNNIScorerBuffer(session, session->deviceFittingRegrets);
    (void)freeNativeVNNIScorerBuffer(session, session->deviceMeasuredP95Regrets);
    discardNativeVNNILeafDiagnosticScratch(session);
    if (session->stream != nullptr) {
        (void)LLAMINAR_SCORER_STREAM_DESTROY(session->stream);
    }
}

/**
 * Grow all subset-dependent scratch arrays as one transactional allocation.
 *
 * New buffers are allocated before the old buffers are released. A failed
 * growth therefore leaves the prepared matrix and prior scratch capacity
 * usable, while this particular score call reports a hard backend error.
 */
bool ensureNativeVNNIScorerSubsetCapacity(
    LlaminarNativeVNNILeafPrimaryScorerSession* session,
    std::uint32_t subsetCount,
    char* error,
    std::size_t errorCapacity) {
    if (subsetCount <= session->subsetCapacity) {
        return true;
    }

    const std::size_t subsetMaskCount =
        static_cast<std::size_t>(subsetCount) * session->subsetWordCount;
    const std::size_t candidateScoreCount =
        static_cast<std::size_t>(subsetCount) * session->candidateCount;
    const std::size_t survivorMaskCount =
        static_cast<std::size_t>(subsetCount) * session->candidateWordCount;
    if (!nativeVNNIScorerByteCountFits(
            subsetMaskCount, sizeof(std::uint64_t))
        || !nativeVNNIScorerByteCountFits(
            candidateScoreCount, sizeof(double))
        || !nativeVNNIScorerByteCountFits(
            candidateScoreCount, sizeof(std::uint32_t))
        || !nativeVNNIScorerByteCountFits(
            survivorMaskCount, sizeof(std::uint64_t))) {
        writeNativeVNNIScorerError(
            error, errorCapacity, "leaf scorer scratch byte count overflows size_t");
        return false;
    }

    std::uint64_t* subsetMasks = nullptr;
    double* candidateFittingP95 = nullptr;
    std::uint32_t* candidateFailedLeaves = nullptr;
    double* bestFittingP95 = nullptr;
    std::uint32_t* bestFailedLeaves = nullptr;
    std::uint64_t* survivors = nullptr;
    NativeVNNIScorerError result = kNativeVNNIScorerSuccess;
    auto allocate = [&](void** pointer, std::size_t bytes, const char* name) {
        result = allocateNativeVNNIScorerBuffer(session, pointer, bytes);
        if (result != kNativeVNNIScorerSuccess) {
            (void)failNativeVNNIScorer(error, errorCapacity, name, result);
            return false;
        }
        return true;
    };
    const bool allocated =
        allocate(reinterpret_cast<void**>(&subsetMasks),
                 subsetMaskCount * sizeof(std::uint64_t), "subset allocation")
        && allocate(reinterpret_cast<void**>(&candidateFittingP95),
                    candidateScoreCount * sizeof(double), "p95 allocation")
        && allocate(reinterpret_cast<void**>(&candidateFailedLeaves),
                    candidateScoreCount * sizeof(std::uint32_t),
                    "leaf-gate allocation")
        && allocate(reinterpret_cast<void**>(&bestFittingP95),
                    subsetCount * sizeof(double), "best-p95 allocation")
        && allocate(reinterpret_cast<void**>(&bestFailedLeaves),
                    subsetCount * sizeof(std::uint32_t),
                    "best-leaf-gate allocation")
        && allocate(reinterpret_cast<void**>(&survivors),
                    survivorMaskCount * sizeof(std::uint64_t),
                    "survivor allocation");
    if (!allocated) {
        (void)freeNativeVNNIScorerBuffer(session, subsetMasks);
        (void)freeNativeVNNIScorerBuffer(session, candidateFittingP95);
        (void)freeNativeVNNIScorerBuffer(session, candidateFailedLeaves);
        (void)freeNativeVNNIScorerBuffer(session, bestFittingP95);
        (void)freeNativeVNNIScorerBuffer(session, bestFailedLeaves);
        (void)freeNativeVNNIScorerBuffer(session, survivors);
        return false;
    }

    (void)freeNativeVNNIScorerBuffer(session, session->deviceSubsetMasks);
    (void)freeNativeVNNIScorerBuffer(session, session->deviceCandidateFittingP95);
    (void)freeNativeVNNIScorerBuffer(session, session->deviceCandidateFailedLeaves);
    (void)freeNativeVNNIScorerBuffer(session, session->deviceBestFittingP95);
    (void)freeNativeVNNIScorerBuffer(session, session->deviceBestFailedLeaves);
    (void)freeNativeVNNIScorerBuffer(session, session->deviceSurvivors);
    session->deviceSubsetMasks = subsetMasks;
    session->deviceCandidateFittingP95 = candidateFittingP95;
    session->deviceCandidateFailedLeaves = candidateFailedLeaves;
    session->deviceBestFittingP95 = bestFittingP95;
    session->deviceBestFailedLeaves = bestFailedLeaves;
    session->deviceSurvivors = survivors;
    session->subsetCapacity = subsetCount;
    return true;
}

} // namespace

extern "C" std::uint32_t llaminarNativeVNNILeafPrimaryScorerAbiVersion() {
    return kLlaminarNativeVNNILeafPrimaryScorerAbiVersion;
}

extern "C" const char* llaminarNativeVNNILeafPrimaryScorerBackend() {
    return kNativeVNNIScorerBackend;
}

extern "C" int llaminarNativeVNNILeafPrimaryScorerDeviceCount(
    char* error,
    std::size_t errorCapacity) {
    int count = 0;
    const NativeVNNIScorerError result = LLAMINAR_SCORER_GET_DEVICE_COUNT(&count);
    if (result != kNativeVNNIScorerSuccess) {
        return -failNativeVNNIScorer(
            error, errorCapacity, "device-count query", result);
    }
    if (error != nullptr && errorCapacity > 0) {
        error[0] = '\0';
    }
    return count;
}

extern "C" int llaminarNativeVNNILeafPrimaryScorerPrepare(
    LlaminarNativeVNNILeafPrimaryScorerSession* session,
    const double* fittingRegrets,
    const double* measuredP95Regrets,
    std::uint32_t pointCount,
    std::uint32_t candidateCount,
    char* error,
    std::size_t errorCapacity) {
    if (session == nullptr || fittingRegrets == nullptr
        || measuredP95Regrets == nullptr) {
        writeNativeVNNIScorerError(
            error, errorCapacity, "leaf scorer preparation received a null input");
        return 1;
    }
    if (pointCount == 0 || candidateCount == 0) {
        writeNativeVNNIScorerError(
            error, errorCapacity, "leaf scorer dimensions must be positive");
        return 1;
    }
    const std::size_t regretCount =
        static_cast<std::size_t>(pointCount) * candidateCount;
    if (regretCount / pointCount != candidateCount
        || !nativeVNNIScorerByteCountFits(regretCount, sizeof(double))) {
        writeNativeVNNIScorerError(
            error, errorCapacity, "regret matrix byte count overflows size_t");
        return 1;
    }

    NativeVNNIScorerError result =
        LLAMINAR_SCORER_SET_DEVICE(session->deviceOrdinal);
    if (result != kNativeVNNIScorerSuccess) {
        return failNativeVNNIScorer(
            error, errorCapacity, "device selection", result);
    }

    if (regretCount > session->matrixCapacity) {
        double* newFitting = nullptr;
        double* newMeasured = nullptr;
        result = allocateNativeVNNIScorerBuffer(session,
            reinterpret_cast<void**>(&newFitting), regretCount * sizeof(double));
        if (result == kNativeVNNIScorerSuccess) {
            result = allocateNativeVNNIScorerBuffer(session,
                reinterpret_cast<void**>(&newMeasured), regretCount * sizeof(double));
        }
        if (result != kNativeVNNIScorerSuccess) {
            (void)freeNativeVNNIScorerBuffer(session, newFitting);
            (void)freeNativeVNNIScorerBuffer(session, newMeasured);
            return failNativeVNNIScorer(
                error, errorCapacity, "regret matrix high-water growth", result);
        }

        // Captured graph nodes retain raw matrix addresses. Destroy them before
        // publishing replacement buffers; metadata-only matrix updates preserve
        // both addresses and all graph executables.
        invalidateNativeVNNITreeGraphs(session);
        (void)freeNativeVNNIScorerBuffer(
            session, session->deviceFittingRegrets);
        (void)freeNativeVNNIScorerBuffer(
            session, session->deviceMeasuredP95Regrets);
        session->deviceFittingRegrets = newFitting;
        session->deviceMeasuredP95Regrets = newMeasured;
        session->matrixCapacity = regretCount;
        ++session->runtimeStats.matrix_growth_count;
    }

    const bool geometryChanged = session->pointCount != pointCount
        || session->candidateCount != candidateCount;
    if (geometryChanged && session->subsetCapacity != 0) {
        discardNativeVNNILeafDiagnosticScratch(session);
    }
    session->pointCount = pointCount;
    session->candidateCount = candidateCount;
    session->subsetWordCount = (pointCount - 1U) / 64U + 1U;
    session->candidateWordCount = (candidateCount - 1U) / 64U + 1U;

    result = copyNativeVNNIScorerBytesAsync(session,
        session->deviceFittingRegrets,
        fittingRegrets,
        regretCount * sizeof(double),
        LLAMINAR_SCORER_HOST_TO_DEVICE);
    if (result == kNativeVNNIScorerSuccess) {
        result = copyNativeVNNIScorerBytesAsync(session,
            session->deviceMeasuredP95Regrets,
            measuredP95Regrets,
            regretCount * sizeof(double),
            LLAMINAR_SCORER_HOST_TO_DEVICE);
    }
    if (result != kNativeVNNIScorerSuccess) {
        return failNativeVNNIScorer(
            error, errorCapacity, "regret matrix publication", result);
    }

    ++session->runtimeStats.prepare_count;
    if (error != nullptr && errorCapacity > 0) {
        error[0] = '\0';
    }
    return 0;
}

extern "C" LlaminarNativeVNNILeafPrimaryScorerSession*
llaminarNativeVNNILeafPrimaryScorerCreate(
    int deviceOrdinal,
    const double* fittingRegrets,
    const double* measuredP95Regrets,
    std::uint32_t pointCount,
    std::uint32_t candidateCount,
    char* error,
    std::size_t errorCapacity) {
    NativeVNNIScorerError result = LLAMINAR_SCORER_SET_DEVICE(deviceOrdinal);
    if (result != kNativeVNNIScorerSuccess) {
        (void)failNativeVNNIScorer(
            error, errorCapacity, "device selection", result);
        return nullptr;
    }
    auto* session = new (std::nothrow)
        LlaminarNativeVNNILeafPrimaryScorerSession();
    if (session == nullptr) {
        writeNativeVNNIScorerError(
            error, errorCapacity, "leaf scorer host-session allocation failed");
        return nullptr;
    }
    session->deviceOrdinal = deviceOrdinal;

    result = LLAMINAR_SCORER_STREAM_CREATE(
        &session->stream, kNativeVNNIScorerStreamFlags);
    if (result != kNativeVNNIScorerSuccess) {
        (void)failNativeVNNIScorer(
            error, errorCapacity, "device-resident scorer preparation", result);
        discardNativeVNNIScorerSessionStorage(session);
        delete session;
        return nullptr;
    }
    const int prepareResult = llaminarNativeVNNILeafPrimaryScorerPrepare(
        session,
        fittingRegrets,
        measuredP95Regrets,
        pointCount,
        candidateCount,
        error,
        errorCapacity);
    if (prepareResult != 0) {
        discardNativeVNNIScorerSessionStorage(session);
        delete session;
        return nullptr;
    }
    if (error != nullptr && errorCapacity > 0) {
        error[0] = '\0';
    }
    return session;
}

extern "C" int llaminarNativeVNNILeafPrimaryScorerScore(
    LlaminarNativeVNNILeafPrimaryScorerSession* session,
    const std::uint64_t* subsetMasks,
    std::uint32_t subsetCount,
    std::uint32_t subsetWordCount,
    double* bestFittingP95,
    std::uint32_t* bestFailedLeafCount,
    std::uint64_t* survivorMasks,
    char* error,
    std::size_t errorCapacity) {
    if (session == nullptr) {
        writeNativeVNNIScorerError(
            error, errorCapacity, "leaf scorer received a null session");
        return 1;
    }
    if (subsetMasks == nullptr || bestFittingP95 == nullptr
        || bestFailedLeafCount == nullptr || survivorMasks == nullptr) {
        writeNativeVNNIScorerError(
            error, errorCapacity, "leaf scorer received a null buffer");
        return 1;
    }
    if (subsetCount == 0) {
        writeNativeVNNIScorerError(
            error, errorCapacity, "leaf scorer dimensions must be positive");
        return 1;
    }
    if (subsetWordCount != session->subsetWordCount) {
        writeNativeVNNIScorerError(
            error, errorCapacity, "subset word count does not match point count");
        return 1;
    }

    NativeVNNIScorerError result =
        LLAMINAR_SCORER_SET_DEVICE(session->deviceOrdinal);
    if (result != kNativeVNNIScorerSuccess) {
        return failNativeVNNIScorer(
            error, errorCapacity, "device selection", result);
    }

    const std::size_t subsetMaskCount =
        static_cast<std::size_t>(subsetCount) * subsetWordCount;
    const std::size_t candidateScoreCount =
        static_cast<std::size_t>(subsetCount) * session->candidateCount;
    const std::size_t survivorMaskCount =
        static_cast<std::size_t>(subsetCount) * session->candidateWordCount;
    if (!nativeVNNIScorerByteCountFits(
            subsetMaskCount, sizeof(std::uint64_t))
        || !nativeVNNIScorerByteCountFits(
            candidateScoreCount, sizeof(double))
        || !nativeVNNIScorerByteCountFits(
            candidateScoreCount, sizeof(std::uint32_t))
        || !nativeVNNIScorerByteCountFits(
            survivorMaskCount, sizeof(std::uint64_t))) {
        writeNativeVNNIScorerError(
            error, errorCapacity, "leaf scorer byte count overflows size_t");
        return 1;
    }
    if (!ensureNativeVNNIScorerSubsetCapacity(
            session, subsetCount, error, errorCapacity)) {
        return 2;
    }

    result = copyNativeVNNIScorerBytesAsync(session,
        session->deviceSubsetMasks,
        subsetMasks,
        subsetMaskCount * sizeof(std::uint64_t),
        LLAMINAR_SCORER_HOST_TO_DEVICE);
    if (result != kNativeVNNIScorerSuccess) {
        return failNativeVNNIScorer(
            error, errorCapacity, "subset upload", result);
    }

    const std::uint64_t pairCount =
        static_cast<std::uint64_t>(subsetCount) * session->candidateCount;
    const std::uint32_t pairBlocks = static_cast<std::uint32_t>(std::min(
        (pairCount + kNativeVNNIScorerThreads - 1U)
            / kNativeVNNIScorerThreads,
        static_cast<std::uint64_t>(kNativeVNNIScorerMaximumBlocks)));
    scoreNativeVNNILeafCandidates<<<
        pairBlocks, kNativeVNNIScorerThreads, 0, session->stream>>>(
        session->deviceFittingRegrets,
        session->deviceMeasuredP95Regrets,
        session->pointCount,
        session->candidateCount,
        session->deviceSubsetMasks,
        subsetCount,
        subsetWordCount,
        session->deviceCandidateFittingP95,
        session->deviceCandidateFailedLeaves);
    const std::uint32_t subsetBlocks =
        std::min(subsetCount, kNativeVNNIScorerMaximumBlocks);
    reduceNativeVNNILeafPrimaryScores<<<
        subsetBlocks, kNativeVNNIScorerThreads, 0, session->stream>>>(
        session->deviceCandidateFittingP95,
        session->deviceCandidateFailedLeaves,
        session->candidateCount,
        subsetCount,
        session->deviceBestFittingP95,
        session->deviceBestFailedLeaves);
    const std::uint64_t survivorWordCount =
        static_cast<std::uint64_t>(subsetCount) * session->candidateWordCount;
    const std::uint32_t survivorBlocks = static_cast<std::uint32_t>(std::min(
        (survivorWordCount + kNativeVNNIScorerThreads - 1U)
            / kNativeVNNIScorerThreads,
        static_cast<std::uint64_t>(kNativeVNNIScorerMaximumBlocks)));
    markNativeVNNILeafPrimarySurvivors<<<
        survivorBlocks, kNativeVNNIScorerThreads, 0, session->stream>>>(
        session->deviceCandidateFittingP95,
        session->deviceCandidateFailedLeaves,
        session->candidateCount,
        session->deviceBestFittingP95,
        session->deviceBestFailedLeaves,
        session->candidateWordCount,
        subsetCount,
        session->deviceSurvivors);

    result = LLAMINAR_SCORER_LAST_ERROR();
    if (result != kNativeVNNIScorerSuccess) {
        return failNativeVNNIScorer(
            error, errorCapacity, "leaf-score kernel launch", result);
    }

    result = copyNativeVNNIScorerBytesAsync(session,
        bestFittingP95,
        session->deviceBestFittingP95,
        subsetCount * sizeof(double),
        LLAMINAR_SCORER_DEVICE_TO_HOST);
    if (result == kNativeVNNIScorerSuccess) {
        result = copyNativeVNNIScorerBytesAsync(session,
            bestFailedLeafCount,
            session->deviceBestFailedLeaves,
            subsetCount * sizeof(std::uint32_t),
            LLAMINAR_SCORER_DEVICE_TO_HOST);
    }
    if (result == kNativeVNNIScorerSuccess) {
        result = copyNativeVNNIScorerBytesAsync(session,
            survivorMasks,
            session->deviceSurvivors,
            survivorMaskCount * sizeof(std::uint64_t),
            LLAMINAR_SCORER_DEVICE_TO_HOST);
    }
    if (result == kNativeVNNIScorerSuccess) {
        result = synchronizeNativeVNNIScorerStream(session);
    }
    if (result != kNativeVNNIScorerSuccess) {
        return failNativeVNNIScorer(
            error, errorCapacity, "score download", result);
    }

    if (error != nullptr && errorCapacity > 0) {
        error[0] = '\0';
    }
    return 0;
}

// Tree search owns a backend-neutral persistent scratch object and graph cache.
// Include it before destruction so cleanup can release those resources through
// the same backend error path as the leaf diagnostic buffers.
#include "NativeVNNITreeSearchGpu.cuh"

extern "C" int llaminarNativeVNNITreeRuntimeStats(
    const LlaminarNativeVNNILeafPrimaryScorerSession* session,
    LlaminarNativeVNNITreeRuntimeStats* stats,
    char* error,
    std::size_t errorCapacity) {
    if (session == nullptr || stats == nullptr) {
        writeNativeVNNIScorerError(
            error, errorCapacity, "tree runtime stats received a null input");
        return 1;
    }
    *stats = session->runtimeStats;
    if (error != nullptr && errorCapacity > 0) {
        error[0] = '\0';
    }
    return 0;
}

extern "C" int llaminarNativeVNNILeafPrimaryScorerDestroy(
    LlaminarNativeVNNILeafPrimaryScorerSession* session,
    char* error,
    std::size_t errorCapacity) {
    if (session == nullptr) {
        writeNativeVNNIScorerError(
            error, errorCapacity, "leaf scorer received a null session");
        return 1;
    }

    NativeVNNIScorerError firstError = kNativeVNNIScorerSuccess;
    const char* firstOperation = nullptr;
    auto record = [&](NativeVNNIScorerError result, const char* operation) {
        if (firstError == kNativeVNNIScorerSuccess
            && result != kNativeVNNIScorerSuccess) {
            firstError = result;
            firstOperation = operation;
        }
    };
    record(LLAMINAR_SCORER_SET_DEVICE(session->deviceOrdinal), "device selection");
    if (session->stream != nullptr) {
        record(synchronizeNativeVNNIScorerStream(session), "stream synchronization");
    }
    record(releaseNativeVNNITreeScratch(session), "tree-scratch release");
    auto release = [&](void* pointer) {
        if (pointer != nullptr) {
            record(freeNativeVNNIScorerBuffer(session, pointer),
                   "device-buffer release");
        }
    };
    release(session->deviceFittingRegrets);
    release(session->deviceMeasuredP95Regrets);
    release(session->deviceSubsetMasks);
    release(session->deviceCandidateFittingP95);
    release(session->deviceCandidateFailedLeaves);
    release(session->deviceBestFittingP95);
    release(session->deviceBestFailedLeaves);
    release(session->deviceSurvivors);
    if (session->stream != nullptr) {
        record(LLAMINAR_SCORER_STREAM_DESTROY(session->stream), "stream destruction");
    }
    delete session;

    if (firstError != kNativeVNNIScorerSuccess) {
        return failNativeVNNIScorer(
            error, errorCapacity, firstOperation, firstError);
    }
    if (error != nullptr && errorCapacity > 0) {
        error[0] = '\0';
    }
    return 0;
}

#undef LLAMINAR_SCORER_GET_DEVICE_COUNT
#undef LLAMINAR_SCORER_SET_DEVICE
#undef LLAMINAR_SCORER_MALLOC
#undef LLAMINAR_SCORER_FREE
#undef LLAMINAR_SCORER_MEMCPY_ASYNC
#undef LLAMINAR_SCORER_MEMSET_ASYNC
#undef LLAMINAR_SCORER_STREAM_CREATE
#undef LLAMINAR_SCORER_STREAM_DESTROY
#undef LLAMINAR_SCORER_STREAM_SYNC
#undef LLAMINAR_SCORER_LAST_ERROR
#undef LLAMINAR_SCORER_ERROR_STRING
#undef LLAMINAR_SCORER_HOST_TO_DEVICE
#undef LLAMINAR_SCORER_DEVICE_TO_HOST
