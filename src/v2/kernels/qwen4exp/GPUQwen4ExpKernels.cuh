/**
 * @file GPUQwen4ExpKernels.cuh
 * @brief Shared CUDA/HIP GR/PLE kernels with capture-safe immutable state.
 *
 * Both native backends compile the same fixed 256-thread arithmetic tree.
 * Pointwise work uses all lanes across channels; each PLE similarity branch
 * gets one block, retaining identical per-row arithmetic at every batch size.
 * CPU-owned immutable bindings are checked before launch, while live counts
 * and acceptance decisions are validated exclusively on device. No allocation,
 * synchronization, transfer or callback appears in this execution boundary.
 * HIP branch normalization retains the exact 256-lane arithmetic tree while
 * its wave64 register tail avoids six redundant CTA barriers. The final sum
 * is explicitly republished to every output lane; CUDA keeps its shared tree.
 */
#pragma once
#include "Qwen4ExpKernels.h"
#include "Qwen4ExpValidation.h"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#if defined(QFN_CUDA)
#include <cuda_runtime.h>
#define QFN_NATIVE_STREAM cudaStream_t
#define QFN_NATIVE_ERROR cudaError_t
#define QFN_GET_ERROR cudaGetLastError
#define QFN_ERROR_STRING cudaGetErrorString
#define QFN_SUCCESS cudaSuccess
#define QFN_DEFAULT_STREAM cudaStreamLegacy
#define QFN_THREAD_STREAM cudaStreamPerThread
#define QFN_FACTORY createCUDAQwen4ExpKernels
#define QFN_IMPLEMENTATION CUDAQwen4ExpKernels
#define QFN_BACKEND_NAMESPACE cuda_detail
#elif defined(QFN_ROCM)
#include <hip/hip_runtime.h>
#define QFN_NATIVE_STREAM hipStream_t
#define QFN_NATIVE_ERROR hipError_t
#define QFN_GET_ERROR hipGetLastError
#define QFN_ERROR_STRING hipGetErrorString
#define QFN_SUCCESS hipSuccess
#define QFN_DEFAULT_STREAM hipStreamLegacy
#define QFN_THREAD_STREAM hipStreamPerThread
#define QFN_FACTORY createROCmQwen4ExpKernels
#define QFN_IMPLEMENTATION ROCmQwen4ExpKernels
#define QFN_BACKEND_NAMESPACE rocm_detail
#else
#error "Compile Qwen4Exp GPU primitives through a native backend translation unit"
#endif

namespace llaminar2::qwen4exp
{
namespace QFN_BACKEND_NAMESPACE
{
/** @brief Canonical arithmetic lanes, independent of hardware warp/wave width. */
inline constexpr unsigned kThreads = 256;
/** @brief Terminate malformed live execution state on its owning backend. */
__device__ inline void invalidDeviceState()
{
#if defined(QFN_CUDA)
    __trap();
#else
    __builtin_trap();
#endif
}
/** @brief Read live execution state in its owning GPU address space. */
__device__ inline int liveRows(RowExtent extent)
{
    const int count = extent.liveRows ? *extent.liveRows : extent.rows;
    if (count < 0 || count > extent.rows) invalidDeviceState();
    return count;
}
/** @brief Read and validate device prefix acceptance without a host mirror. */
__device__ inline int acceptedRows(const AcceptedPrefix &prefix)
{
    const int count = prefix.acceptedRows ? *prefix.acceptedRows : prefix.hostAcceptedRows;
    if (count < 0 || count > liveRows(prefix.candidates)) invalidDeviceState();
    return count;
}
/** @brief Deterministic per-row FP32 sigmoid; no batch-dependent vector route. */
__device__ inline float sigmoid(float value) { return 1.0f / (1.0f + expf(-value)); }
/**
 * @brief One fixed-order branch RMSNorm with branch-specific actual gain.
 * @param a Immutable row extent, original input/gain and admitted output storage.
 * @details Every row/branch owns one 256-thread CTA. HIP's wave64 tail changes
 * storage of the original reduction edges, never their operands or RN order.
 * All output lanes consume the final shared sum after a collective barrier.
 */
__global__ __launch_bounds__(kThreads, 2) void normKernel(BranchRMSNormArgs a)
{
    const auto task = static_cast<std::size_t>(blockIdx.x);
    if (task / 4 >= static_cast<std::size_t>(liveRows(a.extent))) return;
    const auto base = task * a.hidden;
    const unsigned lane = threadIdx.x;
    float partial = 0.0f;
    for (unsigned channel = lane; channel < static_cast<unsigned>(a.hidden); channel += kThreads)
    {
        const float value = a.input[base + channel];
        partial = fmaf(value, value, partial);
    }
    __shared__ float sums[kThreads];
    sums[lane] = partial;
    __syncthreads();
#if defined(QFN_ROCM)
    // Publish edge128 across waves before wave0 reads both sides of edge64.
    // Keep the original left/right operand priority at each shared-memory edge.
    if (lane < 128) sums[lane] = __fadd_rn(sums[lane], sums[lane + 128]);
    __syncthreads();
    if (lane < 64)
    {
        // Edge64 now lives in the owning wave's registers. Every lane executes
        // each shuffle, but only the original edge owners perform RN additions;
        // this preserves the full 128/64/32/.../1 tree, including NaN operands.
        float sum = __fadd_rn(sums[lane], sums[lane + 64]);
#pragma unroll
        for (unsigned stride = 32; stride > 0; stride /= 2)
        {
            const float partner = __shfl_down(sum, stride, 64);
            if (lane < stride) sum = __fadd_rn(sum, partner);
        }
        if (lane == 0) sums[0] = sum;
    }
    // Unlike a lane0-only projection reducer, normalization has consumers in
    // all four waves. Publish lane0's result before any output lane reads it.
    __syncthreads();
#else
#pragma unroll
    for (unsigned stride = kThreads / 2; stride > 0; stride /= 2)
    {
        if (lane < stride) sums[lane] = __fadd_rn(sums[lane], sums[lane + stride]);
        __syncthreads();
    }
#endif
    const float inverse = 1.0f / sqrtf(sums[0] / static_cast<float>(a.hidden) + a.epsilon);
    const auto gainBase = (task % 4) * a.hidden;
    for (unsigned channel = lane; channel < static_cast<unsigned>(a.hidden); channel += kThreads)
        a.output[base + channel] = __fmul_rn(__fmul_rn(a.input[base + channel], inverse), a.gain[gainBase + channel]);
}
/** @brief Expand ordinary embedding channels into four branch-major streams. */
__global__ __launch_bounds__(kThreads, 2) void expandKernel(ResidualExpandArgs a)
{
    const auto i = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
    const auto hidden = static_cast<std::size_t>(a.hidden);
    if (i < static_cast<std::size_t>(liveRows(a.extent)) * 4 * hidden)
        a.output[i] = a.input[(i / (4 * hidden)) * hidden + i % hidden];
}
/** @brief Apply the checkpoint's divide-by-four bottleneck SiLU. */
__global__ __launch_bounds__(kThreads, 2) void downKernel(ResidualDownArgs a)
{
    const auto i = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i < static_cast<std::size_t>(liveRows(a.extent)) * a.lowRank)
    {
        const float scaled = a.input[i] * 0.25f;
        a.output[i] = scaled * sigmoid(scaled);
    }
}
/** @brief Collapse branches with four sigmoid-gated terms in checkpoint order. */
__global__ __launch_bounds__(kThreads, 2) void readKernel(ResidualReadArgs a)
{
    const auto i = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
    const auto hidden = static_cast<std::size_t>(a.hidden);
    if (i >= static_cast<std::size_t>(liveRows(a.extent)) * hidden) return;
    const auto base = (i / hidden) * 4 * hidden + i % hidden;
    float sum = 0.0f;
#pragma unroll
    for (int branch = 0; branch < 4; ++branch)
    {
        const auto offset = base + branch * hidden;
        sum = fmaf(sigmoid(a.gateLogits[offset]), a.normalized[offset], sum);
    }
    a.output[i] = sum * 0.25f;
}
/** @brief Inject one scalar branch gate across a coalesced 256-channel block. */
__global__ __launch_bounds__(kThreads, 2) void writeKernel(ResidualWriteArgs a)
{
    const auto blocks = (static_cast<unsigned>(a.hidden) + kThreads - 1) / kThreads;
    const auto rowBranch = static_cast<std::size_t>(blockIdx.x) / blocks;
    if (rowBranch / 4 >= static_cast<std::size_t>(liveRows(a.extent))) return;
    const auto channel = (blockIdx.x % blocks) * kThreads + threadIdx.x;
    if (channel >= static_cast<unsigned>(a.hidden)) return;
    const float gate = 2.0f * sigmoid(a.gateLogits[rowBranch] * 0.25f);
    const auto offset = rowBranch * a.hidden + channel;
    const float update = __fmul_rn(gate, a.block[(rowBranch / 4) * a.hidden + channel]);
    a.output[offset] = __fadd_rn(a.residual[offset], update);
}
/** @brief Device-friendly hash constants with no host-only std::array methods. */
struct DeviceHashPolicy
{
    std::uint64_t multipliers[3]; ///< Exact unsigned multiplier bit patterns.
    std::uint64_t moduli[kNgramHeads]; ///< Positive divisors per checkpoint head.
    std::uint64_t offsets[kNgramHeads]; ///< Absolute row starts.
    std::int64_t eos; ///< Hash segment boundary token.
};
/** @brief Hash all sixteen suffix heads per live token using uint64 arithmetic. */
__global__ __launch_bounds__(kThreads, 2) void hashKernel(
    RowExtent extent, const std::int32_t *tokens, const std::int32_t *prior,
    std::int64_t *output, DeviceHashPolicy policy)
{
    const auto i = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= static_cast<std::size_t>(liveRows(extent)) * kNgramHeads) return;
    const auto row = i / kNgramHeads;
    const auto head = i % kNgramHeads;
    const auto previous = row > 0 ? tokens[row - 1] : prior[1];
    auto secondPrevious = row > 1 ? tokens[row - 2] : prior[row];
    if (previous == policy.eos) secondPrevious = policy.eos;
    auto hash = (static_cast<std::uint64_t>(tokens[row]) * policy.multipliers[0]) ^
                (static_cast<std::uint64_t>(previous) * policy.multipliers[1]);
    if (head >= 8) hash ^= static_cast<std::uint64_t>(secondPrevious) * policy.multipliers[2];
    output[i] = static_cast<std::int64_t>(hash % policy.moduli[head] + policy.offsets[head]);
}
/** @brief Fixed-lane similarity reduction followed by coalesced gated-value stores. */
__global__ __launch_bounds__(kThreads, 2) void gateKernel(PLEGateArgs a)
{
    const auto task = static_cast<std::size_t>(blockIdx.x);
    if (task / 4 >= static_cast<std::size_t>(liveRows(a.extent))) return;
    const auto base = task * a.hidden;
    const unsigned lane = threadIdx.x;
    float partial = 0.0f;
    for (unsigned channel = lane; channel < static_cast<unsigned>(a.hidden); channel += kThreads)
        partial = fmaf(a.normalizedKey[base + channel], a.normalizedQuery[base + channel], partial);
    __shared__ float sums[kThreads];
    sums[lane] = partial;
    __syncthreads();
#pragma unroll
    for (unsigned stride = kThreads / 2; stride > 0; stride /= 2)
    {
        if (lane < stride) sums[lane] = __fadd_rn(sums[lane], sums[lane + stride]);
        __syncthreads();
    }
    const float similarity = sums[0] / sqrtf(static_cast<float>(a.hidden));
    const float sign = similarity > 0 ? 1.0f : (similarity < 0 ? -1.0f : 0.0f);
    const float gate = sigmoid(sign * sqrtf(fmaxf(fabsf(similarity), 1e-6f)));
    const auto valueBase = (task / 4) * a.hidden;
    for (unsigned channel = lane; channel < static_cast<unsigned>(a.hidden); channel += kThreads)
        a.gated[base + channel] = gate * a.value[valueBase + channel];
}
/** @brief Evaluate four chronological dilated taps without candidate-row dependence. */
__global__ __launch_bounds__(kThreads, 2) void convKernel(PLEConvArgs a)
{
    const auto i = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
    const auto channels = static_cast<std::size_t>(a.channels);
    if (i >= static_cast<std::size_t>(liveRows(a.extent)) * channels) return;
    const auto channel = i % channels;
    const auto row = static_cast<std::ptrdiff_t>(i / channels);
    float sum = 0.0f;
#pragma unroll
    for (int tap = 0; tap < 4; ++tap)
    {
        const auto source = row - 9 + 3 * tap;
        const float value = source < 0 ? a.priorHistory[(source + 9) * channels + channel]
                                      : a.normalized[source * channels + channel];
        sum = fmaf(value, a.weights[channel * 4 + tap], sum);
    }
    const float ple = __fadd_rn(a.gated[i], sum * sigmoid(sum));
    a.output[i] = __fadd_rn(a.residual[i], ple);
}
/** @brief Publish only the normalized tail of the explicitly accepted prefix. */
__global__ __launch_bounds__(kThreads, 2) void commitConvKernel(ConvHistoryCommitArgs a)
{
    const auto i = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
    const auto channels = static_cast<std::size_t>(a.channels);
    if (i >= 9 * channels) return;
    const auto accepted = static_cast<std::ptrdiff_t>(acceptedRows(a.prefix));
    const auto source = accepted - 9 + static_cast<std::ptrdiff_t>(i / channels);
    const auto channel = i % channels;
    a.outputHistory[i] = source < 0 ? a.priorHistory[(source + 9) * channels + channel]
                                  : a.candidates[source * channels + channel];
}
/** @brief Publish two raw token IDs; all other GPU lanes remain inactive. */
__global__ __launch_bounds__(kThreads, 2) void commitTokenKernel(TokenHistoryCommitArgs a)
{
    const int index = static_cast<int>(threadIdx.x);
    const int accepted = acceptedRows(a.prefix);
    if (index < 2)
    {
        const int source = accepted - 2 + index;
        a.outputTokens[index] = source < 0 ? a.priorTokens[source + 2] : a.candidates[source];
    }
}
/** @brief Reject null, legacy-default and per-thread-default stream handles. */
QFN_NATIVE_STREAM explicitStream(void *stream)
{
    const auto native = static_cast<QFN_NATIVE_STREAM>(stream);
    detail::require(native && native != QFN_DEFAULT_STREAM && native != QFN_THREAD_STREAM,
                    "Qwen4Exp GPU launch requires an exact non-default producer stream");
    return native;
}
/** @brief Check launch submission without introducing blocking synchronization. */
void checkLaunch()
{
    const QFN_NATIVE_ERROR status = QFN_GET_ERROR();
    if (status != QFN_SUCCESS)
        throw std::runtime_error(std::string("Qwen4Exp GPU kernel launch failed: ") + QFN_ERROR_STRING(status));
}
/** @brief Compute an exact storage-bound launch grid; live rows are checked on device. */
unsigned blocks(std::size_t count)
{
    const auto value = (count + kThreads - 1) / kThreads;
    detail::require(value <= static_cast<std::size_t>(INT_MAX), "Qwen4Exp launch grid exceeds backend capacity");
    return static_cast<unsigned>(value);
}
/** @brief Native stateless bundle whose pointers and stream remain graph-owned. */
class QFN_IMPLEMENTATION final : public IQwen4ExpKernels
{
public:
    /** @copydoc IQwen4ExpKernels::branchRMSNorm */
    void branchRMSNorm(const BranchRMSNormArgs &a, void *stream) const override
    {
        detail::validate(a);
        detail::require(a.extent.rows <= INT_MAX / 4, "Qwen4Exp branch norm grid exceeds backend capacity");
        normKernel<<<static_cast<unsigned>(a.extent.rows) * 4, kThreads, 0, explicitStream(stream)>>>(a);
        checkLaunch();
    }
    /** @copydoc IQwen4ExpKernels::residualExpand */
    void residualExpand(const ResidualExpandArgs &a, void *stream) const override
    {
        detail::validate(a);
        expandKernel<<<blocks(static_cast<std::size_t>(a.extent.rows) * 4 * a.hidden), kThreads, 0,
                       explicitStream(stream)>>>(a);
        checkLaunch();
    }
    /** @copydoc IQwen4ExpKernels::residualDown */
    void residualDown(const ResidualDownArgs &a, void *stream) const override
    {
        detail::validate(a);
        downKernel<<<blocks(static_cast<std::size_t>(a.extent.rows) * a.lowRank), kThreads, 0,
                     explicitStream(stream)>>>(a);
        checkLaunch();
    }
    /** @copydoc IQwen4ExpKernels::residualRead */
    void residualRead(const ResidualReadArgs &a, void *stream) const override
    {
        detail::validate(a);
        readKernel<<<blocks(static_cast<std::size_t>(a.extent.rows) * a.hidden), kThreads, 0,
                     explicitStream(stream)>>>(a);
        checkLaunch();
    }
    /** @copydoc IQwen4ExpKernels::residualWrite */
    void residualWrite(const ResidualWriteArgs &a, void *stream) const override
    {
        detail::validate(a);
        const auto grid = static_cast<std::size_t>(a.extent.rows) * 4 * blocks(a.hidden);
        detail::require(grid <= INT_MAX, "Qwen4Exp residual write grid exceeds backend capacity");
        writeKernel<<<static_cast<unsigned>(grid), kThreads, 0, explicitStream(stream)>>>(a);
        checkLaunch();
    }
    /** @copydoc IQwen4ExpKernels::ngramHash */
    void ngramHash(const NgramHashArgs &a, void *stream) const override
    {
        detail::validate(a);
        DeviceHashPolicy policy{};
        for (int index = 0; index < 3; ++index) policy.multipliers[index] = a.policy.multipliers[index];
        for (int head = 0; head < kNgramHeads; ++head)
        {
            policy.moduli[head] = a.policy.moduli[head];
            policy.offsets[head] = a.policy.offsets[head];
        }
        policy.eos = a.policy.eosToken;
        hashKernel<<<blocks(static_cast<std::size_t>(a.extent.rows) * kNgramHeads), kThreads, 0,
                     explicitStream(stream)>>>(a.extent, a.tokens, a.priorTokens, a.output, policy);
        checkLaunch();
    }
    /** @copydoc IQwen4ExpKernels::pleGate */
    void pleGate(const PLEGateArgs &a, void *stream) const override
    {
        detail::validate(a);
        detail::require(a.extent.rows <= INT_MAX / 4, "Qwen4Exp similarity grid exceeds backend capacity");
        gateKernel<<<static_cast<unsigned>(a.extent.rows) * 4, kThreads, 0, explicitStream(stream)>>>(a);
        checkLaunch();
    }
    /** @copydoc IQwen4ExpKernels::pleConv */
    void pleConv(const PLEConvArgs &a, void *stream) const override
    {
        detail::validate(a);
        convKernel<<<blocks(static_cast<std::size_t>(a.extent.rows) * a.channels), kThreads, 0,
                     explicitStream(stream)>>>(a);
        checkLaunch();
    }
    /** @copydoc IQwen4ExpKernels::commitConvHistory */
    void commitConvHistory(const ConvHistoryCommitArgs &a, void *stream) const override
    {
        detail::validate(a);
        commitConvKernel<<<blocks(static_cast<std::size_t>(9) * a.channels), kThreads, 0,
                           explicitStream(stream)>>>(a);
        checkLaunch();
    }
    /** @copydoc IQwen4ExpKernels::commitTokenHistory */
    void commitTokenHistory(const TokenHistoryCommitArgs &a, void *stream) const override
    {
        detail::validate(a);
        commitTokenKernel<<<1, kThreads, 0, explicitStream(stream)>>>(a);
        checkLaunch();
    }
};
} // namespace QFN_BACKEND_NAMESPACE

std::unique_ptr<IQwen4ExpKernels> QFN_FACTORY()
{
    return std::make_unique<QFN_BACKEND_NAMESPACE::QFN_IMPLEMENTATION>();
}
} // namespace llaminar2::qwen4exp

#undef QFN_NATIVE_STREAM
#undef QFN_NATIVE_ERROR
#undef QFN_GET_ERROR
#undef QFN_ERROR_STRING
#undef QFN_SUCCESS
#undef QFN_DEFAULT_STREAM
#undef QFN_THREAD_STREAM
#undef QFN_FACTORY
#undef QFN_IMPLEMENTATION
#undef QFN_BACKEND_NAMESPACE
