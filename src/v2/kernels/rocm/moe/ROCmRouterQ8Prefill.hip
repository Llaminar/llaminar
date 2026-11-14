/**
 * @file ROCmRouterQ8Prefill.hip
 * @brief Wave64 INT8 WMMA router with the serial adjacent-pair-first FP32 tree.
 *
 * Cooperating wave64 waves own disjoint K subtrees of sixteen rows and sixteen
 * experts. Two integer WMMA instructions form each exact K32 dot; independently
 * rounded scales are applied before the original 128-owner tree. Register-local
 * subtrees join once through LDS in the same arithmetic order. No global partial
 * storage or atomics are needed. Only actual gfx1100 code objects contain WMMA.
 */
#include <hip/hip_runtime.h>
#include "ROCmRouterQ8Prefill.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include <algorithm>
#include <stdexcept>

namespace llaminar2::rocm
{
namespace
{
thread_local RouterQ8PrefillMode recording_mode = RouterQ8PrefillMode::Automatic;

/** @return Whether the admitted native target owns this exact wave64 instruction. */
bool nativeRouterDevice(int device)
{
    const auto &hardware = rocmHardwareMetadata().at(device);
    const auto &isa = hardware.isa_target;
    return (isa == "gfx1100" || isa.starts_with("gfx1100:")) &&
        hardware.maximum_threads_per_block >= 256 && hardware.lds_per_block >= 4096;
}
}

ScopedRouterQ8PrefillMode::ScopedRouterQ8PrefillMode(RouterQ8PrefillMode mode)
    : previous_(recording_mode)
{
    if (mode != RouterQ8PrefillMode::Automatic && mode != RouterQ8PrefillMode::OrderedDot4 &&
        mode != RouterQ8PrefillMode::NativeWmma)
        throw std::invalid_argument("unknown ROCm Q8 router recording mode");
    recording_mode = mode;
}
ScopedRouterQ8PrefillMode::~ScopedRouterQ8PrefillMode() { recording_mode = previous_; }

bool selectRouterQ8Wmma(int device, int rows, int width, int experts)
{
    if (recording_mode != RouterQ8PrefillMode::NativeWmma) return false;
    if (rows < 0 || width <= 0 || width % 32 || experts <= 0 || !nativeRouterDevice(device))
        throw std::invalid_argument("native Q8 router WMMA requires gfx1100 and complete block32 geometry");
    return true;
}

namespace router_q8_prefill_detail
{
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__)
using I32x4 = int __attribute__((ext_vector_type(4)));
using F32x4 = float __attribute__((ext_vector_type(4)));

/** @brief Fold matching fragment elements with explicit round-to-nearest adds.
 * @param left Earlier subtree. @param right Later subtree. @return Exact parent. */
__device__ __forceinline__ F32x4 add(F32x4 left, F32x4 right)
{
    F32x4 result;
    #pragma unroll
    for (int e = 0; e < 4; ++e) result[e] = __fadd_rn(left[e], right[e]);
    return result;
}

/**
 * @brief Compute one output tile with the same leaf order as serial wave64 decode.
 * @tparam LogLeaves Smallest power-of-two extent covering nonempty serial owners.
 * @tparam Partitioned Whether to publish the compact participant-owned interval.
 * @tparam FixedWidth Compile-time contraction width, or zero for arbitrary admitted widths.
 * @param view Captured banks, physical capacity and optional device live count.
 */
template<int LogLeaves, bool Partitioned, int FixedWidth = 0>
__global__ __launch_bounds__(64 * (1 << (LogLeaves < 2 ? LogLeaves : 2)))
void kernel(RouterQ8PrefillView view)
{
#if defined(__gfx1100__)
    static_assert(__builtin_amdgcn_is_invocable(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64));
    static_assert(LogLeaves >= 0 && LogLeaves <= 7);
    const int width = FixedWidth ? FixedWidth : view.width;
    const int experts = view.experts, blocks = width / 32;
    int rows = view.live_rows ? max(0, min(view.capacity, *view.live_rows)) : view.capacity;
    if constexpr (Partitioned)
    {
        const auto local = view.partition.resolve(view.capacity, view.live_rows);
        view.hidden_codes += static_cast<size_t>(local.first) * width;
        view.hidden_scales += static_cast<size_t>(local.first) * blocks;
        rows = local.count;
    }
    const int row_base = blockIdx.y * 16, expert_base = blockIdx.x * 16;
    if (row_base >= rows || expert_base >= experts) return;
    constexpr int LogKWaves = LogLeaves < 2 ? LogLeaves : 2;
    constexpr int LocalLogLeaves = LogLeaves - LogKWaves;
    const int lane = threadIdx.x % 64, k_wave = threadIdx.x / 64;
    const int operand_row = row_base + lane % 16;
    const int expert = expert_base + lane % 16;
    F32x4 pending[LocalLogLeaves ? LocalLogLeaves : 1];
    F32x4 total = {};
    #pragma unroll 1
    for (int leaf = k_wave * (1 << LocalLogLeaves); leaf < (k_wave + 1) * (1 << LocalLogLeaves); ++leaf)
    {
        F32x4 value = {};
        for (int block = leaf; block < blocks; block += 128)
        {
            I32x4 al = {}, ah = {}, bl = {}, bh = {};
            float activation_scale = 0.0f, weight_scale = 0.0f;
            if (operand_row < rows)
            {
                const auto *a = reinterpret_cast<const I32x4 *>(
                    view.hidden_codes + static_cast<size_t>(operand_row) * width + block * 32);
                al = a[0]; ah = a[1];
                activation_scale = view.hidden_scales[static_cast<size_t>(operand_row) * blocks + block];
            }
            if (expert < experts)
            {
                const auto *b = reinterpret_cast<const I32x4 *>(
                    view.gate_codes + static_cast<size_t>(expert) * width + block * 32);
                bl = b[0]; bh = b[1];
                weight_scale = view.gate_scales[static_cast<size_t>(expert) * blocks + block];
            }
            const I32x4 zero = {};
            const auto low = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true, al, true, bl, zero, false);
            const auto dots = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true, ah, true, bh, low, false);
            #pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                // Every sixteen-lane group loads the same row-scale bank.
                // Stay within that group instead of requesting a cross-half
                // wave64 permute for values already present in each quadrant.
                const float scale = __shfl(activation_scale, e * 4 + lane / 16, 16);
                value[e] = __fmaf_rn(__fmul_rn(static_cast<float>(dots[e]), scale), weight_scale, value[e]);
            }
        }
        // Unlike CUDA's descending shuffle tree, ROCm combines adjacent leaves
        // first. Constant-index pending slots stay registers after unrolling.
        bool carry = true;
        #pragma unroll
        for (int level = 0; level < LocalLogLeaves; ++level)
        {
            if (carry)
            {
                if (leaf & (1 << level)) value = add(pending[level], value);
                else { pending[level] = value; carry = false; }
            }
        }
        total = value;
    }
    if constexpr (LogKWaves > 0)
    {
        // Row/expert exits are block-uniform. Each K wave publishes one complete
        // adjacent subtree before wave zero joins in the serial parent order.
        __shared__ F32x4 subtrees[(1 << LogKWaves)][64];
        subtrees[k_wave][lane] = total;
        __syncthreads();
        if (k_wave != 0) return;
        total = add(subtrees[0][lane], subtrees[1][lane]);
        if constexpr (LogKWaves == 2)
            total = add(total, add(subtrees[2][lane], subtrees[3][lane]));
    }
    #pragma unroll
    for (int e = 0; e < 4; ++e)
    {
        const int row = row_base + e * 4 + lane / 16;
        // Empty upper owners and the serial block's final wave fold contribute
        // positive zero. Retain its signed-zero effect even when omitted.
        if (row < rows && expert < experts)
            view.logits[static_cast<size_t>(row) * experts + expert] = __fadd_rn(total[e], 0.0f);
    }
#endif
}
#endif
}

bool routerQ8Prefill(const RouterQ8PrefillView &view, void *stream)
{
    const auto native_stream = static_cast<hipStream_t>(stream);
    if (!native_stream || native_stream == hipStreamLegacy || native_stream == hipStreamPerThread ||
        !view.hidden_codes || !view.hidden_scales || !view.gate_codes || !view.gate_scales ||
        !view.logits || view.device < 0 || view.capacity <= 0 || view.width <= 0 || view.width % 32 || view.experts <= 0 ||
        (view.partition.participants() > 1 && !view.live_rows) || !nativeRouterDevice(view.device))
        return false;
    int stream_device = -1;
    if (hipStreamGetDevice(native_stream, &stream_device) != hipSuccess || stream_device != view.device)
        return false;
    const int rows = view.partition.capacityFor(view.capacity);
    if (rows == 0) return true;
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__)
    int log_leaves = 0;
    while ((1 << log_leaves) < std::min(128, view.width / 32)) ++log_leaves;
    auto submit = [&]<int LogLeaves, int FixedWidth = 0>()
    {
        const dim3 grid((view.experts + 15) / 16, (rows + 15) / 16);
        constexpr int threads = 64 * (1 << (LogLeaves < 2 ? LogLeaves : 2));
        if (view.partition.participants() > 1)
            router_q8_prefill_detail::kernel<LogLeaves, true, FixedWidth><<<grid, threads, 0, native_stream>>>(view);
        else
            router_q8_prefill_detail::kernel<LogLeaves, false, FixedWidth><<<grid, threads, 0, native_stream>>>(view);
        return hipGetLastError() == hipSuccess;
    };
    // Width specialization removes dynamic row strides and redundant inner K
    // loop tests. It retains the same 128-owner arithmetic and K-subtree fold.
    switch (view.width)
    {
    case 2048: return submit.template operator()<6,2048>();
    case 2560: return submit.template operator()<7,2560>();
    case 3072: return submit.template operator()<7,3072>();
    case 4096: return submit.template operator()<7,4096>();
    case 8192: return submit.template operator()<7,8192>();
    default: break;
    }
    switch (log_leaves)
    {
#define ROUTER_WMMA_LEAVES(LOG) case LOG: return submit.template operator()<LOG>()
        ROUTER_WMMA_LEAVES(0); ROUTER_WMMA_LEAVES(1); ROUTER_WMMA_LEAVES(2); ROUTER_WMMA_LEAVES(3);
        ROUTER_WMMA_LEAVES(4); ROUTER_WMMA_LEAVES(5); ROUTER_WMMA_LEAVES(6); ROUTER_WMMA_LEAVES(7);
#undef ROUTER_WMMA_LEAVES
    }
#endif
    return false;
}
}
