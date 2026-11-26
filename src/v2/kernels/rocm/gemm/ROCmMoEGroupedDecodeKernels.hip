/**
 * @file ROCmMoEGroupedDecodeKernels.hip
 * @brief ROCm NativeQuantised grouped MoE decode kernels and launchers.
 */
#include "ROCmNativeQuantisedMoEDeviceHelpers.hip.inc"

LLAMINAR_DEFINE_ROCM_IQ_GRID_INIT(moe_decode)

namespace llaminar2::rocm
{
namespace { thread_local GateUpProducerLoads gate_up_producer_loads=GateUpProducerLoads::Original; }
/** @copydoc currentGateUpProducerLoads */
GateUpProducerLoads currentGateUpProducerLoads() noexcept {return gate_up_producer_loads;}
/** @copydoc ScopedGateUpProducerLoads::ScopedGateUpProducerLoads */
ScopedGateUpProducerLoads::ScopedGateUpProducerLoads(GateUpProducerLoads loads):previous_(gate_up_producer_loads)
{
    if(llaminar2::isGraphCaptureActive())throw std::logic_error("Gate/up operand loads must be selected before capture");
    if(loads!=GateUpProducerLoads::Original&&loads!=GateUpProducerLoads::PairedK32)
        throw std::invalid_argument("Unknown gate/up operand load policy");
    gate_up_producer_loads=loads;
}
/** @copydoc ScopedGateUpProducerLoads::~ScopedGateUpProducerLoads */
ScopedGateUpProducerLoads::~ScopedGateUpProducerLoads(){gate_up_producer_loads=previous_;}


}

__global__ void moe_grouped_swiglu_quantize_blockwise_kernel(
    const float* const* __restrict__ gate_ptrs,
    const float* const* __restrict__ up_ptrs,
    const int* __restrict__ expert_ids,
    int8_t* __restrict__ A_int8,
    float* __restrict__ scales_A_blockwise,
    int num_active,
    int K)
{
    const int slot = blockIdx.x;
    if (slot >= num_active) return;
    if (expert_ids && expert_ids[slot] < 0) return;

    constexpr int BLOCK_SIZE = 32;
    constexpr int NUM_WARPS = 8;
    const int blocks_per_row = K / BLOCK_SIZE;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    const float* gate = gate_ptrs[slot];
    const float* up = up_ptrs[slot];
    int8_t* row_int8 = A_int8 + static_cast<size_t>(slot) * K;
    float* row_scales = scales_A_blockwise + static_cast<size_t>(slot) * blocks_per_row;

    for (int b = warp_id; b < blocks_per_row; b += NUM_WARPS)
    {
        const int k_start = b * BLOCK_SIZE;
        const float g = gate[k_start + lane];
        const float u = up[k_start + lane];
        const float val =
            llaminar2::rocm::decode_equivalent::swigluValue(g, u);

        float abs_val = fabsf(val);
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            abs_val = fmaxf(abs_val, __shfl_xor(abs_val, mask));

        const float scale =
            llaminar2::rocm::decode_equivalent::q8Scale(abs_val);

        if (lane == 0)
            row_scales[b] = scale;

        row_int8[k_start + lane] = static_cast<int8_t>(
            llaminar2::rocm::decode_equivalent::quantizeQ8FromScale(
                val, scale));
    }
}


/**
 * @brief Publish one weighted ROCm decode row per original router slot.
 *
 * Each `(route, column)` lane computes a complete increasing-K NativeQuantised dot
 * and writes a unique destination. Remote/invalid routes write zero. No route
 * accumulation or floating-point atomic occurs before the LocalTP collective.
 */
template <NativeQuantisedFormat FMT, bool INDEX_BY_EXPERT_ID = true>
__global__ void moe_grouped_native_quantised_down_route_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ descs,
    const int* __restrict__ expert_ids,
    const float* __restrict__ route_weights,
    float* __restrict__ route_output,
    int num_active,
    int N,
    int K)
{
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
                  "Canonical grouped MoE down requires a supported format");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int route = blockIdx.y;
    if (route >= num_active || n >= N)
        return;

    const size_t output_index =
        static_cast<size_t>(route) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    const int descriptor_index = INDEX_BY_EXPERT_ID ? expert_ids[route] : route;
    if (descriptor_index < 0)
    {
        route_output[output_index] = 0.0f;
        return;
    }

    const DeviceNativeQuantisedMatrixDesc desc = descs[descriptor_index];
    if (!native_quantised_desc_shape_ok<FMT>(desc, N, K))
    {
        route_output[output_index] = 0.0f;
        return;
    }

    const int blocks_per_row = K / 32;
    const int8_t* a_slot =
        A_int8 + static_cast<size_t>(route) * static_cast<size_t>(K);
    const float* scale_slot =
        scales_A_blockwise +
        static_cast<size_t>(route) * static_cast<size_t>(blocks_per_row);
    const float expert_acc = moe_native_quantised_dot_desc_range<FMT>(
        desc, n, a_slot, scale_slot, N, K, 0, blocks_per_row);
    route_output[output_index] = rocm_native_quantised_weight_route_rn(
        route_weights[route], expert_acc);
}

/**
 * @brief Fold independently computed route rows in the serial router order.
 *
 * Route projection is parallel across grid Y. This reducer owns one output
 * column and reproduces the serial decode sequence `(((0 + r0) + r1) + ...)`
 * with explicit round-to-nearest operations. No route can race another route's
 * destination, and no scheduler-dependent atomic ordering enters the result.
 */
__global__ void moe_grouped_native_quantised_down_ordered_reduce_kernel(
    const float* __restrict__ route_contributions,
    float* __restrict__ output,
    int num_active,
    int N)
{
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    if (n >= N)
        return;

    float total = 0.0f;
    #pragma unroll 1
    for (int route = 0; route < num_active; ++route)
    {
        total = rocm_native_quantised_accumulate_rn(
            total,
            route_contributions[
                static_cast<size_t>(route) * static_cast<size_t>(N) +
                static_cast<size_t>(n)]);
    }
    output[n] = total;
}

__global__ void moe_grouped_hidden_quantize_blockwise_kernel(
    const float* __restrict__ hidden,
    int8_t* __restrict__ A_int8,
    float* __restrict__ scales_A_blockwise,
    int K)
{
    constexpr int BLOCK_SIZE = 32;
    const int block_idx = blockIdx.x;
    const int lane = threadIdx.x;
    const int k = block_idx * BLOCK_SIZE + lane;
    if (lane >= BLOCK_SIZE || k >= K) return;

    const float val = hidden[k];
    float abs_val = fabsf(val);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        abs_val = fmaxf(abs_val, __shfl_xor(abs_val, mask));

    const float scale =
        llaminar2::rocm::decode_equivalent::q8Scale(abs_val);
    if (lane == 0)
        scales_A_blockwise[block_idx] = scale;

    A_int8[k] = static_cast<int8_t>(
        llaminar2::rocm::decode_equivalent::quantizeQ8FromScale(
            val, scale));
}

/**
 * @brief Compute one ordered split-K partial for each routed expert row.
 *
 * Scratch is laid out as `[KB][route][N]`, matching serial GEMV's outer KB
 * dimension. The first KB plane is therefore also a contiguous `[route][N]`
 * destination and can be overwritten by the following reducer after each lane
 * has consumed its own KB-zero value.
 */
template <NativeQuantisedFormat FMT, bool INDEX_BY_EXPERT_ID = true>
__global__ void moe_grouped_native_quantised_down_kpart_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ descs,
    const int* __restrict__ expert_ids,
    float* __restrict__ partials,
    int num_active,
    int N,
    int K,
    int k_partitions)
{
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
                  "Grouped MoE split-K down requires a supported format");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int route = blockIdx.y;
    const int k_part = blockIdx.z;
    if (route >= num_active || k_part >= k_partitions || n >= N)
        return;

    const size_t partial_index =
        (static_cast<size_t>(k_part) * static_cast<size_t>(num_active) +
         static_cast<size_t>(route)) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    const int descriptor_index = INDEX_BY_EXPERT_ID ? expert_ids[route] : route;
    if (descriptor_index < 0)
    {
        partials[partial_index] = 0.0f;
        return;
    }

    const DeviceNativeQuantisedMatrixDesc desc = descs[descriptor_index];
    if (!native_quantised_desc_shape_ok<FMT>(desc, N, K))
    {
        partials[partial_index] = 0.0f;
        return;
    }

    const int blocks_per_row = K / 32;
    const int blocks_per_part =
        (blocks_per_row + k_partitions - 1) / k_partitions;
    const int b_start = k_part * blocks_per_part;
    int b_end = b_start + blocks_per_part;
    if (b_end > blocks_per_row)
        b_end = blocks_per_row;

    const int8_t* route_a =
        A_int8 + static_cast<size_t>(route) * static_cast<size_t>(K);
    const float* route_scales =
        scales_A_blockwise +
        static_cast<size_t>(route) * static_cast<size_t>(blocks_per_row);
    partials[partial_index] = b_start < b_end
        ? moe_native_quantised_dot_desc_range<FMT>(
              desc, n, route_a, route_scales, N, K, b_start, b_end)
        : 0.0f;
}

/**
 * @brief Reduce one expert's K partials and publish its rounded route row.
 *
 * The loop is deliberately scalar and increasing in KB, exactly matching the
 * serial M=1 reducer. `route_output` may alias the first KB plane of `partials`:
 * each lane reads every source for its own `(route,column)` before replacing
 * that lane's KB-zero value.
 */
__global__ void moe_grouped_native_quantised_down_kpart_reduce_kernel(
    const float* __restrict__ partials,
    const float* __restrict__ route_weights,
    float* __restrict__ route_output,
    int num_active,
    int N,
    int k_partitions)
{
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int route = blockIdx.y;
    if (route >= num_active || n >= N)
        return;

    float expert_sum = 0.0f;
    #pragma unroll 1
    for (int k_part = 0; k_part < k_partitions; ++k_part)
    {
        const size_t partial_index =
            (static_cast<size_t>(k_part) * static_cast<size_t>(num_active) +
             static_cast<size_t>(route)) * static_cast<size_t>(N) +
            static_cast<size_t>(n);
        expert_sum = rocm_native_quantised_accumulate_rn(
            expert_sum, partials[partial_index]);
    }

    route_output[
        static_cast<size_t>(route) * static_cast<size_t>(N) +
        static_cast<size_t>(n)] = rocm_native_quantised_weight_route_rn(
            route_weights[route], expert_sum);
}

/**
 * @brief Compute table-backed gate/up partials with serial-M1 K boundaries.
 */
template <NativeQuantisedFormat FMT>
__global__ void moe_grouped_native_quantised_gate_up_kpart_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const int* __restrict__ expert_ids,
    float* __restrict__ gate_partials,
    float* __restrict__ up_partials,
    int num_active,
    int N,
    int K,
    int k_partitions)
{
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
                  "Grouped MoE split-K gate/up requires a supported format");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int route = blockIdx.y;
    const int k_part = blockIdx.z;
    if (route >= num_active || k_part >= k_partitions || n >= N)
        return;

    const size_t partial_index =
        (static_cast<size_t>(k_part) * static_cast<size_t>(num_active) +
         static_cast<size_t>(route)) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    const int expert_id = expert_ids[route];
    if (expert_id < 0)
    {
        gate_partials[partial_index] = 0.0f;
        up_partials[partial_index] = 0.0f;
        return;
    }

    const DeviceNativeQuantisedMatrixDesc gate_desc = gate_descs[expert_id];
    const DeviceNativeQuantisedMatrixDesc up_desc = up_descs[expert_id];
    if (!native_quantised_desc_shape_ok<FMT>(gate_desc, N, K) ||
        !native_quantised_desc_shape_ok<FMT>(up_desc, N, K))
    {
        gate_partials[partial_index] = 0.0f;
        up_partials[partial_index] = 0.0f;
        return;
    }

    const int blocks_per_row = K / 32;
    const int blocks_per_part =
        (blocks_per_row + k_partitions - 1) / k_partitions;
    const int b_start = k_part * blocks_per_part;
    int b_end = b_start + blocks_per_part;
    if (b_end > blocks_per_row)
        b_end = blocks_per_row;

    gate_partials[partial_index] = b_start < b_end
        ? moe_native_quantised_dot_desc_range<FMT>(
              gate_desc, n, A_int8, scales_A_blockwise,
              N, K, b_start, b_end)
        : 0.0f;
    up_partials[partial_index] = b_start < b_end
        ? moe_native_quantised_dot_desc_range<FMT>(
              up_desc, n, A_int8, scales_A_blockwise,
              N, K, b_start, b_end)
        : 0.0f;
}

/**
 * @brief Publish table/runtime gate/up rows from ordered split-K partials.
 */
__global__ void moe_grouped_native_quantised_gate_up_kpart_reduce_kernel(
    const float* __restrict__ gate_partials,
    const float* __restrict__ up_partials,
    const int* __restrict__ expert_ids,
    float* const* __restrict__ gate_outputs,
    float* const* __restrict__ up_outputs,
    int num_active,
    int N,
    int k_partitions)
{
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int route = blockIdx.y;
    if (route >= num_active || n >= N)
        return;

    if (expert_ids[route] < 0)
    {
        gate_outputs[route][n] = 0.0f;
        up_outputs[route][n] = 0.0f;
        return;
    }

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    #pragma unroll 1
    for (int k_part = 0; k_part < k_partitions; ++k_part)
    {
        const size_t partial_index =
            (static_cast<size_t>(k_part) * static_cast<size_t>(num_active) +
             static_cast<size_t>(route)) * static_cast<size_t>(N) +
            static_cast<size_t>(n);
        gate_sum = rocm_native_quantised_accumulate_rn(
            gate_sum, gate_partials[partial_index]);
        up_sum = rocm_native_quantised_accumulate_rn(
            up_sum, up_partials[partial_index]);
    }

    gate_outputs[route][n] = gate_sum;
    up_outputs[route][n] = up_sum;
}

template <NativeQuantisedFormat FMT>
__global__ void moe_grouped_native_quantised_gate_up_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const int* __restrict__ expert_ids,
    float* const* __restrict__ gate_outputs,
    float* const* __restrict__ up_outputs,
    int num_active,
    int N,
    int K)
{
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
                  "Grouped MoE gate/up requires a supported native-quantised descriptor");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int slot = blockIdx.y;
    if (slot >= num_active || n >= N) return;

    const int expert_id = expert_ids[slot];
    if (expert_id < 0)
        return;
    const DeviceNativeQuantisedMatrixDesc gate_desc = gate_descs[expert_id];
    const DeviceNativeQuantisedMatrixDesc up_desc = up_descs[expert_id];
    if (!native_quantised_desc_shape_ok<FMT>(gate_desc, N, K) ||
        !native_quantised_desc_shape_ok<FMT>(up_desc, N, K))
    {
        return;
    }

    gate_outputs[slot][n] = moe_native_quantised_dot_desc<FMT>(
        gate_desc, n, A_int8, scales_A_blockwise, N, K);
    up_outputs[slot][n] = moe_native_quantised_dot_desc<FMT>(
        up_desc, n, A_int8, scales_A_blockwise, N, K);
}

template <NativeQuantisedFormat FMT>
__global__ void moe_grouped_native_quantised_gate_up_runtime_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* const* __restrict__ gate_outputs,
    float* const* __restrict__ up_outputs,
    int num_active,
    int N,
    int K,
    int num_experts)
{
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
                  "Grouped MoE runtime gate/up requires a supported native-quantised descriptor");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int slot = blockIdx.y;
    if (slot >= num_active || n >= N) return;

    const int expert_id = expert_ids[slot];
    if (expert_id < 0 || expert_id >= num_experts)
    {
        gate_outputs[slot][n] = 0.0f;
        up_outputs[slot][n] = 0.0f;
        return;
    }

    const uint32_t execution_bank = moe_runtime_execution_bank(runtime);
    if (execution_bank > 1u)
    {
        gate_outputs[slot][n] = 0.0f;
        up_outputs[slot][n] = 0.0f;
        return;
    }
    const DeviceMoEPlacementBankView &bank =
        moe_runtime_placement_banks(runtime)[execution_bank];
    if (!moe_runtime_native_descriptor_selected(runtime, bank, expert_id))
    {
        gate_outputs[slot][n] = 0.0f;
        up_outputs[slot][n] = 0.0f;
        return;
    }
    const DeviceNativeQuantisedMatrixDesc gate_desc = bank.experts[expert_id].gate;
    const DeviceNativeQuantisedMatrixDesc up_desc = bank.experts[expert_id].up;
    if (!native_quantised_desc_shape_ok<FMT>(gate_desc, N, K) ||
        !native_quantised_desc_shape_ok<FMT>(up_desc, N, K))
    {
        gate_outputs[slot][n] = 0.0f;
        up_outputs[slot][n] = 0.0f;
        return;
    }

    gate_outputs[slot][n] = moe_native_quantised_dot_desc<FMT>(
        gate_desc, n, A_int8, scales_A_blockwise, N, K);
    up_outputs[slot][n] = moe_native_quantised_dot_desc<FMT>(
        up_desc, n, A_int8, scales_A_blockwise, N, K);
}

/**
 * @brief Compute runtime-placement gate/up partials using serial-M1 K slices.
 */
template <NativeQuantisedFormat FMT>
__global__ void moe_grouped_native_quantised_gate_up_runtime_kpart_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* __restrict__ gate_partials,
    float* __restrict__ up_partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    int k_partitions)
{
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
                  "Runtime grouped MoE split-K gate/up requires a supported format");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int route = blockIdx.y;
    const int k_part = blockIdx.z;
    if (route >= num_active || k_part >= k_partitions || n >= N)
        return;

    const size_t partial_index =
        (static_cast<size_t>(k_part) * static_cast<size_t>(num_active) +
         static_cast<size_t>(route)) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    const int expert_id = expert_ids[route];
    if (expert_id < 0 || expert_id >= num_experts)
    {
        gate_partials[partial_index] = 0.0f;
        up_partials[partial_index] = 0.0f;
        return;
    }

    const uint32_t execution_bank = moe_runtime_execution_bank(runtime);
    if (execution_bank > 1u)
    {
        gate_partials[partial_index] = 0.0f;
        up_partials[partial_index] = 0.0f;
        return;
    }
    const DeviceMoEPlacementBankView &bank =
        moe_runtime_placement_banks(runtime)[execution_bank];
    if (!moe_runtime_native_descriptor_selected(runtime, bank, expert_id))
    {
        gate_partials[partial_index] = 0.0f;
        up_partials[partial_index] = 0.0f;
        return;
    }
    const DeviceNativeQuantisedMatrixDesc gate_desc = bank.experts[expert_id].gate;
    const DeviceNativeQuantisedMatrixDesc up_desc = bank.experts[expert_id].up;
    if (!native_quantised_desc_shape_ok<FMT>(gate_desc, N, K) ||
        !native_quantised_desc_shape_ok<FMT>(up_desc, N, K))
    {
        gate_partials[partial_index] = 0.0f;
        up_partials[partial_index] = 0.0f;
        return;
    }

    const int blocks_per_row = K / 32;
    const int blocks_per_part =
        (blocks_per_row + k_partitions - 1) / k_partitions;
    const int b_start = k_part * blocks_per_part;
    int b_end = b_start + blocks_per_part;
    if (b_end > blocks_per_row)
        b_end = blocks_per_row;

    gate_partials[partial_index] = b_start < b_end
        ? moe_native_quantised_dot_desc_range<FMT>(
              gate_desc, n, A_int8, scales_A_blockwise,
              N, K, b_start, b_end)
        : 0.0f;
    up_partials[partial_index] = b_start < b_end
        ? moe_native_quantised_dot_desc_range<FMT>(
              up_desc, n, A_int8, scales_A_blockwise,
              N, K, b_start, b_end)
        : 0.0f;
}

template <NativeQuantisedFormat FMT>
__global__ void moe_grouped_native_quantised_down_runtime_route_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    const float* __restrict__ route_weights,
    float* __restrict__ route_output,
    int num_active,
    int N,
    int K,
    int num_experts)
{
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
                  "Grouped MoE runtime down requires a supported native-quantised descriptor");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int slot = blockIdx.y;
    if (slot >= num_active || n >= N)
        return;

    const int expert_id = expert_ids[slot];
    const size_t output_index =
        static_cast<size_t>(slot) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    if (expert_id < 0 || expert_id >= num_experts)
    {
        route_output[output_index] = 0.0f;
        return;
    }

    const uint32_t execution_bank = moe_runtime_execution_bank(runtime);
    if (execution_bank > 1u)
    {
        route_output[output_index] = 0.0f;
        return;
    }
    const int blocks_per_row = K / 32;
    const DeviceMoEPlacementBankView &bank =
        moe_runtime_placement_banks(runtime)[execution_bank];
    if (!moe_runtime_native_descriptor_selected(runtime, bank, expert_id))
    {
        route_output[output_index] = 0.0f;
        return;
    }
    const DeviceNativeQuantisedMatrixDesc desc = bank.experts[expert_id].down;
    if (!native_quantised_desc_shape_ok<FMT>(desc, N, K))
    {
        route_output[output_index] = 0.0f;
        return;
    }

    const int8_t* a_slot = A_int8 + static_cast<size_t>(slot) * K;
    const float* scale_slot = scales_A_blockwise + static_cast<size_t>(slot) * blocks_per_row;
    const float expert_acc = moe_native_quantised_dot_desc<FMT>(
        desc, n, a_slot, scale_slot, N, K);
    route_output[output_index] = rocm_native_quantised_weight_route_rn(
        route_weights[slot], expert_acc);
}

/**
 * @brief Compute runtime-placement down partials using serial-M1 K slices.
 */
template <NativeQuantisedFormat FMT>
__global__ void moe_grouped_native_quantised_down_runtime_kpart_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* __restrict__ partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    int k_partitions)
{
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
                  "Runtime grouped MoE split-K down requires a supported format");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int route = blockIdx.y;
    const int k_part = blockIdx.z;
    if (route >= num_active || k_part >= k_partitions || n >= N)
        return;

    const size_t partial_index =
        (static_cast<size_t>(k_part) * static_cast<size_t>(num_active) +
         static_cast<size_t>(route)) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    const int expert_id = expert_ids[route];
    if (expert_id < 0 || expert_id >= num_experts)
    {
        partials[partial_index] = 0.0f;
        return;
    }

    const uint32_t execution_bank = moe_runtime_execution_bank(runtime);
    if (execution_bank > 1u)
    {
        partials[partial_index] = 0.0f;
        return;
    }
    const DeviceMoEPlacementBankView &bank =
        moe_runtime_placement_banks(runtime)[execution_bank];
    if (!moe_runtime_native_descriptor_selected(runtime, bank, expert_id))
    {
        partials[partial_index] = 0.0f;
        return;
    }
    const DeviceNativeQuantisedMatrixDesc desc = bank.experts[expert_id].down;
    if (!native_quantised_desc_shape_ok<FMT>(desc, N, K))
    {
        partials[partial_index] = 0.0f;
        return;
    }

    const int blocks_per_row = K / 32;
    const int blocks_per_part =
        (blocks_per_row + k_partitions - 1) / k_partitions;
    const int b_start = k_part * blocks_per_part;
    int b_end = b_start + blocks_per_part;
    if (b_end > blocks_per_row)
        b_end = blocks_per_row;

    const int8_t* route_a =
        A_int8 + static_cast<size_t>(route) * static_cast<size_t>(K);
    const float* route_scales =
        scales_A_blockwise +
        static_cast<size_t>(route) * static_cast<size_t>(blocks_per_row);
    partials[partial_index] = b_start < b_end
        ? moe_native_quantised_dot_desc_range<FMT>(
              desc, n, route_a, route_scales, N, K, b_start, b_end)
        : 0.0f;
}

template <bool RUNTIME_TABLE>
__global__ void moe_grouped_native_quantised_gate_up_descriptor_kpart_kernel(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* __restrict__ gate_partials,
    float* __restrict__ up_partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31,
    int maximum_k_partitions)
{
    moeGateUpOriginalPartialTask<RUNTIME_TABLE>(
        A_int8, scales_A_blockwise, gate_descs, up_descs, runtime, expert_ids, gate_partials, up_partials, num_active, N, K, num_experts, policy_partitions_0_7, policy_partitions_8_15, policy_partitions_16_23, policy_partitions_24_31, maximum_k_partitions,
        blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x);
}

/**
 * @brief Reduce descriptor-local gate/up partials in each source policy's order.
 */
template <bool RUNTIME_TABLE>
__global__ void moe_grouped_native_quantised_gate_up_descriptor_reduce_kernel(
    const float* __restrict__ gate_partials,
    const float* __restrict__ up_partials,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* const* __restrict__ gate_outputs,
    float* const* __restrict__ up_outputs,
    int num_active,
    int N,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31)
{
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int route = blockIdx.y;
    if (route >= num_active || n >= N)
        return;

    const DeviceNativeQuantisedMatrixDesc* gate_desc = nullptr;
    const DeviceNativeQuantisedMatrixDesc* up_desc = nullptr;
    if (!moe_native_quantised_load_gate_up_descriptors<RUNTIME_TABLE>(
            gate_descs, up_descs, runtime, expert_ids[route], num_experts,
            gate_desc, up_desc))
    {
        gate_outputs[route][n] = 0.0f;
        up_outputs[route][n] = 0.0f;
        return;
    }

    const int gate_partitions =
        moe_native_quantised_descriptor_k_partitions(
            gate_desc,
            policy_partitions_0_7,
            policy_partitions_8_15,
            policy_partitions_16_23,
            policy_partitions_24_31);
    const int up_partitions =
        moe_native_quantised_descriptor_k_partitions(
            up_desc,
            policy_partitions_0_7,
            policy_partitions_8_15,
            policy_partitions_16_23,
            policy_partitions_24_31);
    if (gate_partitions <= 0 || up_partitions <= 0)
    {
        __builtin_trap();
        return;
    }

    // A one-partition serial launch publishes the dot directly.  Copying its
    // sole partial avoids changing a possible signed zero through `0 + value`.
    const size_t first_partial_index =
        static_cast<size_t>(route) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    float gate_sum = gate_partials[first_partial_index];
    if (gate_partitions > 1)
    {
        gate_sum = 0.0f;
        #pragma unroll 1
        for (int part = 0; part < gate_partitions; ++part)
        {
            const size_t partial_index =
                (static_cast<size_t>(part) *
                     static_cast<size_t>(num_active) +
                 static_cast<size_t>(route)) * static_cast<size_t>(N) +
                static_cast<size_t>(n);
            gate_sum = rocm_native_quantised_accumulate_rn(
                gate_sum, gate_partials[partial_index]);
        }
    }

    float up_sum = up_partials[first_partial_index];
    if (up_partitions > 1)
    {
        up_sum = 0.0f;
        #pragma unroll 1
        for (int part = 0; part < up_partitions; ++part)
        {
            const size_t partial_index =
                (static_cast<size_t>(part) *
                     static_cast<size_t>(num_active) +
                 static_cast<size_t>(route)) * static_cast<size_t>(N) +
                static_cast<size_t>(n);
            up_sum = rocm_native_quantised_accumulate_rn(
                up_sum, up_partials[partial_index]);
        }
    }

    gate_outputs[route][n] = gate_sum;
    up_outputs[route][n] = up_sum;
}

template <bool RUNTIME_TABLE>
__global__ void
moe_grouped_native_quantised_gate_up_descriptor_reduce_swiglu_q8_kernel(
    const float* __restrict__ gate_partials,
    const float* __restrict__ up_partials,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    int8_t* __restrict__ swiglu_int8,
    float* __restrict__ swiglu_scales,
    int num_active,
    int N,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31)
{
    moeGateUpOriginalFoldSwiGLUQ8Task<RUNTIME_TABLE>(
        gate_partials, up_partials, gate_descs, up_descs, runtime, expert_ids, swiglu_int8, swiglu_scales, num_active, N, num_experts, policy_partitions_0_7, policy_partitions_8_15, policy_partitions_16_23, policy_partitions_24_31,
        blockIdx.x,blockIdx.y,threadIdx.x);
}

/** @brief Original last-producer arithmetic and load policy; oracle remains independently retained. */
__global__ __launch_bounds__(64) void moe_grouped_native_quantised_gate_up_last_producer_kernel(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* __restrict__ gate_partials,
    float* __restrict__ up_partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31,
    int maximum_k_partitions,
    int8_t* swiglu_int8,float* swiglu_scales,llaminar2::NativeTileCompletionView completion)
{
    moeGateUpLastProducerBody<false>(A_int8,scales_A_blockwise,gate_descs,up_descs,runtime,expert_ids,gate_partials,up_partials,
        num_active,N,K,num_experts,policy_partitions_0_7,policy_partitions_8_15,
        policy_partitions_16_23,policy_partitions_24_31,maximum_k_partitions,swiglu_int8,swiglu_scales,completion);
}

/** @brief Same exact original64-thread completion with paired K32 activation/sum consumption. */
__global__ __launch_bounds__(64) void moe_grouped_native_quantised_gate_up_paired_k32_last_producer_kernel(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* __restrict__ gate_partials,
    float* __restrict__ up_partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31,
    int maximum_k_partitions,
    int8_t* swiglu_int8,float* swiglu_scales,llaminar2::NativeTileCompletionView completion)
{
    moeGateUpLastProducerBody<true>(A_int8,scales_A_blockwise,gate_descs,up_descs,runtime,expert_ids,gate_partials,up_partials,
        num_active,N,K,num_experts,policy_partitions_0_7,policy_partitions_8_15,
        policy_partitions_16_23,policy_partitions_24_31,maximum_k_partitions,swiglu_int8,swiglu_scales,completion);
}

template <bool RUNTIME_TABLE>
__global__ void moe_grouped_native_quantised_down_descriptor_kpart_kernel(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ down_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* __restrict__ partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31,
    int maximum_k_partitions)
{
    moeDownOriginalPartialTask<RUNTIME_TABLE>(A_int8,scales_A_blockwise,down_descs,runtime,expert_ids,partials,num_active,N,K,num_experts,policy_partitions_0_7,policy_partitions_8_15,policy_partitions_16_23,policy_partitions_24_31,maximum_k_partitions,blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x);
}

template <bool RUNTIME_TABLE>
__global__ void moe_grouped_native_quantised_down_descriptor_reduce_kernel(
    const float* __restrict__ partials,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ down_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    const float* __restrict__ route_weights,
    float* __restrict__ route_output,
    int num_active,
    int N,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31)
{
    moeDownOriginalWeightedFoldTask<RUNTIME_TABLE>(partials,down_descs,runtime,expert_ids,route_weights,route_output,num_active,N,num_experts,policy_partitions_0_7,policy_partitions_8_15,policy_partitions_16_23,policy_partitions_24_31,blockIdx.x,blockIdx.y,threadIdx.x);
}
/*
 * the source-policy grid-Z and every per-column partial/fold remain unchanged.
 * One whole-CTA barrier joins both original tasks before each release arrival.
 * The first unique publisher owns only one(tile,route); the second owns only one output tile. No CTA serializes H or
 * recomputes another producer. Invalid descriptors still publish original +0
 * and discharge both counter obligations. Both public planes survive unchanged.
 * All local lanes join before each device-scope release arrival and after each
 * unique acquire; nonfinal CTAs retire immediately. Stream order separates
 * epochs, and only the unique owner resets each counter before kernel retirement.
 */
__global__ __launch_bounds__(llaminar2::NativeDownCompletionPlan::columns_per_tile) void moe_grouped_native_quantised_down_last_producer_kernel(
    const int8_t* A,const float* scales,const DeviceNativeQuantisedMatrixDesc* descriptors,
    const int* expert_ids,const float* weights,float* partials,float* contributions,float* output,
    int routes,int N,int K,int experts,std::uint64_t p0,std::uint64_t p1,std::uint64_t p2,std::uint64_t p3,
    int maximum_partitions,llaminar2::NativeDownCompletionView completion)
{
    constexpr unsigned PhysicalColumns=llaminar2::NativeDownCompletionPlan::columns_per_tile;
    static_assert(PhysicalColumns==128);
    const unsigned tiles=static_cast<unsigned>(N)/PhysicalColumns+(static_cast<unsigned>(N)%PhysicalColumns!=0);
    const int original_tile=static_cast<int>(blockIdx.x)*2+static_cast<int>(threadIdx.x)/64;
    const int original_lane=static_cast<int>(threadIdx.x)%64;
    if(!completion.partitions.counters||!completion.routes||completion.partitions.tiles!=tiles||
        completion.partitions.routes!=static_cast<unsigned>(routes)||
        completion.partitions.contributors!=static_cast<unsigned>(maximum_partitions)||blockDim.x!=PhysicalColumns||
        blockIdx.x>=tiles||blockIdx.y>=completion.partitions.routes||blockIdx.z>=completion.partitions.contributors)
        __builtin_trap();
    moeDownOriginalPartialTask<false>(A,scales,descriptors,nullptr,expert_ids,partials,
        routes,N,K,experts,p0,p1,p2,p3,maximum_partitions,original_tile,blockIdx.y,blockIdx.z,original_lane);
    __syncthreads();
    __shared__ llaminar2::NativeTileCompletionDecision decision;
    auto* const partition_counter=completion.partitions.counters+
        static_cast<std::size_t>(blockIdx.y)*completion.partitions.tiles+blockIdx.x;
    if(threadIdx.x==0)
    {
        decision=llaminar2::native_tile_completion::arriveForPublication(partition_counter,completion.partitions.contributors);
        if(decision==llaminar2::NativeTileCompletionDecision::Invalid)__builtin_trap();
    }
    __syncthreads();
    if(decision!=llaminar2::NativeTileCompletionDecision::Publish)return;
    moeDownOriginalWeightedFoldTask<false>(partials,descriptors,nullptr,expert_ids,weights,contributions,
        routes,N,experts,p0,p1,p2,p3,original_tile,blockIdx.y,original_lane);
    __syncthreads();
    if(threadIdx.x==0)
    {
        llaminar2::native_tile_completion::reset(partition_counter);
        decision=llaminar2::native_tile_completion::arriveForPublication(
            completion.routes+blockIdx.x,completion.partitions.routes);
        if(decision==llaminar2::NativeTileCompletionDecision::Invalid)__builtin_trap();
    }
    __syncthreads();
    if(decision!=llaminar2::NativeTileCompletionDecision::Publish)return;
    const int n=original_tile*64+original_lane;
    if(n<N)
        output[n]=llaminar2::device_ordered_route_fold::sum(contributions+n,static_cast<std::size_t>(N),routes);
    __syncthreads();
    if(threadIdx.x==0)llaminar2::native_tile_completion::reset(completion.routes+blockIdx.x);
}

extern "C" {

/**
 * @brief Execute table-routed gate/up and optionally publish fused Q8 SwiGLU.
 *
 * Supplying both fused publication pointers selects the production decode
 * contract: the ordered gate/up reducer writes the exact Q8 rows consumed by
 * down projection and does not materialize the FP32 output rows.  Supplying
 * neither retains the independently executable two-step oracle.  A half-bound
 * publication is rejected so callers cannot silently read stale workspace.
 */
bool rocmMoE_grouped_gate_up_native_quantised_decode_table(
    const float* d_hidden,
    const DeviceNativeQuantisedMatrixDesc* d_gate_desc_table,
    const DeviceNativeQuantisedMatrixDesc* d_up_desc_table,
    const int* d_expert_ids,
    float* const* d_gate_outputs,
    float* const* d_up_outputs,
    int8_t* d_fused_swiglu_int8,
    float* d_fused_swiglu_scales,
    int8_t* d_hidden_int8,
    float* d_hidden_scales,
    float* d_gate_partials,
    float* d_up_partials,
    bool hidden_prequantized,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint8_t codebook_id,
    uint32_t policy_codebook_mask,
    int device_id,
    void* stream,
    const llaminar2::NativeTileCompletionView* completion)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    const bool fused_swiglu_publication =
        d_fused_swiglu_int8 != nullptr || d_fused_swiglu_scales != nullptr;
    if (!stream || (!hidden_prequantized && !d_hidden) ||
        !d_gate_desc_table || !d_up_desc_table || !d_expert_ids ||
        ((!d_gate_outputs || !d_up_outputs) && !fused_swiglu_publication) ||
        (static_cast<bool>(d_fused_swiglu_int8) !=
         static_cast<bool>(d_fused_swiglu_scales)) ||
        !d_hidden_int8 || !d_hidden_scales ||
        !d_gate_partials || !d_up_partials ||
        num_active <= 0 || N <= 0 || K <= 0 || num_experts <= 0 ||
        (K % 32) != 0)
    {
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_quantised_decode_table] invalid arguments\n");
        return false;
    }

    ROCmMoESerialM1PolicyTable policy_table{};
    if (!llaminar2::rocm::native_quantised_internal::buildROCmMoESerialM1PolicyTable(
            N, K, policy_codebook_mask, policy_table))
    {
        fprintf(stderr,
                "[rocmMoE_grouped_gate_up_native_quantised_decode_table] "
                "missing descriptor-local serial-M1 policy "
                "(codebook=%u policy_mask=0x%08x N=%d K=%d)\n",
                static_cast<unsigned>(codebook_id),
                static_cast<unsigned>(policy_codebook_mask), N, K);
        return false;
    }
    llaminar2::rocm::native_quantised_internal::recordMoEDescriptorLocalDecode(
        "gate_up", "static_table", codebook_id,
        policy_codebook_mask, policy_table);

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    (void)hipGetLastError();
    hipError_t err = hipSuccess;
    if (!hidden_prequantized)
    {
        const int blocks_per_row = K / 32;
        hipLaunchKernelGGL(moe_grouped_hidden_quantize_blockwise_kernel,
            dim3(blocks_per_row), dim3(32), 0, hip_stream,
            d_hidden, d_hidden_int8, d_hidden_scales, K);

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmMoE_grouped_gate_up_native_quantised_decode_table] hidden quantize launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }

    if(completion)
    {
        if(!fused_swiglu_publication||!completion->counters||
            completion->tiles!=(static_cast<unsigned>(N)+63u)/64u||completion->routes!=unsigned(num_active)||
            completion->contributors!=policy_table.maximum_k_partitions)
            throw std::logic_error("Last-producer gate/up changed its exact prepared counter/partition geometry");
        if(llaminar2::rocm::currentGateUpProducerLoads()==llaminar2::rocm::GateUpProducerLoads::PairedK32)
        {
        hipLaunchKernelGGL(moe_grouped_native_quantised_gate_up_paired_k32_last_producer_kernel,
            dim3(completion->tiles,num_active,policy_table.maximum_k_partitions),dim3(64),0,hip_stream,
            d_hidden_int8,d_hidden_scales,d_gate_desc_table,d_up_desc_table,
            static_cast<const DeviceMoELayerRuntimeView*>(nullptr),d_expert_ids,d_gate_partials,d_up_partials,
            num_active,N,K,num_experts,policy_table.k_partitions_0_7,policy_table.k_partitions_8_15,
            policy_table.k_partitions_16_23,policy_table.k_partitions_24_31,int(policy_table.maximum_k_partitions),
            d_fused_swiglu_int8,d_fused_swiglu_scales,*completion);
        }
        else
        {
        hipLaunchKernelGGL(moe_grouped_native_quantised_gate_up_last_producer_kernel,
            dim3(completion->tiles,num_active,policy_table.maximum_k_partitions),dim3(64),0,hip_stream,
            d_hidden_int8,d_hidden_scales,d_gate_desc_table,d_up_desc_table,
            static_cast<const DeviceMoELayerRuntimeView*>(nullptr),d_expert_ids,d_gate_partials,d_up_partials,
            num_active,N,K,num_experts,policy_table.k_partitions_0_7,policy_table.k_partitions_8_15,
            policy_table.k_partitions_16_23,policy_table.k_partitions_24_31,int(policy_table.maximum_k_partitions),
            d_fused_swiglu_int8,d_fused_swiglu_scales,*completion);
        }
        return hipGetLastError()==hipSuccess;
    }

    constexpr int TN = 64;
    dim3 output_grid((N + TN - 1) / TN, num_active);
    dim3 partial_grid(
        (N + TN - 1) / TN,
        num_active,
        policy_table.maximum_k_partitions);
    dim3 block(TN);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            moe_grouped_native_quantised_gate_up_descriptor_kpart_kernel<false>),
        partial_grid, block, 0, hip_stream,
        d_hidden_int8, d_hidden_scales,
        d_gate_desc_table, d_up_desc_table,
        static_cast<const DeviceMoELayerRuntimeView*>(nullptr),
        d_expert_ids, d_gate_partials, d_up_partials,
        num_active, N, K, num_experts,
        policy_table.k_partitions_0_7,
        policy_table.k_partitions_8_15,
        policy_table.k_partitions_16_23,
        policy_table.k_partitions_24_31,
        static_cast<int>(policy_table.maximum_k_partitions));

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_quantised_decode_table] gate/up launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    if (fused_swiglu_publication)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(moe_grouped_native_quantised_gate_up_descriptor_reduce_swiglu_q8_kernel<false>),
            output_grid,block,0,hip_stream,d_gate_partials,d_up_partials,
            d_gate_desc_table,d_up_desc_table,
            static_cast<const DeviceMoELayerRuntimeView*>(nullptr),d_expert_ids,
            d_fused_swiglu_int8,d_fused_swiglu_scales,num_active,N,num_experts,
            policy_table.k_partitions_0_7,policy_table.k_partitions_8_15,
            policy_table.k_partitions_16_23,policy_table.k_partitions_24_31);
    }
    else
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                moe_grouped_native_quantised_gate_up_descriptor_reduce_kernel<false>),
            output_grid, block, 0, hip_stream,
            d_gate_partials, d_up_partials,
            d_gate_desc_table, d_up_desc_table,
            static_cast<const DeviceMoELayerRuntimeView*>(nullptr),
            d_expert_ids, d_gate_outputs, d_up_outputs,
            num_active, N, num_experts,
            policy_table.k_partitions_0_7,
            policy_table.k_partitions_8_15,
            policy_table.k_partitions_16_23,
            policy_table.k_partitions_24_31);
    }
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_gate_up_native_quantised_decode_table] "
                "descriptor-local ordered reduce launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmMoE_grouped_gate_up_native_quantised_decode_runtime(
    const float* d_hidden,
    const void* d_runtime_layer,
    const int* d_expert_ids,
    float* const* d_gate_outputs,
    float* const* d_up_outputs,
    int8_t* d_hidden_int8,
    float* d_hidden_scales,
    float* d_gate_partials,
    float* d_up_partials,
    bool hidden_prequantized,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint8_t codebook_id,
    uint32_t policy_codebook_mask,
    int device_id,
    void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (!stream || (!hidden_prequantized && !d_hidden) ||
        !d_runtime_layer || !d_expert_ids ||
        !d_gate_outputs || !d_up_outputs || !d_hidden_int8 || !d_hidden_scales ||
        !d_gate_partials || !d_up_partials ||
        num_active <= 0 || N <= 0 || K <= 0 || num_experts <= 0 || (K % 32) != 0)
    {
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_quantised_decode_runtime] invalid arguments\n");
        return false;
    }

    ROCmMoESerialM1PolicyTable policy_table{};
    if (!llaminar2::rocm::native_quantised_internal::buildROCmMoESerialM1PolicyTable(
            N, K, policy_codebook_mask, policy_table))
    {
        fprintf(stderr,
                "[rocmMoE_grouped_gate_up_native_quantised_decode_runtime] "
                "missing descriptor-local serial-M1 policy "
                "(codebook=%u policy_mask=0x%08x N=%d K=%d)\n",
                static_cast<unsigned>(codebook_id),
                static_cast<unsigned>(policy_codebook_mask), N, K);
        return false;
    }
    llaminar2::rocm::native_quantised_internal::recordMoEDescriptorLocalDecode(
        "gate_up", "runtime_bank", codebook_id,
        policy_codebook_mask, policy_table);

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    (void)hipGetLastError();
    hipError_t err = hipSuccess;
    if (!hidden_prequantized)
    {
        const int blocks_per_row = K / 32;
        hipLaunchKernelGGL(moe_grouped_hidden_quantize_blockwise_kernel,
            dim3(blocks_per_row), dim3(32), 0, hip_stream,
            d_hidden, d_hidden_int8, d_hidden_scales, K);

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmMoE_grouped_gate_up_native_quantised_decode_runtime] hidden quantize launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }

    constexpr int TN = 64;
    dim3 output_grid((N + TN - 1) / TN, num_active);
    dim3 partial_grid(
        (N + TN - 1) / TN,
        num_active,
        policy_table.maximum_k_partitions);
    dim3 block(TN);
    const auto* runtime = static_cast<const DeviceMoELayerRuntimeView*>(d_runtime_layer);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            moe_grouped_native_quantised_gate_up_descriptor_kpart_kernel<true>),
        partial_grid, block, 0, hip_stream,
        d_hidden_int8, d_hidden_scales,
        static_cast<const DeviceNativeQuantisedMatrixDesc*>(nullptr),
        static_cast<const DeviceNativeQuantisedMatrixDesc*>(nullptr),
        runtime, d_expert_ids, d_gate_partials, d_up_partials,
        num_active, N, K, num_experts,
        policy_table.k_partitions_0_7,
        policy_table.k_partitions_8_15,
        policy_table.k_partitions_16_23,
        policy_table.k_partitions_24_31,
        static_cast<int>(policy_table.maximum_k_partitions));

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_quantised_decode_runtime] gate/up launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            moe_grouped_native_quantised_gate_up_descriptor_reduce_kernel<true>),
        output_grid, block, 0, hip_stream,
        d_gate_partials, d_up_partials,
        static_cast<const DeviceNativeQuantisedMatrixDesc*>(nullptr),
        static_cast<const DeviceNativeQuantisedMatrixDesc*>(nullptr),
        runtime, d_expert_ids, d_gate_outputs, d_up_outputs,
        num_active, N, num_experts,
        policy_table.k_partitions_0_7,
        policy_table.k_partitions_8_15,
        policy_table.k_partitions_16_23,
        policy_table.k_partitions_24_31);
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_gate_up_native_quantised_decode_runtime] "
                "descriptor-local ordered reduce launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

/**
 * @brief Publish grouped expert SwiGLU rows in serial-decode Q8 form.
 *
 * This is the sole production launcher for the fused grouped publisher used
 * by direct, descriptor-table, and device-runtime down projections. Keeping a
 * public launch boundary here also lets integration tests compare the exact
 * Q8 payload and FP32 scales against standalone serial M=1 publication before
 * any weight format or route reduction can obscure the source of a mismatch.
 *
 * @param d_gate_ptrs Device array of per-route gate-row pointers.
 * @param d_up_ptrs Device array of per-route up-row pointers.
 * @param d_expert_ids Optional device route validity array; negative routes
 *        are skipped. Pass null when every route is valid.
 * @param d_swiglu_int8 Device output payload `[num_active, K]`.
 * @param d_swiglu_scales Device output scales `[num_active, K / 32]`.
 * @param num_active Number of grouped route rows.
 * @param K Width of each gate/up row; must be divisible by 32.
 * @param device_id ROCm device ordinal.
 * @param stream Exact non-null producer stream.
 * @return True when the launch was accepted by HIP; false on invalid input or
 *         an immediate launch failure.
 */
bool rocmMoE_grouped_swiglu_quantize_decode_equivalent(
    const float* const* d_gate_ptrs,
    const float* const* d_up_ptrs,
    const int* d_expert_ids,
    int8_t* d_swiglu_int8,
    float* d_swiglu_scales,
    int num_active,
    int K,
    int device_id,
    void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (!stream || !d_gate_ptrs || !d_up_ptrs ||
        !d_swiglu_int8 || !d_swiglu_scales ||
        num_active <= 0 || K <= 0 || (K % 32) != 0)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_quantize_decode_equivalent] "
                "invalid arguments\n");
        return false;
    }

    (void)hipGetLastError();
    hipLaunchKernelGGL(
        moe_grouped_swiglu_quantize_blockwise_kernel,
        dim3(num_active), dim3(256), 0, static_cast<hipStream_t>(stream),
        d_gate_ptrs, d_up_ptrs, d_expert_ids,
        d_swiglu_int8, d_swiglu_scales, num_active, K);

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_quantize_decode_equivalent] "
                "launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmMoE_grouped_swiglu_down_native_quantised_decode(
    const float* const* d_gate_ptrs,
    const float* const* d_up_ptrs,
    const DeviceNativeQuantisedMatrixDesc* d_descs,
    const float* d_weights,
    int8_t* d_swiglu_int8,
    float* d_swiglu_scales,
    float* d_ordered_down_partials,
    float* d_output,
    int num_active,
    int N,
    int K,
    uint8_t codebook_id,
    int device_id,
    void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (!stream || !d_gate_ptrs || !d_up_ptrs || !d_descs || !d_weights ||
        !d_swiglu_int8 || !d_swiglu_scales ||
        !d_ordered_down_partials || !d_output ||
        num_active <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_quantised_decode] invalid arguments\n");
        return false;
    }

    const int k_partitions =
        llaminar2::rocm::native_quantised_internal::resolveSerialM1KPartitions(codebook_id, N, K);
    if (k_partitions <= 0)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_quantised_decode] "
                "missing serial-M1 dispatch policy (codebook=%u N=%d K=%d)\n",
                static_cast<unsigned>(codebook_id), N, K);
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    if (!rocmMoE_grouped_swiglu_quantize_decode_equivalent(
            d_gate_ptrs, d_up_ptrs, nullptr,
            d_swiglu_int8, d_swiglu_scales,
            num_active, K, device_id, stream))
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_quantised_decode] "
                "grouped Q8 publication failed\n");
        return false;
    }
    hipError_t err = hipSuccess;

    constexpr int TN = 64;
    dim3 route_grid((N + TN - 1) / TN, num_active);
    dim3 partial_grid((N + TN - 1) / TN, num_active, k_partitions);
    dim3 block(TN);

    #define LAUNCH_GROUPED_DOWN_DIRECT(FMT_TAG)                              \
        do                                                                    \
        {                                                                     \
            if (k_partitions == 1)                                            \
            {                                                                 \
                hipLaunchKernelGGL(                                           \
                    HIP_KERNEL_NAME(                                          \
                        moe_grouped_native_quantised_down_route_kernel_t<          \
                            FMT_TAG, false>),                                 \
                    route_grid, block, 0, hip_stream,                         \
                    d_swiglu_int8, d_swiglu_scales, d_descs, nullptr,         \
                    d_weights, d_ordered_down_partials, num_active, N, K);    \
            }                                                                 \
            else                                                              \
            {                                                                 \
                hipLaunchKernelGGL(                                           \
                    HIP_KERNEL_NAME(                                          \
                        moe_grouped_native_quantised_down_kpart_kernel_t<          \
                            FMT_TAG, false>),                                 \
                    partial_grid, block, 0, hip_stream,                       \
                    d_swiglu_int8, d_swiglu_scales, d_descs, nullptr,         \
                    d_ordered_down_partials, num_active, N, K, k_partitions); \
            }                                                                 \
        } while (false)

    switch (static_cast<NativeQuantisedFormat>(codebook_id))
    {
    case NATIVE_QUANTISED_Q4_0:     LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_Q4_0);     break;
    case NATIVE_QUANTISED_IQ4_NL:   LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_IQ4_NL);   break;
    case NATIVE_QUANTISED_IQ4_XS_SOURCE:   LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_IQ4_XS_SOURCE);   break;
    case NATIVE_QUANTISED_Q4_1:     LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_Q4_1);     break;
    case NATIVE_QUANTISED_Q5_0:     LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_Q5_0);     break;
    case NATIVE_QUANTISED_Q5_1:     LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_Q5_1);     break;
    case NATIVE_QUANTISED_Q6_K:     LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_Q6_K);     break;
    case NATIVE_QUANTISED_Q3_K:     LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_Q3_K);     break;
    case NATIVE_QUANTISED_Q2_K:     LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_Q2_K);     break;
    case NATIVE_QUANTISED_IQ3_S:    LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_IQ3_S);    break;
    case NATIVE_QUANTISED_IQ3_XXS:  LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_IQ3_XXS);  break;
    case NATIVE_QUANTISED_IQ2_S:    LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_IQ2_S);    break;
    case NATIVE_QUANTISED_IQ2_XS:   LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_IQ2_XS);   break;
    case NATIVE_QUANTISED_IQ2_XXS:  LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_IQ2_XXS);  break;
    case NATIVE_QUANTISED_IQ1_S:    LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_IQ1_S);    break;
    case NATIVE_QUANTISED_IQ1_M:    LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_IQ1_M);    break;
    case NATIVE_QUANTISED_Q8_0:     LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_Q8_0);     break;
    case NATIVE_QUANTISED_EXPANDED_INT8_MIN:
        LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_EXPANDED_INT8_MIN);
        break;
    default:
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_quantised_decode] unsupported codebook_id=%u\n",
                static_cast<unsigned>(codebook_id));
        return false;
    }

    #undef LAUNCH_GROUPED_DOWN_DIRECT

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_quantised_decode] down launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    if (k_partitions > 1)
    {
        hipLaunchKernelGGL(
            moe_grouped_native_quantised_down_kpart_reduce_kernel,
            route_grid, block, 0, hip_stream,
            d_ordered_down_partials, d_weights,
            d_ordered_down_partials,
            num_active, N, k_partitions);
        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmMoE_grouped_swiglu_down_native_quantised_decode] "
                    "K-partition reduce launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }

    hipLaunchKernelGGL(
        moe_grouped_native_quantised_down_ordered_reduce_kernel,
        dim3((N + TN - 1) / TN), block, 0, hip_stream,
        d_ordered_down_partials, d_output, num_active, N);
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_quantised_decode] "
                "route reduce launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

/**
 * @brief Execute table-routed down projection from exact Q8 SwiGLU rows.
 *
 * When @p swiglu_prequantized is true, @p d_swiglu_int8 and
 * @p d_swiglu_scales are the publication produced by the fused gate/up
 * reducer, and the FP32 gate/up pointer arrays are not read.  The false form
 * remains the split-path oracle and performs the ordinary SwiGLU/Q8 kernel
 * before down projection.
 */
bool rocmMoE_grouped_swiglu_down_native_quantised_decode_table(
    const float* const* d_gate_ptrs,
    const float* const* d_up_ptrs,
    const DeviceNativeQuantisedMatrixDesc* d_desc_table,
    const int* d_expert_ids,
    const float* d_weights,
    int8_t* d_swiglu_int8,
    float* d_swiglu_scales,
    bool swiglu_prequantized,
    float* d_output,
    float* d_canonical_route_contributions,
    float* d_ordered_down_partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint8_t codebook_id,
    uint32_t policy_codebook_mask,
    int device_id,
    void* stream,
    const llaminar2::NativeDownCompletionView* completion)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    const bool canonical_publication =
        d_canonical_route_contributions != nullptr;
    if (!stream ||
        ((!d_gate_ptrs || !d_up_ptrs) && !swiglu_prequantized) ||
        !d_desc_table ||
        !d_expert_ids || (!d_weights && (!canonical_publication || completion)) ||
        !d_swiglu_int8 || !d_swiglu_scales ||
        (completion ? (!d_output || !canonical_publication) : (static_cast<bool>(d_output) == canonical_publication)) ||
        !d_ordered_down_partials ||
        num_active <= 0 || N <= 0 || K <= 0 || num_experts <= 0 ||
        (K % 32) != 0)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_quantised_decode_table] invalid arguments\n");
        return false;
    }

    ROCmMoESerialM1PolicyTable policy_table{};
    if (!llaminar2::rocm::native_quantised_internal::buildROCmMoESerialM1PolicyTable(
            N, K, policy_codebook_mask, policy_table))
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_quantised_decode_table] "
                "missing descriptor-local serial-M1 policy "
                "(codebook=%u policy_mask=0x%08x N=%d K=%d)\n",
                static_cast<unsigned>(codebook_id),
                static_cast<unsigned>(policy_codebook_mask), N, K);
        return false;
    }
    llaminar2::rocm::native_quantised_internal::recordMoEDescriptorLocalDecode(
        "down", "static_table", codebook_id,
        policy_codebook_mask, policy_table);

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    if (!swiglu_prequantized &&
        !rocmMoE_grouped_swiglu_quantize_decode_equivalent(
            d_gate_ptrs, d_up_ptrs, d_expert_ids,
            d_swiglu_int8, d_swiglu_scales,
            num_active, K, device_id, stream))
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_quantised_decode_table] "
                "grouped Q8 publication failed\n");
        return false;
    }
    hipError_t err = hipSuccess;

    if(completion)
    {
        const auto expected=llaminar2::NativeDownCompletionPlan::make(N,num_active,policy_table.maximum_k_partitions);
        if(!completion->partitions.counters||!completion->routes||
            completion->partitions.tiles!=expected.partitions().tiles()||
            completion->partitions.routes!=expected.partitions().routes()||
            completion->partitions.contributors!=expected.partitions().contributors()||
            completion->routes!=completion->partitions.counters+expected.partitions().counters())
            return false;
        hipLaunchKernelGGL(moe_grouped_native_quantised_down_last_producer_kernel,
            dim3(expected.partitions().tiles(),num_active,policy_table.maximum_k_partitions),
            dim3(llaminar2::NativeDownCompletionPlan::columns_per_tile),0,hip_stream,
            d_swiglu_int8,d_swiglu_scales,d_desc_table,d_expert_ids,d_weights,d_ordered_down_partials,
            d_canonical_route_contributions,d_output,num_active,N,K,num_experts,
            policy_table.k_partitions_0_7,policy_table.k_partitions_8_15,
            policy_table.k_partitions_16_23,policy_table.k_partitions_24_31,
            static_cast<int>(policy_table.maximum_k_partitions),*completion);
        return hipGetLastError()==hipSuccess;
    }

    constexpr int TN = 64;
    float* const route_contributions = canonical_publication
                                           ? d_canonical_route_contributions
                                           : d_ordered_down_partials;
    dim3 block(TN);
    const dim3 partial_grid((N+TN-1)/TN,num_active,policy_table.maximum_k_partitions);
    const dim3 route_grid((N+TN-1)/TN,num_active);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(moe_grouped_native_quantised_down_descriptor_kpart_kernel<false>),
        partial_grid,block,0,hip_stream,d_swiglu_int8,d_swiglu_scales,d_desc_table,
        static_cast<const DeviceMoELayerRuntimeView*>(nullptr),d_expert_ids,
        d_ordered_down_partials,num_active,N,K,num_experts,
        policy_table.k_partitions_0_7,policy_table.k_partitions_8_15,
        policy_table.k_partitions_16_23,policy_table.k_partitions_24_31,
        static_cast<int>(policy_table.maximum_k_partitions));
    if (hipGetLastError() != hipSuccess)
        return false;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(moe_grouped_native_quantised_down_descriptor_reduce_kernel<false>),
        route_grid,block,0,hip_stream,d_ordered_down_partials,d_desc_table,
        static_cast<const DeviceMoELayerRuntimeView*>(nullptr),d_expert_ids,d_weights,
        route_contributions,num_active,N,num_experts,
        policy_table.k_partitions_0_7,policy_table.k_partitions_8_15,
        policy_table.k_partitions_16_23,policy_table.k_partitions_24_31);
    if (hipGetLastError() != hipSuccess)
        return false;

    if (!canonical_publication)
    {
        hipLaunchKernelGGL(
            moe_grouped_native_quantised_down_ordered_reduce_kernel,
            dim3((N + TN - 1) / TN), block, 0,
            hip_stream,
            route_contributions, d_output, num_active, N);
        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_quantised_decode_table] ordered reduce launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }
    return true;
}

bool rocmMoE_grouped_swiglu_down_native_quantised_decode_runtime(
    const float* const* d_gate_ptrs,
    const float* const* d_up_ptrs,
    const void* d_runtime_layer,
    const int* d_expert_ids,
    const float* d_weights,
    int8_t* d_swiglu_int8,
    float* d_swiglu_scales,
    float* d_output,
    float* d_ordered_down_partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint8_t codebook_id,
    uint32_t policy_codebook_mask,
    int device_id,
    void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (!stream || !d_gate_ptrs || !d_up_ptrs || !d_runtime_layer ||
        !d_expert_ids || !d_weights || !d_swiglu_int8 || !d_swiglu_scales ||
        !d_output || !d_ordered_down_partials ||
        num_active <= 0 || N <= 0 || K <= 0 || num_experts <= 0 || (K % 32) != 0)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_quantised_decode_runtime] invalid arguments\n");
        return false;
    }

    ROCmMoESerialM1PolicyTable policy_table{};
    if (!llaminar2::rocm::native_quantised_internal::buildROCmMoESerialM1PolicyTable(
            N, K, policy_codebook_mask, policy_table))
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_quantised_decode_runtime] "
                "missing descriptor-local serial-M1 policy "
                "(codebook=%u policy_mask=0x%08x N=%d K=%d)\n",
                static_cast<unsigned>(codebook_id),
                static_cast<unsigned>(policy_codebook_mask), N, K);
        return false;
    }
    llaminar2::rocm::native_quantised_internal::recordMoEDescriptorLocalDecode(
        "down", "runtime_bank", codebook_id,
        policy_codebook_mask, policy_table);

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    if (!rocmMoE_grouped_swiglu_quantize_decode_equivalent(
            d_gate_ptrs, d_up_ptrs, d_expert_ids,
            d_swiglu_int8, d_swiglu_scales,
            num_active, K, device_id, stream))
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_quantised_decode_runtime] "
                "grouped Q8 publication failed\n");
        return false;
    }
    hipError_t err = hipSuccess;

    constexpr int TN = 64;
    dim3 route_grid((N + TN - 1) / TN, num_active);
    dim3 partial_grid(
        (N + TN - 1) / TN,
        num_active,
        policy_table.maximum_k_partitions);
    dim3 block(TN);
    const auto* runtime = static_cast<const DeviceMoELayerRuntimeView*>(d_runtime_layer);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            moe_grouped_native_quantised_down_descriptor_kpart_kernel<true>),
        partial_grid, block, 0, hip_stream,
        d_swiglu_int8, d_swiglu_scales,
        static_cast<const DeviceNativeQuantisedMatrixDesc*>(nullptr),
        runtime, d_expert_ids, d_ordered_down_partials,
        num_active, N, K, num_experts,
        policy_table.k_partitions_0_7,
        policy_table.k_partitions_8_15,
        policy_table.k_partitions_16_23,
        policy_table.k_partitions_24_31,
        static_cast<int>(policy_table.maximum_k_partitions));

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_quantised_decode_runtime] route launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            moe_grouped_native_quantised_down_descriptor_reduce_kernel<true>),
        route_grid, block, 0, hip_stream,
        d_ordered_down_partials,
        static_cast<const DeviceNativeQuantisedMatrixDesc*>(nullptr),
        runtime, d_expert_ids, d_weights, d_ordered_down_partials,
        num_active, N, num_experts,
        policy_table.k_partitions_0_7,
        policy_table.k_partitions_8_15,
        policy_table.k_partitions_16_23,
        policy_table.k_partitions_24_31);
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_quantised_decode_runtime] "
                "descriptor-local K-partition reduce launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(
        moe_grouped_native_quantised_down_ordered_reduce_kernel,
        dim3((N + TN - 1) / TN), block, 0, hip_stream,
        d_ordered_down_partials, d_output, num_active, N);
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_quantised_decode_runtime] ordered reduce launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C"

namespace llaminar2::rocm
{
/** @copydoc downLastProducerPlan */
NativeDownCompletionPlan downLastProducerPlan(int columns,int reduction,int routes,std::uint32_t policy_mask)
{
    ROCmMoESerialM1PolicyTable policy{};
    if(columns<=0||reduction<=0||routes<=0||columns%32||reduction%32||
        !llaminar2::rocm::native_quantised_internal::buildROCmMoESerialM1PolicyTable(columns,reduction,policy_mask,policy))
        throw std::invalid_argument("Last-producer down has no original serial-M1 partition policy");
    return NativeDownCompletionPlan::make(columns,routes,policy.maximum_k_partitions);
}
/** @copydoc downLastProducerKernelFunction */
const void* downLastProducerKernelFunction() noexcept
{return reinterpret_cast<const void*>(moe_grouped_native_quantised_down_last_producer_kernel);}

/** @copydoc gateUpLastProducerPlan */
NativeTileCompletionPlan gateUpLastProducerPlan(int columns,int reduction,int routes,std::uint32_t policy_mask)
{
    ROCmMoESerialM1PolicyTable original{};
    if(columns<=0||reduction<=0||routes<=0||columns%32||reduction%32||
        !llaminar2::rocm::native_quantised_internal::buildROCmMoESerialM1PolicyTable(columns,reduction,policy_mask,original))
        throw std::invalid_argument("Last-producer gate/up requires its complete original M1 source policy");
    return NativeTileCompletionPlan::make(columns,routes,original.maximum_k_partitions);
}
/** @return Exact original-grid last-producer program identity, without device resource queries. */
const void* gateUpLastProducerKernelFunction() noexcept
{
    return currentGateUpProducerLoads()==GateUpProducerLoads::PairedK32
        ?reinterpret_cast<const void*>(&moe_grouped_native_quantised_gate_up_paired_k32_last_producer_kernel)
        :reinterpret_cast<const void*>(&moe_grouped_native_quantised_gate_up_last_producer_kernel);
}
/** @copydoc gateUpLastProducerKernelFunction(GateUpProducerLoads) */
const void* gateUpLastProducerKernelFunction(GateUpProducerLoads loads)
{
    if(loads==GateUpProducerLoads::Original)
        return reinterpret_cast<const void*>(&moe_grouped_native_quantised_gate_up_last_producer_kernel);
    if(loads==GateUpProducerLoads::PairedK32)
        return reinterpret_cast<const void*>(&moe_grouped_native_quantised_gate_up_paired_k32_last_producer_kernel);
    throw std::invalid_argument("Unknown gate/up producer loads for native function identity");
}
}
