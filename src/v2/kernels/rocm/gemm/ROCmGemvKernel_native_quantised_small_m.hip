/**
 * @file ROCmGemvKernel_native_quantised_small_m.hip
 * @brief ROCm NativeQuantised small-M and batched grouped verifier GEMV kernels.
 */
#include "ROCmNativeQuantisedMoEDeviceHelpers.hip.inc"

LLAMINAR_DEFINE_ROCM_IQ_GRID_INIT(small_m)


/**
 * @brief Publishes one runtime verifier row from ordered split-K partials.
 *
 * Grid Y owns a bounded row worker, not an allocation-capacity row. A counted
 * replay visits only live rows by grid stride. The reducer's register state
 * and ascending-KB addition order do not depend on speculative capacity; this
 * is serial M=1's reduction applied to each row's `[KB][M][N]` partial slice.
 */
template <int TILE_N, int CPT, bool HasWork, bool Fold>
__global__ void gemv_native_quantised_reduce_small_m_kernel_t(
    const float* __restrict__ d_partial_fp32,    // [KB x M x N] partial buffer
    float* __restrict__ d_C_fp32,                // [M x N] output
    int M, int N, int kblocks, llaminar2::DeviceRowRange rows, llaminar2::ProjectionRowWork work,
    llaminar2::rocm::ROCmFP32DestinationFold destination_fold)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    // The count is borrowed once. Physical M remains the split-K plane stride
    // after a large-to-small replay; no inactive partial is read or published.
    auto rowWork=llaminar2::projectionCursorFor<HasWork>(work,rows,int(blockIdx.y),int(gridDim.y),1);
    int row,active_m;
    while(rowWork.next(row,active_m))
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            if (n >= N)
                continue;

            float sum = 0.0f;
            #pragma unroll 16
            for (int kb = 0; kb < kblocks; ++kb)
            {
                const size_t idx =
                    (static_cast<size_t>(kb) * static_cast<size_t>(M) +
                     static_cast<size_t>(row)) * static_cast<size_t>(N) +
                    static_cast<size_t>(n);
                sum = rocm_native_quantised_accumulate_rn(
                    sum, d_partial_fp32[idx]);
            }
            const auto index = static_cast<size_t>(row) * N + n;
            if constexpr (Fold)
                sum = destination_fold.applyUpdate(sum, d_C_fp32, index, n);
            d_C_fp32[index] = sum;
        }
    }
}

namespace
{
constexpr int kHCDownColumns = 320;
constexpr int kHCDownReduction = 10240;
constexpr int kHCDownPartitions = 16;
constexpr int kHCDownPackedBlocks = kHCDownReduction / 32;
constexpr int kHCDownWindow = kHCDownPackedBlocks / kHCDownPartitions;
constexpr int kHCDownTileRows = 4;
constexpr int kHCDownTileColumns = 32;

/** @brief Existing counted row owner and persistent prepared GEMV storage. */
struct HCDownLongKArgs
{
    const int8_t *activation;
    const uint8_t *payload;
    const uint16_t *weight_scales;
    const float *activation_scales;
    float *partials;
    llaminar2::DeviceRowRange rows;
};

/**
 * @brief Produce the original FP32 K32 × 20 terms for one [partition,row,column] tile.
 *
 * Every lane decodes one physical Q8 block, reuses it across the live row
 * tile and publishes one typed single-scale term per live row. Four lanes
 * fold those terms in increasing original block order. The unchanged public
 * reducer consumes ascending partitions with the existing physical row stride.
 */
__global__ __launch_bounds__(kHCDownTileColumns * kHCDownWindow)
void qfn_hc_down_long_k_contributions(HCDownLongKArgs args)
{
    const int row_base = int(blockIdx.z) * kHCDownTileRows;
    const int width = min(kHCDownTileRows, args.rows.activeRows() - row_base);
    if (width <= 0)
        return;

    const int thread = int(threadIdx.x);
    const int local_group = thread / kHCDownTileColumns;
    const int local_col = thread % kHCDownTileColumns;
    const int group = int(blockIdx.y) * kHCDownWindow + local_group;
    const int col = int(blockIdx.x) * kHCDownTileColumns + local_col;
    const int weight_index = group * kHCDownColumns + col;
    const auto *packed = reinterpret_cast<const int4 *>(args.payload + weight_index * 32);
    const int4 lo = packed[0], hi = packed[1];
    const int weight[] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    const float scale = __half2float(__ushort_as_half(args.weight_scales[weight_index]));
    __shared__ float terms[kHCDownTileRows][kHCDownWindow][kHCDownTileColumns];
#pragma unroll
    for (int row = 0; row < kHCDownTileRows; ++row)
    {
        if (row >= width)
            break;
        const int input_index = (row_base + row) * kHCDownPackedBlocks + group;
        const auto *input = reinterpret_cast<const int *>(args.activation + input_index * 32);
        int dot = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            dot = llaminar2::rocm_integer_contract::signedDot4(input[i], weight[i], dot);
        terms[row][local_group][local_col] =
            llaminar2::device_native_quantised_contract::singleScaleBlock(
                dot, scale, args.activation_scales[input_index]);
    }
    __syncthreads();

    // Each live scratch word has one writer. Inactive words remain dirty.
    if (local_group < width)
    {
        float sum = 0.0f;
#pragma unroll
        for (int group_offset = 0; group_offset < kHCDownWindow; ++group_offset)
            sum = llaminar2::device_native_quantised_contract::accumulate(
                sum, terms[local_group][group_offset][local_col]);
        const auto plane = int(blockIdx.y) * args.rows.physicalRows() + row_base + local_group;
        args.partials[plane * kHCDownColumns + col] = sum;
    }
}

/** @brief Preserve all serial-rounded terms and publish original SiLU/Q8 planes locally. */
__global__ void groupedHCDownReducePublish(
    const float* partials, float* raw, llaminar2::GPUHCDownOutputView publication)
{
    const int column = int(blockIdx.x) * 64 + int(threadIdx.x);
    const int row = int(blockIdx.y);
    if (row >= publication.rows.activeRows())
        return;
    const int offset = row * kHCDownColumns + column;
    float sum = 0.0f;
#pragma unroll
    for (int partition = 0; partition < kHCDownPartitions; ++partition)
        sum = llaminar2::device_native_quantised_contract::accumulate(
            sum, partials[partition * publication.rows.physicalRows() * kHCDownColumns + offset]);
    raw[offset] = sum;
    llaminar2::hc_down_publication::publish(sum, publication, offset);
}

} // namespace

constexpr int NATIVE_QUANTISED_SMALL_M_GRAPH_SAFE_KB_CAP =
    llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions;
constexpr int NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE = 2;
constexpr int NATIVE_QUANTISED_VERIFIER_WIDE_ROWS_PER_TILE = 4;

/**
 * @brief Select the measured verifier-row reuse geometry for a runtime batch.
 *
 * M=2 retains the two-row specialization because a four-row accumulator adds
 * idle predicates without any extra weight reuse.  M>=3 uses four rows: this
 * matches the common four-row transaction and amortizes each packed-weight
 * decode across its live rows. The launch family owns how these fixed tiles
 * are scheduled; counted fused bundles use a bounded grid with live iteration.
 * This arithmetic reuse geometry is total for every positive runtime M.
 *
 * @param m Number of grouped verifier rows in the launch.
 * @return Compile-time kernel geometry selected by the host launch wrapper.
 */
constexpr int native_quantised_verifier_rows_per_tile(int m)
{
    return m <= NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE
        ? NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE
        : NATIVE_QUANTISED_VERIFIER_WIDE_ROWS_PER_TILE;
}

/**
 * @brief Resolve the projection owning one captured column tile.
 * @param args Immutable ordered column intervals and borrowed operand planes.
 * @param global_tile Workgroup X coordinate in the concatenated bundle grid.
 * @return Uniform projection index; launch validation guarantees membership.
 */
__device__ __forceinline__ int native_quantised_batched_projection_for_tile(
    const NativeQuantisedSmallMBatchedArgs& args,
    int global_tile)
{
    int proj = 0;
    #pragma unroll
    for (int i = 1; i < NATIVE_QUANTISED_SMALL_M_MAX_BATCH; ++i)
    {
        if (i < args.num_projections && global_tile >= args.grid_offsets[i])
            proj = i;
    }
    return proj;
}


/**
 * @brief One resolved projection, borrowed by every physical decoder arm.
 *
 * Resolve the bundle before switching codebooks so the eight-projection
 * pointer tables need not stay live through each inlined decoder. The partial
 * pointer already names this CTA's physical K plane, never a live-row stride.
 */
struct NativeQuantisedSmallMProjection
{
    unsigned n; ///< Validated positive physical output width.
    unsigned column_base; ///< This lane's nonnegative output column, including tail lanes.
    const uint8_t* payload; ///< Borrowed packed weight blocks.
    const uint16_t* scales; ///< Primary scale plane.
    const uint16_t* mins; ///< Optional secondary/minimum plane.
    const uint32_t* emins; ///< Optional extended minima.
    const float* bias; ///< Optional output bias.
    float* output; ///< This physical row tile's output.
    float* partial; ///< This CTA's split-K plane, or null for direct output.
};

/** @brief Capture-time publication contract; producers never need both destinations. */
enum class NativeQuantisedVerifierPublication : uint8_t { DirectOutput, OrderedPartial };

/**
 * @brief Resolve one lane's projection before entering a format decoder.
 * @param args Captured bundle inventory; no device allocation is performed.
 * @param projection Bundle index selected uniformly by this workgroup.
 * @param physical_m Physical scratch stride, independent of live rows.
 * @return Borrowed operands with the physical split-K plane already selected.
 */
template <int TILE_N, int CPT, NativeQuantisedVerifierPublication Publication>
__device__ __forceinline__ NativeQuantisedSmallMProjection native_quantised_resolve_small_m_projection(
    const NativeQuantisedSmallMBatchedArgs& args, int projection,
    int physical_m)
{
    const auto n = static_cast<unsigned>(args.N[projection]);
    const auto local_tile = blockIdx.x - static_cast<unsigned>(args.grid_offsets[projection]);
    float* partial = nullptr;
    float* output = nullptr;
    const float* bias = nullptr;
    if constexpr (Publication == NativeQuantisedVerifierPublication::OrderedPartial)
        partial = args.partial[projection] + static_cast<size_t>(blockIdx.y) * physical_m * n;
    else
    {
        output = args.output[projection];
        bias = args.bias[projection];
    }
    return {n, local_tile * TILE_N + threadIdx.x * CPT,
            args.payload[projection], args.scales[projection],
            args.mins[projection], args.emins[projection],
            bias, output, partial};
}

// Shared small-M decode helpers live in ROCmNativeQuantisedDecodeCommon.hip.inc
template <NativeQuantisedFormat FMT>
__device__ __forceinline__ void native_quantised_small_m_accumulate_row(
    float& acc,
    const int8_t* __restrict__ d_A_int8,
    const float* __restrict__ d_scale_A_blockwise,
    const int32_t* __restrict__ d_sum_A_blockwise,
    unsigned row,
    unsigned K,
    unsigned blocks_per_row,
    unsigned b,
    const int32_t* __restrict__ packed_groups,
    const uint8_t* __restrict__ payload,
    const uint16_t* __restrict__ scale_base,
    const uint16_t* __restrict__ min_base,
    const uint32_t* __restrict__ emin_base,
    size_t linear)
{
    const int32_t* a4 = reinterpret_cast<const int32_t*>(
        d_A_int8 + static_cast<size_t>(row) * K + static_cast<size_t>(b) * 32);
    const float scale_ab =
        d_scale_A_blockwise[static_cast<size_t>(row) * blocks_per_row + b];
    int32_t sum_a = 0;
    if constexpr (NATIVE_QUANTISEDTraits<FMT>::is_asymmetric)
    {
        sum_a = d_sum_A_blockwise
            ? d_sum_A_blockwise[static_cast<size_t>(row) * blocks_per_row + b]
            : native_quantised_activation_sum_a<FMT>(a4);
    }
    native_quantised_small_m_accumulate_decoded_block<FMT>(
        acc, a4, packed_groups, payload, scale_base, min_base, emin_base,
        linear, scale_ab, sum_a);
}


/**
 * @brief Decode-equivalent batched small-M GEMV body that shares weight decode across rows.
 *
 * The row-parallel verifier kernel assigns one workgroup row to each verifier
 * row, which preserves the serial decode order but decodes the same packed
 * weight block once per row.  For MTP verifier economics that is the wrong
 * tradeoff on wide projections: dequantization is a material part of the
 * cost, and grouped verifier execution should amortize it across the complete
 * four-row production transaction.
 *
 * This body keeps the same per-row FP32 accumulation order as serial decode:
 * each row sees blocks in increasing `b` order, and split-K partials are
 * reduced by the existing reducer.  The only difference is that each output
 * lane decodes one weight block and then accumulates all verifier rows before
 * moving to the next block.
 */
template <int TILE_N, int CPT, int ROWS_PER_TILE, NativeQuantisedFormat FMT,
          NativeQuantisedVerifierPublication Publication>
__device__ __forceinline__ void gemv_native_quantised_small_m_batched_shared_tile(
    const int8_t* __restrict__ d_A_int8,                // [M x K]
    const float* __restrict__ d_scale_A_blockwise,      // [M x blocks_per_row]
    const int32_t* __restrict__ d_sum_A_blockwise,      // Required [M x blocks_per_row]
    NativeQuantisedSmallMProjection projection,
    unsigned K,
    unsigned partition_blocks,
    unsigned active_m,
    unsigned row_base,
    const float* resolved_bias)
{
    static_assert(CPT == 1 || CPT == 2, "native-quantised small-M supports CPT in {1,2}");
    static_assert(
        ROWS_PER_TILE == NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE ||
            ROWS_PER_TILE == NATIVE_QUANTISED_VERIFIER_WIDE_ROWS_PER_TILE,
        "native-quantised verifier row tile must be one of the measured geometries");
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
                  "small-M native-quantised requires a supported native-quantised format");

    const unsigned N = projection.n;
    const unsigned n_base = projection.column_base;
    const unsigned kb = blockIdx.y;
    // Public launch validation admits only positive N/K and partition widths.
    // Preserve that contract through inlining so address products do not keep
    // unnecessary signed high halves live in the scalar register file.
    __builtin_assume(N > 0);
    __builtin_assume(K > 0);
    __builtin_assume(partition_blocks > 0);
    // Fused callers publish activation sums with the shared quantized input.
    // The public boundary rejects a missing plane; no decoder may reconstruct
    // another sum authority inside its row loop.
    __builtin_assume(d_sum_A_blockwise != nullptr);
    const unsigned blocks_per_row = K / 32;
    const unsigned b_per = partition_blocks;
    const unsigned b_start = kb * b_per;
    const unsigned b_end = min(b_start + b_per, blocks_per_row);

    const uint8_t* payload_base = projection.payload;
    const uint16_t* scale_base = projection.scales;
    const uint16_t* min_base = projection.mins;
    const uint32_t* emin_base = projection.emins;
    float* out_base = projection.output;
    float* partial_base = projection.partial;

    float acc[ROWS_PER_TILE][CPT];
    #pragma unroll
    for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
            acc[tile_row][c] = 0.0f;
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        // Tail lanes borrow the final valid column for read-only arithmetic.
        // Mask only publication below: keeping an EXEC mask through the whole
        // row/decoder loop extends scalar lifetimes and causes register spills.
        // Valid columns retain their exact addresses and operation order.
        const unsigned n = min(n_base + c, N - 1);
        for (unsigned b = b_start; b < b_end; ++b)
        {
            int32_t packed_groups[8];
            const size_t linear =
                static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
            const uint8_t* payload = payload_base + linear * NATIVE_QUANTISEDTraits<FMT>::payload_bytes;
            moe_decode_native_quantised_groups<FMT>(payload, packed_groups);

            #pragma unroll
            for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
            {
                const unsigned row = row_base + tile_row;
                if (row >= active_m)
                    continue;
                native_quantised_small_m_accumulate_row<FMT>(
                    acc[tile_row][c], d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise,
                    row, K, blocks_per_row, b, packed_groups, payload, scale_base,
                    min_base, emin_base, linear);
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const unsigned n = n_base + c;
        if (n >= N)
            continue;

        #pragma unroll
        for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
        {
            const unsigned row = row_base + tile_row;
            if (row >= active_m)
                continue;
            const size_t out_idx =
                static_cast<size_t>(row) * static_cast<size_t>(N) + static_cast<size_t>(n);
            if constexpr (Publication == NativeQuantisedVerifierPublication::DirectOutput)
            {
                out_base[out_idx] = acc[tile_row][c] + resolved_bias[c];
            }
            else
            {
                partial_base[out_idx] = acc[tile_row][c];
            }
        }
    }
}

/**
 * @brief Visit live row tiles with a bounded grid and unchanged tile arithmetic.
 *
 * Column/split-K workers already expose parallelism for wide projections.
 * Grid stride admits additional tiles only when the device count includes
 * them, avoiding thousands of workgroups whose sole job would be to exit.
 * Each tile resets its fixed register accumulator and publishes to the same
 * physical scratch plane as the rectangular launch; no row shares a reduction.
 */
template <int TILE_N, int CPT, int ROWS_PER_TILE, NativeQuantisedFormat FMT,
          NativeQuantisedVerifierPublication Publication>
__device__ __forceinline__ void gemv_native_quantised_small_m_batched_shared_body(
    const int8_t* __restrict__ d_A_int8,
    const float* __restrict__ d_scale_A_blockwise,
    const int32_t* __restrict__ d_sum_A_blockwise,
    NativeQuantisedSmallMProjection projection, unsigned K, unsigned partition_blocks, unsigned active_m,unsigned row_first)
{
    float resolved_bias[CPT] = {};
    if constexpr (Publication == NativeQuantisedVerifierPublication::DirectOutput)
    {
        // Bias is immutable across rows. Materialize each lane's value once,
        // then retire its pointer/presence predicate before decoding weights.
        // Tail lanes read a valid column but never publish their output.
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
            if (projection.bias)
                resolved_bias[c] = projection.bias[min(projection.column_base + c, projection.n - 1)];
    }
    projection.bias = nullptr;
    // A runtime traversal must not speculate extra register-resident row tiles.
    const unsigned row_base = row_first;
    {
        gemv_native_quantised_small_m_batched_shared_tile<TILE_N, CPT, ROWS_PER_TILE, FMT, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise,
            projection, K, partition_blocks, active_m, row_base, resolved_bias);
    }
}

template <int TILE_N, int CPT, int ROWS_PER_TILE, NativeQuantisedFormat FMT,
          NativeQuantisedVerifierPublication Publication, bool HasWork = false>
__global__ __launch_bounds__(TILE_N / CPT) void gemv_native_quantised_small_m_batched_shared_kernel_t(
    const int8_t* __restrict__ d_A_int8,                // [M x K]
    const float* __restrict__ d_scale_A_blockwise,      // [M x blocks_per_row]
    const int32_t* __restrict__ d_sum_A_blockwise,      // [M x blocks_per_row], nullable
    NativeQuantisedSmallMBatchedArgs args,
    int K,
    int partition_blocks)
{
    // The row descriptor is also the sole physical scratch-stride authority.
    const int M = args.rows.physicalRows();
    // Resolve admission before format dispatch. The count's pointer/bounds
    // are dead before the decoder's long scalar-register live ranges begin.
    auto rowWork=llaminar2::projectionCursorFor<HasWork>(args.work,args.rows,int(blockIdx.z),int(gridDim.z),ROWS_PER_TILE);
    int row_first,active_m;
    while(rowWork.next(row_first,active_m))
    {
    const int proj = native_quantised_batched_projection_for_tile(args, blockIdx.x);
    const auto projection = native_quantised_resolve_small_m_projection<TILE_N, CPT, Publication>(
        args, proj, M);
    gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, FMT, Publication>(
        d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
    } // Only authored request tiles enter format dispatch.
}

template <int TILE_N, int CPT, int ROWS_PER_TILE, NativeQuantisedVerifierPublication Publication, bool HasWork = false>
__global__ __launch_bounds__(TILE_N / CPT) void gemv_native_quantised_small_m_batched_shared_mixed_kernel_t(
    const int8_t* __restrict__ d_A_int8,                // [M x K]
    const float* __restrict__ d_scale_A_blockwise,      // [M x blocks_per_row]
    const int32_t* __restrict__ d_sum_A_blockwise,      // [M x blocks_per_row], nullable
    NativeQuantisedSmallMBatchedArgs args,
    int K,
    int partition_blocks)
{
    // The row descriptor is also the sole physical scratch-stride authority.
    const int M = args.rows.physicalRows();
    // Resolve admission before format dispatch. The count's pointer/bounds
    // are dead before the decoder's long scalar-register live ranges begin.
    auto rowWork=llaminar2::projectionCursorFor<HasWork>(args.work,args.rows,int(blockIdx.z),int(gridDim.z),ROWS_PER_TILE);
    int row_first,active_m;
    while(rowWork.next(row_first,active_m))
    {
    const int proj = native_quantised_batched_projection_for_tile(args, blockIdx.x);
    const auto projection = native_quantised_resolve_small_m_projection<TILE_N, CPT, Publication>(
        args, proj, M);
    const auto codebook = static_cast<NativeQuantisedFormat>(
        __builtin_amdgcn_readfirstlane(static_cast<int>(args.codebook[proj])));
    switch (codebook)
    {
    case NATIVE_QUANTISED_Q4_0:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_Q4_0, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_IQ4_NL:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_IQ4_NL, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_IQ4_XS_SOURCE:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_IQ4_XS_SOURCE, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_Q4_1:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_Q4_1, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_Q5_0:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_Q5_0, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_Q5_1:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_Q5_1, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_Q6_K:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_Q6_K, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_Q3_K:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_Q3_K, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_Q2_K:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_Q2_K, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_IQ3_S:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_IQ3_S, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_IQ3_XXS:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_IQ3_XXS, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_IQ2_S:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_IQ2_S, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_IQ2_XS:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_IQ2_XS, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_IQ2_XXS:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_IQ2_XXS, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_IQ1_S:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_IQ1_S, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_IQ1_M:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_IQ1_M, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_Q8_0:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_Q8_0, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_EXPANDED_INT8_MIN:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_EXPANDED_INT8_MIN, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    default:
        break;
    }
    } // Only authored request tiles enter format dispatch.
}

/**
 * @brief Reduces one runtime row for every projection in a fused batch.
 *
 * `blockIdx.y` owns a row residue class, advancing by gridDim.y only while
 * rows are live. The fixed-width register state and ascending-KB fold preserve
 * serial-M1 arithmetic for every depth without launching idle physical rows.
 */
template <int TILE_N, int CPT, bool HasWork = false>
__global__ void gemv_native_quantised_reduce_small_m_batched_kernel_t(
    NativeQuantisedSmallMBatchedArgs args,
    int kblocks)
{
    // The row descriptor is also the sole physical scratch-stride authority.
    const int M = args.rows.physicalRows();
    const int global_tile = blockIdx.x;
    const int proj = native_quantised_batched_projection_for_tile(args, global_tile);
    const int local_tile = global_tile - args.grid_offsets[proj];
    const int N = args.N[proj];
    const int n_base = local_tile * TILE_N + threadIdx.x * CPT;

    const float* partial_base = args.partial[proj];
    const float* bias_base = args.bias[proj];
    float* out_base = args.output[proj];
    // Physical M remains the scratch stride even when fewer rows are live.
    auto rowWork=llaminar2::projectionCursorFor<HasWork>(args.work,args.rows,int(blockIdx.y),int(gridDim.y),1);
    int row,active_m;
    while(rowWork.next(row,active_m))
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            if (n >= N)
                continue;

            float sum = 0.0f;
            // Runtime rows must not multiply sixteen simultaneous address
            // live ranges; the ascending serial-M1 fold remains unchanged.
            #pragma unroll 1
            for (int kb = 0; kb < kblocks; ++kb)
            {
                sum += partial_base[(static_cast<size_t>(kb) * static_cast<size_t>(M) +
                                     static_cast<size_t>(row)) * static_cast<size_t>(N) +
                                    static_cast<size_t>(n)];
            }
            out_base[static_cast<size_t>(row) * N + n] =
                sum + (bias_base ? bias_base[n] : 0.0f);
        }
    }
}

/**
 * @brief Runtime-row NativeQuantised verifier GEMV with fixed four-row work tiles.
 *
 * A workgroup decodes each packed weight block once and applies it to at most
 * four verifier rows. Counted captures use bounded Z-grid workers that visit
 * further live tiles by grid stride, rather than scheduling idle capacity.
 * Workers never enlarge the accumulator array or alter an existing row's
 * block order. Split-K uses the fixed physical workspace stride so the reducer
 * can publish partitions in serial-M1 order on every replay.
 */
template <int TILE_N, int CPT, int ROWS_PER_TILE, NativeQuantisedFormat FMT, bool HasWork, bool Fold>
__global__ void gemv_native_quantised_small_m_kernel_t(
    const int8_t*    __restrict__ d_A_int8,       // [M x K] quantized activations
    const uint8_t*   __restrict__ d_payload,      // [blocks_per_row x N x payload_bytes]
    const uint16_t*  __restrict__ d_block_scales, // [blocks_per_row x N]
    const uint16_t*  __restrict__ d_block_mins,   // [blocks_per_row x N], nullable when unused
    const uint32_t*  __restrict__ d_block_emins,  // [blocks_per_row x N], Q2_K only
    float*           __restrict__ d_C_fp32,       // [M x N] output
    float*           __restrict__ d_partial_fp32, // [KB x M x N] partials
    const float*     __restrict__ d_scale_A_blockwise, // [M x blocks_per_row]
    const int32_t*   __restrict__ d_sum_A_blockwise,   // [M x blocks_per_row], nullable
    int M, int N, int K, int kblocks, llaminar2::DeviceRowRange rows, llaminar2::ProjectionRowWork work,
    llaminar2::rocm::ROCmFP32DestinationFold destination_fold)
{
    static_assert(CPT == 1 || CPT == 2, "native-quantised small-M supports CPT in {1,2}");
    static_assert(
        ROWS_PER_TILE == NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE ||
            ROWS_PER_TILE == NATIVE_QUANTISED_VERIFIER_WIDE_ROWS_PER_TILE,
        "native-quantised verifier row tile must be one of the measured geometries");
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
                  "small-M native-quantised requires a supported native-quantised format");

    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const int kb = blockIdx.y;
    // Geometry supplies enough column/K parallelism before admitting extra
    // row workers. The whole CTA borrows one count; padding owns no dot product.
    int active_m;
    std::uint32_t authored_rows=0;
    if constexpr(HasWork)
    {
        // The existing launcher slices the persistent workspace at this exact
        // bound; this is not a semantic M limit. Observe original publications
        // once before decoder lifetimes, then retain only derived row bits.
        static_assert(llaminar2::kDefaultNativeQuantisedVerifierRowCapacity<=32);
        __shared__ std::uint32_t published_rows;
        if(threadIdx.x==0)
        {
            if(M<=0 || M>llaminar2::kDefaultNativeQuantisedVerifierRowCapacity)
                __builtin_trap();
            std::uint32_t mask=0;
            for(int row=0;row<M;++row)
                if(work.rowIsActive(row))mask|=std::uint32_t(1)<<row;
            published_rows=mask;
        }
        // All threads consume the same derived mask before any packed weight,
        // activation or partial access. It is republished on every graph replay.
        __syncthreads();
        authored_rows=__builtin_amdgcn_readfirstlane(published_rows);
        if(!authored_rows)return;
        active_m=M;
    }
    else active_m=rows.activeRows();

    const int blocks_per_row = K / 32;
    const int b_per = (blocks_per_row + kblocks - 1) / kblocks;
    const int b_start = kb * b_per;
    const int b_end = min(b_start + b_per, blocks_per_row);

    for(int row_base=int(blockIdx.z)*ROWS_PER_TILE;row_base<active_m;
        row_base+=int(gridDim.z)*ROWS_PER_TILE)
    {
        if constexpr(HasWork)
            if(!((authored_rows>>row_base)&((std::uint32_t(1)<<ROWS_PER_TILE)-1)))continue;
        float acc[ROWS_PER_TILE][CPT];
#pragma unroll
        for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
        {
#pragma unroll
            for (int c = 0; c < CPT; ++c)
                acc[tile_row][c] = 0.0f;
        }

#pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            if (n >= N)
                continue;

            for (int b = b_start; b < b_end; ++b)
            {
                int32_t packed_groups[8];
                const size_t linear =
                    static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
                const uint8_t* payload = d_payload + linear * NATIVE_QUANTISEDTraits<FMT>::payload_bytes;
                moe_decode_native_quantised_groups<FMT>(payload, packed_groups);

#pragma unroll
                for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
                {
                    const int row = row_base + tile_row;
                    if (row >= active_m)
                        continue;
                    if constexpr(HasWork)
                        if(!(authored_rows&(std::uint32_t(1)<<row)))continue;
                    native_quantised_small_m_accumulate_row<FMT>(
                        acc[tile_row][c], d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, row, K,
                        blocks_per_row, b, packed_groups, payload, d_block_scales, d_block_mins,
                        d_block_emins, linear);
                }
            }
        }

#pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            if (n >= N)
                continue;

#pragma unroll
            for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
            {
                const int row = row_base + tile_row;
                if (row >= active_m)
                    continue;
                if constexpr(HasWork)
                    if(!(authored_rows&(std::uint32_t(1)<<row)))continue;
                const size_t out_idx =
                    static_cast<size_t>(row) * static_cast<size_t>(N) + static_cast<size_t>(n);
                if (kblocks == 1)
                {
                    auto value = acc[tile_row][c];
                    if constexpr (Fold)
                        value = destination_fold.applyUpdate(value, d_C_fp32, out_idx, n);
                    d_C_fp32[out_idx] = value;
                }
                else
                {
                    const size_t partial_idx = (static_cast<size_t>(kb) * static_cast<size_t>(M) +
                                                static_cast<size_t>(row)) *
                                                   static_cast<size_t>(N) +
                                               static_cast<size_t>(n);
                    d_partial_fp32[partial_idx] = acc[tile_row][c];
                }
            }
        }
    }
}

// ============================================================================
// C API: Launch wrappers
// ============================================================================

// ============================================================================
// IQ LDS kernel: LDS-cached grid lookups for IQ grid formats
//
// Uses 256-thread workgroups (4 wavefronts) with cooperative LDS preload of
// the grid lookup table.  Eliminates constant cache thrashing from divergent
// per-thread grid accesses — the dominant bottleneck for IQ formats.
//
// Microbenchmark results (decode-only, 7B FFN_Up shape):
//   IQ1_S/M: +190% decode speedup  (16KB grid → 4 WGs/CU, 16 waves)
//   IQ3_S:   +115% decode speedup  (2KB grid  → 10 WGs/CU, 40 waves)
//   IQ2_S:   +82%  decode speedup  (8KB grid  → 8 WGs/CU, 32 waves)


// ============================================================================
// NativeQuantised GEMV dispatch: lossless decode with FP16 per-block scales
//
// 2-tier dispatch (matching INT8 scatter pattern):
//   KB=1  → Direct:         single kernel writes final output, no partials
//   KB>1  → Scatter+Reduce: scatter writes partials, reduce sums to output
//
// Eliminates atomicAdd and output memset from the old implementation.
// ============================================================================
/**
 * @brief Record physical grouped tiles with device-owned arithmetic admission.
 *
 * The published count never changes the generated serial-M1 K partitions,
 * scratch size, or captured grid. Each fixed scratch tile receives a slice of
 * the same count owner; its producer and reducer agree on both logical row
 * admission and the unchanged physical partial-buffer stride.
 */
template <NativeQuantisedFormat FMT>
bool rocmLaunch_native_quantised_small_m_fp32(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const uint16_t* d_block_scales,
    const uint16_t* d_block_mins,
    const uint32_t* d_block_emins,
    float* d_C_fp32,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* d_partial_fp32,
    int M, int N, int K,
    int device_id, void* stream,
    uint8_t arithmetic_policy_codebook_id,
    const llaminar2::DeviceRowRange *row_range,
    const llaminar2::GPUHCDownOutputView* publication = nullptr,
    llaminar2::rocm::ROCmFP32DestinationFold destination_fold = {})
{
    const auto rows = row_range
        ? *row_range : llaminar2::DeviceRowRange::fullyActive(M);
    if (!stream || rows.physicalRows() != M)
        return false;

    constexpr int NUM_CUS = 60;
    constexpr int MIN_BLOCKS_PER_WAVE = 1;
    constexpr int KB_CAP = NATIVE_QUANTISED_SMALL_M_GRAPH_SAFE_KB_CAP;
    constexpr int TN = 64;
    constexpr int CPT = 1;
    constexpr int THREADS = TN / CPT;

    const int blocks_per_row = K / 32;
    const int grid_n = (N + TN - 1) / TN;
    const auto& rocm_env = llaminar2::debugEnv().rocm;
    const int api_kb_override = llaminar2::rocm::native_quantised_internal::kbOverride();
    const int api_waves_override = llaminar2::rocm::native_quantised_internal::targetWavesOverride();
    const int explicit_kb_override = (api_kb_override > 0) ? api_kb_override : rocm_env.native_quantised_gemv_kb;
    const int explicit_waves_override = (api_waves_override > 0) ? api_waves_override : rocm_env.native_quantised_gemv_target_waves;
    const bool decode_equivalent = llaminar2::rocm::native_quantised_internal::decodeEquivalentSmallMMode();
    if (!llaminar2::rocm::native_quantised_internal::validateDecodeEquivalentKBOverride(
            "rocmGemv_native_quantised_small_m_fp32",
            explicit_kb_override,
            M,
            N,
            K,
            1))
    {
        return false;
    }
    const auto generated_cfg =
        decode_equivalent
            ? llaminar2::rocm::native_quantised_internal::resolveSerialM1RuntimeConfigForPolicy(
                  arithmetic_policy_codebook_id, N, K)
            : llaminar2::rocm::native_quantised_internal::resolveDecodeGeneratedRuntimeConfigForPolicy(
                  arithmetic_policy_codebook_id, M, N, K);
    if (decode_equivalent &&
        explicit_kb_override <= 0 &&
        !generated_cfg.has_generated)
    {
        fprintf(stderr,
                "[rocmGemv_native_quantised_small_m_fp32] no serial M=1 split-K policy for decode-equivalent verifier launch (M=%d N=%d K=%d codebook=%u)\n",
                M, N, K, static_cast<unsigned>(FMT));
        return false;
    }
    const int kb_override =
        (explicit_kb_override > 0)
            ? explicit_kb_override
            : (generated_cfg.has_generated ? generated_cfg.kb : -1);
    const int waves_override =
        (explicit_waves_override > 0)
            ? explicit_waves_override
            : (generated_cfg.has_generated ? generated_cfg.target_waves_per_cu : -1);
    if (kb_override > KB_CAP)
    {
        fprintf(stderr,
                "[rocmGemv_native_quantised_small_m_fp32] requested kb=%d exceeds graph-safe small-M cap=%d (M=%d N=%d K=%d)\n",
                kb_override, KB_CAP, M, N, K);
        return false;
    }

    constexpr int DEFAULT_TARGET_WAVES_PER_CU = 12;
    const int target_waves_per_cu = (waves_override > 0) ? waves_override : DEFAULT_TARGET_WAVES_PER_CU;
    const int target_total_waves = target_waves_per_cu * NUM_CUS;
    const int verifier_rows_per_tile = native_quantised_verifier_rows_per_tile(M);
    const int logical_row_tiles =
        (M + verifier_rows_per_tile - 1) / verifier_rows_per_tile;
    const int resident_tiles = max(1, grid_n * logical_row_tiles);
    int kb_raw = max(1, (target_total_waves + resident_tiles - 1) / resident_tiles);
    int kb_max = max(1, blocks_per_row / MIN_BLOCKS_PER_WAVE);
    kb_max = min(kb_max, KB_CAP);
    kb_raw = min(kb_raw, kb_max);

    int kb = kb_raw;
    if (blocks_per_row % kb_raw != 0)
    {
        for (int d = 1; d < kb_raw; ++d)
        {
            if (kb_raw - d >= 1 && blocks_per_row % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
            if (kb_raw + d <= kb_max && blocks_per_row % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
        }
    }
    if (kb_override > 0)
    {
        kb = min(kb_override, KB_CAP);
        if (kb > blocks_per_row) kb = blocks_per_row;
    }
    if (kb < 1) kb = 1;

    // A standalone Z or materialized head still belongs to its original stage
    // operation. Preserve the actual output, source and Q8 publication before
    // tiling so a scalar optimization retains the same full operand matching.
    NativeQuantisedSmallMBatchedArgs original_packet{.rows=rows,
        .work=llaminar2::projectionWorkForLaunch(M,row_range)};
    original_packet.num_projections=1; original_packet.total_grid_n=grid_n;
    original_packet.N[0]=N; original_packet.grid_offsets[1]=grid_n;
    original_packet.payload[0]=d_payload; original_packet.scales[0]=d_block_scales;
    original_packet.mins[0]=d_block_mins; original_packet.emins[0]=d_block_emins;
    original_packet.output[0]=d_C_fp32; original_packet.partial[0]=d_partial_fp32;
    original_packet.codebook[0]=static_cast<uint8_t>(FMT);
    // Only the complete materialized head may authorize a new singleton call;
    // existing QKV/GU subgroups retain their separately qualified topology.
    const auto verifier_candidate = llaminar2::rocm::nativeQuantisedVerifierProducer(
        device_id, decode_equivalent, {rows, d_payload, N, K, 1, static_cast<uint8_t>(FMT),
        generated_cfg.has_generated && explicit_kb_override <= 0 && explicit_waves_override <= 0 &&
        kb == generated_cfg.kb && target_waves_per_cu == generated_cfg.target_waves_per_cu &&
        (llaminar2::rocm::native_quantised_internal::serialPartitionN() <= 0 || llaminar2::rocm::native_quantised_internal::serialPartitionN() == N)
            ? llaminar2::rocm::NativeQuantisedVerifierPolicyMatch::CanonicalSerial
            : llaminar2::rocm::NativeQuantisedVerifierPolicyMatch::Unmeasured,
        (llaminar2::rocm::hasNativeQuantisedGDNVerifier() ||
            llaminar2::rocm::hasNativeQuantisedMaterializedHead())?&original_packet:nullptr,
        d_A_int8,d_scale_A_blockwise,d_sum_A_blockwise});
    const bool wmma = verifier_candidate != llaminar2::rocm::NativeQuantisedVerifierCandidate::Dot4;

    if (kb > 1 && !d_partial_fp32)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] kb=%d>1 but no small-M partial buffer (M=%d N=%d K=%d)\n",
                kb, M, N, K);
        return false;
    }

    bool hc_down_long_k = false;
    if constexpr (FMT == NATIVE_QUANTISED_Q8_0)
    {
        if (decode_equivalent && generated_cfg.has_generated &&
            generated_cfg.kb == kHCDownPartitions && kb == kHCDownPartitions &&
            arithmetic_policy_codebook_id == static_cast<uint8_t>(NATIVE_QUANTISED_Q8_0) &&
            N == kHCDownColumns && K == kHCDownReduction && M >= 2 && M <= 8 &&
            row_range && row_range->countOwner() && !d_block_mins && !d_block_emins &&
            !llaminar2::currentProjectionRowWork(M) &&
            (publication || llaminar2::qfnGroupedHCDownLongKExperiment()))
        {
            const auto &hardware = llaminar2::rocmHardwareMetadata().at(device_id);
            hc_down_long_k = (hardware.isa_target == "gfx906" || hardware.isa_target.starts_with("gfx906:")) &&
                hardware.compute_units == 60 && hardware.native_wave_width == 64 &&
                hardware.maximum_threads_per_block >= kHCDownTileColumns * kHCDownWindow;
            if (hc_down_long_k &&
                (reinterpret_cast<std::uintptr_t>(stream) <= 2 ||
                 (reinterpret_cast<std::uintptr_t>(d_payload) & 15u) ||
                 (reinterpret_cast<std::uintptr_t>(d_A_int8) & 3u)))
            {
                fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] invalid exact HC-down alignment or stream\n");
                return false;
            }
        }
    }

    if (publication && (!hc_down_long_k || M != kHCDownTileRows ||
        publication->columns != kHCDownColumns ||
        publication->rows.captureIdentity() != rows.captureIdentity() ||
        !publication->activated || !publication->primary || !publication->residual ||
        !publication->primaryScales || !publication->residualScales))
        return false;

    llaminar2::rocm::native_quantised_internal::recordSmallMLaunch(
        M,
        N,
        K,
        static_cast<unsigned>(FMT),
        hc_down_long_k ? N / kHCDownTileColumns : grid_n,
        kb,
        target_waves_per_cu,
        false,
        1,
        static_cast<int>(arithmetic_policy_codebook_id));

    /*
     * Partials are a reusable fixed-row tile, not a semantic M limit. Wider
     * verifier graphs record multiple grouped kernel/reducer pairs while each
     * pair retains weight reuse and scalar M=1's ordered split-K reduction.
     */
    const int workspace_tile_rows = min(
        M,
        llaminar2::kDefaultNativeQuantisedVerifierRowCapacity);
    for (int row_base = 0; row_base < M; row_base += workspace_tile_rows)
    {
        const int tile_m = min(workspace_tile_rows, M - row_base);
        const auto tile_rows = rows.slice(row_base, tile_m);
        const int tile_rows_per_group = wmma ? 16 : native_quantised_verifier_rows_per_tile(tile_m);
        const int tile_row_groups = llaminar2::deviceRowWorkerGroups(
            tile_rows, tile_rows_per_group,
            static_cast<int64_t>(grid_n) * kb, target_total_waves);
        const int reducer_row_groups = llaminar2::deviceRowWorkerGroups(
            tile_rows, 1, grid_n, target_total_waves);
        const int8_t* tile_a =
            d_A_int8 + static_cast<size_t>(row_base) * K;
        const float* tile_scales =
            d_scale_A_blockwise +
            static_cast<size_t>(row_base) * blocks_per_row;
        const int32_t* tile_sums =
            d_sum_A_blockwise
                ? d_sum_A_blockwise +
                      static_cast<size_t>(row_base) * blocks_per_row
                : nullptr;
        float* tile_c =
            d_C_fp32 + static_cast<size_t>(row_base) * N;

        if (hc_down_long_k)
        {
            static_assert(llaminar2::kDefaultNativeQuantisedVerifierRowCapacity >= 8);
            const dim3 grid(N / kHCDownTileColumns, kb, (tile_m + kHCDownTileRows - 1) / kHCDownTileRows);
            constexpr int threads = kHCDownTileColumns * kHCDownWindow;
            hipLaunchKernelGGL(qfn_hc_down_long_k_contributions, grid, dim3(threads), 0,
                static_cast<hipStream_t>(stream),
                (HCDownLongKArgs{tile_a, d_payload, d_block_scales, tile_scales, d_partial_fp32, tile_rows}));
            llaminar2::PerfStatsCollector::addCounter("gemm", "qfn_grouped_hc_down_long_k_nodes", 1.0,
                "gemm", "rocm:" + std::to_string(device_id));
        }
        else if (wmma)
        {
            NativeQuantisedSmallMBatchedArgs bundle{.rows=tile_rows,
                .work=llaminar2::projectionWorkForLaunch(M,row_range).slice(row_base,tile_m)};
            bundle.num_projections=1; bundle.total_grid_n=grid_n;
            bundle.N[0]=N; bundle.grid_offsets[1]=grid_n;
            bundle.payload[0]=d_payload; bundle.scales[0]=d_block_scales;
            bundle.mins[0]=d_block_mins; bundle.emins[0]=d_block_emins;
            bundle.output[0]=tile_c; bundle.partial[0]=d_partial_fp32;
            bundle.codebook[0]=static_cast<uint8_t>(FMT);
            llaminar2::rocm::submitNativeQuantisedVerifierWmma({
                .activations=tile_a,.activation_scales=tile_scales,.activation_sums=tile_sums,.projections=bundle,
                .device=device_id,.k=K,.partitions=kb,.row_workers=tile_row_groups,
                .has_request_work=llaminar2::currentProjectionRowWork(M)!=nullptr,
                .publication=kb>1 ? llaminar2::rocm::VerifierWmmaPublication::OrderedPartial :
                    (destination_fold.isIdentity() ? llaminar2::rocm::VerifierWmmaPublication::Standalone :
                        llaminar2::rocm::VerifierWmmaPublication::StandaloneFold),
                .destination_fold=destination_fold},stream);
        }
        else
        {
            const auto row_work = llaminar2::projectionWorkForLaunch(M, row_range).slice(row_base, tile_m);
            const auto launch_tile = [&]<int RowTile>() {
                llaminar2::dispatchProjectionWorkMode(M, [&]<bool HasWork>() {
                    const auto submit = [&]<bool Fold>() {
                        hipLaunchKernelGGL(
                            HIP_KERNEL_NAME(gemv_native_quantised_small_m_kernel_t<TN, CPT, RowTile, FMT, HasWork, Fold>),
                            dim3(grid_n, kb, tile_row_groups), dim3(THREADS, 1, 1), 0,
                            static_cast<hipStream_t>(stream),
                            tile_a, d_payload, d_block_scales, d_block_mins, d_block_emins,
                            tile_c, d_partial_fp32, tile_scales, tile_sums,
                            tile_m, N, K, kb, tile_rows, row_work, destination_fold);
                    };
                    if (kb == 1 && !destination_fold.isIdentity())
                        submit.template operator()<true>();
                    else
                        submit.template operator()<false>();
                });
            };
            if (tile_rows_per_group == NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE)
                launch_tile.template operator()<NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE>();
            else
                launch_tile.template operator()<NATIVE_QUANTISED_VERIFIER_WIDE_ROWS_PER_TILE>();
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] kernel launch failed: %s (M=%d tile_base=%d tile_m=%d N=%d K=%d codebook=%u)\n",
                    hipGetErrorString(err), M, row_base, tile_m, N, K,
                    static_cast<unsigned>(FMT));
            return false;
        }

        if (publication)
        {
            hipLaunchKernelGGL(groupedHCDownReducePublish, dim3(grid_n, tile_m), dim3(THREADS), 0,
                static_cast<hipStream_t>(stream), d_partial_fp32, tile_c, *publication);
        }
        else if (kb > 1)
        {
            const auto row_work = llaminar2::projectionWorkForLaunch(M, row_range).slice(row_base, tile_m);
            llaminar2::dispatchProjectionWorkMode(M, [&]<bool HasWork>() {
                const auto submit = [&]<bool Fold>() {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(gemv_native_quantised_reduce_small_m_kernel_t<TN, CPT, HasWork, Fold>),
                        dim3(grid_n, reducer_row_groups), dim3(THREADS, 1, 1), 0,
                        static_cast<hipStream_t>(stream),
                        d_partial_fp32, tile_c, tile_m, N, kb, tile_rows, row_work, destination_fold);
                };
                if (destination_fold.isIdentity())
                    submit.template operator()<false>();
                else
                    submit.template operator()<true>();
            });
        }

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] reduce launch failed: %s (M=%d tile_base=%d tile_m=%d N=%d K=%d codebook=%u)\n",
                    hipGetErrorString(err), M, row_base, tile_m, N, K,
                    static_cast<unsigned>(FMT));
            return false;
        }
    }

    return true;
}

template <NativeQuantisedFormat FMT>
bool rocmLaunch_native_quantised_small_m_batched_fp32(
    const int8_t* d_A_int8,
    const uint8_t* const* d_payloads,
    const uint16_t* const* d_block_scales,
    const uint16_t* const* d_block_mins,
    const uint32_t* const* d_block_emins,
    const float* const* d_biases,
    float* const* d_outputs,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* const* d_partials,
    const int* Ns,
    int num_projections,
    int M, int K,
    int device_id,
    void* stream,
    int call_kb_override = -1,
    int call_waves_override = -1,
    const llaminar2::DeviceRowRange *row_range = nullptr)
{
    constexpr int NUM_CUS = 60;
    constexpr int MIN_BLOCKS_PER_WAVE = 1;
    constexpr int KB_CAP = NATIVE_QUANTISED_SMALL_M_GRAPH_SAFE_KB_CAP;
    constexpr int TN = 64;
    constexpr int CPT = 1;
    constexpr int THREADS = TN / CPT;

    (void)device_id;

    const int blocks_per_row = K / 32;
    const auto rows = row_range ? *row_range :
        llaminar2::DeviceRowRange::fullyActive(M);
    if (!stream || rows.physicalRows() != M)
        return false;
    NativeQuantisedSmallMBatchedArgs args{.rows = rows,.work=llaminar2::projectionWorkForLaunch(M,row_range)};
    args.num_projections = num_projections;
    for (int p = 0; p < num_projections; ++p)
    {
        const int grid_n = (Ns[p] + TN - 1) / TN;
        args.N[p] = Ns[p];
        args.grid_offsets[p] = args.total_grid_n;
        args.total_grid_n += grid_n;
        args.payload[p] = d_payloads[p];
        args.scales[p] = d_block_scales[p];
        args.mins[p] = d_block_mins ? d_block_mins[p] : nullptr;
        args.emins[p] = d_block_emins ? d_block_emins[p] : nullptr;
        args.bias[p] = d_biases ? d_biases[p] : nullptr;
        args.output[p] = d_outputs[p];
        args.partial[p] = d_partials ? d_partials[p] : nullptr;
        args.codebook[p] = static_cast<uint8_t>(FMT);
    }
    args.grid_offsets[num_projections] = args.total_grid_n;

    const auto& rocm_env = llaminar2::debugEnv().rocm;
    const int api_kb_override = llaminar2::rocm::native_quantised_internal::kbOverride();
    const int api_waves_override = llaminar2::rocm::native_quantised_internal::targetWavesOverride();
    const int explicit_kb_override =
        (call_kb_override > 0)
            ? call_kb_override
            : ((api_kb_override > 0) ? api_kb_override : rocm_env.native_quantised_gemv_kb);
    const int explicit_waves_override =
        (call_waves_override > 0)
            ? call_waves_override
            : ((api_waves_override > 0) ? api_waves_override : rocm_env.native_quantised_gemv_target_waves);
    const bool decode_equivalent = llaminar2::rocm::native_quantised_internal::decodeEquivalentSmallMMode();
    if (!llaminar2::rocm::native_quantised_internal::validateDecodeEquivalentKBOverride(
            "rocmGemv_native_quantised_small_m_batched_fp32",
            explicit_kb_override,
            M,
            0,
            K,
            num_projections))
    {
        return false;
    }
    const auto generated_cfg =
        decode_equivalent
            ? llaminar2::rocm::native_quantised_internal::resolveCommonSerialM1RuntimeConfig(
                  static_cast<uint8_t>(FMT), Ns, num_projections, K)
            : (num_projections == 1
                   ? llaminar2::rocm::native_quantised_internal::resolveDecodeGeneratedRuntimeConfig(
                         static_cast<uint8_t>(FMT), M, Ns[0], K)
                   : NativeQuantisedDecodeGeneratedRuntimeConfig{});
    if (decode_equivalent &&
        explicit_kb_override <= 0 &&
        !generated_cfg.has_generated)
    {
        fprintf(stderr,
                "[rocmGemv_native_quantised_small_m_batched_fp32] fused verifier group has no common serial M=1 split-K policy (M=%d K=%d projections=%d codebook=%u)\n",
                M, K, num_projections, static_cast<unsigned>(FMT));
        return false;
    }
    const int kb_override =
        (explicit_kb_override > 0)
            ? explicit_kb_override
            : (generated_cfg.has_generated ? generated_cfg.kb : -1);
    const int waves_override =
        (explicit_waves_override > 0)
            ? explicit_waves_override
            : (generated_cfg.has_generated ? generated_cfg.target_waves_per_cu : -1);
    if (kb_override > KB_CAP)
    {
        fprintf(stderr,
                "[rocmGemv_native_quantised_small_m_batched_fp32] requested kb=%d exceeds graph-safe small-M cap=%d (M=%d K=%d projections=%d)\n",
                kb_override, KB_CAP, M, K, num_projections);
        return false;
    }

    constexpr int DEFAULT_TARGET_WAVES_PER_CU = 12;
    const int target_waves_per_cu = (waves_override > 0) ? waves_override : DEFAULT_TARGET_WAVES_PER_CU;
    const int target_total_waves = target_waves_per_cu * NUM_CUS;
    const int verifier_rows_per_tile = native_quantised_verifier_rows_per_tile(M);
    const int logical_row_tiles =
        (M + verifier_rows_per_tile - 1) / verifier_rows_per_tile;
    const int resident_tiles =
        max(1, args.total_grid_n * logical_row_tiles);
    int kb_raw = max(1, (target_total_waves + resident_tiles - 1) / resident_tiles);
    int kb_max = max(1, blocks_per_row / MIN_BLOCKS_PER_WAVE);
    kb_max = min(kb_max, KB_CAP);
    kb_raw = min(kb_raw, kb_max);

    int kb = kb_raw;
    if (blocks_per_row % kb_raw != 0)
    {
        for (int d = 1; d < kb_raw; ++d)
        {
            if (kb_raw - d >= 1 && blocks_per_row % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
            if (kb_raw + d <= kb_max && blocks_per_row % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
        }
    }
    if (kb_override > 0)
    {
        kb = min(kb_override, KB_CAP);
        if (kb > blocks_per_row) kb = blocks_per_row;
    }
    if (kb < 1) kb = 1;

    bool canonical_members = generated_cfg.has_generated && api_kb_override <= 0 && api_waves_override <= 0 &&
        rocm_env.native_quantised_gemv_kb <= 0 && rocm_env.native_quantised_gemv_target_waves <= 0;
    canonical_members = canonical_members && llaminar2::rocm::matchesNativeQuantisedVerifierSerialPolicies(
        num_projections, {kb,target_waves_per_cu}, [&](int member)
            -> std::optional<llaminar2::rocm::NativeQuantisedVerifierSerialPolicy> {
            const auto policy = llaminar2::rocm::native_quantised_internal::resolveSerialM1RuntimeConfig(static_cast<uint8_t>(FMT), Ns[member], K);
            if(!policy.has_generated || (llaminar2::rocm::native_quantised_internal::serialPartitionN() > 0 &&
                llaminar2::rocm::native_quantised_internal::serialPartitionN() != Ns[member]))return std::nullopt;
            return llaminar2::rocm::NativeQuantisedVerifierSerialPolicy{policy.kb,policy.target_waves_per_cu};
        });
    const auto verifier_candidate = llaminar2::rocm::nativeQuantisedVerifierProducer(
        device_id, decode_equivalent, {rows, args.payload[0], Ns[0], K, num_projections, static_cast<uint8_t>(FMT),
        canonical_members ? llaminar2::rocm::NativeQuantisedVerifierPolicyMatch::CanonicalSerial
            : llaminar2::rocm::NativeQuantisedVerifierPolicyMatch::Unmeasured,
        &args, d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise});
    const bool wmma = verifier_candidate != llaminar2::rocm::NativeQuantisedVerifierCandidate::Dot4;

    if (kb > 1)
    {
        for (int p = 0; p < num_projections; ++p)
        {
            if (!args.partial[p])
            {
                fprintf(stderr,
                        "[rocmGemv_native_quantised_small_m_batched_fp32] kb=%d>1 but projection %d has no partial buffer (M=%d N=%d K=%d)\n",
                        kb, p, M, args.N[p], K);
                return false;
            }
        }
    }

    llaminar2::rocm::native_quantised_internal::recordSmallMLaunch(
        M,
        0,
        K,
        static_cast<unsigned>(FMT),
        args.total_grid_n,
        kb,
        target_waves_per_cu,
        true,
        num_projections);

    const int workspace_tile_rows = min(
        M,
        llaminar2::kDefaultNativeQuantisedVerifierRowCapacity);
    for (int row_base = 0; row_base < M; row_base += workspace_tile_rows)
    {
        const int tile_m = min(workspace_tile_rows, M - row_base);
        const int tile_rows_per_group = wmma ? 16 : native_quantised_verifier_rows_per_tile(tile_m);
        NativeQuantisedSmallMBatchedArgs tile_args = args;
        tile_args.rows = rows.slice(row_base, tile_m);
        tile_args.work = args.work.slice(row_base,tile_m);
        const int tile_row_groups = llaminar2::deviceRowWorkerGroups(
            tile_args.rows, tile_rows_per_group,
            static_cast<int64_t>(args.total_grid_n) * kb, target_total_waves);
        const int reducer_row_groups = llaminar2::deviceRowWorkerGroups(
            tile_args.rows, 1, args.total_grid_n, target_total_waves);
        for (int p = 0; p < num_projections; ++p)
        {
            tile_args.output[p] +=
                static_cast<size_t>(row_base) * tile_args.N[p];
        }

        const int8_t* tile_a =
            d_A_int8 + static_cast<size_t>(row_base) * K;
        const float* tile_scales =
            d_scale_A_blockwise +
            static_cast<size_t>(row_base) * blocks_per_row;
        const int32_t* tile_sums =
            d_sum_A_blockwise
                ? d_sum_A_blockwise +
                      static_cast<size_t>(row_base) * blocks_per_row
                : nullptr;

        // Publication is fixed by the serial-M1 K-partition policy. Compile
        // only its live destination/epilogue into this captured producer.
        const auto launch = [&]<NativeQuantisedVerifierPublication Publication>()
        {
            if (wmma)
            {
                llaminar2::rocm::submitNativeQuantisedVerifierWmma({
                    .activations=tile_a,.activation_scales=tile_scales,.activation_sums=tile_sums,.projections=tile_args,
                    .device=device_id,.k=K,.partitions=kb,.row_workers=tile_row_groups,
                    .has_request_work=llaminar2::currentProjectionRowWork(M)!=nullptr,
                    .publication=Publication==NativeQuantisedVerifierPublication::DirectOutput
                        ? llaminar2::rocm::VerifierWmmaPublication::Batched
                        : llaminar2::rocm::VerifierWmmaPublication::OrderedPartial},stream);
            }
            else if (tile_rows_per_group == NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE)
            {
                llaminar2::dispatchProjectionWorkMode(M,[&]<bool HasWork>() {
        hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_native_quantised_small_m_batched_shared_kernel_t<
                        TN, CPT, NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE, FMT, Publication, HasWork>),
                    dim3(tile_args.total_grid_n, kb, tile_row_groups),
                    dim3(THREADS, 1, 1), 0,
                    static_cast<hipStream_t>(stream),
                    tile_a, tile_scales, tile_sums, tile_args, K,
                    (blocks_per_row + kb - 1) / kb);
        });
            }
            else
            {
                llaminar2::dispatchProjectionWorkMode(M,[&]<bool HasWork>() {
        hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_native_quantised_small_m_batched_shared_kernel_t<
                        TN, CPT, NATIVE_QUANTISED_VERIFIER_WIDE_ROWS_PER_TILE, FMT, Publication, HasWork>),
                    dim3(tile_args.total_grid_n, kb, tile_row_groups),
                    dim3(THREADS, 1, 1), 0,
                    static_cast<hipStream_t>(stream),
                    tile_a, tile_scales, tile_sums, tile_args, K,
                    (blocks_per_row + kb - 1) / kb);
        });
            }
        };
        if (kb == 1)
            launch.template operator()<NativeQuantisedVerifierPublication::DirectOutput>();
        else
            launch.template operator()<NativeQuantisedVerifierPublication::OrderedPartial>();

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmGemv_native_quantised_small_m_batched_fp32] kernel launch failed: %s (M=%d tile_base=%d tile_m=%d K=%d projections=%d codebook=%u)\n",
                    hipGetErrorString(err), M, row_base, tile_m, K,
                    num_projections, static_cast<unsigned>(FMT));
            return false;
        }

        if (kb > 1)
        {
            llaminar2::dispatchProjectionWorkMode(M,[&]<bool HasWork>() {
        hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_native_quantised_reduce_small_m_batched_kernel_t<TN, CPT, HasWork>),
                dim3(tile_args.total_grid_n, reducer_row_groups),
                dim3(THREADS, 1, 1), 0,
                static_cast<hipStream_t>(stream), tile_args, kb);
        });
        }

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmGemv_native_quantised_small_m_batched_fp32] reduce launch failed: %s (M=%d tile_base=%d tile_m=%d K=%d projections=%d codebook=%u)\n",
                    hipGetErrorString(err), M, row_base, tile_m, K,
                    num_projections, static_cast<unsigned>(FMT));
            return false;
        }
    }

    return true;
}

static bool nativeQuantisedCodebookRequiresAux(uint8_t codebook_id)
{
    switch (static_cast<NativeQuantisedFormat>(codebook_id))
    {
    case NATIVE_QUANTISED_IQ3_S:
    case NATIVE_QUANTISED_IQ4_XS_SOURCE:
    case NATIVE_QUANTISED_Q4_1:
    case NATIVE_QUANTISED_Q5_1:
    case NATIVE_QUANTISED_Q6_K:
    case NATIVE_QUANTISED_Q3_K:
    case NATIVE_QUANTISED_Q2_K:
    case NATIVE_QUANTISED_IQ2_S:
    case NATIVE_QUANTISED_IQ2_XS:
    case NATIVE_QUANTISED_IQ1_S:
    case NATIVE_QUANTISED_IQ1_M:
    case NATIVE_QUANTISED_EXPANDED_INT8_MIN:
        return true;
    default:
        return false;
    }
}

static bool nativeQuantisedCodebookRequiresEmins(uint8_t codebook_id)
{
    return static_cast<NativeQuantisedFormat>(codebook_id) == NATIVE_QUANTISED_Q2_K;
}

static bool nativeQuantisedCodebookSupported(uint8_t codebook_id)
{
    switch (static_cast<NativeQuantisedFormat>(codebook_id))
    {
    case NATIVE_QUANTISED_Q4_0:
    case NATIVE_QUANTISED_IQ4_NL:
    case NATIVE_QUANTISED_IQ4_XS_SOURCE:
    case NATIVE_QUANTISED_Q4_1:
    case NATIVE_QUANTISED_Q5_0:
    case NATIVE_QUANTISED_Q5_1:
    case NATIVE_QUANTISED_Q6_K:
    case NATIVE_QUANTISED_Q3_K:
    case NATIVE_QUANTISED_Q2_K:
    case NATIVE_QUANTISED_IQ3_S:
    case NATIVE_QUANTISED_IQ3_XXS:
    case NATIVE_QUANTISED_IQ2_S:
    case NATIVE_QUANTISED_IQ2_XS:
    case NATIVE_QUANTISED_IQ2_XXS:
    case NATIVE_QUANTISED_IQ1_S:
    case NATIVE_QUANTISED_IQ1_M:
    case NATIVE_QUANTISED_Q8_0:
    case NATIVE_QUANTISED_EXPANDED_INT8_MIN:
        return true;
    default:
        return false;
    }
}

bool rocmLaunch_native_quantised_small_m_batched_mixed_fp32(
    const int8_t* d_A_int8,
    const uint8_t* const* d_payloads,
    const uint16_t* const* d_block_scales,
    const uint16_t* const* d_block_mins,
    const uint32_t* const* d_block_emins,
    const float* const* d_biases,
    float* const* d_outputs,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* const* d_partials,
    const int* Ns,
    const uint8_t* codebook_ids,
    int num_projections,
    int M, int K,
    int device_id,
    void* stream,
    int call_kb_override = -1,
    int call_waves_override = -1,
    const llaminar2::DeviceRowRange *row_range = nullptr)
{
    constexpr int NUM_CUS = 60;
    constexpr int MIN_BLOCKS_PER_WAVE = 1;
    constexpr int KB_CAP = NATIVE_QUANTISED_SMALL_M_GRAPH_SAFE_KB_CAP;
    constexpr int TN = 64;
    constexpr int CPT = 1;
    constexpr int THREADS = TN / CPT;

    (void)device_id;

    const int blocks_per_row = K / 32;
    const auto rows = row_range ? *row_range :
        llaminar2::DeviceRowRange::fullyActive(M);
    if (!stream || rows.physicalRows() != M)
        return false;
    NativeQuantisedSmallMBatchedArgs args{.rows = rows,.work=llaminar2::projectionWorkForLaunch(M,row_range)};
    args.num_projections = num_projections;
    for (int p = 0; p < num_projections; ++p)
    {
        const int grid_n = (Ns[p] + TN - 1) / TN;
        args.N[p] = Ns[p];
        args.grid_offsets[p] = args.total_grid_n;
        args.total_grid_n += grid_n;
        args.payload[p] = d_payloads[p];
        args.scales[p] = d_block_scales[p];
        args.mins[p] = d_block_mins ? d_block_mins[p] : nullptr;
        args.emins[p] = d_block_emins ? d_block_emins[p] : nullptr;
        args.bias[p] = d_biases ? d_biases[p] : nullptr;
        args.output[p] = d_outputs[p];
        args.partial[p] = d_partials ? d_partials[p] : nullptr;
        args.codebook[p] = codebook_ids[p];
    }
    args.grid_offsets[num_projections] = args.total_grid_n;

    const auto& rocm_env = llaminar2::debugEnv().rocm;
    const int api_kb_override = llaminar2::rocm::native_quantised_internal::kbOverride();
    const int api_waves_override = llaminar2::rocm::native_quantised_internal::targetWavesOverride();
    const int explicit_kb_override =
        (call_kb_override > 0)
            ? call_kb_override
            : ((api_kb_override > 0) ? api_kb_override : rocm_env.native_quantised_gemv_kb);
    const int explicit_waves_override =
        (call_waves_override > 0)
            ? call_waves_override
            : ((api_waves_override > 0) ? api_waves_override : rocm_env.native_quantised_gemv_target_waves);
    const bool decode_equivalent = llaminar2::rocm::native_quantised_internal::decodeEquivalentSmallMMode();
    if (!llaminar2::rocm::native_quantised_internal::validateDecodeEquivalentKBOverride(
            "rocmGemv_native_quantised_small_m_batched_mixed_fp32",
            explicit_kb_override,
            M,
            0,
            K,
            num_projections))
    {
        return false;
    }
    const auto generated_cfg =
        decode_equivalent
            ? llaminar2::rocm::native_quantised_internal::resolveCommonSerialM1RuntimeConfig(
                  codebook_ids, Ns, num_projections, K)
            : (num_projections == 1
                   ? llaminar2::rocm::native_quantised_internal::resolveDecodeGeneratedRuntimeConfig(
                         codebook_ids[0], M, Ns[0], K)
                   : NativeQuantisedDecodeGeneratedRuntimeConfig{});
    if (decode_equivalent &&
        explicit_kb_override <= 0 &&
        !generated_cfg.has_generated)
    {
        fprintf(stderr,
                "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] fused verifier group has no common serial M=1 split-K policy (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    const int kb_override =
        (explicit_kb_override > 0)
            ? explicit_kb_override
            : (generated_cfg.has_generated ? generated_cfg.kb : -1);
    const int waves_override =
        (explicit_waves_override > 0)
            ? explicit_waves_override
            : (generated_cfg.has_generated ? generated_cfg.target_waves_per_cu : -1);
    if (kb_override > KB_CAP)
    {
        fprintf(stderr,
                "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] requested kb=%d exceeds graph-safe small-M cap=%d (M=%d K=%d projections=%d)\n",
                kb_override, KB_CAP, M, K, num_projections);
        return false;
    }

    constexpr int DEFAULT_TARGET_WAVES_PER_CU = 12;
    const int target_waves_per_cu = (waves_override > 0) ? waves_override : DEFAULT_TARGET_WAVES_PER_CU;
    const int target_total_waves = target_waves_per_cu * NUM_CUS;
    const int verifier_rows_per_tile = native_quantised_verifier_rows_per_tile(M);
    const int logical_row_tiles =
        (M + verifier_rows_per_tile - 1) / verifier_rows_per_tile;
    const int resident_tiles =
        max(1, args.total_grid_n * logical_row_tiles);
    int kb_raw = max(1, (target_total_waves + resident_tiles - 1) / resident_tiles);
    int kb_max = max(1, blocks_per_row / MIN_BLOCKS_PER_WAVE);
    kb_max = min(kb_max, KB_CAP);
    kb_raw = min(kb_raw, kb_max);

    int kb = kb_raw;
    if (blocks_per_row % kb_raw != 0)
    {
        for (int d = 1; d < kb_raw; ++d)
        {
            if (kb_raw - d >= 1 && blocks_per_row % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
            if (kb_raw + d <= kb_max && blocks_per_row % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
        }
    }
    if (kb_override > 0)
    {
        kb = min(kb_override, KB_CAP);
        if (kb > blocks_per_row) kb = blocks_per_row;
    }
    if (kb < 1) kb = 1;

    // Equal K partitions alone do not authenticate a complete subgroup's
    // performance policy: every member must also own the actual wave target.
    bool canonical_members = (llaminar2::rocm::ScopedNativeQuantisedVerifierBundle::current() ||
        llaminar2::rocm::hasNativeQuantisedGDNVerifier()) &&
        generated_cfg.has_generated && api_kb_override <= 0 && api_waves_override <= 0 &&
        rocm_env.native_quantised_gemv_kb <= 0 && rocm_env.native_quantised_gemv_target_waves <= 0;
    canonical_members = canonical_members && llaminar2::rocm::matchesNativeQuantisedVerifierSerialPolicies(
        num_projections, {kb,target_waves_per_cu}, [&](int member)
            -> std::optional<llaminar2::rocm::NativeQuantisedVerifierSerialPolicy> {
            const auto policy = llaminar2::rocm::native_quantised_internal::resolveSerialM1RuntimeConfig(codebook_ids[member], Ns[member], K);
            if(!policy.has_generated || (llaminar2::rocm::native_quantised_internal::serialPartitionN() > 0 &&
                llaminar2::rocm::native_quantised_internal::serialPartitionN() != Ns[member]))return std::nullopt;
            return llaminar2::rocm::NativeQuantisedVerifierSerialPolicy{policy.kb,policy.target_waves_per_cu};
        });
    const auto verifier_candidate = llaminar2::rocm::nativeQuantisedVerifierProducer(
        device_id, decode_equivalent, {rows, args.payload[0], Ns[0], K, num_projections, args.codebook[0],
        canonical_members ? llaminar2::rocm::NativeQuantisedVerifierPolicyMatch::CanonicalSerial
            : llaminar2::rocm::NativeQuantisedVerifierPolicyMatch::Unmeasured,
        &args, d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise});
    const bool wmma = verifier_candidate != llaminar2::rocm::NativeQuantisedVerifierCandidate::Dot4;

    if (kb > 1)
    {
        for (int p = 0; p < num_projections; ++p)
        {
            if (!args.partial[p])
            {
                fprintf(stderr,
                        "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] kb=%d>1 but projection %d has no partial buffer (M=%d N=%d K=%d codebook=%u)\n",
                        kb, p, M, args.N[p], K, static_cast<unsigned>(args.codebook[p]));
                return false;
            }
        }
    }

    llaminar2::rocm::native_quantised_internal::recordSmallMLaunch(
        M,
        0,
        K,
        255,
        args.total_grid_n,
        kb,
        target_waves_per_cu,
        true,
        num_projections);

    const int workspace_tile_rows = min(
        M,
        llaminar2::kDefaultNativeQuantisedVerifierRowCapacity);
    for (int row_base = 0; row_base < M; row_base += workspace_tile_rows)
    {
        const int tile_m = min(workspace_tile_rows, M - row_base);
        const int tile_rows_per_group = wmma ? 16 : native_quantised_verifier_rows_per_tile(tile_m);
        NativeQuantisedSmallMBatchedArgs tile_args = args;
        tile_args.rows = rows.slice(row_base, tile_m);
        tile_args.work = args.work.slice(row_base,tile_m);
        const int tile_row_groups = llaminar2::deviceRowWorkerGroups(
            tile_args.rows, tile_rows_per_group,
            static_cast<int64_t>(args.total_grid_n) * kb, target_total_waves);
        const int reducer_row_groups = llaminar2::deviceRowWorkerGroups(
            tile_args.rows, 1, args.total_grid_n, target_total_waves);
        for (int p = 0; p < num_projections; ++p)
        {
            tile_args.output[p] +=
                static_cast<size_t>(row_base) * tile_args.N[p];
        }

        const int8_t* tile_a =
            d_A_int8 + static_cast<size_t>(row_base) * K;
        const float* tile_scales =
            d_scale_A_blockwise +
            static_cast<size_t>(row_base) * blocks_per_row;
        const int32_t* tile_sums =
            d_sum_A_blockwise
                ? d_sum_A_blockwise +
                      static_cast<size_t>(row_base) * blocks_per_row
                : nullptr;

        // Publication is fixed by the serial-M1 K-partition policy. Compile
        // only its live destination/epilogue into this captured producer.
        const auto launch = [&]<NativeQuantisedVerifierPublication Publication>()
        {
            if (wmma)
            {
                llaminar2::rocm::submitNativeQuantisedVerifierWmma({
                    .activations=tile_a,.activation_scales=tile_scales,.activation_sums=tile_sums,.projections=tile_args,
                    .device=device_id,.k=K,.partitions=kb,.row_workers=tile_row_groups,
                    .has_request_work=llaminar2::currentProjectionRowWork(M)!=nullptr,
                    .publication=Publication==NativeQuantisedVerifierPublication::DirectOutput
                        ? llaminar2::rocm::VerifierWmmaPublication::Batched
                        : llaminar2::rocm::VerifierWmmaPublication::OrderedPartial},stream);
            }
            else if (tile_rows_per_group == NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE)
            {
                llaminar2::dispatchProjectionWorkMode(M,[&]<bool HasWork>() {
        hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_native_quantised_small_m_batched_shared_mixed_kernel_t<
                        TN, CPT, NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE, Publication, HasWork>),
                    dim3(tile_args.total_grid_n, kb, tile_row_groups),
                    dim3(THREADS, 1, 1), 0,
                    static_cast<hipStream_t>(stream),
                    tile_a, tile_scales, tile_sums, tile_args, K,
                    (blocks_per_row + kb - 1) / kb);
        });
            }
            else
            {
                llaminar2::dispatchProjectionWorkMode(M,[&]<bool HasWork>() {
        hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_native_quantised_small_m_batched_shared_mixed_kernel_t<
                        TN, CPT, NATIVE_QUANTISED_VERIFIER_WIDE_ROWS_PER_TILE, Publication, HasWork>),
                    dim3(tile_args.total_grid_n, kb, tile_row_groups),
                    dim3(THREADS, 1, 1), 0,
                    static_cast<hipStream_t>(stream),
                    tile_a, tile_scales, tile_sums, tile_args, K,
                    (blocks_per_row + kb - 1) / kb);
        });
            }
        };
        if (kb == 1)
            launch.template operator()<NativeQuantisedVerifierPublication::DirectOutput>();
        else
            launch.template operator()<NativeQuantisedVerifierPublication::OrderedPartial>();

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] kernel launch failed: %s (M=%d tile_base=%d tile_m=%d K=%d projections=%d)\n",
                    hipGetErrorString(err), M, row_base, tile_m, K,
                    num_projections);
            return false;
        }

        if (kb > 1)
        {
            llaminar2::dispatchProjectionWorkMode(M,[&]<bool HasWork>() {
        hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_native_quantised_reduce_small_m_batched_kernel_t<TN, CPT, HasWork>),
                dim3(tile_args.total_grid_n, reducer_row_groups),
                dim3(THREADS, 1, 1), 0,
                static_cast<hipStream_t>(stream), tile_args, kb);
        });
        }

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] reduce launch failed: %s (M=%d tile_base=%d tile_m=%d K=%d projections=%d)\n",
                    hipGetErrorString(err), M, row_base, tile_m, K,
                    num_projections);
            return false;
        }
    }

    return true;
}

namespace llaminar2::rocm {
bool enqueueGroupedHCDownReducePublish(
    const int8_t* activation, const uint8_t* weights, const uint16_t* weight_scales,
    float* raw, const float* activation_scales, const int32_t* activation_sums,
    float* partials, int device, void* stream, const GPUHCDownOutputView& publication)
{
    if (reinterpret_cast<std::uintptr_t>(stream) <= 2 || !activation || !weights ||
        !weight_scales || !raw || !activation_scales || !activation_sums || !partials)
        return false;
    HipDeviceGuard::setDevice(device);
    return rocmLaunch_native_quantised_small_m_fp32<NATIVE_QUANTISED_Q8_0>(
        activation, weights, weight_scales, nullptr, nullptr, raw, activation_scales, activation_sums,
        partials, kHCDownTileRows, kHCDownColumns, kHCDownReduction, device, stream,
        static_cast<uint8_t>(NATIVE_QUANTISED_Q8_0), &publication.rows, &publication);
}
} // namespace llaminar2::rocm


extern "C" {

/**
 * @brief Execute verifier-depth GEMV with distinct decoder and math policy.
 *
 * `codebook_id` selects the physical weight decoder. The arithmetic policy ID
 * selects the generated split-K geometry whose ordered reduction every row
 * must reproduce. Expert promotion may change only the former.
 */
bool rocmGemv_native_quantised_small_m_fp32_with_sums_policy(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const void* d_block_scales,
    const void* d_block_mins,
    const void* d_block_emins,
    float* d_C_fp32,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* d_partial_fp32,
    int M, int N, int K,
    uint8_t codebook_id,
    uint8_t arithmetic_policy_codebook_id,
    int device_id, void* stream,
    const llaminar2::DeviceRowRange *row_range,
    llaminar2::rocm::ROCmFP32DestinationFold destination_fold)
{
    if (!stream || N <= 0 || K <= 0 ||
        (row_range && row_range->physicalRows() != M))
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] invalid stream or physical row geometry (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (M <= 0 || (M == 1 && (!row_range || !row_range->countOwner())) ||
        (row_range && row_range->physicalRows() != M))
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] M requires positive rows and an explicit count owner at M1 (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }
    if ((K % 32) != 0)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] K must be multiple of 32 (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }
    if (!d_A_int8 || !d_payload || !d_block_scales || !d_C_fp32 || !d_scale_A_blockwise)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] missing required pointer (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    const uint16_t* d_scales_u16 = static_cast<const uint16_t*>(d_block_scales);
    const uint16_t* d_mins_u16 = static_cast<const uint16_t*>(d_block_mins);
    const uint32_t* d_emins_u32 = static_cast<const uint32_t*>(d_block_emins);

    #define LAUNCH_SMALL_M(FMT_TAG)                                                              \
        do {                                                                                      \
            using Traits = NATIVE_QUANTISEDTraits<FMT_TAG>;                                                  \
            if constexpr (Traits::requires_secondary_metadata)                         \
            {                                                                                     \
                if (!d_mins_u16)                                                                  \
                {                                                                                 \
                    fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] codebook_id=%u requires mins/aux buffer (M=%d N=%d K=%d)\n", \
                            static_cast<unsigned>(codebook_id), M, N, K);                         \
                    return false;                                                                 \
                }                                                                                 \
            }                                                                                     \
            if constexpr (Traits::needs_emin_array)                                               \
            {                                                                                     \
                if (!d_emins_u32)                                                                 \
                {                                                                                 \
                    fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] codebook_id=%u requires emins buffer (M=%d N=%d K=%d)\n", \
                            static_cast<unsigned>(codebook_id), M, N, K);                         \
                    return false;                                                                 \
                }                                                                                 \
            }                                                                                     \
            return rocmLaunch_native_quantised_small_m_fp32<FMT_TAG>(                                 \
                d_A_int8, d_payload, d_scales_u16, d_mins_u16, d_emins_u32, d_C_fp32,            \
                d_scale_A_blockwise, d_sum_A_blockwise, d_partial_fp32,                           \
                M, N, K, device_id, stream, arithmetic_policy_codebook_id, row_range, nullptr, destination_fold);                       \
        } while (0)

    switch (static_cast<NativeQuantisedFormat>(codebook_id))
    {
    case NATIVE_QUANTISED_Q4_0:     LAUNCH_SMALL_M(NATIVE_QUANTISED_Q4_0);
    case NATIVE_QUANTISED_IQ4_NL:   LAUNCH_SMALL_M(NATIVE_QUANTISED_IQ4_NL);
    case NATIVE_QUANTISED_IQ4_XS_SOURCE:   LAUNCH_SMALL_M(NATIVE_QUANTISED_IQ4_XS_SOURCE);
    case NATIVE_QUANTISED_Q4_1:     LAUNCH_SMALL_M(NATIVE_QUANTISED_Q4_1);
    case NATIVE_QUANTISED_Q5_0:     LAUNCH_SMALL_M(NATIVE_QUANTISED_Q5_0);
    case NATIVE_QUANTISED_Q5_1:     LAUNCH_SMALL_M(NATIVE_QUANTISED_Q5_1);
    case NATIVE_QUANTISED_Q6_K:     LAUNCH_SMALL_M(NATIVE_QUANTISED_Q6_K);
    case NATIVE_QUANTISED_Q3_K:     LAUNCH_SMALL_M(NATIVE_QUANTISED_Q3_K);
    case NATIVE_QUANTISED_Q2_K:     LAUNCH_SMALL_M(NATIVE_QUANTISED_Q2_K);
    case NATIVE_QUANTISED_IQ3_S:    LAUNCH_SMALL_M(NATIVE_QUANTISED_IQ3_S);
    case NATIVE_QUANTISED_IQ3_XXS:  LAUNCH_SMALL_M(NATIVE_QUANTISED_IQ3_XXS);
    case NATIVE_QUANTISED_IQ2_S:    LAUNCH_SMALL_M(NATIVE_QUANTISED_IQ2_S);
    case NATIVE_QUANTISED_IQ2_XS:   LAUNCH_SMALL_M(NATIVE_QUANTISED_IQ2_XS);
    case NATIVE_QUANTISED_IQ2_XXS:  LAUNCH_SMALL_M(NATIVE_QUANTISED_IQ2_XXS);
    case NATIVE_QUANTISED_IQ1_S:    LAUNCH_SMALL_M(NATIVE_QUANTISED_IQ1_S);
    case NATIVE_QUANTISED_IQ1_M:    LAUNCH_SMALL_M(NATIVE_QUANTISED_IQ1_M);
    case NATIVE_QUANTISED_Q8_0:     LAUNCH_SMALL_M(NATIVE_QUANTISED_Q8_0);
    case NATIVE_QUANTISED_EXPANDED_INT8_MIN:
        LAUNCH_SMALL_M(NATIVE_QUANTISED_EXPANDED_INT8_MIN);
    default:
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] unsupported codebook_id=%u (M=%d N=%d K=%d)\n",
                static_cast<unsigned>(codebook_id), M, N, K);
        return false;
    }

    #undef LAUNCH_SMALL_M
    return false;
}

/** @brief Execute verifier-depth GEMV for self-describing weight bytes. */
bool rocmGemv_native_quantised_small_m_fp32_with_sums(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const void* d_block_scales,
    const void* d_block_mins,
    const void* d_block_emins,
    float* d_C_fp32,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* d_partial_fp32,
    int M, int N, int K,
    uint8_t codebook_id,
    int device_id, void* stream)
{
    return rocmGemv_native_quantised_small_m_fp32_with_sums_policy(
        d_A_int8,
        d_payload,
        d_block_scales,
        d_block_mins,
        d_block_emins,
        d_C_fp32,
        d_scale_A_blockwise,
        d_sum_A_blockwise,
        d_partial_fp32,
        M,
        N,
        K,
        codebook_id,
        defaultNativeQuantisedArithmeticPolicyCodebook(codebook_id),
        device_id,
        stream);
}

/** @brief Execute verifier-depth GEMV with explicit source math policy. */
bool rocmGemv_native_quantised_small_m_fp32_with_policy(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const void* d_block_scales,
    const void* d_block_mins,
    const void* d_block_emins,
    float* d_C_fp32,
    const float* d_scale_A_blockwise,
    float* d_partial_fp32,
    int M, int N, int K,
    uint8_t codebook_id,
    uint8_t arithmetic_policy_codebook_id,
    int device_id, void* stream)
{
    return rocmGemv_native_quantised_small_m_fp32_with_sums_policy(
        d_A_int8,
        d_payload,
        d_block_scales,
        d_block_mins,
        d_block_emins,
        d_C_fp32,
        d_scale_A_blockwise,
        nullptr,
        d_partial_fp32,
        M,
        N,
        K,
        codebook_id,
        arithmetic_policy_codebook_id,
        device_id,
        stream);
}

bool rocmGemv_native_quantised_small_m_fp32(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const void* d_block_scales,
    const void* d_block_mins,
    const void* d_block_emins,
    float* d_C_fp32,
    const float* d_scale_A_blockwise,
    float* d_partial_fp32,
    int M, int N, int K,
    uint8_t codebook_id,
    int device_id, void* stream)
{
    return rocmGemv_native_quantised_small_m_fp32_with_sums(
        d_A_int8, d_payload, d_block_scales, d_block_mins, d_block_emins,
        d_C_fp32, d_scale_A_blockwise, nullptr, d_partial_fp32,
        M, N, K, codebook_id, device_id, stream);
}

static bool rocmGemv_native_quantised_small_m_batched_fp32_with_sums_policy_impl(
    const int8_t* d_A_int8,
    const uint8_t* const* d_payloads,
    const uint16_t* const* d_block_scales,
    const uint16_t* const* d_block_mins,
    const uint32_t* const* d_block_emins,
    const float* const* d_biases,
    float* const* d_outputs,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* const* d_partials,
    const int* Ns,
    int num_projections,
    int M, int K,
    uint8_t codebook_id,
    int device_id,
    void* stream,
    int policy_kb,
    int policy_target_waves,
    const llaminar2::DeviceRowRange *row_range = nullptr)
{
    hipError_t device_err = static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(device_id));
    if (device_err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmGemv_native_quantised_small_m_batched_fp32] hipSetDevice(%d) failed: %s\n",
                device_id, hipGetErrorString(device_err));
        return false;
    }

    if (M <= 0 || (M == 1 && (!row_range || !row_range->countOwner())) ||
        (row_range && row_range->physicalRows() != M))
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] M requires positive rows and an explicit count owner at M1 (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    if (K <= 0 || (K % 32) != 0)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] K must be positive and multiple of 32 (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    if (num_projections <= 0 || num_projections > NATIVE_QUANTISED_SMALL_M_MAX_BATCH)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] invalid projection count=%d (max=%d)\n",
                num_projections, NATIVE_QUANTISED_SMALL_M_MAX_BATCH);
        return false;
    }
    if (!d_A_int8 || !d_payloads || !d_block_scales || !d_outputs ||
        !d_scale_A_blockwise || !d_sum_A_blockwise || !d_partials || !Ns)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] missing required pointer (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    for (int p = 0; p < num_projections; ++p)
    {
        if (Ns[p] <= 0 || !d_payloads[p] || !d_block_scales[p] || !d_outputs[p] || !d_partials[p])
        {
            fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] invalid projection %d (N=%d payload=%p scales=%p output=%p partial=%p)\n",
                    p, Ns[p], static_cast<const void*>(d_payloads[p]),
                    static_cast<const void*>(d_block_scales[p]),
                    static_cast<const void*>(d_outputs[p]),
                    static_cast<const void*>(d_partials[p]));
            return false;
        }
    }

#define DISPATCH_BATCHED_FMT(FMT_TAG)                                                                   \
    do {                                                                                                \
        using Traits = NATIVE_QUANTISEDTraits<FMT_TAG>;                                                            \
        if constexpr (Traits::requires_secondary_metadata) {                                 \
            if (!d_block_mins) {                                                                        \
                fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] codebook_id=%u requires mins/aux array (M=%d K=%d projections=%d)\n", \
                        codebook_id, M, K, num_projections);                                            \
                return false;                                                                           \
            }                                                                                           \
            for (int p = 0; p < num_projections; ++p) {                                                  \
                if (!d_block_mins[p]) {                                                                 \
                    fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] projection %d codebook_id=%u requires mins/aux buffer (M=%d N=%d K=%d)\n", \
                            p, codebook_id, M, Ns[p], K);                                               \
                    return false;                                                                       \
                }                                                                                       \
            }                                                                                           \
        }                                                                                               \
        if constexpr (Traits::needs_emin_array) {                                                       \
            if (!d_block_emins) {                                                                       \
                fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] codebook_id=%u requires emins array (M=%d K=%d projections=%d)\n", \
                        codebook_id, M, K, num_projections);                                            \
                return false;                                                                           \
            }                                                                                           \
            for (int p = 0; p < num_projections; ++p) {                                                  \
                if (!d_block_emins[p]) {                                                                \
                    fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] projection %d codebook_id=%u requires emins buffer (M=%d N=%d K=%d)\n", \
                            p, codebook_id, M, Ns[p], K);                                               \
                    return false;                                                                       \
                }                                                                                       \
            }                                                                                           \
        }                                                                                               \
        return rocmLaunch_native_quantised_small_m_batched_fp32<FMT_TAG>(                                  \
            d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins, d_biases, d_outputs,    \
            d_scale_A_blockwise, d_sum_A_blockwise, d_partials, Ns, num_projections, M, K,             \
            device_id, stream, policy_kb, policy_target_waves, row_range);                                        \
    } while (0)

    switch (codebook_id)
    {
    case NATIVE_QUANTISED_Q4_0:     DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_Q4_0);
    case NATIVE_QUANTISED_IQ4_NL:   DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_IQ4_NL);
    case NATIVE_QUANTISED_IQ4_XS_SOURCE:   DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_IQ4_XS_SOURCE);
    case NATIVE_QUANTISED_Q4_1:     DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_Q4_1);
    case NATIVE_QUANTISED_Q5_0:     DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_Q5_0);
    case NATIVE_QUANTISED_Q5_1:     DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_Q5_1);
    case NATIVE_QUANTISED_Q6_K:     DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_Q6_K);
    case NATIVE_QUANTISED_Q3_K:     DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_Q3_K);
    case NATIVE_QUANTISED_Q2_K:     DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_Q2_K);
    case NATIVE_QUANTISED_IQ3_S:    DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_IQ3_S);
    case NATIVE_QUANTISED_IQ3_XXS:  DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_IQ3_XXS);
    case NATIVE_QUANTISED_IQ2_S:    DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_IQ2_S);
    case NATIVE_QUANTISED_IQ2_XS:   DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_IQ2_XS);
    case NATIVE_QUANTISED_IQ2_XXS:  DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_IQ2_XXS);
    case NATIVE_QUANTISED_IQ1_S:    DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_IQ1_S);
    case NATIVE_QUANTISED_IQ1_M:    DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_IQ1_M);
    case NATIVE_QUANTISED_Q8_0:     DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_Q8_0);
    case NATIVE_QUANTISED_EXPANDED_INT8_MIN:
        DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_EXPANDED_INT8_MIN);
    default:
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] unsupported codebook_id=%u (M=%d K=%d projections=%d)\n",
                codebook_id, M, K, num_projections);
        return false;
    }

#undef DISPATCH_BATCHED_FMT
}

bool rocmGemv_native_quantised_small_m_batched_fp32_with_sums_policy(
    const int8_t* d_A_int8,
    const uint8_t* const* d_payloads,
    const uint16_t* const* d_block_scales,
    const uint16_t* const* d_block_mins,
    const uint32_t* const* d_block_emins,
    const float* const* d_biases,
    float* const* d_outputs,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* const* d_partials,
    const int* Ns,
    int num_projections,
    int M, int K,
    uint8_t codebook_id,
    int device_id,
    void* stream,
    int policy_kb,
    int policy_target_waves,
    const llaminar2::DeviceRowRange *row_range)
{
    return rocmGemv_native_quantised_small_m_batched_fp32_with_sums_policy_impl(
        d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins,
        d_biases, d_outputs, d_scale_A_blockwise, d_sum_A_blockwise,
        d_partials, Ns, num_projections, M, K, codebook_id, device_id,
        stream, policy_kb, policy_target_waves, row_range);
}

bool rocmGemv_native_quantised_small_m_batched_fp32_with_sums(
    const int8_t* d_A_int8,
    const uint8_t* const* d_payloads,
    const uint16_t* const* d_block_scales,
    const uint16_t* const* d_block_mins,
    const uint32_t* const* d_block_emins,
    const float* const* d_biases,
    float* const* d_outputs,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* const* d_partials,
    const int* Ns,
    int num_projections,
    int M, int K,
    uint8_t codebook_id,
    int device_id,
    void* stream)
{
    return rocmGemv_native_quantised_small_m_batched_fp32_with_sums_policy_impl(
        d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins,
        d_biases, d_outputs, d_scale_A_blockwise, d_sum_A_blockwise,
        d_partials, Ns, num_projections, M, K, codebook_id, device_id,
        stream, -1, -1);
}


static bool rocmGemv_native_quantised_small_m_batched_mixed_fp32_with_sums_policy_impl(
    const int8_t* d_A_int8,
    const uint8_t* const* d_payloads,
    const uint16_t* const* d_block_scales,
    const uint16_t* const* d_block_mins,
    const uint32_t* const* d_block_emins,
    const float* const* d_biases,
    float* const* d_outputs,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* const* d_partials,
    const int* Ns,
    const uint8_t* codebook_ids,
    int num_projections,
    int M, int K,
    int device_id,
    void* stream,
    int policy_kb,
    int policy_target_waves,
    const llaminar2::DeviceRowRange *row_range = nullptr)
{
    hipError_t device_err = static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(device_id));
    if (device_err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] hipSetDevice(%d) failed: %s\n",
                device_id, hipGetErrorString(device_err));
        return false;
    }

    if (M <= 0 || (M == 1 && (!row_range || !row_range->countOwner())) ||
        (row_range && row_range->physicalRows() != M))
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] M requires positive rows and an explicit count owner at M1 (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    if (K <= 0 || (K % 32) != 0)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] K must be positive and multiple of 32 (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    if (num_projections <= 0 || num_projections > NATIVE_QUANTISED_SMALL_M_MAX_BATCH)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] invalid projection count=%d (max=%d)\n",
                num_projections, NATIVE_QUANTISED_SMALL_M_MAX_BATCH);
        return false;
    }
    if (!d_A_int8 || !d_payloads || !d_block_scales || !d_outputs ||
        !d_scale_A_blockwise || !d_sum_A_blockwise || !d_partials || !Ns || !codebook_ids)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] missing required pointer (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    for (int p = 0; p < num_projections; ++p)
    {
        const uint8_t codebook = codebook_ids[p];
        if (!nativeQuantisedCodebookSupported(codebook))
        {
            fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] unsupported projection %d codebook_id=%u (M=%d K=%d projections=%d)\n",
                    p, static_cast<unsigned>(codebook), M, K, num_projections);
            return false;
        }
        if (Ns[p] <= 0 || !d_payloads[p] || !d_block_scales[p] || !d_outputs[p] || !d_partials[p])
        {
            fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] invalid projection %d (N=%d payload=%p scales=%p output=%p partial=%p codebook=%u)\n",
                    p, Ns[p], static_cast<const void*>(d_payloads[p]),
                    static_cast<const void*>(d_block_scales[p]),
                    static_cast<const void*>(d_outputs[p]),
                    static_cast<const void*>(d_partials[p]),
                    static_cast<unsigned>(codebook));
            return false;
        }
        if (nativeQuantisedCodebookRequiresAux(codebook) &&
            (!d_block_mins || !d_block_mins[p]))
        {
            fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] projection %d codebook_id=%u requires mins/aux buffer (M=%d N=%d K=%d)\n",
                    p, static_cast<unsigned>(codebook), M, Ns[p], K);
            return false;
        }
        if (nativeQuantisedCodebookRequiresEmins(codebook) &&
            (!d_block_emins || !d_block_emins[p]))
        {
            fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] projection %d codebook_id=%u requires emins buffer (M=%d N=%d K=%d)\n",
                    p, static_cast<unsigned>(codebook), M, Ns[p], K);
            return false;
        }
    }

    return rocmLaunch_native_quantised_small_m_batched_mixed_fp32(
        d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins,
        d_biases, d_outputs, d_scale_A_blockwise, d_sum_A_blockwise,
        d_partials, Ns, codebook_ids, num_projections, M, K, device_id,
        stream, policy_kb, policy_target_waves, row_range);
}

bool rocmGemv_native_quantised_small_m_batched_mixed_fp32_with_sums_policy(
    const int8_t* d_A_int8,
    const uint8_t* const* d_payloads,
    const uint16_t* const* d_block_scales,
    const uint16_t* const* d_block_mins,
    const uint32_t* const* d_block_emins,
    const float* const* d_biases,
    float* const* d_outputs,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* const* d_partials,
    const int* Ns,
    const uint8_t* codebook_ids,
    int num_projections,
    int M, int K,
    int device_id,
    void* stream,
    int policy_kb,
    int policy_target_waves,
    const llaminar2::DeviceRowRange *row_range)
{
    return rocmGemv_native_quantised_small_m_batched_mixed_fp32_with_sums_policy_impl(
        d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins,
        d_biases, d_outputs, d_scale_A_blockwise, d_sum_A_blockwise,
        d_partials, Ns, codebook_ids, num_projections, M, K, device_id,
        stream, policy_kb, policy_target_waves, row_range);
}

bool rocmGemv_native_quantised_small_m_batched_mixed_fp32_with_sums(
    const int8_t* d_A_int8,
    const uint8_t* const* d_payloads,
    const uint16_t* const* d_block_scales,
    const uint16_t* const* d_block_mins,
    const uint32_t* const* d_block_emins,
    const float* const* d_biases,
    float* const* d_outputs,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* const* d_partials,
    const int* Ns,
    const uint8_t* codebook_ids,
    int num_projections,
    int M, int K,
    int device_id,
    void* stream)
{
    return rocmGemv_native_quantised_small_m_batched_mixed_fp32_with_sums_policy_impl(
        d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins,
        d_biases, d_outputs, d_scale_A_blockwise, d_sum_A_blockwise,
        d_partials, Ns, codebook_ids, num_projections, M, K, device_id,
        stream, -1, -1);
}

} // extern "C"
